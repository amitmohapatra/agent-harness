"""Background tasks with no server to host: a sub-agent's run that goes on after the turn that
started it replied, while the chat takes new messages, its outcome brought back to the chat —
the harness's own, for ``ReAct``'s sub-agents (``bridges.background_tasks.BackgroundTasks``),
in the shape of Deep Agents' async sub-agents, whose tools, task record and notification it
keeps: the model sees one interface, governed one way (``task_tools``), whether a task runs on
a server (``AsyncSubAgent``) or here.

* **The task** is a child run of the turn that started it (``Agent._start``, its parent the
  turn: its tenant, user, documents and what it is without), on a conversation of its own (a
  thread of its own, whose id is Deep Agents' ``task_id`` and ``thread_id``), its metadata
  naming it (:data:`BACKGROUND`: the sub-agent's name, and the agent and chat that started
  it). Its attempts run in this process, apart from the turn (:class:`Tasks`), through the
  pipeline as any run's — its tool calls governed, recorded under its run, its budget and time
  its agent's — and held in the run store meanwhile (``holding``): a server killed with it
  leaves a run the store ends (``holder_lost``), the harness closing cuts it off
  (:data:`CUT_OFF`), and cancelling the turn cancels it (``children.end_children``:
  :meth:`Tasks.stop` ends its attempt here); cancelled on
  another replica, its hold here is refused and its attempt stops (``Harness._lost``). Its
  record is the run store's, so a check, the list, an update or a cancel reads it from any
  replica.
* **Its record** in the agent's state is Deep Agents' ``AsyncTask`` (:func:`task_of`): its id,
  the sub-agent's name, its thread, its latest run, a status (:data:`STATUS`, from the run's)
  and three times — kept from turn to turn as Deep Agents' list is
  (``adapters.langgraph.CARRIED``).
* **Its outcome** — ended, or waiting for a person — comes back as Deep Agents' notification
  (:func:`update_of`, :func:`content`): a user-role message the user did not type, marked with
  langchain-core's notification metadata (source :data:`SOURCE`, its data the ``TaskEvent``).
  It reaches the conversation at its next model call, or its next turn, as Deep Agents' pull
  does; and, while the chat is idle, by a turn of the chat's own (:func:`notify`): a run of the
  agent in the chat's thread, its input the notification (:data:`NOTIFIED` in its metadata),
  which the chat's thread stream delivers as it delivers a scheduled run's
  (``agui.threads``). One that a turn of the conversation already gave the model is quiet
  (:func:`stale`): not delivered again.
* **A question it asks** (an approval, an answer) pauses its run; the conversation's next turn
  — the chat's own when it is idle — asks it through the chat (:func:`ask`), as a sub-agent's
  question is asked through its parent, and the answer resumes the task where the answer
  reached (:func:`answered`), on any replica.
* **Follow-ups** (``update_async_task``) are a new run of the task's conversation, in the mode
  Deep Agents' ``update_strategy`` names: ``interrupt`` cancels the run at work and continues
  the conversation (``conversation.Mode.INTERRUPT``), ``enqueue`` waits for it
  (``Mode.QUEUE``).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import html
import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, cast

from trellis.contracts import (
    AgentError,
    ErrorCategory,
    Interrupt,
    InterruptResolution,
    RunRecord,
    RunStatus,
    new_id,
    safe_id,
    stable_id,
)
from trellis.harness import admission, children, stopping, task_tools
from trellis.harness.adapters.langgraph import ASKED
from trellis.harness.bounded import LRU, MAXSIZE
from trellis.harness.conversation import INTERRUPTED, Mode
from trellis.harness.frontend import FRONTEND
from trellis.harness.identity import conversation_key, queued_with
from trellis.harness.journal import Journal

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The metadata key of a background task's run: ``{"name", "agent", "thread"}`` — the
#: sub-agent's name, and the agent and the chat thread that started it (and that chat's front
#: end, :data:`~trellis.harness.frontend.FRONTEND`, when it had one).
BACKGROUND: Final = "trellis_background"
#: The metadata key of a notification run (:func:`notify`): the ``TaskEvent`` it delivers, and
#: the server whose start made it (``by``: its holds' name), the one that runs it.
NOTIFIED: Final = "trellis_notified"
#: A task's status by its run's (Deep Agents' words: ``pending`` not started yet, ``waiting``
#: for a person).
STATUS: Final[Mapping[RunStatus, str]] = {
    RunStatus.QUEUED: "pending",
    RunStatus.RUNNING: "running",
    RunStatus.PAUSED: "waiting",
    RunStatus.SUCCESS: "success",
    RunStatus.PARTIAL: "success",
    RunStatus.ERROR: "error",
    RunStatus.TIMEOUT: "timeout",
    RunStatus.CANCELLED: "cancelled",
    RunStatus.REJECTED: "error",  # refused (its policy, a rule): it failed, as Deep Agents says
}
#: The statuses that can still change: what a pull reads again.
UNFINISHED: Final = frozenset(task_tools.UNFINISHED)
#: What every notification says first (Deep Agents' words): not the user's message.
NOTE: Final = (
    "An async subagent task you started has an update. This is not a message from the user: "
    "don't treat it as their answer to any question."
)
#: What a notification says to do next, by the task's status.
NEXT: Final[Mapping[str, str]] = {
    "success": "Use the result to continue; report back to the user if they're waiting on it.",
    "error": "Use `check_async_task` for details, then decide whether to retry, try another "
    "approach, or tell the user.",
    "timeout": "The task ran too long and was stopped. Decide whether to retry with a smaller "
    "task, or tell the user.",
    "cancelled": "The task's run was cancelled before it finished, outside your tools. Decide "
    "whether to start it again, or tell the user.",
    "waiting": "The subagent waits for the user: its question is asked in this chat, and you "
    "can't answer or approve it for them. Keep working; once they answer, it goes on.",
}
#: The longest result a notification carries (Deep Agents' bound: a large tool result's).
RESULT_CHARS: Final = 4 * 20_000
#: What a failed task's notification says of it: the details are ``check_async_task``'s.
FAILED: Final = "The subagent failed."
#: Why a task at work in this process ended when the harness closed first.
CUT_OFF: Final = "its work in the background was cut off: the harness closed before it ended"
#: The error code of a task cut off.
UNFINISHED_CODE: Final = "BackgroundUnfinished"
#: The most attempts at work in one process (:class:`Tasks`): past it a start is refused
#: (:data:`NO_ROOM`), an idle chat's outcome waits for the conversation's next turn, and an
#: answer waits to be asked for again.
AT_WORK: Final = MAXSIZE
#: How many times in a row a task's outcome, told by a turn of the chat's own (:func:`notify`),
#: may start or update a task with no message of the user's between: past it the chain ends
#: (:data:`CHAINED`).
CHAIN: Final = 3
#: What a start or an update answers when it is not admitted (:func:`start`): too many
#: unfinished tasks in the chat, a chain of outcomes past :data:`CHAIN`, no room in the process.
TOO_MANY: Final = (
    "Not started: this chat has {count} background tasks unfinished, the most at once "
    "({limit}). Wait for one to end, or cancel one that is no longer needed, then start it again."
)
CHAINED: Final = (
    "Not started: the results of background tasks have started or updated a task {limit} "
    "times in a row with no message from the user between. Tell the user what was found; "
    "their next message can start more."
)
NO_ROOM: Final = (
    "Not started: this server has no room for more background work right now. Do the work "
    "with the subagent directly, or try again later."
)
#: The newest runs of a chat read to tell whether it is idle (:func:`idle`).
IDLE_RUNS: Final = 5


def now() -> str:
    """The time now, as Deep Agents' task records keep it (seconds, UTC)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def status_of(record: RunRecord) -> str:
    """The task status of the run ``record`` is."""
    return STATUS[record.status]


def task_of(record: RunRecord, name: str) -> dict[str, Any]:
    """The task record (Deep Agents' ``AsyncTask``) of a task whose run is ``record``."""
    at = now()
    return {
        "task_id": record.thread_id,
        "agent_name": name,
        "thread_id": record.thread_id,
        "run_id": record.run_id,
        "status": status_of(record),
        "created_at": at,
        "last_checked_at": at,
        "last_updated_at": at,
    }


def observed(task: Mapping[str, Any], record: RunRecord) -> dict[str, Any]:
    """``task`` as its run ``record`` says it is now."""
    return moved(task, run_id=record.run_id, status=status_of(record))


def moved(task: Mapping[str, Any], *, run_id: str, status: str) -> dict[str, Any]:
    """``task`` read now, on the run ``run_id`` with ``status`` (Deep Agents' ``_observed``):
    checked now, and updated now when either differs from what its record said."""
    at = now()
    changed = status != task["status"] or run_id != task["run_id"]
    return {
        **task,
        "run_id": run_id,
        "status": status,
        "last_checked_at": at,
        "last_updated_at": at if changed else task["last_updated_at"],
    }


def result_of(task: Mapping[str, Any], record: RunRecord) -> dict[str, Any]:
    """What ``check_async_task`` answers (Deep Agents' check result): the status and the thread,
    and the answer of a task that succeeded, the error of one that failed, the question of one
    that waits."""
    status = status_of(record)
    found: dict[str, Any] = {"status": status, "thread_id": task["thread_id"]}
    if status == "success":
        found["result"] = _text(record.output)
    elif status in ("error", "timeout"):
        found["error"] = record.error.message if record.error is not None else FAILED
    elif status == "waiting":
        found["interrupts"] = _interrupts(record)
        found["note"] = NEXT["waiting"]
    return found


def update_of(task: Mapping[str, Any], record: RunRecord) -> dict[str, Any] | None:
    """The notification of an outcome the task reached (Deep Agents' ``TaskEvent`` and what the
    model reads of it: the result, a failure, the question); ``None`` while it works."""
    status = status_of(record)
    if status in ("pending", "running"):
        return None
    update: dict[str, Any] = {**event_of(task, record)}
    if status == "success":
        update["result"] = _text(record.output)
    elif status == "error":
        update["error"] = FAILED
    elif status == "waiting":
        update["interrupts"] = _interrupts(record)
    return update


def event_of(task: Mapping[str, Any], record: RunRecord) -> dict[str, Any]:
    """The ``TaskEvent`` of ``record``'s run of ``task``."""
    return {
        "task_id": task["task_id"],
        "agent_name": task["agent_name"],
        "status": status_of(record),
        "run_id": record.run_id,
    }


def content(update: Mapping[str, Any], steps: Mapping[str, str] = NEXT) -> list[dict[str, Any]]:
    """A task update as the model reads it, block by block, in Deep Agents' notification's
    shape: the note, the task's header, its result or details, and what to do next (``steps``,
    by status) — the sub-agent's words escaped, so they cannot close the tags and pose as the
    system."""
    header = (
        f"<task-notification>\n<task-id>{_escape(update['task_id'])}</task-id>\n"
        f"<agent-name>{_escape(update['agent_name'])}</agent-name>\n"
        f"<status>{_escape(update['status'])}</status>\n"
    )
    blocks = [_block(f"{NOTE}\n"), _block(header)]
    if "result" in update:
        said = _escape(_clip(str(update["result"])))
        blocks += [_block("<result>\n"), _block(said), _block("\n</result>\n")]
    details = {key: update[key] for key in ("error", "interrupts") if key in update}
    if details:
        said = _escape(_clip(json.dumps(details, default=str)))
        blocks.append(_block(f"<details>{said}</details>\n"))
    blocks.append(_block(f"</task-notification>\n{steps[update['status']]}"))
    return blocks


def _block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def _clip(text: str) -> str:
    if len(text) <= RESULT_CHARS:
        return text
    return text[:RESULT_CHARS] + "\n... [cut off; use `check_async_task` for the full details]"


def _text(output: Any) -> str:
    if output is None:
        return "(completed with no output)"
    return output if isinstance(output, str) else json.dumps(output, default=str)


def _interrupts(record: RunRecord) -> list[dict[str, Any]]:
    asked = record.awaiting
    if asked is None:
        return []
    return [{"id": asked.interrupt_id, "value": asked.question}]


def notification(update: Mapping[str, Any]) -> Any:
    """A task update as the message the model is given: a LangChain ``HumanMessage`` of its
    :func:`content`, marked with langchain-core's notification metadata — source
    ``task_tools.SOURCE``, its data the ``TaskEvent`` — as Deep Agents' notification is."""
    from langchain_core.messages import (  # noqa: PLC0415 - LangChain, an extra's
        HumanMessage,
        add_notification_metadata,
    )

    event = {k: update[k] for k in ("task_id", "agent_name", "status", "run_id")}
    return add_notification_metadata(
        HumanMessage(content=cast(Any, content(update))), source=task_tools.SOURCE, data=event
    )


def input_of(update: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A notification run's input (:func:`notify`): the :func:`notification` as JSON, marked as
    the run's question although the user did not type it (``adapters.langgraph.ASKED``)."""
    message = notification(update)
    return [
        {
            "role": "user",
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "response_metadata": {ASKED: True},
        }
    ]


# --------------------------------------------------------------------------- its runs
class Tasks:
    """The background tasks' attempts at work in this process (``Harness.background``): each
    apart from the turn that started it (a context of its own: the turn is not its parent
    there), held in the run store by its attempt (``pipeline.attempt``)."""

    def __init__(self) -> None:
        self._at_work: dict[str, asyncio.Task[None]] = {}
        self._agents: dict[str, Agent] = {}
        #: the runs a place is kept for while their caller starts them (:meth:`placed`)
        self._coming: set[str] = set()
        #: how the task runs whose attempt ended here ended: their final records
        self._ended: LRU[str, RunRecord] = LRU()

    def ended(self, run_id: str) -> RunRecord | None:
        """The final record of the run ``run_id``, when its attempt ended here: what no read
        of the run store needs to ask again."""
        return self._ended.get(run_id)

    def at_work(self, run_id: str) -> bool:
        return run_id in self._at_work or run_id in self._coming

    @contextlib.contextmanager
    def placed(self, run_id: str) -> Iterator[bool]:
        """A place among the attempts at work here — :data:`AT_WORK` at most — kept for the
        run ``run_id`` while its caller starts it: whether it has one (its earlier attempt's,
        or a free one, taken now and given back when the caller is done: an attempt begun
        meanwhile keeps its own)."""
        held = self.at_work(run_id)
        free = held or len(self._at_work) + len(self._coming) < AT_WORK
        took = free and not held
        if took:
            self._coming.add(run_id)
        try:
            yield free
        finally:
            if took:
                self._coming.discard(run_id)

    def run(
        self,
        agent: Agent,
        record: RunRecord,
        *,
        then: Callable[[RunRecord], Awaitable[None]] | None = None,
        journal: Journal | None = None,
        resolution: InterruptResolution | None = None,
        mode: Mode = Mode.QUEUE,
    ) -> None:
        """The next attempt of ``record``'s run, under way here from now; ``then`` is told how
        it left the run (ended or paused)."""
        work = self._work(
            agent, record, then=then, journal=journal, resolution=resolution, mode=mode
        )
        task = asyncio.get_running_loop().create_task(work, context=contextvars.Context())
        self._at_work[record.run_id] = task
        self._agents[record.run_id] = agent
        task.add_done_callback(lambda _: self._done(record.run_id, task))

    def _done(self, run_id: str, task: asyncio.Task[None]) -> None:
        # the run's next attempt may be at work already (the person answered while this one
        # still handed its outcome on): only this attempt is forgotten
        if self._at_work.get(run_id) is task:
            del self._at_work[run_id]
            del self._agents[run_id]

    async def _work(
        self,
        agent: Agent,
        record: RunRecord,
        *,
        then: Callable[[RunRecord], Awaitable[None]] | None,
        journal: Journal | None,
        resolution: InterruptResolution | None,
        mode: Mode,
    ) -> None:
        from trellis.harness import pipeline  # noqa: PLC0415 - it imports this module

        runs, tenant = agent.harness.runs, record.tenant_id
        try:
            await pipeline.attempt(
                agent, record, record.input, journal=journal, resolution=resolution, mode=mode
            )
        except Exception:
            log.exception("background task %s", record.run_id)
        if then is None:
            return
        try:  # the read too: a run store that fails here must not end this task unheard
            found = await runs.get(record.run_id, tenant=tenant)
            if found is not None and found.final:
                self._ended[record.run_id] = found
            if found is not None and (found.final or found.status is RunStatus.PAUSED):
                await then(found)
        except Exception:
            log.exception("background task %s: its outcome not delivered", record.run_id)

    def stop(self, run_id: str, reason: str) -> None:
        """What this process does for the task's run ``run_id`` stopped, when it is at work
        here, as any run's is (``Agent.stop``): its attempt under way stops (its run is ended
        in the store already, or ends ``CANCELLED`` with ``reason``), and one waiting its turn
        behind the task's earlier run leaves the line now, not once that run does."""
        agent = self._agents.get(run_id)
        if agent is not None:
            agent.stop(run_id, reason)

    async def aclose(self) -> None:
        """The harness closes: each task still at work is cut off — its run ended ``ERROR``
        (:data:`CUT_OFF`), its attempt stopped."""
        error = AgentError(code=UNFINISHED_CODE, category=ErrorCategory.DEPENDENCY, message=CUT_OFF)
        for run_id, task in list(self._at_work.items()):
            agent = self._agents[run_id]
            runtime = agent.running.get(run_id)
            if runtime is not None:
                await stopping.unless_ended(  # (or it ended meanwhile)
                    agent.harness.runs.finish(
                        run_id, RunStatus.ERROR, error=error, tenant=runtime.tenant
                    )
                )
            task.cancel()
        await asyncio.gather(*self._at_work.values(), return_exceptions=True)


class Refused(Exception):
    """A start or an update not admitted: what the model reads instead of its result."""


async def start(
    runtime: Runtime, agent: Agent, name: str, description: str, tasks: Mapping[str, Any]
) -> dict[str, Any]:
    """A task of ``agent`` (the sub-agent called ``name``) on ``description``, started from
    ``runtime``'s turn, whose conversation carries ``tasks``: its run started and at work
    here; its task record. Admitted first (:class:`Refused` otherwise): the turn is not past
    the chain of outcomes (:func:`depth_of`), the process has room (``Tasks.placed``), and the
    conversation has fewer unfinished tasks than a run may have children working at once
    (:func:`_counted`)."""
    place = runtime.idempotency_key or new_id("call_")
    # the task's conversation (its id) and its first turn: follow-ups are turns of it too
    thread = stable_id(place, BACKGROUND, name, prefix="task_")
    run_id = stable_id(place, BACKGROUND, name, prefix="run_")
    origin = {"name": name, "agent": runtime.agent_id, "thread": runtime.identity.thread}
    origin |= {"depth": await depth_of(runtime)}
    origin |= {FRONTEND: runtime.frontend.kept()} if runtime.frontend is not None else {}
    with runtime.agent.harness.background.placed(run_id) as free:
        if not free:
            raise Refused(NO_ROOM)
        _counted(runtime, tasks, thread)
        record = await _started(
            runtime, agent, description, run_id=run_id, thread=thread, origin=origin
        )
        _go(runtime.agent, agent, record)
    return task_of(record, name)


async def depth_of(runtime: Runtime) -> int:
    """How many outcomes in a row, with no message of the user's between, led to ``runtime``'s
    turn: none for a turn a message began; for a turn of the chat's own (:func:`notify`), one
    more than the turn that started the task it tells of — a task here, or one on a server of
    the user's (``bridges.async_tasks``: its run's :data:`~trellis.harness.bridges.helpers.
    HELPER` metadata carries the depth, :func:`origin_of`). Past :data:`CHAIN` the turn starts
    and updates nothing (:class:`Refused`): a result cannot keep work going by itself."""
    if runtime.notification is None:
        return 0
    record = await runtime.agent.harness.runs.get(runtime.run_id, tenant=runtime.tenant)
    told = {} if record is None else record.metadata.get(NOTIFIED) or {}
    depth = int(told.get("depth", CHAIN + 1))  # unknown: past the chain
    if depth > CHAIN:
        raise Refused(CHAINED.format(limit=CHAIN))
    return depth


def _counted(runtime: Runtime, tasks: Mapping[str, Any], task_id: str) -> None:
    """The task ``task_id`` counted among the unfinished ones of ``runtime``'s conversation —
    those its list (``tasks``) says, and those this attempt started since (its calls of one
    model step do not see each other's) — when they are fewer than a run may have children
    working at once (``admission``: the limit falls under pressure); else :class:`Refused`."""
    admission.ADMISSION.admitting()
    limit = admission.ADMISSION.limit
    unfinished = {i for i, t in tasks.items() if t.get("status") in UNFINISHED}
    unfinished |= {i for i in runtime.tasks_started if i not in tasks}
    if task_id not in unfinished and len(unfinished) >= limit:
        raise Refused(TOO_MANY.format(count=len(unfinished), limit=limit))
    runtime.tasks_started.add(task_id)


async def follow_up(
    runtime: Runtime,
    agent: Agent,
    task: Mapping[str, Any],
    latest: RunRecord,
    message: str,
    *,
    mode: Mode,
) -> dict[str, Any]:
    """``message`` sent to ``task``, whose run of now is ``latest`` (the conversation's:
    :func:`owned`): a new run of its conversation, which sees what it did so
    far, under the task's id. ``mode`` is what Deep Agents' ``update_strategy`` says of the
    run at work: ``Mode.INTERRUPT`` stops it first (here, as the conversation's line does;
    elsewhere, by its cancel), ``Mode.QUEUE`` waits for it. The task's record, updated.
    Admitted as a start is, but for the count (it is no new task): :class:`Refused` otherwise."""
    origin = {**latest.metadata[BACKGROUND], "depth": await depth_of(runtime)}
    place = runtime.idempotency_key or new_id("call_")
    run_id = stable_id(place, BACKGROUND, task["task_id"], prefix="run_")
    tasks = runtime.agent.harness.background
    with tasks.placed(run_id) as free:
        if not free:
            raise Refused(NO_ROOM)
        if mode is Mode.INTERRUPT and not latest.final and not tasks.at_work(latest.run_id):
            # at work on another replica, or waiting for a person: stopped there
            await stopping.unless_ended(  # (or it ended meanwhile)
                agent.cancel(latest.run_id, reason=INTERRUPTED, tenant=runtime.tenant)
            )
        record = await _started(
            runtime, agent, message, run_id=run_id, thread=task["task_id"], origin=origin
        )
        _go(runtime.agent, agent, record, mode=mode)
    updated = moved(task, run_id=record.run_id, status="running")
    if mode is Mode.QUEUE and not latest.final:
        updated["superseded_run_ids"] = [*task.get("superseded_run_ids", []), task["run_id"]]
    return updated


async def _started(
    runtime: Runtime,
    agent: Agent,
    input: str,
    *,
    run_id: str,
    thread: str,
    origin: Mapping[str, Any],
) -> RunRecord:
    """The run of a task's turn started in the run store: ``runtime``'s child, on the task's
    own conversation (``thread``) and in its line (``line``: a follow-up queued behind the run
    at work waits for it, in any process), its metadata naming where it came from."""
    conversation = conversation_key(runtime.tenant, runtime.user, thread)
    started = await agent._start(
        input,
        user=runtime.user,
        thread=thread,
        tenant=runtime.tenant,
        run_id=run_id,
        record_input=True,
        parent=runtime,
        concurrency_key=queued_with(conversation),
    )
    started = started.model_copy(
        update={"metadata": {**started.metadata, BACKGROUND: dict(origin)}}
    )
    return await agent.harness.runs.start(started)


def _go(parent: Agent, agent: Agent, record: RunRecord, *, mode: Mode = Mode.QUEUE) -> None:
    """``record``'s run at work here, its outcome then brought back to the chat it came from."""

    async def then(found: RunRecord) -> None:
        await delivered(parent, found)

    parent.harness.background.run(agent, record, then=then, mode=mode)


async def stop(runtime: Runtime, agent: Agent, task: Mapping[str, Any], latest: RunRecord) -> None:
    """``task`` cancelled: its run of now (``latest``), and the earlier ones a queued follow-up
    left behind it that still work (newest first, so cancelling one does not let the next
    begin) — each one the conversation's (:func:`owned`)."""
    earlier = [
        {**task, "run_id": run_id} for run_id in reversed(task.get("superseded_run_ids", []))
    ]
    for record in [latest, *[await owned(runtime, t) for t in earlier]]:
        if record is None:
            continue
        await stopping.unless_ended(  # (or it ended already)
            agent.cancel(
                record.run_id,
                reason="cancelled by the agent that started it",
                tenant=runtime.tenant,
            )
        )


# --------------------------------------------------------------------------- whose it is
async def owned(runtime: Runtime, task: Mapping[str, Any]) -> RunRecord | None:
    """The run ``task``'s record names, when it is a task of ``runtime``'s conversation: a
    background task's run (:data:`BACKGROUND`) of its tenant and user that its agent started
    in its chat thread (:func:`of_chat`), on the conversation the task's id names, of the
    sub-agent the record names. ``None``
    for any other — a record is only what a conversation's state says, and the run it names
    is read, asked about, sent instructions or cancelled only here: every tool, the questions
    asked, each pull and the turn's last look go through this, and what is not owned is
    answered as a task the conversation does not track."""
    harness, run_id = runtime.agent.harness, str(task.get("run_id"))
    # how it ended here, when it did: no read of the store says more
    record = harness.background.ended(run_id) or await harness.runs.get(
        run_id, tenant=runtime.tenant
    )
    origin = None if record is None else record.metadata.get(BACKGROUND)
    if record is None or not isinstance(origin, Mapping):
        return None
    mine = of_chat(runtime, record, agent=origin.get("agent"), thread=origin.get("thread"))
    named = (record.thread_id, origin.get("name")) == (task.get("task_id"), task.get("agent_name"))
    return record if mine and named else None


def of_chat(runtime: Runtime, record: RunRecord, *, agent: Any, thread: Any) -> bool:
    """Whether ``record`` is a run of ``runtime``'s conversation's making: its tenant's and its
    user's, started by its agent (``agent``) in its chat thread (``thread``) — as the record
    says of itself where its kind keeps them (a background task's :data:`BACKGROUND`, a Deep
    Agents task's helper record: ``bridges.async_tasks``)."""
    return (
        record.tenant_id == runtime.tenant
        and record.user_id == runtime.user
        and agent == runtime.agent_id
        and thread == runtime.thread
    )


# --------------------------------------------------------------------------- its outcome
async def delivered(
    parent: Agent,
    record: RunRecord,
    *,
    update: Mapping[str, Any] | None = None,
    ending: str | None = None,
) -> None:
    """A child run of a chat's turn ended or paused — a task here, a Deep Agents async task on
    its server (``bridges.async_tasks``), any run a tool started in the background: its outcome
    (``update``: the ``TaskEvent`` and what the model reads of it, else :func:`update_of` the
    run) told to the chat it came from by a turn of the chat's own (:func:`notify`) when the
    chat is idle — a turn at work, or waiting for a person, gives its model the outcome itself
    (at its next model call, after its answer, or at the conversation's next turn). Told while
    a turn of the chat holds its turn in this process, it is told again once that turn has let
    go (``Live.later``, by the turn's own task — never by the caller's, which that turn may be
    waiting on: a cancel of the task): a turn still at work may read the outcome itself (a turn
    of the chat's own then finds it told, and is quiet: :func:`stale`), but one that is
    finishing looked for the last time already (:func:`left`), and would leave it told by
    nobody. ``ending``: the turn of the chat that is ending now, which took no notice of it
    (:func:`left`)."""
    origin = await origin_of(parent, record)
    user = record.user_id
    if origin is None or not user:
        return
    thread = str(origin["thread"])
    if update is None:
        task = {"task_id": origin["task_id"], "agent_name": origin.get("name", "")}
        update = update_of(task, record)
    live = parent.turns.live(conversation_key(record.tenant_id, user, thread))
    if update is None:  # it works: nothing to tell
        pass
    elif live is not None and live.run_id != ending:
        told = update
        live.later.append(lambda: _told_later(parent, record, told))
    elif await idle(parent, record.tenant_id, user, thread, ending=ending):
        await notify(parent, record, update, user=user, thread=thread, origin=origin)


async def _told_later(parent: Agent, record: RunRecord, update: Mapping[str, Any]) -> None:
    """``record``'s outcome (``update``) told once the chat's turn that held it off has let go
    (:func:`delivered`); what fails is said in the log, never raised in that turn."""
    try:
        await delivered(parent, record, update=update)
    except Exception:
        log.exception("background task %s: its outcome not delivered", record.run_id)


async def origin_of(parent: Agent, record: RunRecord) -> dict[str, Any] | None:
    """Where the child run ``record`` came from, as its outcome is told: the chat ``thread``,
    the sub-agent's ``name``, the ``task_id`` the model knows it by and the chat's front end
    (:data:`~trellis.harness.frontend.FRONTEND`) — its :data:`BACKGROUND` metadata, else its
    parent's record (a framework's helper: its ``task_id`` the one it names, else its run's);
    ``None``: no chat to tell (a run with no thread of its own)."""
    from trellis.harness.bridges.helpers import HELPER  # noqa: PLC0415 - it imports this

    own = record.metadata.get(BACKGROUND)
    if isinstance(own, Mapping):
        found = {**own, "task_id": record.thread_id}
        return found if found.get("thread") else None
    above = None
    if record.parent_run_id is not None:
        above = await parent.harness.runs.get(record.parent_run_id, tenant=record.tenant_id)
    if above is None or not above.thread_id or above.thread_id == above.run_id:
        return None
    helper = record.metadata.get(HELPER) or {}
    found = {
        "thread": above.thread_id,
        "name": str(helper.get("name") or record.agent_id),
        "agent": above.agent_id,
        "task_id": str(helper.get("task_id") or record.run_id),
        "depth": int(helper.get("depth") or 0),  # a remote task's turn's (``depth_of``)
    }
    found |= {FRONTEND: above.metadata[FRONTEND]} if FRONTEND in above.metadata else {}
    return found


async def idle(
    agent: Agent, tenant: str, user: str, thread: str, *, ending: str | None = None
) -> bool:
    """Whether ``user``'s chat ``thread`` with ``agent`` has no turn at work in the run store
    and none waiting for a person — where a turn of its own may begin — but the one ``ending``
    now (one holding the chat's turn in this process, its end written already, is
    :func:`delivered`'s to wait for)."""
    runs = agent.harness.runs
    async for summary in runs.iterate(
        agent_id=agent.id, thread_id=thread, tenant=tenant, limit=IDLE_RUNS, max_pages=1
    ):
        if summary.status.final or summary.run_id == ending:
            continue
        found = await runs.get(summary.run_id, tenant=tenant)
        if found is not None and found.user_id == user and not found.final:
            return False
    return True


async def notify(
    agent: Agent,
    record: RunRecord,
    update: Mapping[str, Any],
    *,
    user: str,
    thread: str,
    origin: Mapping[str, Any],
) -> None:
    """A turn of the chat's own that gives its model the task's ``update``: a run of ``agent``
    in the chat ``thread``, its input the notification (:func:`input_of`), started once per outcome
    (its id is the outcome's: a question the task waits on is one) and run only by the server
    whose start made it — another server telling the same outcome at once finds it made and
    stops —, at work here and delivered by the chat's thread stream."""
    asked = record.awaiting.interrupt_id if record.awaiting is not None else ""
    run_id = stable_id(
        NOTIFIED, update["task_id"], update["run_id"], update["status"], asked, prefix="run_"
    )
    runs, tasks = agent.harness.runs, agent.harness.background
    if tasks.at_work(run_id):
        return  # being told here already
    with tasks.placed(run_id) as free:
        # no room here: the conversation's next turn reads the outcome itself
        if not free or await runs.get(run_id, tenant=record.tenant_id) is not None:
            return  # or told already
        started = await agent._start(
            input_of(update),
            user=user,
            thread=thread,
            tenant=record.tenant_id,
            run_id=run_id,
            record_input=True,
        )
        # who made it: the run store answers a start of a run that exists with that run, so
        # the server whose start made it is the one that finds its own name there, and runs it
        by = agent.harness.holds.name
        told = {k: update[k] for k in ("task_id", "agent_name", "status", "run_id")}
        # one outcome further from the user's last message than the turn that started the task
        depth = int(origin.get("depth", 0)) + 1
        metadata = {**started.metadata, NOTIFIED: {**told, "by": by, "depth": depth}}
        if FRONTEND in origin:  # the chat's front end: a person reads and answers what it asks
            metadata[FRONTEND] = origin[FRONTEND]
        found = await runs.start(started.model_copy(update={"metadata": metadata}))
        ours = (found.metadata.get(NOTIFIED) or {}).get("by") == by
        if ours and found.status is RunStatus.RUNNING:
            tasks.run(agent, found)  # not told meanwhile by another server


async def stale(runtime: Runtime) -> bool:
    """Whether ``runtime`` is a turn of the chat's own for a task's outcome (:func:`notify`)
    that a turn of the conversation already gave the model — the task's record, which the
    conversation carries, says that outcome already, or a later run of the task: it is quiet,
    not delivered again; for a question, once the task waits on it no more (answered,
    cancelled). Only its first attempt is: a resumed one goes on."""
    if runtime.notification is None or runtime.attempt > 1:
        return False
    told = (runtime.notification.notified or {}).get("data") or {}
    if told.get("status") == "waiting":
        found = await runtime.agent.harness.runs.get(str(told.get("run_id")), tenant=runtime.tenant)
        return found is None or found.status is not RunStatus.PAUSED
    tasks = runtime.carried.get("async_tasks") or {}
    task = tasks.get(told.get("task_id")) if isinstance(tasks, Mapping) else None
    if not isinstance(task, Mapping):
        return False
    return task.get("run_id") != told.get("run_id") or task.get("status") == told.get("status")


def quiet(runtime: Runtime, answer: Any) -> bool:
    """Whether ``runtime``'s answer is not delivered: a turn of the chat's own found
    :func:`stale` (it answers ``schedules.NO_REPORT``, as a scheduled run with nothing to report
    does, and the chat's thread stream leaves it out the same way)."""
    from trellis.harness.schedules import NO_REPORT  # noqa: PLC0415 - it imports the runtime

    return runtime.notification is not None and answer == NO_REPORT


def of_turn(runtime: Runtime) -> list[str]:
    """The runs of the background tasks ``runtime``'s conversation carries: they go on whatever
    the turn's clock says (a turn out of time ends its children in its call, not these)."""
    tasks = runtime.carried.get("async_tasks") or {}
    found: list[str] = []
    for task in tasks.values() if isinstance(tasks, Mapping) else ():
        found += [task["run_id"], *task.get("superseded_run_ids", [])]
    return found


async def left(agent: Agent, runtime: Runtime, result: Any) -> None:
    """A turn of a chat ended (not paused): an outcome one of its tasks reached after the turn
    last read it — while the turn was finishing, too late for it and too early for an idle
    chat — told by a turn of the chat's own (:func:`delivered`)."""
    if runtime.parent is not None or result.status is RunStatus.PAUSED:
        return
    tasks = runtime.carried.get("async_tasks") or {}
    for task in tasks.values() if isinstance(tasks, Mapping) else ():
        if task.get("status") not in UNFINISHED:
            continue
        record = await owned(runtime, task)
        if record is None:
            continue  # not the harness's (a task on a server of Deep Agents'), or not its own
        # a new outcome, or a question the turn did not ask (it asks none once at work)
        if status_of(record) != task["status"] or record.status is RunStatus.PAUSED:
            await delivered(agent, record, ending=runtime.run_id)


def agent_for(parent: Agent, name: str, target: Any) -> Agent:
    """The agent the sub-agent ``name`` that ``parent``'s framework runs itself (a Deep Agents
    sub-agent: ``target``, its compiled graph) works as in the background: one per parent and
    name, kept with the harness's agents — a replica that built the same parent finds its runs
    again — with the parent's tools (a sub-agent that names none has the run's), budget, time
    limit, what it is without and framework options."""
    from trellis.harness.agent import Agent  # noqa: PLC0415 - it imports this module

    ident = safe_id(f"{parent.id}.{name}")
    harness = parent.harness
    found = harness.agents.get(ident)
    if found is None:
        found = Agent(
            harness,
            target,
            id=ident,
            budget=parent.budget,
            run_timeout=parent.timeout,
            without=parent.without,
            framework_options=parent.framework_options,
        )
        found.sources = parent.sources
        harness.agents[ident] = found
    return found


# --------------------------------------------------------------------------- its questions
async def ask(runtime: Runtime, record: RunRecord) -> InterruptResolution | None:
    """The question ``record``'s task run waits on asked in ``runtime``'s turn — the chat's, as
    a sub-agent's question is asked through its parent: the turn pauses with it, and its
    answer, once given, is returned (``None``: the run waits on nothing now)."""
    from trellis.harness.subagents import asked_for_background  # noqa: PLC0415 - it imports this

    asked: Interrupt | None = record.awaiting
    if record.status is not RunStatus.PAUSED or asked is None:
        return None
    origin = record.metadata.get(BACKGROUND) or {}
    name = str(origin.get("name") or record.agent_id)
    return await asked_for_background(runtime, asked, agent_id=name, run_id=record.run_id)


async def abandoned(parent: Agent, asked: Interrupt | None, reason: str | None) -> None:
    """The question a chat's turn asked for a task (:func:`ask`, ``asked``) answered with a
    cancel: the task's run is cancelled with the turn, as a sub-agent's is when its question,
    asked through its parent, is — its attempt here stopped, and its own children ended with
    it (``children.end_child``)."""
    from trellis.harness.subagents import asked_by  # noqa: PLC0415 - it imports this

    named = asked_by(asked) if asked is not None else None
    if asked is None or not named or not named.get("background"):
        return
    run_id = str(named["run_id"])
    await children.end_child(
        parent, run_id, asked.tenant_id, status=RunStatus.CANCELLED, reason=reason
    )


async def answered(
    parent: Agent, agent: Agent, record: RunRecord, resolution: InterruptResolution
) -> None:
    """The person's answer to the question ``record``'s task run waits on: the run resumed with
    it in the run store and its next attempt at work here — wherever the answer reached. A run
    no longer waiting on that question (answered already, cancelled) is left as it is."""
    found = await agent.harness.runs.get(record.run_id, tenant=record.tenant_id)
    if (
        found is None
        or found.status is not RunStatus.PAUSED
        or found.awaiting is None
        or found.awaiting.interrupt_id != resolution.interrupt_id
    ):
        return
    tasks = parent.harness.background
    with tasks.placed(found.run_id) as free:
        if not free:  # the question stays the task's: the conversation's next turn asks it
            log.warning("background task %s: no room here to go on with", found.run_id)
            return
        journal, resumed = await agent._resumed(found, resolution)
        if resumed.status is not RunStatus.RUNNING:
            return  # cancelled by the answer: it ended there

        async def then(ended: RunRecord) -> None:
            await delivered(parent, ended)

        tasks.run(agent, resumed, then=then, journal=journal, resolution=resolution)


# --------------------------------------------------------------------------- the five tools
# The model-facing tools: Deep Agents' async sub-agent tools by name (``task_tools``),
# arguments and words; ``bridges.background_tasks`` gives each a LangChain tool's shape.
#: What ``start_async_task`` says: when a task in the background is the right call, and what
#: comes back (the model-facing guidance, here and nowhere else).
START: Final = (
    "Start a subagent on a task in the background of this chat. It returns at once with a "
    "task ID; the subagent works on while this chat goes on.\n\n"
    "Available async agent types:\n{available_agents}\n\n"
    "## Usage notes:\n"
    "1. Use it for work that takes a while (research, a long analysis, several steps of "
    "tools) when the user need not wait for it; for a quick lookup whose answer this reply "
    "needs, call the subagent directly instead.\n"
    "2. It returns at once with a task ID for internal coordination. Briefly tell the user "
    "what was delegated and that the result will come; omit the task ID unless asked. Do NOT "
    "immediately check its status.\n"
    "3. You are told when a task finishes, fails or needs the user's input: a message that is "
    "not from the user brings its result into this chat. Don't check just to see whether it "
    "is done; use `check_async_task` when the user asks, or a notification says to.\n"
    "4. Use `update_async_task` to send a running task new instructions, `cancel_async_task` "
    "to stop one that is no longer needed.\n"
    "5. Several tasks can run at once: start each and let them run.\n"
    "6. A task with status `waiting` waits for the user (an approval or an answer), who is "
    "asked in this chat; you can't answer or approve it for them."
)
CHECK: Final = (
    "Check a background task's status, and its result once it is done (what it asks the user "
    "when its status is `waiting`). Statuses shown earlier in the conversation are stale: call "
    "this for the current one rather than reporting one from an earlier result."
)
UPDATE: Final = (
    "Send a background task new instructions: its current run is stopped and a new one starts "
    "on the same task, which sees what it did so far and your message. The task_id stays the "
    "same."
)
#: What ``update_async_task`` adds for sub-agents whose follow-ups queue (Deep Agents' words).
QUEUED: Final = (
    "\n\nException: for {names}, follow-ups queue behind the current run without interrupting "
    "it; send one follow-up per task per turn, and combine several into one message."
)
CANCEL: Final = (
    "Cancel a running or waiting background task. Use it to stop a task that is no longer needed."
)
LIST: Final = (
    "List the background tasks of this chat with their current statuses ('running', "
    "'waiting', 'success', 'error', 'cancelled'); `status_filter` narrows them. Use "
    "`check_async_task` for a finished task's result. Statuses shown earlier in the "
    "conversation are stale: call this rather than report one from an earlier result."
)
#: What a tool answers for a task id the conversation does not track (Deep Agents' words).
UNTRACKED: Final = "No tracked task found for task_id: {task_id!r}"
NONE: Final = "No async subagent tasks tracked."
#: What a start answers outside a harness run, where nothing can run the task.
NO_RUN: Final = "Background tasks run inside a harness run: none is under way."
#: What a tool's work answers: what the model reads, and the task records it changed.
Answer = tuple[str, dict[str, Any]]


class Offered:
    """The five tools' work for ``subagents`` — each by its name: a wrapped agent, or a target
    the parent's framework runs itself (a Deep Agents sub-agent's compiled graph: its agent is
    :func:`agent_for`'s; ``None`` until :meth:`found`). A call takes the run, the task list its
    conversation carries (the graph's state) and the call's arguments, and answers what the
    model reads and the task records it changed. ``described`` lists each sub-agent with what
    it is for; ``strategies``, what a follow-up to each one's task does with the run at work
    (Deep Agents' ``update_strategy``: ``Mode.INTERRUPT`` by default, or ``Mode.QUEUE``)."""

    def __init__(
        self,
        subagents: Mapping[str, Any],
        *,
        described: Mapping[str, str],
        strategies: Mapping[str, Mode] | None = None,
    ) -> None:
        self.subagents: dict[str, Any] = dict(subagents)
        self.strategies = {name: (strategies or {}).get(name, Mode.INTERRUPT) for name in subagents}
        available = "\n".join(f"- {name}: {what}" for name, what in described.items())
        queued = ", ".join(f"`{n}`" for n, m in self.strategies.items() if m is Mode.QUEUE)
        #: each tool's description, by name
        self.descriptions = {
            task_tools.START: START.format(available_agents=available),
            task_tools.CHECK: CHECK,
            task_tools.UPDATE: UPDATE + (QUEUED.format(names=queued) if queued else ""),
            task_tools.CANCEL: CANCEL,
            task_tools.LIST: LIST,
        }
        #: each tool's work, by name
        self._work: Mapping[str, Callable[..., Awaitable[Answer]]] = {
            task_tools.START: self._start,
            task_tools.CHECK: self._check,
            task_tools.UPDATE: self._update,
            task_tools.CANCEL: self._cancel,
            task_tools.LIST: self._list,
        }

    def found(self, targets: Mapping[str, Any]) -> None:
        """The targets of the sub-agents given none yet, by name (a Deep Agents sub-agent's
        compiled graph, once the graph that runs it is built)."""
        for name in self.subagents:
            if self.subagents[name] is None and name in targets:
                self.subagents[name] = targets[name]

    def agent(self, runtime: Runtime, name: str) -> Agent:
        from trellis.harness.agent import Agent  # noqa: PLC0415 - it imports this module

        found = self.subagents[name]
        return found if isinstance(found, Agent) else agent_for(runtime.agent, name, found)

    async def call(
        self, name: str, runtime: Runtime | None, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        """The tool ``name`` called with ``args`` in ``runtime``'s run, whose conversation
        carries ``tasks``: what the model reads, and the task records the call changed."""
        if runtime is None:
            return NO_RUN, {}
        try:
            return await self._work[name](runtime, tasks, args)
        except Refused as why:
            return str(why), {}

    async def _task(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> tuple[dict[str, Any], RunRecord]:
        """The task a call names and its run of now, when the conversation tracks it and it is
        its own (:func:`owned`); else :class:`Refused`, as a task it does not track."""
        task_id = str(args["task_id"])
        task = tasks.get(task_id.strip())
        tracked = isinstance(task, Mapping) and task.get("agent_name") in self.subagents
        record = await owned(runtime, task) if isinstance(task, Mapping) and tracked else None
        if task is None or record is None:
            raise Refused(UNTRACKED.format(task_id=task_id))
        return dict(task), record

    async def _start(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        name = str(args["subagent_type"])
        if name not in self.subagents:
            allowed = ", ".join(f"`{n}`" for n in self.subagents)
            return f"Unknown async subagent type `{name}`. Available types: {allowed}", {}
        agent = self.agent(runtime, name)
        task = await start(runtime, agent, name, str(args["description"]), tasks)
        return f"Launched async subagent. task_id: {task['task_id']}", {task["task_id"]: task}

    async def _check(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        task, record = await self._task(runtime, tasks, args)
        return json.dumps(result_of(task, record)), {task["task_id"]: observed(task, record)}

    async def _update(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        task, record = await self._task(runtime, tasks, args)
        name, task_id = task["agent_name"], task["task_id"]
        updated = await follow_up(
            runtime,
            self.agent(runtime, name),
            task,
            record,
            str(args["message"]),
            mode=self.strategies[name],
        )
        queued = updated.get("superseded_run_ids") != task.get("superseded_run_ids")
        said = (
            "Queued follow-up to async subagent without interrupting its current run. "
            f"task_id: {task_id}"
            if queued
            else f"Updated async subagent. task_id: {task_id}"
        )
        return said, {task_id: updated}

    async def _cancel(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        task, record = await self._task(runtime, tasks, args)
        await stop(runtime, self.agent(runtime, task["agent_name"]), task, record)
        cancelled = moved(task, run_id=task["run_id"], status="cancelled")
        return f"Cancelled async subagent task: {task['task_id']}", {task["task_id"]: cancelled}

    async def _list(
        self, runtime: Runtime, tasks: Mapping[str, Any], args: Mapping[str, Any]
    ) -> Answer:
        status_filter = args.get("status_filter")
        listed = [dict(t) for t in tasks.values()]
        if status_filter and status_filter != "all":
            listed = [t for t in listed if t["status"] == status_filter]
        if not listed:
            return NONE, {}
        # a task that ended is as its record says: only the unfinished are read
        reading = [t for t in listed if t["status"] in UNFINISHED]
        records = await asyncio.gather(*(owned(runtime, t) for t in reading))
        read = {
            t["task_id"]: observed(t, r)
            for t, r in zip(reading, records, strict=True)
            if r is not None
        }
        updated = {t["task_id"]: read.get(t["task_id"], t) for t in listed}
        entries = [
            f"- task_id: {t['task_id']}  agent: {t['agent_name']}  status: {t['status']}"
            for t in updated.values()
        ]
        return f"{len(entries)} tracked task(s):\n" + "\n".join(entries), updated

    async def asked(self, runtime: Runtime, tasks: Mapping[str, Any]) -> None:
        """The questions ``tasks`` wait on asked in ``runtime``'s turn (:func:`ask`), each
        answer handed on to its task (:func:`answered`)."""
        for task in tasks.values():
            if task.get("status") != "waiting" or task.get("agent_name") not in self.subagents:
                continue
            record = await owned(runtime, task)
            if record is None:
                continue
            resolution = await ask(runtime, record)
            if resolution is not None:
                agent = self.agent(runtime, task["agent_name"])
                await answered(runtime.agent, agent, record, resolution)


async def pulled(
    runtime: Runtime, tasks: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The unfinished ``tasks`` read: their records, updated where their run moved on, and the
    update (:func:`update_of`) of each new outcome — one the conversation was told already
    (through a check, the list, an earlier notification) is not told again. A pull is made at
    every model call, so it asks the run store at most once, however many tasks there are: a
    task whose attempt is at work in this process is running, one whose attempt ended here
    ended as its run did (``Tasks.ended``), and of the rest the one read longest ago is read —
    each in its turn."""
    here = runtime.agent.harness.background
    unfinished = [t for t in tasks.values() if t.get("status") in UNFINISHED]
    working = [t for t in unfinished if here.at_work(t["run_id"])]
    known = [t for t in unfinished if here.ended(t["run_id"]) is not None]
    rest = [t for t in unfinished if t not in working and t not in known]
    oldest = sorted(rest, key=lambda t: (t["last_checked_at"], t["task_id"]))[:1]
    updated: dict[str, Any] = {
        t["task_id"]: moved(t, run_id=t["run_id"], status="running")
        for t in working
        if t["status"] != "running"
    }
    told: list[dict[str, Any]] = []
    for task in [*known, *oldest]:
        record = await owned(runtime, task)
        if record is None:  # none of the conversation's: read, and passed over
            updated[task["task_id"]] = {**task, "last_checked_at": now()}
            continue
        updated[task["task_id"]] = observed(task, record)
        update = update_of(task, record) if status_of(record) != task["status"] else None
        if update is not None:
            told.append(update)
    return updated, told


async def recorded(runtime: Runtime, event: Any, tasks: Mapping[str, Any]) -> dict[str, Any]:
    """The outcome a run's input tells (``event``, its ``TaskEvent``: a turn of the chat's own,
    :func:`notify`) recorded in its task's record, so no read tells it again (Deep Agents'
    ``_record_notification``) — when the run it names is the task's, in ``runtime``'s
    conversation (:func:`owned`): an input is what whoever started the run wrote."""
    task = tasks.get(str(event.get("task_id"))) if isinstance(event, Mapping) else None
    if not isinstance(event, Mapping) or not isinstance(task, Mapping):
        return {}
    if event.get("run_id") != task["run_id"] and task["status"] not in UNFINISHED:
        return {}
    run_id = str(event.get("run_id") or task["run_id"])
    if await owned(runtime, {**task, "run_id": run_id}) is None:
        return {}
    return {task["task_id"]: moved(task, run_id=run_id, status=str(event["status"]))}
