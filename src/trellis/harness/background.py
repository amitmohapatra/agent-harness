"""Background tasks with no server to host: a sub-agent's run that goes on after the turn that
started it replied, while the chat takes new messages, its outcome brought back to the chat —
the harness's own, for an agent whose framework has none (``ReAct``'s sub-agents:
``middleware.BackgroundTasks``), in the shape of Deep Agents' async sub-agents, whose tools,
task record and notification it keeps: the model sees one interface whether a task runs on a
server (``AsyncSubAgent``) or here.

* **The task** is a child run of the turn that started it (``Agent._start``, its parent the
  turn: its tenant, user, documents and what it is without), on a conversation of its own (a
  thread of its own, whose id is Deep Agents' ``task_id`` and ``thread_id``), its metadata
  naming it (:data:`BACKGROUND`: the sub-agent's name, and the agent and chat that started
  it). Its attempts run in this process, apart from the turn (:class:`Tasks`), through the
  pipeline as any run's — its tool calls governed, recorded under its run, its budget and time
  its agent's — and held in the run store meanwhile (``holding``): a server killed with it
  leaves a run the store ends (``holder_lost``), the harness closing cuts it off
  (:data:`CUT_OFF`), and cancelling the turn cancels it (``pipeline._children_ended``,
  ``subagents.cancel_children``: :meth:`Tasks.stop` ends its attempt here). Its record is the
  run store's, so a check, the list, an update or a cancel reads it from any replica.
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
from collections.abc import Awaitable, Callable, Mapping
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
from trellis.harness.adapters.langgraph import ASKED
from trellis.harness.conversation import Mode
from trellis.harness.frontend import FRONTEND
from trellis.harness.identity import conversation_key
from trellis.harness.journal import Journal, content_key
from trellis.runs import ConflictError, LeaseLostError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The metadata key of a background task's run: ``{"name", "agent", "thread"}`` — the
#: sub-agent's name, and the agent and the chat thread that started it (and that chat's front
#: end, :data:`~trellis.harness.frontend.FRONTEND`, when it had one).
BACKGROUND: Final = "trellis_background"
#: The metadata key of a notification run (:func:`notify`): the ``TaskEvent`` it delivers.
NOTIFIED: Final = "trellis_notified"
#: The notification metadata source of a task's update: Deep Agents' (its ``lc_source``).
SOURCE: Final = "async_subagent"
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
}
#: The statuses that can still change: what a pull reads again.
UNFINISHED: Final = frozenset({"pending", "running", "waiting"})
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
#: How often a task at work here looks whether its run ended elsewhere (cancelled on another
#: replica): its attempt is stopped then.
WATCH_SECONDS: Final = 10.0
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
    """``task`` as its run ``record`` says it is now (Deep Agents' ``_observed``)."""
    at, status = now(), status_of(record)
    changed = status != task["status"] or record.run_id != task["run_id"]
    return {
        **task,
        "run_id": record.run_id,
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


def content(update: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A task update as the model reads it, block by block, in Deep Agents' notification's
    shape: the note, the task's header, its result or details, and what to do next — the
    sub-agent's words escaped, so they cannot close the tags and pose as the system."""
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
    blocks.append(_block(f"</task-notification>\n{NEXT[update['status']]}"))
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
    value = asked.question or (asked.tool_call.tool if asked.tool_call is not None else None)
    return [{"id": asked.interrupt_id, "value": value}]


def notification(update: Mapping[str, Any]) -> Any:
    """A task update as the message the model is given: a LangChain ``HumanMessage`` of its
    :func:`content`, marked with langchain-core's notification metadata — source
    :data:`SOURCE`, its data the ``TaskEvent`` — as Deep Agents' notification is."""
    from langchain_core.messages import (  # noqa: PLC0415 - LangChain, an extra's
        HumanMessage,
        add_notification_metadata,
    )

    event = {k: update[k] for k in ("task_id", "agent_name", "status", "run_id")}
    return add_notification_metadata(
        HumanMessage(content=cast(Any, content(update))), source=SOURCE, data=event
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

    def at_work(self, run_id: str) -> bool:
        return run_id in self._at_work

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
        watching = asyncio.get_running_loop().create_task(self._watch(agent, record))
        try:
            await pipeline.attempt(
                agent, record, record.input, journal=journal, resolution=resolution, mode=mode
            )
        except Exception:
            log.exception("background task %s", record.run_id)
        finally:
            watching.cancel()
        found = None if then is None else await runs.get(record.run_id, tenant=tenant)
        if (
            then is not None
            and found is not None
            and (found.final or found.status is RunStatus.PAUSED)
        ):
            try:
                await then(found)
            except Exception:
                log.exception("background task %s: its outcome not delivered", record.run_id)

    async def _watch(self, agent: Agent, record: RunRecord) -> None:
        """Every :data:`WATCH_SECONDS`, whether the run ended elsewhere (a cancel another
        replica took): its attempt here stops then."""
        while True:
            await asyncio.sleep(WATCH_SECONDS)
            found = await agent.harness.runs.get(record.run_id, tenant=record.tenant_id)
            if found is None or found.final:
                self.stop(record.run_id, "ended in the run store")
                return

    def stop(self, run_id: str, reason: str) -> bool:
        """The attempt of ``run_id`` at work here stopped, as ``Agent.cancel`` stops one (its
        run is ended in the store already, or ends ``CANCELLED`` with ``reason``); whether
        one was."""
        agent = self._agents.get(run_id)
        runtime = None if agent is None else agent.running.get(run_id)
        if runtime is None or runtime.running_in is None:
            return False
        runtime.cancelled = runtime.cancelled or reason
        runtime.running_in.cancel()
        return True

    async def aclose(self) -> None:
        """The harness closes: each task still at work is cut off — its run ended ``ERROR``
        (:data:`CUT_OFF`), its attempt stopped."""
        error = AgentError(code=UNFINISHED_CODE, category=ErrorCategory.DEPENDENCY, message=CUT_OFF)
        for run_id, task in list(self._at_work.items()):
            agent = self._agents[run_id]
            runtime = agent.running.get(run_id)
            if runtime is not None:
                with contextlib.suppress(ConflictError, LeaseLostError):  # it ended meanwhile
                    await agent.harness.runs.finish(
                        run_id, RunStatus.ERROR, error=error, tenant=runtime.tenant
                    )
            task.cancel()
        await asyncio.gather(*self._at_work.values(), return_exceptions=True)


async def start(runtime: Runtime, agent: Agent, name: str, description: str) -> dict[str, Any]:
    """A task of ``agent`` (the sub-agent called ``name``) on ``description``, started from
    ``runtime``'s turn: its run started and at work here; its task record."""
    place = runtime.idempotency_key or new_id("call_")
    # the task's conversation (its id) and its first turn: follow-ups are turns of it too
    thread = stable_id(place, BACKGROUND, name, prefix="task_")
    run_id = stable_id(place, BACKGROUND, name, prefix="run_")
    origin = {"name": name, "agent": runtime.agent_id, "thread": runtime.identity.thread}
    if runtime.frontend is not None:
        origin[FRONTEND] = runtime.frontend.kept()
    record = await _started(
        runtime, agent, description, run_id=run_id, thread=thread, origin=origin
    )
    _go(runtime.agent, agent, record)
    return task_of(record, name)


async def follow_up(
    runtime: Runtime,
    agent: Agent,
    task: Mapping[str, Any],
    message: str,
    *,
    strategy: str,
) -> dict[str, Any]:
    """``message`` sent to ``task``: a new run of its conversation, which sees what it did so
    far, under the task's id. ``strategy`` is Deep Agents' ``update_strategy``: ``interrupt``
    stops the run at work first (here, as ``Mode.INTERRUPT`` does; elsewhere, by its
    cancel), ``enqueue`` waits for it (``Mode.QUEUE``). The task's record, updated."""
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    latest = await runs.get(task["run_id"], tenant=tenant)
    origin = dict((latest.metadata if latest is not None else {}).get(BACKGROUND) or {})
    place = runtime.idempotency_key or new_id("call_")
    run_id = stable_id(place, BACKGROUND, task["task_id"], prefix="run_")
    mode = Mode.INTERRUPT if strategy == "interrupt" else Mode.QUEUE
    tasks = runtime.agent.harness.background
    elsewhere = latest is not None and not latest.final and not tasks.at_work(latest.run_id)
    if mode is Mode.INTERRUPT and latest is not None and elsewhere:
        # at work on another replica, or waiting for a person: stopped there
        with contextlib.suppress(ConflictError):  # it ended meanwhile
            await agent.cancel(latest.run_id, reason=INTERRUPTED, tenant=tenant)
    record = await _started(
        runtime, agent, message, run_id=run_id, thread=task["thread_id"], origin=origin
    )
    _go(runtime.agent, agent, record, mode=mode)
    updated = {**observed(task, record), "status": "running", "last_updated_at": now()}
    if mode is Mode.QUEUE and latest is not None and not latest.final:
        updated["superseded_run_ids"] = [*task.get("superseded_run_ids", []), task["run_id"]]
    return updated


#: Why a task's run a follow-up replaced ended (``conversation``'s words for a run a newer
#: message interrupted).
INTERRUPTED: Final = "interrupted by a newer message of the conversation"


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
    own conversation (``thread``), its metadata naming where it came from."""
    started = await agent._start(
        input,
        user=runtime.user,
        thread=thread,
        tenant=runtime.tenant,
        run_id=run_id,
        record_input=True,
        parent=runtime,
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


async def stop(runtime: Runtime, agent: Agent, task: Mapping[str, Any]) -> None:
    """``task`` cancelled: its latest run, and the earlier ones a queued follow-up left behind
    it that still work (newest first, so cancelling one does not let the next begin)."""
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    for run_id in [task["run_id"], *reversed(task.get("superseded_run_ids", []))]:
        found = await runs.get(run_id, tenant=tenant)
        if found is None or found.final:
            continue
        with contextlib.suppress(ConflictError):  # it ended meanwhile
            await agent.cancel(
                run_id, reason="cancelled by the agent that started it", tenant=tenant
            )


# --------------------------------------------------------------------------- its outcome
async def delivered(parent: Agent, record: RunRecord, *, ending: str | None = None) -> None:
    """The run of a task ended or paused: its outcome told to the chat it came from by a turn
    of the chat's own (:func:`notify`) when the chat is idle — a turn at work, or waiting for a
    person, gives its model the outcome itself (at its next model call, after its answer, or at
    the conversation's next turn). ``ending``: the turn of the chat that is ending now, which
    took no notice of it (:func:`left`)."""
    origin = record.metadata.get(BACKGROUND)
    user = record.user_id
    if not isinstance(origin, Mapping) or not origin.get("thread") or not user:
        return
    task = {"task_id": record.thread_id, "agent_name": origin.get("name", "")}
    update = update_of(task, record)
    thread = str(origin["thread"])
    if update is None or not await idle(parent, record.tenant_id, user, thread, ending=ending):
        return
    await notify(parent, record, update, user=user, thread=thread, origin=origin)


async def idle(
    agent: Agent, tenant: str, user: str, thread: str, *, ending: str | None = None
) -> bool:
    """Whether ``user``'s chat ``thread`` with ``agent`` has no turn at work (here, or in the
    run store) and none waiting for a person — where a turn of its own may begin — but the one
    ``ending`` now."""
    live = agent.turns.live(conversation_key(tenant, user, thread))
    if live is not None and live.run_id != ending:
        return False
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
    (its id is the outcome's: a question the task waits on is one), at work here and delivered
    by the chat's thread stream."""
    asked = record.awaiting.interrupt_id if record.awaiting is not None else ""
    run_id = stable_id(
        NOTIFIED, update["task_id"], update["run_id"], update["status"], asked, prefix="run_"
    )
    runs = agent.harness.runs
    if await runs.get(run_id, tenant=record.tenant_id) is not None:
        return  # told already
    started = await agent._start(
        input_of(update),
        user=user,
        thread=thread,
        tenant=record.tenant_id,
        run_id=run_id,
        record_input=True,
    )
    metadata = {
        **started.metadata,
        NOTIFIED: {k: update[k] for k in ("task_id", "agent_name", "status", "run_id")},
    }
    if FRONTEND in origin:  # the chat's front end: a person reads it and answers what it asks
        metadata[FRONTEND] = origin[FRONTEND]
    found = await runs.start(started.model_copy(update={"metadata": metadata}))
    tasks = agent.harness.background
    if found.status is RunStatus.RUNNING and not tasks.at_work(run_id):  # told meanwhile
        tasks.run(agent, found)


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
    runs = agent.harness.runs
    for task in tasks.values() if isinstance(tasks, Mapping) else ():
        if task.get("status") not in UNFINISHED:
            continue
        record = await runs.get(task["run_id"], tenant=runtime.tenant)
        if record is None or BACKGROUND not in record.metadata:
            continue  # not the harness's: a task on a server of Deep Agents'
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
    from trellis.harness.subagents import ASKED, SUBAGENT  # noqa: PLC0415 - it imports this

    asked: Interrupt | None = record.awaiting
    if record.status is not RunStatus.PAUSED or asked is None:
        return None
    origin = record.metadata.get(BACKGROUND) or {}
    named = {
        "agent_id": str(origin.get("name") or record.agent_id),
        "run_id": record.run_id,
        "interrupt_id": asked.interrupt_id,
        "background": True,
    }
    answered = await runtime.interrupt(
        content_key("subagent", record.run_id, asked.interrupt_id),
        payload={**(asked.payload or {}), SUBAGENT: named},
        **{name: getattr(asked, name) for name in ASKED},
    )
    return answered.model_copy(update={"interrupt_id": asked.interrupt_id, "run_id": record.run_id})


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
    journal, resumed = await agent._resumed(found, resolution)
    if resumed.status is not RunStatus.RUNNING:
        return  # cancelled by the answer: it ended there

    async def then(ended: RunRecord) -> None:
        await delivered(parent, ended)

    parent.harness.background.run(agent, resumed, then=then, journal=journal, resolution=resolution)


# --------------------------------------------------------------------------- the five tools
# The model-facing tools, adapter-neutral: Deep Agents' async sub-agent tools by name,
# arguments and words (an adapter gives them its own framework's shape — a LangChain tool,
# ``bridges.background_tasks``; an OpenAI Agents function tool — around :class:`Offered`).
#: The tools, by name, as Deep Agents names its own.
TOOLS: Final = (
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
)
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
#: The tools that hand work on to a sub-agent whose own calls are governed: reads, as the
#: ``task`` hand-over is (``adapters.base.helper``).
HANDS_ON: Final = frozenset({"start_async_task", "update_async_task"})


#: What each tool does: a start and an update hand work on (:data:`HANDS_ON`), a cancel stops
#: work (a write), a check and the list read.
SIDE_EFFECTS: Final[Mapping[str, str]] = {
    name: "read"
    if name in HANDS_ON or name in ("check_async_task", "list_async_tasks")
    else "write"
    for name in TOOLS
}
#: The tools whose result is the record of what was done — the task's id, that it was sent new
#: instructions or stopped — no value to read again.
DONE: Final = frozenset({"start_async_task", "update_async_task", "cancel_async_task"})


class Offered:
    """The five tools' work for ``subagents`` — each by its name: a wrapped agent, or a target
    the parent's framework runs itself (a Deep Agents sub-agent's compiled graph: its agent is
    :func:`agent_for`'s; ``None`` until :meth:`found`) — adapter-neutral: a call takes the run,
    the task list the conversation carries (``Runtime.carried``'s ``async_tasks``, a graph's
    state) and the call's arguments, and returns what the model reads and the task records it
    changed. ``described`` lists each sub-agent with what it is for; ``strategies``, each one's
    Deep Agents ``update_strategy`` (``interrupt`` by default)."""

    def __init__(
        self,
        subagents: Mapping[str, Any],
        *,
        described: Mapping[str, str],
        strategies: Mapping[str, str] | None = None,
    ) -> None:
        self.subagents: dict[str, Any] = dict(subagents)
        self.strategies = {name: (strategies or {}).get(name, "interrupt") for name in subagents}
        available = "\n".join(f"- {name}: {what}" for name, what in described.items())
        queued = ", ".join(f"`{n}`" for n, s in self.strategies.items() if s == "enqueue")
        #: each tool's description, by name
        self.descriptions = {
            "start_async_task": START.format(available_agents=available),
            "check_async_task": CHECK,
            "update_async_task": UPDATE + (QUEUED.format(names=queued) if queued else ""),
            "cancel_async_task": CANCEL,
            "list_async_tasks": LIST,
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
    ) -> tuple[str, dict[str, Any]]:
        """The tool ``name`` called with ``args`` in ``runtime``'s run, whose conversation
        carries ``tasks``: what the model reads, and the task records the call changed."""
        if runtime is None:
            return NO_RUN, {}
        if name == "start_async_task":
            return await self._start(runtime, str(args["subagent_type"]), str(args["description"]))
        if name == "list_async_tasks":
            return await self._list(runtime, tasks, args.get("status_filter"))
        task_id = str(args["task_id"])
        found = tasks.get(task_id.strip())
        if not isinstance(found, Mapping) or found.get("agent_name") not in self.subagents:
            return UNTRACKED.format(task_id=task_id), {}
        task = dict(found)
        if name == "update_async_task":
            return await self._update(runtime, task, str(args["message"]))
        own = self._check if name == "check_async_task" else self._cancel
        return await own(runtime, task)

    async def _start(
        self, runtime: Runtime, name: str, description: str
    ) -> tuple[str, dict[str, Any]]:
        if name not in self.subagents:
            allowed = ", ".join(f"`{n}`" for n in self.subagents)
            return f"Unknown async subagent type `{name}`. Available types: {allowed}", {}
        task = await start(runtime, self.agent(runtime, name), name, description)
        return f"Launched async subagent. task_id: {task['task_id']}", {task["task_id"]: task}

    async def _check(self, runtime: Runtime, task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        record = await runtime.agent.harness.runs.get(task["run_id"], tenant=runtime.tenant)
        if record is None:
            return UNTRACKED.format(task_id=task["task_id"]), {}
        said = json.dumps(result_of(task, record))
        return said, {task["task_id"]: observed(task, record)}

    async def _update(
        self, runtime: Runtime, task: dict[str, Any], message: str
    ) -> tuple[str, dict[str, Any]]:
        name, task_id = task["agent_name"], task["task_id"]
        strategy = self.strategies[name]
        updated = await follow_up(
            runtime, self.agent(runtime, name), task, message, strategy=strategy
        )
        queued = "superseded_run_ids" in updated and strategy == "enqueue"
        said = (
            "Queued follow-up to async subagent without interrupting its current run. "
            f"task_id: {task_id}"
            if queued
            else f"Updated async subagent. task_id: {task_id}"
        )
        return said, {task_id: updated}

    async def _cancel(self, runtime: Runtime, task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        await stop(runtime, self.agent(runtime, task["agent_name"]), task)
        at = now()
        cancelled = {**task, "status": "cancelled", "last_checked_at": at, "last_updated_at": at}
        return f"Cancelled async subagent task: {task['task_id']}", {task["task_id"]: cancelled}

    async def _list(
        self, runtime: Runtime, tasks: Mapping[str, Any], status_filter: Any
    ) -> tuple[str, dict[str, Any]]:
        listed = [dict(t) for t in tasks.values()]
        if status_filter and status_filter != "all":
            listed = [t for t in listed if t["status"] == status_filter]
        if not listed:
            return NONE, {}
        runs = runtime.agent.harness.runs
        records = await asyncio.gather(
            *(runs.get(t["run_id"], tenant=runtime.tenant) for t in listed)
        )
        updated = {
            t["task_id"]: t if r is None else observed(t, r)
            for t, r in zip(listed, records, strict=True)
        }
        entries = [
            f"- task_id: {t['task_id']}  agent: {t['agent_name']}  status: {t['status']}"
            for t in updated.values()
        ]
        return f"{len(entries)} tracked task(s):\n" + "\n".join(entries), updated

    async def asked(self, runtime: Runtime, tasks: Mapping[str, Any]) -> None:
        """The questions ``tasks`` wait on asked in ``runtime``'s turn (:func:`ask`), each
        answer handed on to its task (:func:`answered`)."""
        runs = runtime.agent.harness.runs
        for task in tasks.values():
            if task.get("status") != "waiting" or task.get("agent_name") not in self.subagents:
                continue
            record = await runs.get(task["run_id"], tenant=runtime.tenant)
            if record is None:
                continue
            resolution = await ask(runtime, record)
            if resolution is not None:
                agent = self.agent(runtime, task["agent_name"])
                await answered(runtime.agent, agent, record, resolution)


async def pulled(
    runtime: Runtime, tasks: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The unfinished ``tasks`` read in the run store: their records, updated where their run
    moved on, and the update (:func:`update_of`) of each new outcome — one the conversation was
    told already (through a check, the list, an earlier notification) is not told again."""
    unfinished = [t for t in tasks.values() if t.get("status") in UNFINISHED]
    runs = runtime.agent.harness.runs
    records = await asyncio.gather(
        *(runs.get(t["run_id"], tenant=runtime.tenant) for t in unfinished)
    )
    updated: dict[str, Any] = {}
    told: list[dict[str, Any]] = []
    for task, record in zip(unfinished, records, strict=True):
        if record is None:
            continue
        status = status_of(record)
        if status == task["status"] and record.run_id == task["run_id"]:
            continue
        updated[task["task_id"]] = observed(task, record)
        update = update_of(task, record) if status != task["status"] else None
        if update is not None:
            told.append(update)
    return updated, told


def recorded(event: Any, tasks: Mapping[str, Any]) -> dict[str, Any]:
    """The outcome a run's input tells (``event``, its ``TaskEvent``: a turn of the chat's own,
    :func:`notify`) recorded in its task's record, so no read tells it again (Deep Agents'
    ``_record_notification``)."""
    task = tasks.get(str(event.get("task_id"))) if isinstance(event, Mapping) else None
    if not isinstance(event, Mapping) or not isinstance(task, Mapping):
        return {}
    if event.get("run_id") != task["run_id"] and task["status"] not in UNFINISHED:
        return {}
    at = now()
    status, run_id = str(event["status"]), str(event.get("run_id") or task["run_id"])
    changed = status != task["status"] or run_id != task["run_id"]
    return {
        task["task_id"]: {
            **task,
            "run_id": run_id,
            "status": status,
            "last_checked_at": at,
            "last_updated_at": at if changed else task["last_updated_at"],
        }
    }
