"""A conversation's Claude Code process, kept between its turns.

The Claude Agent SDK's background work — a subagent with ``background``, a shell command run
in the background — belongs to the CLI process that started it: it goes on after the turn's
``ResultMessage`` and ends with the process. So a conversation's turn does not end its CLI:
the process stays (:class:`LiveSession`, the SDK's own ``ClaudeSDKClient`` in streaming-input
mode), a reader goes on consuming its stream, and the conversation's next turn is sent into
it (``client.query``) instead of into a new process that resumes the session.

* **Whose turn.** A harness turn ends at the result of its own prompt — one whose ``origin``
  is none or ``human`` — not when the process goes idle. A turn the session injects by itself
  (a finished task's notification: ``origin.kind == "task-notification"``) is never a user's
  answer: its messages and its result are not given to the turn being served.
* **Whose run.** The process is started once, with one set of hooks, one ``can_use_tool`` and
  one ``trellis`` MCP server, while each turn has its own (bound to its run). The process is
  given relays: each hands the CLI's request to the turn being served, in a copy of that
  turn's context — ``trellis.current()``, the trace and the tool's own run are that turn's.
* **A background subagent's run.** A subagent the CLI runs in the background
  (``task_started`` with ``is_backgrounded``) keeps the child run its ``Agent`` call started
  (``bridges.helpers``) open until its task ends (:class:`Background`), holding it in the run
  store meanwhile (``holding``), so that a server killed with the session (nothing closed)
  leaves a run the store ends once the hold lapses (agent-runs' ticker: ``ERROR``,
  ``holder_lost``), not one ``RUNNING`` for ever: that run's attempts
  go through the pipeline as any run's, and the subagent's hooks, permission requests and
  harness tool calls — which the CLI names it in (``agent_id``: the task's id) — are
  relayed to the attempt under way, turn or no turn. A call that asks a person pauses that
  run as any run pauses, and waits in the CLI's callback: the question is asked in the
  conversation's next turn (as a sub-agent's is asked through its parent), and the attempt
  that follows the answer decides the call from its journal. A dynamic workflow
  (``local_workflow``) is kept so too: its agents name only their kind
  (:data:`WORKFLOW_AGENT`), and one workflow works at a time in the session
  (:data:`ONE_WORKFLOW`), so their calls are its run's.
* **No run.** Any other call made between two turns — the main agent's in a turn the
  session injected, a subagent's with no run — has no run under way: nothing can decide it
  (the run's hooks, governance, a person's approval, its budget) or record it. Such a call is
  refused (:data:`UNATTENDED`), never run unchecked, and the conversation's next run says
  how many were. Messages that arrive then are kept, the most recent
  (:func:`unattended` — the one place a later change delivers them from); Claude's own
  session holds them already, so the next turn's model knows them.
* **A turn that is left** (its run paused, was cancelled, stopped reading) is not stopped
  with the SDK's ``interrupt`` while background work is under way: the CLI stops that work
  with the turn. The turn ends by itself instead: its next tool call is refused with the
  hook's ``continue: false`` (its permission check's ``interrupt``), which end a turn and
  leave the background work. With nothing in the background it is interrupted at once.
* **One prompt at a time.** A prompt sent while a turn is under way may be given to that
  turn (the CLI hands it over at the turn's next tool call) and then has no result of its
  own: a prompt is sent only once the result of the one before came, and no turn of the
  session's own is seen under way (:meth:`LiveSession.settled`).
* **Spending.** A result's cost is the process's running total; its usage is the turn's own.
  A turn left before its result is charged that result when it comes at once (interrupted:
  :meth:`LiveSession.outcome`); never the conversation's next turn.
  What the cost grew by since the last result charged (:meth:`LiveSession.charged`) is
  shared by the tokens each used: the turn the result ends its own usage, each background
  run what its subagent's replies used since (the share of a turn the session injected
  for an ended task is that task's run's). With no background run, what injected turns and
  background shell commands cost is on the conversation's next turn, as their tokens are in
  no turn's usage.
* **Its life.** One process per conversation, in this process only; at most :data:`KEPT`. It
  closes :data:`IDLE_SECONDS` after its last activity when nothing works in it (no background
  task, and no turn of its own for a finished one under way or due), and
  :data:`WORKING_SECONDS` after its last turn whatever does; when a turn needs other options
  than it was started with; and with the harness (``Harness.aclose``). Work that ends with
  it is said in the conversation's next run (``claude_session``), never dropped in silence.
  With no live session — a first turn, another server, a restart, one closed — a turn runs as
  it always did: a new process that resumes the conversation's session (``resume=``) from
  the journal's transcripts, which hold what was mirrored between turns too.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import logging
import time
from collections import Counter, deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import Interrupt, InterruptResolution, RunStatus
from trellis.harness import budget
from trellis.harness.asking import RunCancelled
from trellis.harness.bounded import LRU
from trellis.harness.bridges.claude import JournalSessions, ours
from trellis.harness.bridges.helpers import Kind
from trellis.harness.journal import Journal, content_key
from trellis.runs import ConflictError

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: How long a session nothing works in is kept after its last activity.
IDLE_SECONDS: Final = 300.0
#: How long a session is kept after its last turn, whatever still works in it.
WORKING_SECONDS: Final = 6 * 3600.0
#: The most sessions an agent keeps in this process (each is a CLI process).
KEPT: Final = 16
#: How long a turn waits for the result of an earlier turn that was left (paused, cancelled)
#: before its own prompt is sent; past it the session is closed and the turn runs without.
SETTLE_SECONDS: Final = 30.0
#: The most messages kept of those that arrived with no turn to take them (the most recent).
UNATTENDED_KEPT: Final = 200
#: What the main agent's call in a turn that was left is refused with: the turn ends there.
LEFT: Final = "The user left this turn: it ends here."
#: What a tool call made while no run of the conversation is under way is answered.
UNATTENDED: Final = (
    "This call did not run: no run of the conversation is under way, so nothing can check or "
    "record it. It can be made again once the user's next message is being answered."
)
#: Why a turn whose session's process ended before its result failed.
ENDED: Final = "Claude Code's process ended before the turn's result"
#: How long a background subagent's run waits for a person's answer before it is given up
#: (cancelled, saying so), and how long the CLI is told to wait for a hook's answer (twice
#: that: the harness answers first, and a call the CLI gave up on is one it did not run).
ANSWER_SECONDS: Final = 12 * 3600.0
HOOK_SECONDS: Final = 2 * ANSWER_SECONDS
#: How often a background run that waits for a person looks whether it was cancelled there.
WAITS_SECONDS: Final = 10.0
#: Why a background subagent's run ended with its session, when no person answered it, and
#: when its task was stopped in Claude Code (by Claude, or with the turn it was started in).
LOST: Final = (
    "Claude's session was closed ({why}): the background subagent's task was stopped with it"
)
UNANSWERED: Final = "nobody answered the background subagent's question in {hours:g} hours"
STOPPED: Final = "the background subagent's task was stopped in Claude Code"
CANCELLED_FIRST: Final = "the background subagent's run was cancelled before it began"
#: Why a background subagent's run whose question was answered on a server its session is not
#: on ended (its task went with the server that held the session, or lives on another): its
#: error's message, and what the turn that took the answer is told.
GONE: Final = (
    "the background subagent's Claude session is not on the server the answer reached (it "
    "ended with the server that held it, or lives on another): the answer cannot continue its "
    "task, which is gone from here"
)
GONE_SAID: Final = (
    "A background subagent's question was answered, but its Claude session is gone (the "
    "server that held it stopped): its task cannot go on, and its run ended."
)
#: Claude Code's dynamic workflow, as the bundled CLI (2.1.283) sends it: its tool, the task it
#: runs as (``task_type``: no ``is_backgrounded``, it is always in the background), and what
#: its agents are named in their hooks (``agent_type``; each ``agent_id`` is the agent's own,
#: named in no frame before its first call). One works at a time in a session
#: (:data:`ONE_WORKFLOW`), so its agents' calls are its run's by their kind.
WORKFLOW_TOOL: Final = "Workflow"
WORKFLOW_TASK: Final = "local_workflow"
WORKFLOW_AGENT: Final = "workflow-subagent"
#: What a second workflow's launch is refused with while one works in the session.
ONE_WORKFLOW: Final = (
    "A workflow is already running in this session: one runs at a time. Wait for its "
    "task-notification before starting another, or stop it with TaskStop."
)
#: What a background subagent's run answers when its task ended with no word of its own.
COMPLETED: Final = "The task completed."
#: Why a background subagent's run whose task failed without saying why ended.
FAILED: Final = "the background subagent's task failed"
#: What an attempt of a background run is nudged with: its run paused, or its task ended.
_NUDGE: Final = object()
#: What a request relayed to a background run is answered when that run has ended.
_OVER: Final = object()


def backgrounded(message: Any) -> bool:
    """Whether the task a ``task_started`` ``message`` starts works in the background of its
    call: a subagent that says so (``is_backgrounded``), and a dynamic workflow, always."""
    data = message.data
    return bool(data.get("is_backgrounded")) or data.get("task_type") == WORKFLOW_TASK


class Left(Exception):
    """The turn a request was handed to was left before it answered."""


async def gone(runtime: Runtime, run_id: str, interrupt_id: str, answer: Any) -> None:
    """The person's ``answer`` to the question the background subagent's run ``run_id`` is
    paused on (``interrupt_id``) reached ``runtime``'s turn on a server its session is not on:
    it ended with the server that held it (a crash, a restart) or lives on another, and the
    subagent's task with it — nothing here can continue it. The run records the answer and
    ends ``ERROR`` (``holder_lost``: :data:`GONE`), and the turn is told
    (``claude_session``: :data:`GONE_SAID`). A run no longer waiting on that question (it
    ended, or was answered already) is left as it is."""
    from trellis.contracts import AgentError, ErrorCategory

    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    record = await runs.get(run_id, tenant=tenant)
    if record is None or record.awaiting is None or record.awaiting.interrupt_id != interrupt_id:
        return
    resumed = await runs.resume(
        answer.model_copy(update={"interrupt_id": interrupt_id, "run_id": run_id}), tenant=tenant
    )
    if not resumed.final:  # not a CANCEL, which ended it already
        error = AgentError(
            code="holder_lost",
            category=ErrorCategory.DEPENDENCY,
            message=GONE,
            retryable=False,
            source="claude_agent_sdk",
        )
        await runs.finish(run_id, RunStatus.ERROR, error=error, tenant=tenant)
    runtime.events.warning("claude_session", GONE_SAID)


def unattended(session: LiveSession, message: Any) -> None:
    """A message that arrived while no turn could take it — a background task's progress,
    its notification, the turn the session injected for it and that turn's result: kept, the
    most recent :data:`UNATTENDED_KEPT`. The one place such messages pass: delivering them
    into the conversation is built from here."""
    session.kept.append(message)


@dataclasses.dataclass(eq=False)
class Turn:
    """The harness turn a session serves now: its run's own options (hooks, ``can_use_tool``,
    tool server), and the context its requests run in."""

    runtime: Any
    options: Any
    #: the run's own ``trellis`` server (``None``: it has no harness tool)
    tools: Any
    context: contextvars.Context
    queue: asyncio.Queue[Any] = dataclasses.field(default_factory=asyncio.Queue)
    #: its prompt was sent: the messages that follow are its own
    listening: bool = False
    #: its own result came
    answered: bool = False
    #: it was left before its result and the CLI's turn interrupted: its result comes at once
    stopped: bool = False
    calls: set[asyncio.Task[Any]] = dataclasses.field(default_factory=set)

    async def within[T](self, work: Coroutine[Any, Any, T]) -> T:
        """``work`` run in a copy of the turn's context (its run is ``trellis.current()``):
        its result, or :class:`Left` when the turn was left meanwhile (what is under way then
        is cancelled with it)."""
        task = asyncio.get_running_loop().create_task(work, context=self.context.copy())
        self.calls.add(task)
        task.add_done_callback(self.calls.discard)
        await asyncio.wait({task})
        if task.cancelled():
            raise Left
        return task.result()


def _tokens(usage: Mapping[str, Any] | None) -> int:
    """The tokens a reply (or a turn's result) says it used, read and written."""
    used = usage or {}
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(int(used.get(key) or 0) for key in (*keys, "output_tokens"))


def _waits(answer: Any) -> bool:
    """Whether a relayed request was answered that its run pauses on it — the adapter's
    hook (its ``stopReason``), its permission check (its message), a harness tool (its
    result)."""
    from trellis.harness.tools.convert.claude import WAITING

    if not isinstance(answer, dict):
        return getattr(answer, "message", None) == WAITING
    content = answer.get("content")
    said = isinstance(content, list) and any(c.get("text") == WAITING for c in content)
    return answer.get("stopReason") == WAITING or (bool(answer.get("is_error")) and said)


class Background:
    """A subagent at work in the background of a session, and the child run that is its
    record: the helper its ``Agent`` call started (``bridges.helpers``), kept open from
    there until the SDK says the task ended.

    The run's attempts go through the pipeline (:meth:`_run`): each is served the
    subagent's own messages (:meth:`served`) and ends with a result made of the task's end —
    ``SUCCESS`` with what the subagent said last, ``ERROR`` for a task that failed,
    ``CANCELLED`` for one stopped in Claude Code — or as any run ends: paused, cancelled,
    out of time, its budget spent (its task is stopped then with the SDK's own
    ``stop_task``: an interrupt would stop every task). The requests the CLI makes for the
    subagent are done in the attempt under way (:meth:`asked`); one the run pauses on waits
    there for the attempt that follows the person's answer."""

    def __init__(self, session: LiveSession, task_id: str, call_id: str, helper: Any) -> None:
        turn = session.turn
        assert turn is not None  # the turn that started it is being served
        self.session = session
        #: the SDK's id of the task: the ``agent_id`` of the subagent's hooks and requests
        self.task_id = task_id
        #: the ``Agent`` call that started it: its messages' ``parent_tool_use_id``
        self.call_id = call_id
        self.run_id: str = helper.run_id
        self.parent_run_id: str = helper.parent_run_id
        self.name: str = helper.name
        #: how its call started it (``Kind.WORKFLOW``: a dynamic workflow's agents work in it)
        self.kind = helper.kind
        #: when its ``Agent`` call started it (``time.monotonic``)
        self.began: float = helper.began
        self.agent = turn.runtime.agent
        self.tenant = turn.runtime.tenant
        #: the framework options of the turn that started it: what the process has
        self.given = dict(turn.runtime.framework_options)
        #: the attempt under way
        self.turn: Turn | None = None
        #: the run has ended
        self.over = False
        #: how the task ended, as the SDK said: its status and summary
        self.ended: tuple[str, str] | None = None
        #: why the run cannot go on (:meth:`abandoned`)
        self.lost: str | None = None
        #: why it was cancelled with the run that started it (:meth:`cancel`)
        self.stopped: str | None = None
        #: what the subagent said last
        self.said = ""
        #: the question the run is paused on, until the conversation's next turn asked it
        self.question: Interrupt | None = None
        #: the tokens its replies used since the process's cost was last shared out
        self.unbilled = 0
        #: the cost charged while no attempt was under way, and the replies' usage
        self._owed: list[tuple[Mapping[str, Any] | None, float]] = []
        self._backlog: list[Any] = []
        self._counted: set[str] = set()
        self._waiting: list[asyncio.Future[None]] = []
        self._settled = asyncio.Event()
        self._answer: asyncio.Future[InterruptResolution | None] | None = None
        # held from here until its run ends (its attempts are held too, as every run's)
        self.agent.harness.holds.hold(self.run_id, self.tenant)
        self.task = asyncio.get_running_loop().create_task(
            self._run(), context=contextvars.Context()
        )

    # ------------------------------------------------------------------ its run
    async def _run(self) -> None:
        """The run's attempts, one after each answer, until it ends."""
        from trellis.harness import pipeline

        agent, runs = self.agent, self.agent.harness.runs
        try:
            record = await runs.get(self.run_id, tenant=self.tenant)
            assert record is not None  # its ``Agent`` call started it
            result = await pipeline.attempt(agent, record, record.input)
            while result.status is RunStatus.PAUSED:
                self.question = result.interrupt
                resolution = await self._answered()
                self.question = None
                paused = await runs.get(self.run_id, tenant=self.tenant)
                if resolution is None or paused is None or paused.status is not RunStatus.PAUSED:
                    break
                # as any paused run goes on (one the person abandons ends there)
                result = await agent._continue(paused, resolution)
        except Exception:
            log.exception("background run %s of task %s", self.run_id, self.task_id)
        finally:
            self.agent.harness.holds.let_go(self.run_id)
            self.over = True
            self._woken()
            self.session._over(self)
            await self._finished()

    async def _finished(self) -> None:
        """The run's end on the task tree of the run that started it, while that run is
        still under way (``helper_finished``, as a subagent that ends with its call says
        it); once it has ended there is no stream of it to say so on."""
        parent = self.agent.running.get(self.parent_run_id)
        record = await self.agent.harness.runs.get(self.run_id, tenant=self.tenant)
        if parent is None or record is None or not record.status.final:
            return
        parent.events.helper(
            "finished",
            run_id=self.run_id,
            agent_id=self.name,
            framework=self.agent.adapter.name,
            status=record.status.value.lower(),
            seconds=round(time.monotonic() - self.began, 3),
        )

    async def _answered(self) -> InterruptResolution | None:
        """The person's answer to the question the run is paused on (``None``: the run ended
        there — cancelled, :meth:`abandoned`)."""
        loop = asyncio.get_running_loop()
        self._answer, asked = loop.create_future(), loop.time()
        try:
            while True:
                done, _ = await asyncio.wait({self._answer}, timeout=WAITS_SECONDS)
                if done:
                    return self._answer.result()
                record = await self.agent.harness.runs.get(self.run_id, tenant=self.tenant)
                if record is None or record.status is not RunStatus.PAUSED:
                    return None  # cancelled where it waited (``agent.cancel``)
                if loop.time() - asked > ANSWER_SECONDS:
                    await self.abandoned(UNANSWERED.format(hours=ANSWER_SECONDS / 3600))
        finally:
            self._answer = None

    def answered(self, resolution: InterruptResolution) -> None:
        """The person answered the run's question (asked in the conversation's turn)."""
        self.question = None
        if self._answer is not None and not self._answer.done():
            self._answer.set_result(resolution)

    async def abandoned(self, why: str) -> None:
        """The run cannot go on (its session closed, nobody answered it, its task was stopped
        while it waited): an attempt under way fails saying ``why``, a run that waits for a
        person is cancelled with ``why`` as the reason."""
        self.lost = why
        self._settled.set()
        if self.turn is not None:
            self.turn.queue.put_nowait(None)
        elif self._answer is not None and not self._answer.done():
            try:
                with contextlib.suppress(ConflictError):  # it ended meanwhile
                    await self.agent.harness.runs.cancel(
                        self.run_id, reason=why, tenant=self.tenant
                    )
            finally:  # it waits no more, whether or not the run store took the cancel
                if self._answer is not None and not self._answer.done():
                    self._answer.set_result(None)

    def cancel(self, reason: str) -> None:
        """The run cancelled with the run that started it: its attempt under way as
        ``Agent.cancel`` cancels a run's, and one that has not begun to serve it does not."""
        self.stopped = reason
        runtime = self.agent.running.get(self.run_id)
        if runtime is not None and runtime.running_in is not None:
            runtime.cancelled = reason
            runtime.running_in.cancel()

    # ------------------------------------------------------------------ an attempt
    async def served(self, turn: Turn) -> AsyncGenerator[Any, None]:
        """One attempt of the run: the subagent's messages while its task works, then the
        result of the task's end. It ends before that when the run pauses (the call that
        asked waits: :meth:`asked`)."""
        if self.lost is not None:
            raise RuntimeError(self.lost)
        # cancelled while this attempt was being set up: with the run that started it, or
        # in the run store (``Agent.cancel`` found no attempt of it yet)
        record = await self.agent.harness.runs.get(self.run_id, tenant=self.tenant)
        if self.stopped is not None or record is None or record.final:
            raise RunCancelled(self.stopped or CANCELLED_FIRST)
        self._attach(turn)
        try:
            while True:
                message = await turn.queue.get()
                if message is None:
                    raise RuntimeError(self.lost or ENDED)
                if turn.runtime.pending is not None:
                    if message is not _NUDGE:  # kept for the attempt that follows
                        self._backlog.append(message)
                    return
                if message is not _NUDGE:
                    yield message
                elif self.ended is not None:
                    break
            # what the task cost is known at the session's next result (``LiveSession.charged``)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(SETTLE_SECONDS):
                    await self._settled.wait()
            yield self._result(turn.runtime)
        finally:
            under_way = list(turn.calls)
            self._detach(turn)
            if under_way:  # they end with the attempt, before its run does (as a turn's)
                await asyncio.wait(under_way)

    def _attach(self, turn: Turn) -> None:
        """``turn`` is the attempt under way: it has what came while there was none, and the
        requests that waited for it go on."""
        self.turn = turn
        for message in self._backlog:
            turn.queue.put_nowait(message)
        self._backlog.clear()
        owed, self._owed = self._owed, []
        for usage, cost in owed:
            self.spent(usage, cost)
        if self.ended is not None:
            turn.queue.put_nowait(_NUDGE)
        self._woken()

    def _detach(self, turn: Turn) -> None:
        """The attempt ended: its requests under way end with it, and what it did not take
        is kept for the next."""
        for call in list(turn.calls):
            call.cancel()
        self.turn = None
        while not turn.queue.empty():
            message = turn.queue.get_nowait()
            if message is not None and message is not _NUDGE:
                self._backlog.append(message)

    def _result(self, runtime: Runtime) -> Any:
        """The result an attempt ends with, made of the task's end."""
        from claude_agent_sdk import ResultMessage

        assert self.ended is not None
        status, summary = self.ended
        ended: dict[str, Any] = {"duration_ms": 0, "duration_api_ms": 0, "num_turns": 0}
        if status == "completed" or runtime.budget_spent is not None:
            said = self.said or summary or COMPLETED
            return ResultMessage("success", is_error=False, session_id="", result=said, **ended)
        if status == "failed":
            why = [summary or FAILED]
            return ResultMessage("failed", is_error=True, session_id="", errors=why, **ended)
        raise RunCancelled(STOPPED)

    def took(self, message: Any) -> None:
        """One of the subagent's own messages: its reply's usage on the run, the message to
        the attempt under way (kept for the next when there is none)."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        if self.over:  # its task is being stopped
            return
        if isinstance(message, AssistantMessage):
            self.said = "".join(b.text for b in message.content if isinstance(b, TextBlock)) or (
                self.said
            )
            # a reply comes as one message a block, each with the reply's id and usage
            if message.usage and message.message_id not in self._counted:
                if message.message_id:
                    self._counted.add(message.message_id)
                self.unbilled += _tokens(message.usage)
                self.spent(message.usage, 0.0)
        if self.turn is not None:
            self.turn.queue.put_nowait(message)
        else:
            self._backlog.append(message)

    def finished(self, status: str, summary: str) -> None:
        """The SDK said the task ended (its notification, or an update to a terminal status:
        either may be all that comes, and the notification has the summary)."""
        self.ended = (status, summary or (self.ended or ("", ""))[1])
        if self.turn is not None:
            self.turn.queue.put_nowait(_NUDGE)
        elif self._answer is not None:  # it waits for an answer about a call that is no more
            self.session._aside_run(self.abandoned(STOPPED))

    def spent(self, usage: Mapping[str, Any] | None, cost: float) -> None:
        """A reply's ``usage``, or a share of the process's ``cost``, on the run — the
        attempt under way, else the next — and on its budget: spent, the task is stopped."""
        turn = self.turn
        if turn is None:
            self._owed.append((usage, cost))
            return
        used = usage or {}
        cached = int(used.get("cache_read_input_tokens") or 0)
        output = int(used.get("output_tokens") or 0)
        budget.account(
            turn.runtime,
            model=turn.options.model,
            input_tokens=_tokens(used) - output,
            output_tokens=output,
            cached=cached,
            cost=cost,
            calls=1 if usage else 0,
        )
        reached = budget.reached(turn.runtime)
        if reached is not None and turn.runtime.budget_spent is None:
            budget.stopped(turn.runtime, reached)
            self.session._aside_run(self.session._stop(self.task_id))

    # ------------------------------------------------------------------ its requests
    async def asked(self, make: Callable[[Turn], Coroutine[Any, Any, Any]]) -> Any:
        """A request the CLI makes for the subagent — a hook, a permission check, a harness
        tool's call — done by ``make`` in the attempt under way, in its context. When the
        run pauses on it, the request waits and is done again in the attempt that follows
        the person's answer, whose journal answers it. :data:`_OVER`: the run has ended."""
        while True:
            turn = await self._attempt()
            if turn is None:
                return _OVER
            try:
                answer = await turn.within(make(turn))
            except Left:
                continue
            if turn.runtime.pending is None or not _waits(answer):
                return answer
            turn.queue.put_nowait(_NUDGE)  # the attempt ends paused

    async def _attempt(self) -> Turn | None:
        """The attempt under way, waited for (``None``: the run ended): after a pause, the
        one that follows the answer."""
        while not self.over:
            turn = self.turn
            if turn is not None and turn.runtime.pending is None:
                return turn
            waiter = asyncio.get_running_loop().create_future()
            self._waiting.append(waiter)
            await waiter
        return None

    def _woken(self) -> None:
        waiting, self._waiting = self._waiting, []
        for waiter in waiting:
            if not waiter.done():
                waiter.set_result(None)


class Mirror:
    """The session's ``SessionStore``: every entry the CLI mirrors, kept for the session's
    life, and in the journal of the turn being served — whose run a pause or a crash resumes
    from there, on any worker."""

    def __init__(self, transcripts: Mapping[str, list[dict[str, Any]]]) -> None:
        journal = Journal()
        journal.transcripts = {name: list(entries) for name, entries in transcripts.items()}
        self.all = JournalSessions(journal)
        #: the journal of the turn being served
        self.into: JournalSessions | None = None

    @property
    def transcripts(self) -> dict[str, list[dict[str, Any]]]:
        return self.all.journal.transcripts

    async def append(self, key: Mapping[str, Any], entries: list[dict[str, Any]]) -> None:
        await self.all.append(key, entries)
        if self.into is not None:
            await self.into.append(key, entries)

    async def load(self, key: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        return await self.all.load(key)

    def serve(self, store: JournalSessions) -> None:
        """The turn's journal has what was mirrored so far, and what is mirrored from now."""
        self.into = store
        for name, entries in self.transcripts.items():
            kept = store.journal.transcripts.setdefault(name, [])
            seen = {e.get("uuid") for e in kept}
            kept.extend(e for e in entries if e.get("uuid") is None or e["uuid"] not in seen)


class LiveSession:
    """One conversation's CLI process and what reads it."""

    def __init__(self, key: str, shape: Any, sessions: LiveSessions) -> None:
        self.key = key
        #: what the process was started with and has for good (``adapters.claude._shape``)
        self.shape = shape
        self.sessions = sessions
        self.client: Any = None
        self.session_id: str | None = None
        self.turn: Turn | None = None
        self.mirror: Mirror | None = None
        #: the background tasks under way, by id: what each is
        self.working: dict[str, str] = {}
        #: the background subagents that have a run, by their task's id
        self.background: dict[str, Background] = {}
        #: the workflow's agents (by their own ``agent_id``) whose calls its run decides
        self._workers: dict[str, Background] = {}
        #: the harness tool calls their hook let through, by call: whose each is
        self._calling: dict[str, tuple[str, dict[str, Any], Background]] = {}
        #: why the session was closed (``LiveSessions.forget``)
        self.why = "the harness was closed"
        #: the messages no turn took (:func:`unattended`)
        self.kept: deque[Any] = deque(maxlen=UNATTENDED_KEPT)
        #: the tool calls refused since the last turn for want of a run, by tool
        self.refused: Counter[str] = Counter()
        #: the prompts sent whose result has not come (one at most)
        self.owed = 0
        #: a turn the session injected is under way
        self.injected = False
        #: a background task ended while no turn was served: the session's own turn for it is
        #: due (Claude Code runs it as soon as it gets to it), until that turn begins — waited
        #: for :data:`SETTLE_SECONDS` at least
        self.due = False
        self.closed = False
        #: why the process ended, when it ended of itself with an error
        self.error: BaseException | None = None
        #: the process's running cost as the last charged result carried it
        self.billed = 0.0
        #: the turn left before its own result (paused, cancelled): the result that comes
        #: next is its own, what it spent its run's — never the conversation's next turn's
        self.left: Turn | None = None
        #: where the conversation's session is kept for a turn with no live one
        #: (``Agent.sessions``), once a turn was served
        self.kept_in: LRU[str, Any] | None = None
        self._settled = asyncio.Event()
        self._settled.set()
        self._reader: asyncio.Task[None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._rested = 0.0
        self._aside: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------ its start
    async def start(self, options: Any, transcripts: Mapping[str, list[dict[str, Any]]]) -> None:
        """The process, started with ``options`` — its hooks, permission check and harness
        tools relayed to the turn being served — in a context of no run, so nothing of the
        turn that happened to start it stays with the tasks that read it."""
        from claude_agent_sdk import ClaudeSDKClient

        from trellis.harness.tools.convert import claude as convert

        hooks = self._hooks(options.hooks)
        changes: dict[str, Any] = {"hooks": hooks, "can_use_tool": self._can_use_tool}
        if isinstance(options.session_store, JournalSessions):
            self.mirror = Mirror(transcripts)
            changes["session_store"] = self.mirror
        servers = options.mcp_servers
        if isinstance(servers, dict) and convert.SERVER in servers:
            relayed = convert.relayed(servers[convert.SERVER], self._call)
            changes["mcp_servers"] = {**servers, convert.SERVER: relayed}
        self.client = ClaudeSDKClient(dataclasses.replace(options, **changes))
        try:
            await _apart(self._connected())
        except BaseException:
            await self.close()
            raise

    async def _connected(self) -> None:
        await self.client.connect()
        self._reader = asyncio.get_running_loop().create_task(self._read())

    # ------------------------------------------------------------------ what it is given
    def _hooks(self, hooks: Mapping[str, Any]) -> dict[str, Any]:
        """``hooks`` with each callback replaced by its relay."""
        return {
            event: [
                dataclasses.replace(
                    matcher,
                    hooks=[self._hook(event, m, c) for c in range(len(matcher.hooks))],
                    # a background run's call may wait for a person in the harness's hooks;
                    # a team's keep what they set (none: Claude Code's own default)
                    timeout=HOOK_SECONDS if ours(matcher) else matcher.timeout,
                )
                for m, matcher in enumerate(matchers)
            ]
            for event, matchers in hooks.items()
        }

    def _hook(self, event: str, matcher: int, callback: int) -> Any:
        """The relay of one hook: the same hook of the turn being served — of the attempt of
        its run, for a background subagent that has one (:class:`Background`); with none, a
        tool call is refused (:data:`UNATTENDED`) and any other event passes."""

        async def relay(given: Any, tool_use_id: str | None, hook: Any) -> Any:
            if event == "PreToolUse" and given.get("tool_name") == WORKFLOW_TOOL and self._flow():
                refused = {"permissionDecision": "deny", "permissionDecisionReason": ONE_WORKFLOW}
                return {"hookSpecificOutput": {"hookEventName": event, **refused}}
            found = self._working(given.get("agent_id"), given.get("agent_type"))
            if found is not None:
                answer = await found.asked(
                    lambda at: at.options.hooks[event][matcher].hooks[callback](
                        given, tool_use_id, hook
                    )
                )
                if answer is not _OVER:
                    self._noted(event, given, answer, found)
                    return answer
            # a subagent whose run has ended (cancelled, out of time) has none to decide it
            turn = self.turn if found is None else None
            if turn is not None:
                try:
                    own = turn.options.hooks[event][matcher].hooks[callback]
                    return await turn.within(own(given, tool_use_id, hook))
                except Left:
                    pass
            if event != "PreToolUse":
                return {}
            self.refused[str(given.get("tool_name") or "")] += 1
            ends = self._ends(given.get("agent_id"))
            denied = {
                "hookEventName": event,
                "permissionDecision": "deny",
                "permissionDecisionReason": LEFT if ends else UNATTENDED,
            }
            if ends:
                return {"hookSpecificOutput": denied, "continue_": False, "stopReason": LEFT}
            return {"hookSpecificOutput": denied}

        return relay

    def _noted(self, event: str, given: Any, answer: Any, found: Background) -> None:
        """Whose a harness tool's call is, once its hook let it through (the call itself
        comes with no word of who makes it: :meth:`_call`), with the arguments Claude Code
        makes it with — a hook's ``updatedInput`` over those it was given; forgotten when a
        hook refused it, and once it ran."""
        from trellis.harness.tools.convert.claude import SERVER

        name, ref = str(given.get("tool_name") or ""), str(given.get("tool_use_id") or "")
        if not name.startswith(f"mcp__{SERVER}__"):
            return
        said = (answer.get("hookSpecificOutput") if isinstance(answer, dict) else None) or {}
        decided = said.get("permissionDecision")
        if event == "PreToolUse" and decided == "allow":
            kept = self._calling.get(ref)
            args = said.get("updatedInput") or (
                kept[1] if kept is not None else dict(given.get("tool_input") or {})
            )
            self._calling[ref] = (name.removeprefix(f"mcp__{SERVER}__"), dict(args), found)
        elif event != "PreToolUse" or decided == "deny":
            self._calling.pop(ref, None)

    def _working(self, agent_id: Any, agent_type: Any = None) -> Background | None:
        """The background run whose work a call of the CLI's ``agent_id`` is: a subagent's,
        named by its task's id, or the workflow's, whose agents name their kind
        (:data:`WORKFLOW_AGENT`) in their first call's hook — their permission requests and
        harness tool calls after it name the agent alone."""
        who = str(agent_id or "")
        found = self.background.get(who) or self._workers.get(who)
        if found is None and agent_type == WORKFLOW_AGENT:
            found = self._flow()
            if found is not None:
                self._workers[who] = found
        return found

    def _flow(self) -> Background | None:
        """The dynamic workflow at work in the session, if one is (one at a time)."""
        return next((b for b in self.background.values() if b.kind is Kind.WORKFLOW), None)

    def _forget(self, found: Background) -> None:
        """``found``'s task ended and its run too: it is no longer the session's."""
        self.background.pop(found.task_id, None)
        for who in [w for w, b in self._workers.items() if b is found]:
            del self._workers[who]

    def _ends(self, agent_id: Any) -> bool:
        """Whether a call no turn can take ends the CLI's turn: the main agent's own
        (a subagent's names it), in a turn that was left and whose result is still to come."""
        return bool(self.owed) and not agent_id

    async def _can_use_tool(self, name: str, args: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultDeny

        found = self._working(getattr(context, "agent_id", None))
        if found is not None:
            answer = await found.asked(lambda at: at.options.can_use_tool(name, args, context))
            if answer is not _OVER:
                return answer
        turn = self.turn if found is None else None
        if turn is not None:
            with contextlib.suppress(Left):
                return await turn.within(turn.options.can_use_tool(name, args, context))
        self.refused[name] += 1
        if self._ends(getattr(context, "agent_id", None)):
            return PermissionResultDeny(message=LEFT, interrupt=True)
        return PermissionResultDeny(message=UNATTENDED)

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A harness tool's call: the tool of that name the turn being served has — or, the
        call a background subagent's (:meth:`_noted`), the attempt of its run."""
        from trellis.harness.tools.convert import claude as convert

        ref = next((r for r, (n, a, _) in self._calling.items() if (n, a) == (name, args)), None)
        found = None if ref is None else self._calling.pop(ref)[2]
        if found is not None:

            async def made(at: Turn) -> dict[str, Any]:
                return await _handler(at, name)(args)

            answer = await found.asked(made)
            if answer is not _OVER:
                return answer
        turn = self.turn if found is None else None
        if turn is not None:
            # the turn lists what the process lists (its shape): it has the tool
            own = convert.handler(turn.tools, name)

            async def called() -> dict[str, Any]:
                return await own(args)

            with contextlib.suppress(Left):
                return await turn.within(called())
        self.refused[name] += 1
        return {"content": [{"type": "text", "text": UNATTENDED}], "is_error": True}

    # ------------------------------------------------------------------ its stream
    async def _read(self) -> None:
        try:
            async for message in self.client.receive_messages():
                self._took(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the process ended with an error (the SDK's own)
            self.error = exc
        # the process ended of itself (:meth:`close` cancels this reader before it ends it)
        self._ended()

    def _took(self, message: Any) -> None:
        from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage

        found = getattr(message, "session_id", None)
        if isinstance(found, str) and found:
            self.session_id = found
        self._tracked(message)
        whose = self._of(getattr(message, "parent_tool_use_id", None))
        if whose is not None:  # a background subagent's own message: its run's
            whose.took(message)
            return
        result = isinstance(message, ResultMessage)
        origin = getattr(message, "origin", None)
        own = origin is None or origin.get("kind") == "human"
        # a turn of the session's own (a task's notification) began: a message that names it
        # (a CLI that sends its prompt), or the main agent at work with no prompt owed — a
        # turn's ``init``, its reply (the bundled CLI sends no prompt for such a turn)
        main = getattr(message, "parent_tool_use_id", None) is None
        began = isinstance(message, AssistantMessage) or (
            isinstance(message, SystemMessage) and message.subtype == "init"
        )
        if main and not result and (not own or (began and not self.owed)):
            self.injected = True
            self.due = False
            self._settled.clear()
        turn = self.turn
        mine = turn is not None and turn.listening and not turn.answered
        if result and own:  # a prompt is sent once the one before has its result
            self.owed = 0
        if result:  # a turn of the session's own ended with it (one of a caller's, never)
            self.injected = self.injected and own
        if turn is not None and mine and (own if result else not self.injected):
            turn.answered = result
            charged = self.charged(message, own=own, taken=True) if result else message
            turn.queue.put_nowait(charged)
        elif result and own and self.left is not None:  # the result of the turn left
            left, self.left = self.left, None
            left.answered = True
            left.queue.put_nowait(self.charged(message, own=own, taken=True))
        else:
            if result:
                self.charged(message, own=own, taken=False)
            unattended(self, message)
            if turn is None:
                self._rest()
        if result and not self.owed and not self.injected:
            self._settled.set()

    def _tracked(self, message: Any) -> None:
        """The background tasks under way, as the stream says: started, or ended (its
        notification, or an update to a terminal status: either may be all that comes)."""
        from claude_agent_sdk import (
            TERMINAL_TASK_STATUSES,
            TaskNotificationMessage,
            TaskStartedMessage,
            TaskUpdatedMessage,
        )

        if isinstance(message, TaskStartedMessage):
            self.working[message.task_id] = message.description
            self._launched(message)
        elif isinstance(message, TaskNotificationMessage) or (
            isinstance(message, TaskUpdatedMessage) and message.status in TERMINAL_TASK_STATUSES
        ):
            self.working.pop(message.task_id, None)
            # its end, with no turn served, is followed by a turn of the session's own
            self.due = self.due or self.turn is None
            found = self.background.get(message.task_id)
            if found is not None:
                found.finished(str(message.status), str(getattr(message, "summary", "") or ""))
                if found.over:  # its run ended first: its task was stopped
                    self._forget(found)

    def _launched(self, message: Any) -> None:
        """A subagent the turn being served started in the background (the CLI says which
        it runs so: ``is_backgrounded``): the child run its ``Agent`` call started is kept
        open from here (:class:`Background`), no longer that turn's to end."""
        turn = self.turn
        started = None if turn is None else turn.runtime.helpers
        if started is None or not message.tool_use_id or not backgrounded(message):
            return
        helper = started.detached(message.tool_use_id)
        if helper is not None:
            found = Background(self, message.task_id, message.tool_use_id, helper)
            self.background[message.task_id] = self.sessions.attended[found.run_id] = found

    def _of(self, call_id: Any) -> Background | None:
        """The background run of the subagent the ``Agent`` call ``call_id`` started."""
        return next((b for b in self.background.values() if b.call_id == call_id), None)

    def _over(self, found: Background) -> None:
        """``found``'s run ended: a task still at work is stopped, with the SDK's own stop
        (an interrupt would stop every task of the session), and what it asks until it ends
        is refused."""
        self.sessions.attended.pop(found.run_id, None)
        if found.ended is not None:
            self._forget(found)
        elif not self.closed:
            self._aside_run(self._stop(found.task_id))

    async def _stop(self, task_id: str) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(SETTLE_SECONDS):
                await self.client.stop_task(task_id)

    async def asks(self, runtime: Runtime) -> None:
        """The questions the session's background runs are paused on, asked in ``runtime``'s
        run — a turn of the conversation, before its prompt is sent — as a sub-agent's
        question is asked through its parent (``subagents.child_run``): the turn pauses with
        each, and its answer is handed to the run that asked."""
        from trellis.harness.subagents import ASKED, SUBAGENT

        for found in list(self.background.values()):
            asked = found.question
            if asked is None:
                continue
            record = await found.agent.harness.runs.get(asked.run_id, tenant=found.tenant)
            if record is None or record.status is not RunStatus.PAUSED:
                continue  # cancelled where it waited: nothing to ask any more
            named = {
                "agent_id": found.name,
                "run_id": asked.run_id,
                "interrupt_id": asked.interrupt_id,
                "background": True,
            }
            answered = await runtime.interrupt(
                content_key("subagent", asked.run_id, asked.interrupt_id),
                payload={**(asked.payload or {}), SUBAGENT: named},
                **{name: getattr(asked, name) for name in ASKED},
            )
            found.answered(
                answered.model_copy(
                    update={"interrupt_id": asked.interrupt_id, "run_id": asked.run_id}
                )
            )

    def _ended(self) -> None:
        """The process ended of itself: the turn being served is told, the session is gone."""
        self.sessions.forget(self, "its process ended")
        self._aside_run(self.close())

    # ------------------------------------------------------------------ a turn
    async def served(self, turn: Turn, prompt: str) -> AsyncGenerator[Any, None]:
        """``turn`` served: its prompt sent, then its own messages up to its own result. Left
        before that (the run paused, was cancelled, stopped reading), the CLI's turn is
        interrupted — the session and what works in its background stay."""
        self._attach(turn)
        try:
            await self._send(turn, prompt)
            while True:
                message = await turn.queue.get()
                if message is None:  # the process ended
                    raise self.error or RuntimeError(ENDED)
                yield message
                if turn.answered:
                    return
        finally:
            left = turn.listening and not turn.answered
            under_way = list(turn.calls)
            self._detach(turn)
            # an interrupt stops the background work too (the CLI's own rule): with any
            # under way, the turn ends at its next tool call instead (:meth:`_ends`)
            turn.stopped = left and not self.closed and not self.working
            if turn.stopped:
                self._aside_run(self._interrupt())
            # its result, when it comes, is its own (:meth:`outcome`); one that comes after
            # its run went on with a pause is the resumed attempt's, as a process's that
            # resumes the session
            if left and not self.closed and (turn.stopped or turn.runtime.pending is None):
                self.left = turn
            # the calls it left under way end with it (cancelled above), as a query's own
            # process ends its calls when it is closed: what one says as it ends (a helper's
            # run: ``helper_finished``) is on the run's stream before the run's end
            if under_way:
                await asyncio.wait(under_way)

    def _attach(self, turn: Turn) -> None:
        self.turn = turn
        self.left = None  # its result came (:meth:`settled`), or never will
        self.due = False  # a turn is served: what is due is handed to it
        self.kept_in = turn.runtime.agent.sessions
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.mirror is not None:
            store = turn.options.session_store
            assert isinstance(store, JournalSessions)  # as the process's is (its shape)
            self.mirror.serve(store)
        if self.refused:
            calls = ", ".join(f"{tool} x{n}" for tool, n in sorted(self.refused.items()))
            self.refused.clear()
            turn.runtime.events.warning(
                "claude_session",
                "background work made tool calls while no run of the conversation was under "
                f"way: they were refused, not run ({calls})",
            )

    def _detach(self, turn: Turn) -> None:
        for call in list(turn.calls):
            call.cancel()
        self.turn = None
        for found in self.background.values():
            # a cancelled run's children are cancelled with it
            cancelled: str | None = turn.runtime.cancelled
            if cancelled is not None and found.parent_run_id == turn.runtime.run_id:
                found.cancel(cancelled)
        if self.mirror is not None:
            self.mirror.into = None
        if not self.closed:
            self._rested = asyncio.get_running_loop().time()
            self._rest()

    async def outcome(self, turn: Turn) -> Any:
        """The result of the CLI's turn ``turn`` left before it, charged to it
        (:meth:`charged`), when it comes at once: it came already, or the turn was
        interrupted (:attr:`Turn.stopped`) — waited for :data:`SETTLE_SECONDS` at most.
        ``None`` otherwise: with background work under way the CLI's turn ends at its next
        tool call, after the run went on (a pause: its resumed attempt is charged it, as a
        process's that resumes the session is) or ended (cancelled: no run is charged what
        it spent past then — the run's record is final — and the next turn is not)."""
        from claude_agent_sdk import ResultMessage

        if not (turn.answered or turn.stopped):
            return None
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(SETTLE_SECONDS):
                while (message := await turn.queue.get()) is not None:
                    if isinstance(message, ResultMessage):
                        return message
        if self.left is turn:
            self.left = None
        return None

    async def settled(self) -> bool:
        """Whether the session can take a prompt: the result of the one before came, and no
        turn of the session's own is under way — either is waited for, :data:`SETTLE_SECONDS`
        (a prompt sent meanwhile may be handed to the turn under way, and have no result)."""
        try:
            async with asyncio.timeout(SETTLE_SECONDS):
                await self._settled.wait()
        except TimeoutError:
            return False
        return not self.closed

    async def _send(self, turn: Turn, prompt: str) -> None:
        self.owed = 1
        self.injected = False
        self._settled.clear()
        turn.listening = True
        await self.client.query(prompt)

    async def _interrupt(self) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(SETTLE_SECONDS):
                await self.client.interrupt()

    def charged(self, result: Any, *, own: bool = True, taken: bool = True) -> Any:
        """``result`` with the cost the turn it ends is charged (``taken``: the turn being
        served; ``own``: a turn a caller sent, not one the session injected). The cost a
        result carries is the process's running total: what it grew by since the last result
        charged (a total that did not grow is taken as the growth: never less than nothing)
        is shared by the tokens each used since — the turn its own usage, each background
        run what its subagent's replies used. The share of a turn no run is served for is
        the run's whose task's end the session ran it for; with none, it stays for the
        conversation's next turn, as the whole does with no background run to charge."""
        open_ = [b for b in self.background.values() if not b.over]
        runs = [b for b in open_ if b.unbilled]
        after = None if own else next((b for b in open_ if b.ended is not None), None)
        for found in open_:
            if found.ended is not None:  # what its task cost is known now
                found._settled.set()
        if result.total_cost_usd is None or not (taken or runs or after):
            return result
        total = float(result.total_cost_usd)
        more = total - self.billed
        more = more if more >= 0 else total
        mine = _tokens(result.usage)
        whole = mine + sum(b.unbilled for b in runs)
        share = more * mine / whole if whole else more
        for found in runs:
            found.spent(None, more * found.unbilled / whole)
            found.unbilled = 0
        self.billed = total
        if taken:
            return dataclasses.replace(result, total_cost_usd=share)
        if after is not None:
            after.spent(None, share)
        else:
            self.billed -= share
        return result

    # ------------------------------------------------------------------ its end
    def _rest(self) -> None:
        """No turn is served: the session closes when its time is up (:data:`IDLE_SECONDS`
        after this, nothing working in it — no background task, no turn of its own under way,
        and none due for :data:`SETTLE_SECONDS`; :data:`WORKING_SECONDS` after its last
        turn)."""
        loop = asyncio.get_running_loop()
        if self._timer is not None:
            self._timer.cancel()
        last = self._rested + WORKING_SECONDS - loop.time()
        idle = max(IDLE_SECONDS, SETTLE_SECONDS) if self.due else IDLE_SECONDS
        busy = self.working or self.injected
        wait = max(0.0, last) if busy else min(idle, max(0.0, last))
        self._timer = loop.call_later(wait, self._expired)

    def _expired(self) -> None:
        self._timer = None  # the one of a session that serves no turn and is not closed
        why = "it was kept as long as a session is" if self.working else "it was idle"
        self._aside_run(self.sessions.close(self.key, why))

    def _aside_run(self, work: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.get_running_loop().create_task(work, context=contextvars.Context())
        self._aside.add(task)
        task.add_done_callback(self._aside.discard)

    async def close(self) -> None:
        """The process ended (its stdin closed; stopped if it does not end) and its reader
        with it; the conversation's session kept, with all that was mirrored, for a turn that
        resumes it. The process ends whatever its background runs' ends meet (a run store
        that cannot be reached): the first such error is raised once it has."""
        if self.closed:
            return
        self.closed = True
        self._settled.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for turn in (self.turn, self.left):
            if turn is not None:
                turn.queue.put_nowait(None)
        runs = list(self.background.values())
        try:  # each ends saying why
            ended = await asyncio.gather(
                *(found.abandoned(LOST.format(why=self.why)) for found in runs),
                return_exceptions=True,
            )
        finally:
            kept = None if self.kept_in is None else self.kept_in.get(self.key)
            if self.mirror is not None and kept is not None and kept[0] == self.session_id:
                assert self.kept_in is not None
                self.kept_in[self.key] = (kept[0], dict(self.mirror.transcripts))
            current = asyncio.current_task()
            for task in (self._reader, *self._aside):
                if task is not None and task is not current:
                    task.cancel()
            if self.client is not None:
                with contextlib.suppress(Exception):
                    await self.client.disconnect()
        if runs:
            await asyncio.wait({found.task for found in runs})
        failed = next((e for e in ended if isinstance(e, BaseException)), None)
        if failed is not None:
            raise failed


class LiveSessions:
    """An agent's live sessions in this process, by conversation."""

    def __init__(self) -> None:
        self._open: dict[str, LiveSession] = {}
        #: the background subagents' runs under way in them, by run (:class:`Background`)
        self.attended: dict[str, Background] = {}
        #: what each conversation's next run is told of work that ended with its session
        self._lost: LRU[str, str] = LRU()

    def __len__(self) -> int:
        return len(self._open)

    def get(self, key: str) -> LiveSession | None:
        return self._open.get(key)

    async def open(
        self,
        key: str,
        shape: Any,
        options: Any,
        transcripts: Mapping[str, list[dict[str, Any]]],
    ) -> LiveSession | None:
        """A session started for ``key`` (``None``: :data:`KEPT` are open and each serves a
        turn or has work under way — the turn runs without one, as it always did)."""
        if len(self._open) >= KEPT:
            idle = [s for s in self._open.values() if s.turn is None and not s.working]
            if not idle:
                return None
            await self.close(idle[0].key, "another conversation needed its place")
        session = LiveSession(key, shape, self)
        await session.start(options, transcripts)
        self._open[key] = session
        return session

    def forget(self, session: LiveSession, why: str) -> None:
        """``session`` is no longer the conversation's; the work that ends with it is what
        the conversation's next run is told (:meth:`told`)."""
        self._open.pop(session.key, None)
        session.why = why
        if session.working:
            tasks = "; ".join(f"{task} ({what})" for task, what in session.working.items())
            self._lost[session.key] = (
                f"Claude's session was closed ({why}): the background work under way in it "
                f"was stopped: {tasks}"
            )
            log.warning("conversation %s: %s", session.key, self._lost[session.key])

    def told(self, key: str) -> str | None:
        """What ``key``'s next run is told (once) of work that ended with its session."""
        return self._lost.pop(key)

    async def close(self, key: str, why: str) -> None:
        """``key``'s session closed, if it has one (``why``: said with the work it stops)."""
        session = self._open.get(key)
        if session is not None:
            self.forget(session, why)
            await session.close()

    async def aclose(self) -> None:
        """Every session closed: the harness is closing. Each is closed whatever another's
        close meets; the first error is raised once all are."""
        sessions, self._open = list(self._open.values()), {}
        closed = await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)
        failed = next((e for e in closed if isinstance(e, BaseException)), None)
        if failed is not None:
            raise failed


def _handler(at: Turn, name: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """The harness tool ``name`` a background subagent's call runs in ``at``, its run's
    attempt: the one its run's own server lists — or, one the process lists that its run was
    not offered (the hints narrow each run by its own task), made of its run's tool set and
    offered to it now: the subagent was offered what the process lists."""
    from trellis.harness.tools.convert import claude as convert

    if convert.lists(at.tools, name):
        return convert.handler(at.tools, name)
    found = at.runtime.toolbox.get(name)
    if found is None:
        return _unknown(name)
    at.runtime.offer([name])
    return convert.tool_handler(found)


def _unknown(name: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """What a call of a tool its run does not have is answered."""

    async def refused(args: dict[str, Any]) -> dict[str, Any]:
        said = f"{name} is not a tool of this run: it was not run"
        return {"content": [{"type": "text", "text": said}], "is_error": True}

    return refused


async def _apart[T](work: Awaitable[T]) -> T:
    """``work`` done in a task whose context is no run's (an empty one): what it starts —
    the SDK's reader, the tasks it handles the CLI's requests in — outlives the turn that
    happened to ask, and holds nothing of it."""

    async def done() -> T:
        return await work

    task = asyncio.get_running_loop().create_task(done(), context=contextvars.Context())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise


__all__ = [
    "IDLE_SECONDS",
    "KEPT",
    "UNATTENDED",
    "Background",
    "LiveSession",
    "LiveSessions",
    "Turn",
]
