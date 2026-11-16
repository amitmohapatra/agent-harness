"""A conversation's agent process, kept between its turns while work in its background is under
way: what the Claude Agent SDK's live session (``bridges.claude_live``) and an ACP agent's
(``bridges.acp_live``) share.

A framework's background work — a subagent started in the background, a shell command run in
the background — belongs to the process that started it: it goes on after the turn and ends
with the process. So the process outlives the turn (:class:`Keeper`), and the conversation's
next turn is served by it.

* **Whose turn.** The process is started once, while each turn has its own way of answering
  the agent's requests (:class:`Turn`: its hooks and permission check, its harness tools): the
  process is given relays, each handing a request to the turn being served, in a copy of that
  turn's context — ``trellis.current()``, the trace and the tool's own run are that turn's.
* **A background subagent's run.** The child run its call started (``bridges.helpers``) is
  kept open past the turn, until its task ends (:class:`Kept`), held in the run store meanwhile
  (``holding``) — a server killed with the session leaves a run the store ends once the hold
  lapses, not one ``RUNNING`` for ever. Its attempts go through the pipeline as any run's, and
  the requests the agent makes for the subagent are relayed to the attempt under way, turn or
  no turn. A call that asks a person pauses that run as any run pauses: the question is asked
  in the conversation's next turn (:meth:`Keeper.asks`), and the attempt that follows the
  answer decides the call from its journal. An answer that reaches a server the session is not
  on ends the run (:func:`gone`). A turn on a worker whose session another worker's process
  still holds (a background run it handed on there: :func:`held_elsewhere`) does not resume it
  in a second process.
* **No run.** A call made while no run is under way — between two turns, by no background
  run — has nothing to decide or record it: it is refused (:data:`UNATTENDED`), never run
  unchecked, and the conversation's next run says how many were.
* **Its life.** One process per conversation, in this process only; at most :data:`KEPT` of
  a framework's over all the server process's agents (:meth:`LiveSessions.place`: the place
  is taken before the process starts; an idle one gives its place fairly, :func:`_yielding`).
  It closes :data:`IDLE_SECONDS` after its last activity when nothing
  works in it (no background task, and no turn of its own for a finished one under way or
  due), and :data:`WORKING_SECONDS` after its last turn whatever does; and with the harness.
  Work that ends with it is said in the conversation's next run, never dropped in silence.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import logging
import time
from collections import Counter
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Hashable,
    Iterator,
    Sequence,
)
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from trellis.contracts import Interrupt, InterruptResolution, RunStatus
from trellis.harness import stopping
from trellis.harness.asking import RunCancelled
from trellis.harness.bounded import LRU
from trellis.harness.bridges.helpers import ended

if TYPE_CHECKING:
    from trellis.harness.journal import Pending
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: How long a session nothing works in is kept after its last activity.
IDLE_SECONDS: Final = 300.0
#: How long a session is kept after its last turn, whatever still works in it.
WORKING_SECONDS: Final = 6 * 3600.0
#: The most sessions of a framework kept in this server process, over all its agents. Each is
#: an agent process of its own: a Claude Code CLI holds about 200 MB (16 measured at 3.3 GB on
#: the framework comparison of 2026-11-15), so 8 bound what a server's kept processes take to
#: about 1.6 GB. A turn that finds no place runs as a turn with no session does (a process
#: that resumes the conversation), so the bound costs a start, never a turn.
KEPT: Final = 8
#: The most of them one user (an owner: a tenant's user) holds at once: one user's many
#: working conversations — each kept up to :data:`WORKING_SECONDS` — leave the rest of the
#: server's places to the others.
KEPT_PER_OWNER: Final = 4
#: How long a turn waits for the session to be free (an earlier turn's end, a turn of the
#: session's own) before its own prompt is sent; past it the session is closed and the turn
#: runs without.
SETTLE_SECONDS: Final = 30.0
#: What a tool call made while no run of the conversation is under way is answered.
UNATTENDED: Final = (
    "This call did not run: no run of the conversation is under way, so nothing can check or "
    "record it. It can be made again once the user's next message is being answered."
)
#: How long a background subagent's run waits for a person's answer before it is given up
#: (cancelled, saying so).
ANSWER_SECONDS: Final = 12 * 3600.0
#: How often a background run that waits for a person looks whether it was cancelled there.
WAITS_SECONDS: Final = 10.0
#: Why a background subagent's run ended when no person answered it, and when it was cancelled
#: before its first attempt began.
UNANSWERED: Final = "nobody answered the background subagent's question in {hours:g} hours"
CANCELLED_FIRST: Final = "the background subagent's run was cancelled before it began"
#: What a background subagent's run answers when its task ended with no word of its own, and
#: why it ended when its task failed without saying why.
COMPLETED: Final = "The task completed."
FAILED: Final = "the background subagent's task failed"
#: What an attempt of a background run is nudged with: its run paused, or its task ended.
NUDGE: Final = object()
#: What a request relayed to a background run is answered when that run has ended.
OVER: Final = object()


class Left(Exception):
    """The turn a request was handed to was left before it answered."""


async def gone(
    runtime: Runtime,
    run_id: str,
    interrupt_id: str,
    answer: Any,
    *,
    why: str,
    said: str,
    code: str,
) -> None:
    """The person's ``answer`` to the question the background subagent's run ``run_id`` is
    paused on (``interrupt_id``) reached ``runtime``'s turn on a server its session is not on:
    it ended with the server that held it (a crash, a restart) or lives on another, and the
    subagent's task with it — nothing here can continue it. The run records the answer and
    ends ``ERROR`` (``holder_lost``: ``why``), and the turn is told (``code``: ``said``). A run
    no longer waiting on that question (it ended, or was answered already) is left as it is."""
    from trellis.contracts import AgentError, ErrorCategory, ErrorSource

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
            message=why,
            retryable=False,
            source=cast("ErrorSource", runtime.agent.adapter.name),
        )
        await runs.finish(run_id, RunStatus.ERROR, error=error, tenant=tenant)
    runtime.events.warning(code, said)


def background_answer(
    pending: Pending, resolution: InterruptResolution
) -> tuple[str, str, InterruptResolution] | None:
    """The answer a turn brings to a background subagent's question it was paused on
    (:meth:`Keeper.asks`): that run, its interrupt and the person's resolution — the turn's own
    prompt is still to send. ``None``: the turn paused on a question of its own."""
    from trellis.harness.subagents import asked_by

    named = asked_by(pending.interrupt) or {}
    if not named.get("background"):
        return None
    return (str(named["run_id"]), str(named["interrupt_id"]), resolution)


async def held_elsewhere(runtime: Runtime, sessions: LiveSessions) -> bool:
    """Whether the run's session is held by another process's live session: a background
    subagent's run it handed on there (``Journal.handed_on``) is not one this process keeps,
    and has not ended (its keeper ends it; a keeper that stopped leaves it to the run store —
    its deadline, and a run store that ends a run whose holder stopped heartbeating ends it
    then: the seam where that hold is read). Two processes do not hold one session: a turn
    that resumes it forks it, and the subagent works on in the session it has."""
    runs = runtime.agent.harness.runs
    for run_id in runtime.replay.journal.handed_on:
        if run_id in sessions.attended:
            continue
        record = await runs.get(run_id, tenant=runtime.tenant)
        if record is not None and not record.status.final:
            return True
    return False


def converses(runtime: Runtime) -> bool:
    """Whether the run is a turn of a conversation (a thread of its own, not a sub-agent's
    run): what a later turn follows."""
    return runtime.parent is None and runtime.conversation != runtime.run_id


def said(sessions: LiveSessions, runtime: Runtime, code: str) -> None:
    """The run is told, once, of the background work that ended with its conversation's
    live session."""
    lost = sessions.told(runtime.conversation)
    if lost is not None:
        runtime.events.warning(code, lost)


@dataclasses.dataclass(eq=False)
class Turn:
    """The harness turn a session serves now: its run's own way of answering the agent's
    requests (``options``: Claude's options — its hooks, its ``can_use_tool`` — or an ACP
    attempt, ``adapters.acp.Attempt``), its harness tools, and the context its requests run
    in."""

    runtime: Any
    options: Any
    #: the run's own harness tools (``None``: it has none)
    tools: Any
    context: contextvars.Context
    queue: asyncio.Queue[Any] = dataclasses.field(default_factory=asyncio.Queue)
    #: its prompt was sent: the messages that follow are its own
    listening: bool = False
    #: its own result came
    answered: bool = False
    #: it was left before its result and the agent's turn interrupted: its result comes at once
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


class Kept:
    """A subagent at work in the background of a session, and the child run that is its
    record: the helper its call started (``bridges.helpers``), kept open from there until the
    framework says the task ended (:meth:`finished`).

    The run's attempts go through the pipeline (:meth:`_run`): each is served the subagent's
    own messages, if the framework shows them (:meth:`served`), and ends with a result made of
    the task's end (:meth:`_result`, the framework's) — or as any run ends: paused, cancelled,
    out of time, its budget spent. The requests the agent makes for the subagent are done in
    the attempt under way (:meth:`asked`); one the run pauses on waits there for the attempt
    that follows the person's answer."""

    #: why the run cannot go on when its task was stopped while it waited for a person
    stopped_why: ClassVar[str]
    #: why an attempt failed when the session's process ended under it
    ended_why: ClassVar[str]

    def __init__(
        self,
        session: Keeper[Any],
        task_id: str,
        call_id: str,
        helper: Any,
        turn: Turn | None = None,
    ) -> None:
        turn = turn or session.turn
        assert turn is not None  # the turn that started it (being served, or it just was)
        self.session = session
        #: the framework's id of the task: whose the subagent's requests are
        self.task_id = task_id
        #: the call that started it
        self.call_id = call_id
        self.run_id: str = helper.run_id
        self.parent_run_id: str = helper.parent_run_id
        self.name: str = helper.name
        #: how its call started it (``Kind.WORKFLOW``: a dynamic workflow's agents work in it)
        self.kind = helper.kind
        #: when its call started it (``time.monotonic``)
        self.began: float = helper.began
        self.agent = turn.runtime.agent
        self.tenant = turn.runtime.tenant
        #: the framework options of the turn that started it: what the process has
        self.given = dict(turn.runtime.framework_options)
        #: the attempt under way
        self.turn: Turn | None = None
        #: the run has ended
        self.over = False
        #: how the task ended, as the framework said: its status and summary
        self.ended: tuple[str, str] | None = None
        #: why the run cannot go on (:meth:`abandoned`)
        self.lost: str | None = None
        #: why it was cancelled with the run that started it (:meth:`cancel`)
        self.stopped: str | None = None
        #: the question the run is paused on, until the conversation's next turn asked it
        self.question: Interrupt | None = None
        #: what the task spent while no attempt was under way (:meth:`spent`)
        self._owed: list[tuple[Any, ...]] = []
        self._backlog: list[Any] = []
        self._waiting: list[asyncio.Future[None]] = []
        #: what the task spent is known (an attempt ends with its result once it is)
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
            assert record is not None  # its call started it
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
        ended(
            parent.events,
            run_id=self.run_id,
            name=self.name,
            framework=self.agent.adapter.name,
            status=record.status,
            began=self.began,
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
                await stopping.unless_ended(  # (or it ended meanwhile)
                    self.agent.harness.runs.cancel(self.run_id, reason=why, tenant=self.tenant)
                )
            finally:  # it waits no more, whether or not the run store took the cancel
                if self._answer is not None and not self._answer.done():
                    self._answer.set_result(None)

    def cancel(self, reason: str) -> None:
        """The run cancelled with the run that started it: its attempt under way stops as any
        run's does (``Agent.stop``: the first reason given stands), and one that has not begun
        to serve it does not (:meth:`served` reads why)."""
        self.stopped = reason
        self.agent.stop(self.run_id, reason)

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
                    raise RuntimeError(self.lost or self.ended_why)
                if turn.runtime.pending is not None:
                    if message is not NUDGE:  # kept for the attempt that follows
                        self._backlog.append(message)
                    return
                if message is not NUDGE:
                    yield message
                elif self.ended is not None:
                    break
            # what the task spent is known once the framework says (``_settled``)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(SETTLE_SECONDS):
                    await self._settled.wait()
            yield self._result(turn.runtime)
        finally:
            under_way = list(turn.calls)
            self._detach(turn)
            if under_way:  # they end with the attempt, before its run does (as a turn's)
                await asyncio.wait(under_way)

    #: the result an attempt ends with, made of the task's end (the framework's)
    def _result(self, runtime: Runtime) -> Any: ...

    def _attach(self, turn: Turn) -> None:
        """``turn`` is the attempt under way: it has what came while there was none, and the
        requests that waited for it go on."""
        self.turn = turn
        for message in self._backlog:
            turn.queue.put_nowait(message)
        self._backlog.clear()
        owed, self._owed = self._owed, []
        for spent in owed:
            self.spent(*spent)
        if self.ended is not None:
            turn.queue.put_nowait(NUDGE)
        self._woken()

    def _detach(self, turn: Turn) -> None:
        """The attempt ended: its requests under way end with it, and what it did not take
        is kept for the next."""
        for call in list(turn.calls):
            call.cancel()
        self.turn = None
        while not turn.queue.empty():
            message = turn.queue.get_nowait()
            if message is not None and message is not NUDGE:
                self._backlog.append(message)

    def finished(self, status: str, summary: str) -> None:
        """The framework said the task ended (with ``summary``: what it said last, or why it
        failed; either may be all that comes, and the last word with one stays)."""
        self.ended = (status, summary or (self.ended or ("", ""))[1])
        if self.turn is not None:
            self.turn.queue.put_nowait(NUDGE)
        elif self._answer is not None:  # it waits for an answer about a call that is no more
            self.session._aside_run(self.abandoned(self.stopped_why))

    def spent(self, *spent: Any) -> None:
        """What the task spent (the framework's measure: :meth:`_charge`), on the run — the
        attempt under way, else the next."""
        if self.turn is None:
            self._owed.append(spent)
            return
        self._charge(self.turn, *spent)

    #: what the task spent, on the attempt ``turn`` (the framework's)
    def _charge(self, turn: Turn, *spent: Any) -> None: ...

    # ------------------------------------------------------------------ its requests
    async def asked(self, make: Callable[[Turn], Coroutine[Any, Any, Any]]) -> Any:
        """A request the agent makes for the subagent — a hook, a permission check, a harness
        tool's call — done by ``make`` in the attempt under way, in its context. When the
        run pauses on it, the request waits and is done again in the attempt that follows
        the person's answer, whose journal answers it. :data:`OVER`: the run has ended."""
        while True:
            turn = await self._attempt()
            if turn is None:
                return OVER
            try:
                answer = await turn.within(make(turn))
            except Left:
                continue
            if turn.runtime.pending is None or not self.waits(answer):
                return answer
            turn.queue.put_nowait(NUDGE)  # the attempt ends paused

    #: whether a relayed request was answered that its run pauses on it (the framework's)
    def waits(self, answer: Any) -> bool: ...

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


class Keeper[K: Kept]:
    """One conversation's agent process, kept between its turns: what its turns, its
    background runs (``K``: the framework's) and its life share (the framework's session
    builds on it)."""

    #: why a background run ended with the session (``{why}``: why it was closed)
    lost: ClassVar[str]
    #: what the conversation's next run is told of work that ended with the session
    #: (``{why}``, ``{tasks}``)
    lost_work: ClassVar[str]
    #: the code of what the session says on a run's stream
    code: ClassVar[str]

    def __init__(self, key: str, sessions: LiveSessions, *, owner: Hashable) -> None:
        self.key = key
        #: whose conversation it is (its tenant and user), and when it was last asked for:
        #: what decides which idle session gives its place (:func:`_yielding`)
        self.owner = owner
        self.used = time.monotonic_ns()
        self.sessions = sessions
        self.turn: Turn | None = None
        #: the background tasks under way, by id: what each is
        self.working: dict[str, str] = {}
        #: the background subagents that have a run, by their task's id
        self.background: dict[str, K] = {}
        #: why the session was closed (``LiveSessions.forget``)
        self.why = "the harness was closed"
        #: the tool calls refused since the last turn for want of a run, by tool
        self.refused: Counter[str] = Counter()
        #: a turn of the session's own (for a task's end) is under way
        self.injected = False
        #: a background task ended while no turn was served: the session's own turn for it is
        #: due (the agent runs it as soon as it gets to it), until that turn begins — waited
        #: for :data:`SETTLE_SECONDS` at least
        self.due = False
        self.closed = False
        #: the session can take a prompt: no turn of its own is under way (:meth:`settled`)
        self._settled = asyncio.Event()
        self._settled.set()
        self._timer: asyncio.TimerHandle | None = None
        self._rested = 0.0
        self._aside: set[asyncio.Task[Any]] = set()
        #: where the conversation's session is kept for a turn with no live one
        #: (``Agent.sessions``), once a turn was served
        self.kept_in: LRU[str, Any] | None = None

    # ------------------------------------------------------------------ its turns
    def _attach(self, turn: Turn) -> None:
        """``turn`` is served now: the session's timer stops, and the run is told of the calls
        refused since the last turn."""
        self.turn = turn
        self.kept_in = turn.runtime.agent.sessions
        self.due = False  # a turn is served: what is due is handed to it
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.refused:
            calls = ", ".join(f"{tool} x{n}" for tool, n in sorted(self.refused.items()))
            self.refused.clear()
            turn.runtime.events.warning(
                self.code,
                "background work made tool calls while no run of the conversation was under "
                f"way: they were refused, not run ({calls})",
            )

    def _detach(self, turn: Turn) -> None:
        """``turn`` is served no more: a cancelled run's background children are cancelled with
        it, and the session rests."""
        for call in list(turn.calls):
            call.cancel()
        self.turn = None
        for found in self.background.values():
            cancelled: str | None = turn.runtime.cancelled
            if cancelled is not None and found.parent_run_id == turn.runtime.run_id:
                found.cancel(cancelled)
        if not self.closed:
            self._rested = asyncio.get_running_loop().time()
            self._rest()

    async def settled(self) -> bool:
        """Whether the session can take a prompt: no turn of the session's own is under way
        (and, the framework's, the result of the one before came) — waited for,
        :data:`SETTLE_SECONDS` (a prompt sent meanwhile is not a turn of its own)."""
        try:
            async with asyncio.timeout(SETTLE_SECONDS):
                await self._settled.wait()
        except TimeoutError:
            return False
        return not self.closed

    async def asks(self, runtime: Runtime) -> None:
        """The questions the session's background runs are paused on, asked in ``runtime``'s
        run — a turn of the conversation, before its prompt is sent — as a sub-agent's
        question is asked through its parent (``subagents.child_run``): the turn pauses with
        each, and its answer is handed to the run that asked."""
        from trellis.harness.subagents import asked_for_background

        for found in list(self.background.values()):
            asked = found.question
            if asked is None:
                continue
            record = await found.agent.harness.runs.get(asked.run_id, tenant=found.tenant)
            if record is None or record.status is not RunStatus.PAUSED:
                continue  # cancelled where it waited: nothing to ask any more
            answered = await asked_for_background(
                runtime, asked, agent_id=found.name, run_id=asked.run_id
            )
            found.answered(answered)

    # ------------------------------------------------------------------ its background
    def _of(self, call_id: Any) -> K | None:
        """The background run of the subagent the call ``call_id`` started."""
        return next((b for b in self.background.values() if b.call_id == call_id), None)

    def _over(self, found: Kept) -> None:
        """``found``'s run ended: a task still at work is stopped (:meth:`_stop`), and what it
        asks until it ends is refused."""
        self.sessions.attended.pop(found.run_id, None)
        if found.ended is not None or found.task_id not in self.working:
            self._forget(found)  # nothing works for it in the framework any more
        elif not self.closed:
            self._aside_run(self._stop(found.task_id))

    def _forget(self, found: Kept) -> None:
        """``found``'s task ended and its run too: it is no longer the session's."""
        self.background.pop(found.task_id, None)

    #: the background task ``task_id`` stopped, as the framework can (its own)
    async def _stop(self, task_id: str) -> None: ...

    def _task_ended(self, task_id: str, status: str, summary: str) -> None:
        """The framework said the background task ``task_id`` ended: no longer under way; its
        end, with no turn served, is followed by a turn of the session's own; its run told."""
        self.working.pop(task_id, None)
        self.due = self.due or self.turn is None
        found = self.background.get(task_id)
        if found is not None:
            found.finished(status, summary)
            if found.over:  # its run ended first: its task was stopped
                self._forget(found)

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
        self._aside_run(self.sessions.lapse(self, why))

    def _aside_run(self, work: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.get_running_loop().create_task(work, context=contextvars.Context())
        self._aside.add(task)
        task.add_done_callback(self._aside.discard)

    #: the session is closing: what waits on it is told (the framework's)
    def _closing(self) -> None: ...

    #: the process ended, and what reads it (the framework's)
    async def _disconnect(self) -> None: ...

    async def released(self) -> None:
        """The conversation let go of the session for good (its time is up, its place given,
        the harness closing): the process ended (:meth:`close`). A framework that keeps files
        of the session for the conversation's later turns removes them too."""
        await self.close()

    async def close(self) -> None:
        """The process ended (:meth:`_disconnect`) and its background runs with it, each
        saying why (:attr:`lost`). The process ends whatever its background runs' ends meet (a
        run store that cannot be reached): the first such error is raised once it has."""
        if self.closed:
            return
        self.closed = True
        self._settled.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._closing()
        runs = list(self.background.values())
        try:  # each ends saying why
            ended = await asyncio.gather(
                *(found.abandoned(self.lost.format(why=self.why)) for found in runs),
                return_exceptions=True,
            )
        finally:
            current = asyncio.current_task()
            for task in self._aside:
                if task is not current:
                    task.cancel()
            await self._disconnect()
        if runs:
            await asyncio.wait({found.task for found in runs})
        failed = next((e for e in ended if isinstance(e, BaseException)), None)
        if failed is not None:
            raise failed


class LiveSessions:
    """An agent's live sessions in this process, by conversation."""

    def __init__(self) -> None:
        self._open: dict[str, Keeper[Any]] = {}
        #: the places taken by sessions whose process is starting (:meth:`place`), by owner
        self.starting: Counter[Hashable] = Counter()
        #: the background subagents' runs under way in them, by run (:class:`Kept`)
        self.attended: dict[str, Kept] = {}
        #: what each conversation's next run is told of work that ended with its session
        self._lost: LRU[str, str] = LRU()

    def __len__(self) -> int:
        return len(self._open)

    def __iter__(self) -> Iterator[Keeper[Any]]:
        return iter(list(self._open.values()))

    def get(self, key: str) -> Keeper[Any] | None:
        """``key``'s session, asked for now (the one asked for longest ago gives its place
        first: :func:`_yielding`)."""
        found = self._open.get(key)
        if found is not None:
            found.used = time.monotonic_ns()
        return found

    @contextlib.asynccontextmanager
    async def place(self, among: Sequence[LiveSessions], *, owner: Hashable) -> AsyncIterator[bool]:
        """Whether a session of ``owner``'s can be kept here, its place taken while the code
        inside starts and keeps it (:meth:`keep`). :data:`KEPT` processes are kept by the
        server process, over every agent's sessions (``among``, these included), at most
        :data:`KEPT_PER_OWNER` of them one owner's: the place is taken before the process
        starts, so turns that start together start no more; with every place taken — or every
        one the owner may hold — an idle session gives its own (:func:`_yielding`: of the
        owner's own, at its share), and with none idle there is no place (``False``: the turn
        runs without one, as it always did)."""
        held = sum(s.owner == owner for sessions in among for s in sessions)
        held += sum(sessions.starting[owner] for sessions in among)
        taken = sum(len(sessions) + sessions.starting.total() for sessions in among)
        yielding = None
        if held >= KEPT_PER_OWNER or taken >= KEPT:
            yielding = _yielding(among, owner=owner if held >= KEPT_PER_OWNER else None)
            if yielding is None:
                yield False
                return
        self.starting[owner] += 1
        try:
            if yielding is not None:
                why = "another conversation needed its place"
                await yielding.sessions.lapse(yielding, why)
            yield True
        finally:
            self.starting[owner] -= 1
            if not self.starting[owner]:
                del self.starting[owner]

    def keep(self, session: Keeper[Any]) -> None:
        """``session`` is its conversation's (:meth:`place` took a place for it)."""
        self._open[session.key] = session

    def forget(self, session: Keeper[Any], why: str) -> None:
        """``session`` is no longer the conversation's; the work that ends with it is what
        the conversation's next run is told (:meth:`told`)."""
        if self._open.get(session.key) is session:
            del self._open[session.key]
        session.why = why
        if session.working:
            tasks = "; ".join(f"{task} ({what})" for task, what in session.working.items())
            self._lost[session.key] = session.lost_work.format(why=why, tasks=tasks)
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

    async def lapse(self, session: Keeper[Any], why: str) -> None:
        """``session`` let go of for good (its time is up, its place given), while it is its
        conversation's: closed as :meth:`close` closes it, and what the framework keeps of it
        for the conversation's later turns removed (``Keeper.released``). One its
        conversation no longer has is another's to close (whoever forgot it)."""
        if self._open.get(session.key) is session:
            self.forget(session, why)
            await session.released()

    async def aclose(self) -> None:
        """Every session let go of for good (``Keeper.released``): the harness is closing.
        Each is closed whatever another's close meets; the first error is raised once all
        are."""
        sessions, self._open = list(self._open.values()), {}
        closed = await asyncio.gather(*(s.released() for s in sessions), return_exceptions=True)
        failed = next((e for e in closed if isinstance(e, BaseException)), None)
        if failed is not None:
            raise failed


def among(runtime: Runtime, adapter: Any) -> list[LiveSessions]:
    """The live sessions :data:`KEPT` counts over: those of every agent of ``runtime``'s
    harness whose adapter is of ``adapter``'s kind, ``adapter``'s own included (its
    ``live``)."""
    adapters: list[Any] = [a.adapter for a in runtime.agent.harness.agents.values()]
    found = {id(a): a.live for a in adapters if type(a) is type(adapter)}
    found[id(adapter)] = adapter.live
    return list(found.values())


def _yielding(
    among: Sequence[LiveSessions], *, owner: Hashable | None = None
) -> Keeper[Any] | None:
    """The idle session (no turn, no work under way) that gives its place to another
    conversation: of the user who holds the most sessions — one user's many conversations
    cost that user a place, not another user's only one — the one asked for longest ago;
    ``owner``'s own only, for an owner at its share (:data:`KEPT_PER_OWNER`). ``None``: none
    is idle."""
    kept = [session for sessions in among for session in sessions]
    if owner is not None:
        kept = [session for session in kept if session.owner == owner]
    held = Counter(session.owner for session in kept)
    idle = [session for session in kept if session.turn is None and not session.working]
    return min(idle, key=lambda session: (-held[session.owner], session.used), default=None)


async def apart[T](work: Awaitable[T]) -> T:
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
    "KEPT_PER_OWNER",
    "UNATTENDED",
    "Keeper",
    "Kept",
    "LiveSessions",
    "Turn",
    "among",
    "apart",
    "background_answer",
    "converses",
    "gone",
    "held_elsewhere",
]
