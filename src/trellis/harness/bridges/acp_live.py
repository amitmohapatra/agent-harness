"""A conversation's ACP agent process (Kimi Code's ``kimi acp``), kept between its turns while
work in its background is under way — Kimi's ``Agent(run_in_background=true)``, a ``Bash``
command it runs in the background — so that work finishes. What it shares with Claude's live
session (``bridges.claude_live``) is ``bridges.live``'s: its relays, its background runs' records
(holds, ``Helpers.detached``), the idle and settle rules; here is what ACP needs of its own.

* **One process for the attempt, kept when work goes on.** Every attempt runs in an
  :class:`ACPSession` (``adapters.acp``): the agent's process, its connection and the harness's
  MCP server, each request of the agent's relayed to the turn being served (its
  ``adapters.acp.Attempt``). A conversation's turn that ends with a task still at work — or
  Kimi's own turn for one that ended, under way or due — leaves the process to its
  conversation (``live.LiveSessions``) instead of ending it, and the next turn is served by it.
  Any other ends it, as an attempt always did: a run of no conversation, a sandboxed run, one
  given skills (a home of the run's own, removed with it), a turn whose session another
  worker's process holds (it forks it: ``live.held_elsewhere``).
* **What ACP does not show.** Kimi's ACP server forwards only the events of the turn the
  current prompt started (``driverFor`` in its ACP server): a background sub-agent's work, the
  turn Kimi runs by itself when a task ends (its notification, which its own context keeps for
  the next prompt), and anything after a prompt's answer never reach the client. What the
  harness reads of them is Kimi's log of the session (``adapters.acp.UsageLog``: the
  ``agents/<id>/wire.jsonl`` its usage is read from), as it is written (:meth:`ACPSession.read`):
  a task's start and end (``task.started``, ``task.terminated``; a sub-agent's last word,
  ``subagent.completed``), the main agent's turns (``turn.prompt`` — one whose ``origin`` is
  not the ``user`` is Kimi's own — and ``turn.ended``), whose a permission request is
  (``interaction.request``: ACP's names no agent) and what each sub-agent spends.
* **One prompt at a time.** A prompt sent while Kimi runs a turn of its own is queued behind
  it: Kimi answers ``end_turn`` at once and runs it later, unseen. So a prompt waits until the
  log shows Kimi idle (``Keeper.settled``); a prompt Kimi queued anyway fails the turn saying
  so (:data:`QUEUED`), never an empty answer taken for its own.
* **A background sub-agent's run** (:class:`Background`): its permission requests — approvals
  are the session's, not a turn's: Kimi asks whenever they come — are decided in its run's
  attempt (``adapters.acp.Deciding``), turn or no turn; a call that asks a person pauses that
  run, asked in the conversation's next turn. What it spends is its run's; its end is the
  log's. ACP has no way to stop one task: a run that ends before its task leaves the task at
  work, refused whatever it asks, until it ends or the session closes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Hashable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Final

from trellis.harness import budget
from trellis.harness.asking import RunCancelled
from trellis.harness.bounded import LRU
from trellis.harness.bridges import live
from trellis.harness.bridges.helpers import ATTRIBUTED_SECONDS
from trellis.harness.bridges.live import (
    COMPLETED,
    FAILED,
    OVER,
    UNATTENDED,
    Keeper,
    Kept,
    Left,
    LiveSessions,
    Turn,
    apart,
)

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: Why a background sub-agent's run ended with its session, and when its task was stopped
#: while it waited for a person.
LOST: Final = (
    "Kimi Code's session was closed ({why}): the background sub-agent's task was stopped with it"
)
STOPPED: Final = "the background sub-agent's task was stopped in Kimi Code"
#: Why an attempt of a background sub-agent's run failed when the session's process ended.
ENDED: Final = "Kimi Code's process ended before the background sub-agent's task did"
#: Why a background sub-agent's run whose question was answered on a server its session is not
#: on ended, and what the turn that took the answer is told (``live.gone``).
GONE: Final = (
    "the background sub-agent's Kimi Code session is not on the server the answer reached (it "
    "ended with the server that held it, or lives on another): the answer cannot continue its "
    "task, which is gone from here"
)
GONE_SAID: Final = (
    "A background sub-agent's question was answered, but its Kimi Code session is gone (the "
    "server that held it stopped): its task cannot go on, and its run ended."
)
#: Why a turn whose prompt Kimi queued behind a turn of its own failed.
QUEUED: Final = (
    "Kimi Code was running a turn of its own (for a background task's end) when this turn's "
    "prompt reached it: it queued the prompt behind that turn and runs it out of the "
    "harness's sight, so this turn cannot be followed"
)
#: The harness's MCP server, as the agent names its tools (``tools.convert.mcp.SERVER``: that
#: module needs the MCP SDK, which only a run of an ACP agent imports).
SERVER: Final = "trellis"
HARNESS_PREFIX: Final = f"mcp__{SERVER}__"
#: How often the agent's usage log is read while a turn runs (and the budget checked).
USAGE_SECONDS: Final = 0.5
#: The first wait between two reads of Kimi's log for a line it is about to write.
LOG_STEP: Final = 0.05
#: The most a cancelled turn waits for the agent to say it stopped.
CANCEL_SECONDS: Final = 5.0
#: The last of the agent's stderr kept, for the error of a process that ended mid-turn.
STDERR_TAIL: Final = 4096
#: Why a run failed before its first prompt: the agent's log does not say its session is in
#: the mode the harness set, asking before a call.
UNGOVERNED: Final = (
    "the agent's session mode was set to {wanted!r}, but its log says the session is in "
    "{found!r} with the permission mode {alone!r} (None: it asks before a call). The harness "
    "governs only the calls the agent asks about, so no prompt was sent: check the agent's own "
    "configuration (Kimi's default_permission_mode)"
)
#: What the session says on a run's stream is under this code.
CODE: Final = "acp_session"
#: Kimi's statuses of a task that ended (``task.terminated``): it completed; it was stopped
#: (``killed``); any other, it failed.
TASK_COMPLETED: Final = "completed"
TASK_KILLED: Final = "killed"
#: A request relayed to a background sub-agent's run that its run paused on
#: (``adapters.acp.Deciding``): it waits for the attempt that follows the person's answer.
WAITS: Final = object()


class ACPRunError(RuntimeError):
    """The agent ended the turn as failed, or its process ended in the middle of it."""


def accounted(runtime: Runtime, record: Mapping[str, Any], model: str | None) -> None:
    """One model request's usage, as the agent's log records it (``usage.record``), on
    ``runtime``'s budget, under ``model`` (else the log's)."""
    usage = record["usage"]
    cached = int(usage.get("inputCacheRead") or 0)
    written = int(usage.get("inputCacheCreation") or 0)
    budget.account(
        runtime,
        model=model or str(record.get("model") or "kimi"),
        input_tokens=int(usage.get("inputOther") or 0) + cached + written,
        output_tokens=int(usage.get("output") or 0),
        cached=cached,
    )


def chosen(options: list[Any], kind: str) -> Any:
    """The answer choosing the first option of ``kind`` (``allow_once``, ``reject_once``);
    the agent's question cancelled when it offers none."""
    for option in options:
        if option.kind == kind:
            return selected(option.option_id)
    return dismissed()


def selected(option_id: str) -> Any:
    """The answer to a permission request that chooses its option ``option_id``."""
    from acp.schema import AllowedOutcome, RequestPermissionResponse

    outcome = AllowedOutcome(outcome="selected", option_id=option_id)
    return RequestPermissionResponse(outcome=outcome)


def dismissed() -> Any:
    """The answer to a permission request that chooses none of its options (ACP's
    ``cancelled``): the turn is being stopped."""
    from acp.schema import DeniedOutcome, RequestPermissionResponse

    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _shown(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """What Kimi shows its own user of the call a permission request is for (the ``display``
    of its log's ``interaction.request``: a command whole with the folder it runs in, a file's
    path and content, a URL...), less what kind of display it is; ``None`` when it shows
    nothing."""
    display = (request.get("request") or {}).get("display") or {}
    return {key: value for key, value in display.items() if key != "kind"} or None


def unfilled() -> Any:
    """The answer to a form the agent asked its user to fill that nobody fills now (ACP's
    ``cancel``): the turn is being stopped, or nothing is there to ask."""
    from acp.schema import CancelElicitationResponse

    return CancelElicitationResponse(action="cancel")


async def _logged_in_time(said: Callable[[], Awaitable[bool]], seconds: float) -> bool:
    """Whether Kimi's log ``said`` so within ``seconds``, asked again as it is written: Kimi
    logs an event after it answers for it — a moment after, long after on a busy machine. A
    file of this machine's, read again after a wait that doubles (from :data:`LOG_STEP` to a
    second): some forty reads over the longest wait."""
    step = LOG_STEP
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(seconds):
            while True:
                if await said():
                    return True
                await asyncio.sleep(step)
                step = min(2 * step, 1.0)
    return False


class Background(Kept):
    """A Kimi sub-agent at work in the background of a session (``Agent`` with
    ``run_in_background``), and the child run that is its record (``live.Kept``), kept open
    from its call until Kimi's log says its task ended. ACP shows nothing of its work: each
    attempt waits for the task's end — ``SUCCESS`` with what the sub-agent said last, ``ERROR``
    for a task that failed, ``CANCELLED`` for one stopped in Kimi Code."""

    stopped_why: ClassVar[str] = STOPPED
    ended_why: ClassVar[str] = ENDED

    def __init__(
        self, session: ACPSession, task_id: str, call_id: str, helper: Any, agent_id: str
    ) -> None:
        super().__init__(session, task_id, call_id, helper)
        self.session: ACPSession = session
        #: its sub-agent's id: whose its requests and its usage are, in Kimi's log
        self.agent_id = agent_id

    def _result(self, runtime: Runtime) -> Any:
        """What an attempt ends with, made of the task's end (the answer: its text)."""
        from acp import text_block
        from acp.schema import AgentMessageChunk

        assert self.ended is not None
        status, summary = self.ended
        if status == TASK_COMPLETED or runtime.budget_spent is not None:
            said = text_block(summary or COMPLETED)
            return AgentMessageChunk(session_update="agent_message_chunk", content=said)
        if status == TASK_KILLED:
            raise RunCancelled(STOPPED)
        raise ACPRunError(summary or FAILED)

    def _charge(self, turn: Turn, *spent: Any) -> None:
        """A model request's usage (a ``usage.record`` of its sub-agent's log) on the attempt
        ``turn``, and on its budget: spent, the run ends there (ACP cannot stop its task)."""
        from trellis.harness import budget

        (record,) = spent
        accounted(turn.runtime, record, self.session.model)
        reached = budget.reached(turn.runtime)
        if reached is not None and turn.runtime.budget_spent is None:
            budget.stopped(turn.runtime, reached)
            self.finished(TASK_KILLED, "")

    def finished(self, status: str, summary: str) -> None:
        """Kimi's log says the task ended: what it spent was read with it (its own log's lines
        come before its end in the main agent's)."""
        self._settled.set()
        super().finished(status, summary)

    def waits(self, answer: Any) -> bool:
        return answer is WAITS


class ACPSession(Keeper[Background]):
    """An ACP agent's process for an attempt — kept for its conversation's next turns while
    work in its background goes on (``keeps``) — and what reads its log."""

    lost: ClassVar[str] = LOST
    lost_work: ClassVar[str] = (
        "Kimi Code's session was closed ({why}): the background work under way in it was "
        "stopped: {tasks}"
    )
    code: ClassVar[str] = CODE

    def __init__(
        self,
        key: str,
        shape: Any,
        sessions: LiveSessions,
        *,
        owner: Hashable,
        keeps: bool,
        model: str,
    ) -> None:
        super().__init__(key, sessions, owner=owner)
        #: what the process was started with and has for good (:func:`shape`)
        self.shape = shape
        #: a conversation's: kept past the turn while work in its background goes on
        self.keeps = keeps
        #: the model its usage is counted under
        self.model = model
        self.conn: Any = None
        self.process: Any = None
        #: the harness's MCP server, as ACP hands it to the agent (none: the run has no tool)
        self.servers: list[Any] = []
        self.session_id: str | None = None
        #: the session holds what came before (one resumed, loaded or forked: :meth:`opened`)
        self.continued = False
        #: what the harness knows of the agent (``adapters.acp.Profile``)
        self.profile: Any = None
        #: the agent's shell, as the turn that started the process had it (its background
        #: sub-agents' commands are read by it: ``governance.shell``)
        self.shell: Any = None
        #: the main agent's usage, read for each turn (a background sub-agent's is its run's)
        self.usage: Any = None
        #: the agent's log, read for what ACP does not show (:meth:`read`)
        self.log: Any = None
        self.stderr = bytearray()
        self.drain: asyncio.Task[None] | None = None
        #: how many turns of its own Kimi began (for a task's end), as the log says
        self.own_turns = 0
        #: each permission request as Kimi logs it (``interaction.request``), by call id:
        #: whose it is, and what Kimi shows its own user of the call (:func:`_shown`) — the
        #: most recent (``bounded.MAXSIZE``): a request is read here as it is asked
        self._asked: LRU[str, tuple[str, dict[str, Any] | None]] = LRU()
        #: what each sub-agent said last (Kimi's ``subagent.completed``), by its id
        self._said: dict[str, str] = {}
        #: how each task that ended ended, by id (its call's launch may be read after)
        self._ended: dict[str, tuple[str, str]] = {}
        #: the sub-agents the log says were launched in the background, by the raw id of the
        #: ``Agent`` call each is the task of: its task's id and its own, until the turn
        #: takes them (:meth:`launches`)
        self._launches: dict[str, tuple[str, str]] = {}
        #: the harness tool calls of background sub-agents, by tool: whose each is
        self._calling: dict[str, list[Background]] = {}
        self._stack = contextlib.AsyncExitStack()
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------ its start
    async def start(self, spawned: Any, served: Any) -> None:
        """The process (``spawned``: a context of the connection and the process, the agent's
        requests relayed to this session), and the harness's MCP server (``served``: its
        context, each call relayed; ``None``: no tool), started in a context of no run
        (``live.apart``): nothing of the turn that happened to start them stays with them."""

        async def started() -> None:
            if served is not None:
                self.servers = [await self._stack.enter_async_context(served(self))]
            self.conn, self.process = await self._stack.enter_async_context(spawned(self))
            self.drain = asyncio.create_task(self._drained(self.process.stderr))

        try:
            await apart(started())
        except BaseException:
            await self.close()
            raise

    def watch(self, log: Any) -> None:
        """Kimi's log of the session (``adapters.acp.UsageLog``), read from here on as it is
        written; and the process watched, for an end of its own."""
        self.log = log
        log.start()
        work = apart(self._watching())
        self._tasks.append(asyncio.get_running_loop().create_task(work))

    async def _drained(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(STDERR_TAIL):
            self.stderr.extend(chunk)
            del self.stderr[:-STDERR_TAIL]

    def why_ended(self) -> str:
        tail = self.stderr.decode(errors="replace").strip()
        code = self.process.returncode
        return f"the agent's process ended (exit {code}) in the middle of the turn" + (
            f": {tail}" if tail else ""
        )

    async def answered(self, request: Any) -> Any:
        """The agent's answer to ``request``; when its process ended instead (the connection
        closed with it), why it ended."""
        try:
            return await request
        except Exception as exc:
            raise await self.failed(exc) from exc

    async def failed(self, error: BaseException) -> BaseException:
        """The agent's error — or, when the connection closed (its process ended), why the
        process ended."""
        if not isinstance(error, ConnectionError):
            return error
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(CANCEL_SECONDS):
                await self.process.wait()
                assert self.drain is not None  # started with the process
                await asyncio.shield(self.drain)  # what it said last
        return ACPRunError(self.why_ended())

    async def opened(
        self,
        init: Any,
        cwd: str,
        servers: list[Any],
        runtime: Runtime,
        *,
        fork: bool,
    ) -> str:
        """The session the turn runs in: the one an earlier attempt kept, resumed (else
        loaded) when the agent still holds it — forked (``fork``) when another worker's process
        holds it — else a new one; the journal's: its id."""
        from acp import RequestError

        journal = runtime.replay.journal
        kept = journal.session
        caps = init.agent_capabilities.model_dump() if init.agent_capabilities else {}
        can = caps.get("session_capabilities") or {}
        if kept is not None:
            try:
                if fork and can.get("fork") is not None:
                    forked = await self.answered(
                        self.conn.fork_session(session_id=kept, cwd=cwd, mcp_servers=servers)
                    )
                    kept = forked.session_id
                elif can.get("resume") is not None:
                    await self.answered(
                        self.conn.resume_session(session_id=kept, cwd=cwd, mcp_servers=servers)
                    )
                else:  # its history replayed first: no turn is served to take it
                    await self.answered(
                        self.conn.load_session(session_id=kept, cwd=cwd, mcp_servers=servers)
                    )
                self.session_id = journal.session = kept
                self.continued = True
                return kept
            except RequestError as exc:
                runtime.events.warning(
                    "acp_session", f"session {kept} was not resumed ({exc}): run again"
                )
                journal.session = None
        created = await self.answered(self.conn.new_session(cwd=cwd, mcp_servers=servers))
        self.session_id = journal.session = created.session_id
        return created.session_id

    async def asking(self, mode: str | None, log: Any) -> None:
        """The session put in a mode that asks before a call, before its first prompt:
        ``mode`` on a session the run started; else, for an agent whose log says the mode a
        session is in (``UsageLog.asking_in``), that one — its plan mode kept on, a resumed
        session's too — asking, whatever the agent's own configuration started it in (Kimi's
        ``default_permission_mode`` approves on its own behind ACP's ``default``). The log
        must then say so — waited for as a prompt waits for its session
        (``live.SETTLE_SECONDS``: Kimi writes its log after it answers, long after on a busy
        server) — or the run fails (:data:`UNGOVERNED`): no call is governed in a session that
        asks about none. An agent with no such log keeps the mode it starts in."""
        given = mode if not self.continued else None
        wanted, steps = (given, (given,) if given else ()) if log is None else log.asking_in(given)
        for step in steps:
            await self.answered(
                self.conn.set_session_mode(session_id=self.session_id, mode_id=step)
            )
        if log is None:
            return

        async def confirmed() -> bool:
            return log.mode() == (wanted, None)

        # no prompt goes before the log confirms: waited for as long as a prompt waits for
        # its session (``live.SETTLE_SECONDS``), then the run fails
        if not await _logged_in_time(confirmed, live.SETTLE_SECONDS):
            found, alone = log.mode()
            raise ACPRunError(UNGOVERNED.format(wanted=wanted, found=found, alone=alone))

    # ------------------------------------------------------------------ a turn
    async def served(self, attempt: Any, native_input: Any) -> AsyncGenerator[Any, None]:
        """``attempt`` served: its turns' updates and answers (``Attempt.turns``), each request
        of the agent's in the meantime relayed to it. Then the requests it was answering are
        given up, and its calls under way end with it."""
        import contextvars

        turn = Turn(attempt.runtime, attempt, attempt.served, contextvars.copy_context())
        attempt.bind(self)
        self._attach(turn)
        try:
            async with contextlib.aclosing(attempt.turns(native_input)) as items:
                async for item in items:
                    yield item
        finally:
            under_way = list(turn.calls)
            if attempt.served is not None:  # its calls under way: told it ended first
                await attempt.served.stopped()
            await attempt.stopped()
            self._detach(turn)
            if under_way:
                await asyncio.wait(under_way)

    async def busy(self) -> bool:
        """Whether work in its background goes on (the log read first): a task at work, a
        background sub-agent's run not ended, or Kimi's own turn for a task that ended, under
        way or due."""
        await self.read()
        working = bool(self.working) or any(not b.over for b in self.background.values())
        return working or self.injected or self.due

    async def queued(self, own: int) -> bool:
        """Whether a prompt answered with no turn of its own was queued behind one of Kimi's
        (:data:`QUEUED`): one is under way now, or began since the prompt was sent (``own``:
        how many had, then) — the log read first. With no log, nothing says so."""
        if self.log is None:
            return False
        await self.read()
        return self.injected or self.own_turns > own

    # ------------------------------------------------------------------ the background
    async def launches(self, calls: Mapping[str, Any]) -> list[str]:
        """The turn's calls (``calls``, by id) whose sub-agent Kimi launched in the background,
        as its log says (``task.started``, with the call it is the task of): ACP tells no end
        of such an ``Agent`` call (Kimi 2.6.2 sends no update of its result). Each one's child
        run is the task's from here (:meth:`_launched`); their ids."""
        await self.read()
        found = [c for c in calls if (c.partition(":")[2] or c) in self._launches]
        for call_id in found:
            await self._launched(call_id, *self._launches.pop(call_id.partition(":")[2] or call_id))
        return found

    async def _launched(self, call_id: str, task_id: str, agent_id: str) -> None:
        """The ``Agent`` call ``call_id`` runs its sub-agent (``agent_id``) in the background,
        as the task ``task_id``: its child run — open since the call started — is kept past
        the turn by this session (:class:`Background`) when it is a conversation's; else it is
        the attempt's, ended by its task's end, or cut off with the attempt
        (``Helpers.settled``, ``Helpers.closed``)."""
        assert self.turn is not None  # the call is the turn's
        helpers = self.turn.runtime.helpers
        helper = helpers.detached(call_id) if self.keeps else None
        if helper is None:
            helpers.attribute(task_id, call_id, background=True)
        else:
            kept = Background(self, task_id, call_id, helper, agent_id)
            self.background[task_id] = self.sessions.attended[kept.run_id] = kept
            self.usage.skip.add(agent_id)  # its usage is its run's
        if task_id in self._ended:  # it ended before its launch was read
            await self._task_end(task_id, *self._ended[task_id])

    async def _task_end(self, task_id: str, status: str, summary: str) -> None:
        """The log says the task ``task_id`` ended: its run (kept, or the turn's helper) ends
        as it did."""
        self._ended[task_id] = (status, summary)
        self._task_ended(task_id, status, summary)
        helpers = self.turn.runtime.helpers if self.turn is not None else None
        if helpers is not None:
            failed = status != TASK_COMPLETED
            said = STOPPED if status == TASK_KILLED else summary
            await helpers.settled(task_id, failed=failed, said=said)

    async def _stop(self, task_id: str) -> None:
        """Nothing: ACP has no way to stop one task (``session/cancel`` stops the main agent's
        turn, not its tasks). It works on, refused whatever it asks, until it ends."""

    # ------------------------------------------------------------------ the log
    async def _watching(self) -> None:
        """The log read as it is written; the session closed once its process ended of
        itself."""
        while self.process.returncode is None:
            await self.read()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(USAGE_SECONDS):
                    await self.process.wait()
        self.sessions.forget(self, "its process ended")
        self._aside_run(self.close())

    async def read(self) -> None:
        """What the log says since the last read (:meth:`_logged`, a line at a time); then,
        with no turn served, the session rests or ends as its work stands."""
        if self.log is None:
            return
        for agent, entry in self.log.entries():
            await self._logged(agent, entry)
        if self.turn is None and not self.closed:
            self._rest()

    async def _logged(self, agent: str, entry: Mapping[str, Any]) -> None:
        """One line of ``agent``'s log: a task started or ended, Kimi's own turns, whose a
        permission request is, what a background sub-agent spent (nothing, of an agent the
        harness keeps no run of)."""
        kind = entry.get("type")
        info = entry.get("info") or {}
        if kind == "task.started" and info.get("detached"):
            self.working[str(info.get("taskId"))] = str(info.get("description") or "task")
            if info.get("agentId"):  # a sub-agent's: the ``Agent`` call it is the task of
                launch = (str(info.get("taskId")), str(info.get("agentId")))
                self._launches[str(info.get("parentToolCallId"))] = launch
        elif kind == "subagent.completed":
            self._said[str(entry.get("subagentId"))] = str(entry.get("resultSummary") or "")
        elif kind == "task.terminated":
            summary = self._said.get(str(info.get("agentId")), "") or str(
                info.get("stopReason") or ""
            )
            await self._task_end(str(info.get("taskId")), str(info.get("status")), summary)
        elif kind == "interaction.request":
            self._asked[str(entry.get("toolCallId"))] = (agent, _shown(entry))
        elif kind == "usage.record":
            found = next((b for b in self.background.values() if b.agent_id == agent), None)
            if found is not None and not found.over:
                found.spent(entry)
        elif agent == "main":
            self._turned(kind, entry)

    def _turned(self, kind: Any, entry: Mapping[str, Any]) -> None:
        """A turn of the main agent began or ended: one of Kimi's own (its ``turn.prompt``
        comes from a task's notification, not from the ``user``) is under way until it ends;
        once one ended the session takes a prompt (a task that ended meanwhile was told in it:
        a turn takes what came while it ran)."""
        if kind == "turn.prompt" and (entry.get("origin") or {}).get("kind") != "user":
            self.injected, self.due = True, False
            self.own_turns += 1
            self._settled.clear()
        elif kind == "turn.ended":
            self.injected = self.due = False
            self._settled.set()

    def _task_ended(self, task_id: str, status: str, summary: str) -> None:
        super()._task_ended(task_id, status, summary)
        if self.due:  # Kimi's own turn for it is to come: no prompt meanwhile
            self._settled.clear()

    async def requested(self, call_id: str) -> tuple[str | None, dict[str, Any] | None]:
        """The permission request ``call_id`` as Kimi's log has it (``interaction.request``:
        its id is the request's less its turn): the agent (its id) whose it is, and what Kimi
        shows its own user of the call — waited for while it is not written yet
        (:data:`ATTRIBUTED_SECONDS`). Neither, when no log says."""
        raw = call_id.partition(":")[2] or call_id

        async def asked() -> bool:
            await self.read()
            return raw in self._asked

        if self.log is not None:
            await _logged_in_time(asked, ATTRIBUTED_SECONDS)
        return self._asked.get(raw) or (None, None)

    # ------------------------------------------------------------------ the agent's requests
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """An update: the turn's, when one is served (Kimi sends none of another's)."""
        if self.turn is not None:
            await self.turn.options.session_update(session_id, update, **kwargs)

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        """A call the agent asks permission for, decided by whoever answers for it
        (:meth:`_relayed`); with nobody, refused (``live.UNATTENDED``). A call the turn saw no
        update of (a sub-agent's: ACP shows none, and Kimi's request carries no arguments) is
        decided with what Kimi shows its own user of it, as its arguments."""
        name = tool_call.title or ""
        found, shown = await self._asker(tool_call.tool_call_id)
        if shown is not None and tool_call.raw_input is None:
            tool_call = tool_call.model_copy(update={"raw_input": shown})
        if found is not None and name.startswith(HARNESS_PREFIX):  # its harness tool's call
            self._calling.setdefault(name.removeprefix(HARNESS_PREFIX), []).append(found)
        answer = await self._relayed(
            found, lambda to: to.request_permission(options, session_id, tool_call)
        )
        if answer is not OVER:
            return answer
        self.refused[name] += 1
        return chosen(options, "reject_once")

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        """A form the agent asks its user to fill (Kimi's ``AskUserQuestion``: the main
        agent's — its sub-agents have no such tool), asked by the turn being served; with
        none, left unfilled."""
        answer = await self._relayed(None, lambda to: to.create_elicitation(message, mode))
        return unfilled() if answer is OVER else answer

    async def _asker(self, call_id: str) -> tuple[Background | None, dict[str, Any] | None]:
        """Whose request the one for the call ``call_id`` is — a background sub-agent's run,
        else ``None``: the main agent's, or a sub-agent's of the turn — and what Kimi shows
        its own user of that call (:meth:`requested`; nothing is read for a call the turn
        saw: its own)."""
        turn = self.turn
        main = turn is not None and call_id in turn.options.calls
        agent, shown = (None, None) if main else await self.requested(call_id)
        if agent is not None and turn is not None:
            # a sub-agent just launched may ask before the turn took its launch: taken now
            with contextlib.suppress(Left):
                await turn.within(turn.options.launched())
        return next((b for b in self.background.values() if b.agent_id == agent), None), shown

    async def _relayed(
        self, found: Background | None, asked: Callable[[Any], Coroutine[Any, Any, Any]]
    ) -> Any:
        """A request of the agent's (``asked``: made of the ACP client that answers it) to
        whoever answers for it: a background sub-agent's by its run's attempt
        (``adapters.acp.Deciding``), turn or no turn; any other by the turn being served
        (its ``adapters.acp.Attempt``). ``OVER`` with nothing to answer it — no turn, a
        background run that ended."""
        turn = self.turn  # the turn being served once whose request it is was read
        if found is not None:

            async def decided(at: Turn) -> Any:
                return await asked(at.options)

            return await found.asked(decided)
        if turn is not None:
            with contextlib.suppress(Left):
                return await turn.within(asked(turn.options))
        return OVER

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """A harness tool's call (the MCP server's): a background sub-agent's (its permission
        request said so) made in its run's attempt; else the turn being served's; with neither,
        refused."""
        from trellis.harness.tools.convert import mcp

        waiting = self._calling.get(name)
        found = waiting.pop(0) if waiting else None
        if found is not None:

            async def made(at: Turn) -> Any:
                return await at.tools.called(name, args)

            answer = await found.asked(made)
            if answer is not OVER:
                return answer
        turn = self.turn if found is None else None
        if turn is not None:  # it lists what the process lists (its shape): it has the tool
            with contextlib.suppress(Left):
                return await turn.within(turn.tools.called(name, args))
        self.refused[name] += 1
        return mcp.refused(UNATTENDED)

    # the agent's files and terminals: a sandboxed run's, whose process is never kept — its
    # turn's attempt answers them (``adapters.acp.Sandboxed``)
    def _boxing(self) -> Any:
        assert self.turn is not None  # the requests of a sandboxed run's own turn
        return self.turn.options

    async def read_text_file(self, **kwargs: Any) -> Any:
        return await self._boxing().read_text_file(**kwargs)

    async def write_text_file(self, **kwargs: Any) -> Any:
        return await self._boxing().write_text_file(**kwargs)

    async def create_terminal(self, **kwargs: Any) -> Any:
        return await self._boxing().create_terminal(**kwargs)

    async def terminal_output(self, **kwargs: Any) -> Any:
        return await self._boxing().terminal_output(**kwargs)

    async def wait_for_terminal_exit(self, **kwargs: Any) -> Any:
        return await self._boxing().wait_for_terminal_exit(**kwargs)

    async def kill_terminal(self, **kwargs: Any) -> Any:
        return await self._boxing().kill_terminal(**kwargs)

    async def release_terminal(self, **kwargs: Any) -> Any:
        return await self._boxing().release_terminal(**kwargs)

    # ------------------------------------------------------------------ its end
    def _closing(self) -> None:
        """Nothing waits on the session itself: a turn's own queue is its attempt's."""

    async def _disconnect(self) -> None:
        """The log's reader stopped, the harness's MCP server and the process ended (the
        SDK's own shutdown: a process already gone is no error)."""
        for task in self._tasks:  # the log's reader (the stderr's ends with the process)
            task.cancel()
        await self._stack.aclose()

    async def released(self) -> None:
        """The conversation let go of the session for good: its next turn is not sent to
        continue it (``Agent.sessions``) — at once, so a turn that comes while it closes takes
        its earlier turns from the harness, as a turn on a server that kept no session does —
        and once the process ended, the session's own files leave the agent's home
        (``UsageLog.removed``: its user's Kimi home keeps its login and its config), whatever
        closing meets."""
        assert self.kept_in is not None  # a session is kept once it served a turn
        kept = self.kept_in.get(self.key)
        if kept is not None and kept[0] == self.session_id:
            self.kept_in.pop(self.key)
        try:
            await self.close()
        finally:  # a kept session's agent keeps a log (``adapters.acp._follows``)
            self.log.removed()


def shape(target: Any, env: Mapping[str, str], tools: Any, headers: str) -> str:
    """What a process started for ``target`` with ``env`` and the harness's ``tools`` has for
    good, which a later turn must bring the same to be served by it: its command, its working
    directory, its environment less what each turn brings of its own (its run's gateway
    headers: ``headers``), its own MCP servers, its mode, its model, and the harness tools its
    server lists with their schemas."""
    listed = [] if tools is None else tools.listed
    kept = {k: v for k, v in env.items() if k != headers}
    fixed = [target.command, target.cwd, kept, target.mcp_servers, target.mode, target.model]
    return json.dumps([*fixed, listed], sort_keys=True, default=str)


__all__ = ["CODE", "GONE", "GONE_SAID", "LOST", "QUEUED", "ACPSession", "Background", "shape"]
