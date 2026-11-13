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
* **No run.** Between two turns no run is under way: nothing can decide a tool call (the
  run's hooks, governance, a person's approval, its budget) or record it. Such a call is
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
  A turn is charged what the cost grew by since the last charge
  (:meth:`LiveSession.charged`) — what background work and injected turns cost in between is
  on the conversation's next turn; their tokens are in no turn's usage.
* **Its life.** One process per conversation, in this process only; at most :data:`KEPT`. It
  closes :data:`IDLE_SECONDS` after its last activity when nothing works in it, and
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
from collections import Counter, deque
from collections.abc import AsyncGenerator, Awaitable, Coroutine, Mapping
from typing import Any, Final

from trellis.harness.bounded import LRU
from trellis.harness.bridges.claude import JournalSessions
from trellis.harness.journal import Journal

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


class Left(Exception):
    """The turn a request was handed to was left before it answered."""


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
        #: the messages no turn took (:func:`unattended`)
        self.kept: deque[Any] = deque(maxlen=UNATTENDED_KEPT)
        #: the tool calls refused since the last turn for want of a run, by tool
        self.refused: Counter[str] = Counter()
        #: the prompts sent whose result has not come (one at most)
        self.owed = 0
        #: a turn the session injected is under way
        self.injected = False
        self.closed = False
        #: why the process ended, when it ended of itself with an error
        self.error: BaseException | None = None
        #: the process's running cost as the last charged result carried it
        self.billed = 0.0
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
                    matcher, hooks=[self._hook(event, m, c) for c in range(len(matcher.hooks))]
                )
                for m, matcher in enumerate(matchers)
            ]
            for event, matchers in hooks.items()
        }

    def _hook(self, event: str, matcher: int, callback: int) -> Any:
        """The relay of one hook: the same hook of the turn being served; with none, a tool
        call is refused (:data:`UNATTENDED`) and any other event passes."""

        async def relay(given: Any, tool_use_id: str | None, hook: Any) -> Any:
            turn = self.turn
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

    def _ends(self, agent_id: Any) -> bool:
        """Whether a call no turn can take ends the CLI's turn: the main agent's own
        (a subagent's names it), in a turn that was left and whose result is still to come."""
        return bool(self.owed) and not agent_id

    async def _can_use_tool(self, name: str, args: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultDeny

        turn = self.turn
        if turn is not None:
            with contextlib.suppress(Left):
                return await turn.within(turn.options.can_use_tool(name, args, context))
        self.refused[name] += 1
        if self._ends(getattr(context, "agent_id", None)):
            return PermissionResultDeny(message=LEFT, interrupt=True)
        return PermissionResultDeny(message=UNATTENDED)

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A harness tool's call: the tool of that name the turn being served has."""
        from trellis.harness.tools.convert import claude as convert

        turn = self.turn
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
            self._settled.clear()
        turn = self.turn
        mine = turn is not None and turn.listening and not turn.answered
        if result and own:  # a prompt is sent once the one before has its result
            self.owed = 0
        if turn is not None and mine and (own if result else not self.injected):
            turn.answered = result
            turn.queue.put_nowait(message)
        else:
            unattended(self, message)
            if turn is None:
                self._rest()
        if result:
            self.injected = self.injected and own
            if not self.owed and not self.injected:
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
        elif isinstance(message, TaskNotificationMessage) or (
            isinstance(message, TaskUpdatedMessage) and message.status in TERMINAL_TASK_STATUSES
        ):
            self.working.pop(message.task_id, None)

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
            if left and not self.closed and not self.working:
                self._aside_run(self._interrupt())
            # the calls it left under way end with it (cancelled above), as a query's own
            # process ends its calls when it is closed: what one says as it ends (a helper's
            # run: ``helper_finished``) is on the run's stream before the run's end
            if under_way:
                await asyncio.wait(under_way)

    def _attach(self, turn: Turn) -> None:
        self.turn = turn
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
        if self.mirror is not None:
            self.mirror.into = None
        if not self.closed:
            self._rested = asyncio.get_running_loop().time()
            self._rest()

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

    def charged(self, result: Any) -> Any:
        """``result`` with the cost this turn is charged: what the process's running total
        grew by since the last charged result (a total that did not grow is taken as the
        turn's own: never less than nothing). Its usage is the turn's own already."""
        if result.total_cost_usd is None:
            return result
        total = float(result.total_cost_usd)
        more = total - self.billed
        self.billed = total
        return dataclasses.replace(result, total_cost_usd=more if more >= 0 else total)

    # ------------------------------------------------------------------ its end
    def _rest(self) -> None:
        """No turn is served: the session closes when its time is up (:data:`IDLE_SECONDS`
        after this, nothing working in it; :data:`WORKING_SECONDS` after its last turn)."""
        loop = asyncio.get_running_loop()
        if self._timer is not None:
            self._timer.cancel()
        last = self._rested + WORKING_SECONDS - loop.time()
        wait = max(0.0, last) if self.working else min(IDLE_SECONDS, max(0.0, last))
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
        resumes it."""
        if self.closed:
            return
        self.closed = True
        self._settled.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        turn = self.turn
        if turn is not None:
            turn.queue.put_nowait(None)
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


class LiveSessions:
    """An agent's live sessions in this process, by conversation."""

    def __init__(self) -> None:
        self._open: dict[str, LiveSession] = {}
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
        """Every session closed: the harness is closing."""
        sessions, self._open = list(self._open.values()), {}
        await asyncio.gather(*(s.close() for s in sessions))


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


__all__ = ["IDLE_SECONDS", "KEPT", "UNATTENDED", "LiveSession", "LiveSessions", "Turn"]
