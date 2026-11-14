"""A conversation's Claude Code process, kept between its turns.

The Claude Agent SDK's background work — a subagent with ``background``, a shell command run
in the background — belongs to the CLI process that started it: it goes on after the turn's
``ResultMessage`` and ends with the process. So a conversation's turn does not end its CLI:
the process stays (:class:`LiveSession`, the SDK's own ``ClaudeSDKClient`` in streaming-input
mode), a reader goes on consuming its stream, and the conversation's next turn is sent into
it (``client.query``) instead of into a new process that resumes the session. What it shares
with an ACP agent's (``bridges.acp_live``) — its relays, its background runs' records, its
life — is ``bridges.live``'s.

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
  that follows the answer decides the call from its journal.
* **No run.** Any other call made between two turns — the main agent's in a turn the
  session injected, a subagent's with no run — has no run under way: nothing can decide it
  (the run's hooks, governance, a person's approval, its budget) or record it. Such a call is
  refused (``live.UNATTENDED``), never run unchecked, and the conversation's next run says
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
  session's own is seen under way (``Keeper.settled``).
* **Spending.** A result's cost is the process's running total; its usage is the turn's own.
  A turn left before its result is charged that result when it comes at once (interrupted:
  :meth:`LiveSession.outcome`); never the conversation's next turn.
  What the cost grew by since the last result charged (:meth:`LiveSession.charged`) is
  shared by the tokens each used: the turn the result ends its own usage, each background
  run what its subagent's replies used since (the share of a turn the session injected
  for an ended task is that task's run's). With no background run, what injected turns and
  background shell commands cost is on the conversation's next turn, as their tokens are in
  no turn's usage.
* **Its life.** One process per conversation, in this process only; at most ``live.KEPT``. It
  closes ``live.IDLE_SECONDS`` after its last activity when nothing works in it (no background
  task, and no turn of its own for a finished one under way or due), and
  ``live.WORKING_SECONDS`` after its last turn whatever does; when a turn needs other options
  than it was started with; and with the harness (``Harness.aclose``). Work that ends with
  it is said in the conversation's next run (``claude_session``), never dropped in silence.
  With no live session — a first turn, another server, a restart, one closed — a turn runs as
  it always did: a new process that resumes the conversation's session (``resume=``) from
  the journal's transcripts, which hold what was mirrored between turns too.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Final

from trellis.harness import budget
from trellis.harness.asking import RunCancelled
from trellis.harness.bounded import LRU
from trellis.harness.bridges import live
from trellis.harness.bridges.claude import JournalSessions, ours
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
from trellis.harness.journal import Journal

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The most messages kept of those that arrived with no turn to take them (the most recent).
UNATTENDED_KEPT: Final = 200
#: What the main agent's call in a turn that was left is refused with: the turn ends there.
LEFT: Final = "The user left this turn: it ends here."
#: Why a turn whose session's process ended before its result failed.
ENDED: Final = "Claude Code's process ended before the turn's result"
#: How long the CLI is told to wait for a hook's answer: twice as long as a background
#: subagent's run waits for a person (``live.ANSWER_SECONDS``) — the harness answers first, and
#: a call the CLI gave up on is one it did not run.
HOOK_SECONDS: Final = 2 * live.ANSWER_SECONDS
#: Why a background subagent's run ended with its session, and when its task was stopped in
#: Claude Code (by Claude, or with the turn it was started in).
LOST: Final = (
    "Claude's session was closed ({why}): the background subagent's task was stopped with it"
)
STOPPED: Final = "the background subagent's task was stopped in Claude Code"
#: Why a background subagent's run whose question was answered on a server its session is not
#: on ended (its task went with the server that held the session, or lives on another): its
#: error's message, and what the turn that took the answer is told (``live.gone``).
GONE: Final = (
    "the background subagent's Claude session is not on the server the answer reached (it "
    "ended with the server that held it, or lives on another): the answer cannot continue its "
    "task, which is gone from here"
)
GONE_SAID: Final = (
    "A background subagent's question was answered, but its Claude session is gone (the "
    "server that held it stopped): its task cannot go on, and its run ended."
)
#: What the session says on a run's stream is under this code.
CODE: Final = "claude_session"


def unattended(session: LiveSession, message: Any) -> None:
    """A message that arrived while no turn could take it — a background task's progress,
    its notification, the turn the session injected for it and that turn's result: kept, the
    most recent :data:`UNATTENDED_KEPT`. The one place such messages pass: delivering them
    into the conversation is built from here."""
    session.kept.append(message)


def _tokens(usage: Mapping[str, Any] | None) -> int:
    """The tokens a reply (or a turn's result) says it used, read and written."""
    used = usage or {}
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(int(used.get(key) or 0) for key in (*keys, "output_tokens"))


class Background(Kept):
    """A subagent at work in the background of a Claude session (``task_started`` with
    ``is_backgrounded``), and the child run that is its record (``live.Kept``), kept open from
    its ``Agent`` call until the SDK says the task ended.

    Each attempt is served the subagent's own messages (its ``parent_tool_use_id`` is the
    ``Agent`` call: :meth:`took`) and ends with a result made of the task's end — ``SUCCESS``
    with what the subagent said last, ``ERROR`` for a task that failed, ``CANCELLED`` for one
    stopped in Claude Code. A run that ends first (paused, cancelled, out of time, its budget
    spent) has its task stopped with the SDK's own ``stop_task``: an interrupt would stop
    every task. The CLI names the subagent in its requests by the task's id (``agent_id``)."""

    stopped_why: ClassVar[str] = STOPPED
    ended_why: ClassVar[str] = ENDED

    def __init__(self, session: LiveSession, task_id: str, call_id: str, helper: Any) -> None:
        super().__init__(session, task_id, call_id, helper)
        self.session: LiveSession = session
        #: what the subagent said last
        self.said = ""
        #: the tokens its replies used since the process's cost was last shared out
        self.unbilled = 0
        self._counted: set[str] = set()

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

    def _charge(self, turn: Turn, *spent: Any) -> None:
        """A reply's ``usage``, or a share of the process's ``cost``, on the attempt ``turn``
        and on its budget: spent, the task is stopped."""
        usage, cost = spent
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

    def waits(self, answer: Any) -> bool:
        """Whether a relayed request was answered that its run pauses on it — the adapter's
        hook (its ``stopReason``), its permission check (its message), a harness tool (its
        result)."""
        from trellis.harness.tools.convert.claude import WAITING

        if not isinstance(answer, dict):
            return getattr(answer, "message", None) == WAITING
        content = answer.get("content")
        said = isinstance(content, list) and any(c.get("text") == WAITING for c in content)
        return answer.get("stopReason") == WAITING or (bool(answer.get("is_error")) and said)


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


class LiveSession(Keeper[Background]):
    """One conversation's CLI process and what reads it."""

    lost: ClassVar[str] = LOST
    lost_work: ClassVar[str] = (
        "Claude's session was closed ({why}): the background work under way in it was "
        "stopped: {tasks}"
    )
    code: ClassVar[str] = CODE

    def __init__(self, key: str, shape: Any, sessions: LiveSessions) -> None:
        super().__init__(key, sessions)
        #: what the process was started with and has for good (``adapters.claude._shape``)
        self.shape = shape
        self.client: Any = None
        self.session_id: str | None = None
        self.mirror: Mirror | None = None
        #: the harness tool calls their hook let through, by call: whose each is
        self._calling: dict[str, tuple[str, dict[str, Any], Background]] = {}
        #: the messages no turn took (:func:`unattended`)
        self.kept: deque[Any] = deque(maxlen=UNATTENDED_KEPT)
        #: the prompts sent whose result has not come (one at most)
        self.owed = 0
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
        self._reader: asyncio.Task[None] | None = None

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
            await apart(self._connected())
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
            found = self.background.get(str(given.get("agent_id") or ""))
            if found is not None:
                answer = await found.asked(
                    lambda at: at.options.hooks[event][matcher].hooks[callback](
                        given, tool_use_id, hook
                    )
                )
                if answer is not OVER:
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

    def _ends(self, agent_id: Any) -> bool:
        """Whether a call no turn can take ends the CLI's turn: the main agent's own
        (a subagent's names it), in a turn that was left and whose result is still to come."""
        return bool(self.owed) and not agent_id

    async def _can_use_tool(self, name: str, args: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultDeny

        found = self.background.get(str(getattr(context, "agent_id", None) or ""))
        if found is not None:
            answer = await found.asked(lambda at: at.options.can_use_tool(name, args, context))
            if answer is not OVER:
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
            if answer is not OVER:
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
            summary = str(getattr(message, "summary", "") or "")
            self._task_ended(message.task_id, str(message.status), summary)

    def _launched(self, message: Any) -> None:
        """A subagent the turn being served started in the background (the CLI says which
        it runs so: ``is_backgrounded``): the child run its ``Agent`` call started is kept
        open from here (:class:`Background`), no longer that turn's to end."""
        turn = self.turn
        started = None if turn is None else turn.runtime.helpers
        if started is None or not message.tool_use_id or not message.data.get("is_backgrounded"):
            return
        helper = started.detached(message.tool_use_id)
        if helper is not None:
            found = Background(self, message.task_id, message.tool_use_id, helper)
            self.background[message.task_id] = self.sessions.attended[found.run_id] = found

    async def _stop(self, task_id: str) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(live.SETTLE_SECONDS):
                await self.client.stop_task(task_id)

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
        super()._attach(turn)
        self.left = None  # its result came (:meth:`settled`), or never will
        self.kept_in = turn.runtime.agent.sessions
        if self.mirror is not None:
            store = turn.options.session_store
            assert isinstance(store, JournalSessions)  # as the process's is (its shape)
            self.mirror.serve(store)

    def _detach(self, turn: Turn) -> None:
        super()._detach(turn)
        if self.mirror is not None:
            self.mirror.into = None

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
            async with asyncio.timeout(live.SETTLE_SECONDS):
                while (message := await turn.queue.get()) is not None:
                    if isinstance(message, ResultMessage):
                        return message
        if self.left is turn:
            self.left = None
        return None

    async def _send(self, turn: Turn, prompt: str) -> None:
        self.owed = 1
        self.injected = False
        self._settled.clear()
        turn.listening = True
        await self.client.query(prompt)

    async def _interrupt(self) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(live.SETTLE_SECONDS):
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
    # ------------------------------------------------------------------ its end
    def _closing(self) -> None:
        for turn in (self.turn, self.left):
            if turn is not None:
                turn.queue.put_nowait(None)

    async def _disconnect(self) -> None:
        """The process ended (its stdin closed; stopped if it does not end) and its reader
        with it; the conversation's session kept, with all that was mirrored, for a turn that
        resumes it."""
        kept = None if self.kept_in is None else self.kept_in.get(self.key)
        if self.mirror is not None and kept is not None and kept[0] == self.session_id:
            assert self.kept_in is not None
            self.kept_in[self.key] = (kept[0], dict(self.mirror.transcripts))
        if self._reader is not None and self._reader is not asyncio.current_task():
            self._reader.cancel()
        if self.client is not None:
            with contextlib.suppress(Exception):
                await self.client.disconnect()


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


__all__ = ["Background", "LiveSession", "Mirror"]
