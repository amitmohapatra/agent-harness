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
  that follows the answer decides the call from its journal. A dynamic workflow
  (``local_workflow``) is kept so too: its agents name only their kind
  (:data:`WORKFLOW_AGENT`), so their calls are its run's while it is the one workflow at
  work, and those of a run the workflows share while several are (:data:`SHARED`).
* **No run.** Any other call made between two turns — the main agent's in a turn the
  session injected, a subagent's with no run — has no run under way: nothing can decide it
  (the run's hooks, governance, a person's approval, its budget) or record it. Such a call is
  refused (``live.UNATTENDED``), never run unchecked, and the conversation's next run says
  how many were. Messages that arrive then are nobody's answer; Claude's own session holds
  them, so the next turn's model knows them.
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
import contextvars
import dataclasses
import logging
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final

from trellis.harness import budget
from trellis.harness.asking import RunCancelled
from trellis.harness.bounded import LRU
from trellis.harness.bridges import live
from trellis.harness.bridges.claude import (
    JournalSessions,
    SessionCost,
    TaskEnded,
    TaskStarted,
    ours,
    task_said,
)
from trellis.harness.bridges.helpers import Kind
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
#: Claude Code's dynamic workflow, as the bundled CLI (2.1.283) sends it: its tool, the task it
#: runs as (``task_type``: no ``is_backgrounded``, it is always in the background), and what
#: its agents are named in their hooks (``agent_type``; each ``agent_id`` is the agent's own,
#: named in no frame before its first call). While one works in a session its agents' calls
#: are its run's by their kind; while several do, an agent first seen then cannot be told
#: apart, and its calls are those of a run the workflows share (:data:`SHARED`:
#: :meth:`LiveSession._working`).
WORKFLOW_TOOL: Final = "Workflow"
WORKFLOW_AGENT: Final = "workflow-subagent"
#: The run the workflows at work together in a session share, for the calls of agents no one
#: of them can claim: a child run of the turn that made them concurrent, kept as a background
#: subagent's (:class:`Background`) until one workflow at most is left. Its name and task,
#: and what it answers then.
SHARED: Final = "workflows"
SHARED_TASK: Final = (
    "the calls of agents of Claude Code's dynamic workflows at work together, which none of "
    "them can be told to have made"
)
SHARED_SAID: Final = "{count} call(s) of workflow agents no one workflow could claim: {tools}"


#: The token counts of a reply's (or a turn's result's) usage, read and written.
TOKEN_KEYS: Final = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)


def _added(usage: Mapping[str, Any] | None, more: Mapping[str, Any]) -> dict[str, Any]:
    """``usage`` with the token counts of ``more`` added to its own."""
    used = usage or {}
    return {**used, **{k: int(used.get(k) or 0) + int(more.get(k) or 0) for k in TOKEN_KEYS}}


def _tokens(usage: Mapping[str, Any] | None) -> int:
    """The tokens a reply (or a turn's result) says it used, read and written."""
    used = usage or {}
    return sum(int(used.get(key) or 0) for key in TOKEN_KEYS)


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

    def __init__(
        self,
        session: LiveSession,
        task_id: str,
        call_id: str,
        helper: Any,
        turn: Turn | None = None,
    ) -> None:
        super().__init__(session, task_id, call_id, helper, turn)
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

    def __init__(self, key: str, shape: Any, sessions: LiveSessions, *, owner: Hashable) -> None:
        super().__init__(key, sessions, owner=owner)
        #: what the process was started with and has for good (``adapters.claude._shape``)
        self.shape = shape
        self.client: Any = None
        self.session_id: str | None = None
        self.mirror: Mirror | None = None
        #: the workflow's agents (by their own ``agent_id``) whose calls its run decides
        self._workers: dict[str, Background] = {}
        #: the run the workflows at work together share (:data:`SHARED`), being started or
        #: kept; the calls it decided, by tool; how many such runs the session started
        self._sharing: asyncio.Task[Background] | None = None
        self.shared: Background | None = None
        self._owned: Counter[str] = Counter()
        self._shares = 0
        #: the harness tool calls their hook let through, by call: whose each is
        self._calling: dict[str, Background] = {}
        #: the prompts sent whose result has not come (one at most)
        self.owed = 0
        #: why the process ended, when it ended of itself with an error
        self.error: BaseException | None = None
        #: the session's running cost as the last charged result carried it, shared with
        #: the spending of the query it serves (``bridges.claude.SessionCost``)
        self.cost = SessionCost()
        #: the tokens of the turns the session ran that no run was charged (a background
        #: shell command's end): the next turn served is charged them, with their cost
        self._unclaimed: dict[str, Any] = {}
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
                    hooks=[
                        self._hook(event, m, c, None if ours(matcher) else own)
                        for c, own in enumerate(matcher.hooks)
                    ],
                    # a background run's call may wait for a person in the harness's hooks;
                    # a team's keep what they set (none: Claude Code's own default)
                    timeout=HOOK_SECONDS if ours(matcher) else matcher.timeout,
                )
                for m, matcher in enumerate(matchers)
            ]
            for event, matchers in hooks.items()
        }

    def _hook(self, event: str, matcher: int, callback: int, theirs: Any) -> Any:
        """The relay of one hook: the same hook of the turn being served — of the attempt of
        its run, for a background subagent that has one (:class:`Background`). With no run
        under way to take it (between turns: a subagent's end, the turn the session runs by
        itself), a hook of the team's own (``theirs``: as the process was started with it)
        runs as Claude Code runs it, in no run's context — it is the library's, never
        suppressed, and run once; the harness's own has no run to decide or record for: a
        tool call is refused (:data:`UNATTENDED`) and any other event passes."""
        from trellis.harness.tools.convert import claude as convert

        async def relay(given: Any, tool_use_id: str | None, hook: Any) -> Any:
            found = await self._working(given.get("agent_id"), given.get("agent_type"))
            if found is not None and found is self.shared and event == "PreToolUse":
                tool = str(given.get("tool_name") or "")
                self._owned[convert.bare(tool) or tool] += 1
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
            elif theirs is not None:
                return await theirs(given, tool_use_id, hook)
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
        """Whose a harness tool's call is, once its hook let it through, by Claude's id of
        the call (the request that makes it names the id and not who makes it:
        :meth:`_call`); forgotten when a hook refused it, and once it ran."""
        from trellis.harness.tools.convert import claude as convert

        name, ref = str(given.get("tool_name") or ""), str(given.get("tool_use_id") or "")
        if convert.bare(name) is None:
            return
        said = (answer.get("hookSpecificOutput") if isinstance(answer, dict) else None) or {}
        decided = said.get("permissionDecision")
        if event == "PreToolUse" and decided == "allow":
            self._calling[ref] = found
        elif event != "PreToolUse" or decided == "deny":
            self._calling.pop(ref, None)

    async def _working(self, agent_id: Any, agent_type: Any = None) -> Background | None:
        """The background run whose work a call of the CLI's ``agent_id`` is: a subagent's,
        named by its task's id, or a workflow's, whose agents name their kind
        (:data:`WORKFLOW_AGENT`) in their calls' hooks — the one workflow at work then, else
        the run the workflows at work share (:data:`SHARED`) — the agent remembered for its
        permission requests and harness tool calls, which name it alone."""
        who = str(agent_id or "")
        found = self.background.get(who) or self._workers.get(who)
        if found is not None or agent_type != WORKFLOW_AGENT:
            return found
        flows = self._flows()
        if len(flows) == 1:
            found = flows[0]
        elif self._sharing is not None:
            found = await self._sharing
        if found is not None:
            self._workers[who] = found
        return found

    def _flows(self) -> list[Background]:
        """The dynamic workflows at work in the session (their tasks not ended)."""
        return [
            b
            for b in self.background.values()
            if b.kind is Kind.WORKFLOW and b is not self.shared and b.ended is None
        ]

    async def _shared_run(self, turn: Turn) -> Background:
        """The run the workflows ``turn`` made concurrent share: started as a helper of that
        turn and handed on to the session, as a background subagent's is (:data:`SHARED`)."""
        helpers = turn.runtime.helpers
        assert helpers is not None  # the turn's attempt has its helpers
        self._shares += 1
        key = f"{SHARED}:{self._shares}"
        helper = await helpers.start(key, kind=Kind.WORKFLOW, name=SHARED, task=SHARED_TASK)
        helpers.detached(key)
        found = self.shared = Background(self, key, "", helper, turn)
        self.background[key] = self.sessions.attended[found.run_id] = found
        return found

    def _unshared(self) -> None:
        """One workflow at most is left at work: the run they shared ends, saying what it
        decided, and that workflow's agents' calls are its run's."""
        shared = self.shared
        if shared is None or len(self._flows()) > 1:
            return
        self._sharing = self.shared = None
        owned, self._owned = self._owned, Counter()
        said = SHARED_SAID.format(count=sum(owned.values()), tools=", ".join(sorted(owned)))
        shared.finished("completed", said)
        for who in [w for w, b in self._workers.items() if b is shared]:
            del self._workers[who]

    def _forget(self, found: Kept) -> None:
        """``found``'s task ended and its run too: it is no longer the session's, nor its
        workflow agents' (:meth:`_working`)."""
        super()._forget(found)
        for who in [w for w, b in self._workers.items() if b is found]:
            del self._workers[who]

    def _ends(self, agent_id: Any) -> bool:
        """Whether a call no turn can take ends the CLI's turn: the main agent's own
        (a subagent's names it), in a turn that was left and whose result is still to come."""
        return bool(self.owed) and not agent_id

    async def _can_use_tool(self, name: str, args: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultDeny

        found = await self._working(getattr(context, "agent_id", None))
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

        found = self._calling.pop(convert.call_id() or "", None)
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
            if turn is None:
                self._rest()
        if result and not self.owed and not self.injected:
            self._settled.set()

    def _tracked(self, message: Any) -> None:
        """The background tasks under way, as the stream says (``task_said``): started, or
        ended (its notification, or an update to a terminal status: either may be all that
        comes)."""
        said = task_said(message)
        if isinstance(said, TaskStarted):
            self.working[said.task_id] = said.described
            self._launched(said)
        elif isinstance(said, TaskEnded):
            self._task_ended(said.task_id, said.status, said.summary)

    def _task_ended(self, task_id: str, status: str, summary: str) -> None:
        super()._task_ended(task_id, status, summary)
        self._unshared()

    def _launched(self, message: TaskStarted) -> None:
        """A subagent the turn being served started in the background (the CLI says which
        it runs so: ``is_backgrounded``): the child run its ``Agent`` call started is kept
        open from here (:class:`Background`), no longer that turn's to end."""
        turn = self.turn
        started = None if turn is None else turn.runtime.helpers
        if started is None or not message.call_id or not message.background:
            return
        helper = started.detached(message.call_id)
        if helper is not None:
            found = Background(self, message.task_id, message.call_id, helper)
            self.background[message.task_id] = self.sessions.attended[found.run_id] = found
        if turn is not None and len(self._flows()) > 1 and self._sharing is None:
            # workflows at work together: what none of them can claim is a run they share
            self._sharing = asyncio.get_running_loop().create_task(
                self._shared_run(turn), context=contextvars.Context()
            )

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
        result carries is the session's running total: what it grew by since the last result
        charged (``SessionCost.grew_by``: all of a total that was reset) is shared by the
        tokens each used since — the turn its own usage, each background
        run what its subagent's replies used. The share of a turn no run is served for is
        the run's whose task's end the session ran it for, with that turn's tokens; with
        none, it stays for the conversation's next turn, its tokens too, as the whole does
        with no background run to charge."""
        open_ = [b for b in self.background.values() if not b.over]
        runs = [b for b in open_ if b.unbilled]
        after = None if own else next((b for b in open_ if b.ended is not None), None)
        for found in open_:
            if found.ended is not None:  # what its task cost is known now
                found._settled.set()
        if not taken and after is None:  # nobody's turn: its tokens are the next turn's
            self._unclaimed = _added(self._unclaimed, result.usage or {})
        elif taken and self._unclaimed:
            result = dataclasses.replace(result, usage=_added(result.usage, self._unclaimed))
            self._unclaimed = {}
        if result.total_cost_usd is None or not (taken or runs or after):
            return result
        total = float(result.total_cost_usd)
        more = self.cost.grew_by(total)
        mine = _tokens(result.usage)
        whole = mine + sum(b.unbilled for b in runs)
        share = more * mine / whole if whole else more
        for found in runs:
            found.spent(None, more * found.unbilled / whole)
            found.unbilled = 0
        if taken:
            return dataclasses.replace(result, total_cost_usd=share)
        if after is not None:  # the turn run for its task's end is its run's: tokens too
            after.spent(result.usage, share)
        else:
            self.cost.owed(share)
        return result

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


async def opened(
    sessions: LiveSessions,
    key: str,
    shape: Any,
    options: Any,
    transcripts: Mapping[str, list[dict[str, Any]]],
    *,
    owner: Hashable,
    among: Sequence[LiveSessions],
) -> LiveSession | None:
    """A session started for ``key``, ``owner``'s conversation, and kept by ``sessions`` in a
    place taken before its process starts (``LiveSessions.place``, counted over ``among``).
    ``None``: there is no place — the turn runs without one, as it always did."""
    async with sessions.place(among, owner=owner) as room:
        if not room:
            return None
        session = LiveSession(key, shape, sessions, owner=owner)
        await session.start(options, transcripts)
        sessions.keep(session)
        return session


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


__all__ = ["WORKFLOW_AGENT", "WORKFLOW_TOOL", "Background", "LiveSession", "Mirror", "opened"]
