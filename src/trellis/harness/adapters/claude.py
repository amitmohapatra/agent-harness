"""Claude Agent SDK: the target is the ``ClaudeAgentOptions`` a team already configured.

* input: the prompt; the memory context goes with every prompt as Claude Code's own
  ``UserPromptSubmit`` hook adds context (``bridges.claude.with_context``), beside the options'
  hooks;
* run: ``query(prompt, options)`` on a copy of the options that also carries the harness
  tools as one in-process MCP server (``mcp__trellis__*``) beside the team's own servers — a
  ``mcp_servers`` given as a config file or JSON text is read, since the SDK serves an
  in-process server only from a dict;
* permissions: Claude Code's own, then a person when it asks. The SDK's ``PreToolUse`` hook,
  which the CLI runs before every tool call whatever its ``permission_mode`` or
  ``allowed_tools``, lets a harness tool through (the bridge is its check) and decides any
  other (Claude Code's built-ins: ``Bash``, ``Write``...; a team's own MCP server's) as a
  harness call is — the run's hooks, then governance by its risk (:data:`BUILTIN_SIDE_EFFECTS`)
  and the tool catalog's rules — and leaves the rest to Claude Code: its permission mode, its
  rules (``allowed_tools``, its settings), and when it would ask its user, ``can_use_tool``.
  That callback answers as the harness's: the team's own ``can_use_tool`` when it has one;
  else a call a person already answered about in the hook runs (one call, one question),
  a call that only reads (a read tool, or a shell command ``governance.shell.read_only``
  reads as read-only) runs, and any other asks a person (``bridge.asked``);
* records: every call Claude runs itself (its built-ins, its own MCP servers' tools, a subagent's
  calls) is recorded as the run's tool calls once it ran — its result or its error, and how long
  it took — through the SDK's own ``PostToolUse`` / ``PostToolUseFailure`` hooks
  (``tools.bridge.observed``); the harness's own tools are recorded by the bridge;
* stalls: the query is watched for a stall — a model step that sends nothing for
  ``runtime.STALL_SECONDS`` (the time a tool call runs, from ``PreToolUse`` to its
  ``PostToolUse``, not counted) ends the run with a ``ModelTimeout``;
* spending: each query's usage and cost (its ``ResultMessage``) count toward the run's budget;
* helpers: each subagent Claude runs (``agents=``, its ``Agent`` tool) is a child run of the run
  (``bridges.helpers``), from its call to that call's result; its own tool calls go through the
  same ``PreToolUse`` hook, and what it spends is in its query's ``ResultMessage``;
* conversations: a conversation's turn leaves its CLI process running, and the next turn is
  sent into it (``bridges.claude_live``: the SDK's ``ClaudeSDKClient``), so what Claude started
  in the background — a subagent with ``background``, a shell command — goes on between the
  turns; a turn ends at the result of its own prompt, and a turn the session injects (a
  finished task's notification) is never its answer. A subagent the CLI runs in the
  background keeps its child run open until its task ends, its calls decided and recorded
  for that run (``claude_live.Background``). A run with no conversation, a sub-agent's
  run, and a turn with no live session (another server, a restart, one closed) run as below:
  one process for the query, resuming the conversation's session;
* pause: an approval or ``trellis.current().ask`` stops consuming the query (the CLI process
  ends — a conversation's live one stays, its turn ended, and the resumed attempt is
  sent into it). The session the CLI kept (``ResultMessage.session_id``) is in the run's
  journal, and the next attempt resumes it (``resume=``): Claude goes on from where it was —
  its built-in tools are not run again — and, told which call a person answered about, calls
  that tool
  again, which the journal answers (a call a question was asked in runs as it was, whatever
  words Claude makes it again with: ``Replay.made``). The session is kept in the run's journal
  too, as the SDK's
  ``SessionStore`` (``bridges.claude.JournalSessions``): the SDK writes it back for the CLI on
  any machine. A session the CLI cannot continue (options keeping their own store, a transcript
  it refuses) is a warning, and the query runs again from its prompt against the journal;
* models: with a route that has an Anthropic route (``Capability.ANTHROPIC_ROUTE``: Bifrost)
  serving the harness, the CLI's model calls go through the gateway's Anthropic route with the
  virtual key (:func:`routed_env`; the options' own ``ANTHROPIC_BASE_URL`` wins), carrying
  the deny-all MCP scope, the run's dimensions and its ``traceparent`` as
  ``ANTHROPIC_CUSTOM_HEADERS`` beside the options' own — the CLI reads them as it starts, once
  per attempt;
* framework options: ``ClaudeAgentOptions`` fields, set on the run's copy of the options
  before the harness's own changes. Merged with them: ``hooks`` (the governance and context
  hooks go after theirs), ``can_use_tool`` (asked when Claude Code would ask, after the hooks
  and governance) and ``mcp_servers`` (the harness's ``trellis`` server goes beside them; a
  server of that name is refused). Refused: the fields that choose the session a query
  continues, and the permission prompt tool (:data:`OWNED`).
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness import budget, skills
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    builtin,
    helper,
    unique,
)
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.bridges.claude import JournalSessions, session_store, with_context
from trellis.harness.bridges.claude_live import Background, LiveSession, LiveSessions, Turn
from trellis.harness.bridges.helpers import STOPPED, Helper, Helpers, Kind
from trellis.harness.gateways import NO_KEY, Capability, OpenAICompatible
from trellis.harness.governance.shell import SHELLS, Shell, moves
from trellis.harness.history import Called, Said, Speaker
from trellis.harness.journal import Pending, answers_read, content_key
from trellis.harness.prompts import CUSTOM_HEADERS, GatewayHeaders
from trellis.harness.runtime import Paused, Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.ask import ask_question
from trellis.harness.tools.base import UNDECLARED, SideEffects
from trellis.harness.tools.mcp import data_of_listed, side_effects_of

log = logging.getLogger("trellis.harness")

if TYPE_CHECKING:
    from claude_agent_sdk.types import PreToolUseHookSpecificOutput, SyncHookJSONOutput

#: Claude Code's built-in tools, and what each does as governance sees it (the tool catalog's
#: word overrides it): reads run; writes, ``Bash`` among them, run and are announced, Claude
#: Code's own permission check asking first where it would (a team that wants every shell
#: command approved gives ``Bash`` the risk ``irreversible``, or an ``approve_when``, in the
#: tool catalog); a subagent's
#: call (``Agent``, ``Task``) only hands work on — its own calls are governed one by one; a
#: skill (``Skill``) or a plan (``ExitPlanMode``) is read, a slash command (``SlashCommand``)
#: may do anything its command does: a write. Stopping a background task (``TaskStop``, named
#: ``KillShell`` before) is a write. Tools that only list or look up (``ListAgents``,
#: ``ListConnectors``, ``ReadNotifications``, ``ReportFindings``, ``SearchMcpRegistry``,
#: ``SuggestConnectors``: each read-only by the bundled CLI's own word) are reads. So are the
#: tools Claude keeps its own work with, which change nothing outside its session and which
#: the bundled CLI runs without asking its user: its task list (``TaskCreate``, ``TaskGet``,
#: ``TaskList``, ``TaskUpdate``: ``TodoWrite``'s successors), entering plan mode
#: (``EnterPlanMode``), and the tool it hands a structured answer to (``StructuredOutput``,
#: with ``output_format``: asked about, a run would pause on its own answer). Any other
#: tool the CLI asks about has said nothing of what it does: a person approves each call (its
#: risk in the catalog says more).
BUILTIN_SIDE_EFFECTS: Final[dict[str, SideEffects]] = {
    "Read": "read",
    "Glob": "read",
    "Grep": "read",
    "LS": "read",
    "WebFetch": "read",
    "WebSearch": "read",
    "TodoWrite": "read",
    "Agent": "read",
    "Task": "read",
    "Skill": "read",
    "ExitPlanMode": "read",
    "EnterPlanMode": "read",
    "TaskCreate": "read",
    "TaskGet": "read",
    "TaskList": "read",
    "TaskUpdate": "read",
    "StructuredOutput": "read",
    "AskUserQuestion": "read",
    "BashOutput": "read",
    "ListMcpResourcesTool": "read",
    "ReadMcpResourceTool": "read",
    "ListAgents": "read",
    "ListConnectors": "read",
    "ReadNotifications": "read",
    "ReportFindings": "read",
    "SearchMcpRegistry": "read",
    "SuggestConnectors": "read",
    "Write": "write",
    "Edit": "write",
    "MultiEdit": "write",
    "NotebookEdit": "write",
    "KillShell": "write",
    "TaskStop": "write",
    "SlashCommand": "write",
    "Bash": "write",
}
#: Claude Code's question to its user, asked as the run's (:func:`_answered`), and the mark it
#: ends a recommended option's label with.
QUESTION_TOOL: Final = "AskUserQuestion"
RECOMMENDED: Final = "(Recommended)"
#: What Claude Code is called when a person is asked about a call it asks about.
CLAUDE_CODE: Final = "Claude Code"
#: Claude Code's built-in tools whose work needs the CLI to go on running after the query: a
#: scheduled prompt (``CronCreate``, durable or not, fires only in a session that is running
#: then) and a wake-up (``ScheduleWakeup``). A query's own process ends with it; a
#: conversation's live session (``bridges.claude_live``) outlives the turn, but only until it
#: is idle, and the turn a schedule fires then has no run: its tool calls are refused and its
#: answer reaches nobody. So they stay disallowed (with the tools that list and delete the
#: schedules): offered, Claude promised a schedule that nothing keeps.
OUTLIVES_THE_QUERY: Final = ("CronCreate", "CronDelete", "CronList", "ScheduleWakeup")
#: Claude Code's own limit on an MCP tool's result (``MAX_MCP_OUTPUT_TOKENS``, 25,000 tokens by
#: default: about 50,000 characters of a CSV), raised so the harness keeps a large result first
#: (over ``results.MAX_CHARS``, 80,000 characters, read with ``read_file`` and ``query_result``):
#: a result the CLI saves to its own session's file is one neither reads, nor the journal holds
#: for a resume on another worker. A team's own limit (the options' ``env``, the process's) stays.
MCP_OUTPUT_ENV: Final = "MAX_MCP_OUTPUT_TOKENS"
MCP_OUTPUT_TOKENS: Final = 60_000
#: The most MCP tools a Claude agent was seen calling that join its tool set (the most recent).
SEEN_KEPT: Final = 1024
#: The most one message from the CLI may be (the SDK's ``max_buffer_size``: 1 MiB unless the
#: options say): a tool's result comes back from the CLI in one, and a run keeps results as large
#: as an agent-runs artifact (50 MiB: a journal over a checkpoint's size travels as one) —
#: escaped as JSON, twice that at most.
CLI_MESSAGE_BYTES: Final = 2 * 50 * 1024 * 1024
#: The hook event every tool call is governed at, and the ones that report it ran or failed.
#: Bifrost's route for Anthropic's API, under its root.
ANTHROPIC_ROUTE: Final = "anthropic"
TOOL_EVENT: Final = "PreToolUse"
RAN_EVENT: Final = "PostToolUse"
FAILED_EVENT: Final = "PostToolUseFailure"
#: How a message steered into the run reads to Claude, after a tool call's result.
STEERED_SAYS: Final = "The user adds, while you work: "
#: The key a query's model steps are watched for a stall under (``Runtime.model_started``).
STEP: Final = "claude_agent_sdk"
#: The result a query that reached its ``max_turns`` ends with (no answer).
MAX_TURNS: Final = "error_max_turns"
#: The ``stop_reason`` of a query's result whose last reply reached the model's output limit
#: (Claude Code first continues such a reply itself).
MAX_TOKENS: Final = "max_tokens"
#: When the SDK hands the session store a step: as soon as the CLI mirrors it.
SESSION_FLUSH: Final = "eager"
#: Claude Code's tool that runs a subagent (``agents=``): ``Agent``, named ``Task`` before.
SUBAGENT_TOOLS: Final = frozenset({"Agent", "Task"})
#: The subagent an ``Agent`` call runs when it names none.
GENERAL_PURPOSE: Final = "general-purpose"
#: Why a subagent whose call failed without saying why ended.
SUBAGENT_FAILED: Final = "the subagent's call failed"
#: The statuses a task ends with that say how (the SDK's ``TaskNotificationStatus`` and
#: ``TaskUpdatedStatus``): it completed, it failed; any other terminal one, it was stopped.
TASK_COMPLETED: Final = "completed"
TASK_FAILED: Final = "failed"
TASK_STOPPED: Final = "stopped"
#: The ``ClaudeAgentOptions`` fields the harness owns: framework options cannot set them.
OWNED: Final[dict[str, str]] = {
    "resume": "the harness resumes the run's own session after a pause",
    "continue_conversation": "the harness resumes the run's own session after a pause",
    "permission_prompt_tool_name": "the harness decides Claude's tool permissions with "
    "can_use_tool (give your own check as can_use_tool)",
}
#: Why a conversation's live session is closed before a query that runs in a process of its
#: own (two processes do not hold one session).
OWN_PROCESS: Final = "this query runs in a process of its own"
#: What a resumed session is told: the run goes on, and the call it paused on is made again.
RESUMED: Final = (
    "This task was paused (a person was asked, or the process stopped) and goes on now. If "
    "your last tool call has no result yet, call that tool again with the same arguments: "
    "it runs once. Then carry on with the task."
)
#: What a session resumed after a person answered about one tool call — its approval, or a
#: question it asked while it ran — is told: that call, by the name Claude knows it by. The
#: session holds the call's result as "waiting for a person's approval", which :data:`RESUMED`'s
#: "no result yet" does not describe: a model told only that took the waiting call for a
#: failure and tried other tools instead (or asked the person again itself).
RESUMED_CALL: Final = (
    "This task was paused at your call to {tool}: it was waiting for a person's approval, and "
    "the person has answered. Call {tool} again now with the same arguments: it runs once and "
    "returns its result. Then carry on with the task."
)
#: ... and after a person answered a question that call asked while it ran
#: (``Pending.inside``): named with its arguments, which a session whose transcript ends
#: before the call does not hold (told to repeat a call it had no record of, Claude made none).
#: Whatever arguments it is made again with, it runs as it was (``Replay.made``).
RESUMED_ASKED: Final = (
    "This task was paused at your call to {tool} with the arguments {args}: it was waiting for "
    "a person's answer, and the person has answered. Call {tool} again now with those "
    "arguments: it runs once and returns its result. Then carry on with the task."
)
#: ... and after a person denied that call: it does not run, so it returns no result to wait
#: for, and the session is told so instead of being told to expect one.
RESUMED_DENIED: Final = (
    "This task was paused at your call to {tool}: it was waiting for a person's approval, and "
    "the person denied it. That call will not run; do not make it again. Carry on with the "
    "task without it."
)


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeInput:
    prompt: str
    context: str | None
    #: the tool call a person answered about (``Pending.call``), when the run resumes from
    #: that pause
    answered: ToolCall | None = None
    #: the person denied that call (``REJECT``): it does not run
    denied: bool = False
    #: the person answered a question that call asked while it ran (not its approval)
    asked: bool = False
    #: the run paused before its prompt was sent — on the question of a background
    #: subagent's run of its conversation (``claude_live.LiveSession.asks``): it is sent now
    unsent: bool = False


class ClaudeRunError(RuntimeError):
    """The CLI reported the run as failed."""


class ClaudeAdapter:
    name: ClassVar[str] = "claude_agent_sdk"
    tool_format: ToolFormat = "claude"
    fixed_tools: bool = False
    narrows: Narrowing = "run"
    #: a query takes one prompt, not a conversation: memory's window is in the context's text
    replays_history = False
    #: with memory off, the conversation's earlier turns are in the context's text too
    history_in: HistoryIn = "context"
    #: its session (the journal's) is continued by the conversation's next turn
    sessions = True

    def __init__(self) -> None:
        #: the tools the options declare (:meth:`native`), once read
        self._static: list[ToolSpec] | None = None
        #: the MCP tools the agent was seen calling, the most recent kept
        self.seen: LRUSet[str] = LRUSet(maxsize=SEEN_KEPT)
        #: each conversation's CLI process, kept between its turns (``bridges.claude_live``)
        self.live = LiveSessions()
        #: what each session cost so far, as its last result charged here carried it (the
        #: most recent): a process that resumes one reports the session's cost, not its own
        self.costs: LRU[str, float] = LRU()

    async def aclose(self) -> None:
        """The agent is closing: every conversation's live session ends."""
        await self.live.aclose()

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        return None

    def steers(self, target: Any) -> bool:
        """Every query: a steered message is added after a tool call's result (a
        ``PostToolUse`` hook's ``additionalContext``, :func:`_with_governance`); a query that
        calls no tool takes none."""
        return True

    def asks(self, target: Any) -> bool:
        """Claude Code's own :data:`QUESTION_TOOL`, asked as the run's (:func:`_answered`)."""
        return True

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        prompt = input if isinstance(input, str) else json.dumps(input, default=str)
        return ClaudeInput(prompt=prompt, context=context)

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return [m async for m in self._helped(target, native_input, run)]

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from claude_agent_sdk import AssistantMessage, TextBlock

        messages: list[Any] = []
        async for message in self._helped(target, native_input, run):
            messages.append(message)
            # the main agent's text: a subagent's (forwarded naming its ``Agent`` call) is
            # its own run's, as in the answer and the transcript
            if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield block.text
        yield Output(messages)

    def extract(self, target: Any, output: Any) -> Extracted:
        from claude_agent_sdk import ResultMessage

        transcript = _transcript(output)
        answer: Any = None
        for message in output:
            # the result of the turn's own prompt: one of a turn the session injected (a
            # finished background task's notification) is not the user's answer
            if isinstance(message, ResultMessage) and _own(message):
                if message.is_error:
                    raise ClaudeRunError("; ".join(message.errors or []) or message.subtype)
                answer = (
                    message.structured_output
                    if message.structured_output is not None
                    else message.result
                )
        return Extracted(answer=answer, transcript=transcript)

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        from trellis.harness.subagents import asked_by

        if (asked_by(pending.interrupt) or {}).get("background"):
            return dataclasses.replace(native_input, unsent=True)
        return dataclasses.replace(
            native_input,
            answered=pending.call,
            denied=resolution.decision is InterruptDecision.REJECT,
            asked=pending.interrupt.tool_call is None,
        )

    def check_options(self, options: Mapping[str, Any]) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        from trellis.harness.tools.convert.claude import SERVER

        fields = [f.name for f in dataclasses.fields(ClaudeAgentOptions)]
        unknown = [key for key in options if key not in fields]
        if unknown:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, unknown))}: no ClaudeAgentOptions field"
            )
        owned = [f"{key!r} ({OWNED[key]})" for key in options if key in OWNED]
        if owned:
            raise ConfigurationError(f"framework_options {'; '.join(owned)}")
        servers = options.get("mcp_servers")
        if isinstance(servers, Mapping) and SERVER in servers:
            raise ConfigurationError(
                f"framework_options' mcp_servers names {SERVER!r}: the harness's own server "
                "(its tools) has that name"
            )

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        """Where the CLI works: the options' ``cwd`` (the run's own over the agent's), else
        this process's working directory."""
        return Path(options.get("cwd") or target.cwd or os.getcwd())

    async def native(self, target: Any) -> list[ToolSpec]:
        """Claude Code's built-in tools the options leave it (``tools``, else the ones
        :data:`BUILTIN_SIDE_EFFECTS` knows, less ``disallowed_tools``) and the MCP tools of its
        own servers: those an in-process server (``create_sdk_mcp_server``) lists, those
        ``allowed_tools`` names (``mcp__<server>__<tool>``), and those the agent was seen
        calling (its ``PostToolUse``: another server's tools are listed only by the CLI)."""
        if self._static is None:
            self._static = await _declared(target)
        return unique([*self._static, *(native_spec(name) for name in self.seen)])

    def observed(self, name: str) -> None:
        """A tool Claude ran itself (``PostToolUse``): an MCP server's joins the tool set."""
        if name.startswith("mcp__"):
            self.seen.add(name)

    # ------------------------------------------------------------------ internals
    async def _helped(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages (:meth:`_messages`), each subagent Claude runs a helper of the
        run (``bridges.helpers``) from its ``Agent`` call to that call's result."""
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        # a background subagent's run has no query of its own: its task works in a session
        found = self.live.attended.get(run.runtime.run_id)
        if found is not None:  # the calls its subagent makes are this run's own
            helpers.attribute(found.task_id, None)
        source = (
            self._messages(target, native_input, run)
            if found is None
            else self._attended(target, run, found)
        )
        try:
            async with aclosing(source) as messages:
                async for message in messages:
                    await _subagents(helpers, message)
                    yield message
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)

    async def _attended(
        self, target: Any, run: Invocation, found: Background
    ) -> AsyncGenerator[Any, None]:
        """One attempt of a background subagent's run (``claude_live.Background``): the
        subagent's messages while its task works in its conversation's live session, then
        the result made of the task's end. The attempt's own hooks, permission check and
        harness tools — made as a query's are — decide and record the calls the CLI makes
        for the subagent, and its steps are watched for a stall as a query's."""
        runtime = run.runtime
        target = dataclasses.replace(target, **found.given) if found.given else target
        steps = Steps(runtime, self)
        options = _options(target, None, run, None, steps, kept=True)
        steps.waiting()
        try:
            async with aclosing(found.served(_turn(run, options))) as messages:
                async for message in messages:
                    steps.waiting()
                    yield message
        finally:
            steps.closed = True
            runtime.model_ended(STEP)

    async def _messages(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages (:meth:`_attempted`) — and, when it reached its turn limit
        (``error_max_turns``: no answer), one last answer instead of that error: its session
        resumed (Claude's own ``resume``), told the limit is reached for this turn
        (``budget.TURN_LIMIT``: the session keeps it, and the next turn has its tools again),
        for one turn with no tool, as a ``ReAct`` run at ``max_steps`` answers. A last query
        that gives no answer leaves the error."""
        from claude_agent_sdk import ResultError, ResultMessage

        runtime = run.runtime
        limited: Any = None
        try:
            async with aclosing(self._attempted(target, native_input, run)) as messages:
                async for message in messages:
                    if isinstance(message, ResultMessage) and message.subtype == MAX_TURNS:
                        limited = message
                        continue
                    if isinstance(message, ResultMessage):
                        # past Claude Code's own recovery: what a run with no answer says
                        runtime.cut_off = message.stop_reason == MAX_TOKENS
                    yield message
        except ResultError:
            # the CLI exits with an error after its error result (the SDK raises it then): at
            # the turn limit that is the end the result already said
            if limited is None:
                raise
        if limited is None:
            return
        # the result names the session it ran in (the journal's): resumed for the last answer
        session = runtime.replay.journal.session
        limit = int(runtime.framework_options.get("max_turns") or target.max_turns or 0)
        told = dataclasses.replace(native_input, prompt=budget.TURN_LIMIT.format(limit=limit))
        last: list[Any] = []
        async with aclosing(self._query(target, told, run, session, last=True)) as messages:
            async for message in messages:
                last.append(message)
        answered = [m for m in last if isinstance(m, ResultMessage) and not m.is_error]
        if not answered or not (answered[-1].result or "").strip():
            yield limited
            return
        stopped = f"stopped: the turn limit ({limit}) was reached"
        log.warning("run %s %s", runtime.run_id, stopped)
        runtime.events.warning("step_limit", stopped)
        for message in last:
            yield message

    async def _attempted(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages — the session an earlier attempt kept resumed, when there is
        one the CLI still holds; else from the prompt."""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        runtime = run.runtime
        journal = runtime.replay.journal
        resumed, answered = journal.session, False
        # closed as it is left (``aclosing``): its CLI process ends here, in this loop
        async with aclosing(self._query(target, native_input, run, resumed)) as messages:
            async for message in messages:
                if (
                    resumed is not None
                    and not answered
                    and isinstance(message, ResultMessage)
                    and message.is_error
                ):
                    # the CLI could not continue the session: the query again, from its prompt
                    why = "; ".join(message.errors or []) or message.subtype
                    runtime.events.warning(
                        "claude_session", f"session {resumed} was not resumed ({why}): run again"
                    )
                    journal.session = None
                    break
                answered = answered or isinstance(message, AssistantMessage)
                yield message
            else:
                return
        async with aclosing(self._query(target, native_input, run, None)) as again:
            async for message in again:
                yield message

    async def _query(
        self,
        target: Any,
        native_input: Any,
        run: Invocation,
        session: str | None,
        *,
        last: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """One query (``session``: the one it resumes); the session it runs in kept in the
        journal as it goes. ``last``: the one last answer at the turn limit — its prompt as
        given, one turn, no tool (no built-in, no MCP server), no context added again."""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        runtime = run.runtime
        given = runtime.framework_options
        target = dataclasses.replace(target, **given) if given else target
        # a skill source's plugin written for this run (its pinned skills), removed after it
        plugins, written = await skills.for_run(target.plugins)
        if written:
            target = dataclasses.replace(target, plugins=plugins)
        steps = Steps(runtime, self)
        # a conversation's turn is served by its live session (``bridges.claude_live``); a run
        # with no conversation of its own, a sub-agent's, the one last answer, and options
        # whose parts are a process's alone (a plugin written for the run; an in-process MCP
        # server of the team's, whose tools run in the context of the process's first turn;
        # a budget of the options' own, which the CLI counts for a process)
        # run in a process of their own, as ever
        key = runtime.conversation if _converses(runtime) else None
        kept = key is not None and not last and not written and not _own_process(target)
        context = None if last else native_input.context
        options = _options(target, context, run, session, steps, kept=kept)
        if last:
            options = dataclasses.replace(options, tools=[], mcp_servers={}, max_turns=1)
        # a first attempt resumes only the conversation's session (``Runtime.continued``):
        # the turn's own prompt; a later one, the run's own session after a pause
        first = last or session is None or runtime.attempt == 1 or native_input.unsent
        prompt = native_input.prompt if first else _resumed(native_input, run)
        steps.waiting()
        spending = Spending(runtime, options, self.costs, session)
        live: LiveSession | None = None
        try:
            if kept:
                assert key is not None
                live = await self._live(key, options, run, session, unsent=first)
            elif key is not None:
                await self.live.close(key, OWN_PROCESS)
                _said(self.live, runtime)
            # closed as it is left (``aclosing``): a process of its own ends here, in this
            # loop; a live session's turn is interrupted when left before its result
            source = (
                _queried(prompt, options)
                if live is None
                else live.served(_turn(run, options), prompt)
            )
            async with aclosing(source) as messages:
                async for message in messages:
                    steps.waiting()  # the next message is the next step's
                    found = getattr(message, "session_id", None)
                    if isinstance(found, str):
                        runtime.replay.journal.session = found
                    if isinstance(message, ResultMessage):
                        spending.ended(message, live)
                    elif isinstance(message, AssistantMessage):
                        spending.replied(message)
                    yield message
                    if runtime.pending is not None:
                        # the run paused: stop here; a process of its own goes with it
                        return
        finally:
            steps.closed = True
            runtime.model_ended(STEP)
            skills.removed(written)
            # it ended with no result (paused, stopped), or its subagents went on after its
            # last: what the replies no result counted said — in a live session, the
            # running totals of the next result charged hold it
            if live is None:
                spending.left()

    async def _live(
        self, key: str, options: Any, run: Invocation, session: str | None, *, unsent: bool
    ) -> LiveSession | None:
        """The conversation's live session for this query: the one kept, when it was started
        with these options, holds the session the query continues and owes no earlier turn's
        result; else a new one, started as a query's own process is (``resume=``). ``None``:
        there is no room for one (``claude_live.KEPT``). ``unsent``: the turn's own prompt is
        still to send — what the session's background runs wait to ask a person is asked
        first, in this run (``LiveSession.asks``)."""
        runtime = run.runtime
        shape = _shape(options)
        found = self.live.get(key)
        if found is not None:
            why = None
            if found.shape != shape:
                why = "this turn's options are not the ones it was started with"
            elif found.session_id != session:
                why = "this turn does not continue the session it holds"
            elif not await found.settled():
                why = "an earlier turn's result never came"
            if why is not None:
                await self.live.close(key, why)
                found = None
        if found is None:
            transcripts = runtime.replay.journal.transcripts
            found = await self.live.open(key, shape, options, transcripts)
            if found is not None and session is not None:
                # a process that resumes a session reports the session's cost: what its
                # first turn is charged counts from what the session cost before
                found.billed = self.costs.get(session) or 0.0
        _said(self.live, runtime)
        if found is not None and unsent:
            await found.asks(runtime)
        return found


def _converses(runtime: Runtime) -> bool:
    """Whether the run is a turn of a conversation (a thread of its own, not a sub-agent's
    run): what a later turn follows."""
    return runtime.parent is None and runtime.conversation != runtime.run_id


def _own_process(options: Any) -> bool:
    """Whether the options need a process for every query: they carry an in-process MCP
    server of the team's own (its tools run in the context of the process's first turn), or
    a budget of their own (``max_budget_usd`` caps what a process spends, and the SDK cannot
    give a running one its cap back: the cap a team set for a query stays a query's)."""
    from trellis.harness.tools.convert.claude import SERVER

    return options.max_budget_usd is not None or any(
        name != SERVER and isinstance(config, Mapping) and config.get("type") == "sdk"
        for name, config in configured_servers(options.mcp_servers).items()
    )


def _own(result: Any) -> bool:
    """Whether a ``ResultMessage`` ends a turn its caller submitted (the SDK's
    ``MessageOrigin``: none, or ``human``) — not one the session injected by itself."""
    origin = getattr(result, "origin", None)
    return origin is None or origin.get("kind") == "human"


def _said(live: LiveSessions, runtime: Runtime) -> None:
    """The run is told, once, of the background work that ended with its conversation's
    live session."""
    said = live.told(runtime.conversation)
    if said is not None:
        runtime.events.warning("claude_session", said)


def _turn(run: Invocation, options: Any) -> Turn:
    """The query as a live session serves it: its own options, and this context."""
    from trellis.harness.tools.convert.claude import SERVER

    servers = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
    return Turn(run.runtime, options, servers.get(SERVER), contextvars.copy_context())


def _shape(options: Any) -> tuple[Any, ...]:
    """What a process started with ``options`` has for good, and a later turn must bring the
    same to be served by it: the options less what each turn brings of its own (its hooks'
    and its permission check's code, its harness tools' code, its journal, the session it
    resumes, its run's gateway headers), the hooks' places, and what its ``trellis`` server
    lists."""
    from trellis.harness.tools.convert import claude as convert

    servers = options.mcp_servers
    store = options.session_store
    journaled = isinstance(store, JournalSessions)
    fixed = dataclasses.replace(
        options,
        hooks=None,
        can_use_tool=None,
        resume=None,
        session_store=None if journaled else store,
        mcp_servers={n: c for n, c in servers.items() if n != convert.SERVER}
        if isinstance(servers, dict)
        else servers,
        env={k: v for k, v in options.env.items() if k != CUSTOM_HEADERS},
    )
    hooks = tuple(
        (event, tuple((m.matcher, len(m.hooks), m.timeout) for m in matchers))
        for event, matchers in sorted((options.hooks or {}).items())
    )
    own = servers.get(convert.SERVER) if isinstance(servers, dict) else None
    return (fixed, hooks, None if own is None else convert.listing(own), journaled)


def _queried(prompt: Any, options: Any) -> AsyncGenerator[Any, None]:
    """The query's messages: what ``claude_agent_sdk.query`` runs, for its caller to close.
    ``query`` iterates it with a bare ``async for``, so a caller that leaves early (a paused
    run) leaves it to be collected — after the attempt's task, where its cleanup (the CLI
    process's end) can no longer run, and the process outlives the run."""
    from claude_agent_sdk._internal.client import InternalClient

    messages = InternalClient().process_query(prompt=prompt, options=options)
    return cast("AsyncGenerator[Any, None]", messages)


async def _briefed(
    runtime: Runtime, call_id: str, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """A subagent's call (:data:`SUBAGENT_TOOLS`) as it runs: started as a helper of the run
    (``bridges.helpers``), its ``prompt`` given the memory context for its task before it."""
    helpers = runtime.helpers
    assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
    task = str(args.get("prompt") or args.get("description") or "")
    kind, called = Kind.SUBAGENT, str(args.get("subagent_type") or GENERAL_PURPOSE)
    helper = await helpers.start(call_id, kind=kind, name=called, task=task)
    return {**args, "prompt": helpers.briefed(helper, task)} if "prompt" in args else args


async def _subagents(helpers: Helpers, message: Any) -> None:
    """A subagent's start (the main agent's ``Agent`` call: its ``subagent_type`` and
    ``prompt``; started already when its ``PreToolUse`` came first: :func:`_briefed`) or end
    (that call's result) in ``message``; and whose task it is (``task_started``: the task's
    id is the ``agent_id`` Claude Code names the subagent by in its calls' hooks). A
    subagent Claude Code runs in the background (``is_backgrounded``: its call's result is
    its launch) ends when its task does (``task_notification``; a ``task_updated`` to a
    terminal status may be all that comes). In a conversation's live session such a
    subagent's run is the session's to keep (``claude_live.Background``): it is no longer
    one of these helpers."""
    from claude_agent_sdk import (
        AssistantMessage,
        SystemMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    if isinstance(message, SystemMessage):
        await _tasks(helpers, message)
    elif isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
        for call in message.content:
            if isinstance(call, ToolUseBlock) and call.name in SUBAGENT_TOOLS:
                await helpers.start(
                    call.id,
                    kind=Kind.SUBAGENT,
                    name=str(call.input.get("subagent_type") or GENERAL_PURPOSE),
                    task=str(call.input.get("prompt") or call.input.get("description") or ""),
                )
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for result in message.content:
            if isinstance(result, ToolResultBlock):
                text = _result_text(result.content)
                if result.is_error:
                    await helpers.end(result.tool_use_id, error=text or SUBAGENT_FAILED)
                elif not helpers.waits(result.tool_use_id):  # not a background one's launch
                    await helpers.end(result.tool_use_id, output=text)


async def _tasks(helpers: Helpers, message: Any) -> None:
    """A task's start or end in a system ``message``: whose helper its worker works for,
    and how a background helper's work ended."""
    from claude_agent_sdk import (
        TERMINAL_TASK_STATUSES,
        TaskNotificationMessage,
        TaskStartedMessage,
        TaskUpdatedMessage,
    )

    if isinstance(message, TaskStartedMessage):
        if message.tool_use_id:
            background = bool(message.data.get("is_backgrounded"))
            helpers.attribute(message.task_id, message.tool_use_id, background=background)
    elif isinstance(message, TaskNotificationMessage):
        failed = message.status != TASK_COMPLETED
        said = message.summary if message.status != TASK_STOPPED else STOPPED
        await helpers.settled(message.task_id, failed=failed, said=said)
    elif isinstance(message, TaskUpdatedMessage) and message.status in TERMINAL_TASK_STATUSES:
        failed = message.status != TASK_COMPLETED
        said = "" if message.status in (TASK_COMPLETED, TASK_FAILED) else STOPPED
        await helpers.settled(message.task_id, failed=failed, said=said, final=False)


def _transcript(messages: Sequence[Any]) -> list[Said]:
    """The query's own messages as said (its subagents' are inside their ``Agent`` call): each
    text, each call — a harness tool by its own name, not Claude's ``mcp__trellis__`` one — and
    each call's result, so what memory keeps of the turn holds its steps."""
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from trellis.harness.tools.convert.claude import SERVER

    harness = f"mcp__{SERVER}__"
    out: list[Said] = []
    for message in messages:
        if getattr(message, "parent_tool_use_id", None) is not None:
            continue
        if isinstance(message, AssistantMessage):
            text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
            calls = tuple(
                Called(b.id, b.name.removeprefix(harness), b.input)
                for b in message.content
                if isinstance(b, ToolUseBlock)
            )
            if text or calls:
                out.append(Said(Speaker.ASSISTANT, text, tool_calls=calls))
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            out.extend(
                Said(Speaker.TOOL, _result_text(b.content), tool_call_id=b.tool_use_id)
                for b in message.content
                if isinstance(b, ToolResultBlock)
            )
    return out


def _result_text(content: Any) -> str:
    """A tool result's text (a string, its text blocks, or the text a built-in's structured
    response holds)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        found = content.get("content", content.get("output", content.get("stdout")))
        return _result_text(found) if found is not None else json.dumps(content, default=str)
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


class Spending:
    """What one query spends, charged to its run as its results come. A result's usage is
    the main agent's own for that turn, and its cost its session's running total — the
    subagents' included, and what the session cost before the process that resumed it (the
    bundled CLI's results, driven against a scripted model). So each reply's usage is kept
    by its message id until a result counts it — the main agent's are in the result's
    usage, its subagents' are counted with it — and a result is charged what the session's
    cost grew by since the last one charged: this query's (it gives one for the turn and
    one for every turn the session ran for a background task), an earlier query's of the
    run (the journal's ``session_cost``), or the conversation's last turn's in this process
    (``ClaudeAdapter.costs``)."""

    __slots__ = ("billed", "costs", "helped", "journal", "options", "own", "runtime")

    def __init__(
        self, runtime: Runtime, options: Any, costs: LRU[str, float], session: str | None
    ) -> None:
        self.runtime = runtime
        self.options = options
        self.costs = costs
        self.journal = runtime.replay.journal
        self.own: dict[str, dict[str, Any]] = {}
        self.helped: dict[str, dict[str, Any]] = {}
        if session is None:  # a new session: nothing was spent in it
            self.journal.session_cost = None
        known = self.journal.session_cost
        if known is None and session is not None:
            known = costs.get(session)
        #: the session's running cost, as the last result charged carried it
        self.billed = known or 0.0

    def replied(self, message: Any) -> None:
        """An assistant message: its reply's usage (one reply comes as a message a block)."""
        if message.message_id and message.usage:
            whose = self.own if message.parent_tool_use_id is None else self.helped
            whose[message.message_id] = message.usage

    def ended(self, result: Any, live: LiveSession | None) -> None:
        """A turn's result. One of a ``live`` session comes with what its turn is charged
        (``claude_live.LiveSession.charged``, which keeps the running cost)."""
        cost = result.total_cost_usd
        if live is not None:
            self.billed = live.billed
        elif cost is not None:
            cost, self.billed = max(0.0, cost - self.billed), cost
        self.journal.session_cost = self.billed
        if result.session_id:
            self.costs[result.session_id] = self.billed
        # a turn no reply of the model's is in — one the CLI answered itself, a slash
        # command (``/compact``: ``num_turns`` 0) — is no model call: the run, its answer
        # empty, is one no model replied in (``pipeline._unanswered``)
        replied = bool(self.own or self.helped or result.num_turns)
        _spent(
            self.runtime,
            self.options,
            result,
            self.helped.values(),
            cost=cost,
            calls=int(replied),
        )
        self.own.clear()
        self.helped.clear()

    def left(self) -> None:
        """The query ended: what the replies no result counted said."""
        if self.own or self.helped:
            _spent(self.runtime, self.options, None, [*self.own.values(), *self.helped.values()])


def _spent(
    runtime: Runtime,
    options: Any,
    result: Any,
    replies: Iterable[dict[str, Any]] = (),
    *,
    cost: float | None = None,
    calls: int = 1,
) -> None:
    """What a query spent, on the run's budget (``calls``: the model calls it counts as):
    its result's usage (the main agent's), the
    usage of its subagents' ``replies`` since the result before, and ``cost`` (what the
    result's ``total_cost_usd`` — the subagents' included — is charged for). With no result
    (a query that paused or was stopped), the usage its ``replies`` carried, which the CLI
    gives no cost for."""
    usages = [*([] if result is None else [result.usage or {}]), *replies]

    def total(key: str) -> int:
        return sum(int(u.get(key) or 0) for u in usages)

    cached, written = total("cache_read_input_tokens"), total("cache_creation_input_tokens")
    budget.account(
        runtime,
        model=options.model,
        input_tokens=total("input_tokens") + cached + written,
        output_tokens=total("output_tokens"),
        cached=cached,
        cost=cost,
        **({} if calls == 1 else {"calls": calls}),
    )


def _resumed(native_input: ClaudeInput, run: Invocation) -> str:
    """What the resumed session is told: the call a person answered about, named as Claude
    calls it (a harness tool through the ``trellis`` server, a built-in by its own name)."""
    from trellis.harness.tools.convert.claude import SERVER

    call = native_input.answered
    if call is None:
        return RESUMED
    harness = any(t.name == call.tool for t in run.tools)
    name = f"mcp__{SERVER}__{call.tool}" if harness else call.tool
    return resumed_call(call, name, denied=native_input.denied, asked=native_input.asked)


def resumed_call(call: ToolCall, name: str, *, denied: bool, asked: bool) -> str:
    """What a session resumed after a person answered about ``call`` is told (``name``: the
    tool as its agent calls it): the call was denied, or asked them a question, or was
    approved."""
    told = RESUMED_DENIED if denied else RESUMED_ASKED if asked else RESUMED_CALL
    return told.format(tool=name, args=json.dumps(call.args, default=str))


class Steps:
    """One query's steps as the stall watchdog sees them (``Runtime.model_started``): Claude is
    waited on from one message to the next, except while a tool call runs (from its
    ``PreToolUse`` to its ``PostToolUse``: a tool's own time is not a stalled model); and the
    calls under way, by Claude's id, with when each began (what ``PostToolUse`` records), and
    the subagent each call a subagent makes is the work of (``helped``)."""

    __slots__ = ("adapter", "answered", "calls", "closed", "helped", "runtime")

    def __init__(self, runtime: Runtime, adapter: ClaudeAdapter) -> None:
        self.runtime = runtime
        #: the calls a person answered about before they ran (governance's question, a
        #: hook's ``Ask``: ``PreToolUse``), by Claude's id: Claude Code's own question about
        #: one (``can_use_tool``) is not asked too — one call, one question
        self.answered: set[str] = set()
        #: the query ended: a call of it that ends later (a live session's process goes on)
        #: watches nothing
        self.closed = False
        #: whose query it is: the MCP tools it is seen calling join its tool set
        self.adapter = adapter
        #: the tool calls under way: their name, arguments and start (``time.time()``)
        self.calls: dict[str, tuple[str, dict[str, Any], float]] = {}
        #: the calls a subagent makes, by Claude's id: the helper whose work each is — the
        #: CLI names the subagent in the ``PreToolUse`` hook by its task's id (``agent_id``),
        #: and its permission request (``can_use_tool``) for the same call is the same helper's
        self.helped: dict[str, Helper] = {}

    async def helper_of(self, call_id: str, given: Mapping[str, Any]) -> Helper | None:
        """The helper whose work the call ``call_id`` is, from the ``PreToolUse`` hook's input
        (``given``: the CLI names a subagent's call by its task's id, ``agent_id``, which the
        task's start tied to the ``Agent`` call that started it: ``Helpers.of``, as the call's
        record finds it — two subagents of one type are each their own), kept for the call's
        permission request; ``None``: the run's own call."""
        helpers = self.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        who, kind = given.get("agent_id"), str(given.get("agent_type") or "")
        working = None if who is None else await helpers.of(str(who), kind)
        if working is not None:
            self.helped[call_id] = working
        return working

    def waiting(self) -> None:
        """Claude is to answer now (unless a call runs): watched for a stall."""
        if not self.calls and not self.closed:
            self.runtime.model_started(STEP)

    def started(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.calls[call_id] = (name, args, time.time())
        self.runtime.model_ended(STEP)

    def ended(self, call_id: str) -> tuple[str, dict[str, Any], float] | None:
        self.helped.pop(call_id, None)
        found = self.calls.pop(call_id, None)
        self.waiting()
        return found


def _options(
    options: Any,
    context: str | None,
    run: Invocation,
    session: str | None,
    steps: Steps,
    *,
    kept: bool = False,
) -> Any:
    """The options one query runs with. ``kept``: a conversation's turn, which a live session
    may serve — its context hook is there whether or not this turn has a context, so every
    turn's hooks have the same places."""
    from trellis.harness.tools.convert import claude as convert

    if options.permission_prompt_tool_name:
        raise ConfigurationError(
            "the harness decides Claude's tool calls with a PreToolUse hook: give your own "
            "permission check as can_use_tool (the hook asks it after governance), not "
            "permission_prompt_tool_name"
        )
    runtime = run.runtime
    env = dict(routed_env(options.env, runtime) or options.env)
    if MCP_OUTPUT_ENV not in env and MCP_OUTPUT_ENV not in os.environ:
        env[MCP_OUTPUT_ENV] = str(MCP_OUTPUT_TOKENS)
    shell = _shell(options, env)
    hooks = _with_governance(options.hooks, steps, shell)
    changes: dict[str, Any] = {
        "hooks": with_context(hooks, context or "") if context or kept else hooks,
        "system_prompt": _told(options.system_prompt),
        # what Claude Code asks its user about, the harness answers (the team's own first)
        "can_use_tool": _asked(options.can_use_tool, steps, shell),
        "max_buffer_size": options.max_buffer_size or CLI_MESSAGE_BYTES,
        "env": env,
    }
    store = session_store(options, runtime.replay.journal, runtime.agent_id)
    if store is not None:
        # each step mirrored as it is taken: the journal saved with a pause holds it
        changes["session_store"] = store
        changes["session_store_flush"] = SESSION_FLUSH
    if run.native_tools is not None:
        servers = configured_servers(options.mcp_servers)
        changes["mcp_servers"] = {**servers, convert.SERVER: run.native_tools}
    changes["disallowed_tools"] = _disallowed(runtime, options.disallowed_tools)
    if session is not None:
        changes["resume"] = session
        if not options.fork_session:
            # the session the options named to start (``session_id``) is the one resumed:
            # Claude Code takes both only to fork
            changes["session_id"] = None
    return dataclasses.replace(options, **changes)


def _told(given: Any) -> Any:
    """The options' system prompt with what Claude is told of its folder's ``outputs/``
    (``sandbox.DELIVERS_HERE``: the user's), after it — appended to a preset (``append``), to
    a custom prompt or text, or as the prompt when there is none (the SDK's empty one). A
    prompt read from a file is left as it is: the CLI takes no text beside it."""
    from trellis.harness.sandbox import DELIVERS_HERE

    def after(text: str | None) -> str:
        return f"{text}\n\n{DELIVERS_HERE}" if text else DELIVERS_HERE

    if given is None or isinstance(given, str):
        return after(given)
    kind = given.get("type")
    if kind == "preset":
        return {**given, "append": after(given.get("append"))}
    if kind == "custom":
        return {**given, "prompt": after(given.get("prompt"))}
    return given


def routed_env(env: Mapping[str, str], runtime: Runtime) -> dict[str, str] | None:
    """The CLI's environment with its model calls routed through the gateway when the route
    serving the harness has an Anthropic route (``Capability.ANTHROPIC_ROUTE``: Bifrost;
    ``None``: as it is): the gateway's Anthropic route (``ANTHROPIC_BASE_URL``, the
    virtual key as ``ANTHROPIC_API_KEY``) — unless the options' own ``env`` names a base URL
    (code over the environment) — and, on Bifrost's route, its headers as
    ``ANTHROPIC_CUSTOM_HEADERS`` lines beside the options' own: the deny-all MCP scope, the
    run's dimensions and its ``traceparent`` (read as the CLI starts)."""
    gateway = runtime.agent.harness.model_gateway
    if not gateway.has(Capability.ANTHROPIC_ROUTE):
        return None
    gateway = cast("OpenAICompatible", gateway)
    given = dict(env)
    route = f"{gateway.origin}/{ANTHROPIC_ROUTE}"
    if "ANTHROPIC_BASE_URL" not in given:
        given["ANTHROPIC_BASE_URL"] = route
        given["ANTHROPIC_API_KEY"] = gateway.key or NO_KEY
    if not given["ANTHROPIC_BASE_URL"].startswith(gateway.origin):
        return given if given != env else None  # their own endpoint: no Bifrost headers
    ours = GatewayHeaders().now()
    lines = (given.get(CUSTOM_HEADERS) or "").splitlines()
    kept = [line for line in lines if line.partition(":")[0].strip().lower() not in ours]
    given[CUSTOM_HEADERS] = "\n".join([*kept, *(f"{k}: {v}" for k, v in ours.items())])
    return given


def _disallowed(runtime: Runtime, denied: Sequence[str] | None) -> list[str]:
    """The options' ``disallowed_tools`` with the framework's own tools the run does not offer:
    those whose work outlives the query (:data:`OUTLIVES_THE_QUERY`), and those the tool hints
    narrow away (Claude's built-ins and MCP tools too, for the whole query)."""
    given = list(denied or ())
    narrowed = (
        [n for n in runtime.native if not runtime.offers(n)] if runtime.offered is not None else []
    )
    hidden = [n for n in (*OUTLIVES_THE_QUERY, *narrowed) if n not in given]
    return [*given, *dict.fromkeys(hidden)]


def _with_governance(hooks: Mapping[str, Any] | None, steps: Steps, shell: Shell) -> dict[str, Any]:
    """The options' hooks with the governance hook (:func:`_governance`) and the record hooks
    (:func:`_recorded`) after theirs."""
    given = dict(hooks or {})
    given[TOOL_EVENT] = [*given.get(TOOL_EVENT, ()), _governance(steps)]
    recorded = _recorded(steps, shell)
    for event in (RAN_EVENT, FAILED_EVENT):
        given[event] = [*given.get(event, ()), recorded]
    given[RAN_EVENT] = [*given[RAN_EVENT], _steering()]
    return given


def _steering() -> Any:
    """The ``PostToolUse`` hook that gives Claude the messages steered into the run since its
    last tool call (``Runtime.steered``), as the context a hook adds after the call's result."""
    from claude_agent_sdk import HookMatcher

    async def steered(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        runtime = current()
        taken = runtime.steered() if runtime is not None else []
        if not taken:
            return {}
        said = "\n\n".join(f"{STEERED_SAYS}{s.text}" for s in taken)
        return {"hookSpecificOutput": {"hookEventName": RAN_EVENT, "additionalContext": said}}

    return HookMatcher(hooks=[steered])


def _governance(steps: Steps) -> Any:
    """The ``PreToolUse`` hook every tool call goes through, in every permission mode: a
    harness tool runs (the bridge decides about it); any other is the run's decision
    (``bridge.permitted``: hooks, governance), then Claude Code's own — its permission mode and
    rules, and :func:`_asked` when it would ask — with the arguments as they were decided (a
    call a hook rewrote is asked about as rewritten). A subagent's call is let through with
    its prompt briefed (:func:`_briefed`). Before Claude takes a step, the run's progress is
    saved — its session as mirrored so far among it — so a worker that dies during the step
    continues the session from there, not from an earlier step."""
    from claude_agent_sdk import HookMatcher

    from trellis.harness.tools.convert.claude import SERVER, WAITING

    async def pre_tool_use(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        runtime = current()
        assert runtime is not None  # the CLI asks inside the run it works for
        await runtime.progress(now=True)
        name, args = given["tool_name"], dict(given.get("tool_input") or {})
        call_id = tool_use_id or str(given.get("tool_use_id") or name)
        if name.startswith(f"mcp__{SERVER}__"):
            steps.started(call_id, name, args)  # the bridge records it
            return _decided("allow")
        assert runtime.helpers is not None  # the query runs with the attempt's helpers
        helper = await steps.helper_of(call_id, given)
        try:
            # a subagent's call is decided as its helper's work: an approval names the helper
            with answers_read() as read, runtime.helpers.working(helper):
                decided = await bridge.permitted(_spec(runtime, name), args)
        except Paused:  # the run pauses: Claude stops here, and the resume goes on
            steps.helped.pop(call_id, None)
            return {**_decided("deny", WAITING), "continue_": False, "stopReason": WAITING}
        if isinstance(decided, ToolOutcome):
            steps.helped.pop(call_id, None)
            return _decided("deny", str(decided.output))
        if read:  # a person answered about this call, whatever made governance ask
            steps.answered.add(call_id)
        if name in SUBAGENT_TOOLS:
            ran = await _briefed(runtime, call_id, name, decided.args)
            steps.started(call_id, name, ran)
            return _decided("allow", updated=ran)
        steps.started(call_id, name, decided.args)
        if decided.args != args:
            return _decided("ask", updated=decided.args)
        return {}  # Claude Code's own permission check decides

    return HookMatcher(hooks=[pre_tool_use])


def _shell(options: Any, env: Mapping[str, str]) -> Shell:
    """The CLI's shell (``governance.shell.Shell``): where it works (its ``cwd``, else this
    process's); what else the run may read there - the folders the options add to Claude
    Code's own (``add_dirs``: it reads and works there itself, and a team names a folder of
    its own there, such as the tool results the CLI saves), the plugins it is given by a
    folder (the skills written for the run: ``skills.for_run``), where a kept result is read
    (``results.PREFIX``: the name the model is told, which it also tries in its shell); and
    the environment the SDK starts it with (this process's, under the options' ``env``)."""
    from trellis.harness.tools import results

    plugins = [p for p in options.plugins or () if isinstance(p, Mapping)]
    given = [str(p["path"]) for p in plugins if p.get("path")]
    added = [str(d) for d in options.add_dirs or ()]
    reads = [*added, *given, results.PREFIX]
    return Shell(options.cwd or os.getcwd(), reads, {**os.environ, **env})


def _asked(own: Any, steps: Steps, shell: Shell) -> Any:
    """The ``can_use_tool`` the CLI calls when Claude Code would ask its user about a call (its
    permission mode and rules let it through without asking otherwise): the team's own
    callback, when it has one; else a call that only reads runs — a read tool, or a shell
    command the CLI's ``shell`` reads as read-only (``Shell.read_only``: from every folder it
    may be in, none known while another call that may move it is under way) — and any
    other asks a person (``bridge.asked``); Claude's question to its user is the run's
    (:func:`_answered`). A call not let through ends its step."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from trellis.harness.tools.convert.claude import WAITING

    async def can_use_tool(name: str, args: dict[str, Any], context: Any) -> Any:
        this = str(getattr(context, "tool_use_id", None) or name)

        def moving() -> bool:
            under_way = (a for ref, (n, a, _) in steps.calls.items() if ref != this and n in SHELLS)
            return any(moves(a) for a in under_way)

        runtime = current()
        assert runtime is not None  # the CLI asks inside the run it works for
        spec = _spec(runtime, name)
        if own is not None:
            verdict = await own(name, args, context)
        elif name == QUESTION_TOOL:
            try:
                verdict = PermissionResultAllow(updated_input=await _answered(runtime, args))
            except Paused:  # the run pauses: Claude stops here, and the resume goes on
                verdict = PermissionResultDeny(message=WAITING, interrupt=True)
        elif (
            this in steps.answered
            or spec.side_effects == "read"
            or shell.read_only(name, args, moving=moving())
        ):  # a person approved this call (``PreToolUse``)
            verdict = PermissionResultAllow()
        else:
            helpers = runtime.helpers
            assert helpers is not None  # the query runs with the attempt's helpers
            try:
                # the same helper's work as the call's PreToolUse said (``Steps.helped``)
                with helpers.working(steps.helped.get(this)):
                    decided = await bridge.asked(spec, args, CLAUDE_CODE)
            except Paused:  # the run pauses: Claude stops here, and the resume goes on
                decided = None
            if decided is None:
                verdict = PermissionResultDeny(message=WAITING, interrupt=True)
            elif isinstance(decided, ToolOutcome):
                verdict = PermissionResultDeny(message=str(decided.output))
            else:
                verdict = PermissionResultAllow(updated_input=decided.args)
        if not isinstance(verdict, PermissionResultAllow):
            steps.ended(this)
        return verdict

    return can_use_tool


async def _answered(runtime: Runtime, args: dict[str, Any]) -> dict[str, Any]:
    """``AskUserQuestion``'s input with the person's ``answers`` (by question, the label picked,
    several joined with ", "), each question asked as ``ask_user`` asks it: its options, several
    picks with ``multiSelect``, the one labelled :data:`RECOMMENDED` recommended (its
    description why), another answer in words. The call the resumed session makes again in
    other words is the one the person answered (``Replay.made``): Claude reads its questions
    with their answers."""
    made = runtime.replay.made(QUESTION_TOOL, args)
    answers: dict[str, str] = {}
    # the answers it reads are this call's: read again when it is made again after a pause
    calling = runtime.replay.calling(content_key("call", QUESTION_TOOL, made.args))
    async with runtime.limited(None, made=made):
        with calling:
            for asked in made.args.get("questions") or ():
                options = [o for o in asked.get("options") or () if o.get("label")]
                best = next((o for o in options if str(o["label"]).endswith(RECOMMENDED)), {})
                picked = await ask_question(
                    asked["question"],
                    [str(o["label"]) for o in options],
                    multiple=bool(asked.get("multiSelect")),
                    recommended=best.get("label"),
                    why=best.get("description") or (RECOMMENDED.strip("()") if best else None),
                )
                answers[asked["question"]] = (
                    ", ".join(picked) if isinstance(picked, list) else str(picked)
                )
    runtime.replay.ended(made)
    return {**made.args, "answers": answers}


def _spec(runtime: Runtime, name: str) -> ToolSpec:
    """A call's tool as governance decides it: its risk as the run's tool set has it (an
    in-process server's tool: its annotations), else a built-in's
    (:data:`BUILTIN_SIDE_EFFECTS`)."""
    risk = BUILTIN_SIDE_EFFECTS.get(name, UNDECLARED)
    return runtime.native.get(name) or ToolSpec(name=name, side_effects=risk)


def _recorded(steps: Steps, shell: Shell) -> Any:
    """The ``PostToolUse`` / ``PostToolUseFailure`` hook: a call Claude ran itself (a built-in,
    its own MCP servers' tools), recorded as the run's (``tools.bridge.observed``) with its
    result or its error and its duration (from its ``PreToolUse``); a harness tool's call is
    the bridge's. Either way the stall watch goes on once no call runs. A shell command of
    Claude's own (not a subagent's: the CLI keeps their folder) is where the shell may have
    moved (``Shell.ran``)."""
    from claude_agent_sdk import HookMatcher

    from trellis.harness.tools.convert.claude import SERVER

    async def post_tool_use(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        call_id = tool_use_id or str(given.get("tool_use_id") or given.get("tool_name"))
        found = steps.ended(call_id)
        name = str(given.get("tool_name") or "")
        if name.startswith(f"mcp__{SERVER}__"):
            return {}
        steps.adapter.observed(name)
        _, args, started = found or (name, dict(given.get("tool_input") or {}), time.time())
        if name in SHELLS and "agent_id" not in given:
            shell.ran(args)
        failed = given.get("hook_event_name") == FAILED_EVENT
        outcome = ToolOutcome(
            tool=name,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=given.get("error") if failed else _result_text(given.get("tool_response")),
            error_class="ClaudeToolError" if failed else None,
        )
        spec = steps.runtime.native.get(name) or native_spec(name)
        helpers = steps.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        # a call a subagent made is recorded under its run: the CLI names the subagent by
        # its task's id (``agent_id``), which its ``task_started`` tied to its ``Agent`` call
        who, kind = given.get("agent_id"), str(given.get("agent_type") or "")
        working = None if who is None else await helpers.of(str(who), kind)
        with helpers.working(working):
            await bridge.observed(steps.runtime, spec, args, outcome, ref=call_id, started=started)
        return {}

    return HookMatcher(hooks=[post_tool_use])


async def _declared(target: Any) -> list[ToolSpec]:
    """The tools ``ClaudeAgentOptions`` declare (:meth:`ClaudeAdapter.native`, less what the
    agent was seen calling)."""
    from trellis.harness.tools.convert.claude import SERVER

    given = target.tools if isinstance(target.tools, list) else list(BUILTIN_SIDE_EFFECTS)
    denied = set(target.disallowed_tools or ())
    found = [native_spec(name) for name in given if name not in denied]
    own = f"mcp__{SERVER}__"
    found.extend(
        native_spec(name)
        for name in target.allowed_tools or ()
        if name.startswith("mcp__") and not name.startswith(own) and name not in denied
    )
    for server, config in configured_servers(target.mcp_servers).items():
        instance = config.get("instance") if isinstance(config, Mapping) else None
        if server != SERVER and config.get("type") == "sdk" and instance is not None:
            listed = await _sdk_tools(server, instance)
            found.extend(s for s in listed if s.name not in denied)
    return found


async def _sdk_tools(server: str, instance: Any) -> list[ToolSpec]:
    """The tools an in-process MCP server lists (``create_sdk_mcp_server``), as Claude names
    them (``mcp__<server>__<tool>``); ``[]`` when it cannot be asked here (said once)."""
    try:
        entry = instance.get_request_handler("tools/list")
        listed = (await entry.handler(None, None)).tools
    except Exception as exc:
        _unlisted(server, type(exc).__name__)
        return []
    return [
        ToolSpec(
            name=f"mcp__{server}__{tool.name}",
            description=tool.description or tool.name,
            input_schema=dict(tool.input_schema or {"type": "object"}),
            source="mcp",
            server=server,
            side_effects=side_effects_of(tool.annotations),
            **data_of_listed(tool),
        )
        for tool in listed
    ]


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per server, of the most recent
def _unlisted(server: str, reason: str) -> None:
    log.info(
        "an in-process MCP server's tools could not be listed for the run's tool set: they "
        "join it as the agent calls them",
        extra={"server": server, "reason": reason},
    )


def native_spec(name: str) -> ToolSpec:
    """A tool Claude runs itself, as the harness records and publishes it: a built-in by its
    risk (:data:`BUILTIN_SIDE_EFFECTS`) and tagged ``BUILTIN``, an MCP server's tool
    (``mcp__<server>__<tool>``) by its server."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return ToolSpec(name=name, source="mcp", server=parts[1])
    risk = BUILTIN_SIDE_EFFECTS.get(name, UNDECLARED)
    spec = builtin(ToolSpec(name=name, side_effects=risk))
    return helper(spec) if name in SUBAGENT_TOOLS else spec


def _decided(
    decision: Literal["allow", "deny", "ask"],
    reason: str | None = None,
    *,
    updated: dict[str, Any] | None = None,
) -> SyncHookJSONOutput:
    """A ``PreToolUse`` hook's answer."""
    said: PreToolUseHookSpecificOutput = {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
    }
    if reason is not None:
        said["permissionDecisionReason"] = reason
    if updated is not None:
        said["updatedInput"] = updated
    return {"hookSpecificOutput": said}


def configured_servers(configured: Any) -> dict[str, Any]:
    """The team's MCP servers as a dict, so the harness's in-process server (which the SDK
    serves only from a dict) goes beside them: a dict as it is; a path to, or the JSON text of,
    what the CLI's ``--mcp-config`` reads (``{"mcpServers": {...}}``) read into one."""
    if isinstance(configured, dict):
        return configured
    if not configured:
        return {}
    text = str(configured).strip()
    data = json.loads(text if text.startswith("{") else Path(text).read_text())
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ConfigurationError(f"mcp_servers {text[:80]!r} holds no mcpServers object")
    return servers
