"""Claude Agent SDK: the target is the ``ClaudeAgentOptions`` a team already configured.

* input: the prompt; the memory context goes with every prompt as Claude Code's own
  ``UserPromptSubmit`` hook adds context (``bridges.claude.with_context``), beside the options'
  hooks;
* run: ``query(prompt, options)`` on a copy of the options that also carries the harness
  tools as one in-process MCP server (``mcp__trellis__*``) beside the team's own servers — a
  ``mcp_servers`` given as a config file or JSON text is read, since the SDK serves an
  in-process server only from a dict;
* permissions: the SDK's own ``PreToolUse`` hook, which the CLI runs before every tool call
  whatever its ``permission_mode`` or ``allowed_tools`` (where ``can_use_tool`` is skipped) — a
  harness tool is let through (the bridge is its check); any other (Claude Code's built-ins:
  ``Bash``, ``Write``...; a team's own MCP server's) is decided as a harness call is — the run's
  hooks, governance by its risk (:data:`BUILTIN_SIDE_EFFECTS`), a person when it asks — then by
  the team's own ``can_use_tool``, when it has one, asked by the hook;
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
* pause: an approval or ``trellis.current().ask`` stops consuming the query (the CLI process
  ends). The session the CLI kept (``ResultMessage.session_id``) is in the run's journal, and
  the next attempt resumes it (``resume=``): Claude goes on from where it was — its built-in
  tools are not run again — and, told which call a person answered about, calls that tool
  again, which the journal answers. The session is kept in the run's journal too, as the SDK's
  ``SessionStore`` (``bridges.claude.JournalSessions``): the SDK writes it back for the CLI on
  any machine. A session the CLI cannot continue (options keeping their own store, a transcript
  it refuses) is a warning, and the query runs again from its prompt against the journal;
* models: with Bifrost serving the harness, the CLI's model calls go through Bifrost's
  Anthropic route with the virtual key (:func:`routed_env`; the options' own
  ``ANTHROPIC_BASE_URL`` wins), carrying the deny-all MCP scope, the run's dimensions and its
  ``traceparent`` as ``ANTHROPIC_CUSTOM_HEADERS``; lines there that select a stored prompt the
  run pinned (``h.model_headers(prompt=)``) select the version it pinned — the CLI reads them
  as it starts, once per attempt;
* framework options: ``ClaudeAgentOptions`` fields, set on the run's copy of the options
  before the harness's own changes. Merged with them: ``hooks`` (the governance and context
  hooks go after theirs), ``can_use_tool`` (the
  harness's hook asks it after governance, as the target's)
  and ``mcp_servers`` (the harness's ``trellis`` server goes beside them; a server of that name
  is refused). Refused: the fields that choose the session a query continues, and the
  permission prompt tool (:data:`OWNED`).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness import budget
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    helper,
    unique,
)
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.bridges.claude import session_store, with_context
from trellis.harness.bridges.helpers import Helpers, Kind
from trellis.harness.gateways import NO_KEY, Bifrost
from trellis.harness.history import Said, Speaker
from trellis.harness.journal import Pending
from trellis.harness.prompts import CUSTOM_HEADERS, GatewayHeaders, selected_env
from trellis.harness.runtime import Paused, Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, SideEffects
from trellis.harness.tools.mcp import side_effects_of

log = logging.getLogger("trellis.harness")

if TYPE_CHECKING:
    from claude_agent_sdk.types import PreToolUseHookSpecificOutput, SyncHookJSONOutput

#: Claude Code's built-in tools, and what each does as governance sees it (the tool catalog's
#: word overrides it): reads run, writes run and are announced, ``Bash`` asks; a subagent's
#: call (``Agent``, ``Task``) only hands work on — its own calls are governed one by one; a
#: skill (``Skill``) or a plan (``ExitPlanMode``) is read, a slash command (``SlashCommand``)
#: may do anything its command does: a write. Any other tool the CLI asks about is a write.
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
    "BashOutput": "read",
    "ListMcpResourcesTool": "read",
    "ReadMcpResourceTool": "read",
    "Write": "write",
    "Edit": "write",
    "MultiEdit": "write",
    "NotebookEdit": "write",
    "KillShell": "write",
    "SlashCommand": "write",
    "Bash": "irreversible",
}
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
#: The key a query's model steps are watched for a stall under (``Runtime.model_started``).
STEP: Final = "claude_agent_sdk"
#: When the SDK hands the session store a step: as soon as the CLI mirrors it.
SESSION_FLUSH: Final = "eager"
#: Claude Code's tool that runs a subagent (``agents=``): ``Agent``, named ``Task`` before.
SUBAGENT_TOOLS: Final = frozenset({"Agent", "Task"})
#: The subagent an ``Agent`` call runs when it names none.
GENERAL_PURPOSE: Final = "general-purpose"
#: Why a subagent whose call failed without saying why ended.
SUBAGENT_FAILED: Final = "the subagent's call failed"
#: The ``ClaudeAgentOptions`` fields the harness owns: framework options cannot set them.
OWNED: Final[dict[str, str]] = {
    "resume": "the harness resumes the run's own session after a pause",
    "continue_conversation": "the harness resumes the run's own session after a pause",
    "permission_prompt_tool_name": "the harness decides Claude's tool permissions with "
    "can_use_tool (give your own check as can_use_tool)",
}
#: What a resumed session is told: the run goes on, and the call it paused on is made again.
RESUMED: Final = (
    "This task was paused (a person was asked, or the process stopped) and goes on now. If "
    "your last tool call has no result yet, call that tool again with the same arguments: "
    "it runs once. Then carry on with the task."
)
#: What a session resumed after a person answered about one tool call is told: that call, by
#: the name Claude knows it by. The session holds the call's result as "waiting for a person's
#: approval", which :data:`RESUMED`'s "no result yet" does not describe: a model told only that
#: took the waiting call for a failure and tried other tools instead.
RESUMED_CALL: Final = (
    "This task was paused at your call to {tool}: it was waiting for a person's approval, and "
    "the person has answered. Call {tool} again now with the same arguments: it runs once and "
    "returns its result. Then carry on with the task."
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
    #: the tool call a person answered about, when the run resumes from that pause
    answered: ToolCall | None = None
    #: the person denied that call (``REJECT``): it does not run
    denied: bool = False


class ClaudeRunError(RuntimeError):
    """The CLI reported the run as failed."""


class ClaudeAdapter:
    name: ClassVar[str] = "claude_agent_sdk"
    tool_format: ToolFormat = "claude"
    fixed_tools: bool = False
    narrows: Narrowing = "run"
    #: a query takes one prompt, not a conversation: memory's window is in the context's text
    replays_history = False

    def __init__(self) -> None:
        #: the tools the options declare (:meth:`native`), once read
        self._static: list[ToolSpec] | None = None
        #: the MCP tools the agent was seen calling, the most recent kept
        self.seen: LRUSet[str] = LRUSet(maxsize=SEEN_KEPT)

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        return None

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
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield block.text
        yield Output(messages)

    def extract(self, target: Any, output: Any) -> Extracted:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        transcript: list[Said] = []
        answer: Any = None
        for message in output:
            if isinstance(message, AssistantMessage):
                text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
                if text:
                    transcript.append(Said(Speaker.ASSISTANT, text))
            elif isinstance(message, ResultMessage):
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
        return dataclasses.replace(
            native_input,
            answered=pending.interrupt.tool_call,
            denied=resolution.decision is InterruptDecision.REJECT,
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
    async def _helped(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        """The query's messages (:meth:`_messages`), each subagent Claude runs a helper of the
        run (``bridges.helpers``) from its ``Agent`` call to that call's result."""
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        try:
            async for message in self._messages(target, native_input, run):
                await _subagents(helpers, message)
                yield message
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)

    async def _messages(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncIterator[Any]:
        """The query's messages — the session an earlier attempt kept resumed, when there is
        one the CLI still holds; else from the prompt."""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        runtime = run.runtime
        journal = runtime.replay.journal
        resumed, answered = journal.session, False
        async for message in self._query(target, native_input, run, resumed):
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
                async for again in self._query(target, native_input, run, None):
                    yield again
                return
            answered = answered or isinstance(message, AssistantMessage)
            yield message

    async def _query(
        self, target: Any, native_input: Any, run: Invocation, session: str | None
    ) -> AsyncIterator[Any]:
        """One query (``session``: the one it resumes); the session it runs in kept in the
        journal as it goes."""
        from claude_agent_sdk import ResultMessage, query

        runtime = run.runtime
        given = runtime.framework_options
        target = dataclasses.replace(target, **given) if given else target
        steps = Steps(runtime, self)
        options = _options(target, native_input.context, run, session, steps)
        prompt = native_input.prompt if session is None else _resumed(native_input, run)
        steps.waiting()
        try:
            async for message in query(prompt=prompt, options=options):
                steps.waiting()  # the next message is the next step's
                found = getattr(message, "session_id", None)
                if isinstance(found, str):
                    runtime.replay.journal.session = found
                if isinstance(message, ResultMessage):
                    _spent(runtime, options, message)
                yield message
                if runtime.pending is not None:
                    # the run paused: stop here; the CLI process goes with it
                    return
        finally:
            runtime.model_ended(STEP)


async def _briefed(
    runtime: Runtime, call_id: str, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """A call as it runs: a subagent's (:data:`SUBAGENT_TOOLS`) started as a helper of the run
    (``bridges.helpers``) and its ``prompt`` given the memory context for its task before it;
    any other as it was decided."""
    helpers = runtime.helpers
    if helpers is None or name not in SUBAGENT_TOOLS:
        return args
    task = str(args.get("prompt") or args.get("description") or "")
    kind, called = Kind.SUBAGENT, str(args.get("subagent_type") or GENERAL_PURPOSE)
    helper = await helpers.start(call_id, kind=kind, name=called, task=task)
    return {**args, "prompt": helpers.briefed(helper, task)} if "prompt" in args else args


async def _subagents(helpers: Helpers, message: Any) -> None:
    """A subagent's start (the main agent's ``Agent`` call: its ``subagent_type`` and
    ``prompt``; started already when its ``PreToolUse`` came first: :func:`_briefed`) or end
    (that call's result) in ``message``."""
    from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock, UserMessage

    if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
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
                else:
                    await helpers.end(result.tool_use_id, output=text)


def _result_text(content: Any) -> str:
    """A tool result's text (a string, its text blocks, or the text a built-in's structured
    response holds)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        found = content.get("content", content.get("output", content.get("stdout")))
        return _result_text(found) if found is not None else json.dumps(content, default=str)
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def _spent(runtime: Runtime, options: Any, result: Any) -> None:
    """A query's usage and cost (its ``ResultMessage``: the subagents' included) on the run's
    budget."""
    usage = result.usage or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    budget.account(
        runtime,
        model=options.model,
        input_tokens=int(usage.get("input_tokens") or 0) + cached + written,
        output_tokens=int(usage.get("output_tokens") or 0),
        cached=cached,
        cost=result.total_cost_usd,
    )


def _resumed(native_input: ClaudeInput, run: Invocation) -> str:
    """What the resumed session is told: the call a person answered about, named as Claude
    calls it (a harness tool through the ``trellis`` server, a built-in by its own name)."""
    from trellis.harness.tools.convert.claude import SERVER

    call = native_input.answered
    if call is None:
        return RESUMED
    harness = any(t.name == call.tool for t in run.tools)
    told = RESUMED_DENIED if native_input.denied else RESUMED_CALL
    return told.format(tool=f"mcp__{SERVER}__{call.tool}" if harness else call.tool)


class Steps:
    """One query's steps as the stall watchdog sees them (``Runtime.model_started``): Claude is
    waited on from one message to the next, except while a tool call runs (from its
    ``PreToolUse`` to its ``PostToolUse``: a tool's own time is not a stalled model); and the
    calls under way, by Claude's id, with when each began (what ``PostToolUse`` records)."""

    __slots__ = ("adapter", "calls", "runtime")

    def __init__(self, runtime: Runtime, adapter: ClaudeAdapter) -> None:
        self.runtime = runtime
        #: whose query it is: the MCP tools it is seen calling join its tool set
        self.adapter = adapter
        #: the tool calls under way: their name, arguments and start (``time.time()``)
        self.calls: dict[str, tuple[str, dict[str, Any], float]] = {}

    def waiting(self) -> None:
        """Claude is to answer now (unless a call runs): watched for a stall."""
        if not self.calls:
            self.runtime.model_started(STEP)

    def started(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.calls[call_id] = (name, args, time.time())
        self.runtime.model_ended(STEP)

    def ended(self, call_id: str) -> tuple[str, dict[str, Any], float] | None:
        found = self.calls.pop(call_id, None)
        self.waiting()
        return found


def _options(
    options: Any, context: str | None, run: Invocation, session: str | None, steps: Steps
) -> Any:
    from trellis.harness.tools.convert import claude as convert

    if options.permission_prompt_tool_name:
        raise ConfigurationError(
            "the harness decides Claude's tool calls with a PreToolUse hook: give your own "
            "permission check as can_use_tool (the hook asks it after governance), not "
            "permission_prompt_tool_name"
        )
    runtime = run.runtime
    hooks = _with_governance(options.hooks, options.can_use_tool, steps)
    changes: dict[str, Any] = {
        "hooks": with_context(hooks, context) if context else hooks,
        # the hook asks the team's callback itself: the CLI has no permission prompt to make
        "can_use_tool": None,
        "max_buffer_size": options.max_buffer_size or CLI_MESSAGE_BYTES,
    }
    store = session_store(options, runtime.replay.journal, runtime.agent_id)
    if store is not None:
        # each step mirrored as it is taken: the journal saved with a pause holds it
        changes["session_store"] = store
        changes["session_store_flush"] = SESSION_FLUSH
    if run.native_tools is not None:
        servers = configured_servers(options.mcp_servers)
        changes["mcp_servers"] = {**servers, convert.SERVER: run.native_tools}
    narrowed = _narrowed(runtime, options.disallowed_tools)
    if narrowed is not None:
        changes["disallowed_tools"] = narrowed
    if session is not None:
        changes["resume"] = session
    env = routed_env(selected_env(options.env, runtime) or options.env, runtime)
    if env is not None:
        changes["env"] = env
    return dataclasses.replace(options, **changes)


def routed_env(env: Mapping[str, str], runtime: Runtime) -> dict[str, str] | None:
    """The CLI's environment with its model calls routed through Bifrost when it serves the
    harness (``None``: as it is): Bifrost's Anthropic route (``ANTHROPIC_BASE_URL``, the
    virtual key as ``ANTHROPIC_API_KEY``) — unless the options' own ``env`` names a base URL
    (code over the environment) — and, on Bifrost's route, its headers as
    ``ANTHROPIC_CUSTOM_HEADERS`` lines beside the options' own: the deny-all MCP scope, the
    run's dimensions and its ``traceparent`` (read as the CLI starts)."""
    gateway = runtime.agent.harness.model_gateway
    if not isinstance(gateway, Bifrost):
        return None
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


def _narrowed(runtime: Runtime, denied: Sequence[str] | None) -> list[str] | None:
    """The options' ``disallowed_tools`` with the framework's own tools the run does not offer
    (the tool hints' narrowing: Claude's built-ins and MCP tools too, for the whole query);
    ``None`` when nothing is narrowed."""
    if runtime.offered is None:
        return None
    given = list(denied or ())
    hidden = [n for n in runtime.native if not runtime.offers(n) and n not in given]
    return [*given, *hidden] if hidden else None


def _with_governance(hooks: Mapping[str, Any] | None, own: Any, steps: Steps) -> dict[str, Any]:
    """The options' hooks with the governance hook (:func:`_governance`) and the record hooks
    (:func:`_recorded`) after theirs."""
    given = dict(hooks or {})
    given[TOOL_EVENT] = [*given.get(TOOL_EVENT, ()), _governance(own, steps)]
    recorded = _recorded(steps)
    for event in (RAN_EVENT, FAILED_EVENT):
        given[event] = [*given.get(event, ()), recorded]
    return given


def _governance(own: Any, steps: Steps) -> Any:
    """The ``PreToolUse`` hook every tool call goes through, in every permission mode: a
    harness tool runs (the bridge decides about it); any other is the run's decision
    (``bridge.permitted``: hooks, governance, a person), then ``own``'s — the team's
    ``can_use_tool`` — with the arguments as they were decided. Before Claude takes a step, the
    run's progress is saved — its session as mirrored so far among it — so a worker that dies
    during the step continues the session from there, not from an earlier step."""
    from claude_agent_sdk import HookMatcher, PermissionResultAllow, ToolPermissionContext

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
        side_effects = BUILTIN_SIDE_EFFECTS.get(name, DEFAULT_SIDE_EFFECTS)
        try:
            decided = await bridge.permitted(ToolSpec(name=name, side_effects=side_effects), args)
        except Paused:  # the run pauses: Claude stops here, and the resume goes on
            return {**_decided("deny", WAITING), "continue_": False, "stopReason": WAITING}
        if isinstance(decided, ToolOutcome):
            return _decided("deny", str(decided.output))
        if own is None:
            ran = await _briefed(runtime, call_id, name, decided.args)
            steps.started(call_id, name, ran)
            return _decided("allow", updated=ran)
        verdict = await own(name, decided.args, ToolPermissionContext(tool_use_id=tool_use_id))
        if isinstance(verdict, PermissionResultAllow):
            ran = await _briefed(runtime, call_id, name, verdict.updated_input or decided.args)
            steps.started(call_id, name, ran)
            return _decided("allow", updated=ran)
        refused = _decided("deny", verdict.message)
        return {**refused, "continue_": False} if verdict.interrupt else refused

    return HookMatcher(hooks=[pre_tool_use])


def _recorded(steps: Steps) -> Any:
    """The ``PostToolUse`` / ``PostToolUseFailure`` hook: a call Claude ran itself (a built-in,
    its own MCP servers' tools), recorded as the run's (``tools.bridge.observed``) with its
    result or its error and its duration (from its ``PreToolUse``); a harness tool's call is
    the bridge's. Either way the stall watch goes on once no call runs."""
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
        failed = given.get("hook_event_name") == FAILED_EVENT
        outcome = ToolOutcome(
            tool=name,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=given.get("error") if failed else _result_text(given.get("tool_response")),
            error_class="ClaudeToolError" if failed else None,
        )
        spec = native_spec(name)
        helpers = steps.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        # a call a subagent made (the CLI names it: ``agent_type``) is recorded under its run
        working = helpers.named(str(given["agent_type"])) if "agent_type" in given else None
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
    risk (:data:`BUILTIN_SIDE_EFFECTS`), an MCP server's tool (``mcp__<server>__<tool>``) by its
    server."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return ToolSpec(name=name, source="mcp", server=parts[1])
    spec = ToolSpec(name=name, side_effects=BUILTIN_SIDE_EFFECTS.get(name, DEFAULT_SIDE_EFFECTS))
    return helper(spec) if name in SUBAGENT_TOOLS else spec


def _decided(
    decision: Literal["allow", "deny"],
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
