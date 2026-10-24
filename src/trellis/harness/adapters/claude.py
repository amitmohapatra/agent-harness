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
* model headers: ``ANTHROPIC_CUSTOM_HEADERS`` in the options' ``env`` that select a stored
  prompt the run pinned (``h.model_headers(prompt=)``) select the version it pinned — the CLI
  reads them as it starts, once per attempt;
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
import json
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
)
from trellis.harness import budget
from trellis.harness.adapters.base import Extracted, Invocation, Narrowing, Output, ToolFormat
from trellis.harness.bridges.claude import session_store, with_context
from trellis.harness.bridges.helpers import Helpers, Kind
from trellis.harness.history import Said, Speaker
from trellis.harness.journal import Pending
from trellis.harness.prompts import selected_env
from trellis.harness.runtime import Paused, Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, SideEffects

if TYPE_CHECKING:
    from claude_agent_sdk.types import PreToolUseHookSpecificOutput, SyncHookJSONOutput

#: What Claude Code's built-in tools do, as governance sees them (the tool catalog's word
#: overrides it): reads run, writes run and are announced, ``Bash`` asks; a subagent's call
#: (``Agent``, ``Task``) only hands work on — its own calls are governed one by one. Any other
#: tool the CLI asks about is a write.
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
    "Write": "write",
    "Edit": "write",
    "MultiEdit": "write",
    "NotebookEdit": "write",
    "Bash": "irreversible",
}
#: The most one message from the CLI may be (the SDK's ``max_buffer_size``: 1 MiB unless the
#: options say): a tool's result comes back from the CLI in one, and a run keeps results as large
#: as an agent-runs artifact (50 MiB: a journal over a checkpoint's size travels as one) —
#: escaped as JSON, twice that at most.
CLI_MESSAGE_BYTES: Final = 2 * 50 * 1024 * 1024
#: The hook event every tool call is governed at.
TOOL_EVENT: Final = "PreToolUse"
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

    def native_tools(self, target: Any) -> list[str]:
        """Claude Code's built-in tools the options leave it (``tools``, else the default set
        :data:`BUILTIN_SIDE_EFFECTS` knows, less ``disallowed_tools``) and the MCP tools of
        its own servers the options name (``allowed_tools``' ``mcp__<server>__<tool>``
        entries; a server's other tools are listed only by the CLI)."""
        from trellis.harness.tools.convert.claude import SERVER

        given = target.tools if isinstance(target.tools, list) else list(BUILTIN_SIDE_EFFECTS)
        denied = set(target.disallowed_tools or ())
        builtins = [name for name in given if name not in denied]
        own = f"mcp__{SERVER}__"
        servers = [
            name
            for name in target.allowed_tools or ()
            if name.startswith("mcp__") and not name.startswith(own) and name not in denied
        ]
        return [*builtins, *servers]

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

    @staticmethod
    async def _query(
        target: Any, native_input: Any, run: Invocation, session: str | None
    ) -> AsyncIterator[Any]:
        """One query (``session``: the one it resumes); the session it runs in kept in the
        journal as it goes."""
        from claude_agent_sdk import ResultMessage, query

        runtime = run.runtime
        given = runtime.framework_options
        target = dataclasses.replace(target, **given) if given else target
        options = _options(target, native_input.context, run, session)
        prompt = native_input.prompt if session is None else _resumed(native_input, run)
        async for message in query(prompt=prompt, options=options):
            found = getattr(message, "session_id", None)
            if isinstance(found, str):
                runtime.replay.journal.session = found
            if isinstance(message, ResultMessage):
                _spent(runtime, options, message)
            yield message
            if runtime.pending is not None:
                # the run paused: stop here; the CLI process goes with it
                return


async def _subagents(helpers: Helpers, message: Any) -> None:
    """A subagent's start (the main agent's ``Agent`` call: its ``subagent_type`` and
    ``prompt``) or end (that call's result) in ``message``."""
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
    """A tool result's text (a string, or its text blocks)."""
    if isinstance(content, str):
        return content
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


def _options(options: Any, context: str | None, run: Invocation, session: str | None) -> Any:
    from trellis.harness.tools.convert import claude as convert

    if options.permission_prompt_tool_name:
        raise ConfigurationError(
            "the harness decides Claude's tool calls with a PreToolUse hook: give your own "
            "permission check as can_use_tool (the hook asks it after governance), not "
            "permission_prompt_tool_name"
        )
    runtime = run.runtime
    hooks = _with_governance(options.hooks, options.can_use_tool)
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
    if session is not None:
        changes["resume"] = session
    env = selected_env(options.env, runtime)
    if env is not None:
        changes["env"] = env
    return dataclasses.replace(options, **changes)


def _with_governance(hooks: Mapping[str, Any] | None, own: Any) -> dict[str, Any]:
    """The options' hooks with the governance hook (:func:`_governance`) after theirs."""
    given = dict(hooks or {})
    given[TOOL_EVENT] = [*given.get(TOOL_EVENT, ()), _governance(own)]
    return given


def _governance(own: Any) -> Any:
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
        if name.startswith(f"mcp__{SERVER}__"):
            return _decided("allow")
        side_effects = BUILTIN_SIDE_EFFECTS.get(name, DEFAULT_SIDE_EFFECTS)
        try:
            decided = await bridge.permitted(ToolSpec(name=name, side_effects=side_effects), args)
        except Paused:  # the run pauses: Claude stops here, and the resume goes on
            return {**_decided("deny", WAITING), "continue_": False, "stopReason": WAITING}
        if isinstance(decided, ToolOutcome):
            return _decided("deny", str(decided.output))
        if own is None:
            return _decided("allow", updated=decided.args)
        verdict = await own(name, decided.args, ToolPermissionContext(tool_use_id=tool_use_id))
        if isinstance(verdict, PermissionResultAllow):
            return _decided("allow", updated=verdict.updated_input or decided.args)
        refused = _decided("deny", verdict.message)
        return {**refused, "continue_": False} if verdict.interrupt else refused

    return HookMatcher(hooks=[pre_tool_use])


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
