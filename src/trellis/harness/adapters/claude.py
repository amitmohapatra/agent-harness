"""Claude Agent SDK: the target is the ``ClaudeAgentOptions`` a team already configured.

* input: the prompt, and the memory context appended to the options' system prompt;
* run: ``query(prompt, options)`` on a copy of the options that also carries the harness
  tools as one in-process MCP server (``mcp__trellis__*``) beside the team's own servers — a
  ``mcp_servers`` given as a config file or JSON text is read, since the SDK serves an
  in-process server only from a dict;
* permissions: the SDK's own ``can_use_tool`` — a harness tool is let through (the bridge is
  its check); any other tool the CLI asks about (Claude Code's built-ins: ``Bash``,
  ``Write``...; a team's own MCP server's) is decided as a harness call is — the run's hooks,
  governance by its risk (:data:`BUILTIN_SIDE_EFFECTS`), a person when it asks — then by the
  team's own ``can_use_tool``, when it has one;
* pause: an approval or ``trellis.current().ask`` stops consuming the query (the CLI process
  ends). The session the CLI kept (``ResultMessage.session_id``) is in the run's journal, and
  the next attempt resumes it (``resume=``): Claude goes on from where it was — its built-in
  tools are not run again — and, told which call a person answered about, calls that tool
  again, which the journal answers. A session the CLI no longer holds (another machine, its
  store cleared) is a warning, and the query runs again from its prompt against the journal;
* model headers: ``ANTHROPIC_CUSTOM_HEADERS`` in the options' ``env`` that select a stored
  prompt the run pinned (``h.model_headers(prompt=)``) select the version it pinned — the CLI
  reads them as it starts, once per attempt;
* framework options: ``ClaudeAgentOptions`` fields, set on the run's copy of the options
  before the harness's own changes. Merged with them: ``system_prompt`` (the memory context is
  appended to it), ``can_use_tool`` (the harness asks it after governance, as the target's)
  and ``mcp_servers`` (the harness's ``trellis`` server goes beside them; a server of that name
  is refused). Refused: the fields that choose the session a query continues, and the
  permission prompt tool (:data:`OWNED`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

from trellis.contracts import (
    ConfigurationError,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
    ToolSpec,
)
from trellis.harness.adapters.base import Extracted, Invocation, Narrowing, Output, ToolFormat
from trellis.harness.journal import Pending
from trellis.harness.prompts import selected_env
from trellis.harness.runtime import Paused
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, SideEffects

#: What Claude Code's built-in tools do, as governance sees them (the tool catalog's word
#: overrides it): reads run, writes run and are announced, ``Bash`` asks. Any other tool the
#: CLI asks about is a write.
BUILTIN_SIDE_EFFECTS: Final[dict[str, SideEffects]] = {
    "Read": "read",
    "Glob": "read",
    "Grep": "read",
    "LS": "read",
    "WebFetch": "read",
    "WebSearch": "read",
    "TodoWrite": "read",
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


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeInput:
    prompt: str
    context: str | None
    #: the tool call a person answered about, when the run resumes from that pause
    answered: ToolCall | None = None


class ClaudeRunError(RuntimeError):
    """The CLI reported the run as failed."""


class ClaudeAdapter:
    name: ClassVar[str] = "claude_agent_sdk"
    tool_format: ToolFormat = "claude"
    fixed_tools: bool = False
    narrows: Narrowing = "run"

    def keeps_conversation(self, target: Any) -> bool:
        return False

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
        prompt = input if isinstance(input, str) else json.dumps(input, default=str)
        return ClaudeInput(prompt=prompt, context=context)

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return [m async for m in self._messages(target, native_input, run)]

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from claude_agent_sdk import AssistantMessage, TextBlock

        messages: list[Any] = []
        async for message in self._messages(target, native_input, run):
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield block.text
        yield Output(messages)

    def extract(self, target: Any, output: Any) -> Extracted:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        transcript: list[Any] = []
        answer: Any = None
        for message in output:
            if isinstance(message, AssistantMessage):
                text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
                if text:
                    transcript.append(("assistant", text))
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
        return dataclasses.replace(native_input, answered=pending.interrupt.tool_call)

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

    # ------------------------------------------------------------------ internals
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
        from claude_agent_sdk import query

        runtime = run.runtime
        given = runtime.framework_options
        target = dataclasses.replace(target, **given) if given else target
        options = _options(target, native_input.context, run, session)
        prompt = native_input.prompt if session is None else _resumed(native_input, run)
        async for message in query(prompt=prompt, options=options):
            found = getattr(message, "session_id", None)
            if isinstance(found, str):
                runtime.replay.journal.session = found
            yield message
            if runtime.pending is not None:
                # the run paused: stop here; the CLI process goes with it
                return


def _resumed(native_input: ClaudeInput, run: Invocation) -> str:
    """What the resumed session is told: the call a person answered about, named as Claude
    calls it (a harness tool through the ``trellis`` server, a built-in by its own name)."""
    from trellis.harness.tools.convert.claude import SERVER

    call = native_input.answered
    if call is None:
        return RESUMED
    harness = any(t.name == call.tool for t in run.tools)
    return RESUMED_CALL.format(tool=f"mcp__{SERVER}__{call.tool}" if harness else call.tool)


def _options(options: Any, context: str | None, run: Invocation, session: str | None) -> Any:
    from trellis.harness.tools.convert import claude as convert

    if options.permission_prompt_tool_name:
        raise ConfigurationError(
            "the harness decides Claude's tool permissions with can_use_tool: give your own "
            "permission check as can_use_tool (the harness asks it after governance), not "
            "permission_prompt_tool_name"
        )
    changes: dict[str, Any] = {
        "can_use_tool": _permission(options.can_use_tool),
        "max_buffer_size": options.max_buffer_size or CLI_MESSAGE_BYTES,
    }
    if context:
        changes["system_prompt"] = _with_context(options.system_prompt, context)
    if run.native_tools is not None:
        servers = configured_servers(options.mcp_servers)
        changes["mcp_servers"] = {**servers, convert.SERVER: run.native_tools}
    if session is not None:
        changes["resume"] = session
    env = selected_env(options.env, run.runtime)
    if env is not None:
        changes["env"] = env
    return dataclasses.replace(options, **changes)


def _permission(own: Any) -> Any:
    """The options' ``can_use_tool``: a harness tool runs (the bridge decides about it); any
    other is the run's decision (``bridge.permitted``: hooks, governance, a person), then
    ``own``'s, the team's callback, with the arguments as they were decided."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from trellis.harness.tools.convert.claude import SERVER, WAITING

    async def can_use_tool(name: str, args: dict[str, Any], context: Any) -> Any:
        if name.startswith(f"mcp__{SERVER}__"):
            return PermissionResultAllow()
        side_effects = BUILTIN_SIDE_EFFECTS.get(name, DEFAULT_SIDE_EFFECTS)
        try:
            decided = await bridge.permitted(ToolSpec(name=name, side_effects=side_effects), args)
        except Paused:
            return PermissionResultDeny(message=WAITING, interrupt=True)
        if isinstance(decided, ToolOutcome):
            return PermissionResultDeny(message=str(decided.output))
        if own is None:
            return PermissionResultAllow(updated_input=decided.args)
        verdict = await own(name, decided.args, context)
        if isinstance(verdict, PermissionResultAllow) and verdict.updated_input is None:
            return dataclasses.replace(verdict, updated_input=decided.args)
        return verdict

    return can_use_tool


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


def _with_context(system_prompt: Any, context: str) -> Any:
    if system_prompt is None:
        return context
    if isinstance(system_prompt, str):
        return f"{system_prompt}\n\n{context}"
    if isinstance(system_prompt, dict) and system_prompt.get("type") == "preset":
        appended = system_prompt.get("append")
        return {**system_prompt, "append": f"{appended}\n\n{context}" if appended else context}
    if isinstance(system_prompt, dict) and system_prompt.get("type") == "custom":
        return {**system_prompt, "prompt": f"{system_prompt['prompt']}\n\n{context}"}
    return system_prompt  # a prompt file: the CLI reads it, and the context has no place in it
