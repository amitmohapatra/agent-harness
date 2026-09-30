"""Claude Agent SDK: the target is the ``ClaudeAgentOptions`` a team already configured.

* input: the prompt, and the memory context appended to the options' system prompt;
* run: ``query(prompt, options)`` on a copy of the options that also carries the harness
  tools as one in-process MCP server (``mcp__trellis__*``, pre-allowed: the bridge is the
  permission check);
* pause: ``trellis.current().ask`` inside a harness tool stops consuming the query (the CLI
  process ends) and a resume re-runs it against the journal.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trellis.contracts import InterruptResolution
from trellis.harness.adapters.base import Extracted, Invocation, Output, ToolFormat
from trellis.harness.journal import Pending


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeInput:
    prompt: str
    context: str | None


class ClaudeRunError(RuntimeError):
    """The CLI reported the run as failed."""


class ClaudeAdapter:
    name: ClassVar[str] = "claude_agent_sdk"
    tool_format: ClassVar[ToolFormat] = "claude"
    fixed_tools: ClassVar[bool] = False

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
        return native_input

    # ------------------------------------------------------------------ internals
    async def _messages(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncIterator[Any]:
        from claude_agent_sdk import query

        options = _options(target, native_input.context, run)
        runtime = run.runtime
        async for message in query(prompt=native_input.prompt, options=options):
            yield message
            if runtime.pending is not None:
                # a harness tool paused the run: stop here; the CLI process goes with it
                return


def _options(options: Any, context: str | None, run: Invocation) -> Any:
    from trellis.harness.tools.convert import claude as convert

    changes: dict[str, Any] = {}
    if context:
        changes["system_prompt"] = _with_context(options.system_prompt, context)
    if run.native_tools is not None:
        servers = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
        changes["mcp_servers"] = {**servers, convert.SERVER: run.native_tools}
        changes["allowed_tools"] = [*options.allowed_tools, *convert.allowed_names(run.tools)]
    return dataclasses.replace(options, **changes) if changes else options


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
