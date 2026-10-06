"""Harness tools as OpenAI Agents SDK ``FunctionTool``\\ s. Each is enabled per turn only while
the run offers it (``Runtime.offers``: the tool hints' candidates, the memory tools, the tools
already used), so the model sees the tool schemas that fit the task.

The SDK hands a ``FunctionTool`` the model's arguments as the text the model wrote; text that
is not a JSON object is what the model reads back (:func:`arguments_of`), as the SDK's own
``@function_tool`` does it — the call is not run, and the run goes on — rather than an
exception, which ends a run (the SDK raises everything a ``FunctionTool`` raises)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.harness.runtime import current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import FIX_ARGUMENTS, Tool, arguments_of, not_run
from trellis.harness.tools.convert import text_of

# what the model reads about arguments that are not an object (``tools.base``, as ``ReAct``)
__all__ = ["FIX_ARGUMENTS", "arguments_of", "convert"]


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    return [_one(t) for t in tools]


def _one(tool: Tool) -> FunctionTool:
    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = arguments_of(arguments)
        if isinstance(args, str):
            return not_run(tool.name, args)
        return text_of((await bridge.call(tool, args, call_id=context.tool_call_id)).output)

    return FunctionTool(
        name=tool.name,
        description=tool.spec.description or tool.name,
        params_json_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
        on_invoke_tool=invoke,
        # the schema is the tool's own (MCP, OpenAPI...), not one written for strict mode
        strict_json_schema=False,
        is_enabled=lambda _context, _agent: _offered(tool.name),
    )


def _offered(name: str) -> bool:
    runtime = current()
    return runtime is None or runtime.offers(name)
