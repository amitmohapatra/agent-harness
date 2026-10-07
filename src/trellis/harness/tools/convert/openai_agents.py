"""Harness tools as OpenAI Agents SDK ``FunctionTool``\\ s. Each is enabled per turn only while
the run offers it (``Runtime.offers``: the tool hints' candidates, the memory tools, the tools
already used), so the model sees the tool schemas that fit the task.

The SDK hands a ``FunctionTool`` the model's arguments as the text the model wrote; text that
is not a JSON object is what the model reads back (:func:`arguments_of`), as the SDK's own
``@function_tool`` does it — the call is not run, and the run goes on — rather than an
exception, which ends a run (the SDK raises everything a ``FunctionTool`` raises).

A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; ``read_file`` (with them, enabled once the run keeps one) pages it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.harness.runtime import current
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import FIX_ARGUMENTS, Tool, arguments_of, not_run

# what the model reads about arguments that are not an object (``tools.base``, as ``ReAct``)
__all__ = ["FIX_ARGUMENTS", "arguments_of", "convert"]


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    return [*(_one(t) for t in tools), _read_file()]


def _one(tool: Tool) -> FunctionTool:
    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = arguments_of(arguments)
        if isinstance(args, str):
            return not_run(tool.name, args)
        outcome = await bridge.call(tool, args, call_id=context.tool_call_id)
        return results.readable(outcome.output)

    return FunctionTool(
        name=tool.name,
        description=tool.spec.description or tool.name,
        params_json_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
        on_invoke_tool=invoke,
        # the schema is the tool's own (MCP, OpenAPI...), not one written for strict mode
        strict_json_schema=False,
        is_enabled=lambda _context, _agent: _offered(tool.name),
    )


def _read_file() -> FunctionTool:
    """``read_file``: a page of a large result the run keeps (``tools.results``)."""

    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = arguments_of(arguments)
        return not_run(results.READ_FILE, args) if isinstance(args, str) else results.read(args)

    return FunctionTool(
        name=results.READ_FILE,
        description=results.READ_FILE_DESCRIPTION,
        params_json_schema=results.READ_FILE_SCHEMA,
        on_invoke_tool=invoke,
        strict_json_schema=False,
        is_enabled=lambda _context, _agent: results.kept(),
    )


def _offered(name: str) -> bool:
    runtime = current()
    return runtime is None or runtime.offers(name)
