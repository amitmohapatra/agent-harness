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

import dataclasses
import json
from collections.abc import Sequence
from typing import Any, Final

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.contracts import ToolSpec
from trellis.contracts.errors import is_pause_signal
from trellis.harness.bridges.helpers import Helpers, Kind, text_of
from trellis.harness.runtime import Paused, current
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import FIX_ARGUMENTS, Tool, arguments_of, not_run

# what the model reads about arguments that are not an object (``tools.base``, as ``ReAct``)
__all__ = ["FIX_ARGUMENTS", "arguments_of", "convert", "governed"]

#: The attribute marking the invoke function of a ``FunctionTool`` the harness made (its calls
#: run through the bridge already): kept by a copy of the tool (``dataclasses.replace``).
MADE: Final = "_trellis_made"
#: The attribute the SDK marks an agent called as a tool with (``Agent.as_tool``).
AGENT_TOOL: Final = "_is_agent_tool"


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    made = [*(_one(t) for t in tools), _read_file()]
    for tool in made:
        setattr(tool.on_invoke_tool, MADE, True)
    return made


def governed(tool: Any, *, helpers: Helpers | None = None) -> Any:
    """An agent's own ``FunctionTool`` (one not built with ``h.tools``), its calls run as the
    harness's are — governance (the catalog's risk, else ``write``: announced), a person when
    it asks, the journal (a resume does not run a call again), its time and its trace — as a
    copy: the agent's own tool is not changed. An agent called as a tool (``Agent.as_tool``)
    is, with ``helpers``, a native helper of the run for each call (``bridges.helpers``).
    Anything else is returned as it is."""
    if not isinstance(tool, FunctionTool) or getattr(tool.on_invoke_tool, MADE, False):
        return tool
    original = tool.on_invoke_tool
    if helpers is not None and getattr(tool, AGENT_TOOL, False):
        original = _helped(tool.name, original, helpers)

    async def invoke(context: ToolContext[Any], arguments: str) -> Any:
        args = arguments_of(arguments)
        if isinstance(args, str) or current() is None:
            return await original(context, arguments)

        async def run(given: dict[str, Any]) -> Any:
            return await original(context, json.dumps(given))

        spec = ToolSpec(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=tool.params_json_schema,
            side_effects="write",
        )
        outcome = await bridge.call(Tool(spec, run), args, call_id=context.tool_call_id)
        return results.readable(outcome.output)

    setattr(invoke, MADE, True)
    return dataclasses.replace(tool, on_invoke_tool=invoke)


def _helped(name: str, invoke: Any, helpers: Helpers) -> Any:
    """``invoke`` (an agent tool's) as a helper of the run: started with the call's input,
    ended with the agent's answer or its error — a pause leaves it to the next attempt, and
    stops the run's agent too (the SDK hands the agent tool's pause back as an error text)."""

    async def helped(context: ToolContext[Any], arguments: str) -> Any:
        args = arguments_of(arguments)
        task = args.get("input") if isinstance(args, dict) else None
        key = context.tool_call_id
        await helpers.start(
            key, kind=Kind.AGENT_TOOL, name=name, task=str(task if task is not None else arguments)
        )
        try:
            answer = await invoke(context, arguments)
        except Exception as exc:
            if not is_pause_signal(exc):
                await helpers.end(key, error=f"{type(exc).__name__}: {exc}")
            raise
        pending = helpers.runtime.pending
        if pending is not None:  # the agent paused inside: the SDK told it as an error text
            raise Paused(pending.interrupt)
        await helpers.end(key, output=text_of(answer))
        return answer

    return helped


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
