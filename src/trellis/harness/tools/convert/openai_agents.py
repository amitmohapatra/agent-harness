"""Harness tools as OpenAI Agents SDK ``FunctionTool``\\ s. Each is enabled per turn only while
the run offers it (``Runtime.offers``: the tool hints' candidates, the memory tools, the tools
already used), so the model sees the tool schemas that fit the task — and none on the turn
after a stall (``tools.bridge.stalled``: the same call refused as a loop), so the model's
reply to ``bridge.STALLED`` is the run's answer, as ``StallGuard``'s last call on LangChain.

The SDK hands a ``FunctionTool`` the model's arguments as the text the model wrote; text that
is not a JSON object is what the model reads back (:func:`arguments_of`), as the SDK's own
``@function_tool`` does it — the call is not run, and the run goes on — rather than an
exception, which ends a run (the SDK raises everything a ``FunctionTool`` raises).

A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; ``read_file`` (with them, enabled once the run keeps one) pages it.

A call repeating an earlier one of its model reply (the same tool, the same arguments:
``Runtime.repeats``) does not run: the model reads that its result is the earlier call's
(``tools.bridge.same``)."""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.contracts import ToolOutcome, ToolSpec
from trellis.harness import budget
from trellis.harness.bridges.helpers import Helpers, Kind, text_of
from trellis.harness.journal import content_key
from trellis.harness.runtime import Paused, current
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import FIX_ARGUMENTS, Tool, arguments_of, not_run

# what the model reads about arguments that are not an object (``tools.base``, as ``ReAct``)
__all__ = ["FIX_ARGUMENTS", "arguments_of", "convert", "governed", "held"]

#: The attribute marking the invoke function of a ``FunctionTool`` the harness made (its calls
#: run through the bridge already): kept by a copy of the tool (``dataclasses.replace``).
MADE: Final = "_trellis_made"
#: The attribute the SDK marks an agent called as a tool with (``Agent.as_tool``), and the one
#: holding that agent.
AGENT_TOOL: Final = "_is_agent_tool"
AGENT_OF_TOOL: Final = "_agent_instance"


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    made = [*(_one(t) for t in tools), *(_reader(*r) for r in results.readers())]
    for tool in made:
        setattr(tool.on_invoke_tool, MADE, True)
    return made


def governed(tool: Any, helpers: Helpers | None = None) -> Any:
    """An agent's own ``FunctionTool`` (one not built with ``h.tools``), its calls run as the
    harness's are — governance (the catalog's risk, else ``write``: announced), a person when
    it asks, the journal (a resume does not run a call again), its time and its trace — and
    offered per turn only while the run offers it (the tool hints' narrowing), as a copy: the
    agent's own tool is not changed. An agent called as a tool (``Agent.as_tool``)
    is, with ``helpers``, a native helper of the run for each call (:func:`_helped`). Anything
    else is returned as it is."""
    if not isinstance(tool, FunctionTool) or getattr(tool.on_invoke_tool, MADE, False):
        return tool
    original = tool.on_invoke_tool
    if helpers is not None and getattr(tool, AGENT_TOOL, False):
        original = _helped(tool, original, helpers)

    async def invoke(context: ToolContext[Any], arguments: str) -> Any:
        args = arguments_of(arguments)
        runtime = current()
        if isinstance(args, str) or runtime is None:
            return await original(context, arguments)
        repeat = _repeat(context, arguments, args)
        if repeat is not None:
            return results.readable(repeat.output, tool.name)

        async def run(given: dict[str, Any]) -> Any:
            return await original(context, json.dumps(given))

        # as the run's tool set has it (an agent tool: a helper, read), else undeclared: write
        spec = runtime.native.get(tool.name) or ToolSpec(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=tool.params_json_schema,
            side_effects="write",
        )
        outcome = await bridge.call(Tool(spec, run), args, call_id=context.tool_call_id)
        return results.readable(outcome.output, tool.name)

    setattr(invoke, MADE, True)
    enabled = _narrowed(tool.name, tool.is_enabled)
    return dataclasses.replace(tool, on_invoke_tool=invoke, is_enabled=enabled)


def held(tool: Any, spec: ToolSpec) -> Any:
    """``tool`` (a ``FunctionTool``) held at the SDK's own approval gate (``needs_approval``)
    whenever governance asks a person about a call of it (``bridge.asks_first``: asked then,
    governance's question): the SDK pauses its run before the call and keeps its
    ``RunState``, and the resume continues it — the approved call runs, the model is not asked
    again. A tool with a gate of its own keeps it; anything else is returned as it is."""
    if not isinstance(tool, FunctionTool) or tool.needs_approval is not False:
        return tool

    async def asks(_context: Any, args: dict[str, Any], call_id: str) -> bool:
        if not await bridge.asks_first(spec, args):
            return False
        runtime = current()
        assert runtime is not None and runtime.pending is not None  # it asked: the run's pause
        if runtime.pending.native_id is None and runtime.pending.key == content_key(
            "approve", spec.name, args
        ):  # this call's: the SDK's handle on it, for the resume to approve
            runtime.pending = runtime.pending.model_copy(update={"native_id": call_id})
        return True

    return dataclasses.replace(tool, needs_approval=asks)


def _narrowed(name: str, own: Any) -> Any:
    """An agent's own tool enabled per turn as it was (``own``: a bool, or a sync or async
    callable) and only while the run offers it (``Runtime.offers``: the tool hints' narrowing
    reaches the framework's own tools too)."""

    async def enabled(context: Any, agent: Any) -> bool:
        if not _offered(name):
            return False
        if callable(own):
            given = own(context, agent)
            return bool(await given if inspect.isawaitable(given) else given)
        return bool(own)

    return enabled


def _helped(tool: FunctionTool, invoke: Any, helpers: Helpers) -> Any:
    """An agent tool's ``invoke`` as a helper of the run (``bridges.helpers``): started with the
    call's input, given the memory context for it before it, its tool calls recorded under its
    run while it works, ended with the agent's answer or its error, and what its own run spent
    — the SDK adds it to the calling run's usage — counted toward the run's budget. A pause
    inside leaves it to the next attempt, and stops the calling agent too (the SDK hands the
    agent tool's pause back as an error text)."""
    inner = getattr(tool, AGENT_OF_TOOL, None)
    model = getattr(inner, "model", None)

    async def helped(context: ToolContext[Any], arguments: str) -> Any:
        args = arguments_of(arguments)
        task = args.get("input") if isinstance(args, dict) else None
        key = context.tool_call_id
        helper = await helpers.start(
            key, kind=Kind.AGENT_TOOL, name=tool.name, task=str(task or arguments)
        )
        if isinstance(args, dict) and isinstance(task, str):
            arguments = json.dumps({**args, "input": helpers.briefed(helper, task)})
        before = _spent(context.usage)
        try:
            with helpers.working(helper):
                answer = await invoke(context, arguments)
        except Exception as exc:  # a pause leaves it open (``Helpers.end``: the run is pausing)
            await helpers.end(key, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            given, said, cached = (
                a - b for a, b in zip(_spent(context.usage), before, strict=True)
            )
            budget.account(
                helpers.runtime,
                model=model if isinstance(model, str) else None,
                input_tokens=given,
                output_tokens=said,
                cached=cached,
            )
        if helpers.runtime.pending is not None:  # paused inside: the SDK told it as an error text
            raise Paused(helpers.runtime.pending.interrupt)
        await helpers.end(key, output=text_of(answer))
        return answer

    return helped


def _spent(usage: Any) -> tuple[int, int, int]:
    """A run's usage so far: input, output and cached input tokens."""
    details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0)
    return usage.input_tokens, usage.output_tokens, cached


def _one(tool: Tool) -> FunctionTool:
    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = arguments_of(arguments)
        if isinstance(args, str):
            return not_run(tool.name, args)
        outcome = _repeat(context, arguments, args)
        if outcome is None:
            outcome = await bridge.call(tool, args, call_id=context.tool_call_id)
        return results.readable(outcome.output, tool.name)

    return FunctionTool(
        name=tool.name,
        description=tool.spec.description or tool.name,
        params_json_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
        on_invoke_tool=invoke,
        # the schema is the tool's own (MCP, OpenAPI...), not one written for strict mode
        strict_json_schema=False,
        is_enabled=lambda _context, _agent: _offered(tool.name),
    )


def _repeat(context: ToolContext[Any], arguments: str, args: dict[str, Any]) -> ToolOutcome | None:
    """The outcome of a call repeating an earlier one of its model reply (``Runtime.repeats``):
    not run (``bridge.same``); ``None`` for any other call."""
    runtime = current()
    key = (context.tool_call_id, context.tool_name, arguments)
    first = runtime.repeats.pop(key, None) if runtime is not None else None
    if runtime is None or first is None:
        return None
    return bridge.same(runtime, context.tool_name, args, call_id=context.tool_call_id, first=first)


def _reader(
    name: str,
    description: str,
    schema: dict[str, Any],
    answer: Callable[[dict[str, Any]], Awaitable[str]],
) -> FunctionTool:
    """A tool reading a large result the run keeps (``tools.results.readers``: ``read_file``
    its pages, ``query_result`` SQL over its rows — saving them as a file in a run with no tool
    that runs code), enabled once the run keeps one."""

    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = arguments_of(arguments)
        return not_run(name, args) if isinstance(args, str) else await answer(args)

    return FunctionTool(
        name=name,
        description=description,
        params_json_schema=schema,
        on_invoke_tool=invoke,
        strict_json_schema=False,
        is_enabled=lambda _context, _agent: results.kept() and _offered(name),
    )


def _offered(name: str) -> bool:
    """Whether the run offers ``name`` this turn: none after a stall (``bridge.stalled``), so
    the model's next reply answers."""
    runtime = current()
    return runtime is None or (runtime.offers(name) and not bridge.stalled(runtime))
