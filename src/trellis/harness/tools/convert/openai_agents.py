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

import asyncio
import copy
import dataclasses
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, cast

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.contracts import AgentPaused, ToolOutcome, ToolSpec
from trellis.contracts.errors import is_pause_signal
from trellis.harness import budget
from trellis.harness.bridges.helpers import Helpers, Kind, text_of
from trellis.harness.journal import content_key
from trellis.harness.runtime import Paused, Runtime, current
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import FIX_ARGUMENTS, Tool, arguments_of, not_run

# what the model reads about arguments that are not an object (``tools.base``, as ``ReAct``)
__all__ = [
    "FIX_ARGUMENTS",
    "WaitsInside",
    "arguments_of",
    "asked_again",
    "convert",
    "governed",
    "held",
    "waits_inside",
]

#: The attribute marking the invoke function of a ``FunctionTool`` the harness made (its calls
#: run through the bridge already): kept by a copy of the tool (``dataclasses.replace``).
MADE: Final = "_trellis_made"
#: The attribute the SDK marks an agent called as a tool with (``Agent.as_tool``), and the one
#: holding that agent.
AGENT_TOOL: Final = "_is_agent_tool"
AGENT_OF_TOOL: Final = "_agent_instance"
#: The attribute marking a ``needs_approval`` policy :func:`held` made (governance's gate).
HELD: Final = "_trellis_held"
#: A ``FunctionTool``'s invoke: the call's context and its arguments as the model wrote them.
Invoke = Callable[[ToolContext[Any], str], Awaitable[Any]]
#: Where the invoke of a tool ``function_tool`` built keeps the tool's own function: under the
#: SDK's failure policy for it (the invoke turns what the function raises into what the model
#: reads, or raises it: ``failure_error_function``).
FUNCTION: Final = "_invoke_tool_impl"


class WaitsInside(AgentPaused):
    """An agent tool's call whose own run the SDK holds paused on an approval of its own (a
    ``needs_approval`` tool of the agent it runs): the call is not over — the SDK pauses the
    calling run on that approval and, approved or rejected, makes the call again to continue
    the run inside. To the bridge it is a pause (nothing journaled, nothing recorded: the
    call made again runs); to the SDK, ``answer`` is what the call returned."""

    def __init__(self, tool: str, answer: Any) -> None:
        super().__init__(f"{tool} waits on an approval inside the agent it runs")
        self.answer = answer


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    made = [*(_one(t) for t in tools), *(_reader(*r) for r in results.readers())]
    for tool in made:
        setattr(tool.on_invoke_tool, MADE, True)
    return made


def governed(tool: Any, helpers: Helpers | None = None) -> Any:
    """An agent's own ``FunctionTool`` (one not built with ``h.tools``), its calls run as the
    harness's are — governance (the catalog's risk; the SDK's tool has nowhere to say what it
    does, so without one each call asks), a person when it asks, the journal (a resume does
    not run a call again), its time and its trace — and offered per turn only while the run
    offers it (the tool hints' narrowing), as a copy: the
    agent's own tool is not changed. An agent called as a tool (``Agent.as_tool``)
    is, with ``helpers``, a native helper of the run for each call (:func:`_helped`). Anything
    else is returned as it is.

    What the tool's own function raises is the call's failure — recorded as one — and then
    the SDK's to tell, as without the harness (:func:`_told`): the model reads what the tool's
    ``failure_error_function`` makes of it (the SDK's default text when it names none), and
    with ``failure_error_function=None``, or a ``FunctionTool`` whose invoke raises, the run
    ends with the error. A call past the tool's own ``timeout`` is recorded as timed out
    (:func:`_own_time_is_up`) and is then the SDK's timeout (its text, or its
    ``ToolTimeoutError``). An agent tool's failure is an error its caller's model reads."""
    if not isinstance(tool, FunctionTool) or getattr(tool.on_invoke_tool, MADE, False):
        return tool
    original = tool.on_invoke_tool
    helps = getattr(tool, AGENT_TOOL, False)
    function = None if helps else getattr(original, FUNCTION, None)
    ran = cast("Invoke", function) if callable(function) else original
    if helpers is not None and helps:
        ran = _helped(tool, original, helpers)

    async def invoke(context: ToolContext[Any], arguments: str) -> Any:
        args = arguments_of(arguments)
        runtime = current()
        if isinstance(args, str) or runtime is None:
            return await original(context, arguments)
        repeat = _repeat(context, arguments, args)
        if repeat is not None:
            return results.readable(repeat.output, tool.name)
        raised: list[BaseException] = []
        began = time.monotonic()

        async def run(given: dict[str, Any]) -> Any:
            try:
                answer = await ran(context, json.dumps(given))
            except asyncio.CancelledError as cut:
                if not _own_time_is_up(tool, runtime, began):
                    raise
                raised.append(cut)
                raise PastItsOwnLimit(tool.name) from cut
            except Exception as error:
                if not helps and not is_pause_signal(error):
                    raised.append(error)
                raise
            if waits_inside(context):
                raise WaitsInside(tool.name, answer)
            return answer

        # as the run's tool set has it (an agent tool: a helper, read), else undeclared
        spec = runtime.native.get(tool.name) or ToolSpec(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=tool.params_json_schema,
        )
        try:
            outcome = await bridge.call(Tool(spec, run), args, call_id=context.tool_call_id)
        except WaitsInside as waiting:
            return waiting.answer
        if raised and not outcome.ok:
            return await _told(original, raised[-1], context, arguments)
        return results.readable(outcome.output, tool.name)

    setattr(invoke, MADE, True)
    enabled = _narrowed(tool.name, tool.is_enabled)
    return dataclasses.replace(tool, on_invoke_tool=invoke, is_enabled=enabled)


class PastItsOwnLimit(TimeoutError):
    """A call the SDK cut at its tool's own time limit (``function_tool(timeout=...)``): to
    the bridge a call out of time (recorded so, a write's effect unknown), not tried again —
    the SDK has stopped waiting for it."""

    retryable = False


def _own_time_is_up(tool: FunctionTool, runtime: Runtime, began: float) -> bool:
    """Whether a call cut short while it ran was cut by the SDK at the tool's own time limit
    (``function_tool(timeout=...)``: the SDK's clock for the call started before ``began``,
    so past the limit since then it has rung) and not with its run (cancelled, out of time)."""
    limit = tool.timeout_seconds
    return (
        limit is not None
        and runtime.cancelled is None
        and runtime.remaining() != 0
        and time.monotonic() - began >= limit
    )


async def _told(invoker: Any, error: BaseException, context: Any, arguments: str) -> Any:
    """What the SDK makes of ``error``, which a tool's own function raised (the harness has
    recorded the call as failed): raised as it is when the SDK is to act on it itself — a call
    it cut at the tool's time limit, or a tool whose invoke has no failure policy of the SDK's
    (the run ends) — else given to the tool's own invoke in place of its function, which tells
    the model what its ``failure_error_function`` says, or raises."""
    if not isinstance(error, Exception) or not callable(getattr(invoker, FUNCTION, None)):
        raise error

    async def failed(_context: Any, _arguments: str) -> Any:
        raise error

    failing = copy.copy(invoker)
    setattr(failing, FUNCTION, failed)
    return await failing(context, arguments)


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

    setattr(asks, HELD, True)
    return dataclasses.replace(tool, needs_approval=asks)


def waits_inside(context: ToolContext[Any]) -> bool:
    """Whether the SDK holds the run of the agent tool this call made (``Agent.as_tool``)
    paused on an approval of its own — a ``needs_approval`` tool inside it: what the SDK
    itself reads after the call to pause the calling run (its record of the call's pending
    agent run, ``agents.agent_tool_state``). Never so for any other tool's call."""
    from agents.agent_tool_state import get_agent_tool_state_scope, peek_agent_tool_run_result

    # as the SDK reads it: a call made with its ``ToolContext``, which names the call
    call = context.tool_call if isinstance(context, ToolContext) else None
    held = call is not None and peek_agent_tool_run_result(
        call, scope_id=get_agent_tool_state_scope(context)
    )
    return bool(getattr(held, "interruptions", None))


async def asked_again(item: Any, context: Any) -> bool | None:
    """Whether governance asks now about a call :func:`held` held in an earlier model reply
    (``item``: the SDK's ``ToolApprovalItem`` for it), asked as the gate asks (``True``: asked
    — governance's pause, with its choices; ``False``: it may run). The calls of one reply
    wait at the gate together, and a resumed run keeps each one's pending approval without
    asking the gate again, so the person's answer to the first (an approval for the
    conversation) would never reach the others. ``None``: a call this gate did not hold (your
    own gate's, the SDK's to ask), or one whose arguments it cannot read."""
    tool = next((t for t in item.agent.tools if getattr(t, "name", None) == item.name), None)
    gate = getattr(tool, "needs_approval", None)
    args = arguments_of(item.arguments)
    if not getattr(gate, HELD, False) or isinstance(args, str):
        return None
    assert gate is not None
    return bool(await gate(context, args, item.call_id))


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
    agent tool's pause back as an error text); so does an approval of the SDK's own inside it
    (:func:`waits_inside`): the helper stays open, and the call made again ends it."""
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
        if not waits_inside(context):
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
