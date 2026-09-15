"""``LocalToolClient``: the in-process tool registry (§14).

Registration is explicit — the harness never discovers arbitrary callables — and lookup is
an O(1) dict hit (§64). An MCP or gateway-backed client implements the same port; nothing
in the core knows the difference.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from universal_agent_harness.contracts.errors import ToolNotFoundError
from universal_agent_harness.contracts.tool import ToolCall, ToolOutcome, ToolSpec


class LocalToolClient:
    """Tools that are plain Python callables in this process."""

    name = "local"

    def __init__(self, tools: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._callables: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}
        for tool_name, fn in (tools or {}).items():
            self.register(fn, name=tool_name)

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        spec: ToolSpec | None = None,
        **spec_fields: Any,
    ) -> ToolSpec:
        """Register a callable. Returns the resulting spec (derived from the signature)."""
        tool_name = name or spec.name if spec else name or getattr(fn, "__name__", "tool")
        resolved = spec or ToolSpec(
            name=tool_name,
            description=(inspect.getdoc(fn) or "").split("\n\n")[0],
            input_schema=_schema(fn),
            **spec_fields,
        )
        self._callables[resolved.name] = fn
        self._specs[resolved.name] = resolved
        return resolved

    def spec(self, tool: str) -> ToolSpec | None:
        return self._specs.get(tool)

    async def list_tools(self) -> Sequence[ToolSpec]:
        return list(self._specs.values())

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        fn = self._callables.get(call.tool)
        if fn is None:
            raise ToolNotFoundError(
                f"tool {call.tool!r} is not registered "
                f"(known: {', '.join(sorted(self._callables)) or 'none'})",
                source="tools.local",
            )
        result = fn(**call.args)
        if inspect.isawaitable(result):
            result = await result
        return ToolOutcome(tool=call.tool, status="ok", output=result)


class CallableToolClient:
    """Adapts a single ``async def call(tool, args)`` function to the port. Useful for MCP
    gateways and for tests."""

    name = "callable"

    def __init__(
        self, executor: Callable[[str, dict[str, Any]], Any], specs: Sequence[ToolSpec] = ()
    ) -> None:
        self._executor = executor
        self._specs = list(specs)

    async def list_tools(self) -> Sequence[ToolSpec]:
        return list(self._specs)

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        result = self._executor(call.tool, dict(call.args))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome(tool=call.tool, status="ok", output=result)


class NoToolsClient:
    """The default when no tool runtime is configured."""

    name = "none"

    async def list_tools(self) -> Sequence[ToolSpec]:
        return []

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        name = tool.tool if isinstance(tool, ToolCall) else tool
        raise ToolNotFoundError(
            f"no tool runtime is configured, so {name!r} cannot be called; "
            "pass tools=... to AgentHarness(...) or wrap the tool with harness.wrap_tool()",
            source="tools",
        )


def _schema(fn: Callable[..., Any]) -> dict[str, Any] | None:
    """A minimal JSON-schema-ish description from the signature. Types are best effort;
    the point is recording *which* arguments exist, never their values."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param.name] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    if not properties:
        return None
    return {"type": "object", "properties": properties, "required": required}


_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    default = "string" if annotation is inspect.Parameter.empty else "object"
    return _JSON_TYPES.get(annotation, default)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    if asyncio.iscoroutine(value):  # pragma: no cover - defensive
        return await value
    return value
