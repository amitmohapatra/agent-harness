"""``CompositeToolClient``: several tool clients behind one ``ToolClient`` port.

Local functions, the gateway's MCP tools and memory's own tools are listed together and a
call is routed to the client that listed the tool; the first client to know a name wins,
so a local override of a remote tool is a matter of order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trellis.contracts.errors import ToolError
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec


class CompositeToolClient:
    name = "composite"

    def __init__(self, clients: Sequence[Any]) -> None:
        self.clients = list(clients)
        self._owners: dict[str, Any] = {}

    async def list_tools(self) -> Sequence[ToolSpec]:
        specs: list[ToolSpec] = []
        owners: dict[str, Any] = {}
        for client in self.clients:
            for spec in await client.list_tools():
                if spec.name in owners:
                    continue
                owners[spec.name] = client
                specs.append(spec)
        self._owners = owners
        return specs

    def spec(self, tool: str) -> ToolSpec | None:
        owner = self._owners.get(tool)
        getter = getattr(owner, "spec", None) if owner is not None else None
        spec = getter(tool) if callable(getter) else None
        return spec if isinstance(spec, ToolSpec) else None

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        owner = self._owners.get(call.tool)
        if owner is None:
            await self.list_tools()
            owner = self._owners.get(call.tool)
        if owner is None:
            raise ToolError(f"unknown tool {call.tool!r}", source="tools.composite")
        return await owner.call(call)
