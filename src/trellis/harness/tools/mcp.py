"""``MCPToolClient``: the gateway's MCP tools as a ``ToolClient`` (design §6, mode B).

Bifrost discovers each registered MCP server's tools and filters them per virtual key; it
deliberately does not run them, so a policy check, an audit record or a human approval can
sit in front of a tool that sends email. This client lists those tools as ``ToolSpec``s and
executes one through ``bf.mcp.execute``, so the harness's instrumented client, its policy
and its tool memory apply to a remote tool exactly as they apply to a local one.

The listing is what ``GET /api/mcp/clients`` returns: one entry per server, its ``config``
(``name``), its ``tools`` in the chat-completions function shape (``name``,
``description``, ``parameters``) and its ``state``. The gateway has already named every
tool ``<server>-<tool>`` and routes an execution by that prefix, so the names are used
exactly as listed, never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from trellis.contracts import ToolStatus
from trellis.contracts.errors import ToolError
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec

from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)

SOURCE: Final = "mcp"
#: The gateway's prefix between a server's name and a tool's own name.
PREFIX_SEPARATOR: Final = "-"


class MCPToolClient:
    """Talks to one gateway; the virtual key the gateway client carries decides the tools."""

    name = "mcp"

    def __init__(self, gateway: Any, *, clients: Sequence[str] | None = None) -> None:
        """``gateway`` is a ``bifrost_sdk.Bifrost``, or the harness's ``BifrostModelClient``,
        whose gateway client is reused so inference and tool execution share one virtual
        key, one retry policy and one circuit breaker; ``clients`` restricts the listing to
        these MCP servers (by their registered names)."""
        self._gateway = getattr(gateway, "gateway", None) or gateway
        if not hasattr(self._gateway, "mcp"):
            raise TypeError("MCPToolClient needs a Bifrost client (or a BifrostModelClient)")
        self._clients = tuple(clients) if clients else None
        self._specs: dict[str, ToolSpec] = {}

    async def list_tools(self) -> Sequence[ToolSpec]:
        servers = await self._gateway.mcp.clients()
        specs: dict[str, ToolSpec] = {}
        for server in servers:
            server_name = _server_name(server)
            if self._clients is not None and server_name not in self._clients:
                continue
            for tool in _tools_of(server):
                spec = _spec(server_name, tool)
                if spec is not None:
                    specs[spec.name] = spec
        self._specs = specs
        return list(specs.values())

    def spec(self, tool: str) -> ToolSpec | None:
        return self._specs.get(tool)

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        request = {
            "id": call.idempotency_key or f"call_{call.tool}",
            "type": "function",
            "function": {"name": call.tool, "arguments": json.dumps(call.args)},
        }
        try:
            turn = await self._gateway.mcp.execute(request)
        except Exception as exc:
            raise ToolError(str(exc), source=f"mcp.{call.tool}") from exc
        return _outcome(call, turn)


def _server_name(server: dict[str, Any]) -> str:
    config = server.get("config")
    if isinstance(config, dict) and config.get("name"):
        return str(config["name"])
    return str(server.get("name") or "")


def _tools_of(server: dict[str, Any]) -> list[dict[str, Any]]:
    tools = server.get("tools")
    return [t for t in (tools or []) if isinstance(t, dict)]


def _spec(server: str, tool: dict[str, Any]) -> ToolSpec | None:
    """The listed name, as the gateway routes it; the server is the prefix it carries."""
    name = str(tool.get("name") or "")
    if not name:
        return None
    owner = server if name.startswith(server + PREFIX_SEPARATOR) else None
    return ToolSpec(
        name=name,
        description=str(tool.get("description") or ""),
        input_schema=tool.get("parameters") or tool.get("input_schema"),
        source=SOURCE,
        server=owner,
    )


def _outcome(call: ToolCall, turn: dict[str, Any]) -> ToolOutcome:
    """The ``{"role": "tool", ...}`` turn the gateway returns, as an outcome."""
    content = turn.get("content")
    output: Any = content
    if isinstance(content, str):
        try:
            output = json.loads(content)
        except ValueError:
            output = content
    if turn.get("error") or turn.get("is_error") or turn.get("isError"):
        return ToolOutcome(
            tool=call.tool, status=ToolStatus.ERROR, output=output, error_class="MCPToolError"
        )
    return ToolOutcome(tool=call.tool, status=ToolStatus.OK, output=output)
