"""Harness tools as one in-process MCP server for the Claude Agent SDK.

Claude names them ``mcp__trellis__<tool>``; ``allowed_names`` is what goes into the options'
``allowed_tools`` so the CLI calls them without its own permission prompt (the bridge is the
permission check).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of

#: The in-process server's name, and so the prefix Claude sees.
SERVER: Final = "trellis"


def convert(tools: Sequence[Tool]) -> McpSdkServerConfig:
    return create_sdk_mcp_server(SERVER, tools=[_one(t) for t in tools])


def allowed_names(tools: Sequence[Tool]) -> list[str]:
    return [f"mcp__{SERVER}__{t.name}" for t in tools]


def _one(tool: Tool) -> Any:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        outcome = await bridge.call(tool, dict(args))
        content = [{"type": "text", "text": text_of(outcome.output)}]
        return {"content": content, "is_error": not outcome.ok}

    return sdk_tool(
        tool.name,
        tool.spec.description or tool.name,
        tool.spec.input_schema or {"type": "object", "properties": {}},
    )(handler)
