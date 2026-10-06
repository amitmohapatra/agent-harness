"""Harness tools as one in-process MCP server for the Claude Agent SDK.

Claude names them ``mcp__trellis__<tool>``; the harness's ``can_use_tool`` lets them through
(the bridge is their permission check). A call that pauses the run tells the CLI it waits
(:data:`WAITING`): the run stops there, and the resumed session calls the tool again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from trellis.harness.runtime import Paused
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of

#: The in-process server's name, and so the prefix Claude sees.
SERVER: Final = "trellis"
#: What the CLI is told about a call that paused the run (a harness tool's, or a built-in's
#: the permission check asked a person about).
WAITING: Final = "Waiting for a person's approval."


def convert(tools: Sequence[Tool]) -> McpSdkServerConfig:
    return create_sdk_mcp_server(SERVER, tools=[_one(t) for t in tools])


def _one(tool: Tool) -> Any:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            outcome = await bridge.call(tool, dict(args))
        except Paused:  # the run stops here (the harness reads the pause from the runtime)
            return {"content": [{"type": "text", "text": WAITING}], "is_error": True}
        content = [{"type": "text", "text": text_of(outcome.output)}]
        return {"content": content, "is_error": not outcome.ok}

    return sdk_tool(
        tool.name,
        tool.spec.description or tool.name,
        tool.spec.input_schema or {"type": "object", "properties": {}},
    )(handler)
