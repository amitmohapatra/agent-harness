"""Harness tools as one in-process MCP server for the Claude Agent SDK.

Claude names them ``mcp__trellis__<tool>``; the harness's ``can_use_tool`` lets them through
(the bridge is their permission check). A call that pauses the run tells the CLI it waits
(:data:`WAITING`): the run stops there, and the resumed session calls the tool again.

A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; the server's ``read_file`` pages it. The CLI lists a server's tools once,
when its session starts (the SDK carries no ``list_changed``), so ``read_file`` is listed from
the start: before a result was kept it finds nothing to read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from trellis.harness.runtime import Paused
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import Tool

#: The in-process server's name, and so the prefix Claude sees.
SERVER: Final = "trellis"
#: What the CLI is told about a call that paused the run (a harness tool's, or a built-in's
#: the permission check asked a person about).
WAITING: Final = "Waiting for a person's approval."


def convert(tools: Sequence[Tool]) -> McpSdkServerConfig:
    return create_sdk_mcp_server(SERVER, tools=[*(_one(t) for t in tools), _read_file()])


def _one(tool: Tool) -> Any:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            outcome = await bridge.call(tool, dict(args))
        except Paused:  # the run stops here (the harness reads the pause from the runtime)
            return {"content": [{"type": "text", "text": WAITING}], "is_error": True}
        content = [{"type": "text", "text": results.readable(outcome.output)}]
        return {"content": content, "is_error": not outcome.ok}

    return sdk_tool(tool.name, tool.spec.description or tool.name, _schema(tool))(handler)


def _read_file() -> Any:
    """``read_file``: a page of a large result the run keeps (``tools.results``)."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": results.read(dict(args))}]}

    return sdk_tool(results.READ_FILE, results.READ_FILE_DESCRIPTION, results.READ_FILE_SCHEMA)(
        handler
    )


def _schema(tool: Tool) -> dict[str, Any]:
    """The tool's input schema as the SDK takes a JSON schema: with a ``type`` and
    ``properties`` — any other dict it reads as a map of argument names to Python types, and
    an MCP tool without arguments declares only ``{"type": "object"}``."""
    schema = tool.spec.input_schema or {}
    return {**schema, "type": "object", "properties": schema.get("properties") or {}}
