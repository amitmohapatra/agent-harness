"""Harness tools as OpenAI Agents SDK ``FunctionTool``\\ s."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agents import FunctionTool
from agents.tool_context import ToolContext

from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of


def convert(tools: Sequence[Tool]) -> list[FunctionTool]:
    return [_one(t) for t in tools]


def _one(tool: Tool) -> FunctionTool:
    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        args = json.loads(arguments) if arguments else {}
        return text_of((await bridge.call(tool, args, call_id=context.tool_call_id)).output)

    return FunctionTool(
        name=tool.name,
        description=tool.spec.description or tool.name,
        params_json_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
        on_invoke_tool=invoke,
        # the schema is the tool's own (MCP, OpenAPI...), not one written for strict mode
        strict_json_schema=False,
    )
