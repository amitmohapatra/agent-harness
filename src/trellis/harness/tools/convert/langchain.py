"""Harness tools as LangChain ``BaseTool``\\ s (LangGraph, Deep Agents)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of


def convert(tools: Sequence[Tool]) -> list[BaseTool]:
    return [_one(t) for t in tools]


def _one(tool: Tool) -> BaseTool:
    async def run(**args: Any) -> str:
        return text_of((await bridge.call(tool, args)).output)

    return StructuredTool.from_function(
        coroutine=run,
        name=tool.name,
        description=tool.spec.description or tool.name,
        args_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
    )
