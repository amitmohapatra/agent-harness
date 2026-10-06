"""Harness tools as LangChain ``BaseTool``\\ s (LangGraph, Deep Agents)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from langchain_core.tools import BaseTool, StructuredTool

from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of

#: The metadata key a tool's feature is kept under (``Tool.feature``: ``without=`` turns it
#: off), so the harness's middleware leaves the tools of a part a run is without unoffered.
FEATURE: Final = "trellis_feature"


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
        metadata=None if tool.feature is None else {FEATURE: tool.feature},
    )
