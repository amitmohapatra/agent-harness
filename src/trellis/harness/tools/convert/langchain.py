"""Harness tools as LangChain ``BaseTool``\\ s (LangGraph, Deep Agents)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool

from trellis.contracts import ConfigurationError
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool, ToolStatus, node_retries
from trellis.harness.tools.convert import text_of

#: The metadata key a tool's feature is kept under (``Tool.feature``: ``without=`` turns it
#: off), so the harness's middleware leaves the tools of a part a run is without unoffered.
FEATURE: Final = "trellis_feature"
#: The metadata key marking a tool the harness made (its calls go through the bridge already):
#: a framework's own tool has none, and is governed as one (``middleware.governed_call``).
MADE: Final = "trellis_made"


def convert(tools: Sequence[Tool]) -> list[BaseTool]:
    return [_one(t) for t in tools]


def made(tool: Any) -> bool:
    """Whether ``tool`` is one the harness made (:data:`MADE`)."""
    return bool((getattr(tool, "metadata", None) or {}).get(MADE))


def _one(tool: Tool) -> BaseTool:
    async def run(**args: Any) -> str:
        config = ensure_config()

        def retries(error: Exception) -> bool:
            from trellis.harness.middleware import (
                graph_retries,
            )  # (here: middleware imports this module)

            try:
                return graph_retries(config, error)
            except ConfigurationError:  # LangGraph's internals moved: the harness retries
                return False

        # inside a node whose own ``RetryPolicy`` runs it again on the error, a failed read
        # raises it once the call has ended and been recorded: the node's policy tries it
        # again, and the harness does not on top (``tools.base.NodeRetries``)
        with node_retries(retries) as node:
            outcome = await bridge.call(tool, args)
        failed = outcome.status is ToolStatus.ERROR and not bridge.hooked(outcome)
        if node.error is not None and failed:
            raise node.error
        return text_of(outcome.output)

    return StructuredTool.from_function(
        coroutine=run,
        name=tool.name,
        description=tool.spec.description or tool.name,
        args_schema=tool.spec.input_schema or {"type": "object", "properties": {}},
        metadata={MADE: True} | ({} if tool.feature is None else {FEATURE: tool.feature}),
    )
