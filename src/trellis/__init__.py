"""Trellis: attach memory, tools, approvals, durable runs and evaluation to an agent you
already built.

    from trellis import Harness, mcp, tool, a2a, openapi, ReAct

    h = Harness()
    agent = h.wrap(graph, id="procurement", tools=[mcp("erp")], memory="read_write")
    result = await agent.run("reorder SKU-1", user="u1")

``trellis`` is shared with the other trellis distributions (``trellis.contracts``,
``trellis.memory``): ``__path__`` is extended so they import beside this package, and the
names below load on first use, so ``import trellis.contracts`` does not load the harness.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

if TYPE_CHECKING:
    from trellis.harness.adapters.react import ReAct
    from trellis.harness.agent import Agent, RunHandle
    from trellis.harness.harness import Harness
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime, current
    from trellis.harness.settings import Settings
    from trellis.harness.tools.sources import a2a, mcp, openapi, tool

_EXPORTS = {
    "Harness": "trellis.harness.harness",
    "mcp": "trellis.harness.tools.sources",
    "tool": "trellis.harness.tools.sources",
    "a2a": "trellis.harness.tools.sources",
    "openapi": "trellis.harness.tools.sources",
    "ReAct": "trellis.harness.adapters.react",
    "current": "trellis.harness.runtime",
    "Agent": "trellis.harness.agent",
    "RunHandle": "trellis.harness.agent",
    "Result": "trellis.harness.result",
    "Runtime": "trellis.harness.runtime",
    "Settings": "trellis.harness.settings",
}

__all__ = [
    "Agent",
    "Harness",
    "ReAct",
    "Result",
    "RunHandle",
    "Runtime",
    "Settings",
    "a2a",
    "current",
    "mcp",
    "openapi",
    "tool",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'trellis' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
