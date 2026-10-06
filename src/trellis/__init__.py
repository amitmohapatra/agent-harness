"""Trellis: attach memory, tools, approvals, durable runs and tracing to an agent you already
built.

    from trellis import Harness

    h = Harness()                                   # the deployment is the environment
    agent = h.wrap(graph, id="procurement")
    result = await agent.run("reorder SKU-1", user="u1")

``trellis`` is shared with the other trellis distributions (``trellis.contracts``,
``trellis.memory``, ``trellis.runs``): ``__path__`` is extended so they import beside this
package, and the names below load on first use, so ``import trellis.contracts`` does not load
the harness.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

if TYPE_CHECKING:
    from trellis.harness.adapters.react import ReAct
    from trellis.harness.agent import Agent, RunHandle
    from trellis.harness.harness import Harness
    from trellis.harness.hooks import Ask, Deny, Hooks, ModelCall, Rewrite
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime, current
    from trellis.harness.sandbox import sandbox
    from trellis.harness.settings import Settings
    from trellis.harness.skills import skills
    from trellis.harness.tools.sources import a2a, openapi, tool

_EXPORTS = {
    "Harness": "trellis.harness.harness",
    "Hooks": "trellis.harness.hooks",
    "Deny": "trellis.harness.hooks",
    "Ask": "trellis.harness.hooks",
    "Rewrite": "trellis.harness.hooks",
    "ModelCall": "trellis.harness.hooks",
    "tool": "trellis.harness.tools.sources",
    "a2a": "trellis.harness.tools.sources",
    "openapi": "trellis.harness.tools.sources",
    "skills": "trellis.harness.skills",
    "sandbox": "trellis.harness.sandbox",
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
    "Ask",
    "Deny",
    "Harness",
    "Hooks",
    "ModelCall",
    "ReAct",
    "Result",
    "Rewrite",
    "RunHandle",
    "Runtime",
    "Settings",
    "a2a",
    "current",
    "openapi",
    "sandbox",
    "skills",
    "tool",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'trellis' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
