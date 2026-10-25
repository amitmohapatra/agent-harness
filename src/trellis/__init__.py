"""Trellis: attach memory, tools, approvals, durable runs and tracing to an agent you already
built.

    from trellis import Harness

    h = Harness()                                   # the deployment is the environment
    agent = h.wrap(ReAct("You reorder stock.", "gpt-4o", tools=[reorder]), id="procurement")
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
    from trellis.harness.agent import Agent, RunHandle
    from trellis.harness.budget import Budget
    from trellis.harness.evals import llm_judge
    from trellis.harness.harness import Harness
    from trellis.harness.hooks import Ask, Deny, Hooks, ModelCall, Output, OutputKind, Rewrite
    from trellis.harness.prompts import stored
    from trellis.harness.react import ReAct, middleware
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime, current
    from trellis.harness.sandbox import sandbox
    from trellis.harness.sandbox.backend import docker_sandbox
    from trellis.harness.settings import Settings
    from trellis.harness.tools.mcp import mcp, stdio
    from trellis.harness.tools.sources import a2a, openapi, tool

_EXPORTS = {
    "Harness": "trellis.harness.harness",
    "Hooks": "trellis.harness.hooks",
    "Deny": "trellis.harness.hooks",
    "Ask": "trellis.harness.hooks",
    "Rewrite": "trellis.harness.hooks",
    "ModelCall": "trellis.harness.hooks",
    "Output": "trellis.harness.hooks",
    "OutputKind": "trellis.harness.hooks",
    "tool": "trellis.harness.tools.sources",
    "a2a": "trellis.harness.tools.sources",
    "openapi": "trellis.harness.tools.sources",
    "mcp": "trellis.harness.tools.mcp",
    "stdio": "trellis.harness.tools.mcp",
    "stored": "trellis.harness.prompts",
    "llm_judge": "trellis.harness.evals",
    "sandbox": "trellis.harness.sandbox",
    "ReAct": "trellis.harness.react",
    "middleware": "trellis.harness.react",
    "docker_sandbox": "trellis.harness.sandbox.backend",
    "current": "trellis.harness.runtime",
    "Agent": "trellis.harness.agent",
    "RunHandle": "trellis.harness.agent",
    "Result": "trellis.harness.result",
    "Runtime": "trellis.harness.runtime",
    "Settings": "trellis.harness.settings",
    "Budget": "trellis.harness.budget",
}

__all__ = [
    "Agent",
    "Ask",
    "Budget",
    "Deny",
    "Harness",
    "Hooks",
    "ModelCall",
    "Output",
    "OutputKind",
    "ReAct",
    "Result",
    "Rewrite",
    "RunHandle",
    "Runtime",
    "Settings",
    "a2a",
    "current",
    "docker_sandbox",
    "llm_judge",
    "mcp",
    "middleware",
    "openapi",
    "sandbox",
    "stdio",
    "stored",
    "tool",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'trellis' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
