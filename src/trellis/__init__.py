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
    from trellis.harness.adapters.acp import kimi
    from trellis.harness.agent import Agent, RunHandle
    from trellis.harness.budget import Budget
    from trellis.harness.evals import llm_judge
    from trellis.harness.harness import Harness
    from trellis.harness.hooks import Ask, Deny, Hooks, Output, OutputKind, Rewrite
    from trellis.harness.prompts import stored
    from trellis.harness.react import ReAct, middleware
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime, current
    from trellis.harness.sandbox import sandbox
    from trellis.harness.settings import Settings
    from trellis.harness.skills import deep_agent
    from trellis.harness.tools.ask import ask_user
    from trellis.harness.tools.base import Citation
    from trellis.harness.tools.mcp import mcp, stdio
    from trellis.harness.tools.sources import a2a, openapi, tool

_EXPORTS = {
    "Harness": "trellis.harness.harness",
    "Hooks": "trellis.harness.hooks",
    "Deny": "trellis.harness.hooks",
    "Ask": "trellis.harness.hooks",
    "Rewrite": "trellis.harness.hooks",
    "Output": "trellis.harness.hooks",
    "OutputKind": "trellis.harness.hooks",
    "tool": "trellis.harness.tools.sources",
    "ask_user": "trellis.harness.tools.ask",
    "Citation": "trellis.harness.tools.base",
    "a2a": "trellis.harness.tools.sources",
    "openapi": "trellis.harness.tools.sources",
    "mcp": "trellis.harness.tools.mcp",
    "stdio": "trellis.harness.tools.mcp",
    "stored": "trellis.harness.prompts",
    "llm_judge": "trellis.harness.evals",
    "sandbox": "trellis.harness.sandbox",
    "ReAct": "trellis.harness.react",
    "kimi": "trellis.harness.adapters.acp",
    "middleware": "trellis.harness.react",
    "current": "trellis.harness.runtime",
    "Agent": "trellis.harness.agent",
    "RunHandle": "trellis.harness.agent",
    "Result": "trellis.harness.result",
    "Runtime": "trellis.harness.runtime",
    "Settings": "trellis.harness.settings",
    "Budget": "trellis.harness.budget",
    "deep_agent": "trellis.harness.skills",
}

#: Names ``trellis`` no longer has, and what to use instead (said when one is read).
REMOVED: dict[str, str] = {
    "docker_sandbox": "sandbox() is the one entry: ReAct(backend=sandbox()), "
    "create_deep_agent(**sandbox().deep_agent()), agent.tools.use(sandbox().tools())",
    "ModelCall": "the model hooks were removed: a model call is each framework's own "
    "(LangChain middleware, the OpenAI Agents SDK's AgentHooks, the Claude Agent SDK's hooks)",
}

__all__ = [
    "Agent",
    "Ask",
    "Budget",
    "Citation",
    "Deny",
    "Harness",
    "Hooks",
    "Output",
    "OutputKind",
    "ReAct",
    "Result",
    "Rewrite",
    "RunHandle",
    "Runtime",
    "Settings",
    "a2a",
    "ask_user",
    "current",
    "deep_agent",
    "kimi",
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
    if name in REMOVED:
        raise AttributeError(f"trellis.{name} was removed: {REMOVED[name]}")
    if module is None:
        raise AttributeError(f"module 'trellis' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
