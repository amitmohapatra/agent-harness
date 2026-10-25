"""Test setup for what the public API leaves to an agent's builder.

* :func:`arm` gives a wrapped agent the tools the harness runs for it — what ``ReAct(tools=,
  subagents=)`` declares, and what a function target's code adds with ``agent.tools.use`` —
  whatever its target, so one test reads the same against every framework;
* :func:`native_tools` builds a framework's own tools from the harness's sources (functions,
  ``mcp(...)``, the virtual key's MCP tools, the memory tools), each call going through the
  harness's bridge — a graph built with them, armed with the same sources, is governed as a
  ``ReAct`` is;
* :func:`with_governance` makes ``governance`` the harness's for every tenant (a catalog in process,
  instead of the memory service's).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Final

from trellis import Harness
from trellis.harness.adapters import convert
from trellis.harness.agent import Agent
from trellis.harness.features import features
from trellis.harness.governance import Governance
from trellis.harness.tools.sources import as_source

#: The native tool format each framework's agents are built with.
FORMATS: Final = {
    "langgraph": "langchain",
    "deepagents": "langchain",
    "openai_agents": "openai_agents",
    "claude_agent_sdk": "claude",
    "claude": "claude",
}


def arm(agent: Agent, tools: Sequence[Any]) -> Agent:
    """``agent`` with ``tools`` as the tools the harness runs for it."""
    agent.sources = [as_source(t) for t in tools]
    return agent


def with_governance[H: Harness](harness: H, governance: Governance) -> H:
    """``harness`` with ``governance`` for every tenant."""
    harness._governance_given = governance
    return harness


async def resolved(h: Harness, *sources: Any, without: Collection[str] = ()) -> list[Any]:
    """The tools an agent with ``sources`` is given once: theirs and the MCP tools the virtual
    key allows (as Code Mode's, when on), less the parts ``without`` names."""
    off = features(without)
    box = h._toolbox([as_source(s) for s in sources], tenant=await h._tenant(), without=off)
    return [t for t in await box.tools(code_mode="code_mode" not in off) if t.feature not in off]


async def native_tools(
    h: Harness, *sources: Any, framework: str, without: Collection[str] = ()
) -> Any:
    """``sources``, the MCP tools the virtual key allows and — memory on — the memory tools,
    as ``framework``'s own tools (one in-process MCP server for Claude)."""
    off = features(without)
    tenant = await h._tenant()
    tools = await resolved(h, *sources, without=without)
    if h.memory is not None and "memory_pull" not in off:
        scope = h.memory.scoped(tenant)
        tools.extend(t for t in await h._pull_tools(scope) if t.feature not in off)
    return convert(FORMATS[framework], tools)  # type: ignore[arg-type]
