"""LangGraph adapter for the Universal Agent Harness.

    from universal_agent_harness import AgentHarness

    harness = AgentHarness(memory=memory_client, defaults={"tenant_id": "acme"})
    graph.add_node("inventory", harness.langgraph.wrap_node(node, agent_id="inventory-agent"))

Installing this package is what makes ``harness.langgraph`` available; the harness core
never imports LangGraph.
"""

from universal_agent_harness_langgraph.adapter import LangGraphHarness, langgraph_version
from universal_agent_harness_langgraph.lineage import (
    Lineage,
    Segment,
    context_fields,
    lineage_from_config,
)

__version__ = "0.1.0"

__all__ = [
    "LangGraphHarness",
    "Lineage",
    "Segment",
    "context_fields",
    "langgraph_version",
    "lineage_from_config",
]
