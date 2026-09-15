"""The shipped multi-agent example is executed as a test, so the documentation cannot rot.

It also covers what no smaller test does: a fan-out/fan-in graph where three agents run
concurrently, one of them starting a nested agent run, with tool calls, model calls, an
artifact and a business decision.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


@pytest.fixture(scope="module")
def workflow():
    return importlib.import_module("reorder_workflow")


async def test_workflow_produces_the_expected_business_decision(workflow, spans):
    app = workflow.build_graph()
    out = await app.ainvoke(
        {"question": "Do we need to reorder SKU-1?", "trace": [], "risks": []},
        {"configurable": {"thread_id": "chat-test", "harness": {"tenant_id": "acme"}}},
    )
    decision = out["decision"]

    # position = 95 on hand + 40 on order + 40 from the open PO
    assert decision["position"] == 175
    # safety stock = 1.65 * 6.5 * sqrt(9); reorder point = 22 * 9 + safety stock
    assert decision["safety_stock"] == pytest.approx(32.2, abs=0.1)
    assert decision["reorder_point"] == pytest.approx(230.2, abs=0.1)
    # shortfall rounds up to a pack, then the supplier's minimum order quantity dominates
    assert decision["order_qty"] == 500
    assert decision["cost_usd"] == pytest.approx(6700.0)
    assert decision["budget_capped"] is False
    # 8 days of cover against a 9-day lead time: the stockout risk must fire
    assert decision["covers_lead_time"] is False
    assert any("runs out" in risk for risk in out["risks"])
    assert decision["placeable"] is True
    assert decision["purchase_order_ref"]


async def test_every_node_is_an_agent_run_and_the_middle_three_are_parallel(workflow, spans):
    from universal_agent_harness import AgentExecutionContext

    app = workflow.build_graph()
    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="reorder-workflow", thread_id="chat-parallel",
        turn_id="turn-1",
    )

    started = asyncio.get_running_loop().time()
    # Wrapping the invocation is what collapses the turn into one trace (see the example).
    async with workflow.harness.execution(context, agent_id="reorder-workflow", input="q"):
        out = await app.ainvoke(
            {"question": "Do we need to reorder SKU-1?", "trace": [], "risks": []},
            {"configurable": {"thread_id": "chat-parallel", "harness": {"tenant_id": "acme"}}},
        )
    elapsed = asyncio.get_running_loop().time() - started

    finished = spans.get_finished_spans()
    runs = [s for s in finished if s.name == "agent.run"]
    agent_ids = {s.attributes["agent.id"] for s in runs}
    assert agent_ids == {
        "reorder-workflow", "triage-agent", "inventory-agent", "supplier-risk-agent",
        "demand-agent", "supplier-agent", "decision-agent", "explain-agent",
    }

    # the nested agent records the node that started it as its parent run
    nested = next(s for s in runs if s.attributes["agent.id"] == "supplier-risk-agent")
    inventory = next(s for s in runs if s.attributes["agent.id"] == "inventory-agent")
    assert nested.attributes["agent.parent_run.id"] == inventory.attributes["agent.run.id"]

    # one trace for the whole turn, with the enclosing execution as its only root
    assert len({s.get_span_context().trace_id for s in finished}) == 1
    roots = [s for s in finished if s.parent is None]
    assert len(roots) == 1 and roots[0].attributes["agent.id"] == "reorder-workflow"

    # tools and models were instrumented, not bypassed
    assert [s.name for s in finished].count("agent.tool.call") >= 4
    assert [s.name for s in finished].count("agent.model.invoke") == 3

    # three ~20ms tool sleeps running concurrently, not 60ms+ of serialised work
    assert elapsed < 0.4
    assert set(out["trace"]) == {"triage", "inventory", "demand", "supplier", "decide", "explain"}


async def test_the_explain_node_produces_claims_evidence_and_observations(workflow, spans):
    """The node's state update stays a plain dict; the claims, evidence, recommended actions
    and memory observations ride on the AgentResult the state mapper was given."""
    results = {}

    def capture(event: str, payload: dict) -> None:
        if event == "on_agent_success" and payload["context"].agent_id == "explain-agent":
            results["result"] = payload["result"]

    workflow.harness.on(capture)

    app = workflow.build_graph()
    out = await app.ainvoke(
        {"question": "Do we need to reorder SKU-1?", "trace": [], "risks": []},
        {"configurable": {"thread_id": "chat-claims", "harness": {"tenant_id": "acme"}}},
    )

    assert out["answer"].startswith("Order 500 units of SKU-1")   # the state update is a dict
    result = results["result"]
    assert {c.claim_id for c in result.claims} == {"c-position", "c-reorder"}
    assert {e.source_id for e in result.evidence} == {"inventory_db", "demand_forecast"}
    assert [a.action_type for a in result.recommended_actions] == [
        "raise_purchase_order",
        "expedite_shipment",          # stock runs out before the lead time elapses
    ]
    assert result.memory_observations[0].content.startswith("SKU-1 was reordered")
    assert result.confidence == 0.86
    assert result.metrics["order_qty"] == 500.0
