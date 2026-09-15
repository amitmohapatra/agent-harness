"""A real multi-agent workflow: LangGraph + Memory Service + Langfuse, end to end.

    python examples/reorder_workflow.py                    # runs standalone, no services
    MEMORY_SERVICE_URL=http://localhost:8080 \
    LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... \
        python examples/reorder_workflow.py                # same code, services attached

The scenario is a supply-chain reorder decision for one SKU:

    reorder-workflow                      <- harness.execution(): one trace for the turn
      └── triage ──┬── inventory  (tool + model, starts a nested agent)  ┐
                   ├── demand     (tool + business logic)                ├─ parallel
                   └── supplier   (tool)                                 ┘
                            │
                         decide  (pure business logic, claims, artifact)
                            │
                         explain (model, claims + observations -> memory)

What each part of the harness is doing here:

* every node is an agent run with its own span, nested under one trace for the turn;
* the three middle nodes run **concurrently** — the harness does not serialise them;
* `inventory` also calls a **nested sub-agent** (`supplier-risk-agent`), so the trace shows
  a second level of agent lineage;
* tools are instrumented two ways: registered on the harness (`runtime.tools.call`) and
  wrapped in place (`@harness.wrap_tool`) for a function called directly;
* model calls are instrumented through `runtime.model`, including token and cost metrics;
* `decide` contains the actual business logic and returns claims, evidence, a purchase-order
  artifact and recommended actions;
* `explain` writes observations back to memory — after the result is produced, never before;
* Langfuse, if configured, receives the same span tree as the OTLP backend.
"""

from __future__ import annotations

import asyncio
import math
import operator
import os
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResult,
    AgentRuntime,
    Claim,
    EvidenceRef,
    MemoryObservation,
    RecommendedAction,
)

# --------------------------------------------------------------------------- policy knobs

SERVICE_LEVEL_Z = 1.65      # 95% service level
LEAD_TIME_RISK_DAYS = 21    # above this, a supplier counts as slow
ORDER_BUDGET_USD = 10_000.0  # per-order spending cap


# --------------------------------------------------------------------------- graph state


class OrderState(TypedDict, total=False):
    question: str
    sku: str
    urgency: str
    inventory: dict[str, Any]
    demand: dict[str, Any]
    supplier: dict[str, Any]
    risks: Annotated[list[str], operator.add]
    decision: dict[str, Any]
    answer: str
    trace: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------- fake backends
# Stand-ins for the systems this workflow would really talk to. Swapping in the real thing
# changes these functions only — no harness or agent code moves.


async def inventory_db(sku: str) -> dict[str, Any]:
    """On-hand and on-order stock for a SKU."""
    await asyncio.sleep(0.02)
    return {"sku": sku, "on_hand": 95, "on_order": 40, "unit_cost": 12.5, "warehouse": "EU-1"}


async def demand_forecast(sku: str, horizon_days: int = 30) -> dict[str, Any]:
    """Forecast daily demand and its volatility."""
    await asyncio.sleep(0.02)
    return {"sku": sku, "avg_daily": 22.0, "stddev_daily": 6.5, "horizon_days": horizon_days}


async def supplier_catalog(sku: str) -> dict[str, Any]:
    """Contracted suppliers, lead times and minimum order quantities."""
    await asyncio.sleep(0.02)
    return {
        "sku": sku,
        "suppliers": [
            {"name": "Meridian Parts", "lead_time_days": 24, "moq": 250, "pack_size": 50,
             "unit_price": 12.10},
            {"name": "Castor Supply", "lead_time_days": 9, "moq": 500, "pack_size": 100,
             "unit_price": 13.40},
        ],
    }


class DemoModel:
    """A deterministic stand-in for a provider client.

    Any object with ``ainvoke`` works, so replacing this with a real client is one line in
    :func:`build_harness`. It reports usage the way providers do, which is what lets the
    harness record tokens and cost.
    """

    async def ainvoke(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        text = _canned_reply(str(prompt))
        return {
            "text": text,
            "model": "demo-model-1",
            "usage": {
                "prompt_tokens": max(1, len(str(prompt)) // 4),
                "completion_tokens": max(1, len(text) // 4),
                "cost_usd": 0.00002 * len(text),
            },
        }


def _canned_reply(prompt: str) -> str:
    lowered = prompt.lower()
    if "classify" in lowered:
        return "replenishment request, urgency=high, sku=SKU-1"
    if "summarise the stock" in lowered:
        return "Stock is below the reorder point once the 24-day lead time is accounted for."
    return "Reorder now from the faster supplier; the slower one cannot cover the gap."


# --------------------------------------------------------------------------- harness setup


def build_harness() -> AgentHarness:
    memory = None
    if url := os.environ.get("MEMORY_SERVICE_URL"):
        from universal_memory import MemoryClient  # noqa: PLC0415 - optional in this example

        memory = MemoryClient(url, api_key=os.environ.get("MEMORY_API_KEY"))

    langfuse_on = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )

    return AgentHarness(
        memory=memory,                       # None -> memory degrades to a no-op
        model=DemoModel(),                   # swap for your provider client
        tools=[inventory_db, demand_forecast, supplier_catalog],
        defaults={
            "tenant_id": os.environ.get("MEMORY_TENANT", "acme"),
            "user_id": "planner-7",
            "agent_group_id": "supply-chain",
        },
        config={
            "memory": {
                "observe_claims": True,
                "observe_tool_results": False,   # tool payloads stay out of memory
            },
            "observability": {
                "langfuse": {
                    "enabled": langfuse_on,      # keys come from LANGFUSE_* env vars
                    "environment": os.environ.get("APP_ENV", "local"),
                }
            },
            "evaluation_events": {"enabled": True},
            "timeouts": {"default_seconds": 30.0, "model_seconds": 15.0, "tool_seconds": 10.0},
        },
    )


harness = build_harness()


# A tool called directly rather than through ``runtime.tools`` — wrapping it is what makes
# that call visible to the harness (§13: nothing else can intercept an arbitrary call).
@harness.wrap_tool(idempotent=True)
async def open_purchase_orders(sku: str) -> list[dict[str, Any]]:
    """Purchase orders already raised for this SKU."""
    await asyncio.sleep(0.01)
    return [{"po": "PO-8841", "qty": 40, "eta_days": 12}]


# --------------------------------------------------------------------------- nodes


@harness.langgraph.agent(agent_id="triage-agent", skills=["supply.triage"], query="question")
async def triage(state: OrderState, agent: AgentRuntime) -> dict[str, Any]:
    """Classify the request and pull the SKU out of it (model call)."""
    reply = await agent.model.invoke(f"Classify this request: {state['question']}")
    sku = _extract_sku(reply.text) or "SKU-1"
    urgency = "high" if "urgency=high" in (reply.text or "") else "normal"
    agent.log("triage.done", sku=sku, urgency=urgency)
    return {"sku": sku, "urgency": urgency, "trace": ["triage"]}


@harness.langgraph.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
async def inventory(state: OrderState, agent: AgentRuntime) -> dict[str, Any]:
    """Stock position, plus a nested sub-agent that judges supplier concentration risk."""
    stock = (await agent.tools.call("inventory_db", sku=state["sku"])).output
    pending = await open_purchase_orders(sku=state["sku"])   # wrapped tool, direct call
    stock["on_order"] = stock["on_order"] + sum(po["qty"] for po in pending)

    await agent.model.invoke(f"Summarise the stock position: {stock['on_hand']} on hand")

    # A nested agent run: its spans and memory writes carry this run as their parent.
    risk = await supplier_risk({"sku": state["sku"]})

    return {"inventory": stock, "risks": risk.data["risks"], "trace": ["inventory"]}


@harness.agent(agent_id="supplier-risk-agent", skills=["supply.risk"])
async def supplier_risk(state: dict[str, Any], agent: AgentRuntime) -> AgentResult:
    """A plain (non-node) agent, called from inside a node."""
    catalog = (await agent.tools.call("supplier_catalog", sku=state["sku"])).output
    risks: list[str] = []
    if len(catalog["suppliers"]) < 2:
        risks.append("single-source exposure")
    if all(s["lead_time_days"] > LEAD_TIME_RISK_DAYS for s in catalog["suppliers"]):
        risks.append(f"every supplier exceeds the {LEAD_TIME_RISK_DAYS}-day lead-time limit")
    return AgentResult.ok({"risks": risks})


@harness.langgraph.agent(agent_id="demand-agent", skills=["demand.forecast"])
async def demand(state: OrderState, agent: AgentRuntime) -> dict[str, Any]:
    """Forecast, then derive safety stock and the reorder point (business logic)."""
    forecast = (await agent.tools.call("demand_forecast", sku=state["sku"], horizon_days=30)).output
    return {"demand": forecast, "trace": ["demand"]}


@harness.langgraph.agent(agent_id="supplier-agent", skills=["supply.sourcing"])
async def supplier(state: OrderState, agent: AgentRuntime) -> dict[str, Any]:
    """Pick the supplier that can actually cover the gap in time."""
    catalog = (await agent.tools.call("supplier_catalog", sku=state["sku"])).output
    chosen = min(catalog["suppliers"], key=lambda s: (s["lead_time_days"], s["unit_price"]))
    return {"supplier": {"catalog": catalog["suppliers"], "chosen": chosen},
            "trace": ["supplier"]}


@harness.langgraph.agent(agent_id="decision-agent", skills=["inventory.reorder"])
async def decide(state: OrderState, agent: AgentRuntime) -> dict[str, Any]:
    """The business logic. No model call: this is arithmetic and policy, and it should be
    auditable rather than generated."""
    stock, forecast = state["inventory"], state["demand"]
    chosen = state["supplier"]["chosen"]

    lead_time = chosen["lead_time_days"]
    safety_stock = SERVICE_LEVEL_Z * forecast["stddev_daily"] * math.sqrt(lead_time)
    reorder_point = forecast["avg_daily"] * lead_time + safety_stock
    position = stock["on_hand"] + stock["on_order"]
    shortfall = max(0.0, reorder_point - position)

    # Round up to the supplier's pack size, respect the minimum order quantity.
    packs = math.ceil(shortfall / chosen["pack_size"]) if shortfall else 0
    order_qty = max(packs * chosen["pack_size"], chosen["moq"]) if shortfall else 0
    cost = order_qty * chosen["unit_price"]

    capped = False
    if cost > ORDER_BUDGET_USD:          # budget policy: never exceed; order whole packs
        affordable = int(ORDER_BUDGET_USD // chosen["unit_price"])
        order_qty = (affordable // chosen["pack_size"]) * chosen["pack_size"]
        cost, capped = order_qty * chosen["unit_price"], True

    days_of_cover = position / forecast["avg_daily"] if forecast["avg_daily"] else math.inf

    # Policy checks that decide whether this order is actually placeable.
    blockers: list[str] = []
    if capped and order_qty < chosen["moq"]:
        blockers.append(
            f"budget caps the order at {order_qty} units, below {chosen['name']}'s "
            f"minimum of {chosen['moq']}"
        )
    covers_lead_time = days_of_cover >= lead_time
    if not covers_lead_time:
        blockers.append(
            f"stock runs out in {days_of_cover:.1f} days, before the "
            f"{lead_time}-day lead time"
        )

    decision = {
        "sku": state["sku"],
        "supplier": chosen["name"],
        "order_qty": order_qty,
        "cost_usd": round(cost, 2),
        "lead_time_days": lead_time,
        "reorder_point": round(reorder_point, 1),
        "safety_stock": round(safety_stock, 1),
        "position": position,
        "days_of_cover": round(days_of_cover, 1),
        "budget_capped": capped,
        "covers_lead_time": covers_lead_time,
        "blockers": blockers,
        "placeable": bool(order_qty) and not (capped and order_qty < chosen["moq"]),
    }

    # A purchase-order draft is a document, not a state field: it becomes an artifact and
    # the state keeps only the reference.
    po = await agent.artifacts.put(
        _render_po(decision), type="purchase_order", mime_type="text/plain",
        metadata={"sku": decision["sku"], "supplier": decision["supplier"]},
    )

    agent.log("decision.made", order_qty=order_qty, cost_usd=decision["cost_usd"])
    return {
        "decision": {**decision, "purchase_order_ref": po.artifact_id},
        "risks": blockers,
        "trace": ["decide"],
    }


def _explain_state(result: AgentResult) -> dict[str, Any]:
    """Map the rich result onto the graph's state. The application owns its state shape;
    the harness keeps the claims, evidence and observations on the result."""
    return {"answer": result.data["answer"], "trace": ["explain"]}


@harness.langgraph.agent(
    agent_id="explain-agent",
    skills=["supply.explain"],
    query="question",
    state_mapper=_explain_state,
)
async def explain(state: OrderState, agent: AgentRuntime) -> AgentResult:
    """Write the answer, and hand the durable facts to memory as claims and observations."""
    decision = state["decision"]
    reply = await agent.model.invoke(
        f"Explain the reorder decision for {decision['sku']} in two sentences."
    )
    answer = (
        f"Order {decision['order_qty']} units of {decision['sku']} from "
        f"{decision['supplier']} (${decision['cost_usd']:,.2f}, {decision['lead_time_days']}-day "
        f"lead time). {reply.text}"
    )

    evidence = [
        EvidenceRef(source_type="tool", source_id="inventory_db", citation="[1]"),
        EvidenceRef(source_type="tool", source_id="demand_forecast", citation="[2]"),
    ]
    return AgentResult.ok(
        {"answer": answer},
        claims=[
            Claim(claim_id="c-position", confidence=1.0, evidence_ids=["inventory_db"],
                  text=f"{decision['sku']} has {decision['position']} units of cover "
                       f"({decision['days_of_cover']} days)"),
            Claim(claim_id="c-reorder", confidence=0.9, evidence_ids=["demand_forecast"],
                  text=f"{decision['sku']} reorder point is {decision['reorder_point']} units"),
        ],
        evidence=evidence,
        recommended_actions=_actions(decision),
        memory_observations=[
            MemoryObservation(
                content=(f"{decision['sku']} was reordered from {decision['supplier']}: "
                         f"{decision['order_qty']} units at ${decision['cost_usd']:,.2f}"),
                kind="AGENT_RESULT",
            )
        ],
        confidence=0.86,
        metrics={"order_qty": float(decision["order_qty"]), "cost_usd": decision["cost_usd"]},
    )


# --------------------------------------------------------------------------- graph


def build_graph():
    graph = StateGraph(OrderState)
    graph.add_node("triage", triage)
    graph.add_node("inventory", inventory)
    graph.add_node("demand", demand)
    graph.add_node("supplier", supplier)
    graph.add_node("decide", decide)
    graph.add_node("explain", explain)

    graph.add_edge(START, "triage")
    # fan out: the three investigations run concurrently
    for node in ("inventory", "demand", "supplier"):
        graph.add_edge("triage", node)
        graph.add_edge(node, "decide")       # fan in: decide waits for all three
    graph.add_edge("decide", "explain")
    graph.add_edge("explain", END)
    return graph.compile(checkpointer=InMemorySaver())


async def main() -> None:
    app = build_graph()

    # A turn id belongs to the session that created it, so each run gets its own; the
    # thread stays stable so the conversation accumulates.
    run = uuid.uuid4().hex[:8]
    context = AgentExecutionContext.create(
        tenant_id=os.environ.get("MEMORY_TENANT", "acme"),
        agent_id="reorder-workflow",
        user_id="planner-7",
        # stable thread so the conversation accumulates; override to isolate a run
        thread_id=os.environ.get("MEMORY_THREAD_ID", "chat-supply-42"),
        turn_id=f"turn-{run}",
        work_id="wo-2291",
    )

    question = "Do we need to reorder SKU-1 before the quarter closes?"
    config = {
        "configurable": {
            "thread_id": context.thread_id,
            # identity the graph carries for the harness
            "harness": {
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "work_id": context.work_id,
                "agent_group_id": "supply-chain",
            },
        }
    }

    started = asyncio.get_running_loop().time()
    # Wrapping the invocation gives the turn a single root span, so every node run nests
    # under one trace (and one Langfuse session) instead of starting its own. Without this
    # the graph still works — you just get one trace per node.
    async with harness.execution(context, agent_id="reorder-workflow", input=question):
        out = await app.ainvoke({"question": question, "trace": [], "risks": []}, config)
    elapsed = (asyncio.get_running_loop().time() - started) * 1000

    decision = out["decision"]
    print("\n=== reorder workflow ===")
    print(f"question     : {out['question']}")
    print(f"nodes        : {' -> '.join(out['trace'])}")
    print(f"risks        : {out['risks'] or ['none']}")
    print(f"position     : {decision['position']} units "
          f"({decision['days_of_cover']} days of cover)")
    print(f"reorder point: {decision['reorder_point']} "
          f"(safety stock {decision['safety_stock']})")
    print(f"decision     : {decision['order_qty']} units from {decision['supplier']} "
          f"at ${decision['cost_usd']:,.2f}"
          f"{' (budget capped)' if decision['budget_capped'] else ''}")
    print(f"purchase order artifact: {decision['purchase_order_ref']}")
    print(f"placeable    : {decision['placeable']}"
          f"{'' if decision['placeable'] else ' -> escalate'}")
    print(f"answer       : {out['answer']}")
    print(f"wall clock   : {elapsed:.0f} ms (the three investigations ran in parallel)")
    print(f"memory       : {'attached' if os.environ.get('MEMORY_SERVICE_URL') else 'no-op'}")
    print(f"langfuse     : {'enabled' if harness.langfuse else 'disabled'}")

    await harness.aclose()   # drain memory writeback, flush telemetry


def _actions(decision: dict[str, Any]) -> list[RecommendedAction]:
    """Concise, evidence-linked next steps — never a chain of thought."""
    placeable = decision["placeable"]
    actions = [
        RecommendedAction(
            action_type="raise_purchase_order" if placeable else "escalate_to_procurement",
            description=(
                f"Raise a PO for {decision['order_qty']} units with {decision['supplier']}"
                if decision["placeable"]
                else f"Escalate: {'; '.join(decision['blockers'])}"
            ),
            reason_summary="position is below the reorder point for the chosen lead time",
            evidence_ids=["inventory_db", "demand_forecast"],
            confidence=0.86,
        )
    ]
    if not decision["covers_lead_time"]:
        actions.append(
            RecommendedAction(
                action_type="expedite_shipment",
                description=f"Request expedited delivery from {decision['supplier']}",
                reason_summary="stock runs out before the lead time elapses",
                evidence_ids=["demand_forecast"],
                confidence=0.74,
            )
        )
    return actions


def _extract_sku(text: str | None) -> str | None:
    for token in (text or "").replace(",", " ").split():
        if token.upper().startswith("SKU-"):
            return token.split("=")[-1].upper()
    return None


def _render_po(decision: dict[str, Any]) -> str:
    return (
        "PURCHASE ORDER (draft)\n"
        f"SKU         : {decision['sku']}\n"
        f"Supplier    : {decision['supplier']}\n"
        f"Quantity    : {decision['order_qty']}\n"
        f"Unit lead   : {decision['lead_time_days']} days\n"
        f"Total       : ${decision['cost_usd']:,.2f}\n"
        f"Reorder pt  : {decision['reorder_point']}\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
