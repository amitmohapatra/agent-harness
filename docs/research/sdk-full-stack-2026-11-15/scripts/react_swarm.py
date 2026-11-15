"""``ReAct`` (the harness's own agent: ``create_deep_agent`` plus ``middleware()``) and
langgraph-swarm 0.1 (handoffs), as their docs show them.

``ReAct`` has no "on its own" (it is the harness's); its rows compare with the library feature
it is built on. langgraph-swarm runs ``bare`` (``create_swarm`` of ``create_agent`` agents,
``InMemorySaver``) and wrapped.

Usage: python react_swarm.py [feature[:arm] ...]
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from common import MODEL, case, chosen, uid
from deepagents.backends import StateBackend
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_swarm import create_handoff_tool, create_swarm
from lc_agent import middleware_on, model, read, text, write
from pydantic import BaseModel
from turns import wrapped

from trellis import ReAct, tool
from trellis.harness.plan_mode import PlanMode

SDK = "react"
WIRE = (
    "Wire 300 to account ACME-1 now. The currency is the account default; do not ask, "
    "just call the tool."
)


def react(prompt: str, **kw: Any) -> Any:
    return ReAct(prompt, MODEL, **kw)


async def r_tools(arm: str) -> dict[str, Any]:
    calls: list[str] = []

    @tool(side_effects="read")
    def stock(sku: str) -> str:
        """How many units of a SKU are in stock."""
        calls.append(sku)
        return f"{sku}: 7 units in stock"

    got = await wrapped(
        react("You answer stock questions.", tools=[stock]),
        arm,
        "re-tools",
        ["How many units of SKU-1 are in stock?"],
    )
    return {"ok": "7" in str(got["answers"][-1]) and calls == ["SKU-1"], "calls": calls, **got}


class Weather(BaseModel):
    city: str
    celsius: float


async def r_structured(arm: str) -> dict[str, Any]:
    @tool(side_effects="read")
    def temperature(city: str) -> float:
        """Today's temperature in a city, in Celsius."""
        return 21.5

    target = react("You report weather.", tools=[temperature], response_format=Weather)
    got = await wrapped(target, arm, "re-structured", ["What is the temperature in Oslo?"])
    a = got["answers"][-1]
    return {"ok": isinstance(a, Weather) and a.celsius == 21.5, "type": type(a).__name__, **got}


async def r_approval(arm: str) -> dict[str, Any]:
    made: list[str] = []

    @tool(side_effects="irreversible")
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        made.append(to)
        return f"wired {amount} to {to}"

    got = await wrapped(react("You make payments.", tools=[wire]), arm, "re-approval", [WIRE])
    return {"ok": bool(got["asked"]) and made == ["ACME-1"], "made": made, **got}


async def r_reject(arm: str) -> dict[str, Any]:
    made: list[str] = []

    @tool(side_effects="irreversible")
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        made.append(to)
        return f"wired {amount} to {to}"

    got = await wrapped(
        react("You make payments.", tools=[wire]),
        arm,
        "re-reject",
        [WIRE],
        decide=lambda _i: ("reject", "the account is frozen"),
    )
    said = str(got["answers"][-1]).lower()
    return {"ok": made == [] and ("frozen" in said or "reject" in said), "made": made, **got}


async def r_subagent(arm: str) -> dict[str, Any]:
    calls: list[str] = []

    @lc_tool
    def lookup_order(order: str) -> str:
        """Look up an order's status."""
        calls.append(order)
        return f"order {order}: shipped on 2026-11-02 via DHL"

    sub = {
        "name": "order-checker",
        "description": "Looks up orders' status.",
        "system_prompt": "Look the order up with lookup_order and report its status.",
        "tools": [read(lookup_order)],
    }
    target = react("Delegate order questions to order-checker.", subagents=[sub])
    got = await wrapped(target, arm, "re-sub", ["What is the status of order O-31?"])
    return {
        "ok": calls == ["O-31"] and "dhl" in str(got["answers"][-1]).lower(),
        "sub_calls": calls,
        **got,
    }


async def r_files(arm: str) -> dict[str, Any]:
    """``backend=``: Deep Agents' file tools from the first call; kept across the turns."""
    target = react("You keep notes in files.", backend=StateBackend())
    qs = [
        "Write 'launch code: ORCHID-42' to /notes.md, then say done.",
        "Read /notes.md and tell me the launch code.",
    ]
    got = await wrapped(target, arm, "re-files", qs)
    return {"ok": "ORCHID-42" in str(got["answers"][-1]), **got}


async def r_plan_mode(arm: str) -> dict[str, Any]:
    made: list[str] = []

    @tool(side_effects="read")
    def order(order_id: str) -> str:
        """Look an order up."""
        return f"{order_id}: 2 items, 80 EUR, paid"

    @tool(side_effects="write")
    def refund(order_id: str) -> str:
        """Refund an order."""
        made.append(order_id)
        return f"refunded {order_id}"

    target = react("You handle refunds.", tools=[order, refund], middleware=[PlanMode()])
    got = await wrapped(
        target,
        arm,
        "re-plan",
        ["Refund order O-1."],
        run={"framework_options": {"configurable": {"plan_mode": True}}},
    )
    plan_asked = any("write_todos" in str(a.get("tool_call")) for a in got["asked"])
    return {"ok": plan_asked and made == ["O-1"], "made": made, "plan_asked": plan_asked, **got}


# --------------------------------------------------------------------------- langgraph-swarm
def team(arm: str, made: list[str]) -> Any:
    @lc_tool
    def refund(order: str) -> str:
        """Refund an order."""
        made.append(order)
        return f"refunded {order}"

    mw = middleware_on() if arm == "on" else []
    desk = create_agent(
        model(),
        tools=[create_handoff_tool(agent_name="billing")],
        system_prompt="You are the front desk. Hand refunds to billing.",
        name="desk",
        middleware=mw,
    )
    billing = create_agent(
        model(),
        tools=[write(refund), create_handoff_tool(agent_name="desk")],
        system_prompt="You are billing: you do refunds. Say 'billing here' "
        "at the start of every reply.",
        name="billing",
        middleware=mw,
    )
    return create_swarm([desk, billing], default_active_agent="desk").compile(
        checkpointer=InMemorySaver()
    )


async def s_handoff(arm: str) -> dict[str, Any]:
    """A handoff within the turn, then the next turn answered by the agent handed to."""
    made: list[str] = []
    graph = team(arm, made)
    qs = ["Please refund order O-9.", "Thanks. Who am I talking to now?"]
    if arm == "bare":
        cfg = {"configurable": {"thread_id": uid()}}
        answers, active = [], []
        for q in qs:
            state = await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg)
            answers.append(text(state))
            active.append(state.get("active_agent"))
        got: dict[str, Any] = {"answers": answers, "active": active}
    else:
        got = await wrapped(graph, arm, "sw-handoff", qs)
    ok = made == ["O-9"] and "billing" in str(got["answers"][-1]).lower()
    return {"ok": ok, "made": made, **got}


FEATURES: dict[str, Any] = {
    "r_tools": {"off", "stack"},
    "r_structured": {"off", "stack"},
    "r_approval": {"off", "stack"},
    "r_reject": {"off", "stack"},
    "r_subagent": {"off", "stack"},
    "r_files": {"off", "stack"},
    "r_plan_mode": {"off", "stack"},
    "s_handoff": {"bare", "off", "stack", "on"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        sdk = "swarm" if feature.startswith("s_") else SDK
        await case(sdk, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
