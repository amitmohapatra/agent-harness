"""Plain Python, no framework: wrap what you have, then adopt more when you want to.

    python examples/plain_python.py
"""

from __future__ import annotations

import asyncio

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResult,
    Claim,
    MemoryObservation,
)


# 1. an agent that already exists ------------------------------------------------------
async def inventory_agent(question: str) -> str:
    return f"SKU-1 has 3 units left ({question})"


# 2. a tool you call directly ----------------------------------------------------------
async def inventory_db(sku: str) -> dict:
    """Stock levels for a SKU."""
    return {"sku": sku, "on_hand": 3, "reorder_point": 50}


async def main() -> None:
    # No Memory Service here: everything degrades to a no-op so the example runs anywhere.
    harness = AgentHarness(
        tools=[inventory_db],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config={"evaluation_events": {"enabled": True}},
    )

    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inventory-agent", user_id="u1", thread_id="chat-42",
        turn_id="turn-1",
    )

    # -- level 1: wrap an existing callable --------------------------------------------
    wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent")
    result = await wrapped("how much stock?", context=context)
    print("level 1:", result.status, result.data)

    # -- level 2: runtime-aware agent ---------------------------------------------------
    @harness.agent(agent_id="reorder-agent", skills=["inventory.reorder"])
    async def reorder_agent(state, agent) -> AgentResult:
        stock = (await agent.tools.call("inventory_db", sku=state["sku"])).output
        needed = max(0, stock["reorder_point"] - stock["on_hand"])
        report = await agent.artifacts.put(
            f"reorder report for {state['sku']}: {needed} units", type="report"
        )
        return AgentResult.ok(
            {"reorder": needed},
            claims=[
                Claim(claim_id="c1", text=f"{state['sku']} is {needed} units below reorder")
            ],
            artifacts=[report],
            memory_observations=[MemoryObservation(content=f"{state['sku']} needs {needed} units")],
            confidence=0.9,
        )

    result = await reorder_agent({"sku": "SKU-1"}, context=context)
    print("level 2:", result.data, "| artifacts:", [a.artifact_id[:12] for a in result.artifacts])
    print("        metrics:", result.metrics)

    # -- level 3: instrument a block ----------------------------------------------------
    async with harness.execution(context, agent_id="report-agent", input="summarise") as runtime:
        runtime.logger.info("doing work the harness cannot see inside")
        runtime.state["result"] = AgentResult.ok("summary written")

    # -- child runs inherit identity ----------------------------------------------------
    child = context.for_agent("promotion-agent")
    print("child run:", child.agent_id, "parent:", child.parent_agent_run_id[:12])

    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
