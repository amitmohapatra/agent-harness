"""OpenAI Agents SDK: harness tools are added to a copy of your agent per run, and an
irreversible call waits for a person. (With a memory service, an admin's ``approve_when`` for
the tool in the catalog — ``amount > 10000`` — decides instead.)

    .venv/bin/python examples/openai_agents_agent.py
"""

from __future__ import annotations

import asyncio

from _offline import openai_agents_model
from agents import Agent, set_tracing_disabled

from trellis import Harness, tool

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel


@tool(side_effects="irreversible")
def create_po(supplier: str, amount: float) -> str:
    """Create a purchase order."""
    return f"PO for {amount} EUR to {supplier}"


async def main() -> None:
    async with Harness() as h:
        call = ("create_po", {"supplier": "ACME", "amount": 12000})
        # a resume re-runs the agent: it plans the same call, which now runs approved
        model = openai_agents_model([call, call, "PO created for 12000 EUR."])
        buyer = Agent(name="buyer", instructions="You create purchase orders.", model=model)
        agent = h.wrap(buyer, id="buyer", tools=[create_po])

        result = await agent.run("Order 12000 EUR of steel from ACME.", user="ada")
        while result.interrupt is not None:  # irreversible: a person approves it
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
