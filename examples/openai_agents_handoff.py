"""OpenAI Agents SDK handoffs: ``wrap(tools=...)`` adds harness tools to the agent you wrap; a
specialist reached by a handoff gets its harness tools when it is built, from
``h.tools(..., framework="openai_agents")``. Every call is still the harness's — governance,
approvals, the journal, the record — and a resume re-runs the whole conversation, handoff
included, against the run's journal.

    .venv/bin/python examples/openai_agents_handoff.py
"""

from __future__ import annotations

import asyncio

from _offline import openai_agents_model
from agents import Agent, set_tracing_disabled

from trellis import Harness, tool

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel


@tool(side_effects="irreversible")
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


async def main() -> None:
    async with Harness() as h:
        call = ("refund", {"order": "o-7", "amount": 40.0})
        # a resume re-runs the run: triage hands off again, the specialist calls again, and the
        # approved refund runs once
        refunds = Agent(
            name="refunds",
            instructions="You refund orders.",
            model=openai_agents_model([call, call, "Refunded 40 EUR on o-7."]),
            tools=await h.tools(refund, framework="openai_agents"),
        )
        handoff = ("transfer_to_refunds", {})
        triage = Agent(
            name="triage",
            instructions="Route refund requests to the refunds agent.",
            model=openai_agents_model([handoff, handoff]),
            handoffs=[refunds],
        )
        agent = h.wrap(triage, id="support")

        result = await agent.run("Please refund order o-7 (40 EUR).", user="ada")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lead")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
