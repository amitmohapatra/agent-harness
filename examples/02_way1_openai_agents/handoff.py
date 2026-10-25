"""OpenAI Agents SDK handoffs: the agent you wrap and the specialists it hands off to keep their
own tools, and the harness governs every call of each — approvals (the tool catalog says
``refund`` is irreversible; offline: a scripted memory service's), the journal, the record —
and gives each the memory tools. A resume re-runs the whole conversation, handoff included,
against the run's journal.

    python -m examples.02_way1_openai_agents.handoff
"""

from __future__ import annotations

import asyncio

from agents import Agent, function_tool, set_tracing_disabled
from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, openai_agents_model

from trellis import Harness

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel


@function_tool
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


async def main() -> None:
    catalog = {"refund": {"risk": "irreversible"}}  # an administrator's, in the tool catalog
    async with Harness(**offline_blocks(memory=ScriptedMemory(catalog=catalog))) as h:
        call = ("refund", {"order": "o-7", "amount": 40.0})
        # a resume re-runs the run: triage hands off again, the specialist calls again, and the
        # approved refund runs once
        refunds = Agent(
            name="refunds",
            instructions="You refund orders.",
            model=openai_agents_model([call, call, "Refunded 40 EUR on o-7."]),
            tools=[refund],
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
