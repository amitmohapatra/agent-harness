"""OpenAI Agents SDK, wrapped, with every piece that applies: memory, local and MCP tools,
governance and a person, a hook, ``without=``, a time limit and ``Runner.run``'s own arguments.

The harness adds its tools (yours, the gateway's MCP tools, memory's) to a copy of your
``Agent`` per run, as ``FunctionTool``s whose calls go through the harness. An irreversible call
waits for a person; the resume re-runs the agent against the run's journal (the supplier lookup
is replayed, not run again) and the approved call runs once. (With a memory service, an
administrator's ``approve_when`` for the tool in the catalog — ``amount > 10000`` — decides
instead.)

    python -m examples.02_way1_openai_agents.agent
"""

from __future__ import annotations

import asyncio

from agents import Agent, set_tracing_disabled
from examples._support.gateway import McpTool
from examples._support.offline import offline_blocks, openai_agents_model

from trellis import Harness, Hooks, tool
from trellis.contracts import ToolCall, ToolOutcome

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel


def supplier(name: str) -> str:
    """A supplier's rating and terms."""
    return f"{name}: rated A, net 30"


@tool(side_effects="irreversible")
def create_po(supplier: str, amount: float) -> str:
    """Create a purchase order."""
    return f"PO for {amount} EUR to {supplier}"


class Audit(Hooks):
    """Your audit trail: every tool outcome, as the model will read it."""

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        print(f"audit: {call.tool} -> {outcome.status}")
        return outcome


async def main() -> None:
    crm = {"crm-supplier": McpTool(supplier, read_only=True)}
    async with Harness(**offline_blocks(mcp=crm)) as h:
        look = ("crm-supplier", {"name": "ACME"})
        call = ("create_po", {"supplier": "ACME", "amount": 12000})
        # a resume re-runs the agent: it plans the same calls, the lookup replayed
        model = openai_agents_model([look, call, look, call, "PO created for 12000 EUR."])
        buyer = Agent(name="buyer", instructions="You create purchase orders.", model=model)
        agent = h.wrap(
            buyer,
            id="buyer",
            tools=[create_po],
            hooks=[Audit()],
            timeout=300,
            framework_options={"max_turns": 8},  # Runner.run's own argument
            without={"hints"},  # every tool offered every turn
        )

        result = await agent.run("Order 12000 EUR of steel from ACME.", user="ada")
        while result.interrupt is not None:  # irreversible: a person approves it
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
