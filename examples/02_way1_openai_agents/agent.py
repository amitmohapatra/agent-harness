"""OpenAI Agents SDK, wrapped, with every piece that applies: memory, local and MCP tools,
governance and a person, a hook, ``without=``, a time limit and ``Runner.run``'s own arguments.

Your ``Agent`` keeps its own tools; the harness adds its tools (the gateway's MCP tools,
memory's) to a copy of it per run, and every call — its own and the harness's — goes through
the harness. The tool catalog says ``create_po`` is irreversible (offline: a scripted memory
service's), so its call waits for a person; the resume re-runs the agent against the run's
journal (the supplier lookup is replayed, not run again) and the approved call runs once. (An
administrator's ``approve_when`` for the tool in the catalog — ``amount > 10000`` — can decide
instead.)

    python -m examples.02_way1_openai_agents.agent
"""

from __future__ import annotations

import asyncio

from agents import Agent, function_tool, set_tracing_disabled
from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, openai_agents_model

from trellis import Harness, Hooks
from trellis.contracts import ToolCall, ToolOutcome

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel


def supplier(name: str) -> str:
    """A supplier's rating and terms."""
    return f"{name}: rated A, net 30"


@function_tool
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
    catalog = {"create_po": {"risk": "irreversible"}}  # an administrator's, in the tool catalog
    async with Harness(**offline_blocks(memory=ScriptedMemory(catalog=catalog), mcp=crm)) as h:
        look = ("crm-supplier", {"name": "ACME"})
        call = ("create_po", {"supplier": "ACME", "amount": 12000})
        # a resume re-runs the agent: it plans the same calls, the lookup replayed
        model = openai_agents_model([look, call, look, call, "PO created for 12000 EUR."])
        buyer = Agent(
            name="buyer",
            instructions="You create purchase orders.",
            model=model,
            tools=[create_po],
        )
        agent = h.wrap(
            buyer,
            id="buyer",
            hooks=[Audit()],
            run_timeout=300,
            framework_options={"max_turns": 8},  # Runner.run's own argument
            without={"tool_search"},  # every tool offered every turn
        )

        result = await agent.run("Order 12000 EUR of steel from ACME.", user="ada")
        while result.interrupt is not None:  # irreversible: a person approves it
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
