"""A plain async function, wrapped, with every piece that applies to it: memory, local and MCP
tools, a question to a person, a hook, ``without=`` and a time limit.

A function has no model and no framework: it is your code deciding (a workflow, a router,
glue). It reads memory through ``agent.context`` (pushed) and ``agent.memory`` (its own
reads), calls tools through ``agent.tools.call`` — governed, journaled and recorded like a
model's calls — and asks a person with ``agent.ask``. ``framework_options=`` does not apply: a
function has no framework run call to pass them to (``h.wrap`` refuses them).

    python -m examples.02_way1_function.agent
"""

from __future__ import annotations

import asyncio

from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks

from trellis import Deny, Harness, Hooks, Runtime, tool
from trellis.contracts import ToolCall


def ticket(number: str) -> str:
    """A support ticket's text."""
    return f"{number}: the invoice for order o-7 is wrong (charged twice)"


@tool(side_effects="write")
def credit(order: str, amount: float) -> str:
    """Credit an amount back on an order."""
    return f"credited {amount:g} EUR on {order}"


class NoBigCredits(Hooks):
    """Your rule, in code: credits over 100 go through finance, never through this agent."""

    async def before_tool(self, call: ToolCall) -> Deny | None:
        if call.tool == "credit" and call.args.get("amount", 0) > 100:
            return Deny("credits over 100 go through finance")
        return None


async def support(number: str, agent: Runtime) -> str:
    """Your code: read the ticket (an MCP tool), check memory, ask the customer's account
    manager which remedy, apply it (a local tool)."""
    text = await agent.tools.call("crm-ticket", number=number)
    reads = agent.uses("memory_pull")  # without={"memory"} turns the reads off too
    known = await agent.memory.search("billing preferences") if reads else []
    remedy = await agent.ask(
        f"{text}. Which remedy?",
        options=["credit", "refund"],
        assignee="role:account-managers",
    )
    done = await agent.tools.call("credit", order="o-7", amount=42.5) if remedy == "credit" else ""
    return f"{remedy}: {done} ({len(known)} memories, context: {agent.context or 'none'})"


async def main() -> None:
    memory = ScriptedMemory(context="The customer pays by card.", facts=["Prefers credits."])
    crm = {"crm-ticket": McpTool(ticket, read_only=True)}
    async with Harness(**offline_blocks(memory=memory, mcp=crm)) as h:
        agent = h.wrap(support, id="support", tools=[credit], hooks=[NoBigCredits()], timeout=60)

        result = await agent.run("T-9", user="ada")
        assert result.interrupt is not None  # the account manager's choice
        print("asks", result.interrupt.assignee, "->", result.interrupt.question)
        result = await agent.resume(result.run_id, "answer", answer="credit", reviewer="lee")
        print(result.status.value, result.answer)

        bare = await agent.run("T-9", user="ada", without={"memory"})  # no context, no reads
        assert bare.interrupt is not None
        print("without memory:", (await agent.cancel(bare.run_id, reason="example")).status.value)


if __name__ == "__main__":
    asyncio.run(main())
