"""Deep Agents, wrapped, with every piece that applies: memory, local and MCP tools, governance
and a person (resumed in place), a hook, ``without=``, a time limit and the graph's own config.

``create_deep_agent`` returns a compiled LangGraph graph, so it is wrapped like any graph and its
tools come from ``h.tools(..., framework="deepagents")``. With a checkpointer the approval *is* a
LangGraph interrupt: the resume continues the graph where it stopped and the model is not asked
again. Planning (``write_todos``) is opt-in in Deep Agents: ``TodoListMiddleware`` adds it.

    python -m examples.02_way1_deepagents.agent
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent
from examples._support.gateway import McpTool
from examples._support.offline import langchain_model, offline_blocks
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from trellis import Deny, Harness, Hooks, tool
from trellis.contracts import ToolCall


def order(number: str) -> str:
    """An order's amount and state."""
    return f"order {number}: 40 EUR, delivered damaged"


@tool(side_effects="irreversible")
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


class RefundLimit(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | None:
        too_much = call.tool == "refund" and call.args.get("amount", 0) > 500
        return Deny("refunds over 500 go through finance") if too_much else None


async def main() -> None:
    shop = {"shop-order": McpTool(order, read_only=True)}
    async with Harness(**offline_blocks(mcp=shop)) as h:
        model = langchain_model(
            [
                ("write_todos", {"todos": [{"content": "refund o-7", "status": "in_progress"}]}),
                ("shop-order", {"number": "o-7"}),
                ("refund", {"order": "o-7", "amount": 40.0}),
                "Refunded 40.0 on o-7.",
            ]
        )
        graph = create_deep_agent(
            model=model,
            tools=await h.tools(refund, framework="deepagents"),
            system_prompt="You process refund requests. Plan, look the order up, then refund.",
            middleware=[TodoListMiddleware()],
            checkpointer=InMemorySaver(),
        )
        agent = h.wrap(
            graph,
            id="refunds",
            hooks=[RefundLimit()],
            timeout=300,
            framework_options={"recursion_limit": 50},
            without={"judges"},  # every run of this agent: no online judges
        )

        result = await agent.run("Refund order o-7 (40 EUR).", user="ada", thread="ticket-7")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lead")
        print(result.status.value, result.answer)
        state = await graph.aget_state({"configurable": {"thread_id": "ticket-7"}})
        print("plan:", [todo["content"] for todo in state.values.get("todos", [])])


if __name__ == "__main__":
    asyncio.run(main())
