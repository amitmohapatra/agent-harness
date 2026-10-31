"""Deep Agents, wrapped, with every piece that applies: memory, local and MCP tools, governance
and a person (resumed in place), a hook, ``without=``, a time limit and the graph's own config.

``create_deep_agent`` returns a compiled LangGraph graph, so it is wrapped like any graph: its
own tools are governed as it runs them (the tool catalog says ``refund`` is irreversible;
offline: a scripted memory service's), and ``HarnessTools`` adds the harness's — the gateway's
MCP tools and memory's. With a checkpointer the approval *is* a LangGraph interrupt: the resume
continues the graph where it stopped and the model is not asked again. Planning
(``write_todos``) is opt-in in Deep Agents: ``TodoListMiddleware`` adds it.

    python -m examples.02_way1_deepagents.agent
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import langchain_model, offline_blocks
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from trellis import Deny, Harness, Hooks
from trellis.contracts import ToolCall
from trellis.harness.middleware import HarnessTools


def order(number: str) -> str:
    """An order's amount and state."""
    return f"order {number}: 40 EUR, delivered damaged"


@tool
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


class RefundLimit(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | None:
        too_much = call.tool == "refund" and call.args.get("amount", 0) > 500
        return Deny("refunds over 500 go through finance") if too_much else None


async def main() -> None:
    shop = {"shop-order": McpTool(order, read_only=True)}
    catalog = {"refund": {"risk": "irreversible"}}  # an administrator's, in the tool catalog
    async with Harness(**offline_blocks(memory=ScriptedMemory(catalog=catalog), mcp=shop)) as h:
        model = langchain_model(
            [
                ("write_todos", {"todos": [{"content": "refund o-7", "status": "in_progress"}]}),
                ("shop-order", {"number": "o-7"}),
                ("refund", {"order": "o-7", "amount": 40.0}),
                "Refunded 40.0 on o-7.",
            ]
        )
        stack: list[AgentMiddleware[Any, Any]] = [TodoListMiddleware(), HarnessTools()]
        graph = create_deep_agent(
            model=model,
            tools=[refund],
            system_prompt="You process refund requests. Plan, look the order up, then refund.",
            middleware=stack,
            checkpointer=InMemorySaver(),
        )
        agent = h.wrap(
            graph,
            id="refunds",
            hooks=[RefundLimit()],
            run_timeout=300,
            framework_options={"recursion_limit": 50},
            without={"judges"},  # every run of this agent: no online judges
        )

        result = await agent.run("Refund order o-7 (40 EUR).", user="ada", thread="ticket-7")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lead")
        print(result.status.value, result.answer)
        thread = await h.conversation(user="ada", thread="ticket-7")  # ada's ticket-7
        state = await graph.aget_state({"configurable": {"thread_id": thread}})
        print("plan:", [todo["content"] for todo in state.values.get("todos", [])])


if __name__ == "__main__":
    asyncio.run(main())
