"""Deep Agents: ``create_deep_agent`` returns a compiled graph, so it is wrapped like any
LangGraph graph. With a checkpointer the approval *is* a LangGraph interrupt, and the resume
continues the graph where it stopped. Planning (``write_todos``) is opt-in in Deep Agents:
``TodoListMiddleware`` adds it.

    .venv/bin/python examples/deepagents_agent.py
"""

from __future__ import annotations

import asyncio

from _offline import langchain_model
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from trellis import Harness, tool


@tool(side_effects="irreversible")
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


async def main() -> None:
    async with Harness() as h:
        model = langchain_model(
            [
                ("write_todos", {"todos": [{"content": "refund o-7", "status": "in_progress"}]}),
                ("refund", {"order": "o-7", "amount": 40.0}),
                "Refunded 40.0 on o-7.",
            ]
        )
        graph = create_deep_agent(
            model=model,
            tools=await h.tools(refund, framework="deepagents"),
            system_prompt="You process refund requests. Plan, then refund.",
            middleware=[TodoListMiddleware()],
            checkpointer=InMemorySaver(),
        )
        agent = h.wrap(graph, id="refunds")

        result = await agent.run("Refund order o-7 (40 EUR).", user="ada", thread="ticket-7")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lead")
        print(result.status.value, result.answer)
        state = await graph.aget_state({"configurable": {"thread_id": "ticket-7"}})
        print("plan:", [todo["content"] for todo in state.values.get("todos", [])])


if __name__ == "__main__":
    asyncio.run(main())
