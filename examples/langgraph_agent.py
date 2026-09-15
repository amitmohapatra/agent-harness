"""LangGraph: an existing node and a runtime-aware node in the same graph.

    pip install "universal-agent-harness[langgraph]"
    python examples/langgraph_agent.py
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from universal_agent_harness import AgentHarness


class State(TypedDict, total=False):
    question: str
    stock: dict
    answer: str
    trace: Annotated[list[str], operator.add]


async def inventory_db(sku: str) -> dict:
    """Stock levels for a SKU."""
    return {"sku": sku, "on_hand": 3, "reorder_point": 50}


async def existing_lookup_node(state: State) -> dict:
    """A node that was written before the harness existed. It is not modified."""
    return {"stock": await inventory_db("SKU-1"), "trace": ["lookup"]}


async def main() -> None:
    harness = AgentHarness(
        tools=[inventory_db],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        # In a real deployment: AgentHarness(memory=MemoryClient(...), config="harness.yaml")
    )

    # -- a runtime-aware node: the second argument is the AgentRuntime --------------------
    @harness.langgraph.agent(
        agent_id="answer-agent", skills=["inventory.analysis"], query="question"
    )
    async def answer_node(state: State, agent) -> dict:
        stock = state["stock"]
        shortfall = stock["reorder_point"] - stock["on_hand"]
        agent.logger.info("answering", shortfall=shortfall)
        return {"answer": f"reorder {shortfall} units of {stock['sku']}", "trace": ["answer"]}

    graph = StateGraph(State)
    graph.add_node(
        "lookup",
        harness.langgraph.wrap_node(existing_lookup_node, agent_id="lookup-agent"),
    )
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", "answer")
    graph.add_edge("answer", END)

    app = graph.compile(checkpointer=InMemorySaver())

    out = await app.ainvoke(
        {"question": "how much stock of SKU-1?", "trace": []},
        {
            "configurable": {
                "thread_id": "chat-42",
                # identity the graph carries for the harness (optional)
                "harness": {"tenant_id": "acme", "user_id": "u1", "work_id": "wo-9"},
            }
        },
    )
    print("answer:", out["answer"])
    print("trace :", out["trace"])  # reducers untouched by the wrapper

    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
