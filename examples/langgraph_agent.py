"""LangGraph: build the agent with harness tools, wrap it, approve an irreversible call.

.venv/bin/python examples/langgraph_agent.py      # offline, or with BIFROST_URL / MEMORY_URL
"""

from __future__ import annotations

import asyncio
import os

from _offline import langchain_model
from langchain.agents import create_agent

from trellis import Harness, tool


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


@tool(side_effects="irreversible")
def reorder(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"ordered {qty} x {sku}"


async def main() -> None:
    async with Harness() as h:
        plan = [("stock", {"sku": "SKU-1"}), ("reorder", {"sku": "SKU-1", "qty": 20})]
        # without a checkpointer a resume re-runs the graph: the model plans again, the stock
        # check is replayed from the run's journal, and the approved reorder executes
        model = langchain_model([*plan, *plan, "Ordered 20 x SKU-1."])
        graph = create_agent(
            model,
            tools=await h.tools(stock, reorder, framework="langgraph"),
            system_prompt="You keep stock above 10 units. Check stock, then reorder 20 if low.",
        )
        memory = "read_write" if os.environ.get("MEMORY_URL") else "off"
        agent = h.wrap(graph, id="stock-keeper", memory=memory)

        result = await agent.run("Is SKU-1 low? Top it up if so.", user="ada")
        while result.interrupt is not None:  # the reorder is irreversible: a person approves it
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
