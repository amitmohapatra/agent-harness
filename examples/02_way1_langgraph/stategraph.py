"""A hand-built LangGraph ``StateGraph``: your nodes, edges and tools stay as they are. Compile
with a checkpointer, wrap the graph — and call ``agent.run`` / ``agent.resume`` where you called
``graph.ainvoke``.

The harness's own tools — ``ask_user``, the schedule tools, the memory tools — are bound with
yours when the graph is built (``await h.harness_tools()``): the graph's model can ask the user,
or set them a schedule, as every other agent's can.

Three pauses, each resumed in place from the checkpoint (the model is not asked again):

* the model's question: it calls ``ask_user`` (how many units?), the run waits for the answer,
  and the answer is the call's result;
* the harness's: the tool catalog says ``reorder`` is irreversible (offline: a scripted memory
  service's), so its call inside the tool node waits for a person;
* the graph's own: the ``confirm`` node calls LangGraph's ``interrupt()``, which the harness
  records as a question in the run store and answers with ``Command(resume=...)``.

    python -m examples.02_way1_langgraph.stategraph
"""

from __future__ import annotations

import asyncio
from typing import Any

from examples._support.memory import ScriptedMemory
from examples._support.offline import langchain_model, offline_blocks
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from trellis import Harness


@tool
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


@tool
def reorder(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"ordered {qty} x {sku}"


def build(model: Any, tools: list[Any]) -> Any:
    """The graph as a team would write it: model, tools, and a confirmation step."""
    bound = model.bind_tools(tools)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "confirm"

    def confirm(state: MessagesState) -> dict[str, Any]:
        answer = interrupt({"question": "Send the stock report to the warehouse?"})
        return {
            "messages": [AIMessage(content=f"{state['messages'][-1].content} (sent: {answer})")]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("confirm", confirm)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route)
    graph.add_edge("tools", "model")
    graph.add_edge("confirm", END)
    return graph.compile(checkpointer=InMemorySaver())


async def main() -> None:
    catalog = {"stock": {"risk": "read"}, "reorder": {"risk": "irreversible"}}
    async with Harness(**offline_blocks(memory=ScriptedMemory(catalog=catalog))) as h:
        model = langchain_model(
            [
                ("stock", {"sku": "SKU-1"}),
                ("ask_user", {"question": "How many units of SKU-1?", "kind": "number"}),
                ("reorder", {"sku": "SKU-1", "qty": 20}),
                "SKU-1 was at 3 units; ordered 20.",
            ]
        )
        tools = [stock, reorder, *await h.harness_tools()]
        agent = h.wrap(build(model, tools), id="stock-graph")

        result = await agent.run("Top up SKU-1 if it is low.", user="ada", thread="stock-1")
        while result.interrupt is not None:
            asked = result.interrupt
            print("asks:", asked.question)
            if asked.tool_call is not None:  # the harness's approval of the reorder
                result = await agent.resume(asked.interrupt_id, "approve", reviewer="lead")
            else:  # the model's ask_user, or the graph's own interrupt(): answered
                answer = "20" if asked.component == "ask_user" else "yes"
                result = await agent.resume(
                    asked.interrupt_id, "answer", answer=answer, reviewer="ada"
                )
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
