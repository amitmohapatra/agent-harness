"""A LangGraph graph's own ``interrupt()``, with fields: options with labels, several picks, your
own screen, whose question it is — the same question ``ask(...)`` builds, from a dict.

The ``offer`` node calls ``interrupt({...})``; the harness records it in the run store as a
question (in ``role:sales``'s inbox, rendered by your ``plan-picker`` component where a surface
has it), checks the answer against the options, and resumes the graph in place with
``Command(resume=answer)``. A graph's own interrupt needs a checkpointer.

    python -m examples.05_features.langgraph_interrupt_fields
"""

from __future__ import annotations

import asyncio
from typing import Any, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from trellis import Harness


class Deal(TypedDict):
    customer: str
    plans: NotRequired[list[str]]


def offer(state: Deal) -> dict[str, Any]:
    plans = interrupt(
        {
            "question": f"Which plans do we offer {state['customer']}?",
            "options": [
                {"value": "basic", "label": "Basic, 10 EUR", "description": "up to 3 seats"},
                {"value": "pro", "label": "Pro, 30 EUR"},
                "enterprise",
            ],
            "multiple": True,
            "component": "plan-picker",
            "props": {"customer": state["customer"]},
            "assignee": "role:sales",
        }
    )
    return {"plans": plans}


def build() -> Any:
    graph = StateGraph(Deal)
    graph.add_node("offer", offer)
    graph.add_edge(START, "offer")
    graph.add_edge("offer", END)
    return graph.compile(checkpointer=InMemorySaver())


async def main() -> None:
    async with Harness() as h:
        graph = build()
        agent = h.wrap(graph, id="deals")
        result = await agent.run({"customer": "acme"}, user="ada", thread="deal-1")
        assert result.interrupt is not None
        asked = result.interrupt
        print(asked.question, asked.ui, asked.component, asked.assignee, asked.option_values)

        [waiting] = await h.inbox("role:sales")  # your own screen lists it from here
        print("inbox:", waiting.run_id == result.run_id)
        done = await agent.resume(
            asked.interrupt_id, "answer", answer=["basic", "pro"], reviewer="lee"
        )
        state = await graph.aget_state({"configurable": {"thread_id": "deal-1"}})
        print(done.status.value, "offered:", state.values["plans"])


if __name__ == "__main__":
    asyncio.run(main())
