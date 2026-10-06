"""Deep Agents: ``create_deep_agent`` returns a compiled graph, so the LangGraph adapter
wraps it — with its planning tools, its virtual filesystem and a harness tool side by side."""

from __future__ import annotations

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, tool
from trellis.contracts import RunStatus

refunded: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunded.append(order)
    return f"refunded {order}"


async def test_a_deep_agent_uses_its_own_tools_and_the_harness_tools(harness: Harness) -> None:
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/notes.md", "content": "refund o1"}),
            ("refund", {"order": "o1"}),
            "refund of o1 done",
        ]
    )
    graph = create_deep_agent(
        model=model,
        tools=await harness.tools(refund, framework="deepagents"),
        system_prompt="You process refunds.",
        checkpointer=InMemorySaver(),
    )
    agent = harness.wrap(graph, id="refunds")
    paused = await agent.run("refund o1", user="u1", thread="th")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert refunded == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "refund of o1 done"
    assert refunded == ["o1"]
    state = await graph.aget_state({"configurable": {"thread_id": "th"}})
    assert "/notes.md" in state.values["files"]


async def test_a_tool_of_the_deep_agents_own_asking_with_interrupt_offers_what_it_says(
    harness: Harness,
) -> None:
    from langchain_core.tools import tool as langchain_tool
    from langgraph.types import interrupt

    @langchain_tool
    def pick_plans(customer: str) -> str:
        """Ask sales which plans to offer the customer."""
        picked = interrupt(
            {
                "question": f"Which plans for {customer}?",
                "options": [{"value": "a", "label": "Plan A"}, "b"],
                "multiple": True,
                "component": "plan-picker",
                "props": {"customer": customer},
            }
        )
        return ",".join(picked)

    model = ScriptedChatModel(turns=[("pick_plans", {"customer": "acme"}), "offered a and b"])
    graph = create_deep_agent(model=model, tools=[pick_plans], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="sales")
    paused = await agent.run("offer plans to acme", user="u1", thread="sales-1")
    asked = paused.interrupt
    assert asked is not None and asked.multiple and asked.option_values == ["a", "b"]
    assert (asked.component, asked.props) == ("plan-picker", {"customer": "acme"})
    done = await agent.resume(asked.interrupt_id, "answer", answer=["a", "b"], reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "offered a and b"
    state = await graph.aget_state({"configurable": {"thread_id": "sales-1"}})
    assert any(getattr(m, "content", None) == "a,b" for m in state.values["messages"])
