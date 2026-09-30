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
        tools=await harness.tools(refund, framework="langgraph"),
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
