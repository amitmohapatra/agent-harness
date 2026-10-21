"""Contract: what one approval (W1) and the governance of a graph's own tools (W3) read from
LangChain and LangGraph, checked against the installed releases rather than written by hand —
``HumanInTheLoopMiddleware``'s interrupt (``action_requests`` with each call's ``name`` and
``args``, ``review_configs`` with ``allowed_decisions``), the ``{"decisions": [...]}`` it
resumes with, and the ``ToolCallRequest`` a tool-call middleware gets (``tool_call``,
``tool``). A release that changes any of them fails here, not in a deployment."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.support.chat_model import ScriptedChatModel
from trellis.contracts import InterruptDecision, InterruptResolution
from trellis.harness.adapters.langgraph import hitl_response, is_hitl

ran: list[str] = []


@tool
def wire(to: str) -> str:
    """Wire money."""
    ran.append(to)
    return f"wired {to}"


async def test_the_middlewares_interrupt_and_resume_are_what_one_approval_reads() -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada"}), "wired"])
    graph = create_agent(
        model,
        tools=[wire],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"wire": True})],
        checkpointer=InMemorySaver(),
    )
    config: Any = {"configurable": {"thread_id": "t"}}
    paused = await graph.ainvoke({"messages": [HumanMessage("pay ada")]}, config)
    [interrupt] = paused["__interrupt__"]
    value = interrupt.value
    assert is_hitl(value)
    [action] = value["action_requests"]
    assert (action["name"], action["args"]) == ("wire", {"to": "ada"})
    [review] = value["review_configs"]
    assert review["action_name"] == "wire"
    assert {"approve", "edit", "reject"} <= set(review["allowed_decisions"])
    approve = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.APPROVE
    )
    done = await graph.ainvoke(Command(resume=hitl_response(value, approve)), config)
    assert ran == ["ada"] and done["messages"][-1].content == "wired"


async def test_a_tool_call_middleware_gets_the_call_and_the_tool() -> None:
    seen: list[tuple[str, str, str]] = []

    class Look(AgentMiddleware):
        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            seen.append((request.tool_call["name"], request.tool_call["id"], request.tool.name))
            assert request.tool_call["args"] == {"to": "bo"}
            return await handler(request)

    model = ScriptedChatModel(turns=[("wire", {"to": "bo"}), "wired"])
    graph = create_agent(model, tools=[wire], middleware=[Look()])
    await graph.ainvoke({"messages": [HumanMessage("pay bo")]})
    assert seen and seen[0][0] == "wire" and seen[0][2] == "wire"
