"""LangGraph: real graphs (``create_agent`` and a hand-built ``StateGraph``) driven by a
scripted chat model. Deep Agents has its own file."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import pytest
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from trellis import Harness, current, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus

executed: list[str] = []


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    executed.append(f"stock:{sku}")
    return 7


@tool(side_effects="irreversible")
def order(sku: str, qty: int) -> str:
    """Order more of a SKU."""
    executed.append(f"order:{sku}:{qty}")
    return f"ordered {qty} {sku}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    executed.clear()


async def agent_graph(harness: Harness, model: ScriptedChatModel, **kwargs: Any) -> Any:
    tools = await harness.tools(stock, order, framework="langgraph")
    return create_agent(model, tools=tools, system_prompt="You manage stock.", **kwargs)


async def test_a_create_agent_graph_runs_with_harness_tools(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7 units of a"])
    agent = harness.wrap(await agent_graph(harness, model), id="stock")
    result = await agent.run("how many a?", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "7 units of a"
    assert executed == ["stock:a"]


async def test_streaming_carries_text_and_tool_events(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7 units of a"])
    agent = harness.wrap(await agent_graph(harness, model), id="stock")
    events = [e async for e in agent.stream("how many a?", user="u1")]
    kinds = [e.type for e in events]
    assert kinds[0] is RunEventType.RUN_STARTED and kinds[-1] is RunEventType.RUN_FINISHED
    assert RunEventType.TOOL_CALL_RESULT in kinds
    text = "".join(
        e.data.get("delta", "") for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT
    )
    assert text == "7 units of a"


async def test_with_a_checkpointer_an_approval_is_a_langgraph_interrupt(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("order", {"sku": "a", "qty": 5}), "ordered"])
    graph = await agent_graph(harness, model, checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="buyer")
    paused = await agent.run("order 5 a", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.args == {
        "sku": "a",
        "qty": 5,
    }
    snapshot = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert snapshot.interrupts  # LangGraph holds the pause itself
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "ordered"
    assert executed == ["order:a:5"]
    assert len(model.seen) == 2  # resumed in place: the model was not asked again


async def test_without_a_checkpointer_the_run_is_replayed_from_the_journal(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[
            ("stock", {"sku": "a"}),
            ("order", {"sku": "a", "qty": 5}),
            ("stock", {"sku": "a"}),
            ("order", {"sku": "a", "qty": 5}),
            "ordered",
        ]
    )
    agent = harness.wrap(await agent_graph(harness, model), id="buyer")
    paused = await agent.run("top a up", user="u1")
    assert paused.interrupt is not None and executed == ["stock:a"]
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert executed == ["stock:a", "order:a:5"]  # the read was replayed, not re-run


class Draft(TypedDict):
    topic: str
    draft: NotRequired[str]
    verdict: NotRequired[str]


def write(state: Draft) -> Draft:
    return {"topic": state["topic"], "draft": f"about {state['topic']}"}


def own_question(state: Draft) -> Draft:
    return {"topic": state["topic"], "verdict": interrupt({"question": "Title?"})}


def bare_question(state: Draft) -> Draft:
    return {"topic": state["topic"], "verdict": interrupt("Title?")}


async def review(state: Draft) -> Draft:
    runtime = current()
    assert runtime is not None
    verdict = await runtime.ask(
        f"Publish the draft about {state['topic']}?", ui="diff", expects={"type": "string"}
    )
    return {"topic": state["topic"], "verdict": verdict}


def review_graph(checkpointer: Any = None) -> Any:
    graph = StateGraph(Draft)
    graph.add_node("write", write)
    graph.add_node("review", review)
    graph.add_edge(START, "write")
    graph.add_edge("write", "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.parametrize("checkpointer", [None, InMemorySaver()], ids=["replayed", "checkpointed"])
async def test_ask_inside_a_node_pauses_either_way(harness: Harness, checkpointer: Any) -> None:
    agent = harness.wrap(review_graph(checkpointer), id="editor")
    paused = await agent.run({"topic": "tides"}, user="u1", thread="t9")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.reason.value == "REVIEW" and paused.interrupt.ui == "diff"
    done = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="ship it", reviewer="u1"
    )
    assert done.status is RunStatus.SUCCESS


async def test_a_graphs_own_interrupt_is_resumed_with_the_raw_answer(harness: Harness) -> None:
    graph = StateGraph(Draft)
    graph.add_node("ask", own_question)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    agent = harness.wrap(graph.compile(checkpointer=InMemorySaver()), id="titler")
    paused = await agent.run({"topic": "x"}, user="u1", thread="t2")
    assert paused.interrupt is not None and paused.interrupt.question == "Title?"
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="Tides", reviewer="u1")
    record = await harness.runs.get(paused.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS


async def test_a_graphs_own_interrupt_needs_a_checkpointer(harness: Harness) -> None:
    graph = StateGraph(Draft)
    graph.add_node("ask", bare_question)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    result = await harness.wrap(graph.compile(), id="t").run({"topic": "x"}, user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "checkpointer" in result.error.message


async def test_the_memory_context_leads_the_messages(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChatModel(turns=["you prefer email"])
    graph = create_agent(model, tools=[], system_prompt="You help.")
    await memory_harness.wrap(graph, id="helper", memory="read").run("how to reach me?", user="u1")
    sent = model.seen[0]
    assert [m.type for m in sent][:2] == ["system", "system"]
    assert sent[1].content == memory_service.context_text


async def test_memory_tools_are_built_in_with_h_tools(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    tools = await memory_harness.tools(stock, framework="langgraph", memory=True)
    assert [t.name for t in tools] == ["stock", "memory_search", "memory_remember"]
    model = ScriptedChatModel(turns=[("memory_search", {"query": "x"}), "found"])
    graph = create_agent(model, tools=tools)
    result = await memory_harness.wrap(graph, id="m", memory="read_write").run("x", user="u1")
    assert result.answer == "found"
    assert memory_service.named("call_agent_tool")[0][1]["name"] == "memory_search"


async def test_a_compiled_graph_refuses_tools_at_wrap(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match=r"h\.tools"):
        harness.wrap(review_graph(), id="x", tools=[stock])
