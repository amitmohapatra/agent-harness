"""A tool of the graph's own that returns a ``Command`` — LangGraph's way for a tool to update
the graph's state, or to send it elsewhere — and the list of them and messages a tool may
return: the harness governs and journals the call and hands the graph what the tool made, on
the first run and when a later attempt replays the call from the journal (a resume in place
running a step again, a resume without a checkpointer running the graph from its input). A
Deep Agent's ``task`` is such a tool: what its sub-agent wrote is the update it returns."""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any, NotRequired

import pytest
from deepagents import SubAgent, create_deep_agent
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, ToolRuntime, tools_condition
from langgraph.types import Command

from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, current
from trellis.contracts import RunEventType, RunStatus

#: These tests are about what a write does once it runs, not whether it asks.
pytestmark = pytest.mark.usefixtures("writes_unasked")

ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


class Cart(MessagesState):
    cart: NotRequired[Annotated[list[Any], operator.add]]


class CartState(AgentState[Any]):
    cart: NotRequired[Annotated[list[Any], operator.add]]


class Carts(AgentMiddleware[CartState, Any]):
    """A ``create_agent`` graph's state with a cart (a middleware's ``state_schema``)."""

    state_schema = CartState


@lc_tool
def add(sku: str, runtime: ToolRuntime) -> Command[Any]:
    """Add a SKU to the cart."""
    ran.append(f"add {sku}")
    said = ToolMessage(f"added {sku}", tool_call_id=runtime.tool_call_id or "")
    return Command(update={"cart": [sku], "messages": [said]})


@lc_tool
async def add_slowly(sku: str, runtime: ToolRuntime) -> Command[Any]:
    """Add a SKU to the cart, taking a while."""
    await asyncio.sleep(0.05)
    ran.append(f"add {sku}")
    said = ToolMessage(f"added {sku}", tool_call_id=runtime.tool_call_id or "")
    return Command(update={"cart": [sku], "messages": [said]})


@lc_tool
def add_two(runtime: ToolRuntime) -> list[Any]:
    """Add two SKUs to the cart."""
    ran.append("add two")
    said = ToolMessage("added a and b", tool_call_id=runtime.tool_call_id or "")
    return [Command(update={"cart": ["a"]}), Command(update={"cart": ["b"]}), said]


@lc_tool
def keep(runtime: ToolRuntime) -> Command[Any]:
    """Put in the cart what no serializer takes."""
    ran.append("keep")
    said = ToolMessage("kept", tool_call_id=runtime.tool_call_id or "")
    return Command(update={"cart": [lambda: None], "messages": [said]})


@lc_tool
def cart(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """What is in the cart."""
    return ",".join(str(item) for item in state.get("cart") or []) or "empty"


@lc_tool
def pay(amount: int) -> str:
    """Pay."""
    ran.append(f"pay {amount}")
    return f"paid {amount}"


@lc_tool
def buy(sku: str, runtime: ToolRuntime) -> Command[Any]:
    """Buy a SKU: it goes in the cart."""
    ran.append(f"buy {sku}")
    said = ToolMessage(f"bought {sku}", tool_call_id=runtime.tool_call_id or "")
    return Command(update={"cart": [sku], "messages": [said]})


add.metadata = add_slowly.metadata = add_two.metadata = keep.metadata = {"side_effects": "write"}
cart.metadata = {"side_effects": "read"}
pay.metadata = buy.metadata = {"side_effects": "irreversible"}


def by_hand(model: Any, tools: list[Any], **compile: Any) -> Any:
    """A graph built by hand: a model node, ``ToolNode(tools)``, ``tools_condition`` — its
    tool node runs a step's calls together."""
    bound = model.bind_tools(tools)

    async def think(state: Any) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    graph = StateGraph(Cart)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile(**compile)


async def cart_of(harness: Harness, graph: Any, thread: str) -> list[Any]:
    key = await harness.conversation(user="u", thread=thread)
    state = await graph.aget_state({"configurable": {"thread_id": key}})
    return state.values.get("cart") or []


def answered(model: ScriptedChatModel) -> None:
    """Every call the model made has its result under the call's own id, in its last request."""
    sent = model.seen[-1]
    made = [c["id"] for m in sent if m.type == "ai" for c in getattr(m, "tool_calls", [])]
    assert made == [m.tool_call_id for m in sent if isinstance(m, ToolMessage)]


async def test_a_tool_returning_several_commands_has_each_applied(harness: Harness) -> None:
    """A tool may return a list of ``Command``\\ s and messages: each update is the state's, and
    the call is one call of the run, its result what the list's message says."""
    model = ScriptedChatModel(turns=[("add_two", {}), "added"])
    graph = by_hand(model, [add_two], checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(graph, id="cart").stream("two", user="u", thread="t")]
    said = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [(d["tool"], d["status"], d["output"]) for d in said] == [
        ("add_two", "ok", "added a and b")
    ]
    assert await cart_of(harness, graph, "t") == ["a", "b"]
    assert model.seen[-1][-1].content == "added a and b"


@pytest.mark.parametrize("several", [False, True], ids=["a command", "a list of commands"])
async def test_a_call_replayed_after_a_pause_gives_the_graph_its_update_again(
    harness: Harness, several: bool
) -> None:
    """Without a checkpointer a resume runs the graph again from its input: the calls made
    before the pause are answered from the journal, none run twice — and the state update a
    call made is the state's again, its message under the id the model gives the call now."""
    first = ("add_two", {}) if several else ("add", {"sku": "a"})
    turns: list[Any] = [first, ("pay", {"amount": 5})]
    model = ScriptedChatModel(turns=[*turns, *turns, ("cart", {}), "paid"])
    agent = harness.wrap(by_hand(model, [add, add_two, pay, cart]), id="cart")
    paused = await agent.run("add and pay 5", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "paid"
    assert ran == ["add two" if several else "add a", "pay 5"]
    assert model.seen[-1][-1].content == ("a,b" if several else "a")
    answered(model)


async def test_a_create_agent_graphs_call_replayed_gives_its_update_again(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[
            ("add", {"sku": "a"}),
            ("pay", {"amount": 5}),
            ("add", {"sku": "a"}),
            ("pay", {"amount": 5}),
            ("cart", {}),
            "paid",
        ]
    )
    graph = create_agent(model, tools=[add, pay, cart], middleware=[Carts()])
    agent = harness.wrap(graph, id="cart")
    paused = await agent.run("add a and pay 5", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "paid" and ran == ["add a", "pay 5"]
    assert model.seen[-1][-1].content == "a"
    answered(model)


async def test_what_a_deep_agents_sub_agent_wrote_is_there_after_a_replay(
    harness: Harness,
) -> None:
    """A Deep Agent's ``task`` returns its sub-agent's files as a ``Command``: replayed from
    the journal (no checkpointer: the resume runs the graph again), the files are the main
    agent's still, and the sub-agent did not work twice."""
    sub = ScriptedChatModel(
        turns=[("write_file", {"file_path": "/found.md", "content": "seven"}), "wrote it"]
    )
    turns: list[Any] = [
        ("task", {"description": "find it", "subagent_type": "finder"}),
        ("pay", {"amount": 5}),
    ]
    model = ScriptedChatModel(
        turns=[*turns, *turns, ("read_file", {"file_path": "/found.md"}), "found and paid"]
    )
    finder: SubAgent = {
        "name": "finder",
        "description": "Finds things.",
        "system_prompt": "Find it.",
        "model": sub,
        "tools": [],
    }
    graph = create_deep_agent(model=model, tools=[pay], subagents=[finder])
    agent = harness.wrap(graph, id="deep")
    paused = await agent.run("find it and pay 5", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "found and paid"
    assert len(sub.seen) == 2 and ran == ["pay 5"]
    assert "seven" in str(model.seen[-1][-1].content)


async def test_a_write_running_beside_a_call_that_asks_ends_before_the_run_pauses(
    harness: Harness,
) -> None:
    """A ``ToolNode`` runs a step's calls together and stops at the first that raises: a call
    that asks a person leaves the node once the others have ended, so the write beside it is
    journaled before the pause is recorded. The resume runs the step again: the write is
    answered from the journal — its update the state's — and only the approved call runs."""
    model = ScriptedChatModel(
        turns=[[("add_slowly", {"sku": "a"}), ("pay", {"amount": 5})], "paid"]
    )
    graph = by_hand(model, [add_slowly, pay], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="cart")
    paused = await agent.run("add a and pay 5", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert ran == ["add a"]  # ended before the pause, not behind it
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "paid"
    assert ran == ["add a", "pay 5"]
    assert await cart_of(harness, graph, "t") == ["a"]
    assert [m.content for m in model.seen[-1] if m.type == "tool"] == ["added a", "paid 5"]


async def test_two_calls_of_a_step_that_both_ask_are_approved_one_after_the_other(
    harness: Harness,
) -> None:
    """Both calls of a step ask: each waits for the other, neither runs, and the run pauses;
    approved one after the other, each runs once and its update is the state's."""
    model = ScriptedChatModel(turns=[[("buy", {"sku": "a"}), ("buy", {"sku": "b"})], "bought"])
    graph = by_hand(model, [buy], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="cart")
    result = await agent.run("buy a and b", user="u", thread="t")
    for _ in range(2):
        assert result.status is RunStatus.PAUSED and result.interrupt is not None
        result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "bought"
    assert sorted(ran) == ["buy a", "buy b"]
    assert sorted(await cart_of(harness, graph, "t")) == ["a", "b"]


async def test_a_call_that_asked_made_again_in_other_words_still_updates_the_state(
    harness: Harness,
) -> None:
    """A tool asks a person from within its call; the resume asks the model again, which words
    the call otherwise: it runs as the call that asked, and its ``Command`` is the graph's."""

    @lc_tool
    async def reserve(sku: str, runtime: ToolRuntime) -> Command[Any]:
        """Reserve a SKU for whoever a person names."""
        asking = current()
        assert asking is not None
        who = await asking.ask(f"Reserve {sku} for whom?")
        ran.append(f"reserve {sku} for {who}")
        said = ToolMessage(f"reserved {sku}", tool_call_id=runtime.tool_call_id or "")
        return Command(update={"cart": [sku], "messages": [said]})

    reserve.metadata = {"side_effects": "write"}
    model = ScriptedChatModel(
        turns=[("reserve", {"sku": "a"}), ("reserve", {"sku": " a "}), ("cart", {}), "reserved"]
    )
    agent = harness.wrap(by_hand(model, [reserve, cart]), id="cart")
    paused = await agent.run("reserve a", user="u")
    assert paused.interrupt is not None and paused.interrupt.question == "Reserve a for whom?"
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="ada", reviewer="u")
    assert done.answer == "reserved" and ran == ["reserve a for ada"]
    assert model.seen[-1][-1].content == "a"


async def test_an_update_no_serializer_takes_is_the_graphs_once_and_a_message_when_replayed(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """What a tool returned is kept as LangGraph serializes a step's writes. An update its
    serializer does not take is still the graph's when the tool runs; it cannot be kept for a
    replay, which is said, and the replayed call is the message the model read."""
    turns: list[Any] = [("keep", {}), ("pay", {"amount": 5})]
    model = ScriptedChatModel(turns=[*turns, *turns, ("cart", {}), "paid"])
    agent = harness.wrap(by_hand(model, [keep, pay, cart]), id="cart")
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        paused = await agent.run("keep and pay 5", user="u")
    assert paused.interrupt is not None
    assert "what keep returned in a message's place cannot be kept for a replay" in caplog.text
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "paid" and ran == ["keep", "pay 5"]
    assert model.seen[-1][-1].content == "empty"
    answered(model)
