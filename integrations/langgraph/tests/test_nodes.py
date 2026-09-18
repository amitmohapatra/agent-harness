"""LangGraph adapter (§51-§55, §80): node semantics stay the node's own."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from tests.support import span_by_name, span_names

from universal_agent_harness import AgentResult, AgentRuntime


class State(TypedDict, total=False):
    question: str
    answer: str
    seen: Annotated[list[str], operator.add]


def config(thread_id: str = "chat-42", **configurable):
    return {"configurable": {"thread_id": thread_id, **configurable}}


# --------------------------------------------------------------------------- level 1


async def test_existing_node_is_wrapped_without_changing_its_contract(harness, spans):
    async def existing_node(state: State) -> dict:
        return {"answer": f"answered: {state['question']}"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(existing_node, agent_id="inventory-agent"))
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    app = graph.compile()

    out = await app.ainvoke({"question": "how much stock?"}, config())
    assert out["answer"] == "answered: how much stock?"
    assert "agent.run" in span_names(spans)
    assert span_by_name(spans, "agent.run").attributes["agent.framework"] == "langgraph"


async def test_sync_node_is_supported(harness):
    def sync_node(state: State) -> dict:
        return {"answer": "sync answer"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(sync_node, agent_id="sync-agent"))
    graph.add_edge(START, "answer")
    app = graph.compile()
    assert (await app.ainvoke({"question": "q"}, config()))["answer"] == "sync answer"


async def test_node_receiving_config_still_receives_it(harness):
    from langchain_core.runnables import RunnableConfig

    seen: dict = {}

    async def node(state: State, config: RunnableConfig) -> dict:
        seen["thread_id"] = config["configurable"]["thread_id"]
        return {"answer": "ok"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(node, agent_id="cfg-agent"))
    graph.add_edge(START, "answer")
    await graph.compile().ainvoke({"question": "q"}, config("chat-99"))
    assert seen["thread_id"] == "chat-99"


async def test_lineage_comes_from_the_config(harness, memory, spans):
    async def node(state: State) -> dict:
        return {"answer": "ok"}

    graph = StateGraph(State)
    graph.add_node(
        "answer", harness.langgraph.wrap_node(node, agent_id="inventory-agent", query="question")
    )
    graph.add_edge(START, "answer")
    await graph.compile().ainvoke({"question": "how much stock?"}, config("chat-7"))

    span = span_by_name(spans, "agent.run")
    assert span.attributes["thread.id"] == "chat-7"
    assert span.attributes["turn.id"].startswith("chat-7-step")
    assert span.attributes["task.id"]
    assert memory.retrievals[0]["scope"]["thread_id"] == "chat-7"


async def test_identity_overrides_travel_in_the_config(harness, memory):
    async def node(state: State) -> dict:
        return {"answer": "ok"}

    graph = StateGraph(State)
    graph.add_node(
        "answer", harness.langgraph.wrap_node(node, agent_id="inv", query="question")
    )
    graph.add_edge(START, "answer")
    await graph.compile().ainvoke(
        {"question": "q"},
        config("chat-1", harness={"tenant_id": "other-tenant", "user_id": "u9", "work_id": "w1"}),
    )
    scope = memory.retrievals[0]["scope"]
    assert scope["tenant_id"] == "other-tenant"
    assert scope["user_id"] == "u9"
    assert scope["work_id"] == "w1"


# --------------------------------------------------------------------------- level 2


async def test_runtime_aware_node_receives_the_agent_runtime(harness):
    @harness.langgraph.agent(
        agent_id="inventory-agent", skills=["inventory.analysis"], query="question"
    )
    async def inventory_node(state: State, agent: AgentRuntime) -> dict:
        assert isinstance(agent, AgentRuntime)
        assert agent.memory_context is not None
        return {"answer": f"{agent.agent_id} answered"}

    graph = StateGraph(State)
    graph.add_node("inventory", inventory_node)
    graph.add_edge(START, "inventory")
    out = await graph.compile().ainvoke({"question": "how much stock?"}, config())
    assert out["answer"] == "inventory-agent answered"


async def test_state_mapper_controls_the_state_update(harness):
    @harness.langgraph.agent(
        agent_id="inv", state_mapper=lambda result: {"answer": result.data["text"]}
    )
    async def node(state: State, agent: AgentRuntime) -> AgentResult:
        return AgentResult.ok({"text": "mapped answer"})

    graph = StateGraph(State)
    graph.add_node("inv", node)
    graph.add_edge(START, "inv")
    assert (await graph.compile().ainvoke({"question": "q"}, config()))["answer"] == "mapped answer"


# --------------------------------------------------------------------------- graph semantics


async def test_reducers_are_untouched_by_the_wrapper(harness):
    async def first(state: State) -> dict:
        return {"seen": ["first"]}

    async def second(state: State) -> dict:
        return {"seen": ["second"]}

    graph = StateGraph(State)
    graph.add_node("a", harness.langgraph.wrap_node(first, agent_id="a"))
    graph.add_node("b", harness.langgraph.wrap_node(second, agent_id="b"))
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    out = await graph.compile().ainvoke({"question": "q", "seen": []}, config())
    assert out["seen"] == ["first", "second"]  # the add reducer still accumulates


async def test_parallel_nodes_run_concurrently_and_merge(harness, spans):
    async def slow_a(state: State) -> dict:
        await asyncio.sleep(0.05)
        return {"seen": ["a"]}

    async def slow_b(state: State) -> dict:
        await asyncio.sleep(0.05)
        return {"seen": ["b"]}

    graph = StateGraph(State)
    graph.add_node("a", harness.langgraph.wrap_node(slow_a, agent_id="a"))
    graph.add_node("b", harness.langgraph.wrap_node(slow_b, agent_id="b"))
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", END)
    graph.add_edge("b", END)

    started = asyncio.get_running_loop().time()
    out = await graph.compile().ainvoke({"question": "q", "seen": []}, config())
    elapsed = asyncio.get_running_loop().time() - started

    assert sorted(out["seen"]) == ["a", "b"]
    assert elapsed < 0.09  # genuinely parallel, not serialized by the harness
    assert span_names(spans).count("agent.run") == 2


async def test_exceptions_propagate_to_the_graph(harness):
    class NodeFailure(Exception):
        pass

    async def failing(state: State) -> dict:
        raise NodeFailure("node blew up")

    graph = StateGraph(State)
    graph.add_node("boom", harness.langgraph.wrap_node(failing, agent_id="boom"))
    graph.add_edge(START, "boom")
    with pytest.raises(NodeFailure, match="node blew up"):
        await graph.compile().ainvoke({"question": "q"}, config())


async def test_cancellation_propagates_through_the_graph(harness):
    started = asyncio.Event()

    async def slow(state: State) -> dict:
        started.set()
        await asyncio.sleep(5)
        return {"answer": "never"}

    graph = StateGraph(State)
    graph.add_node("slow", harness.langgraph.wrap_node(slow, agent_id="slow"))
    graph.add_edge(START, "slow")
    app = graph.compile()

    task = asyncio.ensure_future(app.ainvoke({"question": "q"}, config()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_streaming_still_streams(harness):
    async def node(state: State) -> dict:
        return {"answer": "streamed"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(node, agent_id="inv"))
    graph.add_edge(START, "answer")
    app = graph.compile()

    updates = [chunk async for chunk in app.astream({"question": "q"}, config(), stream_mode="updates")]
    assert updates == [{"answer": {"answer": "streamed"}}]


async def test_custom_stream_writer_is_forwarded(harness):
    from langgraph.types import StreamWriter

    async def node(state: State, writer: StreamWriter) -> dict:
        writer("progress: 50%")
        return {"answer": "done"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(node, agent_id="inv"))
    graph.add_edge(START, "answer")
    app = graph.compile()

    chunks = [c async for c in app.astream({"question": "q"}, config(), stream_mode="custom")]
    assert "progress: 50%" in chunks


# --------------------------------------------------------------------------- checkpoints


async def test_checkpoint_retry_does_not_duplicate_memory_observations(harness, memory):
    """A superstep replayed after a failure must reuse the same run id, so the retried
    write deduplicates instead of writing twice (§42, §55, §80)."""
    attempts = 0

    async def flaky(state: State) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("transient upstream failure")
        return {"answer": "recovered"}

    graph = StateGraph(State)
    graph.add_node(
        "answer",
        harness.langgraph.wrap_node(flaky, agent_id="inv", query="question"),
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.001, retry_on=ConnectionError),
    )
    graph.add_edge(START, "answer")
    app = graph.compile(checkpointer=InMemorySaver())

    out = await app.ainvoke({"question": "how much stock?"}, config("chat-retry"))
    assert out["answer"] == "recovered"
    assert attempts == 2

    keys = [o["idempotency_key"] for o in memory.observations]
    assert len(keys) == len(set(keys)), "a replayed superstep produced duplicate memory writes"


async def test_run_id_is_stable_across_replays_of_the_same_superstep(harness):
    run_ids: list[str] = []

    async def node(state: State, agent: AgentRuntime) -> dict:
        run_ids.append(agent.run_id)
        return {"answer": "ok"}

    graph = StateGraph(State)
    graph.add_node("answer", harness.langgraph.wrap_node(node, agent_id="inv"))
    graph.add_edge(START, "answer")
    app = graph.compile(checkpointer=InMemorySaver())

    await app.ainvoke({"question": "q"}, config("chat-stable"))
    await app.ainvoke({"question": "q"}, config("chat-stable-2"))
    assert run_ids[0] != run_ids[1]  # different threads -> different runs

    # ...but the same position replayed gives the same run id
    from universal_agent_harness_langgraph.lineage import lineage_from_config

    lineage = lineage_from_config(
        {"configurable": {"thread_id": "t", "checkpoint_ns": "answer:task-1"}}
    )
    assert lineage.run_id_for("inv") == lineage.run_id_for("inv")


# --------------------------------------------------------------------------- subgraphs


async def test_subgraph_lineage_nests_agent_runs(harness, spans):
    async def inner_node(state: State) -> dict:
        return {"answer": "inner answer"}

    inner = StateGraph(State)
    inner.add_node("inner", harness.langgraph.wrap_node(inner_node, agent_id="inner-agent"))
    inner.add_edge(START, "inner")
    inner.add_edge("inner", END)

    outer = StateGraph(State)
    outer.add_node("research", inner.compile())
    outer.add_edge(START, "research")
    outer.add_edge("research", END)

    out = await outer.compile().ainvoke({"question": "q"}, config("chat-sub"))
    assert out["answer"] == "inner answer"

    span = span_by_name(spans, "agent.run")
    assert span.attributes["agent.id"] == "inner-agent"
    # the enclosing subgraph invocation is recorded as the parent run
    assert span.attributes.get("agent.parent_run.id")


# --------------------------------------------------------------------------- pause (HITL)


async def test_an_interrupt_suspends_the_run_instead_of_failing_it(harness, spans, memory):
    """``interrupt()`` raises to hand control to the graph runtime. That is the mechanism,
    not a failure, and the harness must not report it as one."""
    from langgraph.types import interrupt

    from universal_agent_harness.contracts.events import LifecycleEvent

    interesting = {str(LifecycleEvent.AGENT_PAUSE), str(LifecycleEvent.AGENT_ERROR)}
    seen: list[str] = []
    harness.events.add(lambda event, payload: seen.append(event) if event in interesting else None)

    async def approval_node(state: State) -> dict:
        decision = interrupt({"question": "approve the reorder?"})
        return {"answer": f"human said {decision}"}

    graph = StateGraph(State)
    graph.add_node("approve", harness.langgraph.wrap_node(approval_node, agent_id="approver"))
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    app = graph.compile(checkpointer=InMemorySaver())

    cfg = config("chat-hitl")
    result = await app.ainvoke({"question": "reorder 50?"}, config=cfg)

    # the graph is suspended, and the harness said so
    assert "__interrupt__" in result
    assert seen == ["on_agent_pause"], "a pause must not be reported as an error"

    run = span_by_name(spans, "agent.run")
    assert run.attributes["status"] == "PAUSED"
    assert run.status.status_code.name == "OK", "a suspended run is not an errored span"

    # ...and resuming finishes the turn normally
    from langgraph.types import Command

    resumed = await app.ainvoke(Command(resume="approved"), config=cfg)
    assert resumed["answer"] == "human said approved"
    assert seen == ["on_agent_pause"]
