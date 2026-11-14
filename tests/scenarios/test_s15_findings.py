"""The scenario suite's findings that need agent-runs' own event log, each in its smallest failing
form: ``xfail(strict=True)`` until fixed (the day a fix lands the test passes, and must be
unmarked). Root causes and proposed fixes: docs/testing/TEST_PLAN.md, "Scenario findings"."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from tests.scenarios.conftest import harness as scenario_harness
from tests.scenarios.kit import asked_id
from tests.support.arm import arm
from trellis import Harness, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.middleware import RunCheckpointer

pytestmark = [pytest.mark.timeout(300), pytest.mark.plan("WF-02", "WF-03")]


@tool(side_effects="read")
def look(key: str) -> str:
    """Look a key up."""
    return f"value-of-{key}"


@tool(side_effects="irreversible")
def pay(amount: int) -> str:
    """Pay (asks a person)."""
    return f"paid {amount}"


async def _on_two_workers(harness: Harness, build: Any) -> tuple[Any, Any]:
    """The run queued, paused on a worker of one process, approved, and continued by a worker
    of another (a harness of its own on the same run store, the agent built again)."""
    agent = await build(harness)
    handle = await agent.start("go", user="ada", thread="t1")
    assert await harness.worker([agent], concurrency=1).run_once()
    paused = await handle.result(wait=10)
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    await agent.resume(asked_id(paused), "approve", reviewer="ada")
    async with scenario_harness() as other:
        again = await build(other)
        assert await other.worker([again], concurrency=1).run_once()
    done = await handle.result(wait=10)
    assert done.status is RunStatus.SUCCESS, done.error
    return agent, done


@pytest.mark.xfail(
    strict=True,
    reason="SCN-2: a checkpointed graph's calls after a resume reuse the ids of calls before it",
)
async def test_scn2_call_ids_are_unique_across_a_checkpointed_graphs_attempts(
    h: Harness, uid: str
) -> None:
    """A hand-built graph with the run's checkpointer calls tools from its nodes (no model call
    ids): the calls after an approval are numbered from 1 again (``bridge._called``: ``ref =
    call_id or f"{run_id}:{step}"``, the step counted per attempt), so the payment made after
    the resume carries the id of the lookup made before it — on the stream, in AG-UI's
    ``result:<call>`` message ids, and in what the memory service keeps under them."""
    from tests.support.arm import native_tools

    async def build(on: Harness) -> Any:
        tools = {t.name: t for t in await native_tools(on, look, pay, framework="langgraph")}

        async def first(state: MessagesState) -> dict[str, Any]:
            await tools["look"].ainvoke({"key": "a"})
            return {}

        async def second(state: MessagesState) -> dict[str, Any]:
            await tools["pay"].ainvoke({"amount": 3})
            return {"messages": [AIMessage("done")]}

        graph = StateGraph(MessagesState)
        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        return arm(
            on.wrap(graph.compile(checkpointer=RunCheckpointer()), id=f"scn2-{uid}"), [look, pay]
        )

    agent, done = await _on_two_workers(h, build)
    events = [e async for e in agent.events(done.run_id)]
    started = [
        (e.tool_call_id, e.data["tool"]) for e in events if e.type is RunEventType.TOOL_CALL_START
    ]
    ids = [i for i, _ in started]
    assert len(ids) == len(set(ids)), started


@pytest.mark.xfail(
    strict=True, reason="SCN-3: a graph run again after a resume announces its replayed calls again"
)
async def test_scn3_a_replayed_call_is_not_announced_again(h: Harness, uid: str) -> None:
    """A graph with no checkpointer is run again from its input after an approval; the calls
    made before the pause are replayed from the journal (they do not run again) but each is
    announced again on the stream — a second ``TOOL_CALL_START`` of one call
    (``bridge._called``: ``_events(runtime, ref, tool_call, outcome)`` for a replayed outcome)."""
    from tests.support.arm import native_tools

    async def build(on: Harness) -> Any:
        tools = {t.name: t for t in await native_tools(on, look, pay, framework="langgraph")}

        async def both(state: MessagesState) -> dict[str, Any]:
            await tools["look"].ainvoke({"key": "a"})
            await tools["pay"].ainvoke({"amount": 3})
            return {"messages": [AIMessage("done")]}

        graph = StateGraph(MessagesState)
        graph.add_node("both", both)
        graph.add_edge(START, "both")
        graph.add_edge("both", END)
        return arm(on.wrap(graph.compile(), id=f"scn3-{uid}"), [look, pay])

    agent, done = await _on_two_workers(h, build)
    events = [e async for e in agent.events(done.run_id)]
    started = [e.tool_call_id for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert len(started) == len(set(started)), started
