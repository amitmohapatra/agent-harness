"""The user-scenario suite's findings (SCN-2, SCN-3), each in its smallest form, in process (no
service): a graph's run paused on one worker and continued on a worker of another harness on the
same run store (``LoggedRuns``: agent-runs' event log in process). Each test's docstring says
what was wrong and what holds now."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from tests.support.arm import arm, native_tools
from tests.support.runlog import LoggedRuns
from trellis import Harness, Settings, tool
from trellis.contracts import RunEvent, RunEventType, RunStatus
from trellis.harness.agent import Agent
from trellis.harness.agui.sse import decode
from trellis.harness.middleware import RunCheckpointer

# --------------------------------------------------------------------- calls across attempts


@tool(side_effects="read")
def look(key: str) -> str:
    """Look a key up."""
    return f"value-of-{key}"


@tool(side_effects="irreversible")
def pay(amount: int) -> str:
    """Pay (asks a person)."""
    return f"paid {amount}"


Build = Callable[[Harness], Awaitable[Agent]]


def two_nodes(checkpointed: bool) -> Build:
    """A hand-built graph that calls ``look`` and then ``pay`` from its nodes (no model, so no
    framework call ids): with the run's checkpointer it goes on from its own state after the
    approval of ``pay``; without one it is run again from its input, ``look`` replayed."""

    async def build(on: Harness) -> Agent:
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
        compiled = graph.compile(checkpointer=RunCheckpointer() if checkpointed else None)
        return arm(on.wrap(compiled, id="scn"), [look, pay])

    return build


async def on_two_workers(build: Build) -> tuple[LoggedRuns, str, str]:
    """The run queued, paused on a worker of one harness, approved, and continued by a worker
    of another harness on the same run store (the agent built again there): the run store, the
    run's id and the id of the question it paused on."""
    runs = LoggedRuns()
    async with Harness(config=Settings(), runs=runs) as one:
        agent = await build(one)
        handle = await agent.start("go", user="ada", thread="t1")
        assert await one.worker([agent], concurrency=1).run_once()
        paused = await handle.result(wait=10)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
        asked = paused.interrupt.interrupt_id
        await agent.resume(asked, "approve", reviewer="ada")
    async with Harness(config=Settings(), runs=runs) as other:
        again = await build(other)
        assert await other.worker([again], concurrency=1).run_once()
    assert runs._runs[paused.run_id].status is RunStatus.SUCCESS
    return runs, paused.run_id, asked


_TOOL = frozenset({RunEventType.TOOL_CALL_START, RunEventType.TOOL_CALL_RESULT})


def tool_events(events: list[RunEvent], kind: RunEventType) -> list[tuple[str | None, Any]]:
    return [(e.tool_call_id, e.data.get("tool")) for e in events if e.type is kind]


async def test_scn2_call_ids_are_unique_across_a_checkpointed_graphs_attempts() -> None:
    """SCN-2: the calls a checkpointed graph makes after a resume are numbered by what they
    are, not by when in their attempt they ran — the attempt after the approval (another
    worker's) makes only ``pay``, whose id was once ``<run>:1``, the id of ``look`` made before
    the pause; on the stream, in AG-UI's ``result:<call>`` message ids, and in the memory
    service's records under them."""
    runs, run_id, _ = await on_two_workers(two_nodes(checkpointed=True))
    events = runs.logged(run_id)
    started = tool_events(events, RunEventType.TOOL_CALL_START)
    assert [tool for _, tool in started] == ["look", "pay"], started
    ids = [i for i, _ in started]
    assert len(set(ids)) == 2, started
    results = [e for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert len({e.message_id for e in results}) == 2, results  # ``result:<call>``, one each


async def test_scn2_a_calls_id_is_the_same_in_every_attempt() -> None:
    """The same call made again in a later attempt (a graph with no checkpointer, run again
    from its input) has the id it had: a client, the transcript and the memory service name
    one call one way, whichever attempt made it."""
    runs, run_id, asked = await on_two_workers(two_nodes(checkpointed=False))
    record = runs._runs[run_id]
    looked = [e for e in runs.logged(run_id) if e.data.get("tool") == "look"]
    assert {e.tool_call_id for e in looked} != {None}
    assert len({e.tool_call_id for e in looked}) == 1, looked
    assert record.status is RunStatus.SUCCESS and asked


async def test_scn3_a_replayed_call_is_not_announced_again() -> None:
    """SCN-3: a graph with no checkpointer is run again from its input after an approval; the
    call made before the pause is replayed from the journal (it does not run again) and is not
    announced again: its events are in the first attempt's part of the log already."""
    runs, run_id, _ = await on_two_workers(two_nodes(checkpointed=False))
    events = runs.logged(run_id)
    started = tool_events(events, RunEventType.TOOL_CALL_START)
    assert [tool for _, tool in started] == ["look", "pay"], started
    for kind in (RunEventType.TOOL_CALL_END, RunEventType.TOOL_CALL_RESULT):
        assert [tool for _, tool in tool_events(events, kind)] == ["look", "pay"], kind
    attempts = {e.data.get("tool"): e.attempt for e in events if e.type in _TOOL}
    assert attempts == {"look": 1, "pay": 2}, attempts


async def _served(runs: LoggedRuns, build: Build, call: Callable[[httpx.AsyncClient], Any]) -> Any:
    """``call`` made to the AG-UI surface of a replica of its own on the run store ``runs``."""
    from fastapi import FastAPI

    async with Harness(config=Settings(), runs=runs) as replica:
        app = FastAPI()
        (await build(replica)).serve_chat(app, identity=lambda request: "ada")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agui") as client:
            return await call(client)


async def test_scn3_agui_clients_of_a_run_replayed_on_another_replica_see_each_call_once() -> None:
    """The AG-UI clients of that run: one that reconnects after the run ended (a replica that
    did not serve it: agent-runs' event log) reads each call once — its start, arguments, end
    and result — and the activity line counts two tools; a fresh client of the resumed attempt
    only (the ``resume`` answered on another replica than the one that paused) reads a coherent
    stream: each call it is told of starts and ends, and the call replayed from the earlier
    attempt is not told again."""
    runs = LoggedRuns()
    build = two_nodes(checkpointed=False)

    async def post(client: httpx.AsyncClient, body: dict[str, Any]) -> list[dict[str, Any]]:
        response = await client.post("/agui/run", json={"threadId": "t1", **body})
        assert response.status_code == 200, response.text
        return [event for _, event in decode(response.text)]

    async def reconnect(client: httpx.AsyncClient) -> list[dict[str, Any]]:
        response = await client.get(f"/agui/runs/{run_id}/events")
        assert response.status_code == 200, response.text
        return [event for _, event in decode(response.text)]

    said = [{"id": "u1", "role": "user", "content": "go"}]
    first = await _served(runs, build, lambda c: post(c, {"messages": said}))
    paused = first[-1]
    assert paused["type"] == "RUN_FINISHED", paused
    [entry] = paused["outcome"]["interrupts"]
    run_id = paused["runId"]
    answer = {"messages": [], "resume": [{"interruptId": entry["id"], "payload": True}]}
    fresh = await _served(runs, build, lambda c: post(c, answer))
    assert fresh[-1]["type"] == "RUN_FINISHED" and fresh[-1]["result"] == "done", fresh[-1]
    reconnected = await _served(runs, build, reconnect)

    def calls(stream: list[dict[str, Any]], kind: str) -> list[str]:
        return [e.get("toolCallName") or e["toolCallId"] for e in stream if e["type"] == kind]

    assert calls(fresh, "TOOL_CALL_START") == ["pay"], fresh
    assert calls(fresh, "TOOL_CALL_END") == calls(fresh, "TOOL_CALL_RESULT")
    assert len(calls(fresh, "TOOL_CALL_RESULT")) == 1, fresh
    assert calls(reconnected, "TOOL_CALL_START") == ["look", "pay"], reconnected
    results = calls(reconnected, "TOOL_CALL_RESULT")
    assert len(results) == len(set(results)) == 2, reconnected
    assert calls(reconnected, "TOOL_CALL_END") == results
    activity = [e for e in reconnected if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert activity[-1]["content"]["tools"] == 2, activity


async def test_scn3_the_activity_line_of_a_run_resumed_where_it_paused_counts_every_call() -> None:
    """The client that answered the pause on the replica that served it: the resumed
    attempt's activity line counts the call made before the pause too (the run's calls, as
    a reconnect counts them from the log), not only the attempt's own."""

    async def both(client: httpx.AsyncClient) -> list[dict[str, Any]]:
        said = {"threadId": "t1", "messages": [{"id": "u1", "role": "user", "content": "go"}]}
        paused = [e for _, e in decode((await client.post("/agui/run", json=said)).text)][-1]
        [entry] = paused["outcome"]["interrupts"]
        resume = {"interruptId": entry["id"], "payload": True}
        answer = {"threadId": "t1", "messages": [], "resume": [resume]}
        return [e for _, e in decode((await client.post("/agui/run", json=answer)).text)]

    resumed = await _served(LoggedRuns(), two_nodes(checkpointed=False), both)
    activity = [e for e in resumed if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert activity[-1]["content"]["tools"] == 2, activity
