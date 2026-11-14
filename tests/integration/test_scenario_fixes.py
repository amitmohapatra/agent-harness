"""The fixes of the user-scenario suite's findings SCN-2, SCN-3, SCN-4 and SCN-6 (the test plan's
WF-02/03 and INV-10; CONV-01/02/07 and MEM-01; MEM-10 and INV-2), each in its smallest form, in
process (no service): a run paused on one worker and continued on a worker of another harness on
the same run store (``LoggedRuns``: agent-runs' event log in process), and a memory service that
holds a turn or is down (the in-process fake behind a transport with a gate, no clock). Each
test's docstring says what was wrong and what holds now."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from tests.support.arm import arm, native_tools
from tests.support.memory import URL, FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.runlog import LoggedRuns
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEvent, RunEventType, RunStatus
from trellis.harness import agent as agent_module
from trellis.harness import conversation
from trellis.harness import writes as writes_module
from trellis.harness.agent import Agent
from trellis.harness.agui.sse import decode
from trellis.harness.middleware import RunCheckpointer
from trellis.memory import MemoryClient

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


def a_function() -> Build:
    """A function target that calls ``look`` and then ``pay`` (``agent.tools.call``: no framework
    call ids either); run again from its input after the approval of ``pay``, ``look`` replayed."""

    async def build(on: Harness) -> Agent:
        async def target(input: Any, agent: Runtime) -> str:
            await agent.tools.call("look", key="a")
            await agent.tools.call("pay", amount=3)
            return "done"

        return arm(on.wrap(target, id="scn"), [look, pay])

    return build


#: the targets a resume runs again from their input: every call before the pause is replayed
RUN_AGAIN = pytest.mark.parametrize(
    "build", [two_nodes(checkpointed=False), a_function()], ids=["graph", "function"]
)


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


async def test_scn2_a_calls_id_is_what_the_call_is_however_its_attempt_got_to_it() -> None:
    """The id of ``pay`` says what the call is, not when in its attempt it ran: the same
    (after the run's id) in an attempt that went on from a checkpoint and made it first, and in
    one run again from its input that replayed ``look`` before it — a graph or a function."""
    named: dict[str, set[str]] = {"look": set(), "pay": set()}
    for build in (two_nodes(checkpointed=True), two_nodes(checkpointed=False), a_function()):
        runs, run_id, _ = await on_two_workers(build)
        for call_id, called in tool_events(runs.logged(run_id), RunEventType.TOOL_CALL_START):
            assert call_id is not None and call_id.startswith(f"{run_id}:"), call_id
            named[called].add(call_id.removeprefix(run_id))
    assert len(named["pay"]) == len(named["look"]) == 1 and named["pay"] != named["look"], named


async def test_scn2_the_same_call_made_twice_in_a_run_has_two_ids() -> None:
    """One tool called twice with the same arguments in a run: two calls, two ids (and two
    ``result:<call>`` messages) — what the call is includes which time the run makes it."""
    runs = LoggedRuns()
    async with Harness(config=Settings(), runs=runs) as h:

        async def twice(input: Any, agent: Runtime) -> str:
            return f"{await agent.tools.call('look', key='a')} {await agent.tools.call('look', key='a')}"

        done = await arm(h.wrap(twice, id="twice"), [look]).run("go", user="ada")
        results = [e for e in runs.logged(done.run_id) if e.type is RunEventType.TOOL_CALL_RESULT]
        assert len(results) == 2 and len({e.tool_call_id for e in results}) == 2, results
        assert len({e.message_id for e in results}) == 2, results


async def test_a_paused_and_approved_run_of_a_target_without_call_ids_records_each_call_once() -> (
    None
):
    """H-ID: a target whose framework names no call (a function: ``agent.tools.call``) makes
    the same read twice, then a payment that asks; approved, it is run again from its input,
    both reads replayed. Each call that ran is recorded once in the memory service, starts
    once on the run's stream under one id across both attempts, the two identical reads under
    two ids."""
    memory = FakeMemoryService()
    runs = LoggedRuns()
    async with Harness(config=Settings(), runs=runs, memory=memory.client()) as h:

        async def target(input: Any, agent: Runtime) -> str:
            await agent.tools.call("look", key="a")
            await agent.tools.call("look", key="a")
            await agent.tools.call("pay", amount=3)
            return "done"

        agent = arm(h.wrap(target, id="hid"), [look, pay])
        paused = await agent.run("go", user="ada", thread="t1")
        assert paused.interrupt is not None and paused.interrupt.tool_call is not None, paused
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="ada")
        assert done.status is RunStatus.SUCCESS, done.error
        await h.writes.drain()
    recorded = [call.body["tool"] for call in memory.named("record_tool")]
    assert recorded == ["look", "look", "pay"], recorded
    started = tool_events(runs.logged(done.run_id), RunEventType.TOOL_CALL_START)
    assert [tool for _, tool in started] == ["look", "look", "pay"], started
    assert len({call_id for call_id, _ in started}) == 3, started
    ended = tool_events(runs.logged(done.run_id), RunEventType.TOOL_CALL_RESULT)
    assert ended == started, ended


@RUN_AGAIN
async def test_scn3_a_replayed_call_is_not_announced_again(build: Build) -> None:
    """SCN-3: a target with no state of its own (a graph with no checkpointer, a function) is
    run again from its input after an approval; the call made before the pause is replayed from
    the journal (it does not run again) and is not announced again: its events are in the first
    attempt's part of the log already. The call the resumed attempt does run is announced."""
    runs, run_id, _ = await on_two_workers(build)
    events = runs.logged(run_id)
    for kind in (RunEventType.TOOL_CALL_END, RunEventType.TOOL_CALL_RESULT):
        assert [tool for _, tool in tool_events(events, kind)] == ["look", "pay"], kind
    told = [(e.data.get("tool"), e.attempt) for e in events if e.type in _TOOL]
    assert told.count(("look", 1)) == 2 and ("look", 2) not in told, told  # start and result
    assert ("pay", 2) in told, told


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


# --------------------------------------------------------------------- a slow or absent memory


def _memory(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> MemoryClient:
    return MemoryClient(
        URL,
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(base_url=URL, transport=httpx.MockTransport(handler)),
    )


def _holding(service: FakeMemoryService, kept: asyncio.Event) -> MemoryClient:
    """``service`` behind a transport that holds a turn's write until ``kept`` is set: a
    memory service slower than the run waits for it (no clock: the test says when it answers)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("POST", "/v1/messages"):
            await kept.wait()
        return service._checked(request)

    return _memory(handler)


@pytest.fixture
def kept(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Not set: memory holds each turn's write, and a run stops waiting for it at once
    (``TURN_WRITE_SECONDS``, 5 s, is 0.01 here: its turn is queued)."""
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.01)
    return asyncio.Event()


def _react(h: Harness, answers: list[str], id: str) -> tuple[Agent, ScriptedChat]:
    model = ScriptedChat(answers)
    return h.wrap(ReAct(system_prompt="You help.", model=model), id=id), model


async def test_scn4_the_next_turn_has_the_turn_before_when_memory_writes_slowly(
    kept: asyncio.Event,
) -> None:
    """SCN-4: a memory service that takes longer than ``TURN_WRITE_SECONDS`` to keep a turn:
    the run returns with its transcript queued (``Agent.recorded_run``), and the user's next
    message — sent at once, as people do — is answered from a window that has it all the same:
    the turns memory does not have yet come from the run store's records. Seen on the real
    stack under load: POST /v1/messages taking 3-7 s, a chat's third turn answered without its
    second."""
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        agent, model = _react(h, ["Noted.", "I do not know."], "scn4")
        first = await agent.run("The locker code is 4242.", user="ada", thread="t1")
        assert first.status is RunStatus.SUCCESS
        second = await agent.run("What is the locker code?", user="ada", thread="t1")
        assert second.status is RunStatus.SUCCESS
        sent = [(m["role"], m["content"]) for m in model.requests[-1]["messages"][1:]]
        kept.set()
        assert [text for _, text in sent[:2]] == ["The locker code is 4242.", "Noted."], sent
        assert len(sent) == 3 and sent[2][0] == "user", sent  # then the question, nothing twice


async def test_scn4_the_turn_before_is_said_once_when_memory_has_caught_up(
    kept: asyncio.Event,
) -> None:
    """The third turn, once memory kept the first two: each earlier turn once, not once from
    memory and again from the run store."""
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        agent, model = _react(h, ["Noted.", "Your code is 4242", "Yes."], "scn4b")
        await agent.run("The locker code is 4242.", user="ada", thread="t1")
        await agent.run("What is the locker code?", user="ada", thread="t1")
        kept.set()
        await h.writes.drain()
        await agent.run("Is that right?", user="ada", thread="t1")
        sent = str(model.requests[-1]["messages"])
        for said in ("The locker code is 4242.", "What is the locker code?", "Your code is 4242"):
            assert sent.count(said) == 1, (said, sent)


async def test_scn4_only_the_turns_memory_lacks_are_added_after_its_window(
    kept: asyncio.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory holds the first turn and not the second (its write still on its way): the third
    turn's window is memory's first turn, then the second from the run store — in the order
    said, each once — and only the second is read from the run store."""
    kept.set()
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        agent, model = _react(h, ["Noted.", "Fine.", "4242 and 9191."], "scn4c")
        held = await agent.run("The locker code is 4242.", user="ada", thread="t1")
        kept.clear()
        lacking = await agent.run("The gate code is 9191.", user="ada", thread="t1")
        read = Mock(wraps=h.runs.get)
        monkeypatch.setattr(h.runs, "get", read)
        await agent.run("Which codes?", user="ada", thread="t1")
        turns_read = [call.args[0] for call in read.call_args_list]
        # one run read for the turn memory lacks, none for the one it holds
        assert turns_read.count(lacking.run_id) == 1 and held.run_id not in turns_read
        said = [m["content"] for m in model.requests[-1]["messages"][1:]]
        kept.set()
        assert [text for text in said if isinstance(text, str)][:4] == [
            "The locker code is 4242.",
            "Noted.",
            "The gate code is 9191.",
            "Fine.",
        ], said
        assert len(said) == 5, said


async def test_scn4_the_turn_before_is_had_by_another_replica(kept: asyncio.Event) -> None:
    """The next message served by another replica (another harness on the same run store and
    memory service) while the first replica's write of the turn is still on its way."""
    service, runs = FakeMemoryService(), LoggedRuns()
    async with Harness(config=Settings(), memory=_holding(service, kept), runs=runs) as one:
        agent, _ = _react(one, ["Noted."], "r")
        await agent.run("The locker code is 4242.", user="ada", thread="t1")
        async with Harness(config=Settings(), memory=_holding(service, kept), runs=runs) as other:
            again, model = _react(other, ["4242."], "r")
            await again.run("What is the locker code?", user="ada", thread="t1")
            sent = str(model.requests[-1]["messages"])
            kept.set()
            assert sent.count("The locker code is 4242.") == 1, "not in the other's window"


async def test_scn4_another_users_turns_on_a_thread_named_alike_are_not_added(
    kept: asyncio.Event,
) -> None:
    """Two users' threads named alike are two conversations: the turn of one still on its way
    to memory is never added to the other's window from the run store."""
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        agent, model = _react(h, ["Noted.", "I do not know."], "users")
        await agent.run("The locker code is 4242.", user="ada", thread="t1")
        await agent.run("What is the locker code?", user="bob", thread="t1")
        sent = str(model.requests[-1]["messages"])
        kept.set()
        assert "4242" not in sent, sent


async def test_scn4_a_window_that_names_no_message_is_given_as_memory_gave_it(
    kept: asyncio.Event,
) -> None:
    """A window whose messages name no id (kept before the window gave them back) says nothing
    of which turns memory holds: nothing is added from the run store, so nothing is said
    twice — the turn still on its way is missing, as it was before."""
    from trellis.harness.history import MESSAGE_ID, RUN_RECORD

    kept.set()
    service = FakeMemoryService()
    async with Harness(config=Settings(), memory=_holding(service, kept)) as h:
        agent, model = _react(h, ["Noted.", "Fine.", "I do not know."], "legacy")
        await agent.run("The locker code is 4242.", user="ada", thread="t1")
        for thread in service.threads.values():
            for message in thread:
                message["custom_metadata"].get(RUN_RECORD, {}).pop(MESSAGE_ID, None)
        kept.clear()
        await agent.run("The gate code is 9191.", user="ada", thread="t1")
        await agent.run("Which codes?", user="ada", thread="t1")
        sent = str(model.requests[-1]["messages"])
        kept.set()
        assert sent.count("The locker code is 4242.") == 1 and "9191" not in sent, sent


async def test_scn4_a_run_store_that_cannot_be_read_is_a_warning(
    kept: asyncio.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run store failing while memory lacks the latest turn: a warning, and the window as
    memory gave it — the run goes on."""
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        agent, _ = _react(h, ["Noted.", "I do not know."], "unread")
        await agent.run("The locker code is 4242.", user="ada", thread="t1")
        listed = h.runs.iterate

        def unread(**asked: Any) -> Any:
            # the conversation's line lists the runs still working; the earlier turns are the
            # ended ones, and only their listing fails here
            if asked.get("status") in conversation.ASKED:
                raise RuntimeError("no agent-runs")
            return listed(**asked)

        monkeypatch.setattr(h.runs, "iterate", Mock(side_effect=unread))
        events = [e async for e in agent.stream("What is it?", user="ada", thread="t1")]
        kept.set()
        warned = [e.data for e in events if e.data.get("code") == "earlier_turns"]
        assert warned and "no agent-runs" in warned[0]["message"], warned
        assert events[-1].type is RunEventType.RUN_FINISHED


async def test_scn4_a_window_of_another_agents_turns_is_given_as_memory_gave_it(
    kept: asyncio.Event,
) -> None:
    """Two agents on one thread: the window holds the other agent's turn, which this agent's
    runs in the run store do not have — nothing anchors them, so nothing is added."""
    kept.set()
    async with Harness(config=Settings(), memory=_holding(FakeMemoryService(), kept)) as h:
        first, _ = _react(h, ["Noted."], "a")
        await first.run("The locker code is 4242.", user="ada", thread="t1")
        second, model = _react(h, ["Fine.", "I do not know."], "b")
        kept.clear()
        await second.run("The gate code is 9191.", user="ada", thread="t1")
        await second.run("Which codes?", user="ada", thread="t1")
        sent = str(model.requests[-1]["messages"])
        kept.set()
        assert sent.count("The locker code is 4242.") == 1 and "9191" not in sent, sent


async def test_scn6_a_turn_is_kept_once_memory_comes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCN-6: the memory service goes down for a while (every call answered 503: the client's
    circuit opens, refusing calls at once, and opens again after each probe that fails), then
    comes back: the turn made meanwhile is kept once it is back, and once only (the write sent
    again is the same write: the service keeps a message once by its id). Before, the write got
    3 tries with a short backoff and, with no spool (none in a container:
    ``settings.default_spool``), was given up — "failed and is lost: CircuitOpenError", as seen
    on the real stack for transcripts, tool records and outcomes while memory answered 504s.
    The outage here lasts until three probes of the open circuit have failed (the five calls
    that opened it, then one a cool-down): an event, no clock; a write's backoff is far
    shorter than the circuit's cool-down, as it is with their real values (0.5 s, 30 s)."""
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.1)
    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.001)
    service = FakeMemoryService()
    down, refused, outlasted = asyncio.Event(), [0], asyncio.Event()

    async def flaky(request: httpx.Request) -> httpx.Response:
        if not down.is_set():
            return service._checked(request)
        refused[0] += 1
        if refused[0] >= 5 + 3:
            outlasted.set()
        problem = {
            "title": "unavailable",
            "status": 503,
            "code": "DEPENDENCY_UNAVAILABLE",
            "retryable": True,
        }
        return httpx.Response(
            503, json=problem, headers={"content-type": "application/problem+json"}
        )

    client = _memory(flaky)
    client._transport.breaker.open_seconds = 0.2  # its cool-down (30 s) within the test
    async with Harness(config=Settings(), memory=client) as h:
        agent, _ = _react(h, ["Hello.", "Noted."], "scn6")
        await agent.run("Hi.", user="ada", thread="t1")
        await h.writes.drain()
        down.set()
        done = await agent.run("The locker code is 4242.", user="ada", thread="t1")
        assert done.status is RunStatus.SUCCESS
        async with asyncio.timeout(10):  # a bound for the test, never reached when it passes
            await outlasted.wait()
        assert h.writes.failed == 0, "the write was given up while memory was down"
        down.clear()
        await h.writes.drain()
        stored = [m for t in service.threads.values() for m in t if "4242" in m["content"]]
        assert stored, "the turn made while the memory service was down is lost"
        assert len(stored) == 1, stored  # sent again, kept once
