"""The AG-UI surface over a real FastAPI app: run, pause, resume, reconnect, artifacts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import AgentError, RunEvent, RunEventType, RunOutcome, new_id
from trellis.harness.identity import Identity
from trellis.harness.surfaces.agui.events import AGUIEvent, AGUIEventType
from trellis.harness.surfaces.agui.hub import MAX_EVENTS_PER_RUN, Hub
from trellis.harness.surfaces.agui.sse import decode
from trellis.harness.surfaces.agui.translate import translate

PATH = "/agui"
RELEASE = asyncio.Event()


@tool(side_effects="irreversible")
async def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


async def agent_fn(input: Any, agent: Runtime) -> Any:
    if input == "ask":
        return f"you said {await agent.ask('Proceed?', options=['yes', 'no'])}"
    if input == "refund":
        return await agent.tools.call("refund", order="o1")
    if input == "table":
        rows = [{"n": i} for i in range(60)]
        return await agent.ask("Check these rows", table=rows)
    if input == "wait":
        await RELEASE.wait()
    return f"echo {input}"


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    harness = Harness(config=Settings())
    app = FastAPI()
    harness.wrap(agent_fn, id="chat", tools=[refund]).serve_chat(app, identity=user_of)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://agui", headers={"x-user": "u1"}
    ) as http:
        yield http
    await harness.aclose()


def body(text: str | None = None, **extra: Any) -> dict[str, Any]:
    messages = [{"id": "m1", "role": "user", "content": text}] if text else []
    return {"threadId": "t1", "messages": messages, **extra}


async def post(
    client: httpx.AsyncClient, payload: dict[str, Any]
) -> list[tuple[int, dict[str, Any]]]:
    response = await client.post(f"{PATH}/run", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return decode(response.text)


def finished(events: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    return events[-1][1]


async def test_a_run_streams_numbered_events_ending_in_its_result(
    client: httpx.AsyncClient,
) -> None:
    events = await post(client, body("hello", runId="run-a"))
    assert [n for n, _ in events] == list(range(len(events)))
    assert events[0][1]["type"] == "RUN_STARTED"
    last = finished(events)
    assert last["type"] == "RUN_FINISHED"
    assert last["runId"] == "run-a"
    assert last["outcome"] == {"type": "success"}
    assert last["result"] == "echo hello"


async def test_a_question_pauses_and_a_resume_entry_answers_it(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("ask")))
    assert paused["outcome"]["type"] == "interrupt"
    entry = paused["outcome"]["interrupts"][0]
    assert entry["reason"] == "choice"
    assert entry["message"] == "Proceed?"

    resumed = await post(client, body(resume=[{"interruptId": entry["id"], "payload": "yes"}]))
    first_number = resumed[0][0]
    assert first_number > 0  # the run keeps counting across attempts
    last = finished(resumed)
    assert last["outcome"] == {"type": "success"}
    assert last["result"] == "you said yes"


async def test_an_approval_is_answered_with_true(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("refund")))
    entry = paused["outcome"]["interrupts"][0]
    assert entry["reason"] == "approval"
    assert entry["metadata"]["tool_call"]["tool"] == "refund"

    rejected = await client.post(
        f"{PATH}/run", json=body(resume=[{"interruptId": entry["id"], "payload": "maybe"}])
    )
    assert rejected.status_code == 409

    resumed = await post(client, body(resume=[{"interruptId": entry["id"], "payload": True}]))
    assert finished(resumed)["result"] == "refunded o1"


async def test_a_cancelled_resume_ends_the_run_as_cancelled(client: httpx.AsyncClient) -> None:
    entry = finished(await post(client, body("ask")))["outcome"]["interrupts"][0]
    events = await post(client, body(resume=[{"interruptId": entry["id"], "status": "cancelled"}]))
    assert finished(events)["outcome"] == {"type": "cancelled"}


async def test_an_unknown_interrupt_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(
        f"{PATH}/run", json=body(resume=[{"interruptId": "run_nope.1.1", "payload": 1}])
    )
    assert response.status_code == 404


async def test_a_client_reconnects_after_the_last_event_it_saw(client: httpx.AsyncClient) -> None:
    events = await post(client, body("hello", runId="run-r"))
    replay = await client.get(f"{PATH}/runs/run-r/events", headers={"Last-Event-ID": "0"})
    assert decode(replay.text) == events[1:]
    by_query = await client.get(f"{PATH}/runs/run-r/events", params={"after": 0})
    assert decode(by_query.text) == events[1:]
    stranger = await client.get(f"{PATH}/runs/run-r/events", headers={"x-user": "u2"})
    assert stranger.status_code == 404


async def test_a_reconnect_follows_a_live_run_to_its_end(client: httpx.AsyncClient) -> None:
    RELEASE.clear()
    first = asyncio.create_task(client.post(f"{PATH}/run", json=body("wait", runId="run-live")))
    await asyncio.sleep(0.1)  # the run is open, waiting on RELEASE
    follower = asyncio.create_task(client.get(f"{PATH}/runs/run-live/events"))
    await asyncio.sleep(0.05)
    assert not follower.done()
    RELEASE.set()
    followed = decode((await follower).text)
    assert followed == decode((await first).text)
    assert finished(followed)["result"] == "echo wait"


async def test_a_large_table_travels_as_an_artifact(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("table")))
    reference = paused["outcome"]["interrupts"][0]["metadata"]["payload_ref"]
    content = await client.get(f"{PATH}/artifacts/{reference['artifact_id']}")
    assert content.status_code == 200
    assert content.headers["content-type"] == "application/json"
    assert len(content.json()) == 60
    assert (await client.get(f"{PATH}/artifacts/art_missing")).status_code == 404


# --------------------------------------------------------------------------- translate


def _context() -> Any:
    return Identity(
        tenant="t", user="u", agent_id="a", run_id=new_id("run_"), thread="th"
    ).context()


def test_run_error_is_folded_into_the_finish_event() -> None:
    context = _context()
    error = AgentError(code="Boom", message="it broke")
    assert translate(RunEvent.of(context, RunEventType.RUN_ERROR, 0, error=error)) is None
    finish = translate(RunEvent.finished(context, RunOutcome.ERROR, 1, error=error))
    assert finish is not None
    assert finish.type is AGUIEventType.RUN_ERROR
    assert (finish.message, finish.code) == ("it broke", "Boom")


def test_context_loaded_becomes_a_custom_event() -> None:
    event = translate(RunEvent.of(_context(), RunEventType.CONTEXT_LOADED, 0, data={"chars": 3}))
    assert event is not None
    assert (event.type, event.name, event.value) == (
        AGUIEventType.CUSTOM,
        "context_loaded",
        {"chars": 3},
    )


def test_tool_events_carry_their_call() -> None:
    context = _context()
    start = translate(RunEvent.tool(context, RunEventType.TOOL_CALL_START, "c1", 0, tool="refund"))
    result = translate(
        RunEvent.tool(context, RunEventType.TOOL_CALL_RESULT, "c1", 1, output={"ok": 1})
    )
    assert start is not None and result is not None
    assert (start.tool_call_id, start.tool_call_name) == ("c1", "refund")
    assert (result.role, result.content) == ("tool", '{"ok": 1}')


# --------------------------------------------------------------------------- hub


async def test_a_buffer_keeps_its_last_events_and_numbers_them() -> None:
    buffer = Hub().open("r", "u")
    for n in range(MAX_EVENTS_PER_RUN + 5):
        buffer.publish(AGUIEvent(type=AGUIEventType.CUSTOM, value=n))
    buffer.publish(AGUIEvent(type=AGUIEventType.RUN_FINISHED), final=True)
    read = [(n, e.value) async for n, e in buffer.read(-1)]
    assert len(read) == MAX_EVENTS_PER_RUN
    assert read[0][0] == 6
    assert read[-2] == (MAX_EVENTS_PER_RUN + 4, MAX_EVENTS_PER_RUN + 4)


def test_the_hub_forgets_finished_runs_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness.surfaces.agui import hub as module

    monkeypatch.setattr(module, "MAX_RUNS", 2)
    runs = Hub()
    live = runs.open("live", "u")
    done = runs.open("done", "u")
    done.publish(AGUIEvent(type=AGUIEventType.RUN_FINISHED), final=True)
    runs.open("new", "u")
    assert runs.get("done") is None
    assert runs.get("live") is live
