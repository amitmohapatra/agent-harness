"""The serving surfaces against the real run store (and memory): AG-UI over SSE with a
reconnect that replays, and an A2A round trip over real HTTP between two harnesses."""

from __future__ import annotations

import asyncio
import socket
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import memory_scope
from trellis import Harness, Runtime, a2a
from trellis.contracts import RunStatus
from trellis.harness.surfaces.agui.sse import decode

pytestmark = [pytest.mark.live, needs_runs, needs_memory]


async def concierge(input: Any, agent: Runtime) -> str:
    if "book" in str(input):
        when = await agent.ask("Which evening?", options=["friday", "saturday"])
        return f"Booked for {when}."
    return f"You said {input}."


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "anonymous")


@pytest.fixture
async def chat() -> AsyncIterator[tuple[httpx.AsyncClient, Harness]]:
    async with live_harness() as h:
        app = FastAPI()
        agent = h.wrap(concierge, id=f"live-concierge-{uuid.uuid4().hex[:6]}")
        agent.serve_chat(app, identity=user_of)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://agui", headers={"x-user": "live-ada"}
        ) as client:
            yield client, h


async def test_agui_streams_pauses_resumes_and_replays(
    chat: tuple[httpx.AsyncClient, Harness],
) -> None:
    client, h = chat
    run_id, thread = f"run_{uuid.uuid4().hex}", f"live-chat-{uuid.uuid4().hex[:8]}"
    message = [{"id": "m1", "role": "user", "content": "book a table"}]
    first = await client.post(
        "/agui/run", json={"threadId": thread, "runId": run_id, "messages": message}
    )
    assert first.status_code == 200 and first.headers["content-type"].startswith(
        "text/event-stream"
    )
    events = decode(first.text)
    paused = events[-1][1]
    assert paused["outcome"]["type"] == "interrupt"
    entry = paused["outcome"]["interrupts"][0]
    record = await h.runs.get(run_id)  # the run is agent-runs', paused with its checkpoint
    assert record is not None and record.status is RunStatus.PAUSED
    assert record.checkpoint is not None

    # a client that dropped reconnects after the last event it saw
    replay = await client.get(f"/agui/runs/{run_id}/events", headers={"Last-Event-ID": "1"})
    assert decode(replay.text) == events[2:]

    resumed = await client.post(
        "/agui/run",
        json={"threadId": thread, "resume": [{"interruptId": entry["id"], "payload": "friday"}]},
    )
    done = decode(resumed.text)
    assert done[-1][1]["outcome"] == {"type": "success"}
    assert done[-1][1]["result"] == "Booked for friday."
    assert done[0][0] > events[-1][0]  # numbering continues across the attempts
    record = await h.runs.get(run_id)
    assert record is not None and record.status is RunStatus.SUCCESS and record.checkpoint is None
    await h.writes.drain()
    scope = await memory_scope(h, user="live-ada", agent_id=record.agent_id, thread=thread)
    history = await scope.history()
    assert [m.content for m in history][-1] == "Booked for friday."


# --------------------------------------------------------------------------- A2A
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def planner(input: Any, agent: Runtime) -> str:
    region = await agent.ask("Which region?", options=["eu", "us"])
    return f"deploying {input} to {region}"


@pytest.fixture
def remote_url() -> Iterator[str]:
    """The planner served over A2A by its own harness and uvicorn, on a real port."""
    port = free_port()
    url = f"http://127.0.0.1:{port}/a2a"
    harness = live_harness()
    app = FastAPI()
    harness.wrap(planner, id=f"live-planner-{uuid.uuid4().hex[:6]}").serve_a2a(app, url)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.05)
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_a2a_round_trip_with_a_remote_question(remote_url: str) -> None:
    async def caller(input: Any, agent: Runtime) -> Any:
        return await agent.tools.call("planner", message=input)

    async with live_harness() as h:
        agent = h.wrap(
            caller,
            id=f"live-caller-{uuid.uuid4().hex[:6]}",
            tools=[a2a(remote_url, name="planner")],
        )
        paused = await agent.run(
            "the shop", user="live-ada", thread=f"live-a2a-{uuid.uuid4().hex[:6]}"
        )
        # the remote agent's question became this run's own pause
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.question == "Which region?"
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="eu", reviewer="ada"
        )
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "deploying the shop to eu"
        await asyncio.sleep(0)
