"""SURF against the real services, over real HTTP and real processes:

* AG-UI on two replicas of one deployment (two harness processes' apps, each served by uvicorn):
  a chat run started on one pauses, a client reconnects to the other, which replays the run's
  events from agent-runs' log and resumes it; the thread stream of either replays the thread's
  runs; the surface's own refusals are its documented problem documents (401, 404, 409, 422);
* A2A: the card is public and holds no credential; a call without the trusted identity header,
  or with one past 4096 characters, is 401 before anything runs;
* the worker CLI (``python -m trellis.harness.worker module:h``): claims and finishes a queued
  run, and on SIGTERM releases the run it still holds back to the queue and exits 0;
* doctor (``python -m trellis.harness.doctor``) against the stack: the memory service and
  agent-runs are found and the key's tenant and role are said.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from tests.features.conftest import API_KEY, eventually, harness, runs_http
from tests.features.kit import record
from trellis import Runtime
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = pytest.mark.timeout(300)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def concierge(input: Any, agent: Runtime) -> str:
    if "book" in str(input):
        when = await agent.ask("Which evening?", options=["friday", "saturday"])
        return f"Booked for {when}."
    return f"You said {input}."


class Served:
    """An app served by uvicorn on a port of its own, in this test's event loop (the harness
    the app uses lives in this loop: a server thread of its own would share its clients
    across loops)."""

    def __init__(self, app: FastAPI, port: int | None = None) -> None:
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Served:
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:  # noqa: ASYNC110 - uvicorn offers no event to await
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.server.should_exit = True
        if self.task is not None:
            await self.task


async def read_until_paused(
    c: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> list[tuple[Any, Any]]:
    """A run's AG-UI event stream read until the event that pauses it (the stream itself stays
    open for the run's next attempt), decoded as ``decode`` does."""
    text = ""
    async with c.stream("GET", url, headers=headers) as s:
        assert s.status_code == 200, await s.aread()
        async for chunk in s.aiter_text():
            text += chunk
            if '"interrupt"' in text and text.endswith("\n\n"):
                break
    return decode(text)


def identity(user: str) -> dict[str, str]:
    return {"x-trellis-identity": json.dumps({"user_id": user})}


@pytest.mark.plan("SURF", "F62", "CONV-03")
async def test_agui_a_run_paused_on_one_replica_is_replayed_and_resumed_on_the_other(
    uid: str,
) -> None:
    agent_id, user, thread = f"chat-{uid}", f"gwen-{uid}", f"t-{uid}"
    run_id = f"run_{uid}{uid}"
    async with harness() as h1, harness() as h2:
        apps = []
        for h in (h1, h2):
            app = FastAPI()
            h.wrap(concierge, id=agent_id).serve_chat(app)
            apps.append(app)
        async with (
            Served(apps[0]) as one,
            Served(apps[1]) as two,
            httpx.AsyncClient(timeout=60) as c,
        ):
            message = [{"id": "m1", "role": "user", "content": "book a table"}]
            first = await c.post(
                f"{one.url}/agui/run",
                headers=identity(user),
                json={"threadId": thread, "runId": run_id, "messages": message},
            )
            assert first.status_code == 200, first.text
            events = decode(first.text)
            paused = events[-1][1]
            assert paused["outcome"]["type"] == "interrupt"
            interrupt_id = paused["outcome"]["interrupts"][0]["id"]
            # the other replica: replays the run from agent-runs' log, after event 1
            # (a paused run's stream stays open for its next attempt: read up to the pause)
            replayed = await read_until_paused(
                c, f"{two.url}/agui/runs/{run_id}/events", {**identity(user), "Last-Event-ID": "1"}
            )
            again = [(e["type"], e.get("name")) for _, e in replayed]
            assert again[-1] == ("RUN_FINISHED", None), again  # up to the pause, from the log
            assert replayed[-1][1]["outcome"]["interrupts"][0]["id"] == interrupt_id
            # someone else's run is not theirs to read, nor to cancel
            other = await c.get(
                f"{two.url}/agui/runs/{run_id}/events", headers=identity(f"hal-{uid}")
            )
            assert other.status_code == 404
            nope = await c.post(
                f"{two.url}/agui/runs/{run_id}/cancel", headers=identity(f"hal-{uid}")
            )
            assert nope.status_code == 404
            # problem documents: no identity, a bad mode, an answer that does not fit
            anon = await c.post(
                f"{two.url}/agui/run", json={"threadId": thread, "messages": message}
            )
            assert anon.status_code == 401
            assert anon.headers["content-type"].startswith("application/problem+json")
            bad_mode = await c.post(
                f"{two.url}/agui/run",
                headers=identity(user),
                json={
                    "threadId": f"{thread}-x",
                    "messages": message,
                    "forwardedProps": {"mode": "shout"},
                },
            )
            assert bad_mode.status_code == 422
            misfit = await c.post(
                f"{two.url}/agui/run",
                headers=identity(user),
                json={
                    "threadId": thread,
                    "resume": [{"interruptId": interrupt_id, "payload": "sunday"}],
                },
            )
            assert misfit.status_code == 409 and "BAD_RESUME" in misfit.text
            # resumed on the other replica
            resumed = await c.post(
                f"{two.url}/agui/run",
                headers=identity(user),
                json={
                    "threadId": thread,
                    "resume": [{"interruptId": interrupt_id, "payload": "friday"}],
                },
            )
            done = decode(resumed.text)
            assert done[-1][1]["outcome"] == {"type": "success"}
            assert done[-1][1]["result"] == "Booked for friday."
            # (its numbering starts again on this replica: H-AGUI, the strict xfail below)
            ended = await c.post(f"{one.url}/agui/runs/{run_id}/cancel", headers=identity(user))
            assert ended.status_code == 409  # it ended
    assert (await record(run_id)).status is RunStatus.SUCCESS


@pytest.mark.plan("SURF", "F63", "SEC-06")
async def test_a2a_the_card_is_public_and_a_call_needs_a_trusted_identity(uid: str) -> None:
    async with harness() as h:
        app = FastAPI()
        agent = h.wrap(concierge, id=f"a2a-{uid}", description="Books tables.")
        port = free_port()
        url = f"http://127.0.0.1:{port}/a2a"
        agent.serve_a2a(app, url)
        async with Served(app, port), httpx.AsyncClient(timeout=60) as c:
            card = await c.get(f"{url}/.well-known/agent-card.json")
            assert card.status_code == 200, card.text
            assert "Books tables." in card.text and (API_KEY or "x") not in card.text
            call = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "m1",
                        "role": "ROLE_USER",
                        "parts": [{"text": "hello"}],
                    }
                },
            }
            anon = await c.post(url, json=call)
            assert anon.status_code == 401
            long_identity = {"x-trellis-identity": json.dumps({"user_id": "x" * 4100})}
            too_long = await c.post(url, json=call, headers=long_identity)
            assert too_long.status_code == 401
            foreign = await c.post(
                url,
                json=call,
                headers={
                    "x-trellis-identity": json.dumps(
                        {"tenant_id": "someone-else", "user_id": "ivo"}
                    )
                },
            )
            assert foreign.status_code == 401
            ok = await c.post(
                url, json=call, headers={**identity(f"ivo-{uid}"), "A2A-Version": "1.0"}
            )
            assert ok.status_code == 200, ok.text
            assert "You said hello" in ok.text, ok.text


WORKER_MODULE = """
import asyncio, os
from tests.support.invariants import record_in_process
from tests.features.conftest import harness

record_in_process()
h = harness()


async def slow(input, agent):
    if input == "slow":
        await asyncio.sleep(120)
    with open(os.environ["FEAT_LEDGER"], "a") as out:
        out.write(f"worked:{input}\\n")
    return f"did {input}"


h.wrap(slow, id=os.environ["FEAT_AGENT"])
"""


@pytest.mark.plan("SURF", "F03", "LR-04")
@pytest.mark.timeout(240)
async def test_the_worker_cli_works_the_queue_and_releases_on_sigterm(
    tmp_path: Path, uid: str, invariants: Any
) -> None:
    agent_id = f"cli-{uid}"
    (tmp_path / f"feat_worker_{uid}.py").write_text(WORKER_MODULE)
    ledger = tmp_path / "ledger"
    env = {
        **os.environ,
        "FEAT_AGENT": agent_id,
        "FEAT_LEDGER": str(ledger),
        "PYTHONPATH": f"{tmp_path}:{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}",
        "SCEN_TRAFFIC": str(invariants.traffic_file("worker")),
    }
    slow = None
    process = subprocess.Popen(  # noqa: ASYNC220 - the worker outlives this call: no wait
        [
            sys.executable,
            "-m",
            "trellis.harness.worker",
            f"feat_worker_{uid}:h",
            "--concurrency",
            "2",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        async with harness() as h:

            async def unused(input: Any, agent: Runtime) -> Any:
                return input  # the worker process runs the agent; this side only starts runs

            agent = h.wrap(unused, id=agent_id)
            quick = await agent.start("quick", user=f"jo-{uid}")
            done = await asyncio.wait_for(quick.result(), 90)
            assert done.status is RunStatus.SUCCESS and done.answer == "did quick", done
            slow = await agent.start("slow", user=f"jo-{uid}")

            async def running() -> bool:
                return (await record(slow.id)).status is RunStatus.RUNNING

            assert await eventually(running, within=60)
        os.killpg(process.pid, signal.SIGTERM)
        code = process.wait(timeout=60)  # 25 s grace, then the slow run is released
        assert code == 0, process.stderr.read().decode()[-2000:] if process.stderr else ""
        released = await record(slow.id)
        assert released.status is RunStatus.QUEUED and released.attempt == 2, released
        assert released.metadata is not None
        assert ledger.read_text().splitlines() == ["worked:quick"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if slow is not None:
            async with runs_http() as c:
                await c.post(f"/v1/runs/{slow.id}/cancel", json={"reason": "test over"})


@pytest.mark.plan("SURF")
def test_doctor_finds_the_services_and_says_the_keys_tenant_and_role() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "trellis.harness.doctor"],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
        check=False,
    )
    said = done.stdout
    assert "ok    runs:" in said and "ok    memory:" in said, said  # both services reached
    assert "a service key of tenant feat" in said, said  # the key's tenant and role
    assert "FAIL  runs" not in said and "FAIL  memory" not in said, said
    assert (API_KEY or "x") not in said  # the key itself never


#: H-AGUI (agent-harness, P2): the replica that serves an AG-UI run numbers its events itself
#: (from 0, the interrupt folded into the finishing event), while a reconnect on another replica
#: numbers them by agent-runs' log positions (from 1, the interrupt counted): after
#: ``Last-Event-ID: 1`` the other replica sends again an event the client already has, and the
#: ids of one run differ by replica, and an attempt resumed on another replica starts its
#: numbering again — against docs/surfaces.md ("any replica resumes a
#: stream", ids "those of the run's events").
H_AGUI = (
    "H-AGUI: AG-UI event ids differ between the serving replica and a reconnect on another "
    "(own numbering from 0 vs agent-runs log positions)"
)


@pytest.mark.plan("SURF", "F62", "RUN-06")
@pytest.mark.xfail(strict=True, reason=H_AGUI)
async def test_agui_event_ids_are_the_same_on_every_replica(uid: str) -> None:
    agent_id, user, thread, run_id = f"ids-{uid}", f"hal-{uid}", f"t-{uid}", f"run_{uid}ids1"
    async with harness() as h1, harness() as h2:
        apps = []
        for h in (h1, h2):
            app = FastAPI()
            h.wrap(concierge, id=agent_id).serve_chat(app)
            apps.append(app)
        async with (
            Served(apps[0]) as one,
            Served(apps[1]) as two,
            httpx.AsyncClient(timeout=60) as c,
        ):
            message = [{"id": "m1", "role": "user", "content": "book a table"}]
            first = await c.post(
                f"{one.url}/agui/run",
                headers=identity(user),
                json={"threadId": thread, "runId": run_id, "messages": message},
            )
            events = decode(first.text)
            interrupt_id = events[-1][1]["outcome"]["interrupts"][0]["id"]
            try:
                replayed = await read_until_paused(
                    c,
                    f"{two.url}/agui/runs/{run_id}/events",
                    {**identity(user), "Last-Event-ID": str(events[1][0])},
                )
                served = [(n, e["type"]) for n, e in events[2:]]
                assert [(n, e["type"]) for n, e in replayed] == served
            finally:
                await c.post(
                    f"{one.url}/agui/run",
                    headers=identity(user),
                    json={
                        "threadId": thread,
                        "resume": [{"interruptId": interrupt_id, "payload": "friday"}],
                    },
                )
