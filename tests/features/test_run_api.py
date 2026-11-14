"""RUN-01..RUN-10 and RUN-12 against the real agent-runs — two API replicas on one database
and the ticker — over its own HTTP API, with keys the memory service issued (agent-runs
introspects every key there). Each test proves what a caller relies on: a run moves only
along the documented transitions, the queue hands runs out in order and one per key at a
time, a lease or a hold that lapses is noticed by the ticker, webhooks arrive signed, events
appended on one replica stream from the other, the inbox pages without loss, a start is
idempotent, a cancel works at every status, and a child run names its parent. The run
database is read too.

Every run is of an agent id of the test's own, so claims never see another test's runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from tests.features.conftest import (
    ALICE_KEY,
    RUNS_URL_2,
    eventually,
    harness,
    runs_http,
    runs_sql,
)
from trellis.contracts import (
    Interrupt,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunOutcome,
)
from trellis.runs import parse_delivery, verify_signature
from trellis.runs.webhooks import SIGNATURE_HEADER

pytestmark = pytest.mark.timeout(300)
TENANT = "feat"


def start_body(agent: str, **fields: Any) -> dict[str, Any]:
    return {"tenant_id": TENANT, "agent_id": agent, "input": {"q": "x"}, **fields}


async def start(c: httpx.AsyncClient, agent: str, **fields: Any) -> dict[str, Any]:
    got = await c.post("/v1/runs", json=start_body(agent, **fields))
    assert got.status_code in (200, 201), got.text
    return got.json()


async def claim(c: httpx.AsyncClient, agent: str, worker: str, lease: int = 60) -> dict | None:
    got = await c.post(
        "/v1/runs/claim", json={"worker_id": worker, "agent_ids": [agent], "lease_seconds": lease}
    )
    assert got.status_code in (200, 204), got.text
    return got.json() if got.status_code == 200 else None


def interrupt(run: dict[str, Any], **fields: Any) -> dict[str, Any]:
    made = Interrupt(tenant_id=TENANT, run_id=run["run_id"], question="Go on?", **fields)
    return made.model_dump(mode="json")


def answer(run_id: str, interrupt_id: str, decision: str = "APPROVE", **fields: Any) -> dict:
    made = InterruptResolution(
        interrupt_id=interrupt_id, run_id=run_id, decision=decision, reviewer="user:lee", **fields
    )
    return made.model_dump(mode="json")


async def status(c: httpx.AsyncClient, run_id: str) -> str:
    got = await c.get(f"/v1/runs/{run_id}")
    assert got.status_code == 200, got.text
    return got.json()["status"]


# --------------------------------------------------------------------------- RUN-01
@pytest.mark.plan("RUN-01")
async def test_every_status_is_reached_along_its_transitions_only(uid: str) -> None:
    agent = f"life-{uid}"
    async with runs_http() as c:
        # queued -> running (claimed) -> paused -> queued (answered) -> running -> partial
        run = await start(c, agent, queue=True)
        assert run["status"] == "QUEUED" and run["attempt"] == 1
        claimed = await claim(c, agent, "w-1")
        assert claimed and claimed["run"]["run_id"] == run["run_id"]
        assert claimed["run"]["status"] == "RUNNING"
        asked = interrupt(run)
        paused = await c.post(
            f"/v1/runs/{run['run_id']}/pause",
            params={"worker_id": "w-1"},
            json={"interrupt": asked, "checkpoint": {"asks": {"a": 1}}},
        )
        assert paused.status_code == 200 and paused.json()["status"] == "PAUSED", paused.text
        # invalid while paused: a heartbeat (no lease), a finish SUCCESS, a second pause
        hb = await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                          json={"worker_id": "w-1", "lease_seconds": 60})
        assert hb.status_code == 409 and hb.json()["code"] == "LEASE_LOST"
        bad = await c.post(f"/v1/runs/{run['run_id']}/finish",
                           json={"status": "SUCCESS", "output": None, "error": None})
        assert bad.status_code == 409, bad.text
        resumed = await c.post(f"/v1/runs/{run['run_id']}/resume",
                               json=answer(run["run_id"], asked["interrupt_id"]))
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "QUEUED" and resumed.json()["attempt"] == 2
        again = await claim(c, agent, "w-2")
        assert again and again["run"]["last_resolution"]["decision"] == "APPROVE"
        assert again["run"]["checkpoint"] == {"asks": {"a": 1}}  # the journal travels
        done = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w-2"},
                            json={"status": "PARTIAL", "output": "half", "error": None})
        assert done.status_code == 200 and done.json()["status"] == "PARTIAL"
        assert done.json()["checkpoint"] is None  # an ending clears it
        # a repeat is not an error; another ending is
        same = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w-2"},
                            json={"status": "PARTIAL", "output": "half", "error": None})
        assert same.status_code == 200 and same.json()["output"] == "half"
        other = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w-2"},
                             json={"status": "SUCCESS", "output": "all", "error": None})
        assert other.status_code == 409
        # the other endings, from RUNNING in process
        error = {"code": "X", "category": "TOOL", "message": "m", "retryable": False}
        for ending, err in (("SUCCESS", None), ("ERROR", error), ("TIMEOUT", error),
                            ("REJECTED", error), ("CANCELLED", None)):
            r = await start(c, agent)
            assert r["status"] == "RUNNING"
            f = await c.post(f"/v1/runs/{r['run_id']}/finish",
                             json={"status": ending, "output": None, "error": err})
            assert f.status_code == 200 and f.json()["status"] == ending, f.text
        # an error on a success is refused; QUEUED may only be cancelled or timed out
        r = await start(c, agent)
        bad = await c.post(f"/v1/runs/{r['run_id']}/finish",
                           json={"status": "SUCCESS", "output": None, "error": error})
        assert bad.status_code == 422
        await c.post(f"/v1/runs/{r['run_id']}/finish",
                     json={"status": "SUCCESS", "output": None, "error": None})
        q = await start(c, agent, queue=True)
        bad = await c.post(f"/v1/runs/{q['run_id']}/finish",
                           json={"status": "SUCCESS", "output": None, "error": None})
        assert bad.status_code == 409
        resume_running = await c.post(f"/v1/runs/{q['run_id']}/resume",
                                      json=answer(q["run_id"], "int_none"))
        assert resume_running.status_code == 409
        await c.post(f"/v1/runs/{q['run_id']}/cancel", json={"reason": "done testing"})
        history = await c.get(f"/v1/runs/{run['run_id']}/resolutions")
        assert history.status_code == 200 and len(history.json()) == 1


# --------------------------------------------------------------------------- RUN-02
@pytest.mark.plan("RUN-02")
async def test_the_queue_hands_out_by_priority_and_one_run_per_concurrency_key(
    uid: str, invariants: Any
) -> None:
    invariants.allow("INV-7", "a priority outside -1000..1000 is sent on purpose: refused")
    agent = f"order-{uid}"
    async with runs_http() as c:
        low = await start(c, agent, queue=True, priority=-5)
        mid = await start(c, agent, queue=True)
        high = await start(c, agent, queue=True, priority=10)
        got = [await claim(c, agent, f"w{n}") for n in range(3)]
        assert [g["run"]["run_id"] for g in got if g] == [high["run_id"], mid["run_id"],
                                                         low["run_id"]]
        for g in got:
            await c.post(f"/v1/runs/{g['run']['run_id']}/finish", params={"worker_id": g["lease"]["worker_id"]},
                         json={"status": "SUCCESS", "output": None, "error": None})
        key = f"acct-{uid}"
        a = await start(c, agent, queue=True, concurrency_key=key)
        b = await start(c, agent, queue=True, concurrency_key=key, priority=100)
        first = await claim(c, agent, "wa")
        assert first and first["run"]["run_id"] in (a["run_id"], b["run_id"])
        assert await claim(c, agent, "wb") is None  # its key is busy: the other waits QUEUED
        left = b if first["run"]["run_id"] == a["run_id"] else a
        assert await status(c, left["run_id"]) == "QUEUED"
        await c.post(f"/v1/runs/{first['run']['run_id']}/finish", params={"worker_id": "wa"},
                     json={"status": "SUCCESS", "output": None, "error": None})
        second = await claim(c, agent, "wb")
        assert second and second["run"]["run_id"] == left["run_id"]
        await c.post(f"/v1/runs/{left['run_id']}/finish", params={"worker_id": "wb"},
                     json={"status": "SUCCESS", "output": None, "error": None})
        refused = await c.post("/v1/runs", json=start_body(agent, priority=1001))
        assert refused.status_code == 422


# --------------------------------------------------------------------------- RUN-03
@pytest.mark.plan("RUN-03")
async def test_a_lapsed_lease_is_requeued_by_the_ticker_and_fences_the_old_worker(
    uid: str,
) -> None:
    agent = f"lease-{uid}"
    async with runs_http() as c:
        run = await start(c, agent, queue=True, timeout_seconds=600)
        first = await claim(c, agent, "old", lease=5)
        assert first and first["lease"]["remaining_seconds"] <= 600
        ok = await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                          json={"worker_id": "old", "lease_seconds": 5,
                                "checkpoint": {"tools": {"c1": {"output": "done"}}}})
        assert ok.status_code == 200 and ok.json()["cancel_requested"] is False

        async def requeued() -> bool:  # the old worker stops beating: within a tick or two
            got = (await c.get(f"/v1/runs/{run['run_id']}")).json()
            return got["status"] == "QUEUED" and got["attempt"] == 2

        assert await eventually(requeued, within=40)
        record = (await c.get(f"/v1/runs/{run['run_id']}")).json()
        assert record["checkpoint"] == {"tools": {"c1": {"output": "done"}}}  # progress kept
        nxt = await eventually(lambda: claim(c, agent, "new"), within=60, every=1)
        assert nxt and nxt["run"]["run_id"] == run["run_id"]
        fenced = await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                              json={"worker_id": "old", "lease_seconds": 5})
        assert fenced.status_code == 409 and fenced.json()["code"] == "LEASE_LOST"
        late = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "old"},
                            json={"status": "SUCCESS", "output": "stale", "error": None})
        assert late.status_code == 409 and late.json()["code"] == "LEASE_LOST"
        # release: back on the queue at once, no lapse counted
        rel = await c.post(f"/v1/runs/{run['run_id']}/release", json={"worker_id": "new"})
        assert rel.status_code == 200 and rel.json()["status"] == "QUEUED"
        assert rel.json()["attempt"] == 3
        last = await claim(c, agent, "third")
        assert last and last["run"]["run_id"] == run["run_id"]
        await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "third"},
                     json={"status": "SUCCESS", "output": None, "error": None})
    lapses = runs_sql("select * from agent_runs where run_id = :1", run["run_id"])
    assert lapses and lapses[0]["status"] == "SUCCESS"


@pytest.mark.plan("RUN-03")
async def test_a_hold_that_lapses_ends_the_run_holder_lost(uid: str) -> None:
    agent = f"hold-{uid}"
    async with runs_http() as c:
        run = await start(c, agent)  # RUNNING in its caller's process
        held = await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                            json={"worker_id": "proc-1", "lease_seconds": 5})
        assert held.status_code == 200
        other = await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                             json={"worker_id": "proc-2", "lease_seconds": 5})
        assert other.status_code == 409  # another holder's hold in force

        async def lost() -> dict | None:
            got = (await c.get(f"/v1/runs/{run['run_id']}")).json()
            return got if got["status"] == "ERROR" else None

        ended = await eventually(lost, within=40)
        assert ended, await status(c, run["run_id"])
        assert ended["error"]["code"] == "holder_lost" and ended["error"]["retryable"] is False


@pytest.mark.plan("RUN-03")
@pytest.mark.timeout(240)
async def test_a_held_run_nobody_takes_up_after_its_answer_ends_after_the_resume_grace(
    uid: str,
) -> None:
    agent = f"grace-{uid}"
    async with runs_http() as c:
        run = await start(c, agent)
        await c.post(f"/v1/runs/{run['run_id']}/heartbeat",
                     json={"worker_id": "proc-1", "lease_seconds": 30})
        asked = interrupt(run)
        p = await c.post(f"/v1/runs/{run['run_id']}/pause", json={"interrupt": asked,
                                                                 "checkpoint": None})
        assert p.status_code == 200, p.text
        r = await c.post(f"/v1/runs/{run['run_id']}/resume",
                         json=answer(run["run_id"], asked["interrupt_id"]))
        assert r.status_code == 200 and r.json()["status"] == "RUNNING"
        resumed_at = time.monotonic()
        await asyncio.sleep(40)
        assert await status(c, run["run_id"]) == "RUNNING"  # within the grace: still waits

        async def abandoned() -> dict | None:
            got = (await c.get(f"/v1/runs/{run['run_id']}")).json()
            return got if got["status"] == "ERROR" else None

        ended = await eventually(abandoned, within=60)
        assert ended and ended["error"]["code"] == "holder_lost"
        assert time.monotonic() - resumed_at >= 55  # RESUME_GRACE (60 s), within a tick


# --------------------------------------------------------------------------- RUN-04
@pytest.mark.plan("RUN-04")
@pytest.mark.timeout(240)
async def test_a_retryable_error_is_retried_with_backoff_and_a_later_one_stands(uid: str) -> None:
    agent = f"retry-{uid}"
    flaky = {"code": "Busy", "category": "DEPENDENCY", "message": "try later", "retryable": True}
    async with runs_http() as c:
        run = await start(c, agent, queue=True)
        got = await claim(c, agent, "w1")
        assert got
        f = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w1"},
                         json={"status": "ERROR", "output": None, "error": flaky})
        assert f.status_code == 200 and f.json()["status"] == "QUEUED", f.text
        assert f.json()["attempt"] == 2
        assert await claim(c, agent, "w2") is None  # held back by its backoff (10 s)
        again = await eventually(lambda: claim(c, agent, "w2"), within=40, every=1)
        assert again and again["run"]["run_id"] == run["run_id"]
        hard = {"code": "Bad", "category": "TOOL", "message": "broken", "retryable": False}
        f = await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w2"},
                         json={"status": "ERROR", "output": None, "error": hard})
        assert f.json()["status"] == "ERROR"
        assert "(after 1 of 3 retries)" in f.json()["error"]["message"], f.json()["error"]
        # a run kept in its caller's process is never retried here
        own = await start(c, agent)
        f = await c.post(f"/v1/runs/{own['run_id']}/finish",
                         json={"status": "ERROR", "output": None, "error": flaky})
        assert f.json()["status"] == "ERROR"


# --------------------------------------------------------------------------- RUN-05
class Receiver:
    def __init__(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.url = f"http://127.0.0.1:{self.port}/hook"
        self.got: list[tuple[dict[str, str], bytes]] = []
        #: what ``/gone`` answers: 410 (refused for good: the delivery dies at once) until fixed
        self.gone_status = 410

    def app(self) -> Starlette:
        async def hook(request: Request) -> Response:
            self.got.append((dict(request.headers), await request.body()))
            return Response(status_code=204)

        async def gone(request: Request) -> Response:
            if self.gone_status < 300:
                self.got.append((dict(request.headers), await request.body()))
            return Response(status_code=self.gone_status)

        return Starlette(routes=[Route("/hook", hook, methods=["POST"]),
                                 Route("/gone", gone, methods=["POST"])])


@pytest.fixture
def receiver() -> Iterator[Receiver]:
    r = Receiver()
    server = uvicorn.Server(uvicorn.Config(r.app(), host="127.0.0.1", port=r.port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield r
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.plan("RUN-05", "RUN-12")
async def test_pauses_and_endings_are_delivered_signed_and_a_rotated_secret_signs(
    receiver: Receiver, uid: str
) -> None:
    agent = f"hooks-{uid}"
    async with runs_http() as c:
        made = await c.post("/v1/webhooks", json={"url": receiver.url,
                                                  "events": ["run.paused", "run.finished"]})
        assert made.status_code == 201, made.text
        hook_id, secret = made.json()["webhook_id"], made.json()["secret"]
        try:
            listed = await c.get("/v1/webhooks")
            assert any(w["webhook_id"] == hook_id for w in listed.json())
            assert secret not in listed.text  # the secret is shown once
            one = await c.get(f"/v1/webhooks/{hook_id}")
            assert one.status_code == 200 and secret not in one.text
            run = await start(c, agent)
            asked = interrupt(run, assignee="role:ops")
            await c.post(f"/v1/runs/{run['run_id']}/pause", json={"interrupt": asked,
                                                                 "checkpoint": None})
            await c.post(f"/v1/runs/{run['run_id']}/resume",
                         json=answer(run["run_id"], asked["interrupt_id"]))
            await c.post(f"/v1/runs/{run['run_id']}/finish",
                         json={"status": "SUCCESS", "output": "ok", "error": None})

            def mine() -> list[Any]:
                out = []
                for headers, body in receiver.got:
                    d = parse_delivery(body)
                    if d.data.run.run_id == run["run_id"]:
                        out.append((headers, body, d))
                return out

            got = await eventually(lambda: _async(len(mine()) >= 2), within=60)
            assert got, receiver.got
            kinds = sorted(str(d.type.value) for _, _, d in mine())
            assert kinds == ["run.finished", "run.paused"], kinds
            for headers, body, _ in mine():
                assert verify_signature(secret, headers.get(SIGNATURE_HEADER.lower()), body)
                assert not verify_signature("wrong-secret", headers.get(SIGNATURE_HEADER.lower()),
                                            body)
            rotated = await c.post(f"/v1/webhooks/{hook_id}/rotate-secret")
            assert rotated.status_code == 200
            new_secret = rotated.json()["secret"]
            assert new_secret != secret
            # a receiver that refuses for good: the delivery dies; fixed, it is redelivered
            broken = await c.post("/v1/webhooks", json={
                "url": receiver.url.replace("/hook", "/gone"), "events": ["run.finished"]})
            broken_id, broken_secret = broken.json()["webhook_id"], broken.json()["secret"]
            try:
                r2 = await start(c, agent)
                await c.post(f"/v1/runs/{r2['run_id']}/finish",
                             json={"status": "SUCCESS", "output": "ok", "error": None})

                async def dead() -> list[dict]:
                    got = await c.get("/v1/webhooks/deliveries",
                                      params={"webhook_id": broken_id, "state": "dead"})
                    return got.json()

                died = await eventually(dead, within=60)
                assert died and died[0]["run_id"] == r2["run_id"], died
                receiver.gone_status = 204
                before = len(receiver.got)
                redone = await c.post(
                    f"/v1/webhooks/deliveries/{died[0]['delivery_id']}/redeliver")
                assert redone.status_code == 200, redone.text
                assert await eventually(lambda: _async(len(receiver.got) > before), within=60)
                headers, body = receiver.got[-1]
                assert parse_delivery(body).data.run.run_id == r2["run_id"]
                assert verify_signature(broken_secret, headers.get(SIGNATURE_HEADER.lower()),
                                        body)
                again = await c.post(
                    f"/v1/webhooks/deliveries/{died[0]['delivery_id']}/redeliver")
                assert again.status_code in (404, 409)  # delivered: no longer owed
            finally:
                await c.delete(f"/v1/webhooks/{broken_id}")
            refused = await c.post("/v1/webhooks", json={"url": "ftp://example.com/x",
                                                         "events": ["run.finished"]})
            assert refused.status_code == 422
        finally:
            gone = await c.delete(f"/v1/webhooks/{hook_id}")
            assert gone.status_code == 204
            assert (await c.get(f"/v1/webhooks/{hook_id}")).status_code == 404


async def _async(value: Any) -> Any:
    return value


# --------------------------------------------------------------------------- RUN-06
@pytest.mark.plan("RUN-06", "RUN-12")
async def test_events_appended_on_one_replica_stream_live_from_the_other(uid: str) -> None:
    assert RUNS_URL_2, "FEAT_RUNS_URL_2 names the second replica"
    agent = f"events-{uid}"
    async with runs_http() as one, runs_http(url=RUNS_URL_2) as two:
        run = await start(one, agent)

        def event(n: int, kind: RunEventType, **data: Any) -> dict[str, Any]:
            outcome = RunOutcome.SUCCESS if kind is RunEventType.RUN_FINISHED else None
            return RunEvent(type=kind, tenant_id=TENANT, run_id=run["run_id"], sequence=n,
                            attempt=1, outcome=outcome, data=data).model_dump(mode="json")

        first = await one.post(f"/v1/runs/{run['run_id']}/events",
                               json={"events": [event(1, RunEventType.RUN_STARTED)]})
        assert first.status_code == 200 and first.json()["position"] == 1
        seen: list[dict[str, Any]] = []

        async def follow(after: int, *, last_event_id: str | None = None) -> None:
            headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
            async with two.stream("GET", f"/v1/runs/{run['run_id']}/events/stream",
                                  params={"after": after}, headers=headers) as s:
                assert s.status_code == 200
                data_id = None
                async for line in s.aiter_lines():
                    if line.startswith("id:"):
                        data_id = int(line[3:].strip())
                    if line.startswith("data:"):
                        seen.append({"id": data_id, **json.loads(line[5:])})
                        if seen[-1]["event"]["type"] == "RUN_FINISHED":
                            return

        follower = asyncio.create_task(follow(0))
        await asyncio.sleep(1.0)
        msg = event(2, RunEventType.CUSTOM, name="progress", step=1)
        again = await one.post(f"/v1/runs/{run['run_id']}/events", json={"events": [msg]})
        assert again.json()["position"] == 2
        dup = await one.post(f"/v1/runs/{run['run_id']}/events", json={"events": [msg]})
        assert dup.json()["appended"] == 0  # a retried append is safe
        await one.post(f"/v1/runs/{run['run_id']}/events",
                       json={"events": [event(3, RunEventType.RUN_FINISHED)]})
        await asyncio.wait_for(follower, 30)
        assert [s["id"] for s in seen] == [1, 2, 3]  # live, in order, from the other replica
        await one.post(f"/v1/runs/{run['run_id']}/finish",
                       json={"status": "SUCCESS", "output": None, "error": None})
        seen.clear()
        await asyncio.wait_for(follow(0, last_event_id="2"), 30)  # a reconnect: only what's new
        assert [s["id"] for s in seen] == [3]
        page = await two.get(f"/v1/runs/{run['run_id']}/events", params={"after": 1})
        assert [e["position"] for e in page.json()] == [2, 3]
        late = await one.post(f"/v1/runs/{run['run_id']}/events",
                              json={"events": [event(4, RunEventType.CUSTOM, name="late")]})
        assert late.status_code == 409  # the log takes nothing after the end


# --------------------------------------------------------------------------- RUN-07
@pytest.mark.plan("RUN-07")
async def test_the_inbox_pages_every_waiting_run_once(uid: str) -> None:
    agent, team = f"inbox-{uid}", f"role:team-{uid}"
    async with runs_http() as c:
        made = []
        for _ in range(7):
            run = await start(c, agent)
            await c.post(f"/v1/runs/{run['run_id']}/pause",
                         json={"interrupt": interrupt(run, assignee=team), "checkpoint": None})
            made.append(run["run_id"])
        seen: list[str] = []
        url: str | None = "/v1/runs"
        params: dict[str, Any] | None = {"status": "PAUSED", "assignee": team, "limit": 3}
        while url:
            got = await c.get(url, params=params)
            assert got.status_code == 200
            seen += [r["run_id"] for r in got.json()]
            nxt = got.links.get("next")
            url, params = (nxt["url"], None) if nxt else (None, None)
        assert sorted(seen) == sorted(made) and len(seen) == 7  # each once, none lost
        async with harness() as h:
            inbox = await h.inbox(team)
        assert sorted(s.run_id for s in inbox) == sorted(made)
        assert all(s.awaiting is not None and s.assignee == team for s in inbox)
        for run_id in made:
            await c.post(f"/v1/runs/{run_id}/cancel", json={"reason": "inbox test over"})


# --------------------------------------------------------------------------- RUN-08
@pytest.mark.plan("RUN-08", "CONV-11")
async def test_a_start_repeated_with_its_idempotency_key_is_one_run(uid: str) -> None:
    agent, key = f"idem-{uid}", f"msg-{uid}"
    async with runs_http() as c:
        first = await c.post("/v1/runs", json=start_body(agent, idempotency_key=key, queue=True))
        assert first.status_code == 201 and first.headers["location"].endswith(
            first.json()["run_id"]
        )
        again = await c.post("/v1/runs", json=start_body(agent, idempotency_key=key, queue=True))
        assert again.status_code == 200 and again.json()["run_id"] == first.json()["run_id"]
        assert "location" not in again.headers
        other = await c.post("/v1/runs", json={**start_body(agent, idempotency_key=key,
                                                            queue=True), "input": {"q": "y"}})
        assert other.status_code == 409 and "input" in other.json()["details"]["differing"]
        by_id = await c.post("/v1/runs", json=start_body(agent, run_id=first.json()["run_id"],
                                                         queue=True, idempotency_key=key))
        assert by_id.status_code == 200
        count = runs_sql("select count(*) as n from agent_runs where idempotency_key = :1", key)
        assert count[0]["n"] == 1
        await c.post(f"/v1/runs/{first.json()['run_id']}/cancel", json={"reason": "over"})


# --------------------------------------------------------------------------- RUN-10
@pytest.mark.plan("RUN-10")
async def test_cancel_works_at_every_status_and_a_child_names_its_parent(uid: str) -> None:
    agent = f"cancel-{uid}"
    async with runs_http() as c:
        queued = await start(c, agent, queue=True)
        got = await c.post(f"/v1/runs/{queued['run_id']}/cancel", json={"reason": "no longer"})
        assert got.json()["status"] == "CANCELLED"
        kept = runs_sql("select cancel_reason from agent_runs where run_id = :1", queued["run_id"])
        assert kept[0]["cancel_reason"] == "no longer"
        repeat = await c.post(f"/v1/runs/{queued['run_id']}/cancel", json={"reason": "no longer"})
        assert repeat.status_code == 200  # a retried cancel
        own = await start(c, agent)
        assert (await c.post(f"/v1/runs/{own['run_id']}/cancel", json={})).json()["status"] == \
            "CANCELLED"
        paused = await start(c, agent)
        await c.post(f"/v1/runs/{paused['run_id']}/pause",
                     json={"interrupt": interrupt(paused), "checkpoint": {"x": 1}})
        got = await c.post(f"/v1/runs/{paused['run_id']}/cancel", json={"reason": "stop"})
        assert got.json()["status"] == "CANCELLED" and got.json()["checkpoint"] is None
        working = await start(c, agent, queue=True)
        assert await claim(c, agent, "w")
        got = await c.post(f"/v1/runs/{working['run_id']}/cancel", json={"reason": "halt"})
        assert got.json()["status"] == "RUNNING"  # asked of its worker
        hb = await c.post(f"/v1/runs/{working['run_id']}/heartbeat",
                          json={"worker_id": "w", "lease_seconds": 60})
        assert hb.json()["cancel_requested"] is True
        fin = await c.post(f"/v1/runs/{working['run_id']}/finish", params={"worker_id": "w"},
                           json={"status": "CANCELLED", "output": None, "error": None})
        assert fin.json()["status"] == "CANCELLED"
        ended = await c.post(f"/v1/runs/{working['run_id']}/cancel", json={"reason": "again"})
        assert ended.status_code == 409  # an ended run
        # a child names its parent; top_level hides it from the parent's listing
        parent = await start(c, agent)
        child = await start(c, f"{agent}-child", parent_run_id=parent["run_id"])
        kids = await c.get("/v1/runs", params={"parent_run_id": parent["run_id"]})
        assert [k["run_id"] for k in kids.json()] == [child["run_id"]]
        top = await c.get("/v1/runs", params={"agent_id": f"{agent}-child", "top_level": "true"})
        assert top.json() == []
        for r in (child, parent):
            await c.post(f"/v1/runs/{r['run_id']}/finish",
                         json={"status": "SUCCESS", "output": None, "error": None})


@pytest.mark.plan("RUN-10", "SEC-06")
async def test_a_key_restricted_to_one_person_answers_and_cancels_only_their_runs(
    uid: str,
) -> None:
    agent = f"who-{uid}"
    async with runs_http() as c, runs_http(ALICE_KEY) as alice:
        theirs = await start(c, agent, on_behalf_of="user:alice")
        await c.post(f"/v1/runs/{theirs['run_id']}/pause", json={
            "interrupt": interrupt(theirs, assignee="user:bob"), "checkpoint": None})
        no = await alice.post(f"/v1/runs/{theirs['run_id']}/resume", json={
            **answer(theirs["run_id"], (await c.get(f"/v1/runs/{theirs['run_id']}")).json()[
                "awaiting"]["interrupt_id"]), "reviewer": "user:alice"})
        assert no.status_code == 403  # assigned to bob
        group = await start(c, agent)
        asked = interrupt(group, assignee="role:finance")
        await c.post(f"/v1/runs/{group['run_id']}/pause", json={"interrupt": asked,
                                                               "checkpoint": None})
        no = await alice.post(f"/v1/runs/{group['run_id']}/resume",
                              json={**answer(group["run_id"], asked["interrupt_id"]),
                                    "reviewer": "user:alice"})
        assert no.status_code == 403 and "group" in no.json()["detail"]
        mine = await start(c, agent)
        asked = interrupt(mine, assignee="user:alice")
        await c.post(f"/v1/runs/{mine['run_id']}/pause", json={"interrupt": asked,
                                                              "checkpoint": None})
        yes = await alice.post(f"/v1/runs/{mine['run_id']}/resume",
                               json={**answer(mine["run_id"], asked["interrupt_id"]),
                                     "reviewer": "user:alice"})
        assert yes.status_code == 200, yes.text
        bad = await alice.post("/v1/runs", json=start_body(agent, on_behalf_of="user:bob"))
        assert bad.status_code == 403
        for r in (theirs, group):
            await c.post(f"/v1/runs/{r['run_id']}/cancel", json={"reason": "over"})
        await c.post(f"/v1/runs/{mine['run_id']}/finish",
                     json={"status": "SUCCESS", "output": None, "error": None})


@pytest.mark.plan("RUN-12", "SEC-01")
async def test_another_tenants_runs_are_not_found_and_its_tenant_not_writable(uid: str) -> None:
    from tests.features.conftest import OTHER_KEY

    agent = f"tenancy-{uid}"
    async with runs_http() as c, runs_http(OTHER_KEY) as other:
        run = await start(c, agent)
        for method, path, body in (
            ("GET", f"/v1/runs/{run['run_id']}", None),
            ("GET", f"/v1/runs/{run['run_id']}/events", None),
            ("GET", f"/v1/runs/{run['run_id']}/resolutions", None),
            ("POST", f"/v1/runs/{run['run_id']}/cancel", {"reason": "x"}),
        ):
            got = await other.request(method, path, json=body)
            assert got.status_code == 404, (path, got.status_code)
        foreign = await other.post("/v1/runs", json=start_body(agent))  # names tenant feat
        assert foreign.status_code == 403
        listed = await other.get("/v1/runs", params={"agent_id": agent})
        assert listed.json() == []
        await c.post(f"/v1/runs/{run['run_id']}/finish",
                     json={"status": "SUCCESS", "output": None, "error": None})


@pytest.mark.plan("RUN-12")
async def test_artifacts_and_payloads_are_stored_once_and_served_back(uid: str) -> None:
    agent = f"files-{uid}"
    data = b"sku,qty\nA-1,3\n" * 1000
    checksum = "sha256:" + hashlib.sha256(data).hexdigest()
    async with runs_http() as c:
        run = await start(c, agent, queue=True)
        assert await claim(c, agent, "w")
        up = await c.post(f"/v1/runs/{run['run_id']}/artifacts",
                          params={"worker_id": "w", "checksum": checksum}, content=data,
                          headers={"Content-Type": "text/csv"})
        assert up.status_code == 201, up.text
        again = await c.post(f"/v1/runs/{run['run_id']}/artifacts",
                             params={"worker_id": "w", "checksum": checksum}, content=data,
                             headers={"Content-Type": "text/csv"})
        assert again.status_code == 200 and again.json()["artifact_id"] == up.json()["artifact_id"]
        bad = await c.post(f"/v1/runs/{run['run_id']}/artifacts",
                           params={"worker_id": "w", "checksum": "sha256:" + "0" * 64}, content=data,
                           headers={"Content-Type": "text/csv"})
        assert bad.status_code == 422
        back = await c.get(f"/v1/artifacts/{up.json()['artifact_id']}")
        assert back.status_code == 200 and back.content == data
        # a journal's segments: stored once per content, read back one by one and together
        from trellis.runs import RunsClient
        from trellis.runs.payloads import encode

        async with RunsClient(base_url=c.base_url.__str__(), api_key=c.headers["X-API-Key"]) as rc:
            first = encode({"calls": {"c1": [{"rows": 3}]}})
            second = encode({"calls": {"c2": "done"}})
            refs = [await rc.artifacts.put_payload(run["run_id"], b, worker_id="w")
                    for b in (first, second)]
            again_ref = await rc.artifacts.put_payload(run["run_id"], first, worker_id="w")
            assert again_ref.sha256 == refs[0].sha256
            assert await rc.artifacts.get_payload(refs[1]) == second
            assert await rc.artifacts.get_payloads(refs) == [first, second]
        await c.post(f"/v1/runs/{run['run_id']}/finish", params={"worker_id": "w"},
                     json={"status": "SUCCESS", "output": None, "error": None})


@pytest.mark.plan("RUN-12")
async def test_the_probes_and_metrics_answer_without_a_key() -> None:
    from tests.features.conftest import RUNS_URL

    async with httpx.AsyncClient(base_url=RUNS_URL or "", timeout=30) as c:
        assert (await c.get("/health/live")).json() == {"status": "ok"}
        assert (await c.get("/health/ready")).status_code == 200
        metrics = await c.get("/metrics")
        assert metrics.status_code == 200 and "# TYPE" in metrics.text
        assert (await c.get("/v1/runs")).status_code == 401
