"""RUN-09, RUN-11, RUN-13 against the real agent-runs: the operator's side.

* rate limits: a replica whose tenant budget is 60 a minute with a burst of 5 answers the
  sixth request at once ``429 RATE_LIMIT`` with ``Retry-After``; every counted answer says the
  budget; the SDK waits ``Retry-After`` and its call succeeds (the harness never sees it);
* retention: a run ended longer ago than ``RUNS__LIMITS__RETENTION_DAYS`` (30 here) is
  deleted by the ticker with its events and resolutions; a recent one is kept;
* migrations: on a database holding a run, the schema goes one revision down and up again and
  the run is read back whole; down to base leaves no table, and up to head serves again.

The first starts a replica of its own (``FEAT_RUNS_LAUNCH``); the third a scratch database."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

from tests.features.conftest import (
    OTHER_KEY,
    RUNS_DSN,
    eventually,
    move_clock,
    runs_http,
    runs_sql,
)

pytestmark = pytest.mark.timeout(300)
LAUNCH = os.environ.get("FEAT_RUNS_LAUNCH")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launched(*args: str, **env: str) -> subprocess.Popen[bytes]:
    assert LAUNCH, "FEAT_RUNS_LAUNCH names the agent-runs launcher"
    return subprocess.Popen(
        [LAUNCH, *args],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stopped(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.fixture
def throttled() -> Iterator[str]:
    port = free_port()
    process = launched("api", str(port), FT_RATE_PER_MINUTE="60", FT_RATE_BURST="5")
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + "/health/live", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    yield url
    stopped(process)


@pytest.mark.plan("RUN-09")
async def test_a_tenants_spent_budget_is_429_with_retry_after_and_the_sdk_waits_it(
    throttled: str, uid: str
) -> None:
    # the second tenant: its budget is this test's alone
    async with runs_http(OTHER_KEY, url=throttled) as c:
        answers = [await c.get("/v1/runs", params={"agent_id": f"rl-{uid}"}) for _ in range(8)]
    statuses = [a.status_code for a in answers]
    assert 429 in statuses, statuses
    first = statuses.index(429)
    assert first <= 6, statuses  # the burst, then refused
    refused = answers[first]
    assert refused.json()["code"] == "RATE_LIMIT" and refused.json()["retryable"] is True
    assert 0 < float(refused.headers["retry-after"]) <= 60
    ok = answers[0]
    assert ok.headers["x-ratelimit-limit"] == "60" and "x-ratelimit-remaining" in ok.headers
    from trellis.runs import RunsClient

    async with RunsClient(base_url=throttled, api_key=OTHER_KEY) as runs:
        started = time.monotonic()
        page = await runs.list(agent_id=f"rl-{uid}")  # waits Retry-After, then answers
        assert page is not None and time.monotonic() - started < 40


@pytest.mark.plan("RUN-11")
@pytest.mark.timeout(200)
async def test_a_run_ended_past_the_retention_is_deleted_with_its_log(uid: str) -> None:
    agent = f"ret-{uid}"
    async with runs_http() as c:
        old = (await c.post("/v1/runs", json={"tenant_id": "feat", "agent_id": agent})).json()
        new = (await c.post("/v1/runs", json={"tenant_id": "feat", "agent_id": agent})).json()
        for run in (old, new):
            await c.post(
                f"/v1/runs/{run['run_id']}/finish",
                json={"status": "SUCCESS", "output": "ok", "error": None},
            )
        move_clock(
            RUNS_DSN,
            "update agent_runs set updated_at = now() - interval '45 days',"
            " created_at = now() - interval '46 days' where run_id = :1",
            old["run_id"],
        )

        async def gone() -> bool:
            return (await c.get(f"/v1/runs/{old['run_id']}")).status_code == 404

        assert await eventually(gone, within=150, every=5), runs_sql(
            "select status, updated_at from agent_runs where run_id = :1", old["run_id"]
        )
        assert (await c.get(f"/v1/runs/{new['run_id']}")).status_code == 200  # kept
    assert not runs_sql("select 1 from run_events where run_id = :1", old["run_id"])
    assert not runs_sql("select 1 from run_resolutions where run_id = :1", old["run_id"])


@pytest.mark.plan("RUN-13")
@pytest.mark.timeout(400)
async def test_the_schema_goes_down_and_up_again_with_a_run_in_it(uid: str) -> None:
    db = f"feat_runs_mig_{uid}"
    admin = (RUNS_DSN or "").rsplit("/", 1)[0] + "/postgres"
    await asyncio.to_thread(
        subprocess.run, ["psql", admin, "-qAt", "-c", f"create database {db}"], check=True
    )
    try:

        def step(*args: str) -> None:
            done = launched(*args, FT_DB_RUNS_OVERRIDE=db)
            assert done.wait(timeout=180) == 0, args

        step("migrate")
        port = free_port()
        api = launched("api", str(port), FT_DB_RUNS_OVERRIDE=db)
        url = f"http://127.0.0.1:{port}"
        try:
            await _up(url)
            async with runs_http(url=url) as c:
                run = (
                    await c.post(
                        "/v1/runs",
                        json={"tenant_id": "feat", "agent_id": "mig", "input": {"q": "kept"}},
                    )
                ).json()
        finally:
            stopped(api)
        step("downgrade", "", "-1")  # one revision down and up again: the run is kept
        step("migrate")
        api = launched("api", str(port), FT_DB_RUNS_OVERRIDE=db)
        try:
            await _up(url)
            async with runs_http(url=url) as c:
                kept = await c.get(f"/v1/runs/{run['run_id']}")
                assert kept.status_code == 200 and kept.json()["input"] == {"q": "kept"}
        finally:
            stopped(api)
        step("downgrade", "", "base")
        dsn = admin.rsplit("/", 1)[0] + f"/{db}"
        tables = (
            await asyncio.to_thread(
                subprocess.run,
                [
                    "psql",
                    dsn,
                    "-qAt",
                    "-c",
                    "select count(*) from pg_tables"
                    " where schemaname = 'public' and tablename = 'agent_runs'",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        ).stdout.strip()
        assert tables == "0"  # down to base: no table left
        step("migrate")
        api = launched("api", str(port), FT_DB_RUNS_OVERRIDE=db)
        try:
            await _up(url)
            async with runs_http(url=url) as c:
                again = await c.post("/v1/runs", json={"tenant_id": "feat", "agent_id": "mig"})
                assert again.status_code == 201  # the service works on the new head
                assert (await c.get(f"/v1/runs/{run['run_id']}")).status_code == 404
        finally:
            stopped(api)
    finally:
        await asyncio.to_thread(
            subprocess.run,
            ["psql", admin, "-qAt", "-c", f"drop database if exists {db} with (force)"],
            check=True,
        )


async def _up(url: str) -> None:
    deadline = time.monotonic() + 60
    async with httpx.AsyncClient(timeout=2) as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(url + "/health/ready")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise AssertionError(f"{url} did not come up")
