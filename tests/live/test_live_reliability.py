"""Reliability against agent-runs (and its ticker) and the memory service: a write that timed out
is recorded of unknown effect and is not run again after its worker dies, nor is a write the
worker died in; a running queued run is cancelled from another process; a run's working time
counts across a crash and its next attempt stops on time; a read that meets a busy service is
retried and recorded once."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import pytest
from fastapi import FastAPI, Request, Response

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import eventually, free_port, memory_scope, serving
from trellis import Harness, Runtime, openapi, tool
from trellis.contracts import RunStatus
from trellis.runs import Job

pytestmark = [pytest.mark.live, needs_runs, needs_memory]

#: The ticker sweeps every 5 s and a lapsed lease is queued again after 5 s: a run a dead
#: worker held is claimable again within ~20 s.
SWEEP_SECONDS = 40.0


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def died(h: Harness, agent: Any, handle: Any) -> None:
    """One attempt on a worker that dies (``Crash``) under a short lease, then the wait until
    agent-runs has queued the run again."""
    claimed = await h.runs.claim("w-dies", [agent.id], lease_seconds=5)
    assert claimed is not None
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id="w-dies", lease_seconds=5, store=h.runs)
        )

    async def requeued() -> bool:
        return (await handle.status()).status is RunStatus.QUEUED

    assert await eventually(requeued, within=SWEEP_SECONDS)


async def claimed_again(h: Harness, agent: Any) -> bool:
    """The next attempt, once agent-runs' backoff after the lapse lets a worker claim it."""
    worker = h.worker([agent], concurrency=1)
    return await eventually(worker.run_once, within=SWEEP_SECONDS)


async def test_writes_of_unknown_effect_are_not_run_again_after_a_worker_dies(
    harness: Harness,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    transfers: list[int] = []
    wires: list[int] = []

    @tool(name=f"transfer_{suffix}", side_effects="write", timeout=0.3)
    async def transfer(amount: int) -> str:
        """Transfer money (the bank never answers in time)."""
        transfers.append(amount)
        await asyncio.sleep(30)
        return "sent"

    @tool(name=f"wire_{suffix}", side_effects="write")
    async def wire(amount: int) -> str:
        """Wire money (the worker dies while the bank works on it)."""
        wires.append(amount)
        if len(wires) == 1:
            raise Crash()
        return "wired"

    async def paying(input: str, agent: Runtime) -> list[str]:
        sent = await agent.tools.call(transfer.spec.name, amount=5)
        wired = await agent.tools.call(wire.spec.name, amount=7)
        return [sent, wired]

    agent = harness.wrap(paying, id=f"live-pay-{suffix}", tools=[transfer, wire])
    handle = await agent.start("pay", user="live-user")
    await died(harness, agent, handle)
    assert await claimed_again(harness, agent)
    done = await handle.result(timeout=30)
    assert done.status is RunStatus.SUCCESS, done.error
    timed_out, interrupted = done.answer
    assert timed_out == (
        f"{transfer.spec.name} timed out after 0.3s; it may or may not have taken effect: "
        "check before calling it again"
    )
    assert interrupted.startswith(f"{wire.spec.name} was interrupted by a crash")
    assert transfers == [5] and wires == [7]  # neither ran again
    # each was recorded once, as what the model read
    scope = await memory_scope(harness, user="live-user", agent_id=agent.id)

    async def recorded_once() -> bool:
        names = [transfer.spec.name, wire.spec.name]
        entries = await scope.advanced.tools.catalog(names=names)
        return len(entries) == 2 and all(e.stats.calls == 1 for e in entries)

    await harness.writes.drain()
    assert await eventually(recorded_once)


async def test_a_running_queued_run_is_cancelled_from_another_process() -> None:
    suffix = uuid.uuid4().hex[:8]
    started = asyncio.Event()

    async def ponders(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(120)
        return "never"

    async with live_harness() as worker_side, live_harness() as caller_side:
        working_agent = worker_side.wrap(ponders, id=f"live-ponder-{suffix}")
        worker = worker_side.worker([working_agent], concurrency=1)
        worker.loop.lease_seconds = 6  # a heartbeat every 2 s
        # the caller's side: the same agent id; the run is not in its process
        agent = caller_side.wrap(ponders, id=f"live-ponder-{suffix}")
        handle = await agent.start("x", user="live-user")
        working = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(started.wait(), 30)
        asked = await handle.cancel(reason="nobody is waiting any more")
        assert asked.status is RunStatus.RUNNING  # its worker is asked to stop
        assert await asyncio.wait_for(working, 30)  # and does, at its next heartbeat
        record = await handle.status()
        assert record.status is RunStatus.CANCELLED


async def test_working_time_counts_across_a_crash_and_the_next_attempt_stops_on_time(
    harness: Harness,
) -> None:
    attempts: list[int] = []

    async def slow(input: str, agent: Runtime) -> str:
        attempts.append(agent.attempt)
        if agent.attempt == 1:
            await asyncio.sleep(1)
            raise Crash()
        await asyncio.sleep(120)
        return "never"

    agent = harness.wrap(slow, id=f"live-slow-{uuid.uuid4().hex[:8]}")
    handle = await agent.start("x", user="live-user", timeout=30)
    await died(harness, agent, handle)
    worked = (await handle.status()).worked_seconds
    assert worked > 1  # running until agent-runs took the run back
    began = time.monotonic()
    assert await claimed_again(harness, agent)
    took = time.monotonic() - began
    done = await handle.result(timeout=30)
    assert done.status is RunStatus.TIMEOUT and done.error is not None
    assert done.error.code == "run_timeout" and not done.error.retryable
    assert took < 30 - worked + 15  # what was left (and the backoff), not the limit again
    record = await handle.status()
    assert record.attempt == 2 and record.worked_seconds >= 29


async def test_a_read_that_meets_a_busy_service_is_retried_and_a_write_sends_its_key(
    harness: Harness,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    asked: list[str] = []
    keys: list[str | None] = []
    app = FastAPI()

    @app.get("/prices/{sku}")
    async def price(sku: str) -> Response:
        asked.append(sku)
        if len(asked) < 3:
            return Response(status_code=503)
        return Response(content=f'{{"sku": "{sku}", "eur": 7}}', media_type="application/json")

    @app.post("/orders")
    async def order(request: Request) -> dict[str, Any]:
        keys.append(request.headers.get("idempotency-key"))
        return {"id": "o-1"}

    port = free_port()
    document = {
        "openapi": "3.1.0",
        "servers": [{"url": f"http://127.0.0.1:{port}"}],
        "paths": {
            "/prices/{sku}": {
                "get": {
                    "operationId": f"price_{suffix}",
                    "parameters": [{"name": "sku", "in": "path", "schema": {"type": "string"}}],
                }
            },
            "/orders": {"post": {"operationId": f"order_{suffix}"}},
        },
    }

    async def buying(input: str, agent: Runtime) -> list[Any]:
        quoted = await agent.tools.call(f"price_{suffix}", sku="A-1")
        return [quoted, await agent.tools.call(f"order_{suffix}")]

    with serving(app, port):
        agent = harness.wrap(buying, id=f"live-buy-{suffix}", tools=[openapi(document)])
        result = await agent.run("buy A-1", user="live-user")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == [{"sku": "A-1", "eur": 7}, {"id": "o-1"}]
    assert asked == ["A-1"] * 3  # two 503s, then the price
    [key] = keys
    assert key is not None and key.startswith(f"{result.run_id}:")
    scope = await memory_scope(harness, user="live-user", agent_id=agent.id)

    async def recorded_once() -> bool:
        entries = await scope.advanced.tools.catalog(names=[f"price_{suffix}"])
        return bool(entries) and entries[0].stats.calls == 1

    await harness.writes.drain()
    assert await eventually(recorded_once)
