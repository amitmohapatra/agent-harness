"""Durable runs against agent-runs and its ticker: the run store's whole wire (queue, claim,
heartbeat, lease loss, pause with a checkpoint, resume, finish, inbox, escalation), a run that
asks twice and is continued by a different worker process each time, and a schedule the ticker
fires for a worker."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import eventually, memory_scope
from trellis import Harness
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunStart,
    RunStatus,
    new_id,
)
from trellis.runs import LeaseLostError, RunsClient, RunSummary

pytestmark = [pytest.mark.live, needs_runs, needs_memory]  # the key is the memory service's

ROOT = Path(__file__).resolve().parents[2]
#: The ticker sweeps every 5 s; a lapsed 5 s lease is re-queued within ~10 s.
SWEEP_SECONDS = 30.0


async def in_a_worker(input: object, agent: object) -> None:
    """The caller's handle on an agent a worker process executes (tests.live.worker_app)."""
    raise AssertionError("this agent runs in a worker process")


def start(agent_id: str, tenant: str, **fields: object) -> RunStart:
    return RunStart(
        run_id=new_id("run_"),
        tenant_id=tenant,
        agent_id=agent_id,
        user_id="live-user",
        input="x",
        **fields,  # type: ignore[arg-type]
    )


async def test_the_runs_client_speaks_agent_runs(harness: Harness) -> None:
    runs = harness.runs
    agent_id = f"live-wire-{uuid.uuid4().hex[:8]}"
    tenant = await harness.tenant()
    queued = await runs.start(start(agent_id, tenant), queue=True)
    assert queued.status is RunStatus.QUEUED
    took = await runs.claim("w1", [agent_id], lease_seconds=30)
    assert took is not None and took.run.run_id == queued.run_id
    claimed = took.run
    assert claimed.status is RunStatus.RUNNING and took.lease.worker_id == "w1"
    await runs.heartbeat(claimed.run_id, "w1", lease_seconds=30, tenant=tenant)

    asked = Interrupt(
        interrupt_id=f"{claimed.run_id}.1.1",
        tenant_id=claimed.tenant_id,
        run_id=claimed.run_id,
        question="Which size?",
        assignee="role:live-ops",
    )
    checkpoint = {"answers": {}, "calls": {"k": ["charged"]}}
    paused = await runs.pause(asked, checkpoint=checkpoint, worker_id="w1")
    assert paused.status is RunStatus.PAUSED and paused.checkpoint == checkpoint
    [waiting] = [r for r in await harness.inbox("role:live-ops") if r.run_id == claimed.run_id]
    assert waiting.awaiting is not None and waiting.awaiting.question == "Which size?"

    answer = InterruptResolution(
        interrupt_id=asked.interrupt_id,
        run_id=asked.run_id,
        decision=InterruptDecision.ANSWER,
        answer="L",
        reviewer="live-reviewer",
    )
    resumed = await runs.resume(answer, tenant=tenant)
    assert resumed.status is RunStatus.QUEUED and resumed.attempt == 2  # it came from the queue
    retook = await runs.claim("w2", [agent_id], lease_seconds=30)
    assert retook is not None and retook.run.checkpoint == checkpoint
    again = retook.run
    assert again.last_resolution is not None and again.last_resolution.answer == "L"
    with pytest.raises(LeaseLostError):  # w1 let go of it when it paused
        await runs.heartbeat(again.run_id, "w1", lease_seconds=30, tenant=tenant)
    done = await runs.finish(
        again.run_id, RunStatus.SUCCESS, output="ok", worker_id="w2", tenant=tenant
    )
    assert done.status is RunStatus.SUCCESS and done.checkpoint is None


async def test_a_lapsed_lease_puts_the_run_back_and_fences_the_old_worker(
    harness: Harness,
) -> None:
    runs = harness.runs
    agent_id = f"live-lease-{uuid.uuid4().hex[:8]}"
    tenant = await harness.tenant()
    queued = await runs.start(start(agent_id, tenant), queue=True)
    assert await runs.claim("w1", [agent_id], lease_seconds=5) is not None

    async def requeued() -> bool:
        record = await runs.get(queued.run_id, tenant=tenant)
        return record is not None and record.status is RunStatus.QUEUED

    assert await eventually(requeued, within=SWEEP_SECONDS)
    with pytest.raises(LeaseLostError):
        await runs.heartbeat(queued.run_id, "w1", lease_seconds=5, tenant=tenant)
    taken = await runs.claim("w2", [agent_id], lease_seconds=30)
    assert taken is not None and taken.run.attempt == 2
    with pytest.raises(LeaseLostError):
        await runs.finish(queued.run_id, RunStatus.SUCCESS, worker_id="w1", tenant=tenant)
    await runs.finish(queued.run_id, RunStatus.SUCCESS, worker_id="w2", tenant=tenant)


async def test_a_question_past_its_deadline_escalates_or_times_out(harness: Harness) -> None:
    runs = harness.runs
    agent_id = f"live-escalate-{uuid.uuid4().hex[:8]}"
    soon = datetime.now(UTC) + timedelta(seconds=1)
    tenant = await harness.tenant()
    escalated = await runs.start(start(agent_id, tenant))
    await runs.pause(
        Interrupt(
            interrupt_id=f"{escalated.run_id}.1.1",
            tenant_id=escalated.tenant_id,
            run_id=escalated.run_id,
            question="Approve?",
            assignee="role:live-clerk",
            deadline=soon,
            escalate_to="role:live-lead",
        )
    )
    timed_out = await runs.start(start(agent_id, tenant))
    await runs.pause(
        Interrupt(
            interrupt_id=f"{timed_out.run_id}.1.1",
            tenant_id=timed_out.tenant_id,
            run_id=timed_out.run_id,
            question="Approve?",
            assignee="role:live-clerk",
            deadline=soon,
        )
    )

    async def swept() -> bool:
        lead = await harness.inbox("role:live-lead")
        other = await runs.get(timed_out.run_id, tenant=tenant)
        return escalated.run_id in [r.run_id for r in lead] and (
            other is not None and other.status is RunStatus.TIMEOUT
        )

    assert await eventually(swept, within=SWEEP_SECONDS)


# --------------------------------------------------------------------------- worker processes
@contextmanager
def worker_process(suffix: str, ledger: Path) -> Iterator[subprocess.Popen[bytes]]:
    """``python -m trellis.worker tests.live.worker_app:h`` with this session's agents."""
    env = {**os.environ, "TRELLIS_LIVE_SUFFIX": suffix, "TRELLIS_LIVE_LEDGER": str(ledger)}
    process = subprocess.Popen(
        [sys.executable, "-m", "trellis.worker", "tests.live.worker_app:h"], cwd=ROOT, env=env
    )
    try:
        yield process
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def test_ask_and_resume_continue_in_other_worker_processes(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:8]
    ledger = tmp_path / "ledger.txt"
    ledger.touch()
    async with live_harness() as h:
        # the caller's side: the same agent id the workers serve; it never executes the run
        agent = h.wrap(in_a_worker, id=f"live-billing-{suffix}")
        handle = await agent.start({"order": "o-7", "amount": 40}, user="live-ada")

        with worker_process(suffix, ledger):
            first = await handle.result(timeout=60)
        assert first.status is RunStatus.PAUSED and first.interrupt is not None
        assert first.interrupt.question == "Which size?"
        inbox = await h.inbox("role:ops")
        assert handle.run_id in [r.run_id for r in inbox]
        queued = await agent.resume(
            first.interrupt.interrupt_id, "answer", answer="L", reviewer="live-lee"
        )
        assert queued.status is RunStatus.QUEUED

        with worker_process(suffix, ledger):  # a different process continues it
            second = await handle.result(timeout=60)
        assert second.interrupt is not None and second.interrupt.question == "Ship size L?"
        await agent.resume(second.interrupt.interrupt_id, "answer", answer="yes", reviewer="lee")

        with worker_process(suffix, ledger):
            done = await handle.result(timeout=60)
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "charged o-7 40; size L; ship yes"
        record = await handle.status()
        assert record.checkpoint is None and record.attempt == 3

        # the charge ran once, in the first process, although three processes ran the run
        assert len(ledger.read_text().splitlines()) == 1
        # and it was recorded once, for the user, by the worker's own memory writes
        scope = await memory_scope(h, user="live-ada", agent_id=agent.id)

        async def charged_once() -> bool:
            entries = await scope.advanced.tools.catalog(names=[f"charge_{suffix}"])
            return bool(entries) and entries[0].stats.calls == 1

        assert await eventually(charged_once)


# longer than the suite's 120 s: the ticker's fire is waited for up to 240 s
@pytest.mark.timeout(360)
async def test_a_schedule_fires_from_the_ticker_to_a_worker(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:8]
    async with live_harness() as h:
        agent = h.wrap(in_a_worker, id=f"live-briefing-{suffix}")
        # at most one fire an hour (agent-runs' floor): the next minute but one
        minute = (datetime.now(UTC) + timedelta(minutes=2)).minute
        cron = f"{minute} * * * *"
        first = await agent.schedule(cron, "inbox", on_behalf_of="live-ada", tz="Europe/Berlin")
        # a redeploy schedules it again: agent-runs answers the one that exists, unchanged
        schedule = await agent.schedule(cron, "inbox", on_behalf_of="live-ada")
        assert schedule.schedule_id == first.schedule_id
        assert schedule.timezone == "Europe/Berlin"
        try:
            with worker_process(suffix, tmp_path / "ledger.txt"):

                async def fired() -> bool:
                    return bool(await _succeeded(h, agent.id))

                assert await eventually(fired, within=240, every=5)
            [summary] = await _succeeded(h, agent.id)
            run = await h.runs.get(summary.run_id, tenant=schedule.tenant_id)
            assert run is not None and run.output == "briefing for live-ada: inbox"
            assert run.metadata["schedule_id"] == schedule.schedule_id
        finally:
            assert isinstance(h.runs, RunsClient)  # RUNS_URL is set: agent-runs' client
            await h.runs.schedules.delete(schedule.schedule_id, tenant=schedule.tenant_id)


async def _succeeded(h: Harness, agent_id: str) -> list[RunSummary]:
    listed = h.runs.iterate(agent_id=agent_id, status=RunStatus.SUCCESS, tenant=await h.tenant())
    return [summary async for summary in listed]
