"""Durable runs: ``start`` queues, a worker claims and executes under a lease, a resume sends
the run back to the queue, schedules fire runs — against the in-process run store."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.clients.runs import LocalRuns
from trellis.worker import load, main


async def approve_then_finish(input: dict[str, str], agent: Runtime) -> str:
    ok = await agent.ask(f"Send the report on {input['topic']}?", assignee="role:editor")
    return f"sent={ok}"


async def test_a_started_run_is_queued_until_a_worker_claims_it(harness: Harness) -> None:
    agent = harness.wrap(approve_then_finish, id="reporter")
    handle = await agent.start({"topic": "tides"}, user="u1")
    assert (await handle.status()).status is RunStatus.QUEUED
    worker = harness.worker([agent])
    assert await worker.run_once() is True
    paused = await handle.result(timeout=5)
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    inbox = await harness.runs.list_paused("default", assignee="role:editor")
    assert [r.run_id for r in inbox] == [handle.run_id]

    resumed = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="ed"
    )
    assert resumed.status is RunStatus.QUEUED  # a queued run is continued by a worker
    assert await worker.run_once() is True
    done = await handle.result(timeout=5)
    assert done.status is RunStatus.SUCCESS and done.answer == "sent=yes"
    assert await worker.run_once() is False


async def test_another_process_resumes_from_the_checkpoint_alone() -> None:
    """Two harnesses share nothing but the run store (as two worker processes share
    agent-runs): the second continues from the checkpoint and the last resolution, so no
    question is asked twice and no tool runs twice."""
    charged: list[int] = []

    def charge(amount: int) -> str:
        charged.append(amount)
        return f"charged {amount}"

    async def billing(input: str, agent: Runtime) -> str:
        receipt = await agent.tools.call("charge", amount=5)
        size = await agent.ask("Which size?", assignee="role:ops")
        ship = await agent.ask(f"Ship {size}?", options=["yes", "no"])
        return f"{receipt}; {size}; {ship}"

    store = LocalRuns()
    processes = [Harness(config=Settings()) for _ in range(3)]
    agents = []
    for h in processes:
        h.runs = store
        agents.append(h.wrap(billing, id="billing", tools=[charge]))
    try:
        handle = await agents[0].start("order 7", user="u")
        assert await processes[0].worker([agents[0]]).run_once()
        first = await handle.result(timeout=5)
        assert first.interrupt is not None
        await agents[1].resume(first.interrupt.interrupt_id, "answer", answer="L", reviewer="r")
        assert await processes[1].worker([agents[1]]).run_once()
        second = await handle.result(timeout=5)
        assert second.interrupt is not None and second.interrupt.question == "Ship L?"
        await agents[2].resume(second.interrupt.interrupt_id, "answer", answer="yes", reviewer="r")
        assert await processes[2].worker([agents[2]]).run_once()
        done = await handle.result(timeout=5)
    finally:
        for h in processes:
            await h.aclose()
    assert done.status is RunStatus.SUCCESS and done.answer == "charged 5; L; yes"
    assert charged == [5]  # the tool ran once, in the first process
    assert (await store.get(handle.run_id)).checkpoint is None  # type: ignore[union-attr]


async def test_a_worker_runs_concurrently_and_stops_cleanly(harness: Harness) -> None:
    started = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(0.05)
        return input

    agent = harness.wrap(slow, id="slow")
    handles = [await agent.start(f"job{i}", user="u") for i in range(3)]
    task = asyncio.create_task(harness.worker([agent], concurrency=3).run())
    results = [await h.result(timeout=5) for h in handles]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert [r.answer for r in results] == ["job0", "job1", "job2"]


async def test_a_queued_input_must_be_json(harness: Harness) -> None:
    async def fn(input: object, agent: Runtime) -> str:
        return "x"

    with pytest.raises(ConfigurationError, match="JSON"):
        await harness.wrap(fn, id="j").start(object(), user="u")


async def test_a_schedule_fires_runs_for_the_worker(harness: Harness) -> None:
    async def briefing(input: str, agent: Runtime) -> str:
        return f"briefing for {agent.user}: {input}"

    agent = harness.wrap(briefing, id="briefing")
    schedule = await agent.schedule("0 7 * * 1-5", "inbox", on_behalf_of="ada", tz="Europe/Berlin")
    assert schedule.on_behalf_of == "ada" and schedule.timezone == "Europe/Berlin"
    runs = harness.runs
    assert isinstance(runs, LocalRuns)
    runs._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    assert await harness.worker([agent]).run_once() is True
    [record] = [r for r in runs._runs.values() if r.metadata.get("schedule_id")]
    assert record.status is RunStatus.SUCCESS and record.output == "briefing for ada: inbox"


async def test_a_lost_lease_stops_the_run_without_writing(harness: Harness) -> None:
    release = asyncio.Event()

    async def long(input: str, agent: Runtime) -> str:
        await release.wait()
        return "late"

    agent = harness.wrap(long, id="long")
    handle = await agent.start("x", user="u")
    worker = harness.worker([agent])
    record = await harness.runs.claim(worker.worker_id, ["long"], 30)
    assert record is not None
    execution = asyncio.create_task(agent._claimed(record, worker.worker_id))
    await asyncio.sleep(0.01)
    execution.cancel()  # what the heartbeat does on a lost lease
    with contextlib.suppress(asyncio.CancelledError):
        await execution
    assert (await handle.status()).status is RunStatus.CANCELLED


def test_the_worker_cli_loads_a_harness_by_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "deployed.py").write_text(
        "from trellis import Harness, Settings\n"
        "h = Harness(config=Settings())\n"
        "async def echo(input, agent):\n    return input\n"
        "agent = h.wrap(echo, id='echo')\n"
        "empty = Harness(config=Settings())\n"
        "not_a_harness = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert list(load("deployed:h").agents) == ["echo"]
    for bad in ("deployed:not_a_harness", "deployed:empty", "deployed"):
        with pytest.raises(SystemExit):
            load(bad)
    assert main([]) == 2
