"""The worker loop on the in-process run store: idling, claim and heartbeat failures, a lost
lease, a worker stopped mid-run, and ``python -m trellis.worker``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunRecord, RunStatus
from trellis.harness import worker as worker_module
from trellis.harness.runs import LocalRuns
from trellis.runs import Claimed
from trellis.worker import main, serve


def local(harness: Harness) -> LocalRuns:
    assert isinstance(harness.runs, LocalRuns)
    return harness.runs


async def test_a_worker_needs_an_agent(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="at least one agent"):
        harness.worker([])


async def test_an_idle_worker_keeps_asking_for_work(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    monkeypatch.setattr(worker_module, "IDLE_SECONDS", 0.0)
    agent = harness.wrap(echo, id="echo")
    asked: list[str] = []
    idled = asyncio.Event()
    claim = harness.runs.claim

    async def counting(worker_id: str, agent_ids: Any, **lease: Any) -> Claimed | None:
        asked.append(worker_id)
        if len(asked) == 3:
            idled.set()
        return await claim(worker_id, agent_ids, **lease)

    monkeypatch.setattr(harness.runs, "claim", counting)
    task = asyncio.create_task(harness.worker([agent]).run())
    await idled.wait()
    handle = await agent.start("late work", user="u")  # queued while the worker idles
    assert (await handle.result(timeout=5)).answer == "late work"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_a_failed_claim_is_logged_and_means_no_work(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    async def unreachable(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("agent-runs is down")

    worker = harness.worker([harness.wrap(echo, id="echo")])
    monkeypatch.setattr(harness.runs, "claim", unreachable)
    with caplog.at_level(logging.WARNING, logger="trellis.worker"):
        assert await worker.run_once() is False
    assert "claim failed: agent-runs is down" in caplog.text


async def test_a_run_that_breaks_the_harness_is_logged_and_the_worker_goes_on(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    agent = harness.wrap(echo, id="echo")
    handle = await agent.start("x", user="u")

    async def broken(record: RunRecord, worker_id: str, **lease: Any) -> Any:
        raise RuntimeError("the run store refused the finish")

    monkeypatch.setattr(agent, "_claimed", broken)
    with caplog.at_level(logging.ERROR, logger="trellis.worker"):
        assert await harness.worker([agent]).run_once() is True
    assert f"run {handle.run_id} failed in the worker" in caplog.text


async def test_a_failed_heartbeat_is_logged_and_the_run_continues(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    beat = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        await beat.wait()
        return "done"

    async def flaky(run_id: str, worker_id: str, **lease: Any) -> None:
        beat.set()
        raise ConnectionError("blip")

    monkeypatch.setattr(worker_module, "LEASE_SECONDS", 0.03)
    agent = harness.wrap(slow, id="slow")
    handle = await agent.start("x", user="u")
    monkeypatch.setattr(harness.runs, "heartbeat", flaky)
    with caplog.at_level(logging.WARNING, logger="trellis.worker"):
        assert await harness.worker([agent]).run_once() is True
    assert (await handle.result(timeout=5)).answer == "done"
    assert f"heartbeat for {handle.run_id} failed: blip" in caplog.text


async def test_a_lease_another_worker_took_stops_the_run_and_writes_nothing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = local(harness)

    async def hijacked(input: str, agent: Runtime) -> str:
        # the lease lapses and another worker claims the run while this one still works it
        runs._leases[agent.run_id] = ("other-worker", datetime.now(UTC) + timedelta(minutes=1))
        await asyncio.Event().wait()
        return "never"

    monkeypatch.setattr(worker_module, "LEASE_SECONDS", 0.03)
    agent = harness.wrap(hijacked, id="hijacked")
    handle = await agent.start("x", user="u")
    with caplog.at_level(logging.INFO):
        assert await harness.worker([agent]).run_once() is True
    record = await handle.status()
    assert record.status is RunStatus.RUNNING  # the other worker's run, untouched
    assert f"lease on {handle.run_id} lost" in caplog.text
    assert "taken over by another worker; nothing written" in caplog.text


async def test_stopping_a_worker_cancels_the_runs_it_holds(harness: Harness) -> None:
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    agent = harness.wrap(forever, id="forever")
    handle = await agent.start("x", user="u")
    task = asyncio.create_task(harness.worker([agent]).run())
    await started.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert (await handle.status()).status is RunStatus.CANCELLED


async def test_a_worker_stopped_as_its_run_finishes_still_stops(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run ends normally, but the worker was cancelled in the same moment: the
    cancellation is the worker's own and is not swallowed."""

    async def echo(input: str, agent: Runtime) -> str:
        return input

    agent = harness.wrap(echo, id="echo")
    started = await agent._start("x", user="u", thread=None, tenant=None)
    record = await harness.runs.start(started, queue=True)
    worker = harness.worker([agent])
    claimed = await harness.runs.claim(worker.worker_id, ["echo"])
    assert claimed is not None and claimed.run.run_id == record.run_id
    outer: list[asyncio.Task[Any]] = []

    async def finishing(record: RunRecord, worker_id: str, **lease: Any) -> str:
        asyncio.get_running_loop().call_soon(outer[0].cancel)
        return "finished"

    monkeypatch.setattr(agent, "_claimed", finishing)
    execute = asyncio.create_task(worker._execute(claimed.run))
    outer.append(execute)
    with pytest.raises(asyncio.CancelledError):
        await execute


# --------------------------------------------------------------------------- stopping


async def test_a_stopped_worker_lets_its_runs_finish_and_claims_nothing_more(
    harness: Harness,
) -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        started.set()
        await release.wait()
        return "done"

    agent = harness.wrap(slow, id="slow")
    first = await agent.start("a", user="u")
    worker = harness.worker([agent], concurrency=1)
    task = asyncio.create_task(worker.run())
    await started.wait()
    second = await agent.start("b", user="u")  # queued after the stop: left for later
    worker.stop()
    await asyncio.sleep(0.01)
    assert not task.done()  # waiting for the run it holds
    release.set()
    await asyncio.wait_for(task, 5)
    assert (await first.status()).status is RunStatus.SUCCESS
    assert (await second.status()).status is RunStatus.QUEUED


async def test_a_run_past_the_grace_period_is_released_not_cancelled(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    monkeypatch.setattr(worker_module, "GRACE_SECONDS", 0.05)
    agent = harness.wrap(forever, id="forever")
    handle = await agent.start("x", user="u")
    worker = harness.worker([agent])
    task = asyncio.create_task(worker.run())
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="trellis.worker"):
        worker.stop()
        await asyncio.wait_for(task, 5)
    # nothing written: the lease lapses and another worker runs it again
    assert (await handle.status()).status is RunStatus.RUNNING
    assert f"run {handle.run_id} released" in caplog.text


async def test_a_second_stop_releases_the_runs_at_once(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    monkeypatch.setattr(worker_module, "GRACE_SECONDS", 60.0)
    agent = harness.wrap(forever, id="forever")
    handle = await agent.start("x", user="u")
    worker = harness.worker([agent])
    task = asyncio.create_task(worker.run())
    await started.wait()
    worker.stop()
    await asyncio.sleep(0.01)
    worker.stop()
    await asyncio.wait_for(task, 5)
    assert (await handle.status()).status is RunStatus.RUNNING


async def test_a_stop_while_every_slot_is_busy_is_heard(harness: Harness) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    worker = harness.worker([harness.wrap(echo, id="echo")], concurrency=1)
    slots = asyncio.Semaphore(1)
    await slots.acquire()
    waiting = asyncio.create_task(worker._slot(slots))
    await asyncio.sleep(0)
    worker.stop()
    assert await waiting is False
    assert await worker._slot(asyncio.Semaphore(1)) is False  # stopped: no slot at all


async def test_a_slot_freed_as_the_stop_comes_is_given_back(harness: Harness) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    worker = harness.worker([harness.wrap(echo, id="echo")], concurrency=1)
    slots = asyncio.Semaphore(1)
    await slots.acquire()
    waiting = asyncio.create_task(worker._slot(slots))
    await asyncio.sleep(0)
    slots.release()
    worker.stop()
    assert await waiting is False
    assert not slots.locked()


async def test_an_idle_worker_backs_off_up_to_a_cap(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    real_timeout = asyncio.timeout

    def recording(delay: float | None) -> Any:
        waits.append(delay or 0.0)
        return real_timeout(0)

    async def echo(input: str, agent: Runtime) -> str:
        return input

    worker = harness.worker([harness.wrap(echo, id="echo")])
    monkeypatch.setattr(worker_module.asyncio, "timeout", recording)
    for rounds in (1, 2, 3, 30):
        await worker._idle(rounds)
    first, second, third, capped = waits
    assert 0.25 <= first <= 0.5 and 0.5 <= second <= 1.0 and 1.0 <= third <= 2.0
    assert worker_module.IDLE_MAX_SECONDS / 2 <= capped <= worker_module.IDLE_MAX_SECONDS


async def test_the_default_concurrency_is_the_cpu_count_within_bounds(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    agent = harness.wrap(echo, id="echo")
    for cpus, expected in ((None, 1), (2, 2), (64, 8)):
        monkeypatch.setattr(worker_module.os, "cpu_count", lambda cpus=cpus: cpus)
        assert harness.worker([agent]).concurrency == expected
    assert harness.worker([agent], concurrency=3).concurrency == 3
    configured = Harness(config=Settings(worker_concurrency=5))
    assert configured.worker([configured.wrap(echo, id="echo")]).concurrency == 5
    with pytest.raises(ConfigurationError, match="at least one run"):
        harness.worker([agent], concurrency=-1)


# --------------------------------------------------------------------------- the CLI


async def test_sigterm_stops_the_served_worker_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import signal

    async def echo(input: str, agent: Runtime) -> str:
        return input

    harness = Harness(config=Settings())
    harness.wrap(echo, id="echo")
    task = asyncio.create_task(serve(harness, concurrency=1))
    await asyncio.sleep(0.05)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 5)  # returned on its own: no cancellation needed
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGTERM) is False  # serve removed its handler


async def test_serving_a_harness_closes_it_when_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    harness = Harness(config=Settings())
    harness.wrap(echo, id="echo")
    closed: list[bool] = []
    close = harness.aclose

    async def recording_close() -> None:
        closed.append(True)
        await close()

    monkeypatch.setattr(harness, "aclose", recording_close)
    task = asyncio.create_task(serve(harness))
    await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert closed == [True]


def test_main_serves_the_named_harness_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "served_app.py").write_text(
        "from trellis import Harness, Settings\n"
        "h = Harness(config=Settings())\n"
        "async def echo(input, agent):\n    return input\n"
        "h.wrap(echo, id='echo')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    served: list[list[str | int | None]] = []

    async def once(harness: Harness, *, concurrency: int | None) -> None:
        served.append([*harness.agents, concurrency])

    async def interrupted(harness: Harness, *, concurrency: int | None) -> None:
        raise KeyboardInterrupt

    import trellis.worker as cli

    monkeypatch.setattr(cli, "serve", once)
    assert main(["served_app:h"]) == 0
    assert main(["served_app:h", "--concurrency", "3"]) == 0
    monkeypatch.setattr(cli, "serve", interrupted)
    assert main(["served_app:h"]) == 0  # Ctrl-C is a clean stop
    assert served == [["echo", None], ["echo", 3]]


def test_running_the_module_without_a_target_prints_its_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["trellis.worker"])
    monkeypatch.delitem(sys.modules, "trellis.worker")
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("trellis.worker", run_name="__main__")
    assert exited.value.code == 2
    assert "usage: python -m trellis.worker module:harness_attribute" in capsys.readouterr().err
