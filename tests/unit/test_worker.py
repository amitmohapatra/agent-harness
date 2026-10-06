"""The harness worker: ``trellis.runs.Worker`` (whose loop the SDK's own tests cover) running
wrapped agents — the claimed run's next attempt, the background writes around the loop, a
released or lost run writing nothing — and ``python -m trellis.harness.worker``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.pipeline import STOPPED_EARLY, _out_of_time
from trellis.harness.runs import LocalRuns
from trellis.harness.worker.__main__ import main, serve
from trellis.runs import RELEASED, Job
from trellis.runs import worker as claim_loop


def local(harness: Harness) -> LocalRuns:
    assert isinstance(harness.runs, LocalRuns)
    return harness.runs


async def echo(input: str, agent: Runtime) -> str:
    return input


async def test_a_worker_needs_an_agent_and_at_least_one_slot(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="at least one agent"):
        harness.worker([])
    with pytest.raises(ConfigurationError, match="at least one run"):
        harness.worker([harness.wrap(echo, id="echo")], concurrency=0)


async def test_the_concurrency_is_the_argument_else_the_setting_else_the_sdks_default(
    harness: Harness,
) -> None:
    agent = harness.wrap(echo, id="echo")
    assert harness.worker([agent]).concurrency == claim_loop.default_concurrency()
    assert harness.worker([agent], concurrency=3).concurrency == 3
    configured = Harness(config=Settings(worker_concurrency=5))
    assert configured.worker([configured.wrap(echo, id="echo")]).concurrency == 5
    worker = harness.worker([agent])
    assert worker.worker_id == worker.loop.worker_id and worker.loop.store is harness.runs


async def test_a_claimed_run_is_its_agents_next_attempt_as_the_lease_holder(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = harness.wrap(echo, id="echo")
    other = harness.wrap(echo, id="other")
    handle = await agent.start("x", user="u")
    jobs: list[Job] = []
    attempt = agent.execute

    async def recording(job: Job) -> Any:
        jobs.append(job)
        return await attempt(job)

    monkeypatch.setattr(agent, "execute", recording)
    worker = harness.worker([agent, other])
    assert await worker.run_once() is True
    [job] = jobs
    assert (job.record.run_id, job.worker_id) == (handle.run_id, worker.worker_id)
    assert job.lease_seconds == worker.loop.lease_seconds
    assert (await handle.result(timeout=5)).answer == "x"
    assert await worker.run_once() is False  # nothing queued
    with pytest.raises(ConfigurationError, match=f"run {handle.run_id} is echo's, not other's"):
        await other.execute(job)


async def test_run_starts_the_background_writes_and_drains_them_when_stopped(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    start, drain = harness.writes.start, harness.writes.drain

    def starting() -> None:
        calls.append("start")
        start()

    async def draining() -> None:
        calls.append("drain")
        await drain()

    monkeypatch.setattr(harness.writes, "start", starting)
    monkeypatch.setattr(harness.writes, "drain", draining)
    agent = harness.wrap(echo, id="echo")
    worker = harness.worker([agent])
    task = asyncio.create_task(worker.run())
    handle = await agent.start("late work", user="u")  # queued while the worker idles
    assert (await handle.result(timeout=5)).answer == "late work"
    assert calls == ["start"]
    worker.stop()
    await asyncio.wait_for(task, 5)
    assert calls == ["start", "drain"]


async def test_cancelling_a_running_worker_leaves_the_runs_it_holds_to_their_leases(
    harness: Harness,
) -> None:
    """Nobody asked to cancel the runs: the worker went away, so their leases lapse and
    agent-runs queues them again (``job.cancel_requested`` is what ends a run CANCELLED)."""
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
    assert (await handle.status()).status is RunStatus.RUNNING  # nothing written


async def test_a_worker_run_someone_cancelled_ends_cancelled(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``agent.cancel`` of a run a worker holds: its heartbeat says ``cancel_requested``, the
    worker stops the handler, and the run ends ``CANCELLED``."""
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    async def soon(seconds: float) -> None:  # the heartbeat's wait, cut short
        await asyncio.sleep(0.01)

    monkeypatch.setattr(claim_loop, "_sleep", soon)
    agent = harness.wrap(forever, id="forever")
    handle = await agent.start("x", user="u")
    working = asyncio.create_task(harness.worker([agent]).run_once())
    await started.wait()
    await handle.cancel(reason="not needed")
    assert await working is True
    assert (await handle.status()).status is RunStatus.CANCELLED


async def test_a_stream_closed_after_its_run_ended_elsewhere_writes_nothing(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    stream = harness.wrap(forever, id="forever").stream("x", user="u")
    first = await anext(stream)
    await started.wait()
    await harness.runs.finish(first.run_id, RunStatus.ERROR)  # ended by someone else
    with caplog.at_level(logging.INFO):
        await stream.aclose()  # the caller goes away: the run's task is cancelled
        await asyncio.sleep(0.05)
    assert "was ended or taken over elsewhere; nothing written" in caplog.text
    record = await harness.runs.get(first.run_id)
    assert record is not None and record.status is RunStatus.ERROR


async def test_a_released_run_writes_nothing(harness: Harness) -> None:
    """A second stop releases the runs held (``trellis.runs.RELEASED``): the pipeline records
    no ending, and the worker hands the run back to the queue for another worker at once."""
    started = asyncio.Event()

    async def forever(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    agent = harness.wrap(forever, id="forever")
    handle = await agent.start("x", user="u")
    worker = harness.worker([agent])
    task = asyncio.create_task(worker.run())
    await started.wait()
    worker.stop()
    await asyncio.sleep(0.01)
    assert not task.done()  # the run held gets its grace period
    worker.stop()
    await asyncio.wait_for(task, 5)
    record = await handle.status()
    assert record.status is RunStatus.QUEUED and record.attempt == 2


async def test_a_lease_another_worker_took_stops_the_run_and_writes_nothing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = local(harness)

    async def hijacked(input: str, agent: Runtime) -> str:
        # the lease lapses and another worker claims the run while this one still works it
        runs._leases[agent.run_id] = ("other-worker", datetime.now(UTC) + timedelta(minutes=1))
        await asyncio.Event().wait()
        return "never"

    async def soon(seconds: float) -> None:  # the heartbeat's wait, cut short
        await asyncio.sleep(0.01)

    monkeypatch.setattr(claim_loop, "_sleep", soon)
    agent = harness.wrap(hijacked, id="hijacked")
    handle = await agent.start("x", user="u")
    with caplog.at_level(logging.INFO):
        assert await harness.worker([agent]).run_once() is True
    record = await handle.status()
    assert record.status is RunStatus.RUNNING  # the other worker's run, untouched
    assert f"lease on {handle.run_id} lost" in caplog.text


# --------------------------------------------------------------------------- the CLI


async def test_sigterm_stops_the_served_worker_gracefully() -> None:
    import os
    import signal

    harness = Harness(config=Settings())
    harness.wrap(echo, id="echo")
    task = asyncio.create_task(serve(harness, concurrency=1))
    await asyncio.sleep(0.05)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 5)  # returned on its own: no cancellation needed
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGTERM) is False  # serve removed its handler


async def test_serving_a_harness_closes_it_when_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
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

    import trellis.harness.worker.__main__ as cli

    monkeypatch.setattr(cli, "serve", once)
    assert main(["served_app:h"]) == 0
    assert main(["served_app:h", "--concurrency", "3"]) == 0
    monkeypatch.setattr(cli, "serve", interrupted)
    assert main(["served_app:h"]) == 0  # Ctrl-C is a clean stop
    assert served == [["echo", None], ["echo", 3]]


def test_running_the_module_without_a_target_prints_its_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["trellis.harness.worker"])
    monkeypatch.delitem(sys.modules, "trellis.harness.worker.__main__", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("trellis.harness.worker", run_name="__main__")
    assert exited.value.code == 2
    usage = "usage: python -m trellis.harness.worker module:harness_attribute"
    assert usage in capsys.readouterr().err


def test_a_worker_stopping_an_attempt_out_of_working_time_is_told_from_its_other_stops() -> None:
    """asyncio runs a timer that is due within the clock's resolution early: the worker's
    clock for the working time may stop an attempt with a hair of it left."""

    def job(left: float | None, *, cancelled: bool = False) -> Any:
        return SimpleNamespace(remaining_seconds=left, cancel_requested=cancelled)

    stop = asyncio.CancelledError()
    assert _out_of_time(job(0.0), stop) and _out_of_time(job(STOPPED_EARLY / 1000), stop)
    assert not _out_of_time(job(0.5), stop)  # a lost lease, the worker cancelled
    assert not _out_of_time(job(None), stop)  # no working time to run out of
    assert not _out_of_time(job(0.0, cancelled=True), stop)  # someone cancelled it
    assert not _out_of_time(job(0.0), asyncio.CancelledError(RELEASED))  # released
    assert not _out_of_time(None, stop)  # not a worker's run
