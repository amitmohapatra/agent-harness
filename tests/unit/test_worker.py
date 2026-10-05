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
from typing import Any

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunRecord, RunStatus
from trellis.harness.runs import LocalRuns
from trellis.harness.worker.__main__ import main, serve
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
    claimed: list[tuple[str, str, int | None]] = []
    attempt = agent._claimed

    async def recording(record: RunRecord, worker_id: str, **lease: Any) -> Any:
        claimed.append((record.run_id, worker_id, lease["lease_seconds"]))
        return await attempt(record, worker_id, **lease)

    monkeypatch.setattr(agent, "_claimed", recording)
    worker = harness.worker([agent, other])
    assert await worker.run_once() is True
    assert claimed == [(handle.run_id, worker.worker_id, worker.loop.lease_seconds)]
    assert (await handle.result(timeout=5)).answer == "x"
    assert await worker.run_once() is False  # nothing queued


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


async def test_cancelling_a_running_worker_cancels_the_runs_it_holds(harness: Harness) -> None:
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
    assert "ended or taken over elsewhere; nothing written" in caplog.text


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
