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
from trellis.harness.clients.runs import LocalRuns
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

    async def counting(worker_id: str, agent_ids: Any, lease: float) -> RunRecord | None:
        asked.append(worker_id)
        if len(asked) == 3:
            idled.set()
        return await claim(worker_id, agent_ids, lease)

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

    async def unreachable(*args: Any) -> None:
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

    async def broken(record: RunRecord, worker_id: str) -> Any:
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

    async def flaky(run_id: str, worker_id: str, lease: float) -> None:
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
    record = await harness.runs.queued(await agent._start("x", user="u", thread=None, tenant=None))
    worker = harness.worker([agent])
    claimed = await harness.runs.claim(worker.worker_id, ["echo"], 60)
    assert claimed is not None and claimed.run_id == record.run_id
    outer: list[asyncio.Task[Any]] = []

    async def finishing(record: RunRecord, worker_id: str) -> str:
        asyncio.get_running_loop().call_soon(outer[0].cancel)
        return "finished"

    monkeypatch.setattr(agent, "_claimed", finishing)
    execute = asyncio.create_task(worker._execute(claimed))
    outer.append(execute)
    with pytest.raises(asyncio.CancelledError):
        await execute


# --------------------------------------------------------------------------- the CLI


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
    served: list[list[str]] = []

    async def once(harness: Harness) -> None:
        served.append(list(harness.agents))

    async def interrupted(harness: Harness) -> None:
        raise KeyboardInterrupt

    import trellis.worker as cli

    monkeypatch.setattr(cli, "serve", once)
    assert main(["served_app:h"]) == 0
    monkeypatch.setattr(cli, "serve", interrupted)
    assert main(["served_app:h"]) == 0  # Ctrl-C is a clean stop
    assert served == [["echo"]]


def test_running_the_module_without_a_target_prints_its_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["trellis.worker"])
    monkeypatch.delitem(sys.modules, "trellis.worker")
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("trellis.worker", run_name="__main__")
    assert exited.value.code == 2
    assert "usage: python -m trellis.worker module:harness_attribute" in capsys.readouterr().err
