"""The load benchmark (``benchmark/load``) without a model: a worker runs its lanes at once and
measures each run, the levels climb until a limit, and the leak check reports the memory's
growth after the warm-up — the processes run inline here (the command spawns them)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.load import __main__ as cli
from benchmark.load import run
from benchmark.load.run import Measured, Plan
from trellis import Settings


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark.cost import run as cost_run

    monkeypatch.setattr(cost_run.Settings, "from_env", classmethod(lambda cls: Settings()))


async def test_a_worker_runs_its_lanes_at_once_and_measures_each_run() -> None:
    measured = await run.work(Plan(concurrency=3, runs=6, pace=0.0, sample_seconds=0.01))
    assert len(measured.latencies) == 6 and not measured.errors
    assert measured.cpu_seconds is not None and measured.peak_rss_mb
    assert measured.samples[-1][2] == 6


async def test_a_worker_given_seconds_runs_until_they_pass() -> None:
    measured = await run.work(Plan(concurrency=2, seconds=0.3, pace=0.01))
    assert measured.latencies and not measured.errors


async def test_a_failed_run_is_an_error_not_a_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(run.Paced):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("the model is down")

    monkeypatch.setattr(run, "Paced", Broken)
    measured = await run.work(Plan(concurrency=1, runs=2, pace=0.0))
    assert not measured.latencies and len(measured.errors) == 2

    def boom(h: Any, plan: Plan) -> Any:
        class Agent:
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                raise ValueError("no")

        return Agent()

    monkeypatch.setattr(run, "agent", boom)
    measured = await run.work(Plan(concurrency=1, runs=1))
    assert measured.errors == ["ValueError: no"]


def test_the_levels_climb_until_a_limit_and_the_leak_check_reports_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inline(plans: list[Plan]) -> list[Measured]:
        found = []
        for plan in plans:
            m = Measured(latencies=[0.1] * (plan.runs or 4), cpu_seconds=0.5, peak_rss_mb=100.0)
            if plan.concurrency >= 10:
                m.errors = ["the model is busy"] * (plan.runs or 1)
            m.samples = [(0.0, 90.0, 0), (10.0, 100.0, 10), (50.0, 104.0, 50)]
            found.append(m)
        return found

    monkeypatch.setattr(cli, "in_processes", inline)
    out = tmp_path / "load"
    argv = [
        "--levels",
        "1,10,50",
        "--workers",
        "1,2",
        "--rounds",
        "2",
        "--out",
        str(out),
        "--leak-minutes",
        "1",
        "--watch",
        "me=1",
    ]
    assert cli.main(argv) == 0
    report = json.loads(out.with_suffix(".json").read_text())
    rows = [(r["workers"], r["concurrency"], r["runs"], r["limit"]) for r in report["levels"]]
    assert rows == [
        (1, 1, 2, None),
        (1, 10, 40, "errors: 20 of 40 runs"),  # the climb stops at the limit
        (1, 1, 2, None),  # two workers, one lane: one process
        (2, 10, 20, None),  # five lanes on each: under the fake's errors
        (2, 50, 200, "errors: 100 of 200 runs"),
    ]
    assert report["leak"]["growth_mb"] == 4.0 and report["leak"]["growth_kb_per_run"] == 102.4
    table = out.with_suffix(".md").read_text()
    assert "| 1 | 10 | 40 |" in table and "Leak check: one worker, 10 at once" in table
    assert "me " in table


def test_the_limits_and_the_split() -> None:
    args = cli.arguments(["--max-p95", "2"])
    assert cli.limit_of(args, {"runs": 10, "errors": 0, "p95": 3.0}) == "latency: p95 3.0s > 2s"
    assert cli.limit_of(args, {"runs": 0, "errors": 0, "p95": None}) is None
    assert cli.split(5, 2) == [3, 2] and cli.split(1, 2) == [1]
    for argv in (["--levels", "0"], ["--watch", "x"]):
        with pytest.raises(SystemExit):
            cli.arguments(argv)


def test_a_level_waits_for_free_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "free_mb", lambda: 100.0)
    out = tmp_path / "load"
    assert cli.main(["--levels", "1", "--workers", "1", "--out", str(out)]) == 0
    [row] = json.loads(out.with_suffix(".json").read_text())["levels"]
    assert row["limit"] == "memory: 100 MB free"


def test_free_memory_is_read_from_proc() -> None:
    assert (cli.free_mb() or 0) > 0
