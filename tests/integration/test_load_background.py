"""The background load benchmark's smoke (``benchmark/load/background.py``): both arms, a few
tasks and users, in process — every operation answered, no error, the tasks worked."""

from __future__ import annotations

import pytest

from benchmark.load import background as bench


@pytest.mark.parametrize(("arm", "ops"), [("A", "chat"), ("A", "direct"), ("B", "chat")])
async def test_the_load_benchmark_runs_both_arms(arm: str, ops: str) -> None:
    args = bench.arguments(
        [
            "--arm",
            arm,
            "--ops",
            ops,
            "--workers",
            "0",
            "--tasks",
            "6",
            "--users",
            "2",
            "--chats",
            "0",
            "--seconds",
            "2",
            "--warmup",
            "0.5",
            "--pace",
            "0.01",
            "--steps",
            "2000",
            "--think",
            "0.05",
        ]
    )
    report = await bench.drive(args)
    assert report["errors"] == {}, report
    # every operation answered (a user's tasks may all be cancelled before each kind came up)
    assert report["latency_ms"]["status"]["n"] >= 1, report
