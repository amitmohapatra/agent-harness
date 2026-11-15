"""Cancelling a background task (``background.stop``): its run of now, and each earlier run a
queued follow-up left behind it that is still the conversation's — a run the task's record
names that is not (gone from the store, another chat's) is left alone."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tests.unit.test_background import TASK, runtime, stored
from trellis.harness import background
from trellis.harness.runs import LocalRuns


async def test_a_cancelled_task_stops_its_runs_but_one_its_record_names_that_is_not_its_own() -> (
    None
):
    runs = LocalRuns()
    latest = await stored(runs, run_id="run_2")
    earlier = await stored(runs, run_id="run_1")
    elsewhere = await stored(runs, run_id="run_0", metadata={})  # no task's run
    cancelled: list[str] = []

    async def cancel(run_id: str, **kwargs: Any) -> None:
        cancelled.append(run_id)

    agent = SimpleNamespace(cancel=cancel)
    task = {
        **TASK,
        "run_id": latest.run_id,
        "superseded_run_ids": [elsewhere.run_id, "run_gone", earlier.run_id],
    }
    await background.stop(runtime(runs), agent, task, latest)  # type: ignore[arg-type]
    assert cancelled == ["run_2", "run_1"]  # the newest first, and only the task's own
