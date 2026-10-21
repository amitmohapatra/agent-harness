"""The benchmark gate: a report compared with the stored baseline cell by cell on the tasks
both ran — a fall in success past its margin, or input tokens past theirs, fails it; a better
cell passes; a cell only one report measured is listed, not failed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.cost import gate


def report(*runs: tuple[int, str, str, bool, int], model: str = "m") -> dict[str, Any]:
    return {
        "model": model,
        "runs": [
            {"size": s, "variant": v, "task": t, "success": ok, "input_tokens": n}
            for s, v, t, ok, n in runs
        ],
    }


BASELINE = report(
    (10, "main", "a", True, 1000),
    (10, "main", "b", True, 1000),
    (10, "main", "c", False, 1000),
    (50, "main", "a", True, 7000),
    (100, "main", "a", False, 0),
)


def test_a_cell_as_good_or_better_passes_on_the_common_tasks() -> None:
    now = report((10, "main", "a", True, 900), (10, "main", "c", True, 1000))
    [cell], alone = gate.compare(now, BASELINE)
    assert cell.tasks == 2 and cell.success == (0.5, 1.0) and not cell.failed
    assert alone == [(50, "main"), (100, "main")]


def test_less_success_or_more_tokens_past_the_margin_fails() -> None:
    now = report((10, "main", "a", False, 1300), (10, "main", "b", True, 1000))
    [cell], _ = gate.compare(now, BASELINE)
    assert cell.failed == ("success 0.50 < 1.00 - 0.15", "input tokens 1,150 > 1,000 + 10%")
    [lenient], _ = gate.compare(now, BASELINE, success_margin=0.5, token_margin=0.2)
    assert not lenient.failed


def test_a_cell_with_no_common_task_and_a_baseline_of_no_tokens_are_not_failed() -> None:
    now = report((10, "main", "z", False, 1), (100, "main", "a", False, 5))
    compared, alone = gate.compare(now, BASELINE)
    assert [c.size for c in compared] == [100] and not compared[0].failed
    assert (10, "main") in alone


def test_the_command_exits_by_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(BASELINE))

    def run(now: dict[str, Any], *extra: str) -> int:
        path = tmp_path / "now.json"
        path.write_text(json.dumps(now))
        return gate.main([str(path), str(base), *extra])

    assert run(report((10, "main", "a", True, 1000))) == 0
    out = capsys.readouterr().out
    assert "| 10 | main | 1 | 1.00 → 1.00 | 1,000 → 1,000 | ok |" in out
    assert "Not compared (one report only, or no common task): 50 tools, main" in out
    assert run(report((10, "main", "a", False, 1000))) == 1
    assert run(report((10, "main", "a", False, 1000)), "--success-margin", "1") == 0
    assert run(report((10, "x", "a", True, 1))) == 2  # nothing in common
    assert run(report((10, "main", "a", True, 1), model="other")) == 2
    assert "different models" in capsys.readouterr().err


def test_the_stored_baseline_passes_against_itself() -> None:
    stored = Path(__file__).parents[2] / "benchmark/cost/results/baseline-local.json"
    compared, _ = gate.compare(json.loads(stored.read_text()), json.loads(stored.read_text()))
    assert compared and not any(c.failed for c in compared)
