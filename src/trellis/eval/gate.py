"""The regression gate CI runs (design §11).

    python -m trellis.eval gate \
        --baseline benchmark-results.json --current build/benchmark-results.json \
        --judge build/experiment.json --max-latency-regression 20 --max-score-drop 0.05

Two artifacts, one verdict. The benchmark artifact the harness already publishes says what
the harness *costs*; an experiment summary says what an agent is *worth*. A change that made
answers better and turns 40% slower is a decision somebody has to take deliberately, which is
what a gate is for.

Three rules that keep it honest:

* **a missing baseline fails.** Silence is not a pass; ``--allow-missing-baseline`` is the
  deliberate, visible exception for the first run on a new metric.
* **a missing judge score is not a zero.** An experiment where the judge abstained on
  everything reports "no score", and the gate says so instead of failing on a number nobody
  measured.
* **thresholds come from the command line or configuration, never from the artifact.** An
  artifact cannot widen the gate that is checking it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: The percentiles compared in a benchmark artifact, in the order they are reported.
PERCENTILES = ("p50", "p90", "p95", "p99", "mean")

#: Where the harness publishes its own benchmark numbers.
DEFAULT_BENCHMARK = "benchmark-results.json"


class GateThresholds(BaseModel):
    """How much worse is allowed before the gate fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Percent a latency percentile may regress (20 = "up to 20% slower").
    max_latency_regression_pct: float = Field(default=20.0, ge=0.0)
    #: Absolute drop allowed in the judge's mean score.
    max_score_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Floor the judge's mean score must clear, whatever the baseline was.
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Ignore latencies below this: percent changes on sub-millisecond numbers are noise.
    latency_floor_ms: float = Field(default=0.5, ge=0.0)


class GateFinding(BaseModel):
    """One comparison, and whether it passed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    baseline: float | None = None
    current: float | None = None
    delta: float | None = None
    allowed: float | None = None
    passed: bool = True
    note: str | None = None

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        if self.baseline is None or self.current is None:
            return f"{mark}  {self.metric}: {self.note or 'not compared'}"
        return (
            f"{mark}  {self.metric}: {self.baseline:.4g} -> {self.current:.4g} "
            f"({self.delta:+.4g}, allowed {self.allowed:.4g})"
            + (f" — {self.note}" if self.note else "")
        )


class GateReport(BaseModel):
    """What the gate decided, in full. Printed by the CLI, asserted by tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: list[GateFinding] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings)

    @property
    def failures(self) -> list[GateFinding]:
        return [f for f in self.findings if not f.passed]

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def render(self) -> str:
        head = "regression gate: " + ("passed" if self.passed else "FAILED")
        return "\n".join([head, *(f.line() for f in self.findings)])


class RegressionGate:
    """Compares a candidate against a baseline under fixed thresholds."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def evaluate(
        self,
        *,
        baseline: dict[str, Any] | None,
        current: dict[str, Any] | None,
        baseline_judge: dict[str, Any] | None = None,
        current_judge: dict[str, Any] | None = None,
        allow_missing_baseline: bool = False,
    ) -> GateReport:
        findings: list[GateFinding] = []
        findings.extend(
            self._latency(baseline, current, allow_missing_baseline=allow_missing_baseline)
        )
        findings.extend(
            self._judge(
                baseline_judge, current_judge, allow_missing_baseline=allow_missing_baseline
            )
        )
        if not findings:
            findings.append(GateFinding(metric="gate", passed=False, note="nothing was compared"))
        return GateReport(findings=findings)

    # ------------------------------------------------------------------ latency
    def _latency(
        self,
        baseline: dict[str, Any] | None,
        current: dict[str, Any] | None,
        *,
        allow_missing_baseline: bool,
    ) -> list[GateFinding]:
        if current is None:
            return []
        if baseline is None:
            return [
                GateFinding(
                    metric="latency",
                    passed=allow_missing_baseline,
                    note="no baseline benchmark artifact",
                )
            ]
        findings: list[GateFinding] = []
        base_results = _results(baseline)
        for scenario, values in _results(current).items():
            reference = base_results.get(scenario)
            if not isinstance(reference, dict) or not isinstance(values, dict):
                continue
            for percentile in PERCENTILES:
                finding = self._compare_latency(scenario, percentile, reference, values)
                if finding is not None:
                    findings.append(finding)
        return findings

    def _compare_latency(
        self, scenario: str, percentile: str, reference: dict[str, Any], values: dict[str, Any]
    ) -> GateFinding | None:
        before = _number(reference.get(percentile))
        after = _number(values.get(percentile))
        if before is None or after is None:
            return None
        metric = f"latency.{scenario}.{percentile}"
        if before < self.thresholds.latency_floor_ms:
            return GateFinding(
                metric=metric,
                baseline=before,
                current=after,
                delta=after - before,
                allowed=0.0,
                passed=True,
                note=f"below the {self.thresholds.latency_floor_ms}ms floor; not gated",
            )
        allowed = before * self.thresholds.max_latency_regression_pct / 100.0
        return GateFinding(
            metric=metric,
            baseline=before,
            current=after,
            delta=after - before,
            allowed=allowed,
            passed=(after - before) <= allowed,
        )

    # ------------------------------------------------------------------ judge
    def _judge(
        self,
        baseline: dict[str, Any] | None,
        current: dict[str, Any] | None,
        *,
        allow_missing_baseline: bool,
    ) -> list[GateFinding]:
        if current is None:
            return []
        after = _mean_score(current)
        if after is None:
            return [
                GateFinding(
                    metric="judge.mean_score",
                    passed=True,
                    note="the judge abstained on every item; no score to gate",
                )
            ]
        findings: list[GateFinding] = []
        if self.thresholds.min_score is not None:
            findings.append(
                GateFinding(
                    metric="judge.min_score",
                    baseline=self.thresholds.min_score,
                    current=after,
                    delta=after - self.thresholds.min_score,
                    allowed=0.0,
                    passed=after >= self.thresholds.min_score,
                )
            )
        before = _mean_score(baseline)
        if before is None:
            findings.append(
                GateFinding(
                    metric="judge.mean_score",
                    current=after,
                    passed=allow_missing_baseline,
                    note="no baseline judge score",
                )
            )
            return findings
        findings.append(
            GateFinding(
                metric="judge.mean_score",
                baseline=before,
                current=after,
                delta=after - before,
                allowed=-self.thresholds.max_score_drop,
                passed=(after - before) >= -self.thresholds.max_score_drop,
            )
        )
        return findings


def _results(artifact: dict[str, Any]) -> dict[str, Any]:
    results = artifact.get("results")
    return results if isinstance(results, dict) else {}


def _mean_score(artifact: dict[str, Any] | None) -> float | None:
    """The judge's mean, wherever the artifact keeps it: an experiment file writes a
    ``summary``, a hand-written one may put it at the top level."""
    if not isinstance(artifact, dict):
        return None
    for holder in (artifact.get("summary"), artifact):
        if isinstance(holder, dict):
            value = _number(holder.get("mean_score"))
            if value is not None:
                return value
    return None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return None
    loaded = json.loads(file.read_text())
    return loaded if isinstance(loaded, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    """The CI entry point. Non-zero means "do not merge this"."""
    parser = argparse.ArgumentParser(
        prog="python -m trellis.eval gate",
        description="Fail a build whose latency regressed or whose judge scores dropped.",
    )
    parser.add_argument("--baseline", default=DEFAULT_BENCHMARK, help="baseline benchmark JSON")
    parser.add_argument("--current", help="candidate benchmark JSON")
    parser.add_argument("--judge", help="candidate experiment JSON (mean_score)")
    parser.add_argument("--baseline-judge", help="baseline experiment JSON (mean_score)")
    parser.add_argument("--max-latency-regression", type=float, default=20.0, help="percent")
    parser.add_argument("--max-score-drop", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--allow-missing-baseline", action="store_true")
    args = parser.parse_args(argv)

    gate = RegressionGate(
        GateThresholds(
            max_latency_regression_pct=args.max_latency_regression,
            max_score_drop=args.max_score_drop,
            min_score=args.min_score,
        )
    )
    report = gate.evaluate(
        baseline=_load(args.baseline),
        current=_load(args.current),
        baseline_judge=_load(args.baseline_judge),
        current_judge=_load(args.judge),
        allow_missing_baseline=args.allow_missing_baseline,
    )
    print(report.render())  # noqa: T201 - a CLI: printing is the interface
    return report.exit_code


__all__ = [
    "DEFAULT_BENCHMARK",
    "PERCENTILES",
    "GateFinding",
    "GateReport",
    "GateThresholds",
    "RegressionGate",
    "main",
]
