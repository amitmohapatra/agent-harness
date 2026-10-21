"""The benchmark gate (``make bench-gate``, ``python -m benchmark.cost.gate``): a cost
benchmark report checked against a stored baseline report, and a non-zero exit when a cell got
worse past its margin — so a change that makes the agent fail more tasks or send more tokens is
caught before it lands.

    python -m benchmark.cost.gate build/bench-cost.json benchmark/cost/results/baseline-local.json

Each catalogue size and variant both reports measured is compared on the tasks both ran (a
smoke run of a few tasks is compared with the baseline's runs of those tasks): its success rate
may fall by at most ``--success-margin`` (0.15 by default: a small model's runs vary), and its
mean input tokens may rise by at most ``--token-margin`` (10%) — the tokens of the successful
runs of the tasks both reports did (a run that stopped early sent fewer tokens for less work:
counting it would reward failing). A cell either report did not
measure is listed, not failed. A better cell passes: the gate is non-inferiority, not equality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: How much lower a cell's success rate may be than the baseline's.
SUCCESS_MARGIN: Final = 0.15
#: How much higher (a fraction) a cell's mean input tokens may be than the baseline's.
TOKEN_MARGIN: Final = 0.10


@dataclass(frozen=True)
class Compared:
    size: int
    variant: str
    tasks: int
    success: tuple[float, float]  # baseline, now
    input_tokens: tuple[float, float]
    failed: tuple[str, ...]


def _cells(report: dict[str, Any]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in report["runs"]:
        cells.setdefault((row["size"], row["variant"]), []).append(row)
    return cells


def compare(
    now: dict[str, Any],
    baseline: dict[str, Any],
    *,
    success_margin: float = SUCCESS_MARGIN,
    token_margin: float = TOKEN_MARGIN,
) -> tuple[list[Compared], list[tuple[int, str]]]:
    """Each cell both reports measured, compared on their common tasks; and the cells only one
    of them measured."""
    mine, theirs = _cells(now), _cells(baseline)
    compared, alone = [], sorted(set(mine) ^ set(theirs))
    for cell in sorted(set(mine) & set(theirs)):
        common = {r["task"] for r in mine[cell]} & {r["task"] for r in theirs[cell]}
        if not common:
            alone.append(cell)
            continue
        a = [r for r in theirs[cell] if r["task"] in common]
        b = [r for r in mine[cell] if r["task"] in common]
        success = (_mean(a, "success"), _mean(b, "success"))
        done = {r["task"] for r in a if r["success"]} & {r["task"] for r in b if r["success"]}
        tokens = (
            _mean([r for r in a if r["success"] and r["task"] in done], "input_tokens"),
            _mean([r for r in b if r["success"] and r["task"] in done], "input_tokens"),
        )
        failed = []
        if success[1] < success[0] - success_margin:
            failed.append(f"success {success[1]:.2f} < {success[0]:.2f} - {success_margin:g}")
        if tokens[0] > 0 and tokens[1] > tokens[0] * (1 + token_margin):
            failed.append(f"input tokens {tokens[1]:,.0f} > {tokens[0]:,.0f} + {token_margin:.0%}")
        compared.append(Compared(cell[0], cell[1], len(common), success, tokens, tuple(failed)))
    return compared, alone


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    """The mean of ``key`` over ``rows`` (0 for none: nothing to compare)."""
    return statistics.fmean(float(r[key] or 0) for r in rows) if rows else 0.0


def table(compared: Sequence[Compared], alone: Sequence[tuple[int, str]]) -> str:
    lines = [
        "| tools | variant | tasks | success (baseline → now) | input tokens of the runs both "
        "did (baseline → now) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for c in compared:
        verdict = "; ".join(c.failed) if c.failed else "ok"
        lines.append(
            f"| {c.size} | {c.variant} | {c.tasks} | {c.success[0]:.2f} → {c.success[1]:.2f} | "
            f"{c.input_tokens[0]:,.0f} → {c.input_tokens[1]:,.0f} | {verdict} |"
        )
    lines += [
        f"\nNot compared (one report only, or no common task): {size} tools, {variant}"
        for size, variant in alone
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.cost.gate", description=__doc__)
    parser.add_argument("now", help="the report measured now (<out>.json)")
    parser.add_argument("baseline", help="the stored baseline report")
    parser.add_argument("--success-margin", type=float, default=SUCCESS_MARGIN)
    parser.add_argument("--token-margin", type=float, default=TOKEN_MARGIN)
    args = parser.parse_args(argv)
    now, baseline = (json.loads(Path(p).read_text()) for p in (args.now, args.baseline))
    if now["model"] != baseline["model"]:
        sys.stderr.write(
            f"the reports measured different models: {now['model']}, {baseline['model']}\n"
        )
        return 2
    compared, alone = compare(
        now, baseline, success_margin=args.success_margin, token_margin=args.token_margin
    )
    sys.stdout.write(table(compared, alone))
    if not compared:
        sys.stderr.write("nothing to compare: no cell both reports measured\n")
        return 2
    return 1 if any(c.failed for c in compared) else 0


if __name__ == "__main__":
    raise SystemExit(main())
