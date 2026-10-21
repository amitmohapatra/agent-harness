"""The keep-or-drop decision of each ReAct item (``trellis.harness.adaptive``) from the
framework benchmark's rows: each ``react/C+<item>`` run paired with the ``react/C0`` run of
the same window, cell and index (the same task, the same seed), over the cells where the
item's signal can fire.

    python -m benchmark.cost.frameworks.items build/rb-8k/rows.jsonl build/rb-32k/rows.jsonl

An item is **kept** when, against ``C0`` (``ReAct`` as main builds it):

* its task success is no more than :data:`SUCCESS_MARGIN` below;
* its tokens per run (input and output) are at most :data:`TOKEN_MARGIN` more;
* its target metric (:data:`TARGETS`) is better, the 95% interval of the paired difference
  excluding 0.

Otherwise it is dropped, and the first rule it failed is why.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from benchmark.cost import stats
from benchmark.cost.frameworks.arms import ADAPTIVE
from benchmark.cost.frameworks.report import read_rows, runs_of

#: The arm every item is compared with: ``ReAct`` without the adaptive items.
BASE: Final = "react/C0"
#: Task success may fall at most this much (the gate's non-inferiority margin).
SUCCESS_MARGIN: Final = 0.15
#: Tokens per run may rise at most this share.
TOKEN_MARGIN: Final = 0.10


class Better(StrEnum):
    LOWER = "lower"
    HIGHER = "higher"


@dataclass(frozen=True)
class Target:
    """What an item is for: the row field it should move, which way, and the cells where its
    signal can fire (where it was run)."""

    metric: str
    better: Better
    cells: tuple[str, ...]


CORE: Final = ("short", "long", "many50", "many")
TARGETS: Final[dict[str, Target]] = {
    "token_count": Target("context_overflow", Better.LOWER, CORE),
    "strict_tools": Target("malformed_args", Better.LOWER, CORE),
    "native_output": Target("model_calls", Better.LOWER, ("output",)),
    "tool_repair": Target("model_calls", Better.LOWER, CORE),
    "stable_prefix": Target("cache_ratio", Better.HIGHER, CORE),
    "think": Target("success", Better.HIGHER, ("rules",)),
    "clear_inputs": Target("input_tokens", Better.LOWER, CORE),
    "unsupported_content": Target("errored", Better.LOWER, ("image",)),
    "rubric": Target("success", Better.HIGHER, ("long",)),
}


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("window"), row["cell"], row["index"])


def _value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    return None if value is None else float(value)


def _tokens(row: dict[str, Any]) -> float:
    return float(row["input_tokens"]) + float(row["output_tokens"])


def paired(rows: Sequence[dict[str, Any]], item: str) -> list[tuple[dict, dict]]:
    """The item's runs, each with the base run of the same window, cell and index."""
    runs = runs_of(rows)
    base = {_key(r): r for r in runs if r["arm"] == BASE}
    mine = [r for r in runs if r["arm"] == f"react/C+{item}"]
    return [(base[_key(r)], r) for r in mine if _key(r) in base]


def decide(rows: Sequence[dict[str, Any]], item: str) -> dict[str, Any]:
    """The item's comparison with the base, and whether it is kept (see the module)."""
    target = TARGETS[item]
    pairs = paired(rows, item)
    if not pairs:
        return {"item": item, "kept": False, "why": "not measured", "pairs": 0}
    success = stats.interval([float(m["success"]) - float(b["success"]) for b, m in pairs])
    base_tokens = sum(_tokens(b) for b, _ in pairs) / len(pairs)
    mine_tokens = sum(_tokens(m) for _, m in pairs) / len(pairs)
    tokens = (mine_tokens - base_tokens) / base_tokens if base_tokens else 0.0
    diffs = [
        mv - bv
        for b, m in pairs
        if (bv := _value(b, target.metric)) is not None
        and (mv := _value(m, target.metric)) is not None
    ]
    moved = stats.interval(diffs)
    mean, half = moved["mean"], moved["ci95"]
    if target.better is Better.LOWER:
        improved = mean is not None and half is not None and mean + half < 0
    else:
        improved = mean is not None and half is not None and mean - half > 0
    why = None
    if success["mean"] < -SUCCESS_MARGIN:
        why = f"task success fell {-success['mean']:.2f} (more than {SUCCESS_MARGIN})"
    elif tokens > TOKEN_MARGIN:
        why = f"tokens per run rose {tokens:.0%} (more than {TOKEN_MARGIN:.0%})"
    elif not improved:
        why = f"{target.metric} not {target.better.value}: {stats.shown(moved, 3)}"
    return {
        "item": item,
        "kept": why is None,
        "why": why or "kept",
        "pairs": len(pairs),
        "cells": sorted({b["cell"] for b, _ in pairs}),
        "windows": sorted({b.get("window") for b, _ in pairs}, key=str),
        "success_base": sum(float(b["success"]) for b, _ in pairs) / len(pairs),
        "success_item": sum(float(m["success"]) for _, m in pairs) / len(pairs),
        "success_diff": success,
        "tokens_base": base_tokens,
        "tokens_item": mine_tokens,
        "tokens_change": tokens,
        "metric": target.metric,
        "better": target.better.value,
        "metric_diff": moved,
        "cpu_base": _mean([b.get("cpu_seconds", {}).get("arm") for b, _ in pairs]),
        "cpu_item": _mean([m.get("cpu_seconds", {}).get("arm") for _, m in pairs]),
        "rss_base": _mean([b.get("peak_rss_mb") for b, _ in pairs]),
        "rss_item": _mean([m.get("peak_rss_mb") for _, m in pairs]),
    }


def _mean(values: Sequence[Any]) -> float | None:
    found = [float(v) for v in values if v is not None]
    return sum(found) / len(found) if found else None


def _f(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def table(decisions: Sequence[dict[str, Any]]) -> str:
    """The decisions as a markdown table: before (``C0``) and after (``C+<item>``)."""
    lines = [
        "| item | runs | cells | success C0 → item | tokens C0 → item | target metric "
        "(item - C0, 95% CI) | CPU s C0 → item | peak RSS MB C0 → item | decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in decisions:
        if not d["pairs"]:
            lines.append(f"| {d['item']} | 0 | — | — | — | — | — | — | dropped: {d['why']} |")
            continue
        lines.append(
            f"| {d['item']} | {d['pairs']} | {', '.join(d['cells'])} | "
            f"{d['success_base']:.2f} → {d['success_item']:.2f} | "
            f"{d['tokens_base']:,.0f} → {d['tokens_item']:,.0f} ({d['tokens_change']:+.1%}) | "
            f"{d['metric']} ({d['better']} is better): {stats.shown(d['metric_diff'], 3)} | "
            f"{_f(d['cpu_base'])} → {_f(d['cpu_item'])} | {_f(d['rss_base'], 0)} → "
            f"{_f(d['rss_item'], 0)} | {'**kept**' if d['kept'] else 'dropped: ' + d['why']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.cost.frameworks.items")
    parser.add_argument("rows", nargs="+", help="rows.jsonl files")
    parser.add_argument("--json", default=None, help="write the decisions here too")
    args = parser.parse_args(argv)
    rows = [r for path in args.rows for r in read_rows(Path(path))]
    decisions = [decide(rows, item) for item in ADAPTIVE]
    if args.json:
        Path(args.json).write_text(json.dumps(decisions, indent=2, default=str))
    print(table(decisions), end="")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
