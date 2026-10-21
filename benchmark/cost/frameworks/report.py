"""The head-to-head's report from its rows: per arm (and per cell), the cost benchmark's
statistics — means with 95% intervals (Student's t over the runs) — a ranking per task type,
the load check, the kill survival, what was skipped and why, and the verdict.

**The verdict** says "ReAct best" only when ``react/C`` wins on accuracy — its task success at
least every other arm's, in every task type — and is no worse on cost: its tokens per run no
more than the most accurate other arm's (or within that arm's 95% interval). Otherwise it
names the winning arm (task success first, then fewer tokens) and the feature that won it, as a
candidate to reuse in ReAct.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from benchmark import procstat
from benchmark.cost import stats
from benchmark.cost.frameworks.arms import ARMS
from benchmark.cost.frameworks.runner import CELLS

#: The arm the verdict is about.
OURS: Final = "react/C"
#: What a task type is called in the report.
TYPES: Final = {
    "short": "short (10 tools)",
    "long": "long (10 tools)",
    "many": "many tools (100)",
    "many50": "many tools (50)",
}


def title(cell: str) -> str:
    """What a task type is called in the report."""
    return TYPES.get(cell, cell)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row a rows file holds (a later row of the same run replaces an earlier one)."""
    if not path.exists():
        return []
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for n, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "cell" in row:
            key: tuple[Any, ...] = ("run", row["arm"], row["cell"], row["index"])
        elif row.get("load"):
            key = ("load", row["arm"], row["concurrency"])
        elif row.get("survive"):
            key = ("survive", row["arm"], row["trial"])
        else:
            key = ("other", n)
        found[key] = row
    return list(found.values())


def runs_of(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if "cell" in r]


def tokens(row: dict[str, Any]) -> float:
    return float(row["input_tokens"]) + float(row["output_tokens"])


def cell_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One arm's (or one cell's) statistics over its runs."""

    def of(key: str) -> dict[str, Any]:
        return stats.interval([float(r[key]) for r in rows if r.get(key) is not None])

    seconds = [float(r["seconds"]) for r in rows]
    return {
        "runs": len(rows),
        "success": of("success"),
        "tool_coverage": of("tool_coverage"),
        "input_tokens": of("input_tokens"),
        "output_tokens": of("output_tokens"),
        "tokens": stats.interval([tokens(r) for r in rows]),
        "model_calls": of("model_calls"),
        "errored": of("errored"),
        "p50": procstat.percentile(seconds, 50),
        "p95": procstat.percentile(seconds, 95),
        "cpu_seconds": stats.interval(
            [
                r["cpu_seconds"]["arm"]
                for r in rows
                if r.get("cpu_seconds", {}).get("arm") is not None
            ]
        ),
        "peak_rss_mb": max((r["peak_rss_mb"] for r in rows if r.get("peak_rss_mb")), default=None),
        "mean_peak_rss_mb": stats.interval(
            [r["peak_rss_mb"] for r in rows if r.get("peak_rss_mb")]
        ),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per arm, over all its runs and per cell."""
    runs = runs_of(rows)
    arms = list(dict.fromkeys(r["arm"] for r in runs))
    found: dict[str, Any] = {}
    for arm in arms:
        mine = [r for r in runs if r["arm"] == arm]
        cells = list(dict.fromkeys(r["cell"] for r in mine))
        found[arm] = {
            "all": cell_stats(mine),
            "cells": {c: cell_stats([r for r in mine if r["cell"] == c]) for c in cells},
        }
    return found


def mean(cell: dict[str, Any] | None) -> float:
    if not cell or cell.get("mean") is None:
        return float("nan")
    return float(cell["mean"])


def ranking(summary: dict[str, Any], cell: str) -> list[tuple[str, dict[str, Any]]]:
    """The arms measured on ``cell``, best first: task success, then tool coverage, then
    fewer tokens."""
    measured = [(arm, s["cells"][cell]) for arm, s in summary.items() if cell in s["cells"]]
    return sorted(
        measured,
        key=lambda pair: (
            -mean(pair[1]["success"]),
            -mean(pair[1]["tool_coverage"]),
            mean(pair[1]["tokens"]),
        ),
    )


def no_worse(ours: dict[str, Any], theirs: dict[str, Any]) -> bool:
    """Whether our tokens per run are no more than theirs, or within their 95% interval."""
    mine, other = mean(ours["tokens"]), mean(theirs["tokens"])
    half = theirs["tokens"].get("ci95") or 0.0
    return mine <= other + half


#: What each arm's winning part is, said as a candidate for ReAct.
FEATURES: Final = {
    "summarization": "LangChain's SummarizationMiddleware",
    "planning": "Deep Agents' write_todos planning (TodoListMiddleware)",
    "session": "the OpenAI Agents SDK's session",
}


def feature_of(arm: str) -> str:
    """The part of an arm that would be reused: its signature feature, else what the
    framework does by itself (its loop and prompt), else the harness's (for a B arm)."""
    known = ARMS.get(arm)
    if known is None:
        return "its loop"
    if known.kind == "A":
        if known.feature:
            return FEATURES.get(known.feature, known.feature)
        return f"{known.framework}'s own agent loop and prompt"
    if known.kind == "B":
        return "the harness around it (memory, tool narrowing)"
    return f"ReAct {'without ' + known.feature if known.feature else 'as built'}"


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """ReAct best, or the winning arm and its feature (see the module's docstring)."""
    others = {a: s for a, s in summary.items() if not a.startswith("react/")}
    if OURS not in summary or not others:
        return {"react_best": False, "text": "Not decided: ReAct or every other arm unmeasured."}
    cells = [c for c in summary[OURS]["cells"] if any(c in s["cells"] for s in others.values())]
    lost = []
    for cell in cells:
        ours = mean(summary[OURS]["cells"][cell]["success"])
        best = max(mean(s["cells"][cell]["success"]) for s in others.values() if cell in s["cells"])
        if ours < best:
            lost.append(cell)
    overall = sorted(
        others.items(),
        key=lambda pair: (-mean(pair[1]["all"]["success"]), mean(pair[1]["all"]["tokens"])),
    )
    rival, theirs = overall[0]
    cheap = no_worse(summary[OURS]["all"], theirs["all"])
    ours_all = summary[OURS]["all"]
    if not lost and cheap:
        text = (
            f"**ReAct best.** react/C's task success is at least every other arm's in every task "
            f"type ({', '.join(title(c) for c in cells)}), and its tokens per run "
            f"({mean(ours_all['tokens']):,.0f}) are no more than the most accurate other arm's "
            f"({rival}: {mean(theirs['all']['tokens']):,.0f})."
        )
        return {"react_best": True, "text": text, "rival": rival}
    winners = []
    for cell in cells:
        top = ranking(summary, cell)[0]
        winners.append(f"{title(cell)}: {top[0]} ({mean(top[1]['success']):.2f})")
    why = []
    if lost:
        why.append(f"it is out-scored on task success in {', '.join(title(c) for c in lost)}")
    if not cheap:
        why.append(
            f"its tokens per run ({mean(ours_all['tokens']):,.0f}) are more than {rival}'s "
            f"({stats.shown(theirs['all']['tokens'])})"
        )
    said = "; and ".join(why)
    text = (
        f"**Not ReAct.** {said[:1].upper() + said[1:]}. The winning framework overall is "
        f"**{ARMS[rival].framework if rival in ARMS else rival}** ({rival}: task success "
        f"{mean(theirs['all']['success']):.2f}, {mean(theirs['all']['tokens']):,.0f} tokens per "
        f"run, against ReAct's {mean(ours_all['success']):.2f} and "
        f"{mean(ours_all['tokens']):,.0f}); the feature to consider reusing in ReAct: "
        f"**{feature_of(rival)}**. Best per task type — {'; '.join(winners)}."
    )
    return {"react_best": False, "text": text, "rival": rival, "lost": lost}


# --------------------------------------------------------------------------- markdown
def _m(cell: dict[str, Any] | None, digits: int = 0) -> str:
    return stats.shown(cell, digits)


def _s(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


HEADER: Final = (
    "| arm | runs | task success | tool coverage | input tokens | output tokens | model calls "
    "| p50 s | p95 s | CPU s | peak RSS MB |"
)


def _line(name: str, s: dict[str, Any]) -> str:
    return (
        f"| {name} | {s['runs']} | {_m(s['success'], 2)} | {_m(s['tool_coverage'], 2)} | "
        f"{_m(s['input_tokens'])} | {_m(s['output_tokens'])} | {_m(s['model_calls'], 2)} | "
        f"{_s(s['p50'])} | {_s(s['p95'])} | {_m(s['cpu_seconds'], 1)} | "
        f"{_s(s['peak_rss_mb'], 0)} |"
    )


def _order(arm: str) -> int:
    names = list(ARMS)
    return names.index(arm) if arm in names else len(names)


def markdown(rows: Sequence[dict[str, Any]], meta: dict[str, Any]) -> str:
    """The report as markdown (``docs/eval/frameworks.md``)."""
    summary = summarize(rows)
    order = [a for a in ARMS if a in summary] + [a for a in summary if a not in ARMS]
    lines = [
        "# Framework head-to-head",
        "",
        meta.get("intro", ""),
        "",
        "Each figure is the mean per run ± the half-width of its 95% interval (Student's t over "
        "the runs); p50/p95 are of the runs' wall seconds; CPU s is the arm's process and its "
        "children per run; peak RSS is the largest per-run peak of the arm's process.",
        "",
        "## All task types",
        "",
        HEADER,
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(_line(a, summary[a]["all"]) for a in order),
    ]
    cells = [c for c in CELLS if any(c in s["cells"] for s in summary.values())]
    for cell in cells:
        named = title(cell)
        lines += ["", f"## Ranking: {named}", "", "| # " + HEADER, "|" + "---|" * 12]
        for n, (arm, s) in enumerate(ranking(summary, cell), 1):
            lines.append(f"| {n} " + _line(arm, s))
    loads = [r for r in rows if r.get("load")]
    if loads:
        lines += [
            "",
            "## Load check",
            "",
            "One short task, `runs` runs `concurrency` at a time (the model server has one slot: "
            "a run's latency at concurrency 10 includes its wait for the model).",
            "",
            "| arm | concurrency | runs | succeeded | errors | p50 s | p95 s | wall s | runs/min "
            "| model calls | CPU s | peak RSS MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(loads, key=lambda r: (_order(r["arm"]), r["concurrency"])):
            lines.append(
                f"| {r['arm']} | {r['concurrency']} | {r['runs']} | {r['successes']} | "
                f"{r['errors']} | {_s(r['p50'])} | {_s(r['p95'])} | {_s(r['wall_seconds'])} | "
                f"{_s(r['throughput_per_min'], 2)} | {r['model_calls']} | {_s(r['cpu_seconds'])} "
                f"| {_s(r['peak_rss_mb'], 0)} |"
            )
    trials = [r for r in rows if r.get("survive")]
    if trials:
        lines += [
            "",
            "## Kill mid-run",
            "",
            "A run queued in agent-runs; the worker that claimed it killed (SIGKILL) once the "
            "run's first model call answered; a second worker started. Survived: the run ended "
            "(done, or paused at its approval) on a later attempt.",
            "",
            "| arm | trials | survived | attempts (mean) | seconds to the end (mean) | errors |",
            "|---|---|---|---|---|---|",
        ]
        for arm in dict.fromkeys(r["arm"] for r in trials):
            mine = [r for r in trials if r["arm"] == arm]
            errors = sorted({r["error"] for r in mine if r.get("error")})
            lines.append(
                f"| {arm} | {len(mine)} | {sum(r['survived'] for r in mine)} | "
                f"{sum(r['attempts'] for r in mine) / len(mine):.1f} | "
                f"{sum(r['seconds'] for r in mine) / len(mine):.0f} | "
                f"{'; '.join(errors)[:200] or '—'} |"
            )
    skipped = [r for r in rows if r.get("skipped")]
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"* **{r['arm']}** — {r['skipped']}" for r in skipped]
    decided = verdict(summary)
    lines += ["", "## Verdict", "", decided["text"]]
    for note in meta.get("notes", []):
        lines += ["", note]
    return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"
