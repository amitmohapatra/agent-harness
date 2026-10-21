"""The report's statistics: a mean and its 95% confidence interval (Student's t), and the
report written as JSON and as a markdown table."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any, Final

#: Student's t at 97.5% for 1..30 degrees of freedom (a two-sided 95% interval).
T975: Final = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t975(df: int) -> float:
    """Student's t at 97.5% for ``df`` degrees of freedom (the normal's 1.96 in the limit)."""
    if df <= len(T975):
        return T975[df - 1]
    return 1.96 + 2.4 / df  # within 0.002 of the exact value from 31 on


def interval(values: Sequence[float]) -> dict[str, Any]:
    """The mean of ``values``, the half-width of its 95% interval (``None`` from fewer than
    two values) and how many there were."""
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "ci95": None, "n": n}
    half = t975(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return {"mean": mean, "ci95": half, "n": n}


def shown(cell: dict[str, Any] | None, digits: int = 0, *, scale: float = 1.0) -> str:
    """A mean and its interval as ``mean ± half``; ``—`` when nothing was measured."""
    if not cell or cell.get("mean") is None:
        return "—"
    mean = cell["mean"] * scale
    half = cell.get("ci95")
    text = f"{mean:,.{digits}f}"
    return text if half is None else f"{text} ± {half * scale:,.{digits}f}"


COLUMNS: Final = (
    ("input tokens", "input_tokens", 0, 1.0),
    ("cached", "cached_tokens", 0, 1.0),
    ("output tokens", "output_tokens", 0, 1.0),
    ("model calls", "model_calls", 2, 1.0),
    ("refused prompt tokens", "refused_prompt_tokens", 0, 1.0),
    ("seconds", "seconds", 1, 1.0),
    ("tool coverage", "tool_coverage", 2, 1.0),
    ("success", "success", 2, 1.0),
    ("errors", "errored", 2, 1.0),
)


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def latency(report: dict[str, Any]) -> list[str]:
    """Each cell's model call seconds (p50, p95, over every call of its runs) and CPU seconds
    per run of each measured process, then each process's peak memory."""
    names = list(
        dict.fromkeys(
            key.split(":", 1)[1]
            for cell in report["cells"]
            for key in cell["stats"]
            if key.startswith("cpu_seconds:")
        )
    )
    lines = [
        "",
        "Model call seconds (p50 and p95 over every call of the cell's runs) and CPU seconds "
        "per run of each process:",
        "",
        "| tools | variant | calls | p50 s | p95 s | "
        + " | ".join(f"CPU s ({n})" for n in names)
        + (" |" if names else "|"),
        "|---|---|---|---|---|" + "---|" * len(names),
    ]
    for cell in report["cells"]:
        calls = cell["stats"].get("model_call_seconds") or {}
        lines.append(
            f"| {cell['size']} | {cell['variant']} | {calls.get('n', 0)} | "
            f"{_seconds(calls.get('p50'))} | {_seconds(calls.get('p95'))} | "
            + " | ".join(shown(cell["stats"].get(f"cpu_seconds:{n}"), 2) for n in names)
            + (" |" if names else "")
        )
    peaks = report.get("processes") or {}
    if peaks:
        lines += ["", "Peak resident memory at the end of the run (MB):", ""]
        for name, p in peaks.items():
            peak = p.get("peak_rss_mb")
            lines.append(f"* {name}: {'—' if peak is None else f'{peak:,.0f}'}")
    return lines


def markdown(report: dict[str, Any]) -> str:
    """The report as markdown: per catalogue size and variant, the per-run means with their
    95% intervals, then the dollars per run at each price."""
    lines = [
        f"# Cost benchmark: {report['model']}",
        "",
        f"Up to {report['repeats']} repeats of {report['tasks']} tasks per cell (`runs`); each "
        "figure is the mean per run ± the half-width of its 95% interval (Student's t over "
        "the runs). "
        f"Measured {report['date']} on {report['machine']}.",
        "",
        "| tools | variant | runs | " + " | ".join(c[0] for c in COLUMNS) + " | $ per run |",
        "|---|---|---|" + "---|" * (len(COLUMNS) + 1),
    ]
    for cell in report["cells"]:
        stats = cell["stats"]
        priced = cell["dollars"].get(report["model_price"] or "", None)
        lines.append(
            f"| {cell['size']} | {cell['variant']} | {cell.get('runs', '—')} | "
            + " | ".join(
                shown(stats.get(key), digits, scale=scale) for _, key, digits, scale in COLUMNS
            )
            + f" | {shown(priced, 5) if priced else 'not priced'} |"
        )
    lines += [
        "",
        "By task length (short: two parts or fewer; long: four and five parts; large: the "
        "~100k-character export):",
        "",
        "| tools | variant | length | input tokens | model calls | seconds | tool coverage | "
        "success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in report["cells"]:
        for length, part in cell.get("by_length", {}).items():
            got = part["stats"]
            lines.append(
                f"| {cell['size']} | {cell['variant']} | {length} | "
                f"{shown(got.get('input_tokens'))} | {shown(got.get('model_calls'), 2)} | "
                f"{shown(got.get('seconds'), 1)} | {shown(got.get('tool_coverage'), 2)} | "
                f"{shown(got.get('success'), 2)} |"
            )
    lines += latency(report)
    others = [m for m in report["priced_as"] if m != report["model_price"]]
    if others:
        lines += [
            "",
            "Priced as other models (an estimate: the same token counts at their rates, every "
            "input token at the full input price — the measured model's own cache is not "
            "theirs — and their tokenizers count the same text differently):",
            "",
            "| tools | variant | " + " | ".join(f"$ per run as {m}" for m in others) + " |",
            "|---|---|" + "---|" * len(others),
        ]
        for cell in report["cells"]:
            lines.append(
                f"| {cell['size']} | {cell['variant']} | "
                + " | ".join(shown(cell["dollars"].get(m), 5) for m in others)
                + " |"
            )
    lines += ["", "Prices (per million tokens: input / cached input / cache write / output):", ""]
    for name, price in report["prices"].items():
        write = "—" if price["cache_write"] is None else f"{price['cache_write']}"
        lines.append(
            f"* `{name}`: {price['input']} / {price['cached_input']} / {write} / "
            f"{price['output']} — {price['source']}"
        )
    for note in report.get("notes", []):
        lines += ["", note]
    return "\n".join(lines) + "\n"
