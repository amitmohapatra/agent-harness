"""The verdict on a change of the ReAct loop (``python -m benchmark.cost.analyze``): its arm
against the arm it changes, per task class, every metric's difference with its 95% bootstrap
interval, and the rules applied — nothing decided by hand:

    python -m benchmark.cost.analyze \\
        --arm main=build/main.json --arm limitsA=build/branch.json \\
        --compare limitsA:main:error,memory --latency main=build/serial-main.json \\
        --latency limitsA=build/serial-branch.json --out build/verdict

An arm is ``label=report.json`` (its runs of the variant ``label``) or
``label=report.json:variant``; ``--compare candidate:base:targets`` names the classes the
change is for. Runs are paired by task: each task's mean over its repeats in either arm, the
tasks resampled (:data:`RESAMPLES`, seeded) for the interval of the mean difference — of a
count or a rate, in its unit; of tokens, seconds and dollars, relative to the base. Latency
(p50, p95 of a run's seconds) comes from the serial pass (``--latency``), else from the runs
themselves, marked so (concurrent runs' seconds are not latency).

**The rules** (pre-registered, :func:`verdict`):

1. *Keep rule* — on a target class: success up at least 3 points, or tokens down at least
   10%, or p50 down at least 10%, the whole interval past that; on every other class, the
   success interval's lower end at least -2 points; on the simple class, model calls, tokens
   and p50 not measurably up; the harness's CPU seconds per run (p95) and its peak memory
   not up more than 5%.
2. *Better and never worse* — measurably better (the interval clear of 0 on the good side)
   on at least one metric of one class, and measurably worse on none, on every class.

A change both rules pass is ``KEEP``; one measurably worse anywhere, or past the intervals but
short of the keep rule, ``REVERT``; one with no metric of any class clear of 0 either way a
``TIE`` — more repeats of the classes listed, and if they still tie, a tie reported with its
numbers, never broken by a rule.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from benchmark import procstat
from benchmark.cost import prices

#: How many times the tasks are resampled for an interval, and the seed (reproducible).
RESAMPLES: Final = 2000
SEED: Final = 11
#: The keep rule's thresholds (rule 1).
SUCCESS_GAIN: Final = 0.03
TOKENS_CUT: Final = -0.10
P50_CUT: Final = -0.10
SUCCESS_FLOOR: Final = -0.02
OVERHEAD_FLAT: Final = 0.05


@dataclass(frozen=True)
class Metric:
    """A per-run figure: its key in a run's row (or how it is made of one), whether higher is
    better (``None``: reported, never judged), and whether its difference is relative to the
    base (tokens, seconds, dollars) or in its own unit (rates, counts)."""

    name: str
    of: Callable[[dict[str, Any]], float | None]
    higher_better: bool | None
    relative: bool


def _key(name: str) -> Callable[[dict[str, Any]], float | None]:
    def of(row: dict[str, Any]) -> float | None:
        value = row.get(name)
        return None if value is None else float(value)

    return of


def _cached_share(row: dict[str, Any]) -> float | None:
    cached, sent = row.get("cached_tokens"), row.get("input_tokens")
    return None if cached is None or not sent else float(cached) / float(sent)


def _recall(row: dict[str, Any]) -> float | None:
    """Of a run whose task left a value out: whether it asked."""
    return float(row["asked"]) if row.get("vague") is True else None


def _precision(row: dict[str, Any]) -> float | None:
    """Of a run whose task left nothing out (an asking class's): whether it did not ask."""
    return float(not row["asked"]) if row.get("vague") is False else None


def _targeted(row: dict[str, Any]) -> float | None:
    return None if row.get("targeted") is None else float(row["targeted"])


def _correct(row: dict[str, Any]) -> float | None:
    """Of an ambiguous task's run: whether its change was made with every gold argument."""
    return None if row.get("correct_change") is None else float(row["correct_change"])


def _pauses(row: dict[str, Any]) -> float | None:
    """The times a run waited for its person: its questions and its approvals."""
    if row.get("questions") is None:
        return None
    return float(row["questions"]) + float(row.get("approvals") or 0)


def _tokens(row: dict[str, Any]) -> float:
    return float(row.get("input_tokens") or 0) + float(row.get("output_tokens") or 0)


METRICS: Final = (
    Metric("success", _key("success"), True, relative=False),
    Metric("tool_coverage", _key("tool_coverage"), True, relative=False),
    Metric("wrong_calls", _key("wrong_calls"), False, relative=False),
    Metric("errored", _key("errored"), False, relative=False),
    Metric("timed_out", _key("timed_out"), False, relative=False),
    Metric("model_calls", _key("model_calls"), False, relative=False),
    Metric("failed_model_calls", _key("failed_model_calls"), False, relative=False),
    Metric("tokens", _tokens, False, relative=True),
    Metric("input_tokens", _key("input_tokens"), False, relative=True),
    Metric("output_tokens", _key("output_tokens"), False, relative=True),
    Metric("cached_tokens", _key("cached_tokens"), None, relative=True),
    Metric("cached_share", _cached_share, None, relative=False),
    Metric("ask_recall", _recall, True, relative=False),
    Metric("ask_precision", _precision, True, relative=False),
    Metric("ask_targeted", _targeted, True, relative=False),
    Metric("questions", _key("questions"), None, relative=False),
    Metric("correct_change", _correct, True, relative=False),
    Metric("pauses", _pauses, None, relative=False),
    Metric("edits", _key("edits"), None, relative=False),
)
#: The latency metrics, each a percentile of a class's run seconds.
LATENCY: Final = (("p50", 50), ("p95", 95))


@dataclass(frozen=True)
class Delta:
    """A metric's base and candidate means, their difference (relative to the base where the
    metric says) and its 95% interval, and which way it is clear of 0, if at all."""

    metric: str
    base: float | None
    candidate: float | None
    delta: float | None
    low: float | None
    high: float | None
    tasks: int
    better: bool = False
    worse: bool = False
    note: str | None = None


@dataclass
class ClassResult:
    name: str
    target: bool
    deltas: dict[str, Delta] = field(default_factory=dict)


@dataclass
class Verdict:
    candidate: str
    base: str
    targets: list[str]
    decision: str  # KEEP | REVERT | TIE
    reasons: list[str]
    classes: list[ClassResult]
    overhead: dict[str, Any]
    more_repeats: list[str]


# --------------------------------------------------------------------------- the arms
def runs_of(report: Mapping[str, Any], variant: str, size: int | None) -> list[dict[str, Any]]:
    """The runs of ``variant`` a report holds (of one catalogue size: ``size``, else its
    smallest), less the reasoning class's ``dev`` runs."""
    # a dev run tunes a technique: it is never measured
    rows = [r for r in report["runs"] if r["variant"] == variant and r.get("split") != "dev"]
    if not rows:
        raise SystemExit(f"no runs of {variant!r} in the report")
    chosen = size if size is not None else min(r["size"] for r in rows)
    return [r for r in rows if r["size"] == chosen]


def priced(rows: list[dict[str, Any]], model: str | None) -> None:
    """Each run's dollars at ``model``'s list price (``cost``), every input token in full."""
    price = prices.PRICES.get(model or "")
    for row in rows:
        row["cost"] = (
            None
            if price is None
            else prices.dollars(
                price,
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cached=0,
                written=0,
            )
        )


# --------------------------------------------------------------------------- the intervals
def _per_task(
    rows: Sequence[dict[str, Any]], of: Callable[[dict[str, Any]], float | None]
) -> dict[str, float]:
    found: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = of(row)
        if value is not None:
            found[row["task"]].append(value)
    return {task: statistics.fmean(values) for task, values in found.items()}


def paired(
    base: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    metric: Metric,
    rng: random.Random,
) -> Delta:
    """``metric``'s difference between the arms over the tasks both ran, and its interval."""
    b, c = _per_task(base, metric.of), _per_task(candidate, metric.of)
    tasks = sorted(b.keys() & c.keys())
    if not tasks:
        return Delta(metric.name, None, None, None, None, None, 0, note="no task in both")
    mb, mc = statistics.fmean(b[t] for t in tasks), statistics.fmean(c[t] for t in tasks)

    def diff(sample: Sequence[str]) -> float | None:
        sb, sc = sum(b[t] for t in sample), sum(c[t] for t in sample)
        if metric.relative:
            return None if sb == 0 else sc / sb - 1
        return (sc - sb) / len(sample)

    point = diff(tasks)
    drawn = [
        d for _ in range(RESAMPLES) if (d := diff(rng.choices(tasks, k=len(tasks)))) is not None
    ]
    return _judged(
        metric.name,
        metric.higher_better,
        base=mb,
        candidate=mc,
        point=point,
        drawn=drawn,
        tasks=len(tasks),
    )


def latency(
    base: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    rng: random.Random,
    note: str | None,
) -> dict[str, Delta]:
    """p50 and p95 of a run's seconds, the candidate's relative to the base's, the tasks of
    either arm resampled."""
    by_b, by_c = defaultdict(list), defaultdict(list)
    for row in base:
        by_b[row["task"]].append(float(row["seconds"]))
    for row in candidate:
        by_c[row["task"]].append(float(row["seconds"]))
    tasks = sorted(by_b.keys() & by_c.keys())
    found = {}
    for name, q in LATENCY:

        def at(sample: Sequence[str], by: Mapping[str, list[float]], q: int = q) -> float | None:
            return procstat.percentile([s for t in sample for s in by[t]], q)

        def diff(sample: Sequence[str], q: int = q) -> float | None:
            pb, pc = at(sample, by_b, q), at(sample, by_c, q)
            return None if not pb or pc is None else pc / pb - 1

        if not tasks:
            found[name] = Delta(name, None, None, None, None, None, 0, note="no task in both")
            continue
        drawn = [
            d for _ in range(RESAMPLES) if (d := diff(rng.choices(tasks, k=len(tasks)))) is not None
        ]
        found[name] = _judged(
            name,
            False,
            base=at(tasks, by_b),
            candidate=at(tasks, by_c),
            point=diff(tasks),
            drawn=drawn,
            tasks=len(tasks),
            note=note,
        )
    return found


def _judged(
    name: str,
    higher_better: bool | None,
    *,
    base: float | None,
    candidate: float | None,
    point: float | None,
    drawn: Sequence[float],
    tasks: int,
    note: str | None = None,
) -> Delta:
    if point is None or len(drawn) < 2:
        return Delta(name, base, candidate, point, None, None, tasks, note=note)
    ordered = sorted(drawn)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    up, down = low > 0, high < 0
    better = (up if higher_better else down) if higher_better is not None else False
    worse = (down if higher_better else up) if higher_better is not None else False
    return Delta(name, base, candidate, point, low, high, tasks, better, worse, note)


# --------------------------------------------------------------------------- the rules
def overhead(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    b: list[dict[str, Any]],
    c: list[dict[str, Any]],
) -> dict[str, Any]:
    """The harness's CPU seconds per run (p95) and its peak memory, the candidate's relative
    to the base's, and whether each stayed flat (up at most :data:`OVERHEAD_FLAT`)."""

    def cpu(rows: list[dict[str, Any]]) -> float | None:
        used = [r["cpu_seconds"].get("harness") for r in rows if r.get("cpu_seconds")]
        return procstat.percentile([u for u in used if u is not None], 95)

    def peak(report: Mapping[str, Any]) -> float | None:
        return ((report.get("processes") or {}).get("harness") or {}).get("peak_rss_mb")

    found: dict[str, Any] = {}
    for name, was, now in (
        ("cpu_p95", cpu(b), cpu(c)),
        ("peak_rss_mb", peak(base), peak(candidate)),
    ):
        change = None if not was or now is None else now / was - 1
        found[name] = {
            "base": was,
            "candidate": now,
            "change": change,
            "flat": change is None or change <= OVERHEAD_FLAT,
        }
    return found


def verdict(
    candidate: str,
    base: str,
    targets: Sequence[str],
    classes: list[ClassResult],
    flat: Mapping[str, Any],
) -> Verdict:
    """Both rules applied (the module's doc). Each part of the keep rule is met (its whole
    interval past the threshold), failed (its whole interval short of it) or unresolved (the
    interval across it: more repeats decide). Measurably worse on any metric (the harness's
    overhead up past :data:`OVERHEAD_FLAT` among them): ``REVERT``; nothing clear of 0 either
    way, anywhere: ``TIE`` (never broken by the keep rule); a part failed: ``REVERT``; a part
    unresolved: ``TIE`` (the classes to repeat listed); else ``KEEP``."""
    failed, open_ = _keep_rule(classes)
    worse = [f"{r.name}: {d.metric} worse" for r in classes for d in r.deltas.values() if d.worse]
    worse += [f"harness {name} up {f['change']:+.1%}" for name, f in flat.items() if not f["flat"]]
    better = [
        f"{r.name}: {d.metric} better" for r in classes for d in r.deltas.values() if d.better
    ]
    if worse:
        decision, reasons = "REVERT", worse
    elif not better:
        # nothing clear of 0 either way, anywhere: a tie, never broken by the keep rule
        decision = "TIE"
        reasons = ["no metric of any class is clear of 0", *failed]
        open_ = {r.name: "" for r in classes}
    elif failed:
        decision, reasons = "REVERT", failed
    elif open_:
        decision = "TIE"
        reasons = [f"{name}: {why}" for name, why in open_.items()]
    else:
        decision, reasons = "KEEP", better
    return Verdict(
        candidate, base, list(targets), decision, reasons, classes, dict(flat), list(open_)
    )


def _keep_rule(classes: Sequence[ClassResult]) -> tuple[list[str], dict[str, str]]:
    """The keep rule's parts that failed, and the classes whose part is still open (why); the
    harness's overhead is judged as a metric (:func:`verdict`)."""
    failed: list[str] = []
    open_: dict[str, str] = {}
    for result in classes:
        d = result.deltas
        if result.target:
            parts = [
                _part(d.get("success"), SUCCESS_GAIN, up=True),
                _part(d.get("tokens"), TOKENS_CUT, up=False),
                _part(d.get("p50"), P50_CUT, up=False),
            ]
            if "met" not in parts:
                if all(p == "failed" for p in parts):
                    failed.append(f"{result.name} (target): no gain reaches the keep rule")
                else:
                    open_[result.name] = "its gain may reach the keep rule"
        else:
            part = _part(d.get("success"), SUCCESS_FLOOR, up=True)
            if part == "failed":
                failed.append(f"{result.name}: success below {SUCCESS_FLOOR:+.0%}")
            elif part == "open":
                open_[result.name] = f"its success interval reaches below {SUCCESS_FLOOR:+.0%}"
    # the simple class's calls, tokens and p50 not up: measurably up is worse (rule 2)
    return failed, open_


def _part(delta: Delta | None, threshold: float, *, up: bool) -> str:
    """Whether ``delta``'s interval is past ``threshold`` (``met``: above it, ``up``; else
    below it), short of it (``failed``), or across it (``open``)."""
    if delta is None or delta.low is None or delta.high is None:
        return "open"
    low, high = (delta.low, delta.high) if up else (-delta.high, -delta.low)
    edge = threshold if up else -threshold
    if low >= edge:
        return "met"
    return "failed" if high < edge else "open"


def compare(
    candidate: str,
    base: str,
    targets: Sequence[str],
    arms: Mapping[str, tuple[Mapping[str, Any], list[dict[str, Any]]]],
    timed: Mapping[str, list[dict[str, Any]]],
) -> Verdict:
    """The verdict on ``candidate`` against ``base``, class by class."""
    rng = random.Random(SEED)
    (report_b, rows_b), (report_c, rows_c) = arms[base], arms[candidate]
    names = sorted({r["length"] for r in rows_b} & {r["length"] for r in rows_c})
    cost = Metric("cost", _key("cost"), False, relative=True)
    results = []
    for name in names:
        b = [r for r in rows_b if r["length"] == name]
        c = [r for r in rows_c if r["length"] == name]
        result = ClassResult(name, name in targets)
        for metric in (*METRICS, cost):
            result.deltas[metric.name] = paired(b, c, metric, rng)
        serial = base in timed and candidate in timed
        lb = [r for r in timed[base] if r["length"] == name] if serial else b
        lc = [r for r in timed[candidate] if r["length"] == name] if serial else c
        note = None if serial else "from concurrent runs: not latency"
        result.deltas.update(latency(lb, lc, rng, note))
        results.append(result)
    return verdict(candidate, base, targets, results, overhead(report_b, report_c, rows_b, rows_c))


# --------------------------------------------------------------------------- the report
def _shown(value: float | None, relative: bool) -> str:
    if value is None:
        return "—"
    return f"{value:+.1%}" if relative else f"{value:+.3f}"


def _plain(value: float | None) -> str:
    return "—" if value is None else f"{value:,.3f}"


RELATIVE: Final = {m.name for m in METRICS if m.relative} | {"cost", "p50", "p95"}


def markdown(verdicts: Sequence[Verdict]) -> str:
    lines = ["# ReAct loop: verdicts", ""]
    for v in verdicts:
        lines += [
            f"## {v.candidate} against {v.base}: **{v.decision}**",
            "",
            f"Target classes: {', '.join(v.targets) or '—'}. "
            f"Intervals: 95%, {RESAMPLES} resamples of the tasks.",
            "",
            *(f"* {reason}" for reason in v.reasons),
            "",
        ]
        if v.decision == "TIE":
            lines += [f"More repeats of: {', '.join(v.more_repeats)}.", ""]
        for result in v.classes:
            lines += [
                f"### {result.name}{' (target)' if result.target else ''}",
                "",
                "| metric | base | candidate | Δ | 95% interval | tasks | |",
                "|---|---|---|---|---|---|---|",
            ]
            for d in result.deltas.values():
                rel = d.metric in RELATIVE
                mark = "better" if d.better else "worse" if d.worse else ""
                lines.append(
                    f"| {d.metric} | {_plain(d.base)} | {_plain(d.candidate)} | "
                    f"{_shown(d.delta, rel)} | [{_shown(d.low, rel)}, {_shown(d.high, rel)}] | "
                    f"{d.tasks} | {mark}{' (' + d.note + ')' if d.note else ''} |"
                )
            lines.append("")
        lines += ["Harness overhead:", ""]
        for name, found in v.overhead.items():
            change = found["change"]
            lines.append(
                f"* {name}: {_plain(found['base'])} → {_plain(found['candidate'])} "
                f"({'—' if change is None else f'{change:+.1%}'}; "
                f"{'flat' if found['flat'] else 'up'})"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- the command
def _arm(text: str) -> tuple[str, Path, str]:
    label, _, rest = text.partition("=")
    path, _, variant = rest.partition(":")
    if not label or not path:
        raise argparse.ArgumentTypeError(f"an arm is label=report.json[:variant], not {text!r}")
    return label, Path(path), variant or label


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.cost.analyze", description=__doc__)
    parser.add_argument("--arm", action="append", type=_arm, required=True)
    parser.add_argument("--latency", action="append", type=_arm, default=[])
    parser.add_argument("--compare", action="append", required=True, help="candidate:base:targets")
    parser.add_argument("--size", type=int, default=None, help="the catalogue size compared")
    parser.add_argument("--price", default=prices.REFERENCE[0], help="the price runs are costed at")
    parser.add_argument("--out", default="build/react-verdict")
    return parser.parse_args(argv)


def load(given: Sequence[tuple[str, Path, str]], size: int | None, price: str) -> dict[str, Any]:
    found = {}
    for label, path, variant in given:
        report = json.loads(path.read_text())
        rows = runs_of(report, variant, size)
        priced(rows, price)
        found[label] = (report, rows)
    return found


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    arms = load(args.arm, args.size, args.price)
    timed = {label: rows for label, (_, rows) in load(args.latency, args.size, args.price).items()}
    verdicts = []
    for text in args.compare:
        candidate, base, targets = [*text.split(":"), ""][:3]
        for label in (candidate, base):
            if label not in arms:
                raise SystemExit(f"--compare names {label!r}: no --arm of that label")
        verdicts.append(compare(candidate, base, [t for t in targets.split(",") if t], arms, timed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
    text = markdown(verdicts)
    out.with_suffix(".md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
