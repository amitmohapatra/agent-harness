"""The load benchmark: ``make bench-load`` (``python -m benchmark.load``) — many runs at once on
one and on two worker processes, and the harness's memory over a long run.

    python -m benchmark.load                                   # Paced model: the harness alone
    BIFROST_URL=... BIFROST_VIRTUAL_KEY=... python -m benchmark.load --model provider/model

For each worker count (``--workers 1,2``) and each level of concurrency (``--levels
1,10,50,100``, split across the workers) it runs ``--rounds`` runs per lane and reports the
throughput (runs a second), the run latency's p50, p95 and p99, the errors, and each process's
CPU seconds and peak memory (the workers, and ``--watch name=pid``: the model server, the
gateway). A level stops the climb for its worker count when it hits a limit: errors past
``--max-errors`` of its runs, a p95 past ``--max-p95`` seconds, or the machine's available
memory under ``--min-free-mb`` before it starts — and the report says which.

``--leak-minutes`` then runs one worker at ``--leak-concurrency`` for that long, sampling its
resident memory, and reports the growth from the end of the warm-up (the first fifth) to the
end: a leak shows as memory that keeps rising with the runs done.

Writes ``<out>.json`` and ``<out>.md``. See ``docs/evaluation.md#load-benchmark``."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark import procstat
from benchmark.cost.__main__ import _list, watched
from benchmark.load import queued
from benchmark.load.run import Measured, Plan, process

#: Above this share of a level's runs in error, the level is a limit.
MAX_ERRORS = 0.05


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python -m benchmark.load", description=__doc__)
    parser.add_argument(
        "--model",
        default=env("TRELLIS_BENCH_MODEL"),
        help="a gateway model name (default: Paced, no model)",
    )
    parser.add_argument("--pace", type=float, default=0.2, help="Paced's seconds per call")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--levels", default="1,10,50,100")
    parser.add_argument("--workers", default="1,2")
    parser.add_argument("--rounds", type=int, default=3, help="runs per lane at each level")
    parser.add_argument("--max-errors", type=float, default=MAX_ERRORS)
    parser.add_argument("--max-p95", type=float, default=None, help="seconds")
    parser.add_argument("--min-free-mb", type=float, default=1500.0)
    parser.add_argument("--run-seconds", type=float, default=600.0)
    parser.add_argument("--leak-minutes", type=float, default=0.0)
    parser.add_argument("--leak-concurrency", type=int, default=10)
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--watch", action="append", default=[], help="name=pid")
    parser.add_argument(
        "--queued",
        type=float,
        default=0.0,
        metavar="MINUTES",
        help="the production path for this long instead of the levels: runs queued in agent-runs "
        "(RUNS_URL) and claimed by --workers worker processes, --leak-concurrency in flight",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="with --queued: the agents' memory parts off (the service only as the key registry)",
    )
    parser.add_argument("--out", default="build/bench-load")
    args = parser.parse_args(argv)
    try:
        args.watch = watched(args.watch)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args.levels = [int(n) for n in _list(args.levels)]
    args.workers = [int(n) for n in _list(args.workers)]
    if min(args.levels + args.workers) < 1:
        parser.error("levels and workers are numbers from 1")
    return args


def free_mb() -> float | None:
    """The machine's available memory, in MB (``/proc/meminfo``)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def split(total: int, parts: int) -> list[int]:
    """``total`` split as evenly as it goes over ``parts`` (none of them 0)."""
    parts = min(parts, total)
    return [total // parts + (i < total % parts) for i in range(parts)]


def in_processes(plans: list[Plan]) -> list[Measured]:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(plans), mp_context=context) as pool:
        return list(pool.map(process, plans))


def level(args: argparse.Namespace, workers: int, concurrency: int) -> dict[str, Any]:
    """``concurrency`` lanes over ``workers`` processes, ``--rounds`` runs each: measured."""
    base = Plan(
        concurrency=1,
        model=args.model,
        pace=args.pace,
        window=args.window,
        sample_seconds=args.sample_seconds,
        run_seconds=args.run_seconds,
    )
    plans = [
        replace(base, concurrency=c, runs=c * args.rounds) for c in split(concurrency, workers)
    ]
    before = {name: procstat.cpu_seconds(pid) for name, pid in args.watch.items()}
    began = time.monotonic()
    measured = in_processes(plans)
    wall = time.monotonic() - began
    latencies = [s for m in measured for s in m.latencies]
    errors = [e for m in measured for e in m.errors]
    runs = len(latencies) + len(errors)
    watched_cpu = {}
    for name, pid in args.watch.items():
        now, then = procstat.cpu_seconds(pid), before[name]
        watched_cpu[name] = None if now is None or then is None else round(now - then, 2)
    return {
        "workers": len(plans),
        "concurrency": concurrency,
        "runs": runs,
        "seconds": round(wall, 2),
        "throughput": round(len(latencies) / wall, 3) if wall else None,
        "p50": procstat.percentile(latencies, 50),
        "p95": procstat.percentile(latencies, 95),
        "p99": procstat.percentile(latencies, 99),
        "errors": len(errors),
        "error_examples": sorted(set(errors))[:3],
        "processes": [
            {"name": f"worker {i + 1}", "cpu_seconds": m.cpu_seconds, "peak_rss_mb": m.peak_rss_mb}
            for i, m in enumerate(measured)
        ]
        + [
            {
                "name": name,
                "cpu_seconds": watched_cpu[name],
                "peak_rss_mb": procstat.peak_rss_mb(pid),
            }
            for name, pid in args.watch.items()
        ],
    }


def limit_of(args: argparse.Namespace, found: dict[str, Any]) -> str | None:
    """What limit a level hit, said (``None``: none)."""
    if found["runs"] and found["errors"] / found["runs"] > args.max_errors:
        return f"errors: {found['errors']} of {found['runs']} runs"
    if args.max_p95 is not None and (found["p95"] or 0) > args.max_p95:
        return f"latency: p95 {found['p95']:.1f}s > {args.max_p95:g}s"
    return None


def leak(args: argparse.Namespace) -> dict[str, Any]:
    """One worker at ``--leak-concurrency`` for ``--leak-minutes``: its memory over the runs."""
    plan = Plan(
        concurrency=args.leak_concurrency,
        seconds=args.leak_minutes * 60,
        model=args.model,
        pace=args.pace,
        window=args.window,
        sample_seconds=args.sample_seconds,
        run_seconds=args.run_seconds,
    )
    [measured] = in_processes([plan])
    samples = [s for s in measured.samples if s[1] is not None]
    warm = [s for s in samples if s[0] >= samples[-1][0] / 5] if samples else []
    first, last = (warm[0], warm[-1]) if len(warm) >= 2 else (None, None)
    growth = None if first is None or last is None else last[1] - first[1]  # type: ignore[operator]
    runs = None if first is None or last is None else last[2] - first[2]
    return {
        "minutes": args.leak_minutes,
        "concurrency": args.leak_concurrency,
        "runs": len(measured.latencies) + len(measured.errors),
        "errors": len(measured.errors),
        "samples": samples,
        "after_warm_up_mb": None if first is None else first[1],
        "end_mb": None if last is None else last[1],
        "growth_mb": None if growth is None else round(growth, 1),
        "growth_kb_per_run": None if not runs or growth is None else round(growth * 1024 / runs, 2),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Load benchmark: {report['model']}",
        "",
        f"Measured {report['date']} on {report['machine']}; {report['rounds']} runs per lane; "
        "latency is one run's (a tool call and the answer), in seconds.",
        "",
        "| workers | concurrency | runs | runs/s | p50 s | p95 s | p99 s | errors | CPU s per "
        "process | peak MB per process | limit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def num(value: float | None, digits: int = 2) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    for row in report["levels"]:
        cpu = ", ".join(f"{p['name']} {num(p['cpu_seconds'], 1)}" for p in row["processes"])
        peak = ", ".join(f"{p['name']} {num(p['peak_rss_mb'], 0)}" for p in row["processes"])
        lines.append(
            f"| {row['workers']} | {row['concurrency']} | {row['runs']} | "
            f"{num(row['throughput'])} | {num(row['p50'])} | {num(row['p95'])} | "
            f"{num(row['p99'])} | {row['errors']} | {cpu} | {peak} | {row.get('limit') or '—'} |"
        )
    found = report.get("leak")
    if found:
        lines += [
            "",
            f"Leak check: one worker, {found['concurrency']} at once, {found['minutes']:g} "
            f"minutes, {found['runs']} runs ({found['errors']} errors): resident memory "
            f"{num(found['after_warm_up_mb'], 0)} MB after the warm-up, "
            f"{num(found['end_mb'], 0)} MB at the end — {num(found['growth_mb'], 1)} MB "
            f"({num(found['growth_kb_per_run'])} kB per run).",
        ]
    run = report.get("queued")
    if run:
        lines += [
            "",
            f"Queued in agent-runs, {run['workers']} worker process(es), {run['concurrency']} in "
            f"flight, {run['minutes']:g} minutes, memory {'on' if run['memory'] else 'off'}: "
            f"{run['runs']} runs, {num(run['throughput'])} runs/s, p50 {num(run['p50'])} s, "
            f"p95 {num(run['p95'])} s, p99 {num(run['p99'])} s, {run['errors']} errors.",
            "",
            "| process | resident MB after the warm-up | at the end | slope, MB per hour |",
            "|---|---|---|---|",
        ]
        for p in run["processes"]:
            known = [m for t, m in p["rss"] if m is not None and t >= p["rss"][-1][0] / 5]
            first, last = (known[0], known[-1]) if known else (None, None)
            lines.append(
                f"| {p['name']} | {num(first, 0)} | {num(last, 0)} | "
                f"{num(p['slope_mb_per_hour'], 1)} |"
            )
    for note in report.get("notes", []):
        lines += ["", note]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": args.model or f"Paced ({args.pace:g}s per call, no model)",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "machine": f"{os.uname().machine}, {os.cpu_count()} CPUs",
        "rounds": args.rounds,
        "settings": {k: v for k, v in vars(args).items() if k != "out"},
        "levels": [],
        "notes": [],
    }

    def written() -> None:
        out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
        out.with_suffix(".md").write_text(markdown(report))

    if args.queued > 0:
        plan = Plan(concurrency=1, model=args.model, pace=args.pace, window=args.window,
                    sample_seconds=args.sample_seconds, run_seconds=args.run_seconds)  # fmt: skip
        report["queued"] = asyncio.run(
            queued.drive(
                plan,
                seconds=args.queued * 60,
                concurrency=args.leak_concurrency,
                workers=args.workers[0],
                memory=not args.no_memory,
                watch=args.watch,
            )
        )
        written()
        sys.stdout.write(markdown(report))
        return 0
    for workers in args.workers:
        for concurrency in args.levels:
            free = free_mb()
            if free is not None and free < args.min_free_mb:
                report["levels"].append(
                    {
                        "workers": workers,
                        "concurrency": concurrency,
                        "runs": 0,
                        "seconds": 0,
                        "throughput": None,
                        "p50": None,
                        "p95": None,
                        "p99": None,
                        "errors": 0,
                        "processes": [],
                        "limit": f"memory: {free:,.0f} MB free",
                    }
                )
                written()
                break
            found = level(args, workers, concurrency)
            found["limit"] = limit_of(args, found)
            report["levels"].append(found)
            written()
            sys.stdout.write(markdown(report).splitlines()[-1] + "\n")
            if found["limit"]:
                break
    if args.leak_minutes > 0:
        report["leak"] = leak(args)
        written()
    sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
