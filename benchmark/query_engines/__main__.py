"""The query engine benchmark: which engine ``query_result`` (``tools/query.py``) should run on.
The same questions over the same data through each engine (``engines.py``), against Python's own
answers (``suite.py``)::

    QUERY_ENGINES_SITE=/tmp/engines python -m benchmark.query_engines \\
        --sizes 10000,50000,500000,2000000 --out benchmark/query_engines/results/local

``QUERY_ENGINES_SITE``: a folder DuckDB, Polars and pandasql were installed in (``uv pip install
--target``); they are not the harness's dependencies. Each measurement runs in its own process,
killed past ``--cap-mb`` of memory or when the machine has less than ``--min-free-mb`` left
(reported as such). Writes ``<out>.json`` and ``<out>.md``; see
``docs/research/QUERY_ENGINE.md``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark import procstat
from benchmark.query_engines import suite

ENGINES = ("sqlite", "duckdb", "polars", "pandasql")


def available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("inf")


def worker(mode: str, args: dict[str, Any], cap_mb: float, min_free_mb: float) -> dict[str, Any]:
    """One measurement in its own process, watched: killed past ``cap_mb`` or when the machine
    runs short of memory."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "benchmark.query_engines.worker", mode, json.dumps(args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak = 0.0
    killed = None
    while proc.poll() is None:
        rss = procstat.rss_mb(proc.pid) or 0.0
        peak = max(peak, rss)
        if rss > cap_mb:
            killed = f"killed: over {cap_mb:.0f} MB resident"
        elif available_mb() < min_free_mb:
            killed = f"killed: the machine had under {min_free_mb:.0f} MB free ({rss:.0f} MB here)"
        if killed:
            proc.kill()
            break
        time.sleep(0.05)
    out, err = proc.communicate()
    if killed:
        return {"killed": killed, "peak_seen_mb": peak}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"failed": err.strip().splitlines()[-3:]}


def reps(rows: int) -> tuple[int, int]:
    """Loads and query runs per measurement, fewer as the data grows."""
    if rows <= 50_000:
        return 3, 7
    if rows <= 500_000:
        return 3, 5
    return 2, 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,50000,500000,2000000")
    parser.add_argument("--formats", default="csv,json")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--pandasql-max", type=int, default=50_000)
    parser.add_argument("--concurrent-sizes", default="50000,500000")
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--data", default=None, help="where the data is written (a temp folder)")
    parser.add_argument("--cap-mb", type=float, default=3_000)
    parser.add_argument("--min-free-mb", type=float, default=700)
    parser.add_argument("--out", default="build/query-engines")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    names = args.engines.split(",")
    data = Path(args.data or tempfile.mkdtemp(prefix="query-engines-"))
    cap = (args.cap_mb, args.min_free_mb)
    report: dict[str, Any] = {"sizes": sizes, "measure": {}, "concurrent": {}, "machine": machine()}
    for rows in sizes:
        folder = data / str(rows)
        if not (folder / "truth.json").exists():
            report.setdefault("data", {})[rows] = worker(
                "truth", {"data": str(folder), "rows": rows}, *cap
            )
        loads, queries = reps(rows)
        for fmt in args.formats.split(","):
            for name in names:
                if name == "pandasql" and rows > args.pandasql_max:
                    continue
                key = f"{name}/{rows}/{fmt}"
                sys.stderr.write(f"measure {key}\n")
                report["measure"][key] = worker(
                    "measure",
                    {
                        "engine": name,
                        "data": str(folder),
                        "fmt": fmt,
                        "load_reps": loads,
                        "query_reps": queries,
                    },
                    *cap,
                )
    for rows in [int(s) for s in args.concurrent_sizes.split(",") if s]:
        for name in names:
            if name == "pandasql":
                continue
            key = f"{name}/{rows}"
            sys.stderr.write(f"concurrent {key}\n")
            report["concurrent"][key] = worker(
                "concurrent",
                {
                    "engine": name,
                    "data": str(data / str(rows)),
                    "question": "join_groups",
                    "calls": args.calls,
                },
                *cap,
            )
    smallest = data / str(min(sizes))
    report["safety"] = {
        name: worker("safety", {"engine": name, "data": str(smallest), "scratch": str(data)}, *cap)
        for name in names
    }
    report["fidelity"] = {name: worker("fidelity", {"engine": name}, *cap) for name in names}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(report, names))
    sys.stdout.write(out.with_suffix(".md").read_text())


def machine() -> dict[str, Any]:
    import platform
    import sqlite3

    return {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
    }


# ------------------------------------------------------------------------------------ report


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.1f}"


def markdown(report: dict[str, Any], names: list[str]) -> str:
    lines = [f"Machine: {report['machine']}", ""]
    measured = report["measure"]
    lines += ["## Correctness (18 questions, against Python's answers)", ""]
    lines += [
        "| engine | size/format | as-is | dialect | wrong | error | not as-is |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, found in measured.items():
        qs = found.get("questions")
        if not qs:
            lines.append(
                f"| {key.split('/')[0]} | {'/'.join(key.split('/')[1:])} | {_why(found)} |||||"
            )
            continue
        count = {
            s: sum(q["status"] == s for q in qs.values())
            for s in ("as-is", "dialect", "wrong", "error")
        }
        other = ", ".join(f"{n} ({q['status']})" for n, q in qs.items() if q["status"] != "as-is")
        lines.append(
            f"| {key.split('/')[0]} | {'/'.join(key.split('/')[1:])} | {count['as-is']} | "
            f"{count['dialect']} | {count['wrong']} | {count['error']} | {other} |"
        )
    lines += ["", "## Load (parse + ingest of the 4 tables), peak memory, whole suite", ""]
    lines += [
        "| engine | size/format | text MB | load p50 s | suite query p50 sum ms | slowest p95 ms "
        "| peak RSS +MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, found in measured.items():
        qs = found.get("questions")
        if not qs:
            lines.append(
                f"| {key.split('/')[0]} | {'/'.join(key.split('/')[1:])} | {_why(found)} ||||||"
            )
            continue
        p50 = sum(q.get("p50", 0) for q in qs.values())
        p95 = max(q.get("p95", 0) for q in qs.values())
        lines.append(
            f"| {key.split('/')[0]} | {'/'.join(key.split('/')[1:])} | {found['text_mb']:.1f} | "
            f"{found['load']['p50']:.2f} | {_ms(p50)} | {_ms(p95)} | {found['peak_rss_mb']:.0f} |"
        )
    for rows in report["sizes"]:
        keys = [
            k
            for k in measured
            if k.split("/")[1:] == [str(rows), "csv"] and "questions" in measured[k]
        ]
        if not keys:
            continue
        lines += ["", f"## Query time p50 / p95 (ms), {rows:,} rows, CSV ledger", ""]
        lines += [
            "| question | " + " | ".join(k.split("/")[0] for k in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for q in suite.QUESTIONS:
            cells = []
            for k in keys:
                found = measured[k]["questions"][q.name]
                mark = "" if found["status"] == "as-is" else f" ({found['status']})"
                cells.append(f"{_ms(found.get('p50'))} / {_ms(found.get('p95'))}{mark}")
            lines.append(f"| {q.name} | " + " | ".join(cells) + " |")
    lines += ["", "## 8 calls at once (each loads the 4 tables and runs join_groups)", ""]
    lines += [
        "| engine/size | wall s | call p50 s | call p95 s | errors | peak RSS +MB |",
        "|---|---|---|---|---|---|",
    ]
    for key, found in report["concurrent"].items():
        if "wall_s" not in found:
            lines.append(f"| {key} | {_why(found)} |||||")
            continue
        lines.append(
            f"| {key} | {found['wall_s']:.2f} | {found['p50']:.2f} | {found['p95']:.2f} | "
            f"{len(found['errors'])} | {found['peak_rss_mb']:.0f} |"
        )
    lines += ["", "## Safety probes (a query trying more than reading its own tables)", ""]
    probes = sorted({p for found in report["safety"].values() for p in found})
    lines += ["| probe | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for p in probes:
        lines.append(
            f"| {p} | " + " | ".join(str(report["safety"][n].get(p, "n/a")) for n in names) + " |"
        )
    lines += ["", "## Value fidelity (codes, big integers, empty cells, nested values)", ""]
    for n in names:
        lines.append(f"- {n}: {report['fidelity'][n]}")
    return "\n".join(lines) + "\n"


def _why(found: dict[str, Any]) -> str:
    return str(found.get("killed") or found.get("not_installed") or found.get("failed"))


if __name__ == "__main__":
    main()
