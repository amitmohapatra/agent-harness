"""The framework head-to-head: ``make bench-frameworks`` (``python -m benchmark.cost.frameworks``).

    BIFROST_URL=http://127.0.0.1:8180/v1 BIFROST_VIRTUAL_KEY=... MEMORY_URL=... RUNS_URL=... \\
      TRELLIS_API_KEY=... python -m benchmark.cost.frameworks --model provider/model

Starts the metering proxy (:mod:`.meter`) in front of the gateway, then runs every arm
(:data:`.arms.ARMS`) in a process of its own, in an order shuffled with ``--seed``: the cells'
runs (``--cells``, ``--n`` each), then the load check (concurrency 1 and 10) of each
framework's plain arms, then the kill-mid-run trials of the wrapped ones (B, C). Each child
appends its rows to ``<out>/rows.jsonl``; ``report`` (run at the end) writes the markdown.
A native arm runs with no harness service in its environment (no ``MEMORY_URL``, ``RUNS_URL``):
the wrapped ones use the private ones the environment names.

The Claude Agent SDK's arms run only when the gateway's Anthropic-compatible route answers for
the model (else they are skipped, saying so); CrewAI's runs in its own interpreter
(``--crewai-python``), skipped when that cannot import it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from benchmark.cost.frameworks import arms as built
from benchmark.cost.frameworks import report, runner

ROOT: Final = Path(__file__).resolve().parents[3]
#: The variables of the harness's services, left out of a native arm's environment.
SERVICES: Final = ("MEMORY_URL", "RUNS_URL", "TRELLIS_API_KEY", "TRELLIS_SPOOL_DIR")
#: The plain arm of each framework: the ones the load check runs.
PLAIN: Final = tuple(
    a.name for a in built.ARMS.values() if a.feature is None and a.kind in ("A", "B", "C")
)


def _list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.cost.frameworks", description=__doc__
    )
    parser.add_argument(
        "command", nargs="?", default="run", choices=["run", "arm", "load", "survive", "report"]
    )
    parser.add_argument("--model", default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"))
    parser.add_argument("--arms", default=",".join(built.ARMS), help="arms by name")
    parser.add_argument("--arm", default=None, help="(a child's) its arm")
    parser.add_argument("--cells", default=",".join(runner.DEFAULT_CELLS))
    parser.add_argument("--n", type=int, default=6, help="runs per cell (at least 5)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=int, default=8192, help="the model's context window")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per run")
    parser.add_argument("--load", default="1:5,10:10", help="concurrency:runs pairs ('' skips)")
    parser.add_argument("--concurrency", type=int, default=1, help="(a load child's)")
    parser.add_argument("--runs", type=int, default=5, help="(a load child's)")
    parser.add_argument(
        "--trials", type=int, default=5, help="kill trials per wrapped arm (0 skips)"
    )
    parser.add_argument("--root", default=None, help="(a child's) the metering proxy")
    parser.add_argument("--crewai-python", default=env("FWBENCH_CREWAI_PYTHON"))
    parser.add_argument("--out", default="build/bench-frameworks")
    parser.add_argument("--doc", default="docs/eval/frameworks.md")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command != "report" and not args.model:
        parser.error("no model: --model, TRELLIS_BENCH_MODEL or TRELLIS_LIVE_MODEL")
    unknown = [a for a in _list(args.arms) if a not in built.ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}: the arms are {', '.join(built.ARMS)}")
    if args.n < 1:
        parser.error("--n is at least 1")
    bad = [c for c in _list(args.cells) if c not in runner.CELLS]
    if bad:
        parser.error(f"unknown cells {bad}: the cells are {', '.join(runner.CELLS)}")
    try:
        args.load = loads(args.load)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def loads(text: str) -> list[tuple[int, int]]:
    """``concurrency:runs`` pairs."""
    found = []
    for pair in _list(text):
        c, _, n = pair.partition(":")
        if not (c.isdigit() and n.isdigit() and int(c) >= 1 and int(n) >= 1):
            raise ValueError(f"--load takes concurrency:runs pairs, not {pair!r}")
        found.append((int(c), int(n)))
    return found


def shuffled(names: Sequence[str], seed: int) -> list[str]:
    """``names`` in an order fixed by ``seed``."""
    order = list(names)
    random.Random(seed).shuffle(order)
    return order


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def upstream_of(bifrost_url: str) -> str:
    """The gateway's root, from ``BIFROST_URL`` (its ``/v1``)."""
    url = bifrost_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def child_env(arm: built.Arm, root: str, key: str) -> dict[str, str]:
    """A child's environment: the proxy as the gateway; a native arm without the harness's
    services."""
    env = {**os.environ, "BIFROST_URL": f"{root}/v1", "BIFROST_VIRTUAL_KEY": key}
    env |= {"FWBENCH_ROOT": root, "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true"}
    env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    if not arm.wrapped:
        for name in SERVICES:
            env.pop(name, None)
    return env


# --------------------------------------------------------------------------- availability
def anthropic_route(root: str, key: str, model: str) -> str | None:
    """``None`` when the gateway's Anthropic-compatible route answers for ``model``; else why
    not."""
    try:
        reply = httpx.post(
            f"{root}/anthropic/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say OK."}],
            },
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        return f"the gateway's Anthropic route did not answer ({exc})"
    if reply.status_code != 200 or reply.json().get("type") != "message":
        return f"the gateway's Anthropic route answered {reply.status_code}: {reply.text[:200]}"
    return None


def crewai_problem(python: str | None) -> str | None:
    """``None`` when ``python`` can import CrewAI and the benchmark; else why not."""
    if not python:
        return "no CrewAI interpreter (--crewai-python): CrewAI is installed in a venv of its own"
    probe = subprocess.run(
        [python, "-c", "import crewai, benchmark.cost.frameworks.runner"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return f"CrewAI's interpreter cannot run the arm: {probe.stderr.strip().splitlines()[-1:]}"
    return None


# --------------------------------------------------------------------------- the orchestrator
def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


@dataclass(frozen=True)
class Proxy:
    """The metering proxy as the children reach it, and the gateway's key."""

    root: str
    key: str


def _child(
    args: argparse.Namespace, arm: built.Arm, command: str, proxy: Proxy, *extra: str
) -> int:
    python = args.crewai_python if arm.venv == "crewai" else sys.executable
    line = [
        python, "-m", "benchmark.cost.frameworks", command,
        "--arm", arm.name, "--model", args.model, "--root", proxy.root,
        "--cells", ",".join(_list(args.cells)), "--n", str(args.n), "--seed", str(args.seed),
        "--window", str(args.window), "--timeout", str(args.timeout), "--out", args.out,
        *extra,
    ]  # fmt: skip
    began = time.monotonic()
    env = child_env(arm, proxy.root, proxy.key)
    done = subprocess.run(line, cwd=ROOT, env=env, check=False)
    say(f"[{arm.name}] {command} exited {done.returncode} after {time.monotonic() - began:.0f}s")
    return done.returncode


def _skip(out: Path, arm: str, why: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"arm": arm, "skipped": why}) + "\n")
    say(f"[{arm}] skipped: {why}")


def unavailable(
    args: argparse.Namespace, chosen: Sequence[built.Arm], proxy: Proxy
) -> dict[str, str]:
    """The chosen arms that cannot run here, each with why: the Claude Agent SDK's without an
    Anthropic route to the model, CrewAI's without its interpreter."""
    skipped: dict[str, str] = {}
    if any(a.framework == "claude_agent_sdk" for a in chosen):
        why = anthropic_route(proxy.root, proxy.key, args.model)
        if why:
            skipped |= {
                a.name: f"needs an Anthropic key: {why}"
                for a in chosen
                if a.framework == "claude_agent_sdk"
            }
    if any(a.framework == "trueforge" for a in chosen):
        from benchmark.cost.frameworks.trueforge import install_problem

        why = install_problem()
        if why:
            skipped |= {a.name: why for a in chosen if a.framework == "trueforge"}
    if any(a.venv == "crewai" for a in chosen):
        why = crewai_problem(args.crewai_python)
        if why:
            skipped |= {a.name: why for a in chosen if a.venv == "crewai"}
    return skipped


def _ready(root: str) -> None:
    for _ in range(100):
        try:
            httpx.get(f"{root}/__meter", timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise SystemExit("the metering proxy did not start")


def measure(args: argparse.Namespace, proxy: Proxy, rows: Path) -> None:
    """Every arm's cells, then the load checks, then the kill trials — each in a child."""
    chosen = [built.ARMS[a] for a in _list(args.arms)]
    skipped = unavailable(args, chosen, proxy)
    for name, why in skipped.items():
        _skip(rows, name, why)
    order = shuffled([a.name for a in chosen if a.name not in skipped], args.seed)
    say(f"arm order: {', '.join(order)}")
    for name in order:
        _child(args, built.ARMS[name], "arm", proxy)
    for name in shuffled([n for n in order if n in PLAIN], args.seed + 1):
        for concurrency, runs in args.load:
            extra = ("--concurrency", str(concurrency), "--runs", str(runs))
            _child(args, built.ARMS[name], "load", proxy, *extra)
    if args.trials and os.environ.get("RUNS_URL"):
        wrapped = [n for n in order if n in PLAIN and built.ARMS[n].wrapped]
        for name in shuffled(wrapped, args.seed + 2):
            _child(args, built.ARMS[name], "survive", proxy, "--trials", str(args.trials))


def orchestrate(args: argparse.Namespace) -> int:
    bifrost = os.environ.get("BIFROST_URL", "").strip()
    if not bifrost:
        raise SystemExit("BIFROST_URL is not set")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.jsonl"
    args.out = str(rows)
    port = free_port()
    proxy = Proxy(f"http://127.0.0.1:{port}", os.environ.get("BIFROST_VIRTUAL_KEY", "").strip())
    line = [sys.executable, "-m", "benchmark.cost.frameworks.meter"]
    line += ["--upstream", upstream_of(bifrost), "--port", str(port)]
    server = subprocess.Popen(line, cwd=ROOT)
    try:
        _ready(proxy.root)
        measure(args, proxy, rows)
    finally:
        server.terminate()
        server.wait(timeout=20)
    return write_report(args, rows)


def write_report(args: argparse.Namespace, rows: Path) -> int:
    found = report.read_rows(rows)
    meta = {
        "intro": intro(args, found),
        "notes": args.note,
    }
    text = report.markdown(found, meta)
    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text)
    (rows.parent / "summary.json").write_text(
        json.dumps(report.summarize(found), indent=2, default=str)
    )
    sys.stdout.write(text)
    return 0


def intro(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    runs = report.runs_of(rows)
    dates = sorted({str(r["date"]) for r in runs if r.get("date")})
    cells = ", ".join(report.title(c) for c in _list(args.cells))
    return (
        f"`make bench-frameworks`: {len(runs)} runs, measured "
        f"{dates[0] if dates else datetime.now(UTC).strftime('%Y-%m-%d')} on "
        f"{os.cpu_count()} CPUs. Cells: {cells}; {args.n} runs per cell (its tasks in turn, the "
        f"k-th pass with seed {args.seed}+k), arms in an order shuffled with seed {args.seed}, "
        "each arm in a process of its own (cold start)."
    )


# --------------------------------------------------------------------------- the children
def child_options(args: argparse.Namespace) -> runner.Options:
    if not args.arm or not args.root:
        raise SystemExit("a child takes --arm and --root")
    key = os.environ.get("BIFROST_VIRTUAL_KEY", "")
    return runner.Options(
        arm=built.ARMS[args.arm],
        setup=built.Setup.on(args.root, args.model, key, window=args.window),
        meter_url=args.root,
        cells=tuple(_list(args.cells)),
        n=args.n,
        seed=args.seed,
        timeout=args.timeout,
        out=Path(args.out),
    )


async def survive(args: argparse.Namespace) -> None:
    from benchmark.cost.frameworks import survive as killing

    options = child_options(args)
    log = Path(args.out).parent / "survive-workers.log"
    for index in range(args.trials):
        found = await killing.trial(options, index, root=ROOT, log=log)
        say(json.dumps(found))


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == "run":
        return orchestrate(args)
    if args.command == "report":
        return write_report(args, Path(args.out) / "rows.jsonl")
    options = child_options(args)
    if args.command == "arm":
        asyncio.run(runner.run_arm(options))
    elif args.command == "load":
        asyncio.run(runner.load_check(options, args.concurrency, args.runs))
    else:
        asyncio.run(survive(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
