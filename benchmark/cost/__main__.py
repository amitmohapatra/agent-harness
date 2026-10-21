"""The cost and accuracy benchmark of a ``ReAct`` agent: ``make bench-cost`` (or
``python -m benchmark.cost``), against whatever model the environment configures.

    BIFROST_URL=http://127.0.0.1:8080/v1 BIFROST_VIRTUAL_KEY=... \\
      python -m benchmark.cost --model provider/model --repeats 3

Writes ``<out>.json`` (every run, and each cell's means and 95% intervals) and ``<out>.md``
(the table). The model is a gateway model name (``--model``, else ``TRELLIS_BENCH_MODEL``,
else ``TRELLIS_LIVE_MODEL``); its context window is ``--window`` (a local model's ``-c``), else
unknown — the window-scaled features then off, as ``ReAct`` does — and the report says which
(``setup.window_source``). ``--watch name=pid`` (repeatable: the model server, the gateway)
adds those processes' CPU seconds per run and peak memory to the report. See
``docs/evaluation.md#cost-benchmark``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from benchmark.cost import catalog, prices, run, stats
from benchmark.cost.variants import DEFAULT, VARIANTS
from trellis.harness.react import window_of


def _list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def watched(pairs: list[str]) -> dict[str, int]:
    """``name=pid`` pairs, as a mapping."""
    found = {}
    for pair in pairs:
        name, _, pid = pair.partition("=")
        if not name or not pid.isdigit():
            raise argparse.ArgumentTypeError(f"--watch takes name=pid, not {pair!r}")
        found[name] = int(pid)
    return found


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python -m benchmark.cost", description=__doc__)
    parser.add_argument(
        "--model",
        default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"),
        help="the gateway model name (default: TRELLIS_BENCH_MODEL, TRELLIS_LIVE_MODEL)",
    )
    parser.add_argument("--sizes", default=",".join(map(str, catalog.SIZES)))
    parser.add_argument("--variants", default=",".join(DEFAULT), help=f"of {', '.join(VARIANTS)}")
    parser.add_argument("--sets", default="", help="task sets by name (default: all)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window", type=int, default=None, help="the model's context window")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per run")
    parser.add_argument(
        "--price-as",
        default=",".join(prices.REFERENCE),
        help="also price the tokens as these listed models (an estimate)",
    )
    parser.add_argument("--note", action="append", default=[], help="a note for the report")
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        help="name=pid of another process to measure (its CPU seconds and peak memory)",
    )
    parser.add_argument("--out", default="build/bench-cost")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the runs <out>.json already holds for this model, and run only the others",
    )
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("no model: --model, TRELLIS_BENCH_MODEL or TRELLIS_LIVE_MODEL")
    unknown = [v for v in _list(args.variants) if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}: the variants are {', '.join(VARIANTS)}")
    try:
        args.watch = watched(args.watch)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    unpriced = [m for m in _list(args.price_as) if m not in prices.PRICES]
    if unpriced:
        parser.error(f"no price for {unpriced}: the table has {', '.join(prices.PRICES)}")
    return args


async def bench(args: argparse.Namespace) -> dict:
    sizes = [int(s) for s in _list(args.sizes)]
    variants = _list(args.variants)
    chosen = set(_list(args.sets))
    sets = [s for s in catalog.task_sets() if not chosen or s.name in chosen]
    window, source = window_of(None, args.window)
    setup = {
        "model": args.model,
        "window": window,
        "max_steps": args.max_steps,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    saved = out.with_suffix(".json")
    if args.resume and saved.exists():
        before = json.loads(saved.read_text())
        rows = list(before["runs"]) if before.get("model") == args.model else []
    done = {run.key(r) for r in rows}

    def written() -> dict:
        report = run.report_of(
            rows,
            model=args.model,
            sizes=sorted({*sizes, *(r["size"] for r in rows)}),
            variants=list(dict.fromkeys([*variants, *(r["variant"] for r in rows)])),
            repeats=args.repeats,
            sets=sets,
            setup=setup,
            priced_as=_list(args.price_as),
            notes=args.note,
            watch=args.watch,
        )
        report["setup"]["window_source"] = source
        out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
        out.with_suffix(".md").write_text(stats.markdown(report))
        return report

    def kept(row: dict) -> None:
        """After every run: a benchmark stopped part way keeps what it measured."""
        rows.append(row)
        written()

    async with run.harness() as h:
        for size in sizes:
            for variant in variants:
                await run.run_cell(
                    h,
                    variant,
                    size,
                    setup=setup,
                    repeats=args.repeats,
                    sets=sets,
                    run_seconds=args.timeout,
                    done=done,
                    on_row=kept,
                    watch=args.watch,
                )
    return written()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = asyncio.run(bench(args))
    sys.stdout.write(stats.markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
