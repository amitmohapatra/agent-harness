"""The cost and accuracy benchmark of a ``ReAct`` agent: ``make bench-cost`` (or
``python -m benchmark.cost``), against whatever model the environment configures.

    BIFROST_URL=http://127.0.0.1:8080/v1 BIFROST_VIRTUAL_KEY=... \\
      python -m benchmark.cost --model provider/model --repeats 3

Writes ``<out>.json`` (every run, and each cell's means and 95% intervals) and ``<out>.md``
(the table). The model is a gateway model name (``--model``, else ``TRELLIS_BENCH_MODEL``,
else ``TRELLIS_LIVE_MODEL``); ``--window`` is its context window when it is not 128k (a local
model's ``-c``). See ``docs/evaluation.md#cost-benchmark``.
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
from trellis.harness.react import CONTEXT_WINDOW


def _list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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
    parser.add_argument("--out", default="build/bench-cost")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("no model: --model, TRELLIS_BENCH_MODEL or TRELLIS_LIVE_MODEL")
    unknown = [v for v in _list(args.variants) if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}: the variants are {', '.join(VARIANTS)}")
    unpriced = [m for m in _list(args.price_as) if m not in prices.PRICES]
    if unpriced:
        parser.error(f"no price for {unpriced}: the table has {', '.join(prices.PRICES)}")
    return args


async def bench(args: argparse.Namespace) -> dict:
    sizes = [int(s) for s in _list(args.sizes)]
    variants = _list(args.variants)
    chosen = set(_list(args.sets))
    sets = [s for s in catalog.task_sets() if not chosen or s.name in chosen]
    setup = {
        "model": args.model,
        "window": args.window or CONTEXT_WINDOW,
        "max_steps": args.max_steps,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def written() -> dict:
        report = run.report_of(
            rows,
            model=args.model,
            sizes=sizes,
            variants=variants,
            repeats=args.repeats,
            sets=sets,
            setup=setup,
            priced_as=_list(args.price_as),
            notes=args.note,
        )
        out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
        out.with_suffix(".md").write_text(stats.markdown(report))
        return report

    async with run.harness() as h:
        for size in sizes:
            for variant in variants:
                rows += await run.run_cell(
                    h,
                    variant,
                    size,
                    setup=setup,
                    repeats=args.repeats,
                    sets=sets,
                    run_seconds=args.timeout,
                )
                written()  # after every cell: an interrupted run keeps what it measured
    return written()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = asyncio.run(bench(args))
    sys.stdout.write(stats.markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
