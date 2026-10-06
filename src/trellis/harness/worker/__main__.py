"""``python -m trellis.harness.worker module:attribute [--concurrency N]`` — run a worker for
every agent a Harness wraps (the attribute names the ``Harness``; importing the module wraps
the agents).

SIGTERM and SIGINT stop it gracefully: no new claims, the runs it holds finish (or, past the
grace period, are released for another worker), the memory writes drain, and it exits ``0``.
A second signal releases the runs at once. It logs JSON lines to stderr, or text when stderr is
a terminal (``trellis.harness.logs``)."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys

from trellis.harness import logs
from trellis.harness.harness import Harness

USAGE = "python -m trellis.harness.worker module:harness_attribute [--concurrency N]"


def load(target: str) -> Harness:
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise SystemExit(f"usage: {USAGE}")
    harness = getattr(importlib.import_module(module_name), attribute, None)
    if not isinstance(harness, Harness):
        raise SystemExit(f"{target} is not a trellis Harness")
    if not harness.agents:
        raise SystemExit(f"{target} wraps no agents")
    return harness


async def serve(harness: Harness, *, concurrency: int | None = None) -> None:
    """Work until SIGTERM/SIGINT (or cancellation), then close the harness."""
    worker = harness.worker(list(harness.agents.values()), concurrency=concurrency)
    try:
        await worker.serve()
    finally:
        await harness.aclose()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m trellis.harness.worker", usage=USAGE)
    parser.add_argument("target", help="module:attribute naming the Harness")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="runs executed at once (default TRELLIS_WORKER_CONCURRENCY, else the CPU count, 1-8)",
    )
    args = parser.parse_args(argv)
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    logs.configure(sys.stderr)  # JSON lines unless a terminal reads them
    harness = load(args.target)
    try:
        asyncio.run(serve(harness, concurrency=args.concurrency))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
