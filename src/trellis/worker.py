"""``python -m trellis.worker module:attribute`` — run a worker for every agent a Harness
wraps (the attribute names the ``Harness``; importing the module wraps the agents)."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys

from trellis.harness.harness import Harness


def load(target: str) -> Harness:
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise SystemExit("usage: python -m trellis.worker module:harness_attribute")
    harness = getattr(importlib.import_module(module_name), attribute, None)
    if not isinstance(harness, Harness):
        raise SystemExit(f"{target} is not a trellis Harness")
    if not harness.agents:
        raise SystemExit(f"{target} wraps no agents")
    return harness


async def serve(harness: Harness) -> None:
    try:
        await harness.worker(list(harness.agents.values())).run()
    finally:
        await harness.aclose()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m trellis.worker module:harness_attribute", file=sys.stderr)  # noqa: T201
        return 2
    logging.basicConfig(level=logging.INFO)
    harness = load(argv[0])
    try:
        asyncio.run(serve(harness))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
