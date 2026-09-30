"""What the harness costs per run, measured against calling the agent directly.

Writes ``build/benchmark-results.json``; CI compares it with the committed baseline
(``benchmark-results.json``) through ``python -m trellis.eval gate``. The numbers are the
harness's own overhead (nothing configured: in-process runs, memory off), not a service's.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from trellis import Harness, Runtime, Settings, tool

ITERATIONS = 500
OUT = Path(__file__).resolve().parents[2] / "build" / "benchmark-results.json"


@tool(side_effects="read")
def lookup(sku: str) -> int:
    """Units in stock."""
    return 7


async def answer(input: str, agent: Runtime) -> str:
    return input


async def with_tool(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("lookup", sku=input)


async def measure(call: Callable[[], Awaitable[Any]]) -> dict[str, float]:
    for _ in range(20):
        await call()
    samples = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        await call()
        samples.append((time.perf_counter() - started) * 1000)
    q = statistics.quantiles(samples, n=100)
    return {
        "p50": round(q[49], 4),
        "p90": round(q[89], 4),
        "p95": round(q[94], 4),
        "p99": round(q[98], 4),
        "mean": round(statistics.fmean(samples), 4),
    }


@pytest.mark.performance
async def test_harness_overhead() -> None:
    async with Harness(config=Settings()) as h:
        plain = h.wrap(answer, id="plain")
        tooled = h.wrap(with_tool, id="tooled", tools=[lookup])

        async def stream() -> None:
            async for _ in plain.stream("hi", user="u"):
                pass

        results = {
            "run": await measure(lambda: plain.run("hi", user="u")),
            "run_with_tool": await measure(lambda: tooled.run("a", user="u")),
            "stream": await measure(stream),
        }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"iterations": ITERATIONS, "results": results}, indent=2) + "\n")
    assert results["run"]["p50"] < 5.0  # milliseconds, on anything that can run the suite
