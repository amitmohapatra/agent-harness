"""What the harness costs per run, measured against calling the agent directly.

Writes ``build/benchmark-results.json`` and fails when a median or p95 is more than
:data:`MAX_REGRESSION` (1.75) times the committed baseline (``benchmark-results.json``). The machine
that runs it is not the one the baseline was measured on (a shared CI runner was twice as slow
at the same commit), so each percentile is first scaled by a fixed :func:`reference` workload
timed in the same process (its median on the baseline's machine over its median here): what
fails is the harness getting slower than the machine it runs on, not a slower machine. Each
call is timed in this process's CPU time (``time.process_time``), not the wall clock: the
harness's in-process overhead is CPU work, and the wall clock of a runner shared with other
jobs adds their time slices (a p95 of 4 ms for a run of 0.4). The numbers are the harness's
own overhead (nothing configured: in-process runs, memory off), not
a service's. Whether an *agent* got worse is Langfuse's question (experiments over datasets
with its evaluators), not this benchmark's.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from trellis import Harness, Runtime, Settings, tool

ITERATIONS = 500
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "benchmark-results.json"
BASELINE = ROOT / "benchmark-results.json"
#: How many times the baseline a percentile may reach, scaled to its machine, before the
#: benchmark fails: under 2, so that twice the overhead fails
MAX_REGRESSION = 1.75
#: Below this many milliseconds a percentile is noise, whatever its ratio.
FLOOR_MS = 0.5


@tool(side_effects="read")
def lookup(sku: str) -> int:
    """Units in stock."""
    return 7


async def answer(input: str, agent: Runtime) -> str:
    return input


async def with_tool(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("lookup", sku=input)


async def reference() -> None:
    """A fixed workload of the kind a run's own code does (objects, JSON, awaits), the same on
    every commit: how fast this machine runs Python here and now."""
    for i in range(20):
        payload = {"id": i, "items": [{"sku": str(j), "units": j} for j in range(8)]}
        json.loads(json.dumps(payload))
        await asyncio.sleep(0)


def regressions(
    results: dict[str, dict[str, float]],
    machine: float,
    baseline: dict[str, Any],
) -> list[str]:
    """Each baseline case's p50 and p95 past :data:`MAX_REGRESSION` times the baseline's (and
    past :data:`FLOOR_MS`), once scaled to the baseline's machine: ``machine`` is the
    :func:`reference` median here, the baseline's own is ``baseline["reference"]``."""
    scale = baseline["reference"]["p50"] / machine
    found = []
    for case, was in baseline["results"].items():
        for p in ("p50", "p95"):
            now = round(results[case][p] * scale, 4)
            if now > max(was[p] * MAX_REGRESSION, FLOOR_MS):
                found.append(f"{case} {p}: {now} ms vs {was[p]} ms (scaled by {scale:.2f})")
    return found


async def measure(call: Callable[[], Awaitable[Any]]) -> dict[str, float]:
    for _ in range(20):
        await call()
    samples = []
    for _ in range(ITERATIONS):
        started = time.process_time()
        await call()
        samples.append((time.process_time() - started) * 1000)
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
        tooled = arm(h.wrap(with_tool, id="tooled"), [lookup])

        async def stream() -> None:
            async for _ in plain.stream("hi", user="u"):
                pass

        kid = arm(h.wrap(with_tool, id="kid"), [lookup])

        async def nests(input: str, agent: Runtime) -> Any:
            return (await kid.run(input, user="u")).answer  # one child run

        nested = arm(h.wrap(nests, id="nested"), [kid])
        before = await measure(reference)
        results = {
            "run": await measure(lambda: plain.run("hi", user="u")),
            "run_with_tool": await measure(lambda: tooled.run("a", user="u")),
            "stream": await measure(stream),
            # a run whose code starts one child run (``admission.branch``, ``child_run``): less
            # "run_with_tool", the overhead of a child run in process
            "child_run": await measure(lambda: nested.run("a", user="u")),
        }
        after = await measure(reference)
    # the machine before and after the cases: a runner that slows down in between counts half
    machine = round((before["p50"] + after["p50"]) / 2, 4)
    OUT.parent.mkdir(exist_ok=True)
    report = {"iterations": ITERATIONS, "reference": {"p50": machine}, "results": results}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    found = regressions(results, machine, json.loads(BASELINE.read_text()))
    assert not found, found


BASELINE_FOR_CHECK: dict[str, Any] = {
    "reference": {"p50": 0.2},
    "results": {"run": {"p50": 0.3, "p95": 0.5}, "run_with_tool": {"p50": 0.4, "p95": 0.6}},
}


def test_a_slower_machine_is_not_a_regression_and_a_doubled_overhead_is() -> None:
    same = {"run": {"p50": 0.3, "p95": 0.5}, "run_with_tool": {"p50": 0.4, "p95": 0.6}}
    slower = {case: {p: v * 2.5 for p, v in ps.items()} for case, ps in same.items()}
    assert regressions(same, 0.2, BASELINE_FOR_CHECK) == []
    assert regressions(slower, 0.5, BASELINE_FOR_CHECK) == []  # the machine, 2.5 times slower
    doubled = {**same, "run_with_tool": {"p50": 0.85, "p95": 0.6}}
    assert regressions(doubled, 0.2, BASELINE_FOR_CHECK) == [
        "run_with_tool p50: 0.85 ms vs 0.4 ms (scaled by 1.00)"
    ]
    on_slower = {**slower, "run_with_tool": {"p50": 0.85 * 2.5, "p95": 1.5}}
    assert regressions(on_slower, 0.5, BASELINE_FOR_CHECK) == [
        "run_with_tool p50: 0.85 ms vs 0.4 ms (scaled by 0.40)"
    ]
    small = {"reference": {"p50": 0.2}, "results": {"run": {"p50": 0.1, "p95": 0.2}}}
    tiny = {"run": {"p50": 0.3, "p95": 0.45}}  # three times, under the floor: noise
    assert regressions(tiny, 0.2, small) == []


@pytest.mark.performance
async def test_the_check_catches_the_harness_doing_twice_its_work_on_this_machine() -> None:
    """Measured here, not made up: a run that does a run's work twice against this machine's
    own measure of one run is a regression; the same run measured again is not."""
    async with Harness(config=Settings()) as h:
        tooled = arm(h.wrap(with_tool, id="tooled"), [lookup])

        async def twice() -> None:
            await tooled.run("a", user="u")
            await tooled.run("a", user="u")

        machine = (await measure(reference))["p50"]
        once = await measure(lambda: tooled.run("a", user="u"))
        again = await measure(lambda: tooled.run("a", user="u"))
        doubled = await measure(twice)
    here = {"reference": {"p50": machine}, "results": {"run_with_tool": once}}
    assert regressions({"run_with_tool": again}, machine, here) == []
    [doubled_p50, *_] = regressions({"run_with_tool": doubled}, machine, here)
    assert doubled_p50.startswith("run_with_tool p50")
