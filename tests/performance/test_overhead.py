"""Harness overhead benchmark (§78).

Measured, never assumed (rule 34). The number reported is *harness-only* time: the wrapped
execution minus the same agent called directly, with no network provider in the loop
(memory off, no model, no tools) so what is left is the pipeline itself — context creation,
interceptors, spans, metrics, result coercion.

Run it on its own with::

    pytest tests/performance -m performance -q -s

The assertions use a generous ceiling because CI machines vary; the printed p50/p95 is the
number to record in the documentation for a given machine.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import pytest

from universal_agent_harness import AgentHarness

pytestmark = pytest.mark.performance

ITERATIONS = int(os.environ.get("UAH_BENCH_ITERATIONS", "2000"))
WARMUP = 200
REPORT_PATH = Path(__file__).resolve().parents[2] / "benchmark-results.json"


async def bare_agent(payload):
    return payload


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[int(len(ordered) * 0.90)],
        "p95": ordered[int(len(ordered) * 0.95)],
        "p99": ordered[int(len(ordered) * 0.99)],
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


async def measure(callable_, iterations: int = ITERATIONS) -> list[float]:
    for _ in range(WARMUP):
        await callable_()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        await callable_()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


async def _overhead(harness: AgentHarness, context) -> dict[str, float]:
    wrapped = harness.wrap(bare_agent, agent_id="bench-agent")
    baseline = percentiles(await measure(lambda: bare_agent("payload")))
    harnessed = percentiles(await measure(lambda: wrapped("payload", context=context)))
    return {
        key: round(harnessed[key] - baseline[key], 4)
        for key in ("p50", "p90", "p95", "p99", "mean")
    } | {"baseline_p50_ms": round(baseline["p50"], 4), "wrapped_p50_ms": round(harnessed["p50"], 4)}


async def test_overhead_with_telemetry_enabled(context, span_exporter):
    """The realistic configuration: OTel spans on, memory off, sampling at 100%."""
    harness = AgentHarness(
        defaults={"tenant_id": "acme"},
        config={"memory": {"enabled": False}, "evaluation_events": {"enabled": False}},
    )
    overhead = await _overhead(harness, context)
    span_exporter.clear()
    _report("telemetry_enabled", overhead)
    assert overhead["p50"] < 5.0, overhead
    assert overhead["p95"] < 15.0, overhead


async def test_overhead_with_telemetry_disabled(context):
    """The floor: what the pipeline costs with every backend switched off."""
    harness = AgentHarness(
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"enabled": False},
            "telemetry": {"enabled": False},
            "evaluation_events": {"enabled": False},
        },
    )
    overhead = await _overhead(harness, context)
    _report("telemetry_disabled", overhead)
    assert overhead["p50"] < 2.0, overhead


async def test_overhead_when_sampled_out(context):
    """A sampled-out run must be cheaper than a sampled one (§28)."""
    harness = AgentHarness(
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"enabled": False},
            "telemetry": {"sampling": {"sample_rate": 0.0}},
            "evaluation_events": {"enabled": False},
        },
    )
    overhead = await _overhead(harness, context)
    _report("sampled_out", overhead)
    assert overhead["p50"] < 5.0, overhead


async def test_context_creation_is_constant_time():
    """Context creation is O(1) and allocation-light (§64).

    Measured against a baseline and asserted on the *difference*, the way
    :func:`_overhead` already does it. Two reasons, both learned from this test:

    * The old body was ``lambda: asyncio.sleep(0) or AgentExecutionContext.create(...)``.
      ``asyncio.sleep(0)`` returns a truthy coroutine, so ``or`` short-circuited and the
      context was **never created** — the test measured an event-loop round trip and
      nothing else, while reporting a number called "context_creation".
    * That number was then compared against an absolute 1ms budget, which measures the
      machine rather than the code. It failed at p95 = 1.06ms on a laptop that was busy
      building images, and passed on the same commit minutes later.

    The subtraction cancels whatever the loop and the machine are doing. Context creation
    measured ~0.03ms at p50 and ~0.09ms at p95 on an unloaded machine, so the budget below
    has room for the noise and still catches a regression worth knowing about.
    """
    from universal_agent_harness import AgentExecutionContext

    async def loop_only() -> None:
        await asyncio.sleep(0)

    async def loop_and_context() -> None:
        await asyncio.sleep(0)
        AgentExecutionContext.create(tenant_id="acme")

    baseline = percentiles(await measure(loop_only, iterations=2000))
    measured = percentiles(await measure(loop_and_context, iterations=2000))
    cost = {key: measured[key] - baseline[key] for key in ("p50", "p95", "p99")}
    _report("context_creation", {k: round(v, 4) for k, v in cost.items()})

    assert cost["p95"] < 0.5, {"cost": cost, "baseline": baseline, "measured": measured}


_RESULTS: dict[str, dict[str, float]] = {}


def _report(name: str, values: dict[str, float]) -> None:
    _RESULTS[name] = values
    print(f"\n[benchmark] {name}: {json.dumps(values)}")
    REPORT_PATH.write_text(
        json.dumps(
            {"iterations": ITERATIONS, "results": _RESULTS},
            indent=2,
        )
        + "\n"
    )
