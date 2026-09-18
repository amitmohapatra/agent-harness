"""How many complete turns a second, against the real service, at three concurrencies.

Not harness overhead — that is ``test_overhead.py``, and it is microseconds. This measures a
whole turn: retrieve context, run the agent, write observations back. The number is dominated
by the Memory Service, which is the point: it says what the stack delivers, and whether adding
concurrency buys anything or the work is serialised somewhere.

Writes ``throughput-results.json``. Run with ``make bench-throughput`` against a running
service; skipped when there is none.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import pytest

from universal_agent_harness import AgentExecutionContext, AgentHarness

pytestmark = pytest.mark.performance

CONCURRENCIES = (1, 5, 20)
TURNS_PER_LEVEL = 20
RESULTS = Path(__file__).resolve().parents[2] / "throughput-results.json"


@pytest.fixture
async def live_memory(service_available: bool):
    if not service_available:
        pytest.skip("no Memory Service; throughput is meaningless without one")
    from tests.support import MEMORY_API_KEY, MEMORY_SERVICE_URL
    from universal_memory import MemoryClient

    client = MemoryClient(MEMORY_SERVICE_URL, api_key=MEMORY_API_KEY, timeout=300.0)
    try:
        yield client
    finally:
        await client.aclose()


async def _turn(wrapped, index: int) -> float:
    context = AgentExecutionContext.create(
        tenant_id="acme",
        agent_id="bench-agent",
        thread_id=f"bench-{uuid.uuid4().hex[:8]}",
        user_id=f"bench-user-{index % 4}",
    )
    started = time.perf_counter()
    await wrapped({"question": "how much stock of SKU-1 is on hand?"}, context=context)
    return (time.perf_counter() - started) * 1000


@pytest.mark.live
async def test_turn_throughput_across_concurrencies(live_memory) -> None:
    harness = AgentHarness(
        memory=live_memory,
        defaults={"tenant_id": "acme"},
        config={"timeouts": {"memory_seconds": 300}},
    )

    async def agent(payload, runtime):
        # a turn that uses its context and produces something worth remembering
        bundle = runtime.memory_context
        seen = len(bundle.memories) if bundle is not None else 0
        return f"checked stock against {seen} remembered facts"

    wrapped = harness.wrap(agent, agent_id="bench-agent")
    report: dict[str, dict[str, float]] = {}

    for concurrency in CONCURRENCIES:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(index: int) -> float:
            async with semaphore:
                return await _turn(wrapped, index)

        started = time.perf_counter()
        latencies = await asyncio.gather(*(one(i) for i in range(TURNS_PER_LEVEL)))
        elapsed = time.perf_counter() - started
        await harness.drain()

        report[str(concurrency)] = {
            "turns": TURNS_PER_LEVEL,
            "wall_seconds": round(elapsed, 3),
            "turns_per_second": round(TURNS_PER_LEVEL / elapsed, 3),
            "latency_p50_ms": round(statistics.median(latencies), 1),
            "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        }
        print(f"concurrency {concurrency:>2}: {report[str(concurrency)]}")

    await harness.aclose()
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")

    at_one = report["1"]["turns_per_second"]
    at_twenty = report["20"]["turns_per_second"]
    print(f"\nscaling 1 -> 20: {at_twenty / at_one:.1f}x")
    assert at_twenty >= at_one, "more concurrency must not make the stack slower"
