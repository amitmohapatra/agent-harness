"""How close ``/v1/context`` runs to the memory service's 5 s read deadline (``api/deadline.py``:
past it the service answers ``504 TIMEOUT``, a status its OpenAPI document lists for no
operation) under concurrent load like a busy deployment's: one tenant, users with 40 memories
each, 1, 8 and 16 contexts asked at once (a harness asks one per run). Each level's latencies
are written to ``$TMPDIR/context-deadline.json`` and printed; the test fails if any request is
answered 504 at the load a single harness process makes (8 at once: its four writers and runs
in flight), and reports the share of the deadline the 95th percentile uses."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import pytest

from tests.features.conftest import API_KEY, MEMORY_URL, harness
from trellis.memory import MemoryClient

pytestmark = pytest.mark.timeout(600)
DEADLINE = 5.0
QUESTIONS = (
    "Which suppliers deliver steel bolts and when?",
    "What did I say about my warehouse?",
    "Summarise my standing rules for stock audits.",
    "Who approves purchase orders over 10,000 EUR?",
)


@pytest.mark.plan("MEM-10", "PERF")
async def test_context_latency_against_its_deadline_under_concurrent_load(uid: str) -> None:
    users = [f"load{n}-{uid}" for n in range(16)]
    report: dict[str, dict[str, float]] = {}
    async with harness() as h:
        tenant = await h._tenant()
        assert h.memory is not None
        client = h.memory.client

        async def seed(user: str) -> None:
            ctx = client.bind(tenant_id=tenant, user_id=user)
            for n in range(40):
                await ctx.remember(
                    f"Supplier S{n:02d} delivers {('bolts', 'pallets', 'film', 'boxes')[n % 4]}"
                    f" to warehouse {n % 5} every {('Monday', 'Tuesday', 'Friday')[n % 3]}."
                )

        await asyncio.gather(*(seed(u) for u in users))

    # each request asked once (no retry): its own latency, and a 504 is seen, not retried away
    async with MemoryClient(MEMORY_URL, api_key=API_KEY, timeout=30, max_retries=0) as once:

        async def ask(user: str, n: int) -> tuple[float, int]:
            ctx = once.bind(tenant_id=tenant, user_id=user, agent_id="load")
            started = time.monotonic()
            try:
                await ctx.context(QUESTIONS[n % len(QUESTIONS)], token_budget=8000)
                code = 200
            except Exception as exc:
                code = int(getattr(exc, "status", 0) or 0)
            return time.monotonic() - started, code

        for level in (1, 8, 16):
            results = []
            for round_ in range(3):
                results += await asyncio.gather(
                    *(ask(users[i % len(users)], i + round_) for i in range(level))
                )
            seconds = sorted(s for s, _ in results)
            p95 = seconds[max(0, int(len(seconds) * 0.95) - 1)]
            report[str(level)] = {
                "requests": len(results),
                "p50": round(statistics.median(seconds), 3),
                "p95": round(p95, 3),
                "max": round(seconds[-1], 3),
                "share_of_deadline_p95": round(p95 / DEADLINE, 2),
                "timeouts_504": sum(1 for _, c in results if c == 504),
                "other_errors": sum(1 for _, c in results if c not in (200, 504)),
            }
    out = Path(os.environ.get("TMPDIR", "/tmp")) / "context-deadline.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n/v1/context latency by concurrency:", json.dumps(report, indent=2))
    assert report["1"]["timeouts_504"] == 0 and report["8"]["timeouts_504"] == 0, report
    assert report["8"]["other_errors"] == 0, report
