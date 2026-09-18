"""End-to-end against a **running** Memory Service.

Skipped unless ``MEMORY_SERVICE_URL`` is set, so the normal suite stays hermetic:

    make dev-up                       # in the agent-memory-service checkout
    MEMORY_SERVICE_URL=http://localhost:8080 MEMORY_API_KEY=dev-key \
        pytest tests/e2e/test_live_memory_service.py -q -s

Every suite in this repository talks to a running service; this file goes furthest — it drives the
real SDK against the real service and asserts that what the harness sends is accepted and
that what comes back is usable — the one test that proves the integration rather than the
harness's idea of it.
"""

from __future__ import annotations

import os
import uuid

import pytest

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResult,
    Claim,
    MemoryObservation,
)

URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")


def _reachable(url: str) -> bool:
    """Gate on the service being *there*, not on someone having exported a variable.

    An env-var gate turns "the service is down" and "I forgot to set MEMORY_SERVICE_URL"
    into the same green run. This asks the service.
    """
    import httpx

    try:
        return httpx.get(f"{url}/health/live", timeout=5).status_code == 200
    except Exception:
        return False


LIVE = _reachable(URL)
API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE, reason=f"no Memory Service at {URL} — start it with `make dev-up`"
    ),
]

TENANT = os.environ.get("MEMORY_TENANT", "acme")


@pytest.fixture
async def live_client():
    """One client per test.

    ``MemoryClient`` holds an httpx connection pool bound to the loop it was created on,
    and pytest-asyncio gives each test its own loop — a module-scoped client therefore
    works for the first test and fails with "Event loop is closed" for the rest.
    """
    from universal_memory import MemoryClient

    client = MemoryClient(URL, api_key=API_KEY, timeout=30.0)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def live_harness(live_client):
    return AgentHarness(
        memory=live_client,
        defaults={"tenant_id": TENANT, "user_id": "live-user"},
        config={"memory": {"writeback": False}, "evaluation_events": {"enabled": True}},
    )


@pytest.fixture
def live_context():
    run = uuid.uuid4().hex[:8]
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="live-agent",
        user_id="live-user",
        thread_id=f"live-thread-{run}",
        turn_id=f"turn-{run}",
        work_id=f"wo-{run}",
        # share() publishes to the agent group; USER-visible memories need the user
        agent_group_id=f"crew-{run}",
        workspace_id=f"ws-{run}",
    )


async def test_service_is_reachable(live_client):
    assert await live_client.alive()
    health = await live_client.health()
    assert health  # readiness includes the dependency probes


async def test_a_full_turn_writes_and_reads_back(live_harness, live_context, spans):
    """The automatic path: retrieve before, observe after, against the real service."""

    @live_harness.agent(agent_id="live-agent", skills=["live.check"])
    async def agent(state, runtime):
        assert runtime.memory.enabled
        # the bundle was fetched before this ran
        bundle = runtime.memory_context
        facts = runtime.memory.describe(bundle)
        assert facts["evidence_status"] in ("COMPLETE", "INCOMPLETE", "INSUFFICIENT")
        return AgentResult.ok(
            "SKU-1 has 3 units left",
            claims=[Claim(claim_id="c1", text="SKU-1 has 3 units left")],
            memory_observations=[
                MemoryObservation(content="SKU-1 stock was checked and found low")
            ],
        )

    result = await agent({"question": "how much stock of SKU-1?"}, context=live_context)
    assert result.succeeded

    from tests.support import span_by_name

    assert span_by_name(spans, "agent.memory.retrieve")
    assert span_by_name(spans, "agent.memory.observe")


async def test_every_memory_operation_against_the_real_service(live_harness, live_context):
    """Each operation the harness exposes, executed for real. Failures here mean the wire
    contract is wrong — not that our idea of it disagrees with us."""
    outcome: dict[str, object] = {}

    @live_harness.agent(agent_id="live-memory", memory_policy={"retrieve_before": False})
    async def agent(state, runtime):
        m = runtime.memory

        # -- push
        await m.record_input("How much stock of SKU-1 do we have?")
        await m.observe(
            MemoryObservation(content="Stock check for SKU-1 returned 95 units at EU-1.")
        )
        await m.remember(
            "SKU-1 is reordered from Castor Supply below 10 days of cover.",
            memory_type="SEMANTIC", lifetime="LONG_TERM", visibility="USER",
        )
        await m.remember(
            "The planner prefers weekly digests.",
            memory_type="PREFERENCE", lifetime="LONG_TERM", visibility="USER",
        )
        await m.share("SKU-1 reorder raised with Castor Supply.")
        await m.record_output("You have 95 units, about 4 days of cover.")

        # -- get
        bundle = await m.retrieve("Should we reorder SKU-1?")
        outcome["bundle"] = m.describe(bundle)
        outcome["recall"] = len(await m.recall("reorder policy", limit=5))
        outcome["history"] = len(await m.history(limit=20))
        graph = await m.graph_query("who supplies SKU-1?", hops=2)
        outcome["graph_facts"] = len(getattr(graph, "facts", []) or []) if graph else 0
        held = await m.memories(limit=50)
        outcome["held"] = len(held)
        report = await m.verify("SKU-1 is reordered from Castor Supply.", query="SKU-1 supplier")
        outcome["grounded"] = getattr(report, "grounded", None) if report else None

        # -- delete what we can identify as ours
        if held:
            await m.forget(held[0].memory_id)
            outcome["forgot"] = held[0].memory_id
        return AgentResult.ok(outcome)

    result = await agent({}, context=live_context)
    assert result.succeeded, result.error

    bundle = outcome["bundle"]
    assert isinstance(bundle, dict)
    assert bundle["evidence_status"] in ("COMPLETE", "INCOMPLETE", "INSUFFICIENT")
    assert isinstance(bundle["token_estimate"], int)
    assert outcome["history"] >= 1, "the chat turns we just wrote should be readable back"
    print("\n[live] memory operations:", outcome)


async def test_document_ingestion_and_retrieval(live_harness, live_context, tmp_path):
    doc = tmp_path / "reorder-policy.txt"
    doc.write_text(
        "Reorder policy v3.\n"
        "Safety stock is held at a 95 percent service level.\n"
        "Class-A parts never exceed thirty days of cover.\n"
    )

    @live_harness.agent(agent_id="live-ingest", memory_policy={"retrieve_before": False})
    async def agent(state, runtime):
        handle = await runtime.memory.add_document(doc, title="Reorder policy v3")
        return AgentResult.ok({"document_id": getattr(handle, "document_id", None)})

    result = await agent({}, context=live_context)
    assert result.succeeded
    assert result.data["document_id"], "the service must return a document handle"
    print(f"\n[live] ingested document: {result.data['document_id']}")


async def test_idempotency_is_enforced_by_the_service(live_harness, live_context):
    """The same logical write, replayed, must not create a second memory."""
    acks: list[object] = []

    @live_harness.agent(agent_id="live-idem", memory_policy={"retrieve_before": False})
    async def agent(state, runtime):
        ack = await runtime.memory.observe(
            MemoryObservation(content="a replayed observation", kind="EVENT")
        )
        acks.append(ack)
        return AgentResult.ok("ok")

    await agent({}, context=live_context)
    await agent({}, context=live_context)      # same context -> same idempotency key

    first, second = acks
    assert getattr(first, "observation_id", None) == getattr(second, "observation_id", None), (
        "a replayed write produced a different observation: idempotency is not holding"
    )
    print(f"\n[live] replayed write deduplicated: {getattr(second, 'deduplicated', '?')}")


def test_the_shipped_examples_run_against_the_live_service():
    """The examples, run as the user would run them, with the real service behind them.

    A subprocess rather than an import: the examples build their harness at import time
    from the environment, which is exactly the path a reader will take.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "MEMORY_SERVICE_URL": URL or "",
        "MEMORY_API_KEY": API_KEY,
        "MEMORY_TENANT": TENANT,
        # a thread of its own, so repeated runs never interfere with each other
        "MEMORY_THREAD_ID": f"live-example-{uuid.uuid4().hex[:8]}",
        "PYTHONPATH": str(root),
    }
    for example in ("memory_tour.py", "reorder_workflow.py"):
        proc = subprocess.run(
            [sys.executable, str(root / "examples" / example)],
            capture_output=True, text=True, env=env, timeout=180, check=False,
        )
        assert proc.returncode == 0, f"{example} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
        summary = [line for line in proc.stdout.splitlines() if not line.startswith("{")]
        print(f"\n[live] {example}:\n" + "\n".join(summary[-14:]))
