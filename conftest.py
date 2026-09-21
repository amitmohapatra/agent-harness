"""Root fixtures. Every fixture here binds to a **running** Memory Service.

There is no fake. The suite needs `docker compose up -d` in the agent-memory-service
checkout (or `MEMORY_SERVICE_URL` pointing at one); without it the memory-backed tests skip
with a message rather than quietly testing a stub.

Two deliberate choices, both about dev/prod parity:

* the harness fixture leaves ``memory.writeback`` at its production default (asynchronous),
  so tests exercise the path production takes — they call ``await harness.drain()`` before
  asserting on what was written;
* every test gets unique thread/turn/user ids, because the service is stateful and a shared
  conversation would make tests depend on each other's leftovers.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (
    DEAD_SERVICE_URL,
    MEMORY_API_KEY,
    MEMORY_SERVICE_URL,
    FaultInjectingTransport,
    RecordingMemoryClient,
)

from universal_agent_harness import AgentExecutionContext, AgentHarness

TENANT = "acme"


@pytest.fixture(scope="session")
def service_available() -> bool:
    """Whether a Memory Service is reachable. Checked once, not mocked around."""
    try:
        return httpx.get(f"{MEMORY_SERVICE_URL}/health/live", timeout=5).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def span_exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def spans(span_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    span_exporter.clear()
    return span_exporter


@pytest.fixture
def run_id() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
async def memory(service_available: bool, run_id: str) -> Any:
    """The real SDK client, with its calls recorded so tests can assert on requests too."""
    if not service_available:
        pytest.skip(
            f"no Memory Service at {MEMORY_SERVICE_URL} — start it with `make dev-up` in the "
            "agent-memory-service checkout, or set MEMORY_SERVICE_URL"
        )
    from universal_memory import MemoryClient

    client = MemoryClient(MEMORY_SERVICE_URL, api_key=MEMORY_API_KEY, timeout=120.0)
    recording = RecordingMemoryClient(client)
    try:
        yield recording
    finally:
        await client.aclose()


@pytest.fixture
async def faulty_memory(service_available: bool) -> Any:
    """The real client and the real service, with a fault injector in the socket path.

    Tests reach it through ``memory.faults``: ``faults.drop.add("/v1/context")`` refuses that
    connection for real, ``faults.stall["/v1/context"] = 10`` makes it genuinely slow. Paths
    left alone still reach the service.
    """
    if not service_available:
        pytest.skip(f"no Memory Service at {MEMORY_SERVICE_URL}")
    import httpx as _httpx
    from universal_memory import MemoryClient

    faults = FaultInjectingTransport()
    client = MemoryClient(
        MEMORY_SERVICE_URL,
        api_key=MEMORY_API_KEY,
        max_retries=0,
        http_client=_httpx.AsyncClient(
            transport=faults, timeout=120.0, base_url=MEMORY_SERVICE_URL
        ),
    )
    recording = RecordingMemoryClient(client, faults)
    try:
        yield recording
    finally:
        await client.aclose()


@pytest.fixture
async def dead_memory() -> Any:
    """A client pointed at a port nothing listens on — a real outage, not a simulated one."""
    from universal_memory import MemoryClient

    client = MemoryClient(DEAD_SERVICE_URL, api_key="unused", timeout=2.0, max_retries=0)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def context(run_id: str) -> AgentExecutionContext:
    """A fresh conversation per test: the service is stateful and tests must not share one."""
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="test-agent",
        user_id=f"user-{run_id}",
        workspace_id=f"ws-{run_id}",
        agent_group_id=f"crew-{run_id}",
        thread_id=f"thread-{run_id}",
        turn_id=f"turn-{run_id}",
        work_id=f"work-{run_id}",
    )


@pytest.fixture
async def harness(memory: Any) -> Any:
    """Configured as production is: asynchronous writeback, real timeouts.

    Tests that assert on what was written call ``await harness.drain()`` first — which is
    also what a production process does before it exits.
    """
    instance = AgentHarness(
        memory=memory,
        defaults={"tenant_id": TENANT},
        config={
            "telemetry": {"capture": {"inputs": True, "outputs": True}},
            "evaluation_events": {"enabled": True, "synchronous": True},
            # generous enough for CPU-bound model inference in the service
            "timeouts": {"memory_seconds": 120.0, "default_seconds": 300.0},
        },
    )
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture(autouse=True)
def _reset_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient Langfuse credentials must not silently change what the tests exercise."""
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "UAH_LANGFUSE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
