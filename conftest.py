"""Root fixtures, shared by the harness tests and the adapter tests.

Defined once at the repository root so a single OpenTelemetry provider (and therefore a
single in-memory span exporter) serves every test root in one run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import FakeMemoryClient

from universal_agent_harness import AgentExecutionContext, AgentHarness


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
def memory() -> FakeMemoryClient:
    return FakeMemoryClient()


@pytest.fixture
def context() -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id="acme",
        agent_id="test-agent",
        user_id="u1",
        thread_id="chat-1",
        turn_id="turn-1",
    )


@pytest.fixture
def harness(memory: FakeMemoryClient) -> AgentHarness:
    return AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config={
            "memory": {"writeback": False},
            "telemetry": {"capture": {"inputs": True, "outputs": True}},
            "evaluation_events": {"enabled": True, "synchronous": True},
        },
    )


@pytest.fixture(autouse=True)
def _reset_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient Langfuse credentials must not silently change what the tests exercise."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
                 "LANGFUSE_BASE_URL", "UAH_LANGFUSE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
