"""Shared fixtures: a harness with nothing configured (in-process runs, memory off), and one
whose memory service is the in-process fake."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from agents import set_tracing_disabled
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support import memory as fake_memory
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings
from trellis.harness import telemetry

set_tracing_disabled(True)


@pytest.fixture(autouse=True)
def no_spool_on_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test reads its settings from the mapping it gives, never this machine's folders: no
    spool is detected (the test of the detection restores it)."""
    from trellis.harness import settings

    monkeypatch.setattr(settings, "default_spool", lambda environ: None)


@pytest.fixture(autouse=True)
def no_real_claude(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Claude Agent SDK client a test starts without the scripted CLI (``cli_path``) must
    reach no model: looking for the real CLI fails the test. Live tests run the real CLI."""
    if request.node.get_closest_marker("live") is not None:
        return
    from claude_agent_sdk._internal.transport import subprocess_cli

    def refused(self: Any) -> str:
        raise AssertionError("a test started the real Claude Code CLI: give cli_path")

    monkeypatch.setattr(subprocess_cli.SubprocessCLITransport, "_find_cli", refused)


@pytest.fixture(autouse=True)
def memory_contract() -> Iterator[None]:
    """Every memory fake a test made spoke the memory service's OpenAPI document: what the
    harness sent and what the fake answered."""
    fake_memory.MADE.clear()
    yield
    violations = [v for fake in fake_memory.MADE for v in fake.violations]
    fake_memory.MADE.clear()
    assert not violations, "\n".join(violations)


@pytest.fixture(autouse=True)
def openai_on_httpx(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ChatOpenAI`` a test builds (``ReAct`` with a gateway model name) sends through httpx,
    where ``respx`` mocks the gateway (the openai client's own transport is its httpx fork).
    Live tests talk to the real gateway as they are."""
    if request.node.get_closest_marker("live") is not None:
        return
    import langchain_openai

    class OnHttpx(langchain_openai.ChatOpenAI):
        def __init__(self, **kwargs: Any) -> None:
            given = kwargs.get("http_async_client")
            if not isinstance(given, httpx.AsyncClient):
                # the hooks the harness gave its client (what it reads of a gateway's error)
                hooks = getattr(given, "event_hooks", None)
                kwargs["http_async_client"] = httpx.AsyncClient(event_hooks=hooks)
            super().__init__(**kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", OnHttpx)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """The spans the harness ends during the test, in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with Harness(config=Settings()) as h:
        yield h


@pytest.fixture
def memory_service() -> FakeMemoryService:
    return FakeMemoryService()


@pytest.fixture
async def memory_harness(memory_service: FakeMemoryService) -> AsyncIterator[Harness]:
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        yield h
