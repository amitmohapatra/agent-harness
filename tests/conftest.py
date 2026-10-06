"""Shared fixtures: a harness with nothing configured (in-process runs, memory off), and one
whose memory service is the in-process fake."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from agents import set_tracing_disabled

from tests.support import memory as fake_memory
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings

set_tracing_disabled(True)


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
            if not isinstance(kwargs.get("http_async_client"), httpx.AsyncClient):
                kwargs["http_async_client"] = httpx.AsyncClient()
            super().__init__(**kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", OnHttpx)


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
