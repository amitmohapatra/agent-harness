"""Shared fixtures: a harness with nothing configured (in-process runs, memory off), and one
whose memory service is the in-process fake."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

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
