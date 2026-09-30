"""Shared fixtures: a harness with nothing configured (in-process runs, memory off), and one
whose memory service is the in-process fake."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from agents import set_tracing_disabled

from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings
from trellis.harness.clients.memory import Memory

set_tracing_disabled(True)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with Harness(config=Settings()) as h:
        yield h


@pytest.fixture
def memory_service() -> FakeMemoryService:
    return FakeMemoryService()


@pytest.fixture
async def memory_harness(memory_service: FakeMemoryService) -> AsyncIterator[Harness]:
    async with Harness(config=Settings(memory_url="http://memory.test")) as h:
        h.memory = Memory("http://memory.test", None, client=memory_service.client())
        yield h
