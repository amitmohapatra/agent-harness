"""A memory service that is down, slow, refusing or answering garbage degrades a run — it
never fails it: who the key is stays what it was, the memory tools are left out with a
warning, the catalog falls back to the tools' own tiers with every tool that does more than
read asking, and context and records are warnings. Only a key the service refuses, or one it
could never be asked about, is a configuration error."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from tests.support.memory import FakeMemoryService
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, InterruptReason, RunEventType, RunStatus
from trellis.harness import fresh as fresh_module
from trellis.harness import writes as writes_module


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.0)


@pytest.fixture
async def harness(memory_service: FakeMemoryService) -> AsyncIterator[Harness]:
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        yield h


async def echo(input: str, agent: Runtime) -> str:
    return input


@pytest.mark.parametrize("failure", [503, 429, "timeout", "malformed"])
async def test_a_key_never_read_is_a_clear_configuration_error(
    harness: Harness, memory_service: FakeMemoryService, failure: int | str
) -> None:
    memory_service.failures["key"] = failure
    with pytest.raises(ConfigurationError, match="could not be reached to say who"):
        await harness.wrap(echo, id="echo").run("q", user="u")


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_key_is_a_configuration_error(
    harness: Harness, memory_service: FakeMemoryService, status: int
) -> None:
    memory_service.failures["key"] = status
    with pytest.raises(ConfigurationError, match="refused TRELLIS_API_KEY"):
        await harness.tenant()


async def test_a_known_key_outlives_an_outage_and_is_read_again_later(
    harness: Harness,
    memory_service: FakeMemoryService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [100.0]
    monkeypatch.setattr(fresh_module.time, "monotonic", lambda: now[0])
    assert await harness.tenant() == "acme"
    memory_service.failures["key"] = 503
    now[0] += 601  # past the key's TTL: asked again, and the service is down
    with caplog.at_level(logging.WARNING, logger="trellis.fresh"):
        assert await harness.tenant() == "acme"
        assert await harness.tenant() == "acme"  # not asked again before the retry interval
    assert len(memory_service.named("key")) == 1
    assert caplog.text.count("could not be read again") == 1
    del memory_service.failures["key"]
    memory_service.tenant = "acme"
    now[0] += 31
    assert await harness.tenant() == "acme"
    assert len(memory_service.named("key")) == 2
    memory_service.failures["key"] = 401  # revoked meanwhile: no stale answer hides that
    now[0] += 601
    with pytest.raises(ConfigurationError, match="refused"):
        await harness.tenant()


async def test_memory_tools_that_cannot_be_listed_leave_the_run_without_them(
    harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.failures["agent_tools"] = 503
    offered: list[list[str]] = []

    async def looks(input: str, agent: Runtime) -> str:
        offered.append(sorted(agent.toolbox))
        return input

    agent = harness.wrap(looks, id="looks")
    events = [e async for e in agent.stream("q", user="u")]
    assert events[-1].type is RunEventType.RUN_FINISHED and offered == [[]]
    warnings = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any(w["name"] == "warning" and "no memory tools" in w["message"] for w in warnings)


async def test_a_catalog_that_cannot_be_read_makes_writes_ask_and_the_run_goes_on(
    harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.failures["catalog"] = "timeout"

    @tool(side_effects="read")
    def stock(sku: str) -> int:
        """Units in stock."""
        return 3

    @tool(side_effects="write")
    def reorder(sku: str) -> str:
        """Reorder a SKU."""
        return "ordered"

    async def restocks(input: str, agent: Runtime) -> str:
        await agent.tools.call("stock", sku="a")  # reads: still runs
        return await agent.tools.call("reorder", sku="a")  # writes: now asks

    agent = harness.wrap(restocks, id="restocks", tools=[stock, reorder])
    paused = await agent.run("restock a", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.reason is InterruptReason.APPROVAL
    assert "tool catalog" in paused.interrupt.question
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="ops")
    assert done.status is RunStatus.SUCCESS and done.answer == "ordered"


@pytest.mark.parametrize("failure", [503, 429, 422, "timeout", "malformed"])
async def test_a_context_that_fails_any_way_is_a_warning(
    harness: Harness, memory_service: FakeMemoryService, failure: int | str
) -> None:
    memory_service.failures["context"] = failure
    events = [e async for e in harness.wrap(echo, id="echo").stream("q", user="u")]
    assert events[-1].type is RunEventType.RUN_FINISHED
    warnings = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any(w["name"] == "warning" and "no memory context" in w["message"] for w in warnings)


@pytest.mark.parametrize("failure", [401, 403, 422, 429, 503, "timeout"])
async def test_a_record_that_fails_any_way_never_fails_the_run(
    harness: Harness, memory_service: FakeMemoryService, failure: int | str
) -> None:
    memory_service.failures["messages"] = failure
    result = await harness.wrap(echo, id="echo").run("q", user="u")
    assert result.status is RunStatus.SUCCESS
    await harness.writes.drain()
    assert harness.writes.failed == 1  # reported, counted
