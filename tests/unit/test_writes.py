from __future__ import annotations

import asyncio

import pytest

from trellis.contracts import AgentExecutionContext, RunEvent
from trellis.harness.events import RunEvents
from trellis.harness.writes import Writes


async def test_writes_run_in_the_background_and_drain_waits_for_them() -> None:
    writes, done = Writes(), []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    for _ in range(10):
        writes.submit("slow", slow)
    assert done == []
    await writes.drain()
    assert len(done) == 10
    await writes.aclose()


async def test_a_failed_write_is_counted_and_reported_to_the_run() -> None:
    writes, seen = Writes(), []
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    events.listen(seen.append)

    async def boom() -> None:
        raise ConnectionError("memory is down")

    writes.submit("memory.transcript", boom, events=events)
    await writes.drain()
    assert writes.failed == 1
    event: RunEvent = seen[0]
    assert event.data["name"] == "warning"
    assert "memory is down" in event.data["message"]
    await writes.aclose()


def test_the_loop_shutting_down_drains_the_queue() -> None:
    done: list[int] = []
    writes = Writes()

    async def write() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    async def main() -> None:
        for _ in range(20):
            writes.submit("w", write)
        # returning without drain: asyncio.run cancels the workers, which finish first

    asyncio.run(main())
    assert len(done) == 20


async def test_a_full_queue_refuses_the_write_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness import writes as module

    monkeypatch.setattr(module, "MAX_PENDING", 1)
    writes, seen = Writes(), []
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    events.listen(seen.append)

    async def write() -> None:
        return None

    writes.submit("first", write)
    writes.submit("second", write, events=events)  # nothing drained yet: no room
    assert writes.failed == 1
    assert seen[0].data["message"] == "second: the write queue is full"
    await writes.aclose()
