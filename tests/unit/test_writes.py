from __future__ import annotations

import asyncio

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


async def test_a_delayed_write_is_run_now_by_drain() -> None:
    writes, done = Writes(), []

    async def later() -> None:
        done.append(1)

    writes.submit("later", later, delay=60)
    await writes.drain()
    assert done == [1]
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
