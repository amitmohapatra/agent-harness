"""Queue order: ``priority`` and ``concurrency_key`` on queued runs.

* ``agent.start(..., priority=100)`` — a higher priority (-1000 to 1000) is claimed first;
  equal priorities in the order they were queued;
* ``concurrency_key`` — runs sharing one run one at a time. By default it is the conversation's
  (``thread:<thread>``): a second message to a busy conversation waits for the first, in the
  queue, while other conversations go ahead.

Offline the queue is the in-process store (the same order agent-runs keeps with ``RUNS_URL``).

    python -m examples.05_features.priority_and_concurrency
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime

ran: list[str] = []
all_claimed = asyncio.Event()


async def triage(ticket: str, agent: Runtime) -> str:
    ran.append(ticket)
    if len(ran) == 4:
        all_claimed.set()
    await asyncio.sleep(0.4 if ticket == "routine-1" else 0.05)  # routine-1 takes a while
    return f"triaged {ticket}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(triage, id="triage")
        await agent.start("routine-1", user="ada", thread="t1")
        await agent.start("routine-2", user="ada", thread="t1")  # same conversation: waits
        await agent.start("outage", user="bob", thread="t2", priority=100)  # first
        await agent.start("nightly", user="ops", priority=-10, concurrency_key="nightly-jobs")

        worker = h.worker([agent], concurrency=2)  # two at a time
        serving = asyncio.create_task(worker.run())
        await all_claimed.wait()
        worker.stop()
        await serving
        # outage first (priority); routine-2 only after routine-1 ended (one conversation),
        # so nightly overtakes it
        print("claimed in this order:", ran)


if __name__ == "__main__":
    asyncio.run(main())
