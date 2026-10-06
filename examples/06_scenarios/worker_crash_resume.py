"""Scenario: a worker dies in the middle of a run; another worker finishes it, and nothing the
first one did happens twice.

* a queued run is claimed by worker A under a short lease; it pays an invoice (a write: its
  result is saved as the run's progress checkpoint on the heartbeat right after) and then the
  process dies — no finish, no release, nothing written;
* the lease lapses; agent-runs (here the in-process store) puts the run back on the queue as
  attempt 2;
* worker B claims it: the journal in the checkpoint answers the payment (it is not paid again)
  and the run goes on to its end.

``agent.execute(job)`` is what any worker calls for a claimed run (``h.worker`` does too).

    python -m examples.06_scenarios.worker_crash_resume
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime, tool
from trellis.runs import Job

LEASE_SECONDS = 1
paid: list[int] = []
crashes = ["the machine"]


class Died(BaseException):
    """The worker process dies: nothing more is written, the lease is not renewed."""


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an invoice."""
    paid.append(amount)
    return f"paid {amount}"


async def billing(invoice: str, agent: Runtime) -> str:
    receipt = await agent.tools.call("pay", amount=120)
    if crashes:  # attempt 1 dies here, after the payment
        raise Died(crashes.pop())
    return f"{invoice}: {receipt}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(billing, id="billing", tools=[pay])
        handle = await agent.start("invoice 7", user="ada")

        claimed = await h.runs.claim("worker-a", [agent.id], lease_seconds=LEASE_SECONDS)
        assert claimed is not None
        job = Job(
            record=claimed.run, worker_id="worker-a", lease_seconds=LEASE_SECONDS, store=h.runs
        )
        try:
            await agent.execute(job)
        except Died as died:
            print("worker A died:", died)
        record = await handle.status()
        print(
            "after the crash:",
            record.status.value,
            "checkpoint saved:",
            record.checkpoint is not None,
        )

        worker_b = h.worker([agent])
        # the lease lapses, the run is queued again: a worker polls the store for it
        while not await worker_b.run_once():  # noqa: ASYNC110 - polling is what a worker does
            await asyncio.sleep(0.25)
        done = await handle.result(timeout=10)
        print(done.status.value, done.answer, "| attempt", (await handle.status()).attempt)
        print("paid:", paid)  # once


if __name__ == "__main__":
    asyncio.run(main())
