"""``ReAct`` with blocks of your own: a ``Harness`` is the blocks you give it, the rest from the
environment.

* the run store is yours: agent-runs' ``RunsClient`` with ``RUNS_URL``, else the in-process one;
* the scheduler is yours: a loop that claims the agent's queued runs and hands each to
  ``agent.execute`` — the run's journal, governance and time limit go with it;
* governance is yours: a ``Governance`` (the tools' own risks here; ``Governance.from_env()``
  reads the tool catalog), so ``refund`` waits for a person;
* memory is off (``memory=False``), whatever the environment names.

    python -m examples.05_features.own_scheduler
"""

from __future__ import annotations

import asyncio
from typing import Final

from examples._support.offline import Runs, react_model, runs_store

from trellis import Agent, Harness, ReAct, tool
from trellis.harness.governance import Governance
from trellis.runs import Job

LEASE_SECONDS: Final = 60


@tool(side_effects="read")
def order(number: str) -> str:
    """An order's state."""
    return f"order {number}: paid, 40 EUR"


@tool(side_effects="irreversible")
def refund(number: str) -> str:
    """Refund an order."""
    return f"refunded {number}"


async def scheduler(runs: Runs, agent: Agent, worker_id: str) -> None:
    """Your own scheduler: claim the agent's queued runs and run each, until none is left."""
    while claimed := await runs.claim(worker_id, [agent.id], lease_seconds=LEASE_SECONDS):
        job = Job(record=claimed.run, worker_id=worker_id, lease_seconds=LEASE_SECONDS, store=runs)
        result = await agent.execute(job)
        print(f"{worker_id}: run {result.run_id} is {result.status.value}")


async def main() -> None:
    runs = runs_store()
    async with Harness(runs=runs, memory=False, governance=Governance()) as h:
        model = react_model(
            [("order", {"number": "o1"}), ("refund", {"number": "o1"}), "Refunded order o1."]
        )
        target = ReAct(system="You handle refunds. Check the order first.", model=model)
        agent = h.wrap(target, id="refunds", tools=[order, refund], timeout=300)
        handle = await agent.start("Refund order o1 if it was paid.", user="ada")
        await scheduler(runs, agent, "my-scheduler")
        paused = await handle.result()
        assert paused.interrupt is not None  # refund is irreversible: a person approves it
        print("waiting:", paused.interrupt.question)
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
        await scheduler(runs, agent, "my-scheduler")  # the approved run is queued again
        done = await handle.result()
        print(done.status.value, done.answer)
    await runs.aclose()  # a block you gave is yours to close


if __name__ == "__main__":
    asyncio.run(main())
