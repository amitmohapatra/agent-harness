"""Way 2, runs: a schedule of your own on agent-runs' queue, executed by ``trellis.runs.Worker``
— no harness. The handler is plain code (it could call any framework).

* ``runs.schedules.create(ScheduleSpec(...))`` — a cadence, the person it acts for, the input,
  and ``metadata`` every fired run carries (your own options: here which report and a limit);
* agent-runs' ticker queues a run at each tick; ``runs.schedules.fire(id)`` queues one now;
* ``Worker(runs, handle, [agent])`` claims the fired run under a lease and runs ``handle``.

Offline (no ``RUNS_URL``) the store is the harness's in-process one, which takes the same calls.

    python -m examples.03_way2_runs.schedule
"""

from __future__ import annotations

import asyncio

from examples._support.offline import runs_store

from trellis.contracts import RunStatus, ScheduleSpec
from trellis.runs import Job, Worker

TENANT = "default"
AGENT = "reports"


async def handle(job: Job) -> None:
    """One fired run: the schedule's input, and the options it carried in its metadata."""
    record = job.record
    options = record.metadata.get("report", {})
    text = (
        f"{record.input} for {record.on_behalf_of}: {options.get('kind')}, top {options['limit']}"
    )
    await job.finish(RunStatus.SUCCESS, output=text)


async def main() -> None:
    runs = runs_store()
    spec = ScheduleSpec(
        tenant_id=TENANT,
        agent_id=AGENT,
        name="weekly sales",
        cadence="weekly",
        timezone="Europe/Paris",
        on_behalf_of="ada",
        input="Sales report",
        metadata={"report": {"kind": "by region", "limit": 5}},
    )
    schedule = await runs.schedules.create(spec)
    print("next tick:", schedule.next_fire_at)
    fired = await runs.schedules.fire(schedule.schedule_id, tenant=TENANT)  # now, not Monday

    worker = Worker(runs, handle, [AGENT], tenant=TENANT)
    while await worker.run_once():
        pass
    record = await runs.get(fired.run_id, tenant=TENANT)
    assert record is not None
    print(record.status.value, record.output)
    await runs.aclose()


if __name__ == "__main__":
    asyncio.run(main())
