"""Schedules: a run of the agent on a cadence, acting for the person who set it up; a worker
executes the runs the schedule queues.

    python -m examples.02_way1_function.schedule

In production the schedule lives in agent-runs (``RUNS_URL``) and its ticker queues each run at
its time; this example does not wait for the time: it fires the schedule now
(``h.runs.schedules.fire``, agent-runs' "run it now", which the in-process store also takes).
With ``MEMORY_URL`` set a scheduled run is the person's like any other: its context is pushed
from their memory and its transcript and outcome are recorded.
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime


async def briefing(input: str, agent: Runtime) -> str:
    return f"Morning briefing for {agent.user}: {input}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(briefing, id="briefing")
        schedule = await agent.schedule("weekdays", "3 new tickets", on_behalf_of="ada")
        print("next run at", schedule.next_fire_at)

        fired = await h.runs.schedules.fire(schedule.schedule_id)  # now, not at 00:00
        assert await h.worker([agent]).run_once()  # a worker claims the queued run
        record = await h.runs.get(fired.run_id)
        assert record is not None
        print(record.status.value, record.output)


if __name__ == "__main__":
    asyncio.run(main())
