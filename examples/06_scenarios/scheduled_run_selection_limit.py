"""Scenario: a scheduled run with its own selection of the harness's parts, a time limit, the
framework's options and a queue priority — the options a schedule keeps for every run it fires.

``agent.schedule(cron, user=, input=, ...)`` takes what ``agent.start`` takes: ``without=``
(here: no memory, no online judges — a nightly digest needs neither), ``run_timeout=`` (the
most working time of each fired run), ``framework_options=`` (the graph's
``recursion_limit``) and ``priority=``. agent-runs keeps them in the schedule's
metadata and copies them into each run it fires; the worker that runs it applies them.

    python -m examples.06_scenarios.scheduled_run_selection_limit
"""

from __future__ import annotations

import asyncio

from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, react_model

from trellis import Harness, ReAct, tool


@tool(side_effects="read")
def open_tickets() -> int:
    """How many support tickets are open."""
    return 12


async def main() -> None:
    memory = ScriptedMemory()
    model = react_model([("open_tickets", {}), "12 tickets are open."])
    async with Harness(**offline_blocks(memory=memory)) as h:
        target = ReAct(
            system_prompt="You write a one-line digest of open tickets.",
            model=model,
            tools=[open_tickets],
        )
        agent = h.wrap(target, id="digest")
        schedule = await agent.schedule(
            "0 7 * * 1-5",
            user="ada",
            input="Morning digest",
            tz="Europe/Paris",
            without={"memory", "judges"},
            run_timeout=120,
            framework_options={"recursion_limit": 12},
            priority=10,
        )
        print("kept with the schedule:", schedule.metadata, schedule.timeout_seconds)

        fired = await h.runs.schedules.fire(schedule.schedule_id)  # the 07:00 tick, now
        assert await h.worker([agent]).run_once()
        record = await h.runs.get(fired.run_id)
        assert record is not None
        print(record.status.value, record.output, "| priority", record.priority)
        print("the fired run's options:", record.metadata, record.timeout_seconds)
        print("contexts pushed:", memory.asked["context_"])  # 0: without memory


if __name__ == "__main__":
    asyncio.run(main())
