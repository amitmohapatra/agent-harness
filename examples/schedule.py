"""Schedules: a run of the agent every minute, acting for the person who set it up; a worker
executes the runs the schedule queues.

    .venv/bin/python examples/schedule.py      # waits for the next minute (≤ 60 s), then exits

In production the schedule lives in agent-runs (``RUNS_URL``) and its ticker queues the runs;
here the in-process run store fires it when the worker asks for work. With ``MEMORY_URL`` set a
scheduled run is the person's like any other: its context is pushed from their memory, and its
transcript and outcome are recorded (its own thread: the run id).
"""

from __future__ import annotations

import asyncio
import contextlib

from trellis import Harness, Runtime

TIMEOUT_SECONDS = 70


async def main() -> None:
    briefed = asyncio.Event()

    async def briefing(input: str, agent: Runtime) -> str:
        text = f"Morning briefing for {agent.user}: {input}"
        if agent.context:  # memory on: what the service knows about the person and the task
            text += f" (memory context: {len(agent.context)} characters)"
        print(text)
        briefed.set()
        return text

    async with Harness() as h:
        agent = h.wrap(briefing, id="briefing")
        schedule = await agent.schedule("* * * * *", "3 new tickets", on_behalf_of="ada")
        print("next run at", schedule.next_fire_at)

        worker = asyncio.create_task(h.worker([agent]).run())
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                await briefed.wait()
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker


if __name__ == "__main__":
    asyncio.run(main())
