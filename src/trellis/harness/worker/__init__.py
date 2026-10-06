"""The worker: ``trellis.runs.Worker`` running wrapped agents.

The claim loop is agent-runs' SDK's (``trellis.runs.Worker``): it claims queued runs of the
agents, keeps their leases with heartbeats (a lost lease cancels the run, which writes nothing;
a heartbeat that says the run was cancelled cancels it too, and it ends ``CANCELLED``), backs
off while the queue is empty, runs ``concurrency`` at a time, and on a stop lets the runs it
holds finish for a grace period before it releases the rest (cancelled with
``trellis.runs.RELEASED``: nothing written, and the run goes back on the queue for another
worker at once). This module hands it each claimed run's agent (``Agent.execute``: the next
attempt, with the run's checkpoint as its journal and the working time its lease says is
left) and, around the loop, the harness's background writes: started before the first claim
(it replays what an earlier process spooled), drained when the loop ends.

    h.worker([agent]).run()                         # in your own process
    python -m trellis.harness.worker app.main:h     # every agent wrapped by that Harness
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Sequence
from typing import TYPE_CHECKING

from trellis.contracts import ConfigurationError
from trellis.harness.writes import DRAIN_SECONDS
from trellis.runs import Job
from trellis.runs import Worker as ClaimLoop

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.harness import Harness
    from trellis.harness.result import Result


class Worker:
    """Claims the queued runs of ``agents`` and executes each with its agent, ``concurrency``
    at a time (else ``TRELLIS_WORKER_CONCURRENCY``, else ``trellis.runs``' default: the CPU
    count from 1 to 8)."""

    def __init__(
        self, harness: Harness, agents: Sequence[Agent], *, concurrency: int | None = None
    ) -> None:
        self.harness = harness
        self.agents = {agent.id: agent for agent in agents}
        chosen = harness.settings.worker_concurrency if concurrency is None else concurrency
        try:
            #: the claim loop: claims, leases, heartbeats, the graceful stop
            self.loop = ClaimLoop(harness.runs, self._execute, [*self.agents], concurrency=chosen)
        except ValueError as exc:  # no agent, or a concurrency under one
            raise ConfigurationError(str(exc)) from exc

    @property
    def worker_id(self) -> str:
        return self.loop.worker_id

    @property
    def concurrency(self) -> int:
        return self.loop.concurrency

    def stop(self) -> None:
        """Stop claiming and let the runs held finish (at most ``trellis.runs``'
        ``GRACE_SECONDS``, 25 s); a second call releases them at once. ``run`` and ``serve``
        return when they are done."""
        self.loop.stop()

    async def run(self) -> None:
        """Claim and execute until stopped (or cancelled: the runs held then stop, writing
        nothing — their leases lapse, and agent-runs queues them again)."""
        await self._writing(self.loop.run())

    async def serve(self) -> None:
        """:meth:`run`, stopped gracefully by SIGTERM or SIGINT (a second signal releases the
        runs held at once)."""
        await self._writing(self.loop.serve())

    async def run_once(self) -> bool:
        """Claim one run and execute it to its end or pause. ``False`` when none was queued."""
        return await self.loop.run_once()

    async def _execute(self, job: Job) -> Result:
        """The claimed run's next attempt, as the worker holding its lease."""
        return await self.agents[job.record.agent_id].execute(job)

    async def _writing(self, loop: Awaitable[None]) -> None:
        self.harness.writes.start()  # replays what an earlier process could not deliver
        try:
            await loop
        finally:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(DRAIN_SECONDS):
                    await self.harness.writes.drain()


__all__ = ["Worker"]
