"""The worker: claims queued runs of its agents from agent-runs and executes them.

A claimed run is leased to this worker; a heartbeat extends the lease while the run
executes, and a worker that dies lets the lease lapse, after which agent-runs queues the
run again (the next attempt, which the journal makes idempotent). Schedules and resumed
durable runs arrive the same way: as queued runs.

    h.worker([agent]).run()                 # in your own process
    python -m trellis.worker app.main:h     # every agent wrapped by that Harness
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from trellis.contracts import ConfigurationError, RunRecord
from trellis.harness.clients.runs import LeaseLost

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.worker")

#: Runs one worker executes at once.
WORKER_CONCURRENCY: Final = 4
#: How long a claim is leased; the heartbeat renews it at a third of that.
LEASE_SECONDS: Final = 60.0
#: How long an idle worker waits before asking for work again.
IDLE_SECONDS: Final = 1.0


class Worker:
    def __init__(
        self, harness: Harness, agents: Sequence[Agent], *, concurrency: int = WORKER_CONCURRENCY
    ) -> None:
        if not agents:
            raise ConfigurationError("a worker needs at least one agent")
        self.harness = harness
        self.agents = {agent.id: agent for agent in agents}
        self.concurrency = concurrency
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{id(self):x}"

    async def run(self) -> None:
        """Claim and execute until cancelled; ``concurrency`` runs at a time."""
        slots = asyncio.Semaphore(self.concurrency)
        running: set[asyncio.Task[None]] = set()
        try:
            while True:
                await slots.acquire()
                record = await self._claim()
                if record is None:
                    slots.release()
                    await asyncio.sleep(IDLE_SECONDS)
                    continue
                task = asyncio.create_task(self._execute(record))
                running.add(task)
                task.add_done_callback(running.discard)
                task.add_done_callback(lambda _: slots.release())
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            await self.harness.writes.drain()

    async def run_once(self) -> bool:
        """Claim one run and execute it to its end or pause. ``False`` when none was queued."""
        record = await self._claim()
        if record is None:
            return False
        await self._execute(record)
        return True

    async def _claim(self) -> RunRecord | None:
        try:
            return await self.harness.runs.claim(self.worker_id, list(self.agents), LEASE_SECONDS)
        except Exception as exc:
            log.warning("claim failed: %s", exc)
            return None

    async def _execute(self, record: RunRecord) -> None:
        """Run it while the lease holds; a lost lease stops it without writing anything."""
        execution = asyncio.create_task(
            self.agents[record.agent_id]._claimed(record, self.worker_id)
        )
        heartbeat = asyncio.create_task(self._heartbeat(record.run_id, execution))
        try:
            await execution
        except asyncio.CancelledError:
            if not execution.cancelled():
                raise
        except Exception:
            log.exception("run %s failed in the worker", record.run_id)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, run_id: str, execution: asyncio.Task[object]) -> None:
        while True:
            await asyncio.sleep(LEASE_SECONDS / 3)
            try:
                await self.harness.runs.heartbeat(run_id, self.worker_id, LEASE_SECONDS)
            except LeaseLost:
                log.warning("lease on %s lost: stopping it", run_id)
                execution.cancel()
                return
            except Exception as exc:
                log.warning("heartbeat for %s failed: %s", run_id, exc)
