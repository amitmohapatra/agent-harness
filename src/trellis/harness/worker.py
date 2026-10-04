"""The worker: claims queued runs of its agents from agent-runs and executes them.

A claimed run is leased to this worker; a heartbeat extends the lease while the run
executes, and a worker that dies lets the lease lapse, after which agent-runs queues the
run again (the next attempt, which the journal makes idempotent). Schedules and resumed
durable runs arrive the same way: as queued runs.

An idle worker asks again after a growing pause (exponential, jittered, at most
:data:`IDLE_MAX_SECONDS`), and asks at once again after it got work. ``stop()`` (what
``python -m trellis.worker`` calls on SIGTERM or SIGINT) stops claiming and lets the runs it
holds finish for up to :data:`GRACE_SECONDS`; a run still going then is released — stopped
without writing anything, so its lease lapses and another worker runs it again.

    h.worker([agent]).run()                 # in your own process
    python -m trellis.worker app.main:h     # every agent wrapped by that Harness
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import socket
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from trellis.contracts import ConfigurationError, RunRecord
from trellis.harness.clients.runs import LeaseLost
from trellis.harness.pipeline import RELEASED
from trellis.harness.writes import DRAIN_SECONDS

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.worker")

#: The most runs one worker executes at once by default (the CPU count, at most this).
MAX_DEFAULT_CONCURRENCY: Final = 8
#: How long a claim is leased; the heartbeat renews it at a third of that.
LEASE_SECONDS: Final = 60.0
#: An idle worker's first pause before asking for work again; doubled while the queue stays
#: empty, up to :data:`IDLE_MAX_SECONDS` (equal jitter: half of it fixed, half random).
IDLE_SECONDS: Final = 0.5
IDLE_MAX_SECONDS: Final = 10.0
#: How long a stopping worker lets the runs it holds finish before it releases them.
GRACE_SECONDS: Final = 25.0


def default_concurrency() -> int:
    """Runs one worker executes at once when nothing says: the CPU count, from 1 to 8."""
    return max(1, min(MAX_DEFAULT_CONCURRENCY, os.cpu_count() or 1))


class Worker:
    def __init__(
        self, harness: Harness, agents: Sequence[Agent], *, concurrency: int | None = None
    ) -> None:
        if not agents:
            raise ConfigurationError("a worker needs at least one agent")
        chosen = concurrency or harness.settings.worker_concurrency or default_concurrency()
        if chosen < 1:
            raise ConfigurationError("a worker executes at least one run at a time")
        self.harness = harness
        self.agents = {agent.id: agent for agent in agents}
        self.concurrency = chosen
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{id(self):x}"
        self._stopping = asyncio.Event()
        self._hurry = asyncio.Event()
        #: the executions this worker holds, by run id (what a stop releases)
        self._held: dict[str, asyncio.Task[object]] = {}

    def stop(self) -> None:
        """Stop claiming and let the runs held finish (at most :data:`GRACE_SECONDS`); a
        second call releases them at once. ``run`` returns when they are done."""
        if self._stopping.is_set():
            self._hurry.set()
        self._stopping.set()

    async def run(self) -> None:
        """Claim and execute until stopped (or cancelled: the runs held are then cancelled
        and end ``CANCELLED``); ``concurrency`` runs at a time."""
        self.harness.writes.start()  # replays what an earlier process could not deliver
        slots = asyncio.Semaphore(self.concurrency)
        running: set[asyncio.Task[None]] = set()
        idle = 0
        try:
            while await self._slot(slots):
                record = await self._claim()
                if record is None:
                    slots.release()
                    idle += 1
                    await self._idle(idle)
                    continue
                idle = 0
                task = asyncio.create_task(self._execute(record))
                running.add(task)
                task.add_done_callback(running.discard)
                task.add_done_callback(lambda _: slots.release())
            await self._wind_down(running)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(DRAIN_SECONDS):
                    await self.harness.writes.drain()

    async def run_once(self) -> bool:
        """Claim one run and execute it to its end or pause. ``False`` when none was queued."""
        record = await self._claim()
        if record is None:
            return False
        await self._execute(record)
        return True

    # ------------------------------------------------------------------ internals
    async def _slot(self, slots: asyncio.Semaphore) -> bool:
        """A free slot, or ``False`` once the worker is told to stop."""
        if self._stopping.is_set():
            return False
        if not slots.locked():
            await slots.acquire()
            return True
        acquiring = asyncio.ensure_future(slots.acquire())
        stopping = asyncio.ensure_future(self._stopping.wait())
        try:
            await asyncio.wait({acquiring, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            got = acquiring.done()
            for waiter in (acquiring, stopping):
                waiter.cancel()
        if self._stopping.is_set():
            if got:  # a slot freed as the stop came: give it back
                slots.release()
            return False
        return True

    async def _idle(self, rounds: int) -> None:
        """Wait before asking again: longer each empty round, cut short by a stop."""
        ceiling = min(IDLE_MAX_SECONDS, IDLE_SECONDS * 2 ** min(rounds - 1, 16))
        delay = ceiling / 2 + random.uniform(0, ceiling / 2)  # jitter, not a secret
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(delay):
                await self._stopping.wait()

    async def _wind_down(self, running: set[asyncio.Task[None]]) -> None:
        """Let the runs held finish within the grace period, then release the rest."""
        if running:
            log.info(
                "worker %s stopping: %d run(s) in flight, %.0f s to finish",
                self.worker_id,
                len(running),
                GRACE_SECONDS,
            )
            hurry = asyncio.ensure_future(self._hurry.wait())
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(GRACE_SECONDS):
                    while running and not hurry.done():
                        await asyncio.wait({*running, hurry}, return_when=asyncio.FIRST_COMPLETED)
            hurry.cancel()
        for run_id, execution in list(self._held.items()):
            log.warning("run %s released: its lease lapses and another worker runs it", run_id)
            execution.cancel(RELEASED)
        await asyncio.gather(*running, return_exceptions=True)

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
        self._held[record.run_id] = execution
        heartbeat = asyncio.create_task(self._heartbeat(record.run_id, execution))
        try:
            await execution
        except asyncio.CancelledError:
            if not execution.cancelled():
                raise
        except Exception:
            log.exception("run %s failed in the worker", record.run_id)
        finally:
            self._held.pop(record.run_id, None)
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
