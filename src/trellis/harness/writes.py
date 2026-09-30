"""Side effects that must not hold a run: transcript and tool records, outcomes, verdicts.

A write is queued and the run moves on. A few workers drain the queue; a write that fails is
logged, counted, and reported to the run's event listeners as a ``warning`` event — never
silently dropped and never raised into the run.

Draining is automatic: when the event loop shuts down (``asyncio.run`` returning, a server
stopping) it cancels the workers, and a cancelled worker finishes the queue first, bounded by
:data:`DRAIN_SECONDS`. ``await harness.aclose()`` does the same explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from trellis.harness.events import RunEvents
from trellis.harness.telemetry import metrics

log = logging.getLogger("trellis.writes")

#: Workers draining the queue concurrently.
WRITERS: Final = 4
#: Pending writes held at most; past it a write is refused (and reported), never blocking.
MAX_PENDING: Final = 10_000
#: How long a shutdown waits for the queue to empty.
DRAIN_SECONDS: Final = 10.0

Work = Callable[[], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class _Item:
    label: str
    work: Work
    events: RunEvents | None


class Writes:
    """The queue and its workers, bound to the event loop that first used them."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_Item] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: list[asyncio.Task[None]] = []
        self.failed = 0

    def submit(self, label: str, work: Work, *, events: RunEvents | None = None) -> None:
        """Queue ``work``."""
        item = _Item(label, work, events)
        queue = self._ensure()
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            self._report(item, "the write queue is full")

    async def drain(self) -> None:
        """Wait until every queued write has been attempted."""
        if self._queue is not None and self._loop is asyncio.get_running_loop():
            await self._queue.join()

    async def aclose(self) -> None:
        await self.drain()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()
        self._queue = None

    # ------------------------------------------------------------------ internals
    def _ensure(self) -> asyncio.Queue[_Item]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._loop = loop
            self._queue = asyncio.Queue(MAX_PENDING)
            self._workers = [
                loop.create_task(self._work(self._queue), name=f"trellis-writes-{i}")
                for i in range(WRITERS)
            ]
        return self._queue

    async def _work(self, queue: asyncio.Queue[_Item]) -> None:
        item: _Item | None = None
        try:
            while True:
                item = await queue.get()
                await self._attempt(item)
                queue.task_done()
                item = None
        except asyncio.CancelledError:
            # The loop is going away: finish what was promised (the write that was cut off
            # included, writes are idempotent), within bounds.
            pending = [item] if item is not None else []
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                async with asyncio.timeout(DRAIN_SECONDS):
                    while pending or not queue.empty():
                        current = pending.pop() if pending else queue.get_nowait()
                        try:
                            await self._attempt(current)
                        finally:
                            queue.task_done()
            raise

    async def _attempt(self, item: _Item) -> None:
        try:
            await item.work()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._report(item, f"{type(exc).__name__}: {exc}")

    def _report(self, item: _Item, reason: str) -> None:
        self.failed += 1
        metrics.write_failed(item.label)
        log.warning("trellis write failed: %s: %s", item.label, reason)
        if item.events is not None:
            item.events.warning("write_failed", f"{item.label}: {reason}")
