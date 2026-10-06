"""A run's events in agent-runs' event log, so any replica streams any run.

With ``RUNS_URL`` set the run store is agent-runs' ``RunsClient``, which keeps each run's
events (``POST /v1/runs/{id}/events``; read with ``GET …/events`` or streamed from
``…/events/stream`` by any process). Every attempt then writes its events there as they
happen: :class:`RunLog` takes each one from the attempt's emitter (``RunEvents.record``) and
appends them in order, in batches, one append at a time, in the background — the run never
waits for it, except at its end: the pipeline :meth:`RunLog.flush`\\ es the attempt's last
events (``INTERRUPT``, ``RUN_FINISHED``...) before it records the pause or the ending, since
the log takes nothing after. A run kept in process (``LocalRuns``) has no log: its events go
to whoever listens, as always.

Best-effort: an append agent-runs refuses or cannot take (it already retried) drops those
events with one warning (logged, counted ``trellis.run_events.undelivered``, and a ``warning``
event to whoever listens) — the run goes on. A lost lease stops the log. A repeated append is
safe: agent-runs adds an event once per attempt and sequence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from trellis.contracts import RunEvent
from trellis.harness.telemetry import metrics
from trellis.runs import LeaseLostError, RunEventEntry

if TYPE_CHECKING:
    from trellis.harness.events import RunEvents

log = logging.getLogger("trellis.run")

#: The most events one append carries (agent-runs takes 1 to 500).
BATCH: Final = 500


@runtime_checkable
class EventLog(Protocol):
    """A run store that keeps runs' events: ``trellis.runs.RunsClient``."""

    async def append_events(
        self,
        run_id: str,
        events: Sequence[RunEvent],
        *,
        worker_id: str | None = ...,
        tenant: str | None = ...,
    ) -> Any: ...

    def stream_events(
        self, run_id: str, *, after: int = ..., tenant: str | None = ...
    ) -> AsyncIterator[RunEventEntry]: ...


class RunLog:
    """One attempt's events on their way to the run's log (a listener of its emitter)."""

    def __init__(
        self,
        store: EventLog,
        run_id: str,
        *,
        tenant: str,
        worker_id: str | None,
        events: RunEvents,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.tenant = tenant
        self.worker_id = worker_id
        self.events = events
        self._pending: list[RunEvent] = []
        self._sending: asyncio.Task[None] | None = None
        #: the log takes nothing more: the attempt ended, or the lease was lost
        self.closed = False
        self._warned = False

    def __call__(self, event: RunEvent) -> None:
        if self.closed:
            return
        self._pending.append(event)
        if self._sending is None or self._sending.done():
            self._sending = asyncio.get_running_loop().create_task(self._send())

    async def flush(self) -> None:
        """Wait until every event so far is in the log (or dropped)."""
        while self._sending is not None and not self._sending.done():
            await asyncio.shield(self._sending)

    async def close(self) -> None:
        """The attempt's last events are in the log: it takes no more."""
        await self.flush()
        self.closed = True

    async def _send(self) -> None:
        while self._pending and not self.closed:
            batch = self._pending[:BATCH]
            try:
                await self.store.append_events(
                    self.run_id, batch, worker_id=self.worker_id, tenant=self.tenant
                )
            except LeaseLostError:
                self.closed = True  # another worker has the run: write nothing more
                self._pending.clear()
                return
            except Exception as exc:
                self._dropped(len(batch), f"{type(exc).__name__}: {exc}")
            del self._pending[: len(batch)]

    def _dropped(self, count: int, why: str) -> None:
        metrics.events_undelivered(count)
        if self._warned:
            return
        self._warned = True
        message = f"run {self.run_id}: {count} event(s) not in agent-runs' event log: {why}"
        log.warning("%s", message)
        self.events.warning("events_undelivered", message)


def event_log(store: object) -> EventLog | None:
    """``store`` when it keeps runs' events (agent-runs), else ``None`` (in process)."""
    return store if isinstance(store, EventLog) else None


__all__ = ["BATCH", "EventLog", "RunLog", "event_log"]
