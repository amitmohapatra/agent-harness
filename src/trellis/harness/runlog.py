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

Best-effort: an append agent-runs cannot be reached for (``DependencyUnavailableError``, its
client's own retries spent: a restart takes longer) is tried again, with backoff, for up to
:data:`UNREACHABLE_SECONDS` — the later events wait behind it, in order. One it refuses, or
still cannot take by then, drops those events with one warning (logged, counted
``trellis.run_events.undelivered``, and a ``warning`` event to whoever listens) — the run goes
on. A lost lease stops the log, and so does a run that no longer runs (cancelled from another
process): the attempt is told (``ended``), and one cancelled stops. A repeated append is safe:
agent-runs adds an event once per attempt and sequence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from trellis.contracts import RunEvent
from trellis.harness.telemetry import metrics
from trellis.runs import ConflictError, DependencyUnavailableError, LeaseLostError, RunEventEntry

if TYPE_CHECKING:
    from trellis.harness.events import RunEvents

log = logging.getLogger("trellis.run")

#: The most events one append carries (agent-runs takes 1 to 500).
BATCH: Final = 500
#: How long an append agent-runs cannot be reached for is tried again before its events are
#: dropped: long enough for agent-runs to restart.
UNREACHABLE_SECONDS: Final = 60.0
#: The first wait before such an append is tried again, doubled up to :data:`RETRY_MAX_SECONDS`.
RETRY_SECONDS: Final = 0.5
RETRY_MAX_SECONDS: Final = 5.0


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
        ended: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.tenant = tenant
        self.worker_id = worker_id
        self.events = events
        #: told when the log refuses the run's events because the run no longer runs (it was
        #: cancelled from another process, or ended by agent-runs' ticker)
        self.ended = ended
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

    def unsent(self) -> list[RunEvent]:
        """The events not in the log yet (at most one append's worth, oldest first): a
        progress save sends them with its checkpoint (``Runtime``), so a worker killed right
        after a step leaves both or neither. They stay queued here: sent again, agent-runs
        adds each once."""
        return [] if self.closed else self._pending[:BATCH]

    async def flush(self) -> None:
        """Wait until every event so far is in the log (or dropped)."""
        while self._sending is not None and not self._sending.done():
            await asyncio.shield(self._sending)

    async def close(self) -> None:
        """The attempt's last events are in the log: it takes no more."""
        await self.flush()
        self.closed = True

    async def _send(self) -> None:
        loop = asyncio.get_running_loop()
        down_since: float | None = None
        wait = RETRY_SECONDS
        while self._pending and not self.closed:
            batch = self._pending[:BATCH]
            try:
                await self.store.append_events(
                    self.run_id, batch, worker_id=self.worker_id, tenant=self.tenant
                )
            except DependencyUnavailableError as exc:
                down_since = loop.time() if down_since is None else down_since
                if loop.time() - down_since < UNREACHABLE_SECONDS:
                    await asyncio.sleep(wait)  # agent-runs is not there now: the batch waits
                    wait = min(wait * 2, RETRY_MAX_SECONDS)
                    continue
                self._dropped(len(batch), f"{type(exc).__name__}: {exc}")
            except LeaseLostError:
                self.closed = True  # another worker has the run: write nothing more
                self._pending.clear()
                return
            except ConflictError:
                self.closed = True  # the run no longer runs: the log takes nothing more
                self._pending.clear()
                if self.ended is not None:
                    await self.ended()
                return
            except Exception as exc:
                self._dropped(len(batch), f"{type(exc).__name__}: {exc}")
            del self._pending[: len(batch)]
            down_since, wait = None, RETRY_SECONDS

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
