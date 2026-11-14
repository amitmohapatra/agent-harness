"""A conversation's line in the run store: its runs answered one at a time, in the order they
arrived, on whichever process each arrives (ADR 0005).

A turn of a conversation that a process executes outside the queue — ``agent.run``, ``stream``,
``serve_chat``, ``serve_a2a``, or a run continued after its answer — is recorded ``RUNNING``
with the conversation's ``concurrency_key`` (``thread:<conversation>``, the key a queued turn
gets), and held while it waits and while it works (``holding``). Its place in the line
(:func:`placed`) is when it took its place: its ``created_at``, or the ``recorded_at`` of its
latest answer — set by agent-runs, which places a run of a key after every run of that key
already running or queued, in the order they commit. A queued turn (``start``) holds its place
the same way while it waits for a worker, and a worker claims it only once no run placed before
it runs (agent-runs' claim): a stream of messages cannot keep it waiting.

A run takes its turn once no run of the same agent and conversation that is ``RUNNING`` or
``QUEUED`` in the store is placed before it (:meth:`Line.turn`): a run that ended, paused for a
person or whose process died (its hold lapsed, and the store ended it ``holder_lost``) has left
the line. A run that left the line before its turn came — cancelled while it waited, here or on
another process — takes no turn (:meth:`Line.turn` says so). A run waits for the run just ahead
of it: one of this process until it leaves the line here; one elsewhere (another process's, a
worker's) by following its event log, which says when it ends or pauses (agent-runs' event stream),
reading the line again then — and at the latest every :data:`STILL_SECONDS`, or after a pause
growing with each failure when the stream cannot be followed. A store with no event log
(``LocalRuns``: one process) is read again every :data:`POLL_SECONDS`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

from trellis.contracts import RunEventType, RunRecord, RunStatus
from trellis.harness.bounded import LRU
from trellis.harness.runlog import event_log
from trellis.runs import RunsError

if TYPE_CHECKING:
    from trellis.harness.runs import RunStore

#: How often a run waiting behind another reads the line again where the store keeps no event
#: log to follow (``LocalRuns``, in this process).
POLL_SECONDS: Final = 0.5
#: The longest a run following the event log of the run ahead goes without reading the line
#: again (a pause whose event never reached the log).
STILL_SECONDS: Final = 30.0
#: The pause before the line is read again when the run ahead's event log cannot be followed:
#: this, doubling with each failure in a row, up to :data:`DROPPED_MAX_SECONDS`, jittered.
DROPPED_SECONDS: Final = 0.5
DROPPED_MAX_SECONDS: Final = 10.0
#: The statuses of the runs in a line.
IN_LINE: Final = (RunStatus.RUNNING, RunStatus.QUEUED)

#: A run's place: when it took it, then its id (two placed at once).
Place = tuple[datetime, str]


@dataclass(frozen=True, slots=True)
class Seen:
    """A run as the line read it: its ``updated_at`` then (a run that moved is read again),
    its place (``None``: outside the line — another user's thread named alike, a child run)
    and its attempt (the one whose end its event log says)."""

    updated_at: datetime
    place: Place | None
    attempt: int


#: The runs a process's lines read, shared by its waiting runs (each read once).
Cache = LRU[str, Seen]


async def placed(runs: RunStore, record: RunRecord) -> datetime:
    """When the run ``record`` took its place, as the run store says: its ``created_at``,
    or the ``recorded_at`` of its latest answer (it took its place again then)."""
    place = record.created_at
    if record.last_resolution is None:
        return place
    cursor: str | None = None
    while True:
        page = await runs.resolutions(record.run_id, cursor=cursor, tenant=record.tenant_id)
        place = max([place, *(entry.recorded_at for entry in page.items)])
        if (cursor := page.next_cursor) is None:
            return place


@dataclass(slots=True)
class Line:
    """The line the run ``record`` waits in: the ``RUNNING`` and ``QUEUED`` runs of its agent
    and conversation (its ``concurrency_key``) in the run store, by their places. ``seen`` is
    what this process's lines read of the runs (``Turns.seen``)."""

    runs: RunStore
    record: RunRecord
    seen: Cache = field(default_factory=LRU)
    #: how far each run ahead's event log was read (its position)
    _read: dict[str, int] = field(default_factory=dict)
    #: the failures in a row to follow an event log
    _dropped: int = 0

    async def ahead(self) -> list[str] | None:
        """The runs placed before this one, the first first (the run holding the turn); ``None``
        once this one no longer runs (it left the line)."""
        record = self.record
        listed = [
            summary
            for status in IN_LINE
            async for summary in self.runs.iterate(
                agent_id=record.agent_id,
                thread_id=record.thread_id,
                status=status,
                tenant=record.tenant_id,
            )
        ]
        if not any(s.run_id == record.run_id and s.status is RunStatus.RUNNING for s in listed):
            return None
        if len(listed) == 1:
            return []
        places: dict[str, Place | None] = {}
        for summary in listed:
            seen = self.seen.get(summary.run_id)
            if seen is None or seen.updated_at != summary.updated_at:
                seen = await self._seen(summary.run_id, summary.updated_at)
            places[summary.run_id] = seen.place
        mine = places.pop(record.run_id)
        if mine is None:
            return []
        return [run for _, run in sorted(p for p in places.values() if p is not None and p < mine)]

    async def turn(self, leaving: Mapping[str, asyncio.Event], poked: asyncio.Event) -> bool:
        """Wait until no run is placed before this one: ``True``, its turn; ``False``, it left
        the line first (it no longer runs). It waits for the run just ahead of it to leave the
        line: one of this process (``leaving``: each run here, and the event set as it leaves)
        until that event; one elsewhere, by following its event log. ``poked``: look at the line
        again now (it may have been cancelled)."""
        while True:
            poked.clear()
            ahead = await self.ahead()
            if ahead is None:
                return False
            if not ahead:
                return True
            here = leaving.get(ahead[-1])
            waits = [
                asyncio.ensure_future(poked.wait()),
                asyncio.ensure_future(here.wait() if here is not None else self._left(ahead[-1])),
            ]
            try:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiting in waits:
                    waiting.cancel()
                await asyncio.gather(*waits, return_exceptions=True)

    async def _left(self, run_id: str) -> None:
        """Until the run ``run_id``, ahead in the line elsewhere, may have left it: its event log
        says its attempt ended (or paused), or the run ended; at the latest
        :data:`STILL_SECONDS`. Where the store keeps no event log, :data:`POLL_SECONDS`."""
        log = event_log(self.runs)
        if log is None:
            await asyncio.sleep(POLL_SECONDS)
            return
        seen = self.seen.get(run_id)
        attempt = 1 if seen is None else seen.attempt
        tenant = self.record.tenant_id
        try:
            async with asyncio.timeout(STILL_SECONDS):
                after = self._read.get(run_id, 0)
                async for entry in log.stream_events(run_id, after=after, tenant=tenant):
                    self._read[run_id] = entry.position
                    event = entry.event
                    if event.type is RunEventType.RUN_FINISHED and event.attempt >= attempt:
                        break
            self._dropped = 0
        except TimeoutError:
            self._dropped = 0
        except RunsError:  # the log cannot be followed now: read the line again after a pause
            pause = min(DROPPED_MAX_SECONDS, DROPPED_SECONDS * 2**self._dropped)
            self._dropped += 1
            await asyncio.sleep(pause * random.uniform(0.5, 1.0))  # jittered

    async def _seen(self, run_id: str, updated_at: datetime) -> Seen:
        found = await self.runs.get(run_id, tenant=self.record.tenant_id)
        if (
            found is None
            or found.status not in IN_LINE
            or found.concurrency_key != self.record.concurrency_key
        ):
            seen = Seen(updated_at, None, 1)
        else:
            seen = Seen(updated_at, (await placed(self.runs, found), run_id), found.attempt)
        self.seen[run_id] = seen
        return seen


__all__ = [
    "DROPPED_MAX_SECONDS",
    "DROPPED_SECONDS",
    "IN_LINE",
    "POLL_SECONDS",
    "STILL_SECONDS",
    "Cache",
    "Line",
    "Place",
    "Seen",
    "placed",
]
