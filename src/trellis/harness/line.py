"""A conversation's line in the run store: its runs answered one at a time, in the order they
arrived, on whichever process each arrives (ADR 0005).

A turn of a conversation that a process executes outside the queue — ``agent.run``, ``stream``,
``serve_chat``, ``serve_a2a``, or a run continued after its answer — is recorded ``RUNNING``
with the conversation's ``concurrency_key`` (``thread:<conversation>``, the key a queued turn
gets: a worker's claim waits while it runs), and held while it waits and while it works
(``holding``). Its place in the line (:func:`placed`) is when it started running: its
``created_at``, or the ``recorded_at`` of its latest answer — set by agent-runs, which places a
run of a key after every run of that key already running, in the order their starts commit.
It takes its turn once no run of the same agent and conversation that is ``RUNNING`` in the
store is placed before it (:meth:`Line.turn`): a run that ended, paused for a person or whose
process died (its hold lapsed, and the store ended it ``holder_lost``) has left the line. A
queued turn a worker claimed is placed when it was queued or last answered — before any run
started after its claim, since a claim waits while a run of its key is running. A queued turn
still waiting for a worker holds no place: it is claimed once no turn of the conversation runs.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

from trellis.contracts import RunRecord, RunStatus

if TYPE_CHECKING:
    from trellis.harness.runs import RunStore

#: How often a run waiting for a turn held elsewhere reads the line again (a turn ending in
#: this process wakes it at once).
POLL_SECONDS: Final = 0.5

#: A run's place: when it started running, then its id (two placed at once).
Place = tuple[datetime, str]


async def placed(runs: RunStore, record: RunRecord) -> datetime:
    """When the run ``record`` is started running, as the run store says: its ``created_at``,
    or the ``recorded_at`` of its latest answer (it continued then)."""
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
    """The line the run ``record`` waits in: the ``RUNNING`` runs of its agent and conversation
    (its ``concurrency_key``) in the run store, by their places."""

    runs: RunStore
    record: RunRecord
    #: each run read: its ``updated_at`` then (a run continued after an answer moves) and its
    #: place, ``None`` for one outside the line (another user's thread named alike, a child run)
    _seen: dict[str, tuple[datetime, Place | None]] = field(default_factory=dict)

    async def ahead(self) -> list[str]:
        """The runs placed before this one, the first first (the run holding the turn); none
        once this one no longer runs."""
        record = self.record
        listed = [
            summary
            async for summary in self.runs.iterate(
                agent_id=record.agent_id,
                thread_id=record.thread_id,
                status=RunStatus.RUNNING,
                tenant=record.tenant_id,
            )
        ]
        if all(summary.run_id == record.run_id for summary in listed):
            return []
        places: dict[str, Place | None] = {}
        for summary in listed:
            seen = self._seen.get(summary.run_id)
            if seen is None or seen[0] != summary.updated_at:
                seen = (summary.updated_at, await self._place(summary.run_id))
                self._seen[summary.run_id] = seen
            places[summary.run_id] = seen[1]
        mine = places.pop(record.run_id, None)
        if mine is None:
            return []
        return [run for _, run in sorted(p for p in places.values() if p is not None and p < mine)]

    async def turn(self, ended: Callable[[], asyncio.Event]) -> None:
        """Wait until no run is placed before this one. ``ended`` is the event a turn of the
        conversation ending in this process sets; a turn held elsewhere is read again every
        :data:`POLL_SECONDS`."""
        while True:
            woken = ended()  # taken first: a turn ending while the line is read wakes it
            if not await self.ahead():
                return
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(POLL_SECONDS):
                    await woken.wait()

    async def _place(self, run_id: str) -> Place | None:
        found = await self.runs.get(run_id, tenant=self.record.tenant_id)
        if (
            found is None
            or found.status is not RunStatus.RUNNING
            or found.concurrency_key != self.record.concurrency_key
        ):
            return None
        return (await placed(self.runs, found), run_id)


__all__ = ["POLL_SECONDS", "Line", "Place", "placed"]
