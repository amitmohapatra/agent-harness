"""A conversation's line: its runs answered one at a time, in the order they arrived, on
whichever process each arrives (ADR 0005). The one ordering of a conversation's turns: the run
store keeps the order, and ``conversation.Turns`` is its part in this process (which runs wait
or work here, and who wakes whom).

A turn of a conversation that a process executes outside the queue — ``agent.run``, ``stream``,
``serve_chat``, ``serve_a2a``, or a run continued after its answer — is recorded ``RUNNING``
with the conversation's ``concurrency_key`` (``thread:<conversation>``, the key a queued turn
gets), and held while it waits and while it works (``holding``). Its place in the line
(``runs.place``) is when it took it: its ``created_at``, or the ``recorded_at`` of its latest
answer — set by agent-runs, which places a run of a key after every run of that key already
running or queued, in the order they commit. A queued turn (``start``) holds its place the same
way while it waits for a worker, and a worker claims it only once no run placed before it runs
(the run store's claim): a stream of messages cannot keep it waiting.

A run takes its turn once no run of the same agent and conversation that is ``RUNNING`` or
``QUEUED`` in the store is placed before it (:meth:`Line.turn`): a run that ended, paused for a
person or whose process died (its hold lapsed, and the store ended it) has left the line. A run
cancelled while it waited, or ended in the store before its turn came, takes none. It waits for
the run just ahead of it: one of this process until it leaves the line here; one elsewhere
(another process's, a worker's) by following its event log, which says when its attempt ends or
pauses (agent-runs' event stream: one stream a run ahead, shared, :class:`Followers`), then
reads the line again — at the latest after :data:`STILL_SECONDS`. Only when that log cannot be
followed (the stream dropped, a store that keeps none, or :data:`MAX_FOLLOWED` streams open)
does it read the line again after a pause: :data:`DROPPED_SECONDS`, doubling with
each failure in a row up to :data:`DROPPED_MAX_SECONDS`, jittered.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from trellis.contracts import RunEventType, RunRecord, RunStatus
from trellis.harness.bounded import LRU
from trellis.harness.runlog import EventLog
from trellis.harness.runs import Place, RunStore, place
from trellis.runs import RunsError

#: The longest a run following the event log of the run ahead goes without reading the line
#: again (the log is best-effort: a pause whose event never reached it).
STILL_SECONDS: Final = 30.0
#: The pause before the line is read again when the run ahead's event log cannot be followed:
#: this, doubling with each failure in a row, up to :data:`DROPPED_MAX_SECONDS`, jittered.
DROPPED_SECONDS: Final = 0.05
DROPPED_MAX_SECONDS: Final = 5.0
#: The most runs ahead whose event logs one process follows at once (each an open stream): a
#: run waiting behind another one elsewhere reads the line again after a pause instead.
MAX_FOLLOWED: Final = 32
#: The statuses of the runs in a line.
IN_LINE: Final = (RunStatus.RUNNING, RunStatus.QUEUED)


#: The ``RunStart.metadata`` key a run sent with ``mode="interrupt"`` keeps, under it the run
#: that held its conversation's turn in another process: that process stops it (:meth:`Line.
#: interrupted`), and the message takes its turn after it.
INTERRUPTS: Final = "interrupts"


@dataclass(frozen=True, slots=True)
class Seen:
    """A run as the line read it: its ``updated_at`` then (a run that moved is read again),
    its place (``None``: outside the line — another user's thread named alike, a child run),
    its attempt (the one whose end its event log says) and the run it asks to interrupt."""

    updated_at: datetime
    place: Place | None
    attempt: int
    interrupts: str | None = None


#: The runs a process's lines read, shared by its waiting runs (each read once; bounded).
Cache = LRU[str, Seen]


def _leaving() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


@dataclass(eq=False, slots=True)
class Here:
    """A run in its conversation's line in this process, waiting its turn or holding it."""

    #: done as it leaves the line here (the run behind it looks at the line again)
    left: asyncio.Future[None] = field(default_factory=_leaving)
    #: set to have it look at the line again now
    again: asyncio.Event = field(default_factory=asyncio.Event)
    #: why it was cancelled while it waited its turn (it takes none)
    cancelled: str | None = None


class Followers:
    """The event logs this process follows of the runs ahead in its conversations' lines that
    work elsewhere: one stream a run, shared by every run waiting behind it here, at most
    :data:`MAX_FOLLOWED` at once — on ``log``, a client of its own where the harness builds its
    clients, so that streams never take the connections heartbeats and writes need."""

    def __init__(self, log: Callable[[], EventLog | None]) -> None:
        self._log = log
        self._following: dict[str, asyncio.Future[bool]] = {}
        #: how far each run's event log was read (its position)
        self._read: LRU[str, int] = LRU()

    async def left(self, run_id: str, attempt: int, tenant: str) -> bool:
        """Until the event log of run ``run_id`` says its ``attempt`` ended (or paused), the
        run ended, or :data:`STILL_SECONDS` passed; ``False`` when it cannot be followed now
        (no event log, the stream dropped, or :data:`MAX_FOLLOWED` are followed already)."""
        following = self._following.get(run_id)
        if following is None:
            log = self._log()
            if log is None or len(self._following) >= MAX_FOLLOWED:
                return False
            following = asyncio.ensure_future(self._followed(log, run_id, attempt, tenant))
            self._following[run_id] = following
            following.add_done_callback(lambda _: self._following.pop(run_id, None))
        return await asyncio.shield(following)  # (the stream is every waiter's)

    async def _followed(self, log: EventLog, run_id: str, attempt: int, tenant: str) -> bool:
        try:
            async with asyncio.timeout(STILL_SECONDS):
                async for entry in log.stream_events(
                    run_id, after=self._read.get(run_id) or 0, tenant=tenant
                ):
                    self._read[run_id] = entry.position
                    ended = entry.event.type is RunEventType.RUN_FINISHED
                    if ended and entry.event.attempt >= attempt:
                        break
        except TimeoutError:
            pass
        except RunsError:
            return False
        return True


async def placed(runs: RunStore, record: RunRecord) -> Place:
    """The place of the run ``record`` is (``runs.place``), its answers read from the store."""
    if record.last_resolution is None:
        return place(record, ())
    answers = []
    cursor: str | None = None
    while True:
        page = await runs.resolutions(record.run_id, cursor=cursor, tenant=record.tenant_id)
        answers += page.items
        if (cursor := page.next_cursor) is None:
            return place(record, answers)


@dataclass(slots=True)
class Line:
    """The line the run ``record`` waits in: the ``RUNNING`` and ``QUEUED`` runs of its agent
    and conversation (its ``concurrency_key``) in the run store, by their places. ``seen`` is
    what this process's lines read of the runs (``Turns.seen``), ``followers`` the event logs it
    follows (``Harness.followers``)."""

    runs: RunStore
    record: RunRecord
    seen: Cache
    followers: Followers
    #: the times in a row the event log of the run ahead could not be followed
    _dropped: int = 0

    async def ahead(self) -> list[str] | None:
        """The runs placed before this one, the first first (the run holding the turn); ``None``
        once this one no longer runs (it left the line)."""
        standing = await self._standing(*IN_LINE)
        mine = next((p for p in standing if p[1] == self.record.run_id), None)
        if mine is None:
            return None
        return [run for at, run in sorted(standing) if (at, run) < mine]

    async def holder(self) -> str | None:
        """The run holding the conversation's turn now — the first in its line — before this
        one is started; ``None`` when the line is empty."""
        standing = await self._standing(*IN_LINE)
        return min(standing)[1] if standing else None

    async def interrupted(self) -> bool:
        """Whether a run in the line asks to interrupt this one (:data:`INTERRUPTS`: a newer
        message of the conversation, sent to another process with ``mode="interrupt"``)."""
        standing = await self._standing(RunStatus.RUNNING)
        return any(seen.interrupts == self.record.run_id for seen in standing.values())

    async def _standing(self, *statuses: RunStatus) -> dict[Place, Seen]:
        """The runs in the line with one of ``statuses``, by their places (each run read once:
        :attr:`seen`)."""
        record = self.record
        standing: dict[Place, Seen] = {}
        for status in statuses:
            async for summary in self.runs.iterate(
                agent_id=record.agent_id,
                thread_id=record.thread_id,
                status=status,
                tenant=record.tenant_id,
            ):
                seen = self.seen.get(summary.run_id)
                if seen is None or seen.updated_at != summary.updated_at:
                    seen = await self._seen(summary.run_id, summary.updated_at)
                if seen.place is not None:
                    standing[seen.place] = seen
        return standing

    async def turn(self, here: Mapping[str, Here], mine: Here) -> bool:
        """Wait until no run is placed before this one: ``True``, its turn; ``False``, it takes
        none (cancelled while it waited, or it no longer runs). ``here`` are the conversation's
        runs in line in this process, ``mine`` this one among them: it waits for the run just
        ahead of it to leave the line — one of ``here`` until it leaves, one elsewhere by its
        event log (:meth:`_left`) — or to be told to look again."""
        while mine.cancelled is None:
            mine.again.clear()
            ahead = await self.ahead()
            if not ahead:
                return ahead is not None
            before = here.get(ahead[-1])
            elsewhere = None if before is not None else asyncio.ensure_future(self._left(ahead[-1]))
            again = asyncio.ensure_future(mine.again.wait())
            try:
                leaving = elsewhere if before is None else before.left
                await asyncio.wait({leaving, again}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                again.cancel()
                if elsewhere is not None:  # (a run's ``left`` is its own: all behind it wait on it)
                    elsewhere.cancel()
        return False

    async def _left(self, run_id: str) -> None:
        """Until the run ``run_id``, ahead in the line elsewhere, may have left it: its event
        log followed to its attempt's end (``Followers.left``); where it cannot be followed,
        after a pause that grows with each failure in a row."""
        seen = self.seen.get(run_id)
        attempt = 1 if seen is None else seen.attempt
        if await self.followers.left(run_id, attempt, self.record.tenant_id):
            self._dropped = 0
            return
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
            asks = found.metadata.get(INTERRUPTS)
            seen = Seen(updated_at, await placed(self.runs, found), found.attempt, asks)
        self.seen[run_id] = seen
        return seen


__all__ = [
    "DROPPED_MAX_SECONDS",
    "DROPPED_SECONDS",
    "INTERRUPTS",
    "IN_LINE",
    "MAX_FOLLOWED",
    "STILL_SECONDS",
    "Cache",
    "Followers",
    "Here",
    "Line",
    "Seen",
    "placed",
]
