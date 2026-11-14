"""Where a chat run's AG-UI events wait for whoever reads them: the response that started the
run, and any client that reconnects to it later.

Every run served here gets a buffer of its translated events, each numbered by where its
harness event is in the run's log (:func:`number_of`: its attempt and its sequence in it), so
a client that lost its connection asks for what came after the last number it saw — of this
replica or of any other: a reconnect that reaches another replica reads the run's log there
and numbers each event the same, so it neither repeats nor skips one. An event derived from
the one before it (the activity line after a tool's result) shares that event's number.
Bounded both ways: a run keeps its last :data:`MAX_EVENTS_PER_RUN` events, and past
:data:`MAX_RUNS` runs the least recently used finished run is forgotten. Publishing is O(1).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from typing import Final

from trellis.contracts import RunEvent
from trellis.harness.agui.activity import Activity
from trellis.harness.agui.events import AGUIEvent

MAX_EVENTS_PER_RUN: Final = 2048
MAX_RUNS: Final = 256
#: How many events one attempt of a run may number (its events' ``sequence``): an event's
#: number is its attempt's place times this, plus its sequence (2**32: room for 2**21 attempts
#: within the integers a JavaScript client holds exactly).
ATTEMPT_EVENTS: Final = 2**32


def number_of(event: RunEvent) -> int:
    """A harness event's number on the AG-UI streams (an SSE ``id``): where it is in the run's
    log — its attempt, then its sequence in it, as agent-runs orders the log — the same on the
    replica that serves the run and on any a client reconnects to, in every attempt."""
    return (event.attempt - 1) * ATTEMPT_EVENTS + event.sequence


class RunBuffer:
    """One run's events, and the readers waiting on the next one."""

    __slots__ = ("_changed", "activity", "appended", "events", "finished", "last", "user")

    def __init__(self, user: str) -> None:
        self.user = user
        #: the run's activity line, as its calls end in any of its attempts served here
        self.activity = Activity()
        #: the last events, each with its number
        self.events: deque[tuple[int, AGUIEvent]] = deque(maxlen=MAX_EVENTS_PER_RUN)
        #: the number of the latest event published (-1: none yet)
        self.last = -1
        #: how many events were ever appended
        self.appended = 0
        self.finished = False
        self._changed = asyncio.Event()

    def publish(self, event: AGUIEvent, number: int, *, final: bool = False) -> None:
        """Append an event under ``number`` (:func:`number_of`, the harness event's it is
        translated or derived from) — also after the run finished: a late warning is kept."""
        self.events.append((number, event))
        self.appended += 1
        self.last = max(self.last, number)
        self.finished = self.finished or final
        self._changed.set()
        self._changed = asyncio.Event()

    def finish(self) -> None:
        """The run's attempt returned: its readers' streams end (its events are all in)."""
        self.finished = True
        self._changed.set()
        self._changed = asyncio.Event()

    def reopen(self) -> None:
        """A resumed run: readers wait for its next attempt again."""
        self.finished = False

    async def read(self, after: int) -> AsyncIterator[tuple[int, AGUIEvent]]:
        """``(number, event)`` for every event numbered above ``after``, live until the run
        finishes. Events that fell out of the buffer are skipped, not invented."""
        index = 0  # the next event's place among all those ever appended
        while True:
            oldest = self.appended - len(self.events)
            for place in range(max(index, oldest), self.appended):
                number, event = self.events[place - oldest]
                if number > after:
                    yield number, event
            index = max(index, self.appended)
            if self.finished:
                return
            await self._changed.wait()


class Hub:
    """The buffers of the runs this surface served."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, RunBuffer] = OrderedDict()

    def get(self, run_id: str) -> RunBuffer | None:
        found = self._runs.get(run_id)
        if found is not None:
            self._runs.move_to_end(run_id)
        return found

    def open(self, run_id: str, user: str) -> RunBuffer:
        buffer = self._runs.get(run_id)
        if buffer is None:
            buffer = self._runs[run_id] = RunBuffer(user)
            self._evict()
        else:
            buffer.reopen()
        self._runs.move_to_end(run_id)
        return buffer

    def _evict(self) -> None:
        excess = len(self._runs) - MAX_RUNS
        for run_id in [r for r, b in self._runs.items() if b.finished][: max(excess, 0)]:
            del self._runs[run_id]
