"""Where a chat run's AG-UI events wait for whoever reads them: the response that started the
run, and any client that reconnects to it later.

Every run served here gets a buffer of its translated events, numbered from 0 across all its
attempts (a resumed run keeps counting), so a client that lost its connection asks for what
came after the last number it saw. Bounded both ways: a run keeps its last
:data:`MAX_EVENTS_PER_RUN` events, and past :data:`MAX_RUNS` runs the least recently used
finished run is forgotten. Publishing is O(1).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from typing import Final

from trellis.harness.surfaces.agui.events import AGUIEvent

MAX_EVENTS_PER_RUN: Final = 2048
MAX_RUNS: Final = 256


class RunBuffer:
    """One run's events, and the readers waiting on the next one."""

    __slots__ = ("_changed", "events", "finished", "published", "user")

    def __init__(self, user: str) -> None:
        self.user = user
        self.events: deque[AGUIEvent] = deque(maxlen=MAX_EVENTS_PER_RUN)
        #: how many events were ever published: the next event's number
        self.published = 0
        self.finished = False
        self._changed = asyncio.Event()

    def publish(self, event: AGUIEvent, *, final: bool = False) -> None:
        """Append an event (also after the run finished: a late warning is kept)."""
        self.events.append(event)
        self.published += 1
        self.finished = self.finished or final
        self._changed.set()
        self._changed = asyncio.Event()

    def reopen(self) -> None:
        """A resumed run: readers wait for its next attempt again."""
        self.finished = False

    async def read(self, after: int) -> AsyncIterator[tuple[int, AGUIEvent]]:
        """``(number, event)`` for every event numbered above ``after``, live until the run
        finishes. Events that fell out of the buffer are skipped, not invented."""
        next_number = after + 1
        while True:
            oldest = self.published - len(self.events)
            for number in range(max(next_number, oldest), self.published):
                yield number, self.events[number - oldest]
            next_number = max(next_number, self.published)
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
