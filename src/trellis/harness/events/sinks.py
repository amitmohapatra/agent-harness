"""Sinks a harness can be given: collect (tests and the SSE bridge), compose, filter."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
from typing import Any

from trellis.contracts.runs import RunEvent, RunEventType

from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)


#: How many runs a collecting sink remembers, and how many events per run.
MAX_RUNS = 256
MAX_EVENTS_PER_RUN = 2000

RunKey = tuple[str, str]  # (tenant_id, run_id): a run id alone is caller-chosen material


class CollectingEventSink:
    """Keeps each run's recent events and fans them out to the run's subscribers.

    Runs are keyed by tenant *and* run id, so a caller naming another tenant's run never
    hears it. ``subscribe`` returns a queue that receives the run's events as they happen
    and ``None`` once the run is over (a ``RUN_FINISHED`` event), which is what a streaming
    surface waits on; events published before a subscriber arrived are replayed to it
    unless it asks for new events only. Memory is bounded: the oldest runs and the oldest
    events of a run are dropped past the caps.
    """

    name = "collecting"

    def __init__(self, *, max_runs: int = MAX_RUNS, max_events: int = MAX_EVENTS_PER_RUN) -> None:
        self._runs: OrderedDict[RunKey, deque[RunEvent]] = OrderedDict()
        self._subscribers: dict[RunKey, list[asyncio.Queue[RunEvent | None]]] = {}
        self.max_runs = max_runs
        self.max_events = max_events

    @property
    def events(self) -> list[RunEvent]:
        """Every remembered event, oldest run first (tests read this)."""
        return [event for run in self._runs.values() for event in run]

    async def publish(self, event: RunEvent) -> None:
        key = (event.tenant_id, event.run_id)
        run = self._runs.get(key)
        if run is None:
            run = self._runs[key] = deque(maxlen=self.max_events)
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)
        run.append(event)
        for queue in self._subscribers.get(key, []):
            queue.put_nowait(event)
            if event.type is RunEventType.RUN_FINISHED:
                queue.put_nowait(None)

    def subscribe(
        self, tenant_id: str, run_id: str, *, replay: bool = True
    ) -> asyncio.Queue[RunEvent | None]:
        key = (tenant_id, run_id)
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        if replay:
            for event in self._runs.get(key, ()):
                queue.put_nowait(event)
                if event.type is RunEventType.RUN_FINISHED:
                    queue.put_nowait(None)
        self._subscribers.setdefault(key, []).append(queue)
        return queue

    def unsubscribe(
        self, tenant_id: str, run_id: str, queue: asyncio.Queue[RunEvent | None]
    ) -> None:
        key = (tenant_id, run_id)
        queues = self._subscribers.get(key, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(key, None)

    def for_run(self, run_id: str, tenant_id: str | None = None) -> list[RunEvent]:
        return [
            event
            for (tenant, run), events in self._runs.items()
            if run == run_id and (tenant_id is None or tenant == tenant_id)
            for event in events
        ]

    def types(self, run_id: str | None = None) -> list[RunEventType]:
        events = self.for_run(run_id) if run_id else self.events
        return [e.type for e in events]

    def clear(self) -> None:
        self._runs.clear()


class CompositeEventSink:
    """Publishes to several sinks; one that fails does not stop the others."""

    name = "composite"

    def __init__(self, sinks: Iterable[Any]) -> None:
        self.sinks: Sequence[Any] = list(sinks)

    async def publish(self, event: RunEvent) -> None:
        for sink in self.sinks:
            try:
                await sink.publish(event)
            except Exception as exc:
                log.warning("run_event.sink_failed", sink=type(sink).__name__, error=str(exc))


class FilteringEventSink:
    """Forwards only the event types a sink cares about."""

    name = "filtering"

    def __init__(self, sink: Any, types: Iterable[RunEventType | str]) -> None:
        self.sink = sink
        self.types = frozenset(RunEventType(t) for t in types)

    async def publish(self, event: RunEvent) -> None:
        if event.type in self.types:
            await self.sink.publish(event)
