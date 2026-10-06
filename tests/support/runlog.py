"""A run store that keeps runs' events as agent-runs does (``POST``/``GET /v1/runs/{id}/events``,
``…/events/stream``): ``LocalRuns`` and an event log per run — appended only while the run is
``RUNNING``, fenced like a heartbeat, an event added once per attempt and sequence, positions
from 1 — so the harness's run log is driven exactly as against agent-runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from trellis.contracts import RunEvent, RunStatus
from trellis.harness.runs import LocalRuns
from trellis.runs import (
    ConflictError,
    EventsAppended,
    NotFoundError,
    RunEventEntry,
    ValidationError,
)

#: How often a stream looks for new events (agent-runs: every 0.5 s).
POLL_SECONDS = 0.005


class LoggedRuns(LocalRuns):
    def __init__(self) -> None:
        super().__init__()
        self.logs: dict[str, list[RunEventEntry]] = {}
        #: every append, as (run id, worker id, how many events)
        self.appends: list[tuple[str, str | None, int]] = []

    async def append_events(
        self,
        run_id: str,
        events: Sequence[RunEvent],
        *,
        worker_id: str | None = None,
        tenant: str | None = None,
    ) -> EventsAppended:
        record = self._fenced(run_id, worker_id, tenant)
        if record.status is not RunStatus.RUNNING or (worker_id is None and run_id in self._leases):
            raise ConflictError(f"run {run_id} takes no events now", code="CONFLICT", status=409)
        log = self.logs.setdefault(run_id, [])
        seen = {(entry.event.attempt, entry.event.sequence) for entry in log}
        added = 0
        for event in events:
            if event.run_id != run_id:
                raise ValidationError("an event of another run", code="VALIDATION", status=422)
            if (event.attempt, event.sequence) in seen:
                continue
            log.append(RunEventEntry(position=len(log) + 1, event=event))
            added += 1
        self.appends.append((run_id, worker_id, added))
        return EventsAppended(appended=added, position=len(log))

    async def events(
        self, run_id: str, *, after: int = 0, limit: int = 50, tenant: str | None = None
    ) -> list[RunEventEntry]:
        if await self.get(run_id, tenant=tenant) is None:
            raise NotFoundError(f"no run {run_id}", code="NOT_FOUND", status=404)
        return self.logs.get(run_id, [])[after : after + limit]

    async def stream_events(
        self, run_id: str, *, after: int = 0, tenant: str | None = None
    ) -> AsyncIterator[RunEventEntry]:
        while True:
            record = await self.get(run_id, tenant=tenant)
            if record is None:
                raise NotFoundError(f"no run {run_id}", code="NOT_FOUND", status=404)
            log = self.logs.get(run_id, [])
            for entry in log[after:]:
                after = entry.position
                yield entry
            if record.final and after >= len(log):
                return
            await asyncio.sleep(POLL_SECONDS)

    def logged(self, run_id: str) -> list[RunEvent]:
        return [entry.event for entry in self.logs.get(run_id, [])]
