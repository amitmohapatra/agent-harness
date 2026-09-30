"""Where runs and schedules are kept: agent-runs over HTTP, or this process when ``RUNS_URL``
is unset. Both behave the same way:

* a run is started ``RUNNING`` (someone runs it in process) or ``QUEUED`` (for a worker);
* a pause keeps the interrupt; a resume continues the run as its next attempt — ``QUEUED``
  again when it ever came from the queue, ``RUNNING`` for the process that resumes it — and
  a ``CANCEL`` ends it;
* a worker claims a queued run under a lease, heartbeats it, and names itself on the pause
  and the finish so a worker whose lease lapsed cannot write over another's run.

A pause also carries the run's checkpoint — its :class:`~trellis.harness.journal.Journal`, what
a re-run needs — which the store returns as ``RunRecord.checkpoint`` on every read and claim
until the run ends, so whichever worker resumes the run repeats no question and no side
effect. Writes raise when the store refuses or cannot be reached: a pause that was not
recorded cannot be resumed, so it is not reported.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import httpx
from croniter import croniter

from trellis.contracts import (
    AgentError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
    Schedule,
    ScheduleSpec,
    new_id,
)

#: How long one call to agent-runs may take before the run fails with it.
TIMEOUT_SECONDS: Final = 10.0
#: The header a platform key names the tenant it acts for with.
TENANT_HEADER: Final = "X-Trellis-Tenant"
NO_CONTENT: Final = 204
NOT_FOUND: Final = 404
CONFLICT: Final = 409
#: Schedules of one agent read when a name is taken (agent-runs' page limit).
SCHEDULES_PAGE: Final = 500
#: What a repeated ``schedule`` updates on the schedule of that name.
SCHEDULE_CHANGES: Final = {"cadence", "timezone", "input", "enabled", "metadata"}


class RunStoreError(RuntimeError):
    """The run store refused a write or could not be reached."""


class LeaseLost(RunStoreError):
    """The worker no longer holds the run: stop working it, and write nothing more."""


class Runs(Protocol):
    """The run store the harness drives: the contracts ``RunStore``, the worker queue, the
    inbox and schedules."""

    async def queued(self, start: RunStart) -> RunRecord: ...
    async def started(self, start: RunStart) -> RunRecord: ...
    async def paused(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> RunRecord: ...
    async def resumed(self, resolution: InterruptResolution) -> RunRecord: ...
    async def finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
        worker_id: str | None = None,
    ) -> RunRecord: ...
    async def get(self, run_id: str) -> RunRecord | None: ...
    async def list_paused(
        self, tenant_id: str, *, limit: int = 100, assignee: str | None = None
    ) -> Sequence[RunRecord]: ...
    async def claim(
        self, worker_id: str, agent_ids: Sequence[str], lease_seconds: float
    ) -> RunRecord | None: ...
    async def heartbeat(self, run_id: str, worker_id: str, lease_seconds: float) -> None: ...
    async def schedule(self, spec: ScheduleSpec) -> Schedule: ...
    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- agent-runs


class HttpRuns:
    """agent-runs 0.2 (``docs/api.md`` there). Bodies are the contracts' own models."""

    def __init__(
        self, base_url: str, api_key: str | None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        headers = {"X-Api-Key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=TIMEOUT_SECONDS
        )
        #: the tenant of every run this client wrote or read, so later calls can name it
        self._tenants: dict[str, str] = {}

    async def queued(self, start: RunStart) -> RunRecord:
        return await self._start(start, queue=True)

    async def started(self, start: RunStart) -> RunRecord:
        return await self._start(start, queue=False)

    async def _start(self, start: RunStart, *, queue: bool) -> RunRecord:
        body = {**start.model_dump(mode="json"), "queue": queue}
        return self._record(await self._send("POST", "/v1/runs", start.tenant_id, json=body))

    async def paused(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        body = {"interrupt": interrupt.model_dump(mode="json"), "checkpoint": checkpoint}
        return self._record(
            await self._send(
                "POST",
                f"/v1/runs/{interrupt.run_id}/pause",
                interrupt.tenant_id,
                json=body,
                params=_worker(worker_id),
            )
        )

    async def resumed(self, resolution: InterruptResolution) -> RunRecord:
        return self._record(
            await self._send(
                "POST",
                f"/v1/runs/{resolution.run_id}/resume",
                self._tenants.get(resolution.run_id),
                json=resolution.model_dump(mode="json"),
            )
        )

    async def finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        body: dict[str, Any] = {"status": status.value, "output": output}
        if error is not None:
            body["error"] = error.model_dump(mode="json")
        return self._record(
            await self._send(
                "POST",
                f"/v1/runs/{run_id}/finish",
                self._tenants.get(run_id),
                json=body,
                params=_worker(worker_id),
            )
        )

    async def get(self, run_id: str) -> RunRecord | None:
        response = await self._call("GET", f"/v1/runs/{run_id}", self._tenants.get(run_id))
        if response.status_code == NOT_FOUND:
            return None
        return self._record(self._body(response))

    async def list_paused(
        self, tenant_id: str, *, limit: int = 100, assignee: str | None = None
    ) -> Sequence[RunRecord]:
        params: dict[str, Any] = {"status": RunStatus.PAUSED.value, "limit": limit}
        if assignee is not None:
            params["assignee"] = assignee
        rows = await self._send("GET", "/v1/runs", tenant_id, params=params)
        return [self._record(row) for row in rows]

    async def claim(
        self, worker_id: str, agent_ids: Sequence[str], lease_seconds: float
    ) -> RunRecord | None:
        response = await self._call(
            "POST",
            "/v1/runs/claim",
            None,
            json={
                "worker_id": worker_id,
                "agent_ids": list(agent_ids),
                "lease_seconds": lease_seconds,
            },
        )
        if response.status_code == NO_CONTENT:
            return None
        return self._record(self._body(response)["run"])

    async def heartbeat(self, run_id: str, worker_id: str, lease_seconds: float) -> None:
        response = await self._call(
            "POST",
            f"/v1/runs/{run_id}/heartbeat",
            self._tenants.get(run_id),
            json={"worker_id": worker_id, "lease_seconds": lease_seconds},
        )
        if response.status_code == CONFLICT:
            raise LeaseLost(f"{worker_id} no longer holds {run_id}")
        self._body(response)

    async def schedule(self, spec: ScheduleSpec) -> Schedule:
        """Create the schedule, or update the one of the same name (a redeploy)."""
        created = await self._call(
            "POST", "/v1/schedules", spec.tenant_id, json=spec.model_dump(mode="json")
        )
        if created.status_code != CONFLICT:
            return Schedule.model_validate(self._body(created))
        listed = await self._send(
            "GET",
            "/v1/schedules",
            spec.tenant_id,
            params={"agent_id": spec.agent_id, "limit": SCHEDULES_PAGE},
        )
        existing = next((s for s in listed if s["name"] == spec.name), None)
        if existing is None:  # the name is taken by another agent's schedule
            raise RunStoreError(f"schedule name {spec.name!r} is taken: {created.text[:300]}")
        changes = spec.model_dump(mode="json", include=SCHEDULE_CHANGES)
        data = await self._send(
            "PATCH", f"/v1/schedules/{existing['schedule_id']}", spec.tenant_id, json=changes
        )
        return Schedule.model_validate(data)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ internals
    def _record(self, data: Any) -> RunRecord:
        record = RunRecord.model_validate(data)
        self._tenants[record.run_id] = record.tenant_id
        return record

    async def _send(self, method: str, path: str, tenant: str | None, **kwargs: Any) -> Any:
        return self._body(await self._call(method, path, tenant, **kwargs))

    async def _call(
        self, method: str, path: str, tenant: str | None, **kwargs: Any
    ) -> httpx.Response:
        headers = {TENANT_HEADER: tenant} if tenant else {}
        try:
            return await self._client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise RunStoreError(f"agent-runs unreachable: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _body(response: httpx.Response) -> Any:
        if response.is_error:
            error = LeaseLost if response.status_code == CONFLICT else RunStoreError
            raise error(
                f"agent-runs {response.request.method} {response.request.url.path}: "
                f"HTTP {response.status_code} {response.text[:300]}"
            )
        return response.json()


def _worker(worker_id: str | None) -> dict[str, str]:
    return {"worker_id": worker_id} if worker_id else {}


# --------------------------------------------------------------------------- in process


class LocalRuns:
    """The same store in this process, for development and tests: nothing survives a restart,
    only workers in this process claim, and a due schedule fires when a worker claims."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._queue: deque[str] = deque()
        #: runs that were ever queued: a resume sends them back to the queue
        self._queued: set[str] = set()
        self._leases: dict[str, tuple[str, datetime]] = {}
        self._schedules: dict[str, Schedule] = {}
        self._lock = asyncio.Lock()

    async def queued(self, start: RunStart) -> RunRecord:
        return self._start(start, RunStatus.QUEUED)

    async def started(self, start: RunStart) -> RunRecord:
        return self._start(start, RunStatus.RUNNING)

    def _start(self, start: RunStart, status: RunStatus) -> RunRecord:
        existing = self._runs.get(start.run_id)
        if existing is not None:  # idempotent on the run id
            return existing
        record = RunRecord.from_start(start, status=status)
        self._runs[record.run_id] = record
        if status is RunStatus.QUEUED:
            self._enqueue(record.run_id)
        return record

    async def paused(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        record = self._fenced(interrupt.run_id, worker_id)
        self._leases.pop(record.run_id, None)
        return self._move(record, RunStatus.PAUSED, awaiting=interrupt, checkpoint=checkpoint)

    async def resumed(self, resolution: InterruptResolution) -> RunRecord:
        record = self._require(resolution.run_id)
        if record.awaiting is None or not resolution.resolves(record.awaiting):
            raise RunStoreError(f"run {record.run_id} is not waiting on {resolution.interrupt_id}")
        if resolution.decision is InterruptDecision.CANCEL:
            return self._move(record, RunStatus.CANCELLED, last_resolution=resolution)
        requeue = record.run_id in self._queued
        moved = self._move(
            record,
            RunStatus.QUEUED if requeue else RunStatus.RUNNING,
            awaiting=None,
            last_resolution=resolution,
            attempt=record.attempt + 1,
        )
        if requeue:
            self._enqueue(moved.run_id)
        return moved

    async def finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        record = self._fenced(run_id, worker_id)
        self._leases.pop(run_id, None)
        return self._move(record, status, output=output, error=error)

    async def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def list_paused(
        self, tenant_id: str, *, limit: int = 100, assignee: str | None = None
    ) -> Sequence[RunRecord]:
        paused = [
            r
            for r in self._runs.values()
            if r.tenant_id == tenant_id
            and r.status is RunStatus.PAUSED
            and (assignee is None or (r.awaiting is not None and r.awaiting.assignee == assignee))
        ]
        return sorted(paused, key=lambda r: r.updated_at, reverse=True)[:limit]

    async def claim(
        self, worker_id: str, agent_ids: Sequence[str], lease_seconds: float
    ) -> RunRecord | None:
        async with self._lock:
            now = datetime.now(UTC)
            self._fire_due(now)
            self._expire_leases(now)
            wanted = set(agent_ids)
            for run_id in self._queue:
                record = self._runs[run_id]
                if record.agent_id in wanted and record.status is RunStatus.QUEUED:
                    self._queue.remove(run_id)
                    self._leases[run_id] = (worker_id, now + timedelta(seconds=lease_seconds))
                    return self._move(record, RunStatus.RUNNING)
            return None

    async def heartbeat(self, run_id: str, worker_id: str, lease_seconds: float) -> None:
        self._fenced(run_id, worker_id)
        self._leases[run_id] = (worker_id, datetime.now(UTC) + timedelta(seconds=lease_seconds))

    async def schedule(self, spec: ScheduleSpec) -> Schedule:
        existing = next((s for s in self._schedules.values() if s.name == spec.name), None)
        schedule = Schedule.from_spec(spec, next_fire_at=_next_fire(spec, datetime.now(UTC)))
        if existing is not None:
            schedule = schedule.model_copy(update={"schedule_id": existing.schedule_id})
        self._schedules[schedule.schedule_id] = schedule
        return schedule

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------ internals
    def _require(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise RunStoreError(f"no run {run_id}")
        return record

    def _fenced(self, run_id: str, worker_id: str | None) -> RunRecord:
        record = self._require(run_id)
        if worker_id is not None:
            held = self._leases.get(run_id)
            if held is None or held[0] != worker_id or record.status is not RunStatus.RUNNING:
                raise LeaseLost(f"{worker_id} no longer holds {run_id}")
        return record

    def _enqueue(self, run_id: str) -> None:
        self._queue.append(run_id)
        self._queued.add(run_id)

    def _move(self, record: RunRecord, status: RunStatus, **changes: Any) -> RunRecord:
        if record.status is not status and not record.status.can_become(status):
            raise RunStoreError(f"run {record.run_id}: {record.status} cannot become {status}")
        if status.final:  # an ending clears what the run waited on and would resume from
            changes.update(awaiting=None, checkpoint=None)
        moved = RunRecord.model_validate(
            {**record.model_dump(), **changes, "status": status, "updated_at": datetime.now(UTC)}
        )
        self._runs[moved.run_id] = moved
        return moved

    def _expire_leases(self, now: datetime) -> None:
        for run_id, (_, until) in list(self._leases.items()):
            if until < now:
                del self._leases[run_id]
                record = self._runs[run_id]
                self._move(record, RunStatus.QUEUED, attempt=record.attempt + 1)
                self._enqueue(run_id)

    def _fire_due(self, now: datetime) -> None:
        for schedule in list(self._schedules.values()):
            if not schedule.enabled or schedule.next_fire_at is None or schedule.next_fire_at > now:
                continue
            start = RunStart(
                run_id=new_id("run_"),
                tenant_id=schedule.tenant_id,
                agent_id=schedule.agent_id,
                user_id=schedule.on_behalf_of,
                on_behalf_of=schedule.on_behalf_of,
                workspace_id=schedule.workspace_id,
                input=schedule.input,
                metadata={"schedule_id": schedule.schedule_id},
            )
            record = self._start(start, RunStatus.QUEUED)
            self._schedules[schedule.schedule_id] = schedule.model_copy(
                update={
                    "last_fired_at": now,
                    "last_run_id": record.run_id,
                    "next_fire_at": _next_fire(schedule, now),
                }
            )


def _next_fire(spec: ScheduleSpec, after: datetime) -> datetime:
    local = after.astimezone(ZoneInfo(spec.timezone))
    return croniter(spec.cadence, local).get_next(datetime).astimezone(UTC)
