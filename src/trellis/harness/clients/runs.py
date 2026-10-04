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
effect. Data too large for a question (an ``ask`` table or diff) is a run artifact, stored
beside the run and referenced from its interrupt (``payload_ref``). Writes raise when the store
refuses or cannot be reached: a pause that was not recorded cannot be resumed, so it is not
reported.

Every call to agent-runs is retried — up to :data:`RETRIES` times, with exponential backoff and
full jitter, or after the ``Retry-After`` the service sent (at most
:data:`RETRY_AFTER_MAX_SECONDS`) — when it failed on the way: a transport error, ``429``,
``502``, ``503``, ``504``. That is safe for every call: starting a run is idempotent on its id,
a repeated pause or finish from the same worker with the same status answers the stored record,
a repeated artifact is the same artifact, a repeated schedule is the one that exists. A refusal
is read from the service's problem document (RFC 9457) by its ``code``: ``LEASE_LOST`` is
:class:`LeaseLost`, ``CONFLICT`` :class:`Conflict`, ``NOT_FOUND`` :class:`NotFound`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import httpx
from croniter import croniter
from pydantic import BaseModel, ConfigDict

from trellis.contracts import (
    AgentError,
    ArtifactRef,
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
from trellis.contracts.ids import now

log = logging.getLogger("trellis.runs")

#: How long one call to agent-runs may take before it counts as failed (and is retried).
TIMEOUT_SECONDS: Final = 10.0
#: The header a platform key names the tenant it acts for with.
TENANT_HEADER: Final = "X-Trellis-Tenant"
NO_CONTENT: Final = 204
NOT_FOUND: Final = 404
CONFLICT: Final = 409
#: Retries of a call that failed on the way, after the first attempt.
RETRIES: Final = 3
#: The answers that mean "try again": too many requests, and a gateway or service in trouble.
RETRY_STATUSES: Final = frozenset({429, 502, 503, 504})
#: The backoff ceiling of the first retry, doubled for each one after it (full jitter: the
#: wait is uniform between 0 and the ceiling), and the highest ceiling.
BACKOFF_SECONDS: Final = 0.25
BACKOFF_MAX_SECONDS: Final = 5.0
#: The longest ``Retry-After`` honoured.
RETRY_AFTER_MAX_SECONDS: Final = 30.0
#: Paused runs one inbox page asks for (agent-runs' page limit), and the pages one inbox
#: read follows at most: past that the newest are returned and a warning is logged.
INBOX_LIMIT: Final = 500
INBOX_MAX_PAGES: Final = 10
#: What an artifact of JSON is sent as.
JSON_MIME: Final = "application/json"


class RunSummary(BaseModel):
    """A run as a listing shows it (agent-runs ``GET /v1/runs``): enough for an inbox; read
    the run for the rest."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    run_id: str
    agent_id: str
    status: RunStatus
    awaiting: Interrupt | None = None
    assignee: str | None = None
    deadline: datetime | None = None
    updated_at: datetime


class RunStoreError(RuntimeError):
    """The run store refused a call or could not be reached. ``code`` is the problem's
    (agent-runs' error code), ``status`` the HTTP status (``None`` without a response), and
    ``retryable`` whether the same call may succeed later — what a run's ``AgentError``
    keeps."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable


class LeaseLost(RunStoreError):
    """The worker no longer holds the run: stop working it, and write nothing more."""


class Conflict(RunStoreError):
    """The run (or schedule, or artifact) is not in a state that allows the call: a run id
    taken, an answer to an interrupt the run does not wait on, an illegal transition."""


class NotFound(RunStoreError):
    """The store has no such run (or artifact) for this tenant."""


#: The problem codes that have a class of their own; any other refusal is a RunStoreError.
ERRORS: Final[dict[str, type[RunStoreError]]] = {
    "LEASE_LOST": LeaseLost,
    "CONFLICT": Conflict,
    "NOT_FOUND": NotFound,
}
#: The class of a refusal without a ``code`` (a proxy's answer, an older service).
ERRORS_BY_STATUS: Final[dict[int, type[RunStoreError]]] = {404: NotFound, 409: Conflict}


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
    async def inbox(self, tenant_id: str, assignee: str | None) -> Sequence[RunSummary]: ...
    async def claim(
        self, worker_id: str, agent_ids: Sequence[str], lease_seconds: float
    ) -> RunRecord | None: ...
    async def heartbeat(self, run_id: str, worker_id: str, lease_seconds: float) -> None: ...
    async def schedule(self, spec: ScheduleSpec) -> Schedule: ...
    async def put_artifact(
        self, run_id: str, data: bytes, *, worker_id: str | None = None
    ) -> ArtifactRef: ...
    async def artifact(self, artifact_id: str, tenant_id: str) -> bytes | None: ...
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

    async def inbox(self, tenant_id: str, assignee: str | None) -> Sequence[RunSummary]:
        """Every paused run waiting on ``assignee``, newest first, page by page (the
        ``Link: rel="next"`` cursor agent-runs sends), at most :data:`INBOX_MAX_PAGES`."""
        params: dict[str, Any] = {"status": RunStatus.PAUSED.value, "limit": INBOX_LIMIT}
        if assignee is not None:
            params["assignee"] = assignee
        rows: list[Any] = []
        for _ in range(INBOX_MAX_PAGES):
            response = await self._call("GET", "/v1/runs", tenant_id, params=params)
            rows.extend(self._body(response))
            cursor = next_cursor(response.headers.get("link"))
            if cursor is None:
                break
            params = {**params, "cursor": cursor}
        else:
            log.warning(
                "the inbox of %s holds more than %d paused runs: the newest are returned",
                assignee or tenant_id,
                len(rows),
            )
        return [RunSummary.model_validate(row) for row in rows]

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
            # whatever its code: this worker does not hold a running lease on the run
            raise LeaseLost(f"{worker_id} no longer holds {run_id}", status=CONFLICT)
        self._body(response)

    async def schedule(self, spec: ScheduleSpec) -> Schedule:
        """Create the schedule; the same agent, person, cadence and input answer the one
        that exists (agent-runs upserts on them)."""
        data = await self._send(
            "POST", "/v1/schedules", spec.tenant_id, json=spec.model_dump(mode="json")
        )
        return Schedule.model_validate(data)

    async def put_artifact(
        self, run_id: str, data: bytes, *, worker_id: str | None = None
    ) -> ArtifactRef:
        """Store ``data`` (JSON) as an artifact of the run; the same bytes again answer the
        artifact already stored."""
        params = {**_worker(worker_id), "checksum": f"sha256:{hashlib.sha256(data).hexdigest()}"}
        body = await self._send(
            "POST",
            f"/v1/runs/{run_id}/artifacts",
            self._tenants.get(run_id),
            content=data,
            params=params,
            headers={"Content-Type": JSON_MIME},
        )
        return ArtifactRef.model_validate(body)

    async def artifact(self, artifact_id: str, tenant_id: str) -> bytes | None:
        response = await self._call("GET", f"/v1/artifacts/{artifact_id}", tenant_id)
        if response.status_code == NOT_FOUND:
            return None
        self._body_ok(response)
        return response.content

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
        self,
        method: str,
        path: str,
        tenant: str | None,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """One call, retried while it fails on the way (see the module's docstring)."""
        sent = {**(headers or {}), **({TENANT_HEADER: tenant} if tenant else {})}
        retry = 0
        while True:
            try:
                response = await self._client.request(method, path, headers=sent, **kwargs)
            except httpx.TransportError as exc:
                if retry == RETRIES:
                    raise RunStoreError(
                        f"agent-runs unreachable: {type(exc).__name__}: {exc}", retryable=True
                    ) from exc
                delay = backoff(retry, None)
            except httpx.HTTPError as exc:
                raise RunStoreError(f"agent-runs call failed: {type(exc).__name__}: {exc}") from exc
            else:
                if response.status_code not in RETRY_STATUSES or retry == RETRIES:
                    return response
                delay = backoff(retry, response.headers.get("retry-after"))
            retry += 1
            await _pause(delay)

    @classmethod
    def _body(cls, response: httpx.Response) -> Any:
        cls._body_ok(response)
        return response.json()

    @staticmethod
    def _body_ok(response: httpx.Response) -> None:
        if response.is_error:
            raise refusal(response)


def refusal(response: httpx.Response) -> RunStoreError:
    """The error an error response means: by the problem's ``code`` (else by the status),
    keeping its words and whether it may be retried."""
    try:
        body = response.json()
    except ValueError:
        body = None
    problem: dict[str, Any] = body if isinstance(body, dict) else {}
    status = response.status_code
    code = problem.get("code") if isinstance(problem.get("code"), str) else None
    cls = ERRORS.get(code, RunStoreError) if code else ERRORS_BY_STATUS.get(status, RunStoreError)
    retryable = problem.get("retryable")
    detail = problem.get("detail") or problem.get("title") or response.text[:300]
    return cls(
        f"agent-runs {response.request.method} {response.request.url.path}: "
        f"HTTP {status}{f' {code}' if code else ''}: {detail}",
        status=status,
        code=code,
        retryable=retryable if isinstance(retryable, bool) else status in RETRY_STATUSES,
    )


def backoff(retry: int, retry_after: str | None) -> float:
    """How long to wait before retry number ``retry + 1``: the service's ``Retry-After``
    (seconds or an HTTP date, at most :data:`RETRY_AFTER_MAX_SECONDS`) when it sent one,
    else full jitter under an exponentially growing ceiling."""
    asked = _retry_after(retry_after)
    if asked is not None:
        return min(asked, RETRY_AFTER_MAX_SECONDS)
    ceiling = min(BACKOFF_MAX_SECONDS, BACKOFF_SECONDS * 2**retry)
    return random.uniform(0, ceiling)  # jitter, not a secret


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


_NEXT_LINK: Final = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')


def next_cursor(link: str | None) -> str | None:
    """The ``cursor`` of a ``Link`` header's ``rel="next"`` target; None on the last page."""
    for part in (link or "").split(","):
        match = _NEXT_LINK.search(part)
        if match is not None:
            cursor = httpx.URL(match.group(1)).params.get("cursor")
            return cursor or None
    return None


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
        #: artifact id -> (tenant, bytes)
        self._artifacts: dict[str, tuple[str, bytes]] = {}
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
            raise Conflict(f"run {record.run_id} is not waiting on {resolution.interrupt_id}")
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

    async def inbox(self, tenant_id: str, assignee: str | None) -> Sequence[RunSummary]:
        paused = [
            r
            for r in self._runs.values()
            if r.tenant_id == tenant_id
            and r.status is RunStatus.PAUSED
            and (assignee is None or (r.awaiting is not None and r.awaiting.assignee == assignee))
        ]
        paused.sort(key=lambda r: r.updated_at, reverse=True)
        return [
            RunSummary(
                run_id=r.run_id,
                agent_id=r.agent_id,
                status=r.status,
                awaiting=r.awaiting,
                assignee=r.awaiting.assignee if r.awaiting is not None else None,
                deadline=r.deadline,
                updated_at=r.updated_at,
            )
            for r in paused[: INBOX_LIMIT * INBOX_MAX_PAGES]
        ]

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
        identity = schedule_identity(spec)
        existing = next(
            (s for s in self._schedules.values() if schedule_identity(s) == identity), None
        )
        if existing is not None:  # the upsert agent-runs does: the existing one, unchanged
            return existing
        schedule = Schedule.from_spec(spec, next_fire_at=_next_fire(spec, datetime.now(UTC)))
        self._schedules[schedule.schedule_id] = schedule
        return schedule

    async def put_artifact(
        self, run_id: str, data: bytes, *, worker_id: str | None = None
    ) -> ArtifactRef:
        record = self._fenced(run_id, worker_id)
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"art_{digest[:24]}"
        self._artifacts[artifact_id] = (record.tenant_id, data)
        return ArtifactRef(
            artifact_id=artifact_id,
            type="blob",
            mime_type=JSON_MIME,
            checksum=f"sha256:{digest}",
            size_bytes=len(data),
            created_at=now(),
        )

    async def artifact(self, artifact_id: str, tenant_id: str) -> bytes | None:
        found = self._artifacts.get(artifact_id)
        return found[1] if found is not None and found[0] == tenant_id else None

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------ internals
    def _require(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise NotFound(f"no run {run_id}")
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
            raise Conflict(f"run {record.run_id}: {record.status} cannot become {status}")
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


def schedule_identity(spec: ScheduleSpec) -> tuple[str, str, str, str, str]:
    """What makes two schedules one (agent-runs' upsert key): tenant, agent, person, cadence
    and the SHA-256 of the input as canonical JSON."""
    canonical = json.dumps(spec.input, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return (spec.tenant_id, spec.agent_id, spec.on_behalf_of, spec.cadence, digest)


#: agent-runs' named cadences, as cron (local midnight; weekly on Monday); ``manual`` never
#: fires on its own.
NAMED_CADENCES: Final = {
    "hourly": "0 * * * *",
    "daily": "0 0 * * *",
    "weekly": "0 0 * * 1",
    "weekdays": "0 0 * * 1-5",
}


def _next_fire(spec: ScheduleSpec, after: datetime) -> datetime | None:
    if spec.cadence == "manual":
        return None
    local = after.astimezone(ZoneInfo(spec.timezone))
    cron = NAMED_CADENCES.get(spec.cadence, spec.cadence)
    return croniter(cron, local).get_next(datetime).astimezone(UTC)
