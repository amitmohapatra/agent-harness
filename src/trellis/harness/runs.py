"""Where runs and schedules are kept: agent-runs (``trellis.runs.RunsClient``), or this process
when ``RUNS_URL`` is unset (:class:`LocalRuns`). :class:`RunStore` is what the harness calls on
either: the part of ``RunsClient`` it uses, with the same signatures, so the two are
interchangeable. Both behave the same way:

* a run is started ``RUNNING`` (someone runs it in process) or, with ``queue=True``,
  ``QUEUED`` (for a worker);
* a pause keeps the interrupt; a resume continues the run as its next attempt — ``QUEUED``
  again when it ever came from the queue, ``RUNNING`` for the process that resumes it — and
  a ``CANCEL`` ends it;
* a worker claims a queued run under a lease — the highest ``priority`` first, then the
  oldest, among the runs with room under their ``concurrency_key`` (one running at a time per
  key and tenant) —, heartbeats it, and names itself on the pause and the finish so a worker
  whose lease lapsed cannot write over another's run; each lease says the working time the
  run has left; a stopping worker releases the runs it holds back to the queue;
* a schedule's fire copies its ``timeout_seconds`` and ``agent_version`` into the run, and
  the thread its metadata names (``thread_id``) as the run's;
* a cancel ends a queued or paused run (or one kept in its caller's process) at once, and asks
  the worker holding a running one to stop: its heartbeats say ``cancel_requested``, and the
  run is cancelled when its lease runs out;
* a heartbeat on a run kept in its caller's process (started ``RUNNING``, never queued) holds
  it for the ``worker_id`` (the process running it) instead: once the hold lapses the run ends
  ``ERROR`` (``holder_lost``), never queued; a hold fences no write, and a pause lets go of it.
  Answered, a run held when it paused must be held again within :data:`RESUME_GRACE_SECONDS`
  (whoever continues it does so at once) or it ends so too; a ``release`` by its holder lets go
  of the hold, the run ``RUNNING``, unheld.

A pause also carries the run's checkpoint — its :class:`~trellis.harness.journal.Journal`, what
a re-run needs — and so may a heartbeat (progress: the journal after a side-effecting call, so
the attempt after a worker crash replays it), which the store returns as
``RunRecord.checkpoint`` on every read and claim until the run ends, so whichever worker
resumes the run repeats no question and no side effect. Data too large for a question (an
``ask`` table or diff) is a run artifact, stored beside the run and referenced from its
interrupt (``payload_ref``); the journal is saved as segments, payload objects the checkpoint
lists (``Journal.checkpoint``).

The tenant is explicit: a call whose body names it (a start, a pause, a schedule) carries it
there, and every other call takes ``tenant=`` (the run's own, from its record or its runtime).
Reads by id answer ``None`` for a record that does not exist; writes raise the SDK's errors
(``trellis.runs``): ``NotFoundError``, ``ConflictError``, and ``LeaseLostError`` — which is not
a ``ConflictError``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import tempfile
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from croniter import croniter

from trellis.contracts import (
    AgentError,
    ArtifactRef,
    ConfigurationError,
    ErrorCategory,
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
from trellis.runs import (
    Claimed,
    ConflictError,
    FireResult,
    Lease,
    LeaseLostError,
    NotFoundError,
    Page,
    ResolutionEntry,
    RunSummary,
    ScheduleUpdate,
)
from trellis.runs.artifacts import JSON_MIME
from trellis.runs.blob import BlobExists, BlobNotFound, FilesystemBlobStore, read_whole
from trellis.runs.client import LEASE_SECONDS
from trellis.runs.payloads import MEDIA_TYPE, PayloadRef, owned_by, ref_of

#: Items a listing page holds when the caller names no limit (agent-runs' default).
PAGE_LIMIT: Final = 50
#: How many finished runs :class:`LocalRuns` keeps — the most recently finished — and for how
#: long, in seconds: an in-process store never grows without bound. Paused, queued and running
#: runs are always kept. A deployment keeps its runs in agent-runs (``RUNS_URL``).
KEEP_FINISHED: Final = 1_000
KEEP_FINISHED_SECONDS: Final = 3600.0
#: How long an answered run that was held when it paused may run unheld before it is ended
#: (agent-runs' ``RESUME_GRACE``): whoever continues it holds it at once.
RESUME_GRACE_SECONDS: Final = 60.0


class RunArtifacts(Protocol):
    """``runs.artifacts``: payloads too large for a run's row."""

    async def upload(
        self,
        run_id: str,
        data: bytes,
        *,
        mime_type: str = ...,
        worker_id: str | None = ...,
        tenant: str | None = ...,
    ) -> ArtifactRef: ...

    async def download(self, artifact_id: str, *, tenant: str | None = ...) -> bytes | None: ...

    async def put_payload(
        self,
        run_id: str,
        data: bytes,
        *,
        worker_id: str | None = ...,
        tenant: str | None = ...,
    ) -> PayloadRef:
        """Store a payload object (``trellis.runs.payloads``: a checkpoint's segment)."""
        ...

    async def get_payloads(
        self, refs: Sequence[PayloadRef], *, tenant: str | None = ...
    ) -> list[bytes]:
        """Several payload objects (a checkpoint's segments) at once, in their order."""
        ...


class RunSchedules(Protocol):
    """``runs.schedules``: what ``Agent.schedule`` creates, and what the chat's schedule tools
    (``schedules.py``) list, change and cancel."""

    async def create(self, spec: ScheduleSpec) -> Schedule: ...

    async def list(
        self,
        *,
        enabled: bool | None = ...,
        agent_id: str | None = ...,
        on_behalf_of: str | None = ...,
        cursor: str | None = ...,
        limit: int = ...,
        tenant: str | None = ...,
    ) -> Page[Schedule]: ...

    async def get(self, schedule_id: str, *, tenant: str | None = ...) -> Schedule | None: ...

    async def update(
        self, schedule_id: str, changes: ScheduleUpdate, *, tenant: str | None = ...
    ) -> Schedule: ...

    async def delete(self, schedule_id: str, *, tenant: str | None = ...) -> None: ...

    async def fire(
        self, schedule_id: str, *, at: datetime | None = ..., tenant: str | None = ...
    ) -> FireResult: ...


@runtime_checkable
class RunStore(Protocol):
    """The run store the harness drives: ``trellis.runs.RunsClient`` and :class:`LocalRuns`
    both are one. (The worker loop, ``trellis.runs.Worker``, needs its ``WorkerStore`` part:
    ``claim``, ``heartbeat``, ``release``, ``pause`` and ``finish``.)"""

    async def start(self, start: RunStart, *, queue: bool = ...) -> RunRecord: ...

    async def claim(
        self,
        worker_id: str,
        agent_ids: Sequence[str],
        *,
        lease_seconds: int = ...,
        tenant: str | None = ...,
    ) -> Claimed | None: ...

    async def heartbeat(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: int = ...,
        checkpoint: dict[str, Any] | None = ...,
        tenant: str | None = ...,
    ) -> Lease: ...

    async def release(
        self,
        run_id: str,
        worker_id: str,
        *,
        checkpoint: dict[str, Any] | None = ...,
        tenant: str | None = ...,
    ) -> RunRecord: ...

    async def cancel(
        self, run_id: str, *, reason: str | None = ..., tenant: str | None = ...
    ) -> RunRecord: ...

    async def pause(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = ...,
        worker_id: str | None = ...,
    ) -> RunRecord: ...

    async def resume(
        self, resolution: InterruptResolution, *, tenant: str | None = ...
    ) -> RunRecord: ...

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = ...,
        error: AgentError | None = ...,
        worker_id: str | None = ...,
        tenant: str | None = ...,
    ) -> RunRecord: ...

    async def get(self, run_id: str, *, tenant: str | None = ...) -> RunRecord | None: ...

    def iterate(
        self,
        *,
        status: RunStatus | None = ...,
        assignee: str | None = ...,
        agent_id: str | None = ...,
        thread_id: str | None = ...,
        parent_run_id: str | None = ...,
        limit: int = ...,
        tenant: str | None = ...,
        max_pages: int | None = ...,
    ) -> AsyncIterator[RunSummary]: ...

    async def resolutions(
        self,
        run_id: str,
        *,
        cursor: str | None = ...,
        limit: int = ...,
        tenant: str | None = ...,
    ) -> Page[ResolutionEntry]:
        """One page of every interrupt the run paused on and how it was answered, oldest
        first: what a later run of the conversation reads of what the person allowed for it
        (``conversation.allowed``)."""
        ...

    @property
    def artifacts(self) -> RunArtifacts: ...

    @property
    def schedules(self) -> RunSchedules: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- in process


class LocalRuns:
    """The same store in this process, for development and tests: nothing survives a restart,
    only workers in this process claim, and a due schedule fires when a worker claims. A
    ``tenant=`` is checked as agent-runs checks a key's tenant: a run of another tenant is not
    there; without one, every tenant's runs are. A run's working time (``worked_seconds``)
    grows each time it stops running, as agent-runs keeps it; its limits are the attempts'
    to keep (``pipeline.Budget``).

    It keeps the ``keep_finished`` most recently finished runs (:data:`KEEP_FINISHED`), each
    for at most ``keep_finished_seconds`` (:data:`KEEP_FINISHED_SECONDS`), with their artifacts:
    an older one is forgotten (``get`` answers ``None``), so a long-lived process does not grow.
    Paused, queued and running runs are never forgotten."""

    def __init__(
        self,
        *,
        keep_finished: int = KEEP_FINISHED,
        keep_finished_seconds: float = KEEP_FINISHED_SECONDS,
        payload_root: Path | None = None,
    ) -> None:
        if keep_finished < 0 or keep_finished_seconds < 0:
            raise ConfigurationError("LocalRuns keeps a number of finished runs from 0")
        self.keep_finished = keep_finished
        self.keep_finished_seconds = keep_finished_seconds
        #: the finished runs, oldest first, and when each finished (``time.monotonic``)
        self._finished: OrderedDict[str, float] = OrderedDict()
        #: artifact id -> the runs that uploaded it; run id -> its artifacts
        self._artifact_runs: dict[str, set[str]] = {}
        self._run_artifacts: dict[str, set[str]] = {}
        #: run id -> the schedule tick that queued it (``_fired``'s key)
        self._fired_by_run: dict[str, str] = {}
        self._runs: dict[str, RunRecord] = {}
        #: run id -> every interrupt it paused on and how it was answered, oldest first
        self._resolutions: dict[str, list[ResolutionEntry]] = {}
        self._queue: deque[str] = deque()
        #: runs that were ever queued: a resume sends them back to the queue
        self._queued: set[str] = set()
        self._leases: dict[str, tuple[str, datetime]] = {}
        #: the runs kept in their caller's process that it holds: by whom (``None``: an answered
        #: run whoever continues it must hold), until when
        self._holds: dict[str, tuple[str | None, datetime]] = {}
        #: the paused runs that were held when they paused
        self._held_before: set[str] = set()
        #: when each running run last started running
        self._running_since: dict[str, datetime] = {}
        #: the running runs a worker holds that were asked to stop, and why
        self._cancelling: dict[str, str | None] = {}
        self._schedules: dict[str, Schedule] = {}
        #: the run each schedule's tick queued, by ``<schedule_id>@<tick>``
        self._fired: dict[str, str] = {}
        #: artifact id -> (tenant, bytes)
        self._artifacts: dict[str, tuple[str, bytes]] = {}
        #: payload objects (a checkpoint's segments), in agent-runs' own filesystem store and
        #: format (``trellis.runs.payloads``), in ``payload_root`` (a temporary folder unset)
        self._payload_root = payload_root or Path(tempfile.mkdtemp(prefix="trellis-runs-"))
        #: the temporary folder's removal (never a folder the caller gave): at :meth:`aclose`,
        #: or when the store is collected or the process exits unclosed, as
        #: ``tempfile.TemporaryDirectory`` removes its own
        self._made = (
            None
            if payload_root
            else weakref.finalize(self, shutil.rmtree, self._payload_root, ignore_errors=True)
        )
        self._payloads = FilesystemBlobStore(self._payload_root)
        self._lock = asyncio.Lock()
        self.artifacts = LocalArtifacts(self)
        self.schedules = LocalSchedules(self)

    async def start(self, start: RunStart, *, queue: bool = False) -> RunRecord:
        return self._start(start, RunStatus.QUEUED if queue else RunStatus.RUNNING)

    def _start(self, start: RunStart, status: RunStatus) -> RunRecord:
        existing = self._runs.get(start.run_id)
        if existing is not None:  # idempotent on the run id
            return existing
        record = RunRecord.from_start(start, status=status)
        self._runs[record.run_id] = record
        if status is RunStatus.QUEUED:
            self._enqueue(record.run_id)
        else:
            self._running_since[record.run_id] = record.created_at
        return record

    async def claim(
        self,
        worker_id: str,
        agent_ids: Sequence[str],
        *,
        lease_seconds: int = LEASE_SECONDS,
        tenant: str | None = None,
    ) -> Claimed | None:
        async with self._lock:
            now = datetime.now(UTC)
            self._fire_due(now)
            self._expire_leases(now)
            wanted = set(agent_ids)
            ready = [
                record
                for record in (self._runs[run_id] for run_id in self._queue)
                if record.agent_id in wanted
                and record.status is RunStatus.QUEUED
                and tenant in (None, record.tenant_id)
                and self._room(record)
            ]
            if not ready:
                return None
            # the highest priority, then the oldest (the queue's order: max keeps the first)
            record = max(ready, key=lambda r: r.priority)
            self._queue.remove(record.run_id)
            lease = self._lease(record.run_id, worker_id, lease_seconds)
            return Claimed(run=self._move(record, RunStatus.RUNNING), lease=lease)

    def _room(self, record: RunRecord) -> bool:
        """Whether a queued run may run now: no run of its tenant sharing its
        ``concurrency_key`` is running (agent-runs' default limit, one)."""
        key = record.concurrency_key
        return key is None or not any(
            r.status is RunStatus.RUNNING
            and r.concurrency_key == key
            and r.tenant_id == record.tenant_id
            for r in self._runs.values()
        )

    async def heartbeat(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: int = LEASE_SECONDS,
        checkpoint: dict[str, Any] | None = None,
        tenant: str | None = None,
    ) -> Lease:
        found = self._require(run_id, tenant)
        if found.status is RunStatus.RUNNING and run_id not in (*self._leases, *self._queued):
            return self._hold(found, worker_id, lease_seconds, checkpoint)
        record = self._fenced(run_id, worker_id, tenant)
        if checkpoint is not None:
            record = self._move(record, record.status, checkpoint=checkpoint)
        if run_id in self._cancelling:  # asked to stop: the lease is not extended
            return self._leased(record, worker_id, self._leases[run_id][1])
        return self._lease(run_id, worker_id, lease_seconds)

    async def release(
        self,
        run_id: str,
        worker_id: str,
        *,
        checkpoint: dict[str, Any] | None = None,
        tenant: str | None = None,
    ) -> RunRecord:
        found = self._require(run_id, tenant)
        if found.status is RunStatus.RUNNING and run_id not in (*self._leases, *self._queued):
            # kept in its caller's process: its holder lets go of it, and it goes on unheld
            if self._holds.get(run_id, (None, None))[0] != worker_id:
                raise LeaseLostError(
                    f"{worker_id} does not hold {run_id}", code="LEASE_LOST", status=409
                )
            del self._holds[run_id]
            if checkpoint is not None:
                found = self._move(found, found.status, checkpoint=checkpoint)
            return found
        record = self._fenced(run_id, worker_id, tenant)
        del self._leases[run_id]
        if checkpoint is not None:
            record = self._move(record, record.status, checkpoint=checkpoint)
        return self._requeued(record)

    async def cancel(
        self, run_id: str, *, reason: str | None = None, tenant: str | None = None
    ) -> RunRecord:
        record = self._require(run_id, tenant)
        if run_id in self._cancelling:
            return record
        if record.final:
            raise _conflict(f"run {run_id} already ended {record.status.value}")
        if record.status is RunStatus.RUNNING and run_id in self._leases:
            self._cancelling[run_id] = reason  # its worker stops it
            return record
        if run_id in self._queue:
            self._queue.remove(run_id)
        return self._move(record, RunStatus.CANCELLED)

    async def pause(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        record = self._fenced(interrupt.run_id, worker_id, interrupt.tenant_id)
        self._leases.pop(record.run_id, None)
        return self._move(record, RunStatus.PAUSED, awaiting=interrupt, checkpoint=checkpoint)

    async def resume(
        self, resolution: InterruptResolution, *, tenant: str | None = None
    ) -> RunRecord:
        record = self._require(resolution.run_id, tenant)
        if record.awaiting is None or not resolution.resolves(record.awaiting):
            raise _conflict(f"run {record.run_id} is not waiting on {resolution.interrupt_id}")
        entry = ResolutionEntry(
            interrupt=record.awaiting,
            resolution=resolution,
            attempt=record.attempt,
            recorded_at=datetime.now(UTC),
        )
        self._resolutions.setdefault(record.run_id, []).append(entry)
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
        elif moved.run_id in self._held_before:  # whoever continues it holds it
            self._held_before.discard(moved.run_id)
            grace = timedelta(seconds=RESUME_GRACE_SECONDS)
            self._holds[moved.run_id] = (None, datetime.now(UTC) + grace)
        return moved

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
        worker_id: str | None = None,
        tenant: str | None = None,
    ) -> RunRecord:
        record = self._fenced(run_id, worker_id, tenant)
        self._leases.pop(run_id, None)
        return self._move(record, status, output=output, error=error)

    async def get(self, run_id: str, *, tenant: str | None = None) -> RunRecord | None:
        record = self._runs.get(run_id)
        return record if record is not None and tenant in (None, record.tenant_id) else None

    async def iterate(
        self,
        *,
        status: RunStatus | None = None,
        assignee: str | None = None,
        agent_id: str | None = None,
        thread_id: str | None = None,
        parent_run_id: str | None = None,
        limit: int = PAGE_LIMIT,
        tenant: str | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[RunSummary]:
        """The runs that match, newest first, as summaries: at most ``max_pages`` pages of
        ``limit`` when given."""
        matching = [
            r
            for r in self._runs.values()
            if tenant in (None, r.tenant_id)
            and status in (None, r.status)
            and agent_id in (None, r.agent_id)
            and thread_id in (None, r.thread_id)
            and parent_run_id in (None, r.parent_run_id)
            and assignee in (None, _assignee(r))
        ]
        matching.sort(key=lambda r: r.updated_at, reverse=True)
        for record in matching[: None if max_pages is None else limit * max_pages]:
            yield RunSummary(
                run_id=record.run_id,
                agent_id=record.agent_id,
                status=record.status,
                awaiting=record.awaiting,
                assignee=_assignee(record),
                deadline=record.deadline,
                thread_id=record.thread_id,
                updated_at=record.updated_at,
            )

    async def resolutions(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = PAGE_LIMIT,
        tenant: str | None = None,
    ) -> Page[ResolutionEntry]:
        self._require(run_id, tenant)
        start = int(cursor or 0)
        entries = self._resolutions.get(run_id, [])
        after = start + limit
        return Page[ResolutionEntry](
            items=entries[start:after], next_cursor=str(after) if after < len(entries) else None
        )

    async def aclose(self) -> None:
        """The temporary folder of the store's payloads removed, when it made one (once)."""
        if self._made is not None:
            self._made()

    # ------------------------------------------------------------------ internals
    def _require(self, run_id: str, tenant: str | None) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None or tenant not in (None, record.tenant_id):
            raise NotFoundError(f"no run {run_id}", code="NOT_FOUND", status=404)
        return record

    def _hold(
        self,
        record: RunRecord,
        holder: str,
        lease_seconds: int,
        checkpoint: dict[str, Any] | None,
    ) -> Lease:
        """Hold a run kept in its caller's process for ``holder``, as agent-runs does: another
        holder's hold in force is ``LeaseLostError``."""
        moment = datetime.now(UTC)
        held = self._holds.get(record.run_id)
        if held is not None and held[0] not in (None, holder) and held[1] >= moment:
            raise LeaseLostError(
                f"{record.run_id} is held by {held[0]}", code="LEASE_LOST", status=409
            )
        if checkpoint is not None:
            record = self._move(record, record.status, checkpoint=checkpoint)
        until = moment + timedelta(seconds=lease_seconds)
        self._holds[record.run_id] = (holder, until)
        return self._leased(record, holder, until)

    def _fenced(self, run_id: str, worker_id: str | None, tenant: str | None) -> RunRecord:
        record = self._require(run_id, tenant)
        if worker_id is not None:
            held = self._leases.get(run_id)
            if held is None or held[0] != worker_id or record.status is not RunStatus.RUNNING:
                raise LeaseLostError(
                    f"{worker_id} no longer holds {run_id}", code="LEASE_LOST", status=409
                )
        return record

    def _lease(self, run_id: str, worker_id: str, lease_seconds: int) -> Lease:
        until = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        self._leases[run_id] = (worker_id, until)
        return self._leased(self._runs[run_id], worker_id, until)

    def _leased(self, record: RunRecord, worker_id: str, until: datetime) -> Lease:
        """The lease as agent-runs answers it: the working time left, a cancel asked for."""
        remaining = None
        if record.timeout_seconds is not None:
            since = self._running_since.get(record.run_id)
            stretch = (datetime.now(UTC) - since).total_seconds() if since is not None else 0.0
            remaining = record.timeout_seconds - record.worked_seconds - stretch
        return Lease(
            run_id=record.run_id,
            worker_id=worker_id,
            expires_at=until,
            remaining_seconds=remaining,
            cancel_requested=record.run_id in self._cancelling,
        )

    def _enqueue(self, run_id: str) -> None:
        self._queue.append(run_id)
        self._queued.add(run_id)

    def _move(self, record: RunRecord, status: RunStatus, **changes: Any) -> RunRecord:
        if record.status is not status and not record.status.can_become(status):
            raise _conflict(f"run {record.run_id}: {record.status} cannot become {status}")
        if status.final:  # an ending clears what the run waited on and would resume from
            changes.update(awaiting=None, checkpoint=None)
            self._cancelling.pop(record.run_id, None)
        moment = datetime.now(UTC)
        if status is not RunStatus.RUNNING:  # a hold is on a running run only
            held = self._holds.pop(record.run_id, None)
            if status is RunStatus.PAUSED and held is not None and held[0] is not None:
                self._held_before.add(record.run_id)
            elif status.final:
                self._held_before.discard(record.run_id)
        since = self._running_since.pop(record.run_id, None)
        if since is not None and status is not RunStatus.RUNNING:  # it stops working
            changes["worked_seconds"] = record.worked_seconds + (moment - since).total_seconds()
        elif status is RunStatus.RUNNING:
            self._running_since[record.run_id] = since or moment
        moved = RunRecord.model_validate(
            {**record.model_dump(), **changes, "status": status, "updated_at": moment}
        )
        self._runs[moved.run_id] = moved
        if status.final:
            self._finished[moved.run_id] = time.monotonic()
            self._finished.move_to_end(moved.run_id)
            self._forget()
        return moved

    def _forget(self) -> None:
        """Past the cap, or past their time, the oldest finished runs go, with what is theirs."""
        horizon = time.monotonic() - self.keep_finished_seconds
        while self._finished:
            run_id, finished = next(iter(self._finished.items()))
            if len(self._finished) <= self.keep_finished and finished >= horizon:
                return
            del self._finished[run_id]
            self._drop(run_id)

    def _drop(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        self._resolutions.pop(run_id, None)
        self._queued.discard(run_id)
        self._leases.pop(run_id, None)
        self._running_since.pop(run_id, None)
        self._cancelling.pop(run_id, None)  # (a finished run is in no queue: claim, cancel)
        tick = self._fired_by_run.pop(run_id, None)
        if tick is not None:
            self._fired.pop(tick, None)
        for artifact_id in self._run_artifacts.pop(run_id, set()):
            owners = self._artifact_runs.get(artifact_id, set())
            owners.discard(run_id)
            if not owners:
                self._artifact_runs.pop(artifact_id, None)
                if self._artifacts.pop(artifact_id, None) is None:  # a payload object's key
                    (self._payload_root / artifact_id).unlink(missing_ok=True)

    def _expire_leases(self, now: datetime) -> None:
        for run_id, (_, until) in list(self._leases.items()):
            if until < now:
                del self._leases[run_id]
                self._requeued(self._runs[run_id])
        for run_id, (holder, until) in list(self._holds.items()):
            if until < now:  # its holder stopped: nothing else can run it
                stopped = (
                    f"its holder stopped: {holder} kept the run in its own process and "
                    f"stopped heartbeating (held until {until}), so nothing runs it"
                    if holder is not None
                    else "nobody took the run up after its answer: it ran in its caller's "
                    f"process, and nothing held it by {until} (the process that ran it before "
                    "it paused stopped while it waited)"
                )
                error = AgentError(
                    code="holder_lost",
                    category=ErrorCategory.TIMEOUT,
                    message=stopped,
                    retryable=False,
                    source="agent-runs",
                )
                self._move(self._runs[run_id], RunStatus.ERROR, error=error)

    def _requeued(self, record: RunRecord) -> RunRecord:
        """A running run nobody holds any more back on the queue as its next attempt — or,
        asked to stop, ``CANCELLED``."""
        if record.run_id in self._cancelling:
            return self._move(record, RunStatus.CANCELLED)
        moved = self._move(record, RunStatus.QUEUED, attempt=record.attempt + 1)
        self._enqueue(record.run_id)
        return moved

    def _fire_due(self, now: datetime) -> None:
        for schedule in list(self._schedules.values()):
            if not schedule.enabled or schedule.next_fire_at is None or schedule.next_fire_at > now:
                continue
            self._fire(schedule, now, now)

    def _fire(self, schedule: Schedule, tick: datetime, now: datetime) -> FireResult:
        """Queue ``schedule``'s run for ``tick`` (a repeated tick answers the run it queued)
        and advance the schedule past ``now``."""
        key = f"{schedule.schedule_id}@{tick.astimezone(UTC).isoformat()}"
        run_id = self._fired.get(key)
        if run_id is None:
            start = RunStart(
                run_id=new_id("run_"),
                tenant_id=schedule.tenant_id,
                agent_id=schedule.agent_id,
                user_id=schedule.on_behalf_of,
                on_behalf_of=schedule.on_behalf_of,
                thread_id=_thread(schedule),
                workspace_id=schedule.workspace_id,
                input=schedule.input,
                timeout_seconds=schedule.timeout_seconds,
                agent_version=schedule.agent_version,
                # as agent-runs fires it: the schedule's metadata, the fire's own keys over it
                metadata={**schedule.metadata, "schedule_id": schedule.schedule_id},
                priority=getattr(schedule, "priority", 0),
                concurrency_key=getattr(schedule, "concurrency_key", None),
            )
            run_id = self._fired[key] = self._start(start, RunStatus.QUEUED).run_id
            self._fired_by_run[run_id] = key
            schedule = self._schedules[schedule.schedule_id] = schedule.model_copy(
                update={
                    "last_fired_at": now,
                    "last_run_id": run_id,
                    "next_fire_at": _next_fire(schedule, now),
                }
            )
        return FireResult(
            schedule_id=schedule.schedule_id,
            run_id=run_id,
            fire_time=tick,
            idempotency_key=key,
            schedule=schedule,
        )


class LocalArtifacts:
    """``LocalRuns.artifacts``: kept per tenant, by the SHA-256 of their bytes."""

    def __init__(self, runs: LocalRuns) -> None:
        self._runs = runs

    async def upload(
        self,
        run_id: str,
        data: bytes,
        *,
        mime_type: str = JSON_MIME,
        worker_id: str | None = None,
        tenant: str | None = None,
    ) -> ArtifactRef:
        record = self._runs._fenced(run_id, worker_id, tenant)
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"art_{digest[:24]}"
        self._runs._artifacts[artifact_id] = (record.tenant_id, data)
        self._runs._artifact_runs.setdefault(artifact_id, set()).add(run_id)
        self._runs._run_artifacts.setdefault(run_id, set()).add(artifact_id)
        return ArtifactRef(
            artifact_id=artifact_id,
            type="blob",
            mime_type=mime_type,
            checksum=f"sha256:{digest}",
            size_bytes=len(data),
            created_at=now(),
        )

    async def download(self, artifact_id: str, *, tenant: str | None = None) -> bytes | None:
        found = self._runs._artifacts.get(artifact_id)
        return found[1] if found is not None and tenant in (None, found[0]) else None

    async def put_payload(
        self,
        run_id: str,
        data: bytes,
        *,
        worker_id: str | None = None,
        tenant: str | None = None,
    ) -> PayloadRef:
        """The object, stored once (the same bytes are the same key) and whole or not at all;
        kept while a run that wrote it is."""
        record = self._runs._fenced(run_id, worker_id, tenant)
        ref = ref_of(record.tenant_id, data)
        with contextlib.suppress(BlobExists):
            await self._runs._payloads.put(ref.ref, data, content_type=MEDIA_TYPE)
        self._runs._artifact_runs.setdefault(ref.ref, set()).add(run_id)
        self._runs._run_artifacts.setdefault(run_id, set()).add(ref.ref)
        return ref

    async def get_payloads(
        self, refs: Sequence[PayloadRef], *, tenant: str | None = None
    ) -> list[bytes]:
        """The objects' bytes, read at once and each checked against its reference;
        ``NotFoundError`` for another tenant's object or one that is gone."""
        if tenant is not None and not all(owned_by(ref, tenant) for ref in refs):
            raise NotFoundError("a payload object is another tenant's")
        store = self._runs._payloads
        try:
            return list(
                await asyncio.gather(
                    *(read_whole(store, r.ref, sha256=r.sha256, size=r.size) for r in refs)
                )
            )
        except BlobNotFound as exc:
            raise NotFoundError(f"no payload object {exc}") from exc


class LocalSchedules:
    """``LocalRuns.schedules``: a due schedule fires when a worker claims."""

    def __init__(self, runs: LocalRuns) -> None:
        self._runs = runs

    async def create(self, spec: ScheduleSpec) -> Schedule:
        """The upsert agent-runs does: a schedule with the same identity answers the existing
        one, unchanged."""
        kept = self._runs._schedules
        identity = schedule_identity(spec)
        existing = next((s for s in kept.values() if schedule_identity(s) == identity), None)
        if existing is not None:
            return existing
        schedule = Schedule.from_spec(spec, next_fire_at=_next_fire(spec, datetime.now(UTC)))
        kept[schedule.schedule_id] = schedule
        return schedule

    async def list(
        self,
        *,
        enabled: bool | None = None,
        agent_id: str | None = None,
        on_behalf_of: str | None = None,
        cursor: str | None = None,
        limit: int = PAGE_LIMIT,
        tenant: str | None = None,
    ) -> Page[Schedule]:
        """The tenant's schedules, newest first, filtered as agent-runs filters them; the
        cursor is the offset of the next page."""
        wanted = {"enabled": enabled, "agent_id": agent_id, "on_behalf_of": on_behalf_of}
        found = [
            s
            for s in reversed(self._runs._schedules.values())
            if tenant in (None, s.tenant_id)
            and all(v is None or getattr(s, k) == v for k, v in wanted.items())
        ]
        start = int(cursor or 0)
        after = start + limit
        return Page[Schedule](
            items=found[start:after], next_cursor=str(after) if after < len(found) else None
        )

    async def get(self, schedule_id: str, *, tenant: str | None = None) -> Schedule | None:
        found = self._runs._schedules.get(schedule_id)
        return found if found is not None and tenant in (None, found.tenant_id) else None

    async def update(
        self, schedule_id: str, changes: ScheduleUpdate, *, tenant: str | None = None
    ) -> Schedule:
        """Change the fields ``changes`` sets, as agent-runs does: ``metadata`` merged into the
        schedule's, a new cadence or zone re-arming the next fire."""
        schedule = self._owned(schedule_id, tenant)
        changed = changes.model_dump(exclude_unset=True)
        if "metadata" in changed:
            changed["metadata"] = {**schedule.metadata, **(changed["metadata"] or {})}
        updated = Schedule.model_validate({**schedule.model_dump(), **changed})
        if {"cadence", "timezone"} & changed.keys():
            updated = updated.model_copy(
                update={"next_fire_at": _next_fire(updated, datetime.now(UTC))}
            )
        self._runs._schedules[schedule_id] = updated
        return updated

    async def delete(self, schedule_id: str, *, tenant: str | None = None) -> None:
        del self._runs._schedules[self._owned(schedule_id, tenant).schedule_id]

    def _owned(self, schedule_id: str, tenant: str | None) -> Schedule:
        schedule = self._runs._schedules.get(schedule_id)
        if schedule is None or tenant not in (None, schedule.tenant_id):
            raise NotFoundError(f"no schedule {schedule_id}", code="NOT_FOUND", status=404)
        return schedule

    async def fire(
        self, schedule_id: str, *, at: datetime | None = None, tenant: str | None = None
    ) -> FireResult:
        """Queue the schedule's run now, as agent-runs' ``schedules.fire`` does: for the tick
        ``at`` (a repeat for it answers the same run), else for the tick it is due for, else
        for now — a ``manual`` schedule's only way to run."""
        schedule = self._owned(schedule_id, tenant)
        current = datetime.now(UTC)
        due = schedule.next_fire_at
        tick = at or (due if due is not None and due <= current else current)
        return self._runs._fire(schedule, tick, current)


def _conflict(message: str) -> ConflictError:
    return ConflictError(message, code="CONFLICT", status=409)


def _assignee(record: RunRecord) -> str | None:
    return record.awaiting.assignee if record.awaiting is not None else None


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


#: The schedule metadata key naming the conversation its runs are turns of (agent-runs fires
#: them with it as their ``thread_id``).
THREAD: Final = "thread_id"


def _thread(schedule: Schedule) -> str | None:
    thread = schedule.metadata.get(THREAD)
    return thread if isinstance(thread, str) and thread else None


def _next_fire(spec: ScheduleSpec, after: datetime) -> datetime | None:
    if spec.cadence == "manual":
        return None
    local = after.astimezone(ZoneInfo(spec.timezone))
    cron = NAMED_CADENCES.get(spec.cadence, spec.cadence)
    return croniter(cron, local).get_next(datetime).astimezone(UTC)
