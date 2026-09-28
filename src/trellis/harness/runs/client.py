"""Run records: the contracts ``RunStore`` port over the agent-runs service (design §3).

A run outlives a process and a framework swap; a UI needs an inbox of paused runs that
knows nothing about the agent. The client here is that adapter; the recorder feeds it from
the lifecycle bus, in order, without holding the turn.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Sequence
from typing import Any, Final

from pydantic import ValidationError
from trellis.contracts.errors import AgentError, AgentPaused
from trellis.contracts.events import LifecycleEvent
from trellis.contracts.runs import (
    Interrupt,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
)

from trellis.harness.events.stream import WEBHOOK_URL_FIELD
from trellis.harness.interrupts.signals import interrupt_from_signal
from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)

#: How many runs' tenants a client remembers, so a resume or a finish names the tenant the
#: run was opened under without the port having to carry it.
KNOWN_RUNS: Final = 4096


class NoRunStore:
    """No runs service configured: every record is a no-op, every read is empty."""

    name = "noop"
    required = False
    webhook_url: str | None = None

    async def started(self, start: RunStart) -> RunRecord:
        return RunRecord.from_start(start)

    async def paused(self, interrupt: Interrupt) -> RunRecord:
        return _paused_record(interrupt)

    async def resumed(self, resolution: InterruptResolution) -> RunRecord:
        return RunRecord(
            run_id=resolution.run_id,
            tenant_id="",
            agent_id="",
            status=RunStatus.RUNNING,
            last_resolution=resolution,
        )

    async def finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
    ) -> RunRecord:
        return RunRecord(
            run_id=run_id, tenant_id="", agent_id="", status=status, output=output, error=error
        )

    async def get(self, run_id: str) -> RunRecord | None:
        return None

    async def list_paused(self, tenant_id: str, *, limit: int = 100) -> Sequence[RunRecord]:
        return []


class RunStoreClient:
    """Records runs in the agent-runs service; the contracts ``RunStore`` port over its wire.

    Every write is **best-effort by default**. The runs service is a system of record, not a
    dependency of the turn: an agent that refused to answer a customer because its bookkeeping
    was unreachable would have inverted the priority. Failures are logged and the turn
    continues. ``required=True`` inverts that for deployments where an unrecorded run is worse
    than a failed one — a regulated workflow, typically.
    """

    name = "agent-runs"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        tenant_id: str | None = None,
        required: bool = False,
        timeout: float = 5.0,
        webhook_url: str | None = None,
        client: Any = None,
    ) -> None:
        import httpx  # noqa: PLC0415 - optional at import time, required to construct

        if not base_url or not api_key:
            raise ValueError("RunStoreClient needs base_url and api_key")
        self.tenant_id = tenant_id
        self.required = required
        #: Where agent-runs should POST when a run pauses or finishes. Without it a UI has
        #: to poll ``GET /v1/runs?status=PAUSED`` to notice that the 3am job is waiting on
        #: an approval — which is the wrong shape for a product where most runs do nothing
        #: interesting for minutes at a time. Set per deployment, because only the
        #: deployment knows where its own UI backend lives; a run's own ``webhook_url``
        #: (from the request metadata) wins over it.
        self.webhook_url = webhook_url
        self._httpx = httpx
        self._auth = {"X-Api-Key": api_key}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(3.0, timeout)),
        )
        self._owns_client = client is None
        self._tenants: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------ writes
    async def started(self, start: RunStart) -> RunRecord:
        """Open a run. Idempotent on the run id, so a retried turn does not open a second."""
        webhook = start.webhook_url or self.webhook_url
        self._remember(start.run_id, start.tenant_id)
        metadata = {**start.metadata}
        if start.workspace_id:
            metadata["workspace_id"] = start.workspace_id
        body = await self._send(
            "POST",
            "/v1/runs",
            start.tenant_id,
            {
                "tenant_id": start.tenant_id,
                "agent_id": start.agent_id,
                "run_id": start.run_id,
                "parent_run_id": start.parent_run_id,
                "thread_id": start.thread_id,
                "user_id": start.user_id,
                "on_behalf_of": start.on_behalf_of,
                # The run id *is* the natural idempotency key: it is derived, not random, so
                # a retried turn produces the same one and reopens nothing.
                "idempotency_key": start.run_id,
                # Omitted entirely when unset: agent-runs forbids unknown fields, and
                # sending an explicit null is not the same as not asking.
                **({WEBHOOK_URL_FIELD: webhook} if webhook else {}),
                **({"metadata": metadata} if metadata else {}),
            },
        )
        return _record(body) or RunRecord.from_start(start)

    async def paused(self, interrupt: Interrupt) -> RunRecord:
        """Mark the run as waiting on something outside it — a human, typically. This is what
        makes ``GET /v1/runs?status=PAUSED`` a human inbox that survives the process and a
        framework swap; ``awaiting`` is the interrupt as the contracts spell it."""
        self._remember(interrupt.run_id, interrupt.tenant_id)
        body = await self._transition(
            interrupt.run_id, interrupt.tenant_id, "PAUSED", awaiting=interrupt.awaiting()
        )
        return _record(body) or _paused_record(interrupt)

    async def resumed(self, resolution: InterruptResolution) -> RunRecord:
        """The person answered: ``POST /v1/runs/{id}/resume`` moves the run back to RUNNING
        with the answer on the record (the answer is also feedback, recorded by the
        harness, and what the resumed agent finds)."""
        tenant = await self._tenant_of(resolution.run_id)
        body = await self._send(
            "POST",
            f"/v1/runs/{resolution.run_id}/resume",
            tenant,
            {"answer": resolution.model_dump(mode="json", exclude_none=True)},
        )
        return _record(body) or RunRecord(
            run_id=resolution.run_id,
            tenant_id=tenant,
            agent_id="",
            status=RunStatus.RUNNING,
            last_resolution=resolution,
        )

    async def finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output: Any = None,
        error: AgentError | None = None,
    ) -> RunRecord:
        tenant = await self._tenant_of(run_id)
        body = await self._transition(
            run_id,
            tenant,
            status.value,
            output=output,
            error=error.model_dump(mode="json", exclude_none=True) if error else None,
        )
        self._tenants.pop(run_id, None)
        return _record(body) or RunRecord(
            run_id=run_id, tenant_id=tenant, agent_id="", status=status, output=output, error=error
        )

    # ------------------------------------------------------------------ reads
    async def get(self, run_id: str) -> RunRecord | None:
        data = await self._fetch(f"/v1/runs/{run_id}")
        record = _record(data)
        if record is not None:
            self._remember(record.run_id, record.tenant_id)
        return record

    async def list_paused(self, tenant_id: str, *, limit: int = 100) -> Sequence[RunRecord]:
        data = await self._fetch(
            "/v1/runs", params={"status": "PAUSED", "limit": limit}, tenant_id=tenant_id
        )
        rows = data.get("runs", data) if isinstance(data, dict) else data
        return [r for r in (_record(row) for row in (rows or [])) if r is not None]

    # ------------------------------------------------------------------ internals
    def _remember(self, run_id: str, tenant_id: str) -> None:
        self._tenants[run_id] = tenant_id
        self._tenants.move_to_end(run_id)
        while len(self._tenants) > KNOWN_RUNS:
            self._tenants.popitem(last=False)

    async def _tenant_of(self, run_id: str) -> str:
        """The tenant a run was opened under: remembered from its start or pause, else the
        client's own, else read back from the service."""
        known = self._tenants.get(run_id) or self.tenant_id
        if known:
            return known
        record = await self.get(run_id)
        return record.tenant_id if record is not None else ""

    async def _transition(self, run_id: str, tenant_id: str, status: str, **body: Any) -> Any:
        return await self._send(
            "POST",
            f"/v1/runs/{run_id}/transition",
            tenant_id,
            {"status": status, **{k: v for k, v in body.items() if v is not None}},
        )

    async def _send(self, method: str, path: str, tenant_id: str, body: dict[str, Any]) -> Any:
        """The response body of a write (a run row, when the service answers with one)."""
        headers = {**self._auth, "X-Tenant-Id": self.tenant_id or tenant_id}
        try:
            response = await self._client.request(method, path, json=body, headers=headers)
        except self._httpx.HTTPError as exc:
            self._degrade(path, f"{type(exc).__name__}: {exc}")
            return None
        if response.status_code == 409:
            # The service refusing an illegal or repeated transition protected the record
            # rather than failing, so it is not a degradation.
            log.debug("runs.transition_refused", path=path, body=response.text[:200])
            return None
        if response.status_code >= 400:
            self._degrade(path, f"HTTP {response.status_code}: {response.text[:200]}")
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def _fetch(
        self, path: str, *, params: dict[str, Any] | None = None, tenant_id: str | None = None
    ) -> Any:
        headers = {**self._auth}
        if self.tenant_id and tenant_id and self.tenant_id != tenant_id:
            raise RunStoreUnavailable(
                f"this runs client is bound to tenant {self.tenant_id!r}, not {tenant_id!r}"
            )
        if self.tenant_id or tenant_id:
            headers["X-Tenant-Id"] = self.tenant_id or str(tenant_id)
        try:
            response = await self._client.get(path, params=params, headers=headers)
        except self._httpx.HTTPError as exc:
            self._degrade(path, f"{type(exc).__name__}: {exc}")
            return None
        if response.status_code >= 400:
            if response.status_code != 404:
                self._degrade(path, f"HTTP {response.status_code}: {response.text[:200]}")
            return None
        try:
            return response.json()
        except ValueError:
            self._degrade(path, "non-JSON body")
            return None

    def _degrade(self, path: str, error: str) -> None:
        if self.required:
            raise RunStoreUnavailable(f"agent-runs is required but {path} failed: {error}")
        log.warning("runs.unavailable", path=path, error=error, effect="turn continues unrecorded")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RunStoreUnavailable(RuntimeError):
    """Raised only when ``required=True``: the deployment asked to hear about it."""


def _paused_record(interrupt: Interrupt) -> RunRecord:
    return RunRecord(
        run_id=interrupt.run_id,
        tenant_id=interrupt.tenant_id,
        agent_id="",
        status=RunStatus.PAUSED,
        awaiting=interrupt,
    )


def _record(data: Any) -> RunRecord | None:
    """A run as agent-runs returns it, as a ``RunRecord``; None when it is not one."""
    if not isinstance(data, dict) or not data.get("run_id"):
        return None
    fields = {
        "run_id": data.get("run_id"),
        "tenant_id": data.get("tenant_id") or "",
        "agent_id": data.get("agent_id") or "",
        "parent_run_id": data.get("parent_run_id"),
        "thread_id": data.get("thread_id"),
        "user_id": data.get("user_id"),
        "status": data.get("status"),
        "output": data.get("output"),
        "attempt": data.get("attempt") or 1,
    }
    awaiting = _awaiting(data)
    if awaiting is not None:
        fields["awaiting"] = awaiting
    try:
        return RunRecord.model_validate(fields)
    except ValidationError as exc:
        log.error("runs.unreadable_record", run_id=data.get("run_id"), error=str(exc)[:500])
        return None


def _awaiting(data: dict[str, Any]) -> dict[str, Any] | None:
    """The interrupt a paused row carries. Rows written before interrupts had ids (0.2.0:
    ``{"reason": "AgentPaused", "question": ...}``) are read as questions, so the inbox
    keeps showing them."""
    raw = data.get("awaiting")
    if not isinstance(raw, dict):
        return None
    if "interrupt_id" in raw:
        return raw
    question = raw.get("question")
    legacy = Interrupt(
        tenant_id=str(data.get("tenant_id") or ""),
        run_id=str(data["run_id"]),
        question=question if isinstance(question, str) and question.strip() else "paused",
        expects=raw.get("expects") if isinstance(raw.get("expects"), dict) else None,
        payload=raw.get("payload") if isinstance(raw.get("payload"), dict) else None,
    )
    return legacy.awaiting()


class RunRecorder:
    """Turns lifecycle events into run records, in order, without blocking the turn.

    Two constraints that pull against each other:

    * **Order matters.** ``started`` must reach the service before ``finished``, or the
      transition arrives for a run that does not exist yet.
    * **The turn must not wait.** Bookkeeping latency is not the customer's problem.

    The lifecycle bus schedules async listeners with ``create_task`` and forgets them, which
    satisfies the second and breaks the first — two independent tasks finish in whatever
    order the loop happens to run them, and in a short-lived process they may not run at
    all. So ``on_event`` is *synchronous*: it appends to a queue that one worker drains in
    order, and :meth:`drain` is what a caller awaits to know the record is written.
    """

    #: Harness statuses that end a run. PAUSED is deliberately absent — a paused run is not
    #: finished, and reporting it as such is what made "waiting for a human" look like a
    #: completed turn in the first place.
    FINAL = frozenset({"SUCCESS", "PARTIAL", "ERROR", "TIMEOUT", "CANCELLED", "REJECTED"})

    def __init__(self, store: Any) -> None:
        self._store = store
        self._queue: deque[tuple[str, tuple[Any, ...], dict[str, Any]]] = deque()
        self._worker: asyncio.Task[None] | None = None

    def on_event(self, event: str, payload: Any) -> None:
        """Synchronous on purpose — see the class docstring."""
        context = payload.get("context")
        if context is None:
            return
        if event == LifecycleEvent.AGENT_START:
            request = payload.get("request")
            if request is None:
                log.debug("runs.start_without_request", run_id=context.agent_run_id)
                return
            start = RunStart.from_request(
                request, webhook_url=request.metadata.get(WEBHOOK_URL_FIELD)
            )
            self._enqueue("started", (start,), {})
        elif event == LifecycleEvent.AGENT_PAUSE:
            interrupt = payload.get("interrupt") or interrupt_from_signal(
                payload.get("signal") or AgentPaused("paused"), context
            )
            self._enqueue("paused", (interrupt,), {})
        elif event == LifecycleEvent.AGENT_FINISH:
            status = str(payload.get("status") or "ERROR")
            if status not in self.FINAL:
                return
            result = payload.get("result")
            self._enqueue(
                "finished",
                (context.agent_run_id, RunStatus.from_agent_status(status)),
                {"output": getattr(result, "data", None), "error": payload.get("error")},
            )

    def _enqueue(self, action: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._queue.append((action, args, kwargs))
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(self._drain())
            except RuntimeError:
                # No loop: a synchronous caller. The queue holds, and drain() writes it.
                self._worker = None

    async def _drain(self) -> None:
        while self._queue:
            action, args, kwargs = self._queue.popleft()
            try:
                await getattr(self._store, action)(*args, **kwargs)
            except Exception:
                if getattr(self._store, "required", False):
                    # Re-raised on drain() rather than swallowed: a deployment that asked
                    # for records to be mandatory has to hear about it somewhere.
                    self._queue.appendleft((action, args, kwargs))
                    raise
                log.warning("runs.record_failed", action=action)

    async def drain(self) -> None:
        """Await every pending record. Called by the harness on drain and on close."""
        if self._worker is not None and not self._worker.done():
            await self._worker
        await self._drain()
