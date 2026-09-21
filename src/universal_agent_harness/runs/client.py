"""Durable runs: the seam between a turn and the service that outlives it.

A run that pauses to ask a person a question may wait longer than any worker stays alive.
LangGraph's checkpointer can resume it *within LangGraph*; what it cannot do is answer "what
is waiting for a human right now" from outside the framework, or survive the framework being
swapped. That is what the runs service is for, and this is the client.

Nothing here executes anything. The harness runs the agent; this records what happened to
it, so the record outlives the process.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from universal_agent_contracts.events import LifecycleEvent

from universal_agent_harness.runtime.logging import get_logger

log = get_logger("universal_agent_harness.runs")

_ERROR = 400


class NoRunStore:
    """The default. A deployment with no runs service behaves exactly as it did before."""

    name = "noop"

    async def started(self, context: Any) -> None:
        return None

    async def paused(self, context: Any, *, reason: str, awaiting: Any = None) -> None:
        return None

    async def finished(self, context: Any, *, status: str, output: Any = None) -> None:
        return None


class RunStoreClient:
    """Records runs in the agent-runs service.

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
        client: Any = None,
    ) -> None:
        import httpx  # noqa: PLC0415 - optional at import time, required to construct

        if not base_url or not api_key:
            raise ValueError("RunStoreClient needs base_url and api_key")
        self.tenant_id = tenant_id
        self.required = required
        self._httpx = httpx
        self._auth = {"X-Api-Key": api_key}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(3.0, timeout)),
        )
        self._owns_client = client is None

    # ------------------------------------------------------------------ lifecycle
    async def started(self, context: Any) -> None:
        """Open a run. Idempotent on the run id, so a retried turn does not open a second."""
        await self._send(
            "POST",
            "/v1/runs",
            context.tenant_id,
            {
                "tenant_id": context.tenant_id,
                "agent_id": context.agent_id,
                "run_id": context.agent_run_id,
                "parent_run_id": context.parent_agent_run_id,
                "thread_id": context.thread_id,
                "user_id": context.user_id,
                # The run id *is* the natural idempotency key: it is derived, not random, so
                # a retried turn produces the same one and reopens nothing.
                "idempotency_key": context.agent_run_id,
            },
        )

    async def paused(self, context: Any, *, reason: str, awaiting: Any = None) -> None:
        """Mark the run as waiting on something outside it — a human, typically.

        This is what makes ``GET /v1/runs?status=PAUSED`` a human inbox that survives the
        process, and a framework swap.
        """
        await self._transition(context, "PAUSED", awaiting={"reason": reason, **(awaiting or {})})

    async def finished(self, context: Any, *, status: str, output: Any = None) -> None:
        await self._transition(context, status, output=output)

    # ------------------------------------------------------------------ internals
    async def _transition(self, context: Any, status: str, **body: Any) -> None:
        await self._send(
            "POST",
            f"/v1/runs/{context.agent_run_id}/transition",
            context.tenant_id,
            {"status": status, **{k: v for k, v in body.items() if v is not None}},
        )

    async def _send(self, method: str, path: str, tenant_id: str, body: dict[str, Any]) -> None:
        headers = {**self._auth, "X-Tenant-Id": self.tenant_id or tenant_id}
        try:
            response = await self._client.request(method, path, json=body, headers=headers)
        except self._httpx.HTTPError as exc:
            self._degrade(path, str(exc))
            return
        if response.status_code >= _ERROR:
            # 409 is the service refusing an illegal or repeated transition — it protected
            # the record rather than failing, so it is not a degradation.
            level = log.debug if response.status_code == 409 else None
            if level is not None:
                level("runs.transition_refused", path=path, body=response.text[:200])
                return
            self._degrade(path, f"{response.status_code}: {response.text[:200]}")

    def _degrade(self, path: str, error: str) -> None:
        if self.required:
            raise RuntimeError(f"agent-runs is required but {path} failed: {error}")
        log.warning("runs.unavailable", path=path, error=error, effect="turn continues unrecorded")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["NoRunStore", "RunRecorder", "RunStoreClient"]


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
        self._queue: deque[tuple[str, Any, dict[str, Any]]] = deque()
        self._worker: asyncio.Task[None] | None = None

    def on_event(self, event: str, payload: Any) -> None:
        """Synchronous on purpose — see the class docstring."""
        context = payload.get("context")
        if context is None:
            return
        if event == LifecycleEvent.AGENT_START:
            self._enqueue("started", context, {})
        elif event == LifecycleEvent.AGENT_PAUSE:
            signal = payload.get("signal")
            reason = type(signal).__name__ if signal is not None else "paused"
            self._enqueue("paused", context, {"reason": reason})
        elif event == LifecycleEvent.AGENT_FINISH:
            status = str(payload.get("status") or "ERROR")
            if status not in self.FINAL:
                return
            result = payload.get("result")
            self._enqueue(
                "finished", context, {"status": status, "output": getattr(result, "data", None)}
            )

    def _enqueue(self, action: str, context: Any, kwargs: dict[str, Any]) -> None:
        self._queue.append((action, context, kwargs))
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(self._drain())
            except RuntimeError:
                # No loop: a synchronous caller. The queue holds, and drain() writes it.
                self._worker = None

    async def _drain(self) -> None:
        while self._queue:
            action, context, kwargs = self._queue.popleft()
            try:
                await getattr(self._store, action)(context, **kwargs)
            except Exception:
                if getattr(self._store, "required", False):
                    # Re-raised on drain() rather than swallowed: a deployment that asked
                    # for records to be mandatory has to hear about it somewhere.
                    self._queue.appendleft((action, context, kwargs))
                    raise
                log.warning("runs.record_failed", action=action)

    async def drain(self) -> None:
        """Await every pending record. Called by the harness on drain and on close."""
        if self._worker is not None and not self._worker.done():
            await self._worker
        await self._drain()
