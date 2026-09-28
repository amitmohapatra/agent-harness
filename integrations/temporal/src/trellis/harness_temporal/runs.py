"""Temporal as the contracts ``RunStore`` (design §10, §15 row 6).

The same port the agent-runs HTTP client implements, so choosing Temporal is a configuration
decision (``runs.engine``) and not a change to a single agent. One workflow per run, started
under the run's own id; the transitions are signals; the reads are queries.

Best-effort like the agent-runs client, and for the same reason: the run store is a system of
record, not a dependency of the turn. An agent that refused to answer because its bookkeeping
was unreachable would have inverted the priority. ``required=True`` inverts it back for
deployments where an unrecorded run is worse than a failed one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from temporalio.common import WorkflowIDConflictPolicy
from temporalio.service import RPCError, RPCStatusCode
from trellis.contracts.errors import AgentError
from trellis.contracts.runs import (
    Interrupt,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
)

from trellis.harness.runtime.logging import get_logger
from trellis.harness_temporal.client import ClientSource, TemporalConnection
from trellis.harness_temporal.state import ENDINGS_WITH_ERROR, RunEnding
from trellis.harness_temporal.workflow import (
    FINISH_SIGNAL,
    PAUSE_SIGNAL,
    RECORD_QUERY,
    RESUME_SIGNAL,
    RUN_WORKFLOW,
    AgentRunWorkflow,
)

log = get_logger("trellis.harness_temporal.runs")

#: How many running workflows ``list_paused`` will query before it gives up. A stock namespace
#: has no search attribute for "is this run paused", so the inbox is assembled by asking; the
#: README shows the search-attribute variant for a namespace that has one registered.
DEFAULT_SCAN_LIMIT = 1000


class TemporalRunsUnavailable(RuntimeError):
    """Raised only when ``required=True``: the deployment asked to hear about it."""


class TemporalRunStore:
    """A workflow per run. Implements :class:`trellis.contracts.ports.RunStore`."""

    name = "temporal"

    def __init__(
        self,
        client: ClientSource,
        *,
        task_queue: str = "trellis-runs",
        namespace: str = "default",
        required: bool = False,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
        connection: TemporalConnection | None = None,
        **connect_options: Any,
    ) -> None:
        if not task_queue:
            raise ValueError("TemporalRunStore needs a task_queue")
        #: Shared with a scheduler in the same process when one is passed in.
        self.connection = connection or TemporalConnection(
            client, namespace=namespace, **connect_options
        )
        self.task_queue = task_queue
        self.required = required
        self.scan_limit = scan_limit

    # ------------------------------------------------------------------ writes
    async def started(self, start: RunStart) -> RunRecord:
        """Open the run's workflow. Idempotent by construction: the workflow id *is* the run
        id, so a retried turn reopens nothing (``USE_EXISTING`` rather than ``FAIL``, because
        a retry finding its own run is the normal case, not an error)."""
        record = RunRecord.from_start(start)
        try:
            client = await self.connection.client()
            await client.start_workflow(
                AgentRunWorkflow.run,
                start,
                id=start.run_id,
                task_queue=self.task_queue,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
                memo={"tenant_id": start.tenant_id, "agent_id": start.agent_id},
            )
        except Exception as exc:
            self._degrade(f"start {start.run_id}", exc)
        return record

    async def paused(self, interrupt: Interrupt) -> RunRecord:
        """The run is waiting on a person. ``GET`` then shows it with its question, which is
        what makes a paused-run inbox that knows nothing about the agent possible."""
        await self._signal(interrupt.run_id, PAUSE_SIGNAL, interrupt)
        return await self._read(interrupt.run_id) or RunRecord(
            run_id=interrupt.run_id,
            tenant_id=interrupt.tenant_id,
            agent_id="",
            status=RunStatus.PAUSED,
            awaiting=interrupt,
        )

    async def resumed(self, resolution: InterruptResolution) -> RunRecord:
        await self._signal(resolution.run_id, RESUME_SIGNAL, resolution)
        return await self._read(resolution.run_id) or RunRecord(
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
        ending = RunEnding(status=status, output=output, error=error)
        await self._signal(run_id, FINISH_SIGNAL, ending)
        return await self._read(run_id) or RunRecord(
            run_id=run_id,
            tenant_id="",
            agent_id="",
            status=status,
            output=output,
            error=error if status in ENDINGS_WITH_ERROR else None,
        )

    # ------------------------------------------------------------------ reads
    async def get(self, run_id: str) -> RunRecord | None:
        """The run's own record, queried from its workflow. ``None`` when there is no such
        workflow — including one the namespace's retention has already swept away."""
        return await self._read(run_id)

    async def list_paused(self, tenant_id: str, *, limit: int = 100) -> Sequence[RunRecord]:
        """The tenant's paused runs: the human inbox.

        Assembled by listing this workflow type's running executions and asking each for its
        record, because a stock namespace has no search attribute saying which runs are
        paused. Bounded twice — ``limit`` results, ``scan_limit`` questions — so an inbox
        query can never walk a whole cluster.
        """
        if limit <= 0:
            return []
        try:
            client = await self.connection.client()
            query = f"WorkflowType = '{RUN_WORKFLOW}' AND ExecutionStatus = 'Running'"
            found: list[RunRecord] = []
            scanned = 0
            async for execution in client.list_workflows(query, limit=self.scan_limit):
                scanned += 1
                if scanned > self.scan_limit:
                    break
                if not await _claims_tenant(execution, tenant_id):
                    continue
                record = await self._read(execution.id)
                if record is None:
                    continue
                # The memo is a cheap pre-filter; the record is the decision. Both must name
                # this tenant — a memo that could not be read says "ask", never "admit", or an
                # unreadable memo would be a way to read another tenant's inbox.
                if record.tenant_id != tenant_id or record.status != RunStatus.PAUSED:
                    continue
                found.append(record)
                if len(found) >= limit:
                    break
            return found
        except Exception as exc:
            self._degrade(f"list paused runs for {tenant_id}", exc)
            return []

    # ------------------------------------------------------------------ internals
    async def _signal(self, run_id: str, signal: str, payload: Any) -> None:
        try:
            client = await self.connection.client()
            await client.get_workflow_handle(run_id).signal(signal, payload)
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                # The run is gone (retention swept it, or it was never opened here). Nothing
                # to record and nothing to fail: the same shape as agent-runs refusing a
                # transition it has no row for.
                log.debug("temporal.signal_no_run", run_id=run_id, signal=signal)
                return
            self._degrade(f"{signal} {run_id}", exc)
        except Exception as exc:
            self._degrade(f"{signal} {run_id}", exc)

    async def _read(self, run_id: str) -> RunRecord | None:
        try:
            client = await self.connection.client()
            record = await client.get_workflow_handle(run_id).query(RECORD_QUERY)
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return None
            self._degrade(f"query {run_id}", exc)
            return None
        except Exception as exc:
            self._degrade(f"query {run_id}", exc)
            return None
        if record is None:
            return None
        return record if isinstance(record, RunRecord) else RunRecord.model_validate(record)

    def _degrade(self, what: str, error: Exception) -> None:
        if self.required:
            raise TemporalRunsUnavailable(f"temporal is required but {what} failed: {error}")
        log.warning(
            "temporal.unavailable",
            operation=what,
            error=f"{type(error).__name__}: {error}",
            effect="turn continues unrecorded",
        )

    async def aclose(self) -> None:
        await self.connection.aclose()


async def _claims_tenant(execution: Any, tenant_id: str) -> bool:
    """Whether a listed execution's memo names this tenant.

    The memo is written at start, so the cheap filter needs no extra round trip; an execution
    whose memo cannot be read (an older run, another writer) is *not* excluded here — it is
    asked, and ``list_paused`` then checks the record's own ``tenant_id``. Dropping it here
    would hide a paused run from its owner; admitting it on the memo alone would show one to
    somebody else.
    """
    try:
        memo = await execution.memo()
    except Exception:  # pragma: no cover - an unreadable memo must not hide a run
        return True
    claimed = memo.get("tenant_id") if isinstance(memo, dict) else None
    return claimed is None or claimed == tenant_id


__all__ = ["DEFAULT_SCAN_LIMIT", "TemporalRunStore", "TemporalRunsUnavailable"]
