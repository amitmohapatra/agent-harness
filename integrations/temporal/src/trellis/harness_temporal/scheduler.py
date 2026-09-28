"""Temporal Schedules as the contracts ``Scheduler`` (design §10, §15 row 6).

A standing intent ("every weekday at 8") is editable without touching the history of what
fired, which is exactly what a Temporal Schedule is. Firing starts the same
:class:`~trellis.harness_temporal.workflow.AgentRunWorkflow` with the same ``RunStart`` as any
other entry point, carrying ``on_behalf_of`` — so a 6 a.m. run acts as the person who set the
schedule and memory and policy see the right identity.

The platform's spec travels in the schedule's **memo**. Temporal owns *when* a schedule fires;
the contract owns *what* it is, and reverse-engineering a cadence out of a Temporal spec would
lose the difference between ``weekdays`` and the cron it compiled to.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from temporalio.client import (
    Schedule as TemporalSchedule,
)
from temporalio.client import (
    ScheduleActionStartWorkflow,
    ScheduleState,
)
from temporalio.service import RPCError, RPCStatusCode
from trellis.contracts.runs import RunStart, Schedule, ScheduleSpec

from trellis.harness.runtime.logging import get_logger
from trellis.harness_temporal.cadence import DEFAULT_FLOOR_SECONDS, is_manual, parse_cadence
from trellis.harness_temporal.client import ClientSource, TemporalConnection
from trellis.harness_temporal.workflow import AgentRunWorkflow

log = get_logger("trellis.harness_temporal.scheduler")

#: The memo key the contract's own spec is kept under.
SPEC_MEMO_KEY = "trellis_schedule"

#: How many schedules ``list_for_tenant`` will read before it stops.
DEFAULT_SCAN_LIMIT = 1000


class TemporalScheduler:
    """Temporal Schedules behind :class:`trellis.contracts.ports.Scheduler`.

    Unlike the run store, a scheduler is *not* best-effort: creating a schedule is the user's
    explicit act, and a "saved" schedule that does not exist is worse than an error. Failures
    raise.
    """

    name = "temporal"

    def __init__(
        self,
        client: ClientSource,
        *,
        task_queue: str = "trellis-runs",
        namespace: str = "default",
        floor_seconds: int = DEFAULT_FLOOR_SECONDS,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
        connection: TemporalConnection | None = None,
        **connect_options: Any,
    ) -> None:
        if not task_queue:
            raise ValueError("TemporalScheduler needs a task_queue")
        self.connection = connection or TemporalConnection(
            client, namespace=namespace, **connect_options
        )
        self.task_queue = task_queue
        self.floor_seconds = floor_seconds
        self.scan_limit = scan_limit

    # ------------------------------------------------------------------ writes
    async def create(self, spec: ScheduleSpec) -> Schedule:
        """Register the standing intent. The cadence is validated here, not at 6 a.m."""
        schedule = Schedule.from_spec(spec)
        temporal_spec = parse_cadence(
            spec.cadence, timezone=spec.timezone, floor_seconds=self.floor_seconds
        )
        client = await self.connection.client()
        handle = await client.create_schedule(
            schedule.schedule_id,
            TemporalSchedule(
                action=ScheduleActionStartWorkflow(
                    AgentRunWorkflow.run,
                    _run_start(schedule),
                    id=schedule.schedule_id,
                    task_queue=self.task_queue,
                    memo={"tenant_id": spec.tenant_id, "agent_id": spec.agent_id},
                ),
                spec=temporal_spec,
                # ``manual`` means "only when triggered": paused, with no automatic spec.
                state=ScheduleState(paused=is_manual(spec.cadence) or not spec.enabled),
            ),
            memo=_memo(schedule),
        )
        return await self._describe(handle.id) or schedule

    async def set_enabled(self, schedule_id: str, enabled: bool) -> Schedule:
        """Pause or unpause. A ``manual`` schedule stays paused: unpausing it would give it a
        cadence it does not have."""
        current = await self.get(schedule_id)
        if current is None:
            raise KeyError(f"no schedule {schedule_id!r}")
        if enabled and is_manual(current.cadence):
            raise ValueError(f"schedule {schedule_id!r} is manual: trigger it, do not enable it")
        client = await self.connection.client()
        handle = client.get_schedule_handle(schedule_id)
        note = "enabled by the platform" if enabled else "disabled by the platform"
        if enabled:
            await handle.unpause(note=note)
        else:
            await handle.pause(note=note)
        updated = current.model_copy(update={"enabled": enabled})
        return await self._describe(schedule_id) or updated

    async def delete(self, schedule_id: str) -> None:
        client = await self.connection.client()
        try:
            await client.get_schedule_handle(schedule_id).delete()
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            # Deleting what is already gone is the outcome the caller asked for.
            log.debug("temporal.schedule_absent", schedule_id=schedule_id)

    # ------------------------------------------------------------------ reads
    async def get(self, schedule_id: str) -> Schedule | None:
        return await self._describe(schedule_id)

    async def list_for_tenant(self, tenant_id: str, *, limit: int = 100) -> Sequence[Schedule]:
        """This tenant's schedules. Filtered on the memo the platform wrote, so a namespace
        shared with another product never leaks into somebody else's list."""
        if limit <= 0:
            return []
        client = await self.connection.client()
        found: list[Schedule] = []
        scanned = 0
        async for listed in await client.list_schedules():
            scanned += 1
            if scanned > self.scan_limit:  # pragma: no cover - bound, not behaviour
                break
            schedule = await _from_memo(listed)
            if schedule is None or schedule.tenant_id != tenant_id:
                continue
            found.append(_with_temporal(schedule, listed))
            if len(found) >= limit:
                break
        return found

    # ------------------------------------------------------------------ internals
    async def _describe(self, schedule_id: str) -> Schedule | None:
        client = await self.connection.client()
        try:
            description = await client.get_schedule_handle(schedule_id).describe()
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return None
            raise
        schedule = await _from_memo(description)
        return None if schedule is None else _with_temporal(schedule, description)


def _run_start(schedule: Schedule) -> RunStart:
    """The run a firing creates: the same request contract as every other entry point.

    ``run_id`` is the schedule's id here and the *firing's* workflow id at run time — Temporal
    appends the nominal time to the action's id, and the workflow takes its own id as the run
    id, so every firing is its own run rather than a reopening of the template.
    """
    return RunStart(
        run_id=schedule.schedule_id,
        tenant_id=schedule.tenant_id,
        agent_id=schedule.agent_id,
        workspace_id=schedule.workspace_id,
        on_behalf_of=schedule.on_behalf_of,
        user_id=schedule.on_behalf_of,
        input=schedule.input,
        webhook_url=schedule.webhook_url,
        metadata={**schedule.metadata, "schedule_id": schedule.schedule_id},
    )


def _memo(schedule: Schedule) -> dict[str, Any]:
    return {
        "tenant_id": schedule.tenant_id,
        "agent_id": schedule.agent_id,
        SPEC_MEMO_KEY: schedule.model_dump(mode="json"),
    }


async def _from_memo(description: Any) -> Schedule | None:
    """The platform's schedule as the memo recorded it; ``None`` for somebody else's."""
    try:
        memo = await description.memo()
    except Exception:  # pragma: no cover - an unreadable memo is not ours
        return None
    payload = memo.get(SPEC_MEMO_KEY) if isinstance(memo, dict) else None
    if not isinstance(payload, dict):
        return None
    return Schedule.model_validate({**payload, "schedule_id": description.id})


def _with_temporal(schedule: Schedule, description: Any) -> Schedule:
    """The memo's record with what only Temporal knows: paused, and the firing times.

    ``paused`` is read back rather than kept in the memo because a memo cannot be updated on
    an existing schedule — so ``enabled`` has exactly one source of truth, and it is the one
    that decides whether the thing actually fires.
    """
    info = getattr(description, "info", None)
    state = getattr(getattr(description, "schedule", None), "state", None)
    paused = getattr(state, "paused", None)
    next_times = list(getattr(info, "next_action_times", ()) or ())
    recent = list(getattr(info, "recent_actions", ()) or ())
    return schedule.model_copy(
        update={
            **({"enabled": not paused} if paused is not None else {}),
            "next_fire_at": next_times[0] if next_times else None,
            "last_fired_at": getattr(recent[-1], "scheduled_at", None) if recent else None,
        }
    )


__all__ = ["DEFAULT_SCAN_LIMIT", "SPEC_MEMO_KEY", "TemporalScheduler"]
