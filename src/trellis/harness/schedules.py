"""Schedules a user sets in chat ("every Monday at 8 check stock and tell me if under 5"): four
built-in tools every adapter gets while the run store can schedule (off: ``without=
["schedules"]``), on agent-runs' schedules (``RUNS_URL``) or ``LocalRuns``'.

* ``schedule_task(task, when, condition=None, once=False, timezone="UTC")``: ``when`` is a
  5-field cron the model writes, checked with ``croniter`` (agent-runs' own);
  ``list_schedules()``; ``update_schedule(id, ...)``; ``cancel_schedule(id)``. Each user sees
  and changes only their own (the run's user and tenant), and only those set in chat.
* A fired run is the same agent, the same user and the same thread: the schedule's metadata
  names the thread (agent-runs fires the run into it), so its reply is in that chat's history.
  It takes the agent's own ``run_timeout`` and budget, and its writes go through the same
  governance and approvals as any run's.
* With a ``condition``, the run is told to answer exactly :data:`NO_REPORT` when it does not
  hold. Such a run is not delivered: its output is ``NO_REPORT``, it is not written to the
  conversation, and the thread stream leaves it out. A condition that still holds is reported
  once: the next runs are quiet until it stops holding (:data:`REPORTED`, in the schedule's
  metadata). ``once=True`` deletes the schedule after its run.
* Limits: ``TRELLIS_SCHEDULE_MIN_INTERVAL`` and ``TRELLIS_SCHEDULE_MAX_PER_USER`` (``Settings``),
  per user ``Harness(schedule_limits=)``; agent-runs' own floor, rate limits and
  ``max_running_per_tenant`` still apply. A limit reached is a sentence the model tells the
  user, not an error.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final

from croniter import croniter

from trellis.contracts import RunStatus, Schedule, ScheduleSpec, ToolSpec
from trellis.harness.runs import THREAD
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import Tool
from trellis.runs import ConflictError, NotFoundError, ScheduleUpdate, ValidationError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.result import Result

log = logging.getLogger("trellis.schedules")

#: What a scheduled run answers when its condition does not hold: it is not delivered.
NO_REPORT: Final = "NO_REPORT"
#: The schedule metadata key of what the user asked (``task``, ``condition``, ``once``): it
#: marks a schedule set in chat.
TASK: Final = "trellis_task"
#: The schedule metadata key of whether the last run found the condition holding (reported).
REPORTED: Final = "trellis_reported"
#: The tools' names.
SCHEDULE_TOOLS: Final = ("schedule_task", "list_schedules", "update_schedule", "cancel_schedule")
#: Occurrences measured to find a cron's shortest interval (as agent-runs measures its floor).
PROBES: Final = 8
_PROBE_FROM: Final = datetime(2024, 1, 1, tzinfo=UTC)
#: A cron's fields: minute, hour, day of month, month, day of week.
CRON_FIELDS: Final = 5
MINUTE: Final = 60


@dataclass(frozen=True)
class ScheduleLimits:
    """One user's limits (``Harness(schedule_limits=)``); a field left ``None`` is the
    environment's."""

    #: the least seconds between two runs of a schedule
    min_interval: float | None = None
    #: how many schedules the user may have, per agent
    max_per_user: int | None = None


#: ``Harness(schedule_limits=)``: ``async (user, tenant) -> ScheduleLimits | None``.
LimitsHook = Callable[[str, str], Awaitable[ScheduleLimits | None]]


@dataclass
class Fired:
    """A run a chat schedule fired, as its metadata says: what decides whether it is
    delivered, and what its end changes on the schedule."""

    schedule_id: str
    condition: str | None
    once: bool
    #: whether the run before this one reported (the condition held)
    reported: bool
    #: whether this run found the condition holding (set when it answers)
    met: bool | None = None

    @classmethod
    def of(cls, metadata: dict[str, Any]) -> Fired | None:
        task, schedule_id = metadata.get(TASK), metadata.get("schedule_id")
        if not isinstance(task, dict) or not isinstance(schedule_id, str):
            return None
        return cls(
            schedule_id=schedule_id,
            condition=task.get("condition") or None,
            once=bool(task.get("once")),
            reported=bool(metadata.get(REPORTED)),
        )


# --------------------------------------------------------------------------- fired runs


def quiet(runtime: Runtime, answer: Any) -> bool:
    """Whether a run's answer is not delivered: a fired run that answered :data:`NO_REPORT`,
    or whose condition held already at the run before (reported once)."""
    fired = runtime.scheduled
    if fired is None:
        return False
    fired.met = not (isinstance(answer, str) and re.sub(r"\W", "", answer) == NO_REPORT)
    return not fired.met or (fired.condition is not None and fired.reported)


async def ended(agent: Agent, runtime: Runtime, result: Result) -> None:
    """After a fired run ended: a ``once`` schedule is deleted; a condition's state is kept
    when it changed. A failure is logged: the run's own result stands."""
    fired = runtime.scheduled
    if fired is None or result.status is RunStatus.PAUSED:
        return
    store = agent.harness.runs.schedules
    try:
        if fired.once:
            await store.delete(fired.schedule_id, tenant=runtime.tenant)
        elif fired.condition is not None and fired.met not in (None, fired.reported):
            changes = ScheduleUpdate(metadata={REPORTED: fired.met})
            await store.update(fired.schedule_id, changes, tenant=runtime.tenant)
    except NotFoundError:  # cancelled meanwhile
        pass
    except Exception:
        log.warning("schedule %s not updated after its run", fired.schedule_id, exc_info=True)


# --------------------------------------------------------------------------- the tools

_TASK: Final = {"type": "string"}
_WHEN: Final = {
    "type": "string",
    "description": "5-field cron: '0 9 * * *' daily 9:00, '0 9 * * 1-5' weekdays, "
    "'0 8 * * 1' Mondays 8:00, '0 9 1 * *' the 1st of the month.",
}
_CONDITION: Final = {"type": "string", "description": "Report only when this holds."}
_ID: Final = {"type": "string"}
#: ``update_schedule``'s ``when``: as ``schedule_task``'s, said once.
_NEW_WHEN: Final = {"type": "string", "description": "A new 5-field cron."}


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    side_effects = "read" if name == "list_schedules" else "write"
    schema = {"type": "object", "properties": properties, "required": required}
    return ToolSpec(
        name=name,
        description=description,
        input_schema=schema,
        source="local",
        side_effects=side_effects,
    )


SPECS: Final = (
    _spec(
        "schedule_task",
        "Run a task later or on a timetable for this user; the reply comes back to this chat.",
        {
            "task": _TASK,
            "when": _WHEN,
            "condition": _CONDITION,
            "once": {"type": "boolean", "description": "Run once, then end."},
            "timezone": {"type": "string", "description": "The user's IANA zone; default UTC."},
        },
        ["task", "when"],
    ),
    _spec("list_schedules", "The user's scheduled tasks.", {}, []),
    _spec(
        "update_schedule",
        "Change one of the user's scheduled tasks (an id from list_schedules); enabled=false "
        "pauses it, true resumes it.",
        {
            "id": _ID,
            "when": _NEW_WHEN,
            "task": {"type": "string"},
            "condition": {"type": "string"},
            "enabled": {"type": "boolean"},
        },
        ["id"],
    ),
    _spec("cancel_schedule", "Cancel one of the user's scheduled tasks.", {"id": _ID}, ["id"]),
)


def schedule_tools() -> list[Tool]:
    return [Tool(spec, _RUN[spec.name], feature="schedules") for spec in SPECS]


async def _schedule_task(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    task, when = str(args["task"]).strip(), str(args["when"]).strip()
    condition = (args.get("condition") or "").strip() or None
    min_interval, max_per_user = await _limits(runtime)
    if (refused := _refused(when, min_interval)) is not None:
        return refused
    mine = await _mine(runtime)
    if len(mine) >= max_per_user:
        return (
            f"You have {len(mine)} of {max_per_user} schedules: cancel one before adding another."
        )
    agent = runtime.agent
    about = {"task": task, "condition": condition, "once": bool(args.get("once"))}
    metadata = {**agent._metadata(runtime.without, None), TASK: about}
    if runtime.conversation != runtime.run_id:  # the chat it was set in
        metadata[THREAD] = runtime.identity.thread
    try:
        spec = ScheduleSpec(
            tenant_id=runtime.tenant,
            agent_id=agent.id,
            name=task[:200],
            cadence=when,
            timezone=str(args.get("timezone") or "UTC"),
            on_behalf_of=runtime.user,
            input=_input(task, condition),
            timeout_seconds=agent.timeout,
            agent_version=agent.version,
            metadata=metadata,
        )
        made = await agent.harness.runs.schedules.create(spec)
    except (ValueError, ValidationError, ConflictError) as exc:
        return f"That schedule was refused: {exc}"
    return f"Scheduled {made.schedule_id}: {_described(made)}"


async def _list_schedules(args: dict[str, Any]) -> Any:
    mine = await _mine(_runtime())
    return [_shown(s) for s in mine] if mine else "You have no schedules."


async def _update_schedule(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    schedule = await _own(runtime, str(args["id"]))
    if schedule is None:
        return f"No schedule {args['id']} of yours."
    changes: dict[str, Any] = {}
    if args.get("when"):
        min_interval, _ = await _limits(runtime)
        if (refused := _refused(str(args["when"]).strip(), min_interval)) is not None:
            return refused
        changes["cadence"] = str(args["when"]).strip()
    about = dict(schedule.metadata[TASK])
    if args.get("task") or "condition" in args:
        about["task"] = str(args.get("task") or about["task"]).strip()
        if "condition" in args:
            about["condition"] = (args["condition"] or "").strip() or None
        changes.update(
            name=about["task"][:200],
            input=_input(about["task"], about["condition"]),
            metadata={TASK: about, REPORTED: False},
        )
    if args.get("enabled") is not None:
        changes["enabled"] = bool(args["enabled"])
    store = runtime.agent.harness.runs.schedules
    try:
        updated = await store.update(
            schedule.schedule_id, ScheduleUpdate(**changes), tenant=runtime.tenant
        )
    except (ValidationError, ConflictError) as exc:
        return f"That change was refused: {exc}"
    return _shown(updated)


async def _cancel_schedule(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    schedule = await _own(runtime, str(args["id"]))
    if schedule is None:
        return f"No schedule {args['id']} of yours."
    await runtime.agent.harness.runs.schedules.delete(schedule.schedule_id, tenant=runtime.tenant)
    return f"Cancelled {schedule.schedule_id}: {schedule.metadata[TASK]['task']}."


_RUN: Final = {
    "schedule_task": _schedule_task,
    "list_schedules": _list_schedules,
    "update_schedule": _update_schedule,
    "cancel_schedule": _cancel_schedule,
}


# --------------------------------------------------------------------------- helpers


def _runtime() -> Runtime:
    runtime = current()
    assert runtime is not None  # the bridge runs a tool inside a run
    return runtime


def _input(task: str, condition: str | None) -> str:
    """What each fired run is asked."""
    said = f"Scheduled task: {task}"
    if condition is None:
        return said
    return (
        f"{said}\nReport only if this holds: {condition}. If it does not, reply exactly "
        f"{NO_REPORT} and nothing else."
    )


async def _limits(runtime: Runtime) -> tuple[float, int]:
    settings, hook = runtime.agent.harness.settings, runtime.agent.harness.schedule_limits
    own = await hook(runtime.user, runtime.tenant) if hook is not None else None
    own = own or ScheduleLimits()
    return (
        settings.schedule_min_interval if own.min_interval is None else own.min_interval,
        settings.schedule_max_per_user if own.max_per_user is None else own.max_per_user,
    )


def _refused(when: str, min_interval: float) -> str | None:
    """Why ``when`` cannot be a schedule's time, in a sentence; ``None`` when it can."""
    if len(when.split()) != CRON_FIELDS or not croniter.is_valid(when):
        return (
            f"'{when}' is not a cron expression: write 5 fields (minute hour day-of-month month "
            "day-of-week), e.g. '0 9 * * *' for daily at 9:00."
        )
    cursor = croniter(when, _PROBE_FROM)
    times = [cursor.get_next(datetime) for _ in range(PROBES)]
    shortest = min((b - a).total_seconds() for a, b in pairwise(times))
    if shortest < min_interval:
        least = (
            f"{min_interval / MINUTE:g} minutes"
            if min_interval >= MINUTE
            else f"{min_interval:g} seconds"
        )
        return f"'{when}' runs more often than every {least}, the least allowed: pick a longer one."
    return None


async def _mine(runtime: Runtime) -> list[Schedule]:
    """The run's user's schedules set in chat with this agent, newest first."""
    store, found, cursor = runtime.agent.harness.runs.schedules, [], None
    while True:
        page = await store.list(
            agent_id=runtime.agent.id,
            on_behalf_of=runtime.user,
            cursor=cursor,
            tenant=runtime.tenant,
        )
        found += [s for s in page.items if _theirs(runtime, s)]
        if (cursor := page.next_cursor) is None:
            return found


async def _own(runtime: Runtime, schedule_id: str) -> Schedule | None:
    found = await runtime.agent.harness.runs.schedules.get(schedule_id, tenant=runtime.tenant)
    return found if found is not None and _theirs(runtime, found) else None


def _theirs(runtime: Runtime, schedule: Schedule) -> bool:
    return (
        schedule.on_behalf_of == runtime.user
        and schedule.agent_id == runtime.agent.id
        and isinstance(schedule.metadata.get(TASK), dict)
    )


def _shown(schedule: Schedule) -> dict[str, Any]:
    about = schedule.metadata[TASK]
    due = schedule.next_fire_at if schedule.enabled else None
    return {
        "id": schedule.schedule_id,
        "task": about["task"],
        "when": schedule.cadence,
        "timezone": schedule.timezone,
        "condition": about.get("condition"),
        "once": about.get("once", False),
        "status": "active" if schedule.enabled else "paused",
        "next": due.isoformat() if due is not None else None,
    }


def _described(schedule: Schedule) -> str:
    shown = _shown(schedule)
    said = f"{shown['task']}, '{shown['when']}' ({shown['timezone']})"
    if shown["condition"]:
        said += f", reported only when {shown['condition']}"
    return f"{said}; next run {shown['next']}."
