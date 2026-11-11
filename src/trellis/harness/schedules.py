"""Schedules a user sets in chat ("every Monday at 8 check stock and tell me if under 5"): four
built-in tools every adapter gets while the run store can schedule (off: ``without=
["schedules"]``), on agent-runs' schedules (``RUNS_URL``) or ``LocalRuns``'.

* ``schedule_task(task, when, condition=None, once=False, in_minutes=None, timezone=None)``:
  ``when`` is a 5-field cron the model writes, checked with ``croniter`` (agent-runs' own);
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
* The user's zone: the one the model names, else the one their app sends
  (``forwardedProps.timezone``, ``frontend.py``), else UTC. Every run reads the time now in
  it in ``schedule_task``'s description (:func:`now_said`), so "tomorrow at 9" has a date;
  "in 2 hours" is ``in_minutes``. A fired run is offered none of these tools: it does its
  task.
* Asked with buttons (``Runtime.ask(options=)``) where the answer changes the outcome: a
  schedule like one the user has (:func:`_twin`: update it, recommended, keep both, or
  neither; the very same again is "already set"), ``cancel_schedule(id="all")``, several that
  fit the user's words (``ids``) and a vague ask (``choices``).
* Unattended safety, with no setting: after :data:`MAX_FAILURES` failed runs in a row
  (``ERROR``, ``TIMEOUT``) the schedule is paused and the last run's error says so (the reply in
  the thread, the run's record and its ``run.finished`` webhook); a schedule the user has not
  touched for :data:`STALE` asks at its next run "Still need this?" (Keep, recommended, or
  Cancel), paused until answered. The count and the touch are in the schedule's metadata
  (:data:`FAILURES`, :data:`TOUCHED`). agent-runs runs one fire of a schedule at a time and
  spreads the fires over the minute.
* Limits: ``TRELLIS_SCHEDULE_MIN_INTERVAL`` and ``TRELLIS_SCHEDULE_MAX_PER_USER`` (``Settings``),
  per user ``Harness(schedule_limits=)``; agent-runs' own floor, rate limits and
  ``max_running_per_tenant`` still apply. A limit reached is a sentence the model tells the
  user, not an error.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from trellis.contracts import AgentError, Option, RunStatus, Schedule, ScheduleSpec, ToolSpec
from trellis.harness import snapshots
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
#: The schedule metadata key of its runs' failures in a row.
FAILURES: Final = "trellis_failures"
#: Failed runs in a row that pause a schedule (as agent-runs pauses one whose fires fail).
MAX_FAILURES: Final = 3
#: The schedule metadata key of when the user last made or changed it (ISO 8601).
TOUCHED: Final = "trellis_touched"
#: How long a schedule runs untouched before its next run asks whether it is still needed.
STALE: Final = timedelta(days=90)
#: The tools' names.
SCHEDULE_TOOLS: Final = ("schedule_task", "list_schedules", "update_schedule", "cancel_schedule")
#: Occurrences measured to find a cron's shortest interval (as agent-runs measures its floor).
PROBES: Final = 8
_PROBE_FROM: Final = datetime(2024, 1, 1, tzinfo=UTC)
#: A cron's fields: minute, hour, day of month, month, day of week.
CRON_FIELDS: Final = 5
MINUTE: Final = 60
#: The most schedules a vague ask is offered to pick from.
CHOICES: Final = 3
#: The values of the options the user is asked with: update it (or cancel all), keep both,
#: neither.
UPDATE: Final = "update"
BOTH: Final = "both"
NEITHER: Final = "none"
#: The values of the options a schedule whose task names writes is confirmed with: schedule it
#: allowing them in its runs, or schedule it asking at each run (or :data:`NEITHER`).
ALLOW: Final = "allow"
EACH_TIME: Final = "ask"
#: The schedule metadata key of the writes the user allowed its runs when they confirmed it:
#: a fired run calls them without asking (``tools.bridge``); any other write still asks.
ALLOWED: Final = "trellis_allowed"
#: How alike two tasks' words are (``difflib``'s ratio, digits left out) for one to be the
#: other with a different time or value, where they name no identifier in common.
ALIKE: Final = 0.8


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
    #: what the user asked, as they said it
    task: str = ""
    #: its runs' failures in a row before this one
    failures: int = 0
    #: when the user last made or changed it
    touched: datetime | None = None
    #: whether this run found the condition holding (set when it answers)
    met: bool | None = None
    #: the writes the user allowed its runs when they confirmed it (:data:`ALLOWED`)
    allowed: tuple[str, ...] = ()

    @classmethod
    def of(cls, metadata: dict[str, Any]) -> Fired | None:
        task, schedule_id = metadata.get(TASK), metadata.get("schedule_id")
        if not isinstance(task, dict) or not isinstance(schedule_id, str):
            return None
        touched = metadata.get(TOUCHED)
        return cls(
            schedule_id=schedule_id,
            condition=task.get("condition") or None,
            once=bool(task.get("once")),
            reported=bool(metadata.get(REPORTED)),
            task=str(task.get("task") or ""),
            failures=int(metadata.get(FAILURES) or 0),
            touched=datetime.fromisoformat(touched) if isinstance(touched, str) else None,
            allowed=tuple(str(name) for name in metadata.get(ALLOWED) or ()),
        )


# --------------------------------------------------------------------------- the user's time


def zone_of(name: Any) -> str | None:
    """``name`` when it is an IANA zone, else ``None``."""
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError):
        return None
    return name.strip()


def now_said(runtime: Runtime) -> str:
    """When it is, as ``schedule_task``'s description says it: the day and the hour in the
    user's zone (not the minute: the tool's text stays the same for an hour, as a prompt cache
    keeps it; ``in_minutes`` counts from the minute), or in UTC, said as not theirs."""
    return snapshots.said(snapshots.now(), snapshots.zone(runtime), hour=True)


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
    when it changed; a success clears the failures in a row. A failure is logged: the run's
    own result stands."""
    fired = runtime.scheduled
    if fired is None or result.status is RunStatus.PAUSED:
        return
    store = agent.harness.runs.schedules
    kept: dict[str, Any] = {}
    if fired.condition is not None and fired.met not in (None, fired.reported):
        kept[REPORTED] = fired.met
    if fired.failures and result.status in (RunStatus.SUCCESS, RunStatus.PARTIAL):
        kept[FAILURES] = 0
    try:
        if fired.once:
            await store.delete(fired.schedule_id, tenant=runtime.tenant)
        elif kept:
            await store.update(
                fired.schedule_id, ScheduleUpdate(metadata=kept), tenant=runtime.tenant
            )
    except NotFoundError:  # cancelled meanwhile
        pass
    except Exception:
        log.warning("schedule %s not updated after its run", fired.schedule_id, exc_info=True)


async def failed(agent: Agent, runtime: Runtime, error: AgentError) -> AgentError:
    """A fired run's failure, counted on its schedule before the run ends: the
    :data:`MAX_FAILURES`-th in a row pauses the schedule, and the error then says so (the
    Kubernetes CronJob and Temporal leave a failing schedule running; an unattended one that
    fails every time only costs)."""
    fired = runtime.scheduled
    if fired is None or fired.once:
        return error
    fired.failures += 1
    stop = fired.failures >= MAX_FAILURES
    counted = {FAILURES: fired.failures}
    change = (
        ScheduleUpdate(metadata=counted, enabled=False)
        if stop
        else ScheduleUpdate(metadata=counted)
    )
    try:
        await agent.harness.runs.schedules.update(fired.schedule_id, change, tenant=runtime.tenant)
    except NotFoundError:
        return error
    except Exception:
        log.warning("schedule %s: failure not counted", fired.schedule_id, exc_info=True)
        return error
    if not stop:
        return error
    said = (
        f"{error.message} This schedule failed {fired.failures} times in a row and is paused: "
        f"{fired.task}. Ask me to resume it once the cause is fixed."
    )
    return error.model_copy(update={"message": said})


async def stale(runtime: Runtime) -> str | None:
    """Before a fired run works: a schedule the user has not touched for :data:`STALE` asks
    once whether it is still needed, paused until they answer. Keep (recommended) resumes it
    and the run does its task (``None``); Cancel deletes it and is the run's answer."""
    fired = runtime.scheduled
    if fired is None or fired.touched is None or datetime.now(UTC) - fired.touched < STALE:
        return None
    store, tenant = runtime.agent.harness.runs.schedules, runtime.tenant
    await store.update(fired.schedule_id, ScheduleUpdate(enabled=False), tenant=tenant)
    keep = await runtime.ask(
        f"Still need this? {fired.task}",
        options=[
            Option(value=UPDATE, label="Keep", recommended=True),
            Option(value=NEITHER, label="Cancel"),
        ],
    )
    if keep == UPDATE:
        change = ScheduleUpdate(enabled=True, metadata={TOUCHED: _now()})
        await store.update(fired.schedule_id, change, tenant=tenant)
        return None
    await store.delete(fired.schedule_id, tenant=tenant)
    runtime.scheduled = None  # gone: nothing of it to keep after the run
    return f"Cancelled: {fired.task}."


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- the tools

_TASK: Final = {"type": "string", "description": "What to do, as the user said it."}
_WHEN: Final = {
    "type": "string",
    "description": "5-field cron in the user's zone: '0 9 * * *' daily 9:00, '0 9,17 * * *' "
    "9:00 and 17:00, '0 9 * * 1-5' weekdays, '0 8 * * 1' Mondays, '0 9 1 * *' the 1st; one "
    "day ('tomorrow at 9', 'next Monday at 10') counted from today: '0 9 4 11 *' with once.",
}
_CONDITION: Final = {"type": "string", "description": "Report only when this holds."}
_ID: Final = {"type": "string", "description": "An id from list_schedules."}
#: The schedules the user's words fit, when several do: the user picks one.
_IDS: Final = {
    "type": "array",
    "items": {"type": "string"},
    "description": "When the user's words fit several: all their ids; the user picks one.",
}
#: ``update_schedule``'s and a choice's ``when``: as ``schedule_task``'s, said once.
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
        "Run a task later or on a timetable when the user asks for that ('every Monday...', "
        "'tomorrow at 9...', 'tell me when...'), not when a question only mentions a time or a "
        "habit; the reply comes back to this chat. A condition: first check it once now with "
        "your tools and tell the user the value; one that holds already: say so, and "
        "schedule only if they want to hear again. If no tool of yours can check it, do not "
        "schedule: say so and offer two or three things your tools can watch, by name. A "
        "vague ask ('keep an eye on sales'): call with choices, 2-3 concrete schedules your "
        "tools can run, the best first; the user picks one. No time said: "
        "9:00. A schedule like one the user has is asked about here. One schedule a request.",
        {
            "task": _TASK,
            "when": _WHEN,
            "condition": _CONDITION,
            "once": {"type": "boolean", "description": "Run once, then end."},
            "in_minutes": {
                "type": "number",
                "description": "Instead of when: run once this many minutes from now "
                "('in 2 hours': 120).",
            },
            "choices": {
                "type": "array",
                "description": "Only for a vague ask: 2-3 concrete schedules, the best first; "
                "the user picks one, which is made.",
                "items": {
                    "type": "object",
                    "properties": {"task": _TASK, "when": _NEW_WHEN, "condition": _CONDITION},
                    "required": ["task", "when"],
                },
            },
            "timezone": {
                "type": "string",
                "description": "An IANA zone, only when the user names one; default theirs.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tools its runs will call that change something: the user "
                "allows them now.",
            },
        },
        ["task"],
    ),
    _spec(
        "list_schedules",
        "The user's scheduled tasks: read them when the user refers to one ('the stock one', "
        "'my reports'). A reference that fits one is that one; one that fits several: pass "
        "all their ids to update_schedule or cancel_schedule, and the user picks.",
        {},
        [],
    ),
    _spec(
        "update_schedule",
        "Change one of the user's scheduled tasks (an id from list_schedules): its time, task "
        "or condition; enabled=false pauses it, true resumes it.",
        {
            "id": _ID,
            "ids": _IDS,
            "when": _NEW_WHEN,
            "task": {"type": "string"},
            "condition": {"type": "string"},
            "enabled": {"type": "boolean"},
        },
        ["id"],
    ),
    _spec(
        "cancel_schedule",
        "Cancel one of the user's scheduled tasks, or id 'all' (the user is asked to confirm).",
        {
            "id": {"type": "string", "description": "An id from list_schedules, or 'all'."},
            "ids": _IDS,
        },
        ["id"],
    ),
)


def schedule_tools(runtime: Runtime | None = None) -> list[Tool]:
    """The tools; in a run, ``schedule_task``'s description says when it is (:func:`now_said`)."""
    specs = list(SPECS)
    if runtime is not None:
        dated = f"{specs[0].description} {now_said(runtime)}"
        specs[0] = specs[0].model_copy(update={"description": dated})
    return [Tool(spec, _RUN[spec.name], feature="schedules", confined=True) for spec in specs]


async def _schedule_task(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    if choices := args.get("choices"):  # a vague ask: the user picks one of these
        return await _chosen(runtime, args, list(choices)[:CHOICES])
    task, when = str(args["task"]).strip(), str(args.get("when") or "").strip()
    condition = (args.get("condition") or "").strip() or None
    zone = str(args.get("timezone") or "").strip() or snapshots.zone(runtime) or "UTC"
    once = bool(args.get("once"))
    counted = ""
    if minutes := args.get("in_minutes"):  # once, at that minute
        start = datetime.now(ZoneInfo(zone_of(zone) or "UTC"))
        at = start + timedelta(minutes=float(minutes))
        when, once = f"{at.minute} {at.hour} {at.day} {at.month} *", True
        # the minute counted from, said where the model reads the time it got: the tool's
        # text says only the hour (cached), so 03:53 for "in 2 hours" at 01:53 is not an error
        counted = f" That is {float(minutes):g} minutes from {start:%H:%M} ({zone}), as asked."
    min_interval, max_per_user = await _limits(runtime)
    if (refused := _refused(when, min_interval)) is not None:
        return refused
    mine = await _mine(runtime)
    about = {"task": task, "condition": condition, "once": once}
    twin = await _twin_asked(runtime, mine, about, when, zone)
    if (said := twin or _full(mine, max_per_user)) is not None:
        return said
    agent = runtime.agent
    metadata = {**agent._metadata(runtime.without, None), TASK: about, TOUCHED: _now()}
    if runtime.conversation != runtime.run_id:  # the chat it was set in
        metadata[THREAD] = runtime.identity.thread
    writes = await _asking(runtime, args.get("tools") or [])
    if writes:
        picked = await _writes_asked(runtime, _said(about, when, zone), writes)
        if picked == NEITHER:
            return "Nothing scheduled: the user did not want it."
        if picked == ALLOW:
            metadata[ALLOWED] = writes
    try:
        spec = ScheduleSpec(
            tenant_id=runtime.tenant,
            agent_id=agent.id,
            name=task[:200],
            cadence=when,
            timezone=zone,
            on_behalf_of=runtime.user,
            input=_input(task, condition),
            timeout_seconds=agent.timeout,
            agent_version=agent.version,
            metadata=metadata,
        )
        made = await agent.harness.runs.schedules.create(spec)
    except (ValueError, ValidationError, ConflictError) as exc:
        return f"That schedule was refused: {exc}"
    return f"Scheduled {made.schedule_id}: {_described(made)}{counted}"


def _full(mine: list[Schedule], max_per_user: int) -> str | None:
    """What the user is told when they have as many schedules as they may, else ``None``."""
    if len(mine) < max_per_user:
        return None
    return f"You have {len(mine)} of {max_per_user} schedules: cancel one before adding another."


async def _asking(runtime: Runtime, names: list[Any]) -> list[str]:
    """Of the tools ``names`` says a schedule's runs will call, those a call of would ask a
    person about because they write (or say nothing of what they do), in the run's own tool
    set: never an ``irreversible`` one, which asks every call, nor one that does not ask."""
    governance = runtime.agent.harness._governance(runtime.tenant)
    asking: list[str] = []
    for name in dict.fromkeys(str(n) for n in names):
        found = runtime.toolbox.get(name)
        spec = found.spec if found is not None else runtime.native.get(name)
        if spec is None or name not in runtime.own_tools():
            continue
        decision = await governance.check(
            name, {}, side_effects=spec.side_effects, agent_id=runtime.agent_id
        )
        confined = found is not None and found.confined  # announced, never asked
        if decision.asks and decision.risk != "irreversible" and not confined:
            asking.append(name)
    return asking


async def _writes_asked(runtime: Runtime, said: str, writes: list[str]) -> Any:
    """The user's confirmation of a schedule whose runs will call ``writes``: schedule it
    allowing them, schedule it asking at each run, or not at all."""
    named = ", ".join(writes)
    return await runtime.ask(
        f"Schedule {said}? Its runs will call {named}, which change things.",
        options=[
            Option(value=ALLOW, label=f"Schedule it, allowing {named} in its runs"),
            Option(value=EACH_TIME, label="Schedule it, asking me at each run"),
            Option(value=NEITHER, label="Do not schedule it"),
        ],
    )


async def _twin_asked(
    runtime: Runtime, mine: list[Schedule], about: dict[str, Any], when: str, zone: str
) -> Any:
    """What a schedule like one the user has comes to: nothing new (the same again), or the one
    they have updated, or none, as the user picks; ``None``: make it (no twin, or both kept)."""
    twin = _twin(mine, about)
    if twin is None:
        return None
    if _same(twin, about, when, zone):
        return f"Already set, nothing changed: {_described(twin)}"
    said = _said(about, when, zone)
    choice = await runtime.ask(
        f"You already have a schedule like this: {_described(twin)} Update it to {said}, "
        "or keep both?",
        options=[
            Option(
                value=UPDATE,
                label=f"Update it to {said}",
                description="what you asked for now",
                recommended=True,
            ),
            Option(value=BOTH, label="Keep both"),
            Option(value=NEITHER, label="Cancel"),
        ],
    )
    if choice == UPDATE:
        return await _changed(runtime, twin, {"cadence": when, "timezone": zone, **_about(about)})
    return None if choice == BOTH else "Nothing scheduled: the user keeps the one they have."


async def _chosen(runtime: Runtime, args: dict[str, Any], choices: list[Any]) -> Any:
    """The schedule the user picked among ``choices`` (the best first), made; or none."""
    zone = str(args.get("timezone") or "").strip() or snapshots.zone(runtime) or "UTC"
    options = [
        Option(value=str(n), label=_said(c, str(c.get("when") or ""), zone))
        for n, c in enumerate(choices)
        if isinstance(c, dict) and c.get("task")
    ]
    if not options:
        return "Each choice needs a task and a when."
    picked = await runtime.ask(
        "Which should I set up?", options=[*options, Option(value=NEITHER, label="None of these")]
    )
    if picked == NEITHER:
        return "Nothing scheduled: the user wants none of these."
    return await _schedule_task({**args, "choices": None, **choices[int(picked)]})


async def _list_schedules(args: dict[str, Any]) -> Any:
    mine = await _mine(_runtime())
    return [_shown(s) for s in mine] if mine else "You have no schedules."


async def _update_schedule(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    schedule = await _which(runtime, args, "change")
    if isinstance(schedule, str):
        return schedule
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
        changes.update(_about(about))
    if args.get("enabled") is not None:
        changes["enabled"] = bool(args["enabled"])
    return await _changed(runtime, schedule, changes)


async def _changed(runtime: Runtime, schedule: Schedule, changes: dict[str, Any]) -> Any:
    store = runtime.agent.harness.runs.schedules
    # the user's own change: touched now, failures cleared
    changes = {**changes, "metadata": {**changes.get("metadata", {}), TOUCHED: _now(), FAILURES: 0}}
    try:
        updated = await store.update(
            schedule.schedule_id, ScheduleUpdate(**changes), tenant=runtime.tenant
        )
    except (ValidationError, ConflictError) as exc:
        return f"That change was refused: {exc}"
    return _shown(updated)


async def _cancel_schedule(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    store = runtime.agent.harness.runs.schedules
    if str(args.get("id")).strip().lower() == "all":
        mine = await _mine(runtime)
        if not mine:
            return "You have no schedules."
        sure = await runtime.ask(
            f"Cancel all {len(mine)} of your schedules?",
            options=[
                Option(value=UPDATE, label=f"Cancel all {len(mine)}"),
                Option(value=NEITHER, label="Keep them"),
            ],
        )
        if sure != UPDATE:
            return "Nothing cancelled: the user keeps them."
        for schedule in mine:
            await store.delete(schedule.schedule_id, tenant=runtime.tenant)
        return f"Cancelled all {len(mine)}: " + "; ".join(_label(s) for s in mine)
    schedule = await _which(runtime, args, "cancel")
    if isinstance(schedule, str):
        return schedule
    await store.delete(schedule.schedule_id, tenant=runtime.tenant)
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


async def _which(runtime: Runtime, args: dict[str, Any], verb: str) -> Schedule | str:
    """The schedule a call names (``id``), or — the user's words fitting several (``ids``) —
    the one the user picks; a sentence when there is none."""
    ids = [str(i) for i in args.get("ids") or [] if i] or [str(args.get("id") or "")]
    found = [s for i in dict.fromkeys(ids) if (s := await _own(runtime, i)) is not None]
    if not found:
        return f"No schedule {', '.join(ids)} of yours."
    if len(found) == 1:
        return found[0]
    picked = await runtime.ask(
        f"Which one should I {verb}?",
        options=[Option(value=s.schedule_id, label=_label(s)) for s in found],
    )
    return next(s for s in found if s.schedule_id == picked)  # the answer is one of them


def _words(about: dict[str, Any]) -> str:
    return f"{about.get('task') or ''} {about.get('condition') or ''}".lower()


def _names(text: str) -> set[str]:
    """The identifiers a task names (a SKU, an order number): words of letters and digits
    that are not a time of day."""
    return {
        w
        for w in re.findall(r"[a-z0-9][\w-]*", text)
        if re.search(r"\d", w) and re.search(r"[a-z]", w) and not re.fullmatch(r"\d+(am|pm|h)", w)
    }


def _twin(mine: list[Schedule], about: dict[str, Any]) -> Schedule | None:
    """The user's schedule that watches what ``about`` does, perhaps with another time or
    value: one naming an identifier in common, else one whose words are alike."""
    words = _words(about)
    for schedule in mine:
        theirs = _words(schedule.metadata[TASK])
        if _names(words) & _names(theirs):
            return schedule
    plain = re.sub(r"\d", "", words)
    for schedule in mine:
        theirs = re.sub(r"\d", "", _words(schedule.metadata[TASK]))
        if difflib.SequenceMatcher(None, plain, theirs).ratio() >= ALIKE:
            return schedule
    return None


def _same(schedule: Schedule, about: dict[str, Any], when: str, zone: str) -> bool:
    """Whether ``about`` at ``when`` is ``schedule`` again: its time, zone and the figures and
    presence of its condition."""
    theirs = schedule.metadata[TASK]

    def figures(condition: Any) -> tuple[bool, list[str]]:
        found = re.findall(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])", str(condition or ""))
        return condition is not None, found

    return (
        schedule.cadence == when
        and schedule.timezone == zone
        and bool(theirs.get("once")) == about["once"]
        and figures(theirs.get("condition")) == figures(about["condition"])
    )


def _about(about: dict[str, Any]) -> dict[str, Any]:
    """The changes that make a schedule ``about``'s task and condition."""
    return {
        "name": about["task"][:200],
        "input": _input(about["task"], about["condition"]),
        "metadata": {TASK: about, REPORTED: False},
    }


def _said(about: dict[str, Any], when: str, zone: str) -> str:
    said = f"{about.get('task')}, '{when}' ({zone})"
    return f"{said}, only when {about['condition']}" if about.get("condition") else said


def _label(schedule: Schedule) -> str:
    return _said(schedule.metadata[TASK], schedule.cadence, schedule.timezone)


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
