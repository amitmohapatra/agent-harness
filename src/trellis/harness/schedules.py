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
  schedule the model judged like one the user has (``alike``, read from ``list_schedules``:
  update it, recommended, keep both, or neither; the very same again, :func:`_same`, is
  "already set" whatever was passed), ``cancel_schedule(id="all")``, several that
  fit the user's words (``ids``) and a vague ask (``choices``).
* Unattended safety: after ``TRELLIS_SCHEDULE_PAUSE_AFTER_FAILURES`` failed runs in a row
  (``ERROR``, ``TIMEOUT``) the schedule is paused and the last run's error says so (the reply in
  the thread, the run's record and its ``run.finished`` webhook); a schedule the user has not
  touched for ``TRELLIS_SCHEDULE_EXPIRE_DAYS`` asks at its next run "Still need this?" (Keep,
  recommended, or Cancel), paused until answered. The count and the touch are in the
  schedule's metadata (:data:`FAILURES`, :data:`TOUCHED`). agent-runs runs one fire of a
  schedule at a time and spreads the fires over the minute.
* Limits: ``TRELLIS_SCHEDULE_*`` (``Settings``: the least time between runs, schedules per
  user, each fired run's time and tool calls, failures before a pause, days before asking),
  per user ``Harness(schedule_limits=)``, the same names; agent-runs' own floor, rate limits and
  ``max_running_per_tenant`` still apply. A limit reached is a sentence the model tells the
  user, not an error.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time as time_module
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from trellis.contracts import (
    AgentError,
    Option,
    RunStatus,
    Schedule,
    ScheduleSpec,
    ToolSpec,
    stable_id,
)
from trellis.harness import snapshots
from trellis.harness.budget import Budget
from trellis.harness.notifications import Kind, Notification
from trellis.harness.runs import ERROR_SAID, THREAD
from trellis.harness.runtime import Runtime, current
from trellis.harness.settings import Settings
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
#: Failed runs in a row that pause a schedule, by default (``TRELLIS_SCHEDULE_PAUSE_AFTER_
#: FAILURES``; agent-runs pauses one whose fires fail as it does).
MAX_FAILURES: Final[int] = Settings.model_fields["schedule_pause_after_failures"].default
#: The schedule metadata key of when the user last made or changed it (ISO 8601).
TOUCHED: Final = "trellis_touched"
#: The schedule metadata key of its zone, for its runs: a fired run has no app to say it.
ZONE: Final = "trellis_zone"
#: The schedule metadata key of how a condition reports (:data:`REPORTS`), when not the default.
REPORT: Final = "trellis_report"
#: How a condition reports: until it clears (once while it holds, again after it stopped: the
#: default), every run it holds, or once and then the schedule ends.
REPORTS: Final = ("until_clears", "every_time", "once")
#: The schedule metadata key of why it is paused, when it was paused for a reason the user is
#: told (its runs failed, its one run failed, the user paused it, it waits to be kept).
PAUSED_BY: Final = "trellis_paused_by"
#: The schedule metadata key of its fired run that waits on a person, while it waits: the store
#: ends one nobody answers (its deadline), never the harness, so the next run counts it.
WAITS: Final = "trellis_waits"
#: How long a schedule runs untouched before its next run asks whether it is still needed, by
#: default (``TRELLIS_SCHEDULE_EXPIRE_DAYS``).
STALE: Final = timedelta(days=Settings.model_fields["schedule_expire_days"].default)
#: How long agent-runs' schedule policy read is kept before it is read again, in seconds.
FLOOR_KEPT: Final = 60.0
#: Why a schedule unchanged for its days is paused (``paused_because``).
KEEP_IT: Final = "it waits for the user to keep it"
#: What a fired run of such a schedule is told, after its task: its model asks the user, in
#: their words and language, never a fixed text of code (handoff 2.6).
EXPIRING: Final = (
    "This schedule has run unchanged for a long time and is now paused. Do the task, then end "
    "your reply by asking the user, in their own language, whether they still need this "
    "schedule: saying so keeps it (it resumes), or they can cancel it. Ask even if there is "
    "nothing to report."
)
#: The tools' names.
SCHEDULE_TOOLS: Final = ("schedule_task", "list_schedules", "update_schedule", "cancel_schedule")
#: Occurrences measured to find a cron's shortest interval (as agent-runs measures its floor).
PROBES: Final = 8
_PROBE_FROM: Final = datetime(2024, 1, 1, tzinfo=UTC)
#: A cron's fields: minute, hour, day of month, month, day of week.
CRON_FIELDS: Final = 5
MINUTE: Final = 60
#: The longest ``in_minutes``: a cron names no year, so a year less its last day.
YEAR: Final = 364 * 24 * MINUTE
#: The most schedules a vague ask is offered to pick from.
CHOICES: Final = 3
#: The values of the options the user is asked with: update it (or cancel all), keep both,
#: neither.
UPDATE: Final = "update"
BOTH: Final = "both"
NEITHER: Final = "none"
#: What an answer to a schedule question may be besides its suggestions, as a JSON Schema the
#: app renders from the user's own settings (a time picker, the days to tick, a calendar, or
#: words): how often, which days, the time, a date, when it ends. The model reads the answer
#: and makes the schedule it says; nothing here turns it into one.
WHEN_ANSWER: Final = {
    "type": "object",
    "title": "When",
    "properties": {
        "repeat": {"type": "string", "title": "How often",
                   "enum": ["once", "daily", "weekdays", "weekly", "monthly"]},
        "days": {"type": "array", "title": "Days", "uniqueItems": True,
                 "items": {"type": "string",
                           "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}},
        "time": {"type": "string", "format": "time", "title": "Time"},
        "date": {"type": "string", "format": "date", "title": "Date"},
        "ends": {"type": "string", "format": "date", "title": "Ends"},
    },
}  # fmt: skip
#: What the model reads of an answer that is none of the suggestions.
OWN_ANSWER: Final = "Nothing scheduled yet: the user answered in their own words: {answer}"
#: The confirmation's recommended answer: make the schedule with each open detail as offered.
CONFIRM: Final = "confirm"
#: The schedule metadata key of what its unattended runs do with a call nobody approved: skip it
#: and say so in the reply (:data:`UNATTENDED`'s first, the default), or wait for its user.
UNAPPROVED: Final = "trellis_unapproved"
UNATTENDED: Final = ("skip", "wait")
#: What the model reads of what the user assumed nothing about (``inferred``): those details
#: of a schedule, confirmed with the user in its one question.
INFERRED: Final = ("time", "days", "zone", "condition", "end")
#: The schedule metadata key of the writes the user allowed its runs when they confirmed it:
#: a fired run calls them without asking (``tools.bridge``); any other write still asks.
ALLOWED: Final = "trellis_allowed"


@dataclass(frozen=True)
class ScheduleLimits:
    """One user's limits (``Harness(schedule_limits=)``); a field left ``None`` is the
    environment's, ``Settings.schedule_<field>``: the same names, one source each."""

    #: the least seconds between two runs of a schedule
    min_interval: float | None = None
    #: how many schedules the user may have, per agent
    max_per_user: int | None = None
    #: the most seconds one fired run may take
    run_timeout: float | None = None
    #: the most harness tool calls one fired run may make
    run_max_tool_calls: int | None = None
    #: fired runs failed in a row that pause the schedule
    pause_after_failures: int | None = None
    #: days a schedule runs unchanged before its next run asks to keep it
    expire_days: int | None = None


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
    #: the schedule's zone: the day and hour its runs are told
    zone: str | None = None
    #: how its condition reports (:data:`REPORTS`)
    report: str = REPORTS[0]
    #: whether this run asks the user to keep it (unchanged for its days: :func:`stale`)
    expiring: bool = False
    #: whether its answer is delivered (:func:`quiet`), once it answered
    delivered: bool | None = None
    #: why this run paused its schedule (:func:`failed`), when it did
    paused: str | None = None
    #: what a call nobody approved does in its runs (:data:`UNATTENDED`); a schedule made
    #: before it was asked waits, as it did
    unapproved: str = UNATTENDED[1]
    #: its run before this one, when that one waited on a person (:data:`WAITS`)
    waits: str | None = None

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
            zone=metadata.get(ZONE) if isinstance(metadata.get(ZONE), str) else None,
            report=str(metadata.get(REPORT) or REPORTS[0]),
            unapproved=str(metadata.get(UNAPPROVED) or UNATTENDED[1]),
            waits=metadata.get(WAITS) if isinstance(metadata.get(WAITS), str) else None,
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
    fired.delivered = not _quiet(fired, answer)
    return not fired.delivered


def _quiet(fired: Fired, answer: Any) -> bool:
    fired.met = not (isinstance(answer, str) and re.sub(r"\W", "", answer) == NO_REPORT)
    if fired.expiring or fired.paused:  # its question (its pause) reaches the user, met or not
        return not fired.met
    held_before = fired.condition is not None and fired.reported
    return not fired.met or (held_before and fired.report == REPORTS[0])


async def ended(agent: Agent, runtime: Runtime, result: Result) -> None:
    """After a fired run ended: a ``once`` schedule is deleted; a condition's state is kept
    when it changed; a success clears the failures in a row. A failure is logged: the run's
    own result stands."""
    fired = runtime.scheduled
    if fired is None:
        return
    await _told(agent, runtime, fired, result)
    store = agent.harness.runs.schedules
    # a run that paused its schedule did nothing else: it clears no failure, ends no schedule
    ran = result.status in (RunStatus.SUCCESS, RunStatus.PARTIAL) and not fired.paused
    waits = result.status is RunStatus.PAUSED  # on a person: the next run sees how it ended
    kept, change = ({WAITS: runtime.run_id}, {}) if waits else _after(fired, ran=ran)
    told = fired.report == REPORTS[2] and fired.condition is not None and bool(fired.met)
    try:
        if (fired.once and ran) or told:  # done: a failed one-time run stays, paused
            await store.delete(fired.schedule_id, tenant=runtime.tenant)
        elif kept or change:
            await store.update(
                fired.schedule_id, ScheduleUpdate(metadata=kept, **change), tenant=runtime.tenant
            )
    except NotFoundError:  # cancelled meanwhile
        pass
    except Exception:
        log.warning("schedule %s not updated after its run", fired.schedule_id, exc_info=True)


def _after(fired: Fired, *, ran: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """What a fired run's end changes on its schedule: the metadata kept (a condition's state
    that changed, the failures a success clears) and its other fields (a schedule whose run
    was told to ask to be kept, and asked nothing, back on: never paused unsaid)."""
    kept: dict[str, Any] = {}
    if fired.condition is not None and fired.met not in (None, fired.reported):
        kept[REPORTED] = fired.met
    if fired.failures and ran:
        kept[FAILURES] = 0
    if fired.waits is not None:  # the run before that waited ended, and was counted
        kept[WAITS] = None
    change: dict[str, Any] = {}
    if fired.expiring and not (ran and fired.met):
        change = {"enabled": True}
        kept[PAUSED_BY] = None
    return kept, change


async def _told(agent: Agent, runtime: Runtime, fired: Fired, result: Result) -> None:
    """What a fired run comes to, on the user's channels (``notifications``), once each: its
    result when delivered, its failure (the pause it caused, when it did), its wait for an
    approval. A channel failing never fails the run."""
    if result.status is RunStatus.PAUSED and result.interrupt is not None:
        kind, text = Kind.APPROVAL, f"{fired.task}: {result.interrupt.question}"
    elif result.status in (RunStatus.ERROR, RunStatus.TIMEOUT):
        error = result.error.message if result.error is not None else result.status.value
        kind = Kind.PAUSED if fired.paused else Kind.FAILURE
        text = error
    elif fired.delivered and isinstance(result.answer, str):
        kind, text = Kind.RESULT, result.answer
    else:
        return
    notification = Notification(
        kind=kind,
        tenant=runtime.tenant,
        user=runtime.user,
        text=text,
        key=stable_id(runtime.run_id, kind.value),
        thread=runtime.identity.thread,
        run_id=runtime.run_id,
        schedule_id=fired.schedule_id,
    )
    try:
        await agent.harness.notify(notification)
    except Exception:
        log.warning("schedule %s: its run not told", fired.schedule_id, exc_info=True)


async def failed(agent: Agent, runtime: Runtime, error: AgentError) -> AgentError:
    """A fired run's failure, counted on its schedule before the run ends (:func:`_counted`):
    the one that pauses the schedule says so in the run's error (the Kubernetes CronJob and
    Temporal leave a failing schedule running; an unattended one that fails every time only
    costs)."""
    fired = runtime.scheduled
    if fired is None or (why := await _counted(runtime, fired)) is None:
        return error
    return error.model_copy(update={"message": f"{error.message} {_paused_said(why, fired)}"})


async def _counted(runtime: Runtime, fired: Fired) -> str | None:
    """One more failed run in a row, kept on the schedule: the :data:`MAX_FAILURES`-th (a
    one-time schedule's first) pauses it, and why is returned; ``None`` when it does not, or
    when the count is not kept (the schedule gone meanwhile, or the store failing: logged)."""
    fired.failures += 1
    stop = fired.once or fired.failures >= int((await limits(runtime)).pause_after_failures or 1)
    counted: dict[str, Any] = {FAILURES: fired.failures}
    why = "its one run failed" if fired.once else f"its runs failed {fired.failures} times in a row"
    if stop:
        counted[PAUSED_BY] = why
        fired.paused = why
    change = (
        ScheduleUpdate(metadata=counted, enabled=False)
        if stop
        else ScheduleUpdate(metadata=counted)
    )
    try:
        await runtime.agent.harness.runs.schedules.update(
            fired.schedule_id, change, tenant=runtime.tenant
        )
    except NotFoundError:
        return None
    except Exception:
        log.warning("schedule %s: failure not counted", fired.schedule_id, exc_info=True)
        return None
    return why if stop else None


def _paused_said(why: str, fired: Fired) -> str:
    """What the run that paused its schedule says of it."""
    return (
        f"This schedule is paused because {why}: {fired.task}. Ask me to resume it once the "
        "cause is fixed."
    )


async def _unanswered(runtime: Runtime, fired: Fired) -> str | None:
    """Its run before this one waited on a person and the store ended it (nobody answered by
    its deadline, or its holder stopped): a failed run in a row the harness never saw end,
    counted now (:func:`_counted`). The one that pauses the schedule is said as this run's
    answer, and the run does nothing else."""
    if fired.waits is None:
        return None
    before = await runtime.agent.harness.runs.get(fired.waits, tenant=runtime.tenant)
    if (
        before is None
        or before.status not in (RunStatus.ERROR, RunStatus.TIMEOUT)
        or before.error is None
        or before.error.source != "agent-runs"
    ):
        return None
    why = await _counted(runtime, fired)
    return None if why is None else f"{before.error.message}. {_paused_said(why, fired)}"


async def stale(runtime: Runtime) -> str | None:
    """Before a fired run works: a schedule the user has not changed for its days
    (``TRELLIS_SCHEDULE_EXPIRE_DAYS``) is paused until they keep it, and this run asks them, in
    its model's words and their language (handoff 2.6), after doing its task
    (:func:`expiring`): the user keeps it in chat (``update_schedule``, which renews it) or
    cancels it. What a run that does nothing else answers (:func:`_unanswered`), else
    ``None``."""
    fired = runtime.scheduled
    if fired is None:
        return None
    if (said := await _unanswered(runtime, fired)) is not None or fired.touched is None:
        return said
    days = int((await limits(runtime)).expire_days or 0)
    if datetime.now(UTC) - fired.touched < timedelta(days=days):
        return None
    waits = ScheduleUpdate(enabled=False, metadata={PAUSED_BY: KEEP_IT})
    await runtime.agent.harness.runs.schedules.update(
        fired.schedule_id, waits, tenant=runtime.tenant
    )
    fired.expiring = True
    return None


def expiring(runtime: Runtime, input: Any) -> Any:
    """A fired run's input, with the question :func:`stale` leaves to its model: after the
    task, ask whether the user still needs the schedule, in their language."""
    fired = runtime.scheduled
    if fired is None or not fired.expiring or not isinstance(input, str):
        return input
    return f"{input}\n\n{EXPIRING}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- the tools

_TASK: Final = {"type": "string", "description": "What to do, as the user said it."}
_WHEN: Final = {
    "type": "string",
    "description": "5-field cron in the user's zone: '0 9 * * *' daily 9:00, '0 9,17 * * *' "
    "9:00 and 17:00, '0 9 * * 1-5' weekdays, '0 8 * * 1' Mondays, '0 9 1 * *' the 1st, "
    "'0 18 L * *' the last day of the month; one day ('tomorrow at 9', 'next Monday at 10') "
    "counted from today: '0 9 4 11 *' with once.",
}
_CONDITION: Final = {"type": "string", "description": "Report only when this holds."}
_ID: Final = {"type": "string", "description": "An id from list_schedules."}
#: When a schedule ends: the user's last day, or an exact time, in their zone.
_END_AT: Final = {
    "type": "string",
    "description": "When it ends, in the user's zone: the last day it runs ('2026-12-31' for "
    "'until 31 December'; count 'for 3 weeks' from today), or 'YYYY-MM-DDTHH:MM' for an exact "
    "time. In update_schedule, '' removes the end.",
}
#: How a condition reports: the model picks it from the user's words.
_REPORT: Final = {
    "type": "string",
    "enum": list(REPORTS),
    "description": "For a condition: until_clears (default) reports when it holds, then not "
    "again until it stopped holding; every_time each run it holds; once the first time it "
    "holds, then the schedule ends.",
}
#: The schedules the user's words fit, when several do: the user picks one.
_IDS: Final = {
    "type": "array",
    "items": {"type": "string"},
    "description": "When the user's words fit several: all their ids; the user picks one.",
}
#: ``update_schedule``'s and a choice's ``when``: as ``schedule_task``'s, said once.
_NEW_WHEN: Final = {"type": "string", "description": "A new 5-field cron."}
#: ``update_schedule``'s pause and resume.
_ENABLED: Final = {"type": "boolean", "description": "false pauses it, true resumes it."}


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
        "tools can run, the best first; the user picks one. With no time said, use "
        "{default_time} and list time in inferred. One schedule can run at several times or days "
        "('0 9,17 * * *', '0 9 * * 1,4'). Runs at most every {every}. A condition reports as "
        "report says. When the user may already have one like it, read list_schedules first; "
        "one doing the same thing, perhaps at another time or value: pass its id as alike, "
        "and the user is asked to update it or keep both. The very same again is not made "
        "twice. Tell the user when it runs as days and times in their zone, never the cron.",
        {
            "task": _TASK,
            "when": _WHEN,
            "condition": _CONDITION,
            "once": {"type": "boolean", "description": "Run once, then end."},
            "report": _REPORT,
            "end_at": _END_AT,
            "inferred": {
                "type": "array",
                "items": {"type": "string", "enum": list(INFERRED)},
                "description": "What you assumed rather than the user said: the user confirms "
                "it in one question, with everything else still open.",
            },
            "alike": {
                "type": "string",
                "description": "The id (list_schedules) of the user's schedule that does the "
                "same thing as this one: the user picks, update it or keep both.",
            },
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
        "or condition, or pause or resume it.",
        {
            "id": _ID,
            "ids": _IDS,
            "when": _NEW_WHEN,
            "task": _TASK,
            "condition": _CONDITION,
            "report": _REPORT,
            "end_at": _END_AT,
            "enabled": _ENABLED,
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


def schedule_tools(
    runtime: Runtime | None = None, *, settings: Settings | None = None
) -> list[Tool]:
    """The tools; ``schedule_task``'s description says the deployment's default time and
    shortest repeat (the run's harness's settings, else ``settings``, else the defaults) and,
    in a run, when it is (:func:`now_said`)."""
    given = runtime.agent.harness.settings if runtime is not None else settings or Settings()
    said = SPECS[0].description.format(
        default_time=given.schedule_default_time, every=_every(given.schedule_min_interval)
    )
    if runtime is not None:
        said = f"{said} {now_said(runtime)}"
    specs = [SPECS[0].model_copy(update={"description": said}), *SPECS[1:]]
    return [Tool(spec, _RUN[spec.name], feature="schedules", confined=True) for spec in specs]


def _every(seconds: float) -> str:
    """A shortest repeat in words: whole hours, else whole minutes, else seconds."""
    for unit, size in (("hour", 3600), ("minute", 60)):
        if seconds >= size and seconds % size == 0:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return f"{seconds:g} seconds"


async def _schedule_task(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    if choices := args.get("choices"):  # a vague ask: the user picks one of these
        return await _chosen(runtime, args, list(choices)[:CHOICES])
    task, when = str(args["task"]).strip(), str(args.get("when") or "").strip()
    condition = (args.get("condition") or "").strip() or None
    zone = str(args.get("timezone") or "").strip() or snapshots.zone(runtime) or "UTC"
    once, counted, place = bool(args.get("once")), "", zone
    if minutes := args.get("in_minutes"):  # once, at that minute
        timed = _in_minutes(float(minutes), zone)
        if isinstance(timed, str):
            return timed
        (when, counted, place), once = timed, True
    held = await limits(runtime)
    min_interval, max_per_user = float(held.min_interval or 0), int(held.max_per_user or 0)
    ends = _ending(args.get("end_at"), zone)
    mine = await _mine(runtime)
    about = {"task": task, "condition": condition, "once": once}
    if said := (
        _refused(when, min_interval)
        or (ends if isinstance(ends, str) else _unending(ends, when, place))
        or _already(mine, about, when, place)
        or _full(mine, max_per_user)
    ):
        return said
    metadata = _made_metadata(runtime, held, about, zone, args.get("report"))
    when, answer = await _settled(runtime, args, mine, about, (when, place), metadata=metadata)
    if answer is not None:
        return answer
    agent = runtime.agent
    try:
        spec = ScheduleSpec(
            tenant_id=runtime.tenant,
            agent_id=agent.id,
            name=task[:200],
            cadence=when,
            timezone=place,
            on_behalf_of=runtime.user,
            input=_input(task, condition),
            timeout_seconds=min(t for t in (agent.timeout, held.run_timeout) if t is not None),
            agent_version=agent.version,
            metadata=metadata,
            end_at=ends if isinstance(ends, datetime) else None,
        )
        made = await agent.harness.runs.schedules.create(spec)
    except (ValueError, ValidationError, ConflictError) as exc:
        return _refusal(exc, "That schedule was refused")
    return f"Scheduled {made.schedule_id}: {_described(made)}{counted}"


async def _settled(
    runtime: Runtime,
    args: dict[str, Any],
    mine: list[Schedule],
    about: dict[str, Any],
    at: tuple[str, str],
    *,
    metadata: dict[str, Any],
) -> tuple[str, Any]:
    """The schedule's ``when`` once every open detail is settled, in its one question when
    anything is (:func:`_confirmed`): what the model assumed (``inferred``), the schedule it
    judged alike, the writes its runs make. ``metadata`` takes what the user allowed. With an
    answer beside it, that is the tool's: nothing made, or the alike schedule updated."""
    when, zone = at
    open_ = {
        "time": _assumed_time(when, args.get("inferred")),
        "twin": next((m for m in mine if m.schedule_id == str(args.get("alike") or "")), None),
        "writes": await _asking(runtime, args.get("tools") or []),
    }
    if not any(open_.values()) and not set(args.get("inferred") or ()) & set(INFERRED):
        return when, None  # said in full: made with no question
    confirmed = await _confirmed(runtime, _said(about, when, zone), open_)
    if isinstance(confirmed, str):
        return when, confirmed
    when = _at(when, confirmed.time)
    metadata[UNAPPROVED] = confirmed.unapproved
    if confirmed.allow:
        metadata[ALLOWED] = open_["writes"]
    if open_["twin"] is None or confirmed.alike != UPDATE:
        return when, None
    update = {"cadence": when, "timezone": zone, **_about(about)}
    kept = {k: metadata[k] for k in (ALLOWED, UNAPPROVED) if k in metadata}
    update["metadata"] = {**update["metadata"], **kept}
    return when, await _changed(runtime, open_["twin"], update)


def _in_minutes(
    minutes: float, zone: str, start: datetime | None = None
) -> tuple[str, str, str] | str:
    """``in_minutes``' one run: its cron for the next whole minute that many minutes of real
    time on (never one gone), the minute it was counted from, said where the model reads the
    time it got (the tool's text says only the hour, cached: 03:53 for "in 2 hours" at 01:53
    is right), and the zone the cron is in: ``zone``, or UTC where a clock change makes that
    wall-clock minute come twice or not at all. A delay before now, or one no cron can name
    (it has no year), is said back."""
    start = (start or datetime.now(UTC)).astimezone(UTC)
    place = zone_of(zone) or "UTC"
    local = start.astimezone(ZoneInfo(place))
    if not 0 <= minutes <= YEAR:
        return (
            f"{minutes:g} minutes from now ({local:%H:%M}, {zone}) cannot be scheduled: "
            "in_minutes is a delay ahead of at most a year; for a later day, say a date with "
            "when and once."
        )
    # up to the next whole minute: a delay that ends inside one fires in the next
    at = start + timedelta(minutes=minutes, microseconds=-1)
    at = at.replace(second=0, microsecond=0) + timedelta(minutes=1)
    counted = f" That is {minutes:g} minutes from {local:%H:%M} ({zone}), as asked."
    cron = _date_cron(at.astimezone(local.tzinfo))
    if croniter(cron, local).get_next(datetime) != at:  # the user's clock says it twice or never
        return _date_cron(at), counted, "UTC"
    return cron, counted, zone


def _date_cron(at: datetime) -> str:
    """The cron of ``at``'s minute on its day, in its own zone."""
    return f"{at.minute} {at.hour} {at.day} {at.month} *"


def _made_metadata(
    runtime: Runtime, held: ScheduleLimits, about: dict[str, Any], zone: str, report: Any
) -> dict[str, Any]:
    """What a schedule made in chat keeps: its runs' budget, its task, when the user made it,
    its zone, how its condition reports (when not the default), and the chat it was set in."""
    agent = runtime.agent
    budget = _run_budget(agent.budget, int(held.run_max_tool_calls or 0))
    metadata = {
        **agent._metadata(runtime.without, None, budget=budget),
        TASK: about,
        TOUCHED: _now(),
        ZONE: zone,
        UNAPPROVED: UNATTENDED[0],  # never waiting for nobody, unless the user says so
    }
    if (report := str(report or REPORTS[0])) in REPORTS[1:]:
        metadata[REPORT] = report
    if runtime.conversation != runtime.run_id:  # the chat it was set in
        metadata[THREAD] = runtime.identity.thread
    return metadata


def _limit_said(error: Any) -> str | None:
    """agent-runs' limit (``RUNS__SCHEDULES__*``, its code and numbers) as the user is told it:
    which limit, its number, what to do. ``None`` for anything else."""
    code, details = getattr(error, "code", None), getattr(error, "details", None) or {}
    try:
        if code == "SCHEDULE_LIMIT":
            most, count = int(details["max_per_tenant"]), int(details["schedule_count"])
            return (
                f"Your organisation has {count} of {most} schedules, the most it allows: cancel "
                "one of yours, or ask your admin to raise the limit."
            )
        if code == "CADENCE_TOO_FREQUENT":
            gap, least = details["interval_seconds"], details["min_interval_seconds"]
            return (
                f"That runs every {_every(float(gap))}, more often than every "
                f"{_every(float(least))}, the most often a schedule may run here: pick a longer "
                "gap."
            )
        if code == "FIRES_PER_DAY":
            return (
                f"Your organisation's {int(details['max_fires_per_day'])} scheduled runs for the "
                "day were used up, so this one did not run; it runs again at its next time."
            )
    except (KeyError, TypeError, ValueError):  # a limit without its numbers: as it was said
        return None
    return None


def _refusal(exc: Exception, lead: str) -> str:
    """Why the store refused a schedule or a change, as the model tells the user."""
    return _limit_said(exc) or f"{lead}: {exc}"


def _ending(given: Any, zone: str) -> datetime | str | None:
    """``end_at`` as the instant the schedule ends: a day is its last day, so the instant that
    day ends in ``zone``; a date and time, that time there; ``''`` none (``update_schedule``
    removes the end). What cannot be one, or is already past, is said back."""
    text = str(given or "").strip()
    if not text:
        return None
    place = ZoneInfo(zone_of(zone) or "UTC")
    try:
        if len(text) == len("2026-12-31"):
            last = date.fromisoformat(text)
            ends = datetime.combine(last + timedelta(days=1), time(), tzinfo=place)
        else:
            ends = datetime.fromisoformat(text).replace(tzinfo=place)
    except ValueError:
        return f"{text!r} is not a day ('2026-12-31') or a time ('2026-12-31T17:00')."
    if ends <= datetime.now(UTC):
        return f"{text} is already past in {zone}: say a later end."
    return ends.astimezone(UTC)


def _unending(ends: datetime | None, when: str, zone: str) -> str | None:
    """What the user is told of an end that leaves the schedule no run, else ``None``."""
    place = ZoneInfo(zone_of(zone) or "UTC")
    if (
        ends is None
        or not croniter.is_valid(when)
        or croniter(when, datetime.now(place)).get_next(datetime) < ends
    ):
        return None
    first = runs_said(when, zone, once=True).removeprefix("next ")
    return (
        f"It would never run: its first run, {first}, is not before its end, "
        f"{_ends_said(ends, zone)}: say a later end or another time."
    )


def _ends_said(ends: datetime, zone: str) -> str:
    """When a schedule ends, in words in its zone: after its last day, or at a time."""
    local = ends.astimezone(ZoneInfo(zone_of(zone) or "UTC"))
    if local.time() == time():
        last = local - timedelta(days=1)
        return f"after {last:%a} {last.day} {last:%b} {last.year} ({zone})"
    return f"at {local:%a} {local.day} {local:%b} {local.year}, {local:%H:%M} ({zone})"


def _run_budget(own: Budget | None, max_tool_calls: int) -> Budget:
    """A fired run's budget: the agent's, its tool calls at most ``max_tool_calls`` (the
    agent's own lower limit stays)."""
    own = own or Budget()
    held = own.max_tool_calls
    calls = max_tool_calls if held is None else min(held, max_tool_calls)
    return dataclasses.replace(own, max_tool_calls=calls)


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


def _already(mine: list[Schedule], about: dict[str, Any], when: str, zone: str) -> str | None:
    """The very same schedule again (:func:`_same`): said to be set, nothing made."""
    same = next((m for m in mine if _same(m, about, when, zone)), None)
    return None if same is None else f"Already set, nothing changed: {_described(same)}"


@dataclass(frozen=True)
class _Confirmed:
    """What the user confirmed of a new schedule: its time (``HH:MM``, when it was open), the
    schedule they keep when one is alike (:data:`UPDATE` it or :data:`BOTH`), whether its runs
    may make its writes unasked, and what they do with a call nobody approved."""

    time: str | None
    alike: str
    allow: bool
    unapproved: str


async def _confirmed(
    runtime: Runtime,
    said: str,
    open_: dict[str, Any],
) -> _Confirmed | str:
    """The one question a new schedule asks, every open detail in it (``open_``: the time it
    assumed, the schedule the model judged alike, the writes its runs make): Confirm with
    each as offered (recommended), the details changed in the app's form, Cancel, or the
    user's own words — what the model then reads (:data:`OWN_ANSWER`)."""
    fields = _fields(open_)
    question = " ".join(
        [f"Set up this schedule? {said}."]
        + ([f"You already have one like it: {_described(open_['twin'])}"] if open_["twin"] else [])
        + ([f"Its runs will call {', '.join(open_['writes'])}, which change things."]
           if open_["writes"] else [])
    )  # fmt: skip
    answer = await runtime.ask(
        question,
        options=[
            Option(value=CONFIRM, label="Confirm", recommended=True),
            Option(value=NEITHER, label="Cancel"),
        ],
        expects={"anyOf": [{"type": "string"}, {"type": "object", "properties": fields}]},
    )
    if answer == NEITHER:
        return "Nothing scheduled: the user did not want it."
    if answer != CONFIRM and not isinstance(answer, dict):
        return OWN_ANSWER.format(answer=json.dumps(answer))
    picked = {name: field.get("default") for name, field in fields.items()}
    picked |= answer if isinstance(answer, dict) else {}
    unapproved = picked["unapproved"]
    return _Confirmed(
        time=picked.get("time"),
        alike=str(picked.get("alike") or BOTH),
        allow=picked.get("allow") is True,
        unapproved=unapproved if unapproved in UNATTENDED else UNATTENDED[0],
    )


def _fields(open_: dict[str, Any]) -> dict[str, Any]:
    """The confirmation's form: a field for each open detail, its recommended answer its
    default (the app renders a time picker, a choice, a box to tick)."""
    fields: dict[str, Any] = {}
    if open_["time"] is not None:
        fields["time"] = {
            "type": "string",
            "format": "time",
            "title": "Time",
            "default": open_["time"],
        }
    if open_["twin"] is not None:
        fields["alike"] = {
            "type": "string",
            "title": "The one you have",
            "enum": [UPDATE, BOTH],
            "default": UPDATE,
        }
    if open_["writes"]:
        fields["allow"] = {
            "type": "boolean",
            "default": True,
            "title": f"Allow {', '.join(open_['writes'])} in its runs unasked",
        }
    fields["unapproved"] = {
        "type": "string",
        "title": "A step nobody approved",
        "enum": list(UNATTENDED),
        "default": UNATTENDED[0],
    }
    return fields  # fmt: skip


def _at(when: str, time_: str | None) -> str:
    """``when`` at ``time_`` (``HH:MM``) where the user changed the time it assumed."""
    if not time_ or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time_):
        return when
    hour, minute = time_.split(":")
    return " ".join([str(int(minute)), str(int(hour)), *when.split()[2:]])


def _assumed_time(when: str, inferred: Any) -> str | None:
    """The time a schedule runs at, ``HH:MM``, when the model assumed it (``inferred``)."""
    minute, hour, *_ = [*when.split(), "", ""]
    if "time" not in (inferred or ()) or not (minute.isdigit() and hour.isdigit()):
        return None
    return f"{int(hour):02d}:{int(minute):02d}"


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
        "Which should I set up?",
        options=[*options, Option(value=NEITHER, label="None of these")],
        # a suggestion's value, the user's own words, or the app's fields for any time
        expects={"anyOf": [{"type": "string"}, WHEN_ANSWER]},
    )
    if picked == NEITHER:
        return "Nothing scheduled: the user wants none of these."
    if not isinstance(picked, str) or picked not in {o.value for o in options}:
        return OWN_ANSWER.format(answer=json.dumps(picked, ensure_ascii=False))
    return await _schedule_task({**args, "choices": None, **choices[int(picked)]})


async def _list_schedules(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    mine = await _mine(runtime)
    if not mine:
        return "You have no schedules."
    shown = []
    for schedule in mine:
        shown.append({**_shown(schedule), **await _last_run(runtime, schedule)})
    return shown


async def _last_run(runtime: Runtime, schedule: Schedule) -> dict[str, str]:
    """What the schedule's last run's own record says: what it waits on while it waits on a
    person (point 29: its next tick starts no other run meanwhile), or why it failed (its
    error, as the run store keeps it: "why did my report fail?" answered with memory off)."""
    if schedule.last_run_id is None:
        return {}
    record = await runtime.agent.harness.runs.get(schedule.last_run_id, tenant=runtime.tenant)
    if record is None:
        return {}
    if record.status is RunStatus.PAUSED and record.awaiting is not None:
        return {"waiting": f"for your answer: {record.awaiting.question}"}
    if record.status in (RunStatus.ERROR, RunStatus.TIMEOUT) and record.error is not None:
        return {"last_run_error": record.error.message[:ERROR_SAID]}
    return {}


async def _update_schedule(args: dict[str, Any]) -> Any:
    runtime = _runtime()
    schedule = await _which(runtime, args, "change")
    if isinstance(schedule, str):
        return schedule
    changes: dict[str, Any] = {}
    if args.get("when"):
        min_interval = float((await limits(runtime)).min_interval or 0)
        if (refused := _refused(str(args["when"]).strip(), min_interval)) is not None:
            return refused
        changes["cadence"] = str(args["when"]).strip()
    about = dict(schedule.metadata[TASK])
    if args.get("task") or "condition" in args:
        about["task"] = str(args.get("task") or about["task"]).strip()
        if "condition" in args:
            about["condition"] = (args["condition"] or "").strip() or None
        changes.update(_about(about))
    more = _more_changes(args, schedule)
    if isinstance(more, str):
        return more
    ends = more.get("end_at", getattr(schedule, "end_at", None))
    if refused := _unending(ends, changes.get("cadence", schedule.cadence), schedule.timezone):
        return refused
    metadata = {**changes.get("metadata", {}), **more.pop("metadata", {})}
    return await _changed(runtime, schedule, {**changes, **more, "metadata": metadata})


def _more_changes(args: dict[str, Any], schedule: Schedule) -> dict[str, Any] | str:
    """``update_schedule``'s pause or resume (and why, the user's own), end and report, as a
    schedule's changes; what cannot be one, said back."""
    changes: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    if args.get("enabled") is not None:
        changes["enabled"] = bool(args["enabled"])
        metadata[PAUSED_BY] = None if changes["enabled"] else "the user paused it"
    if args.get("end_at") is not None:
        ends = _ending(args["end_at"], schedule.timezone)
        if isinstance(ends, str):
            return ends
        changes["end_at"] = ends
    if (report := args.get("report")) is not None:
        if report not in REPORTS:
            return f"report is one of {', '.join(REPORTS)}, not {report!r}."
        metadata[REPORT] = report
    return {**changes, "metadata": metadata}


async def _changed(runtime: Runtime, schedule: Schedule, changes: dict[str, Any]) -> Any:
    store = runtime.agent.harness.runs.schedules
    # the user's own change: touched now, failures cleared
    kept = {**changes.get("metadata", {}), TOUCHED: _now(), FAILURES: 0}
    if changes.get("timezone"):
        kept[ZONE] = changes["timezone"]
    changes = {**changes, "metadata": kept}
    try:
        updated = await store.update(
            schedule.schedule_id, ScheduleUpdate(**changes), tenant=runtime.tenant
        )
    except (ValidationError, ConflictError) as exc:
        return _refusal(exc, "That change was refused")
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


async def limits(runtime: Runtime) -> ScheduleLimits:
    """The limits in force for the run's user: each field the hook's
    (``Harness(schedule_limits=)``), else the settings' ``schedule_<field>``: none left
    ``None``."""
    settings, hook = runtime.agent.harness.settings, runtime.agent.harness.schedule_limits
    own = (
        await hook(runtime.user, runtime.tenant) if hook is not None else None
    ) or ScheduleLimits()
    held = ScheduleLimits(
        **{
            f.name: getattr(settings, f"schedule_{f.name}")
            if getattr(own, f.name) is None
            else getattr(own, f.name)
            for f in dataclasses.fields(ScheduleLimits)
        }
    )
    least = max(float(held.min_interval or 0), await _service_floor(runtime))
    return dataclasses.replace(held, min_interval=least)


async def _service_floor(runtime: Runtime) -> float:
    """agent-runs' own least time between a schedule's runs (``schedules.policy``), read at
    most once a :data:`FLOOR_KEPT` per tenant: 0 where the store says none (``LocalRuns``) or
    it cannot be read now (its own refusal, said in words, still applies)."""
    harness = runtime.agent.harness
    kept = harness.schedule_floors.get(runtime.tenant)
    now = time_module.monotonic()
    if kept is not None and now - kept[1] < FLOOR_KEPT:
        return kept[0]
    policy = getattr(harness.runs.schedules, "policy", None)
    floor = 0.0
    if policy is not None:
        try:
            floor = float((await policy(tenant=runtime.tenant)).min_interval_seconds)
        except Exception:
            log.warning("agent-runs' schedule policy not read", exc_info=True)
            return kept[0] if kept is not None else 0.0
    harness.schedule_floors[runtime.tenant] = (floor, now)
    return floor


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
        return (
            f"That runs every {_every(shortest)}, more often than every {_every(min_interval)}, "
            "the most often a schedule may run here: pick a longer gap."
        )
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
    the one the user picks; a sentence when there is none. A run resuming with a pick whose
    schedule went meanwhile never takes another for it: with several left it asks again (the
    question changed), with one it says the pick is gone."""
    ids = list(dict.fromkeys(str(i) for i in args.get("ids") or [] if i))
    ids = ids or [str(args.get("id") or "")]
    found = [s for i in ids if (s := await _own(runtime, i)) is not None]
    if not found:
        return f"No schedule {', '.join(ids)} of yours."
    if len(found) == 1:
        gone = _picked(runtime, ids) or found[0].schedule_id
        return found[0] if gone == found[0].schedule_id else f"No schedule {gone} of yours now."
    picked = await runtime.ask(
        f"Which one should I {verb}?",
        options=[Option(value=s.schedule_id, label=_label(s)) for s in found],
    )
    return next(s for s in found if s.schedule_id == picked)  # the answer is one of them


def _picked(runtime: Runtime, ids: list[str]) -> str | None:
    """The one of ``ids`` this run's journal holds as a person's answer: the pick a resumed
    run's question was answered with (a first call has none)."""
    answers = runtime.replay.journal.answers.values()
    return next((a["answer"] for given in answers for a in given if a.get("answer") in ids), None)


def _same(schedule: Schedule, about: dict[str, Any], when: str, zone: str) -> bool:
    """Whether ``about`` at ``when`` is ``schedule`` again: its time, zone and the figures and
    presence of its condition."""
    theirs = schedule.metadata[TASK]

    def figures(condition: Any) -> tuple[bool, list[str]]:
        found = re.findall(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])", str(condition or ""))
        return condition is not None, found

    return (
        " ".join(str(theirs.get("task", "")).lower().split())
        == " ".join(str(about["task"]).lower().split())
        and schedule.cadence == when
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


def runs_said(when: str, zone: str, *, once: bool = False, after: datetime | None = None) -> str:
    """When a schedule runs, as the model and the user read it: its next run (and the one after,
    for a repeating one) as a day, a date and an hour in its zone — never its cron, which stays
    inside (handoff 2.4). A cron that does not parse is said as it is."""
    try:
        start = (after or datetime.now(UTC)).astimezone(ZoneInfo(zone))
        cursor = croniter(when, start)
        first = cursor.get_next(datetime)
    except (ValueError, KeyError, ZoneInfoNotFoundError):
        return f"'{when}' ({zone})"
    year = start.year  # a date in another year says its year

    def day(at: datetime) -> str:
        dated = f"{at:%a} {at.day} {at:%b}" + (f" {at.year}" if at.year != year else "")
        return f"{dated}, {at:%H:%M}"

    if once:
        return f"next {day(first)} ({zone})"
    return f"next {day(first)}, then {day(cursor.get_next(datetime))} ({zone})"


def _said(about: dict[str, Any], when: str, zone: str) -> str:
    said = f"{about.get('task')}, {runs_said(when, zone, once=bool(about.get('once')))}"
    return f"{said}, only when {about['condition']}" if about.get("condition") else said


def _label(schedule: Schedule) -> str:
    return _said(schedule.metadata[TASK], schedule.cadence, schedule.timezone)


def _shown(schedule: Schedule) -> dict[str, Any]:
    about = schedule.metadata[TASK]
    due = schedule.next_fire_at if schedule.enabled else None
    once = bool(about.get("once", False))
    return {
        "id": schedule.schedule_id,
        "task": about["task"],
        "runs": runs_said(schedule.cadence, schedule.timezone, once=once) if due else None,
        "timezone": schedule.timezone,
        "condition": about.get("condition"),
        "once": once,
        "status": "active" if schedule.enabled else "paused",
        **_history(schedule),
    }


def _history(schedule: Schedule) -> dict[str, Any]:
    """What the schedule keeps of its runs, said where there is something: the last run (in its
    zone), its failed runs in a row, why its last fire failed, and for a condition how it
    reports and whether it was reported (why a watch went quiet)."""
    said: dict[str, Any] = {}
    if (ends := getattr(schedule, "end_at", None)) is not None:
        said["ends"] = _ends_said(ends, schedule.timezone)
    if (at := schedule.last_fired_at) is not None:
        local = at.astimezone(ZoneInfo(zone_of(schedule.timezone) or "UTC"))
        said["last_run"] = f"{local:%a} {local.day} {local:%b}, {local:%H:%M} ({schedule.timezone})"
    if failures := int(schedule.metadata.get(FAILURES) or 0):
        said["failed_runs_in_a_row"] = failures
    if schedule.consecutive_failures and schedule.last_error is not None:
        said["fire_error"] = _limit_said(schedule.last_error) or schedule.last_error.message
    if not schedule.enabled and (why := schedule.metadata.get(PAUSED_BY)):
        said["paused_because"] = str(why)
    if schedule.metadata[TASK].get("condition"):
        said["reports"] = str(schedule.metadata.get(REPORT) or REPORTS[0])
        said["reported"] = bool(schedule.metadata.get(REPORTED))
    return said


def _described(schedule: Schedule) -> str:
    shown = _shown(schedule)
    said = f"{shown['task']}, {shown['runs'] or 'paused'}"
    if shown["condition"]:
        said += f", reported only when {shown['condition']}"
    return f"{said}."
