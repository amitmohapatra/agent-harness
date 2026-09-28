"""``ScheduleSpec.cadence`` (contracts) as a Temporal schedule spec.

The contract says a cadence is "a cron expression or one of the scheduler's named buckets
(``hourly``, ``daily``, ``weekly``, ``weekdays``, ``manual``) evaluated in ``timezone``; the
scheduler validates the expression and its floor". This module is that scheduler's half of the
bargain: the buckets are spelled here once, and an expression that would fire faster than the
floor is refused at ``create`` time rather than discovered as a bill.
"""

from __future__ import annotations

from enum import StrEnum

from temporalio.client import ScheduleSpec as TemporalScheduleSpec
from trellis.contracts.errors import ConfigurationError


class Cadence(StrEnum):
    """The named buckets. An enum because it is a closed vocabulary: anything else is cron."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    WEEKDAYS = "weekdays"
    #: fires only when something triggers it; created paused
    MANUAL = "manual"


#: Bucket → cron, in the schedule's own timezone. Midnight rather than an invented "9am":
#: a bucket that guessed a business hour would be wrong in half the deployments that use it.
_CRON: dict[Cadence, str] = {
    Cadence.HOURLY: "0 * * * *",
    Cadence.DAILY: "0 0 * * *",
    Cadence.WEEKLY: "0 0 * * 1",
    Cadence.WEEKDAYS: "0 0 * * 1-5",
}

#: The fastest a schedule may fire. Cron's own resolution is a minute, so the floor only ever
#: bites the six-field form (seconds first), which is the one that can ask for 1/second.
DEFAULT_FLOOR_SECONDS = 60


def parse_cadence(
    cadence: str, *, timezone: str = "UTC", floor_seconds: int = DEFAULT_FLOOR_SECONDS
) -> TemporalScheduleSpec:
    """The Temporal spec for one cadence, or ``ConfigurationError`` saying why not.

    ``manual`` yields a spec with nothing in it: a schedule that never fires on its own, which
    together with ``ScheduleState(paused=True)`` is what "standing intent, triggered by hand"
    means to Temporal.
    """
    text = cadence.strip()
    bucket = _bucket(text)
    if bucket is Cadence.MANUAL:
        return TemporalScheduleSpec(time_zone_name=timezone)
    expression = _CRON[bucket] if bucket is not None else text
    if bucket is None:
        _validate_cron(expression, floor_seconds=floor_seconds)
    return TemporalScheduleSpec(cron_expressions=[expression], time_zone_name=timezone)


def is_manual(cadence: str) -> bool:
    return _bucket(cadence.strip()) is Cadence.MANUAL


def _bucket(text: str) -> Cadence | None:
    try:
        return Cadence(text.lower())
    except ValueError:
        return None


def _validate_cron(expression: str, *, floor_seconds: int) -> None:
    fields = expression.split()
    if len(fields) not in (5, 6, 7):
        raise ConfigurationError(
            f"cadence {expression!r} is not a cron expression: expected 5 fields "
            f"(minute hour day month weekday), 6 with leading seconds, or 7 with a trailing "
            f"year, got {len(fields)}"
        )
    if len(fields) == 5:
        return
    seconds = fields[0]
    if seconds in ("0", "00"):
        return
    step = _step(seconds)
    if step is None or step < floor_seconds:
        raise ConfigurationError(
            f"cadence {expression!r} would fire more often than every {floor_seconds}s; "
            f"use a coarser expression or raise the scheduler's floor deliberately"
        )


def _step(field: str) -> int | None:
    """The interval a seconds field asks for, when it asks for a regular one."""
    if field in ("*", "?"):
        return 1
    _, _, step = field.partition("/")
    if not step.isdigit():
        return None
    return int(step)


__all__ = ["DEFAULT_FLOOR_SECONDS", "Cadence", "is_manual", "parse_cadence"]
