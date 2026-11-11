"""Tool results as snapshots: a value a tool returned is what it was when it was fetched, and a
conversation that replays it in a later turn says so.

* **fetched** — when a call returned (``Runtime.fetched``) is said in one place: the run's
  record of the call (``Said.facts``, :data:`FETCHED`), which the run writes with its turn
  (``pipeline._with_facts``) and the memory service's steps list shows on the call's line. A
  result's own text is never stamped with it.
* **out of date** — a tool's data is live unless it declares otherwise (``tools.base.max_age``:
  0 by default, ``max_age`` seconds, or none for ``freshness="static"`` reference data), so an
  earlier result replayed past that age says it is out of date (:data:`STALE`), the run's
  context says the same in one line naming the tool (:func:`note`), and the tool stays offered
  — the model fetches the value again instead of answering from the old one. Memory is never
  the source of a tool's data; a repeated call is served from an earlier one only within its
  own model step (``tools.bridge.same``) and run (the journal), never across turns.
* **today** — the dates all of these carry mean something only against the day it is, which
  the run's model is told once, in its instructions (:func:`today`: the run's start, to the
  day), where its framework does not say it itself; ``schedule_task`` says the hour too
  (:func:`said`, the one wording of when it is).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Final
from zoneinfo import ZoneInfo

from trellis.harness.history import Said, Speaker

#: The run-record fact of when a call returned (UTC, :data:`TIME`), and of how long its value
#: stays current (seconds; absent for a tool whose data is static).
FETCHED: Final = "fetched"
MAX_AGE: Final = "max_age"
#: The line a replayed result past its tool's max age starts with.
STALE: Final = "[out of date: call {tool} again for the current value]"
#: The mark of a live tool's result from an earlier turn of a thread its framework keeps (a
#: graph's checkpointer: no fetch time is kept with it, and it is from before this turn).
EARLIER: Final = (
    "[{tool}'s result from an earlier turn: out of date, {tool}'s values change — call {tool} "
    "again for the current value]"
)
#: What the context says of the tools whose replayed results are out of date.
NOTE: Final = (
    "The values {tools} returned in earlier turns are out of date (older than they stay "
    "current): for a value the question needs, call its tool again; never answer with an "
    "earlier one."
)
#: The time format of a fetch time: to the second, UTC.
TIME: Final = "%Y-%m-%dT%H:%M:%SZ"


def declared_max_age(runtime: Any, name: str) -> float | None:
    """How long a value of ``runtime``'s tool ``name`` stays current, as it declared it
    (``tools.base.max_age``; ``None``: it did not say, or it is not the run's tool)."""
    from trellis.harness.tools.base import max_age  # noqa: PLC0415 - the tools import this

    found = runtime.toolbox.get(name)
    spec = found.spec if found is not None else runtime.native.get(name)
    return None if spec is None else max_age(spec)


def now() -> datetime:
    """The current time, in UTC (one place, so a test sets it)."""
    return datetime.now(UTC)


def zone(runtime: Any) -> str | None:
    """The user's zone, as their app said it (``Frontend.timezone``: a valid one, or none)."""
    return runtime.frontend.timezone if runtime.frontend is not None else None


def said(when: datetime, zone: str | None, *, hour: bool = False) -> str:
    """When it is, as a model reads it: the day — ``hour``: and the hour, as a range — in the
    user's ``zone``, or in UTC, said as not theirs. Never the minute: the text stays the same
    for the day (the hour), as a prompt cache keeps it."""
    local = when.astimezone(ZoneInfo(zone or "UTC"))
    at = f", {local:%H}:00-{local:%H}:59" if hour else ""
    whose = f"{zone}, the user's zone" if zone else "UTC; the user's zone is not known"
    return f"It is now {local:%a %Y-%m-%d}{at} ({whose})."


def today(runtime: Any) -> str:
    """The day it is, as ``runtime``'s model is told in its instructions: the day its run
    started (``Runtime.started_at``), so every model call of the run reads the same line."""
    return said(runtime.started_at or now(), zone(runtime))


def out_of_date(
    history: Sequence[Said], max_age: Callable[[str], float | None]
) -> tuple[list[Said], list[str]]:
    """``history`` with each tool result whose recorded fetch time (``Said.facts``,
    :data:`FETCHED`) is past its tool's ``max_age`` (seconds; ``None``: the tool did not say)
    starting with :data:`STALE`, and those tools, in the order met."""
    at = now()
    tools = {c.id: c.name for m in history for c in m.tool_calls}
    stale: dict[str, None] = {}

    def one(m: Said) -> Said:
        tool = tools.get(m.tool_call_id or "") if m.role is Speaker.TOOL else None
        when = _parsed((m.facts or {}).get(FETCHED))
        limit = None if tool is None else max_age(tool)
        if tool is None or when is None or limit is None or (at - when).total_seconds() <= limit:
            return m
        stale[tool] = None
        return dataclasses.replace(m, content=f"{STALE.format(tool=tool)}\n{m.content}")

    marked = [one(m) for m in history]
    return marked, list(stale)


def note(tools: Iterable[str]) -> str | None:
    """The context's line for the tools whose replayed results are out of date (``None``:
    none)."""
    named = ", ".join(tools)
    return NOTE.format(tools=named) if named else None


def _parsed(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, TIME).replace(tzinfo=UTC)
    except ValueError:
        return None


__all__ = [
    "EARLIER",
    "FETCHED",
    "MAX_AGE",
    "NOTE",
    "STALE",
    "TIME",
    "declared_max_age",
    "note",
    "now",
    "out_of_date",
    "said",
    "today",
    "zone",
]
