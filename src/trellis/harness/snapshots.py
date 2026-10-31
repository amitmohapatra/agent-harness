"""Tool results as snapshots: a value a tool returned is what it was when it was fetched, and a
conversation that replays it in a later turn says so.

* **stamped** — where memory owns the conversation, each tool result the run writes to it
  (``Agent.recorded_run``) starts with :data:`STAMP`: the tool, and when the call returned
  (``Runtime.fetched``, else the time it is written). A later turn's window replays it with
  that line, so the model can say where a number came from and when, and can tell an old
  value from a current one.
* **out of date** — a tool's data is live unless it declares otherwise (``tools.base.max_age``:
  0 by default, ``max_age`` seconds, or none for ``freshness="static"`` reference data), so an
  earlier result replayed past that age says it is out of date (:data:`STALE`), the run's
  context says the same in one line naming the tool (:func:`note`), and the tool stays offered
  — the model fetches the value again instead of answering from the old one. Memory is never
  the source of a tool's data; a repeated call is served from an earlier one only within its
  own model step (``tools.bridge.same``) and run (the journal), never across turns.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from trellis.harness.history import Said, Speaker

#: The line a replayed tool result starts with: which tool returned it, and when (UTC).
STAMP: Final = "[snapshot of {tool} from {at}]"
#: The stamp of a result past its tool's max age.
STALE: Final = (
    "[snapshot of {tool} from {at}: out of date, {tool}'s values change — call {tool} again "
    "for the current value]"
)
_STAMPED: Final = re.compile(
    r"^\[snapshot of (?P<tool>[^\]\s]+) from (?P<at>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)[^\]\n]*\]\n"
)
#: What the context says of the tools whose replayed results are out of date.
NOTE: Final = (
    "The values {tools} returned in earlier turns are out of date (older than they stay "
    "current): call {tools} again for the current values; never answer from the earlier ones."
)
#: The time format of a stamp: to the second, UTC.
TIME: Final = "%Y-%m-%dT%H:%M:%SZ"


def now() -> datetime:
    """The current time, in UTC (one place, so a test sets it)."""
    return datetime.now(UTC)


def stamped(messages: Sequence[Said], fetched: Mapping[str, datetime]) -> list[Said]:
    """``messages`` with each tool result that has no stamp yet starting with :data:`STAMP`:
    its tool (named by the call it answers), and when it was fetched (``fetched``, by call id;
    else now)."""
    tools = {c.id: c.name for m in messages for c in m.tool_calls}
    written = now()

    def one(m: Said) -> Said:
        if m.role is not Speaker.TOOL or m.tool_call_id is None or _STAMPED.match(m.content):
            return m
        when = fetched.get(m.tool_call_id, written)
        stamp = STAMP.format(tool=tools.get(m.tool_call_id, "a tool"), at=when.strftime(TIME))
        return dataclasses.replace(m, content=f"{stamp}\n{m.content}")

    return [one(m) for m in messages]


def out_of_date(
    history: Sequence[Said], max_age: Callable[[str], float | None]
) -> tuple[list[Said], list[str]]:
    """``history`` with each stamped tool result past its tool's ``max_age`` (seconds; ``None``:
    the tool did not say) stamped :data:`STALE` instead, and those tools, in the order met."""
    at = now()
    stale: dict[str, None] = {}

    def one(m: Said) -> Said:
        found = _STAMPED.match(m.content) if m.role is Speaker.TOOL else None
        if found is None:
            return m
        tool, when = found.group("tool"), _parsed(found.group("at"))
        limit = max_age(tool)
        if when is None or limit is None or (at - when).total_seconds() <= limit:
            return m
        stale[tool] = None
        said = STALE.format(tool=tool, at=found.group("at"))
        return dataclasses.replace(m, content=f"{said}\n{m.content[found.end() :]}")

    marked = [one(m) for m in history]
    return marked, list(stale)


def note(tools: Iterable[str]) -> str | None:
    """The context's line for the tools whose replayed results are out of date (``None``:
    none)."""
    named = ", ".join(tools)
    return NOTE.format(tools=named) if named else None


def _parsed(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, TIME).replace(tzinfo=UTC)
    except ValueError:
        return None


__all__ = ["NOTE", "STALE", "STAMP", "note", "now", "out_of_date", "stamped"]
