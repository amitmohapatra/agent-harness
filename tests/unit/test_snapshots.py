"""``trellis.harness.snapshots``: when a tool result was fetched is said once, in the run's
record of its call; replayed past its tool's declared age, the result says it is out of date."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from trellis.contracts import ToolSpec
from trellis.harness import snapshots
from trellis.harness.history import Called, Said, Speaker

T0 = datetime(2026, 10, 30, 9, 0, 0, tzinfo=UTC)
REAL_NOW = snapshots.now


def turn(result: str = "40", call: str = "c1", fetched: str | None = "2026-10-30T09:00:00Z"):
    facts = None if fetched is None else {"status": "ok", snapshots.FETCHED: fetched}
    return [
        Said(Speaker.USER, "stock?"),
        Said(Speaker.ASSISTANT, "", (Called(call, "get_stock", {"sku": "a"}),)),
        Said(Speaker.TOOL, result, tool_call_id=call, facts=facts),
        Said(Speaker.ASSISTANT, "40 units."),
    ]


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    now = [T0]
    monkeypatch.setattr(snapshots, "now", lambda: now[0])
    return now


def test_a_replayed_result_past_its_tools_max_age_is_out_of_date(clock: list[datetime]) -> None:
    history = turn()
    ages = {"get_stock": 60.0}
    clock[0] = T0 + timedelta(seconds=30)
    assert snapshots.out_of_date(history, ages.get) == (history, [])  # within its age
    clock[0] = T0 + timedelta(hours=1)
    marked, stale = snapshots.out_of_date(history, ages.get)
    assert stale == ["get_stock"]
    assert marked[2].content == snapshots.STALE.format(tool="get_stock") + "\n40"
    assert "2026-10-30" not in marked[2].content  # the fetch time is the run record's to say
    assert snapshots.out_of_date(history, {}.get) == (history, [])  # no age declared
    assert snapshots.note(stale) is not None and "get_stock" in str(snapshots.note(stale))
    assert snapshots.note([]) is None


def test_a_result_with_no_recorded_or_an_unreadable_fetch_time_is_left_as_it_is() -> None:
    plain = turn(fetched=None)
    assert snapshots.out_of_date(plain, {"get_stock": 0.0}.get) == (plain, [])
    odd = turn(fetched="2026-13-45T09:00:00Z")
    assert snapshots.out_of_date(odd, {"get_stock": 0.0}.get) == (odd, [])
    orphan = [Said(Speaker.TOOL, "7", tool_call_id="zz", facts={"fetched": "2020-01-01T00:00:00Z"})]
    assert snapshots.out_of_date(orphan, {"get_stock": 0.0}.get) == (orphan, [])
    _, stale = snapshots.out_of_date(turn() + turn(call="c2"), {"get_stock": 0.0}.get)
    assert stale == []  # fetched just now: not older than 0 seconds


def test_a_tools_declared_age_is_read_from_the_run() -> None:
    live = ToolSpec(name="get_stock", description="d", input_schema={"type": "object"})
    runtime = SimpleNamespace(toolbox={"get_stock": SimpleNamespace(spec=live)}, native={})
    assert snapshots.declared_max_age(runtime, "get_stock") == 0.0
    assert snapshots.declared_max_age(runtime, "other") is None


def test_the_time_is_utc() -> None:
    assert REAL_NOW().tzinfo is UTC
