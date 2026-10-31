"""``trellis.harness.snapshots``: a tool result written to the conversation says which tool
returned it and when; replayed past its tool's declared age, it says it is out of date."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trellis.harness import snapshots
from trellis.harness.history import Called, Said, Speaker

T0 = datetime(2026, 10, 30, 9, 0, 0, tzinfo=UTC)
REAL_NOW = snapshots.now


def turn(result: str = "40", call: str = "c1") -> list[Said]:
    return [
        Said(Speaker.USER, "stock?"),
        Said(Speaker.ASSISTANT, "", (Called(call, "get_stock", {"sku": "a"}),)),
        Said(Speaker.TOOL, result, tool_call_id=call),
        Said(Speaker.ASSISTANT, "40 units."),
    ]


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    now = [T0]
    monkeypatch.setattr(snapshots, "now", lambda: now[0])
    return now


def test_a_tool_result_is_stamped_once_with_its_tool_and_when_it_was_fetched() -> None:
    fetched = {"c1": T0 - timedelta(seconds=3)}
    stamped = snapshots.stamped(turn(), fetched)
    assert stamped[2].content == "[snapshot of get_stock from 2026-10-30T08:59:57Z]\n40"
    assert [m.content for m in stamped[:2] + stamped[3:]] == ["stock?", "", "40 units."]
    assert snapshots.stamped(stamped, {}) == stamped  # never twice
    # a call the attempt did not time (another worker's, a framework's own): when written;
    # a result whose call is not in the messages: "a tool"
    orphan = [Said(Speaker.TOOL, "7", tool_call_id="zz")]
    assert snapshots.stamped(orphan, {})[0].content.startswith("[snapshot of a tool from 2026-")


def test_a_replayed_result_past_its_tools_max_age_is_out_of_date(clock: list[datetime]) -> None:
    history = snapshots.stamped(turn(), {})
    ages = {"get_stock": 60.0}
    clock[0] = T0 + timedelta(seconds=30)
    assert snapshots.out_of_date(history, ages.get) == (history, [])  # within its age
    clock[0] = T0 + timedelta(hours=1)
    marked, stale = snapshots.out_of_date(history, ages.get)
    assert stale == ["get_stock"]
    assert marked[2].content == (
        snapshots.STALE.format(tool="get_stock", at="2026-10-30T09:00:00Z") + "\n40"
    )
    assert snapshots.out_of_date(history, {}.get) == (history, [])  # no age declared
    assert snapshots.note(stale) is not None and "get_stock" in str(snapshots.note(stale))
    assert snapshots.note([]) is None


def test_a_result_with_no_stamp_or_an_unreadable_time_is_left_as_it_is() -> None:
    plain = turn()
    assert snapshots.out_of_date(plain, {"get_stock": 0.0}.get) == (plain, [])
    odd = turn("[snapshot of get_stock from 2026-13-45T09:00:00Z]\n40")
    assert snapshots.out_of_date(odd, {"get_stock": 0.0}.get) == (odd, [])
    twice = turn() + turn(call="c2")
    _, stale = snapshots.out_of_date(snapshots.stamped(twice, {}), {"get_stock": 0.0}.get)
    assert stale == []  # fetched just now: not older than 0 seconds


def test_the_time_is_utc() -> None:
    assert REAL_NOW().tzinfo is UTC
