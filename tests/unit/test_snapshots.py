"""``trellis.harness.snapshots``: when a tool result was fetched is said once, in the run's
record of its call; replayed past its tool's declared age, the result says it is out of date."""

from __future__ import annotations

import random
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
    # the mark is the result's state alone: what to do about it is the note's, said once with
    # the question and only for a value the question needs (on ``conversation-02`` a model told
    # on the July result to "call sales_summary again" volunteered July when asked for August)
    assert marked[2].content == "[out of date]\n40"
    assert "2026-10-30" not in marked[2].content  # the fetch time is the run record's to say
    # where no note goes with the question, the mark says it itself
    alone, _ = snapshots.out_of_date(history, ages.get, mark=snapshots.STALE_CALL)
    assert alone[2].content == "[out of date: call get_stock again for the current value]\n40"
    assert snapshots.out_of_date(history, {}.get) == (history, [])  # no age declared
    assert snapshots.note(stale) is not None and "get_stock" in str(snapshots.note(stale))
    assert snapshots.note([]) is None


@pytest.mark.parametrize("seed", range(40))
def test_what_is_out_of_date_is_marked_as_a_fact_and_what_to_do_is_said_once(
    clock: list[datetime], seed: int
) -> None:
    """Any conversation — any number of turns, results of any tools at any position, of any
    age against any declared age, some of calls that failed: each out-of-date result says so
    as a fact, on itself and nothing more; what to do about it is said once, naming each such
    tool once, and limited to what the question needs. A result still current, a failed call's
    and everything else are as they were."""
    rng = random.Random(seed)
    tools = [f"tool_{n}" for n in range(rng.randrange(1, 6))]
    ages = {name: rng.choice([0.0, 60.0, 3600.0, None]) for name in tools}
    history: list[Said] = []
    expected: dict[int, str] = {}  # where an out-of-date result is, and its tool
    clock[0] = T0 + timedelta(hours=2)
    for step in range(rng.randrange(1, 12)):
        tool, call = rng.choice(tools), f"c{step}"
        old = timedelta(seconds=rng.choice([10, 600, 7000]))
        failed = rng.random() < 0.2
        facts = {
            "status": "error" if failed else "ok",
            snapshots.FETCHED: (clock[0] - old).strftime(snapshots.TIME),
        }
        history += [
            Said(Speaker.USER, f"question {step}"),
            Said(Speaker.ASSISTANT, "", (Called(call, tool, {}),)),
            Said(Speaker.TOOL, f"value {step}", tool_call_id=call, facts=facts),
            Said(Speaker.ASSISTANT, f"answer {step}"),
        ]
        limit = ages[tool]
        if not failed and limit is not None and old.total_seconds() > limit:
            expected[len(history) - 2] = tool
    marked, stale = snapshots.out_of_date(history, ages.get)
    for n, (before, after) in enumerate(zip(history, marked, strict=True)):
        if n in expected:
            assert after.content == f"[out of date]\n{before.content}"  # the state, no more
        else:
            assert after == before
    assert sorted(stale) == sorted(set(expected.values()))
    note = snapshots.note(stale)
    if not expected:
        assert note is None  # nothing out of date: nothing said
        return
    assert note is not None and note.count("call its tool again") == 1
    assert "for a value the question needs" in note
    assert all(note.count(name) == 1 for name in stale)
    # where no note goes with the question, each mark names its own tool and what to do
    alone, _ = snapshots.out_of_date(history, ages.get, mark=snapshots.STALE_CALL)
    for n, tool in expected.items():
        assert alone[n].content.startswith(f"[out of date: call {tool} again")


def test_a_result_with_no_recorded_or_an_unreadable_fetch_time_is_left_as_it_is() -> None:
    plain = turn(fetched=None)
    assert snapshots.out_of_date(plain, {"get_stock": 0.0}.get) == (plain, [])
    odd = turn(fetched="2026-13-45T09:00:00Z")
    assert snapshots.out_of_date(odd, {"get_stock": 0.0}.get) == (odd, [])
    orphan = [Said(Speaker.TOOL, "7", tool_call_id="zz", facts={"fetched": "2020-01-01T00:00:00Z"})]
    assert snapshots.out_of_date(orphan, {"get_stock": 0.0}.get) == (orphan, [])
    _, stale = snapshots.out_of_date(turn() + turn(call="c2"), {"get_stock": 0.0}.get)
    assert stale == []  # fetched just now: not older than 0 seconds


@pytest.mark.parametrize("status", ["error", "timeout"])
def test_a_failed_call_has_no_value_to_be_out_of_date(clock: list[datetime], status: str) -> None:
    """A call that failed returned no value: its replayed result ("Error: ...") was marked
    out of date all the same, and its tool counted as one whose earlier values are out of
    date. Only a call that returned a value is marked."""
    failed = turn(result="Error: the warehouse is down")
    failed[2] = Said(
        Speaker.TOOL,
        failed[2].content,
        tool_call_id="c1",
        facts={"status": status, snapshots.FETCHED: "2026-10-30T09:00:00Z"},
    )
    clock[0] = T0 + timedelta(hours=1)
    assert snapshots.out_of_date(failed, {"get_stock": 0.0}.get) == (failed, [])
    marked, stale = snapshots.out_of_date(failed + turn(call="c2"), {"get_stock": 0.0}.get)
    assert stale == ["get_stock"]  # the call that returned 40 still is
    assert marked[2].content == "Error: the warehouse is down"
    assert marked[6].content == snapshots.STALE + "\n40"


def test_a_tools_declared_age_is_read_from_the_run() -> None:
    live = ToolSpec(name="get_stock", description="d", input_schema={"type": "object"})
    runtime = SimpleNamespace(toolbox={"get_stock": SimpleNamespace(spec=live)}, native={})
    assert snapshots.declared_max_age(runtime, "get_stock") == 0.0
    assert snapshots.declared_max_age(runtime, "other") is None


def test_the_time_is_utc() -> None:
    assert REAL_NOW().tzinfo is UTC


def test_the_note_sends_the_model_back_only_for_what_the_question_needs() -> None:
    """The context's line said "call get_stock, find_sku again for the current values; never
    answer from the earlier ones". On a turn that needs none of them ("Put that in a PDF for
    finance"), ReAct re-fetched every input and answered "I can't produce a PDF, but here is
    the content", with no reason (break-and-fix bf3-mixed-01: 6 of 10 Haiku runs). It now asks
    for a call only for a value the question needs, and still never an answer from an
    earlier one."""
    said = str(snapshots.note(["get_stock", "find_sku"]))
    assert "for a value the question needs, call its tool again" in said
    assert "never answer with an earlier one" in said
    assert "again for the current values;" not in said
