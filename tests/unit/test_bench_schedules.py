"""The scheduling benchmark's checks (``benchmark/cost/schedules.py``), on chats made here."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from benchmark.cost import schedules as b
from trellis.contracts import Schedule, ScheduleSpec
from trellis.harness.schedules import NO_REPORT, TASK

NOW = datetime(2026, 11, 3, 8, 30, tzinfo=UTC)


def made(cron: str, zone: str = "UTC", *, once: bool = False, condition: str | None = None,
         sid: str = "s1") -> Schedule:  # fmt: skip
    spec = ScheduleSpec(
        tenant_id="t", agent_id="a", name="x", cadence=cron, timezone=zone, on_behalf_of="u",
        metadata={TASK: {"task": "check stock", "condition": condition, "once": once}},
    )  # fmt: skip
    return Schedule.from_spec(spec).model_copy(update={"schedule_id": sid})


def chat(
    *after: Schedule, answer: str = "Done.", calls: Any = (), zone: str | None = None
) -> b.Chat:
    return b.Chat(now=NOW, zone=zone, answers=[answer], calls=[list(calls)], after=list(after))


def test_a_schedule_is_compared_by_the_instants_it_fires_at_in_its_zone() -> None:
    check = b.scheduled("0 9 * * *", "Asia/Kolkata")
    assert check(chat(made("30 3 * * *"))) == []  # 9:00 IST is 3:30 UTC
    assert check(chat(made("0 9 * * *", "Asia/Kolkata"))) == []
    assert check(chat(made("0 9 * * *"))) == [
        "fires at '0 9 * * *' UTC, not '0 9 * * *' Asia/Kolkata"
    ]
    assert (
        b.scheduled("0 9 * * *")(chat(made("0 9 * * *", "Asia/Kolkata"), zone="Asia/Kolkata")) == []
    )
    assert b.scheduled("0 9 * * *")(chat()) == ["0 schedules made, not 1"]


def test_once_a_condition_and_a_weekday() -> None:
    tomorrow = b.scheduled(once=True, at=b._tomorrow_at(9))
    assert tomorrow(chat(made("0 9 4 11 *", once=True))) == []
    today = tomorrow(chat(made("0 9 * * *")))  # at 8:30 its first fire is today's 9:00
    assert today == [
        "first fire Tue 03 09:00Z ('0 9 * * *' UTC), not 2026-11-04 09:00:00+00:00",
        "once=False",
    ]
    cond = b.scheduled("0 8 * * 1", condition=("SKU-1", "5"))
    assert cond(chat(made("0 8 * * 1", condition="stock of SKU-1 under 5"))) == []
    assert cond(chat(made("0 8 * * 1"))) == ["condition None lacks ['SKU-1', '5']"]
    assert b.scheduled("0 8 * * 1")(chat(made("0 8 * * 1", condition="x"))) == [
        "a condition nobody asked for: 'x'"
    ]
    monday = b.scheduled(weekday=0)
    assert monday(chat(made("15 7 * * 1"))) == []
    assert monday(chat(made("0 9 * * 2"))) == ["not weekly on day 0: '0 9 * * 2' UTC"]


def test_a_zone_not_known_may_be_asked_and_nothing_written() -> None:
    check = b.or_asks_zone(b.scheduled("0 9 * * *"))
    assert check(chat(answer="Which time zone are you in?")) == []
    assert check(chat(answer="Which report?")) == ["0 schedules made, not 1"]


def test_a_time_the_words_leave_open_is_settled_and_said_or_asked() -> None:
    nine = b.or_asks(b.says_any(b.scheduled("0 9 * * *"), "am", "09:00"), "am", "pm")
    assert nine(chat(made("0 9 * * *"), answer="Set for 9:00 AM daily.")) == []
    assert nine(chat(answer="9 in the morning or the evening, AM or PM?")) == []
    # a silent guess fails: scheduled, but the time not said in words the user cannot misread
    assert nine(chat(made("0 9 * * *"), answer="Done.")) == ["answer says none of ['am', '09:00']"]
    assert nine(chat(made("0 21 * * *"), answer="Set for 9 PM.")) != []
    # asked, but a schedule made anyway is not a question
    assert nine(chat(made("0 21 * * *"), answer="Set for 21:00; AM instead?")) != []


def test_a_question_is_matched_however_its_words_are_hyphenated() -> None:
    """"every-other-week" is "every other week": a question naming what the words left open
    counts whatever joins them (the answers the every-other-week case was scored down on)."""
    other = b.or_asks(b.alternate_weeks, "every week", "weekly", "every other", "two weeks")
    asked = "I can't set an every-other-week pattern. Which should I use: every Monday, or?"
    assert other(chat(answer=asked)) == []
    assert other(chat(answer="A true every-other-Monday schedule isn't possible. Which?")) == []
    assert other(chat(answer="What should the sales summary cover?")) != []  # asks other


def test_schedules_together_and_a_time_of_day_in_words() -> None:
    both = b.covered("0 9 * * 1,4")
    assert both(chat(made("0 9 * * 1,4"))) == []
    assert both(chat(made("0 9 * * 1", sid="a"), made("0 9 * * 4", sid="b"))) == []
    assert both(chat(made("0 9 * * 1"))) != []
    assert both(chat()) == ["0 schedules made"]
    morning = b.hours(6, 11, once=True)
    assert morning(chat(made("0 9 4 11 *", once=True))) == []
    assert morning(chat(made("0 14 4 11 *", once=True))) == [
        "first run at 14:00, not between 6:00 and 11:00"
    ]
    assert b.hours(17, 21)(chat(made("0 18 * * *", once=True))) == ["once=True"]
    assert b.hours(17, 21)(chat()) == ["0 schedules made, not 1"]


def test_every_other_week_is_a_weekly_schedule_that_skips_weeks_or_says_so() -> None:
    skips = made("0 9 * * 1", condition="the ISO week number is even")
    assert b.alternate_weeks(chat(skips)) == []
    assert b.alternate_weeks(chat(made("0 9 * * 1"), answer="Set weekly: every Monday.")) == []
    assert b.alternate_weeks(chat(made("0 9 * * 1"), answer="Done.")) != []  # said nothing
    assert b.alternate_weeks(chat(made("0 9 * * 2", condition="even week"))) != []
    assert b.alternate_weeks(chat()) == ["0 schedules made, not 1"]


def test_an_end_date_is_kept() -> None:
    until = made("0 9 * * *")
    until.metadata[TASK]["until"] = "2026-11-30"
    assert b.ends(b.scheduled("0 9 * * *"))(chat(until)) == []
    assert b.ends(b.scheduled("0 9 * * *"))(chat(made("0 9 * * *"))) == ["no end date kept"]


def test_no_schedule_cancel_all_and_fired_runs() -> None:
    answered = b.unscheduled("1234", tools=("stock_level",))  # said as 1,234 too
    assert answered(chat(answer="1,234 units", calls=[("stock_level", {}, "")])) == []
    wrote = chat(answer="12", calls=[("schedule_task", {}, "")])
    assert answered(wrote) == [
        "wrote schedule_task",
        "answer lacks '1234'",
        "did not call stock_level",
    ]
    seeds = [made("0 9 * * *", sid="s1"), made("0 8 * * 1", sid="s2")]
    asked = b.Chat(now=NOW, answers=["Cancel both?", "Done."], seeded=seeds,
                   calls=[[("list_schedules", {}, "")], [("cancel_schedule", {}, "")] * 2])  # fmt: skip
    assert b.cancelled(0, 1, ask_first=True)(asked) == []
    asked.answers[0] = "Cancelled both."
    assert b.cancelled(0, 1, ask_first=True)(asked) == ["did not confirm before cancelling all"]
    fired = b.Chat(now=NOW, seeded=seeds[:1], after=seeds[:1], fired=["23 left", NO_REPORT])
    assert b.reports("report", None, says=("23",))(fired) == []
    fired.fired_calls = ["schedule_task"]
    assert b.reports("report", None)(fired) == ["fired run called schedule_task"]


def test_the_set_has_every_category() -> None:
    cases = b.cases()
    assert {c.category for c in cases} == set(b.CATEGORIES)
    assert len({c.id for c in cases}) == len(cases) >= 40


def test_check_first_cannot_vague_and_a_twin() -> None:
    first = chat(made("0 9 * * *"), answer="23 now; I'll tell you.",
                 calls=[("stock_level", {}, ""), ("schedule_task", {}, "")])  # fmt: skip
    assert b.checked_first(first, ("23",)) == []
    blind = chat(made("0 9 * * *"), answer="Done.", calls=[("schedule_task", {}, "")])
    assert b.checked_first(blind, ("23",)) == ["scheduled without checking", "answer lacks '23'"]
    assert b.checked_first(chat(), ()) == ["no schedule_task call"]
    offers = "I can't watch competitors. I can watch stock levels or our sales instead."
    assert b.cannot(chat(answer=offers)) == []
    assert b.cannot(chat(answer="I can't.")) == ["suggests 0 things the tools can watch, not 2+"]
    vague = b.Chat(now=NOW, answers=["Which?", "Done."], asked=[["A", "B"], None],
                   calls=[[], [("schedule_task", {}, "")]], after=[made("0 9 * * *")])  # fmt: skip
    assert b.clarifies(vague) == []
    seed = made("0 9 * * *", condition="under 5")
    twin = b.Chat(now=NOW, answers=["?", "Done."], asked=[["Update it (Recommended)", "Keep both"], None],
                  seeded=[seed], after=[made("0 9 * * *", condition="under 3")])  # fmt: skip
    assert b.updated_twin("3")(twin) == []
    assert b.already_set(chat(answer="That is already set.")) == []
    assert (
        b.already_true("23", "50")(chat(made("0 9 * * *", condition="under 50"), answer="23 now"))
        == []
    )


def test_a_turn_answers_the_question_the_run_waits_on() -> None:
    from trellis.contracts import Interrupt, InterruptReason, Option

    waits = Interrupt(interrupt_id="i", tenant_id="t", run_id="r", reason=InterruptReason.QUESTION,
                      question="Which?", options=[Option(value="update", label="Update it"), "both"])  # fmt: skip
    assert b.picks(waits, "@0|Update it.") == "update" and b.picks(waits, "@5|x") == "both"
    assert b.picks(waits, "keep both") == "keep both" and b.said_of("@1|Yes.") == "Yes."
    assert b.labels(waits) == ["Update it", "both"]


def test_the_scheduling_and_asking_sets_measure_any_adapter_the_complex_suite_wraps() -> None:
    from benchmark.cost import asking

    for bench in (b, asking):
        assert bench.arguments(["--model", "m"]).adapter == "react"
        assert bench.arguments(["--model", "m", "--adapter", "claude"]).adapter == "claude"
