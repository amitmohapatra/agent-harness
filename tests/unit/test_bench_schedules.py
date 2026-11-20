"""The scheduling benchmark's checks (``benchmark/cost/schedules.py``), on chats made here."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

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


# --------------------------------------------------------------------------- the groups added


def seeded(*schedules: Schedule, answers: Any = ("Done.",), calls: Any = None,
           after: Any = None, asked: Any = None, zone: str | None = "UTC") -> b.Chat:  # fmt: skip
    """A chat that began with ``schedules`` (kept unchanged unless ``after`` says)."""
    answers = list(answers)
    return b.Chat(now=NOW, zone=zone, answers=answers, seeded=list(schedules),
                  after=list(schedules if after is None else after),
                  calls=calls or [[] for _ in answers],
                  asked=asked or [None for _ in answers])  # fmt: skip


def test_an_explanation_says_the_facts_and_no_made_up_cause() -> None:
    why = b.explains(*b.SIGN_IN)
    assert why(seeded(answers=["It fails because the Gmail sign-in expired."])) == []
    assert why(seeded(answers=["Your Gmail login has expired; sign in again."])) == []
    assert why(seeded(answers=["It hit a rate limit."])) == [
        "answer lacks /sign.?in|log.?in|signed out/",
        "answer lacks /expir/",
        "a made-up claim: 'rate limit'",
    ]
    # the real reason, with a cause the record does not hold besides, is made up too
    assert why(seeded(answers=["The sign-in expired after a network timeout."])) != []
    assert why(seeded(answers=["Sign-in expired."], calls=[[("update_schedule", {}, "")]])) == [
        "wrote update_schedule"
    ]


def test_a_count_of_failures_is_the_one_recorded() -> None:
    two = b.counted(2)
    assert two(seeded(answers=["It has failed 2 times in a row."])) == []
    assert two(seeded(answers=["Two failed runs so far, both on sign-in."])) == []
    assert two(seeded(answers=["It failed twice."])) == []
    assert two(seeded(answers=["It failed 3 times."])) == ["counts 3, not 2"]
    assert two(seeded(answers=["It has had 2 failures, 5 times in total."])) == ["counts 5, not 2"]
    assert two(seeded(answers=["It keeps failing."])) == ["says no count"]
    # a time of day or a run a day is no count of failures
    assert two(seeded(answers=["It runs once a day at 9; it failed twice."])) == []
    # the rule it pauses by, and a failure to come, are no count of those that happened
    assert two(seeded(answers=["2 failures so far; it pauses after 3 failures in a row."])) == []
    assert two(seeded(answers=["It failed 2 times: one more failure and it pauses."])) == []


def test_the_last_run_is_said_at_its_own_time() -> None:
    ran = made("0 9 * * *", "Asia/Kolkata").model_copy(
        update={"last_fired_at": datetime(2026, 11, 3, 8, 19, tzinfo=UTC)}
    )
    last = b.last_ran(0)
    assert last(seeded(ran, answers=["It last ran today at 13:49 IST."])) == []
    assert last(seeded(ran, answers=["Last run: 1:49 pm today."])) == []
    assert last(seeded(ran, answers=["It last ran at 9:00 this morning."])) == [
        "answer lacks the last run's time (13:49)"
    ]
    assert b.last_ran(0)(seeded(made("0 9 * * *"), answers=["Never."])) == [
        "seed 0 has no last run"
    ]


def test_a_paused_schedule_is_said_paused_and_a_user_pause_is_no_failure() -> None:
    off = b.explains(b.PAUSED, wrong=b.FAILED_CLAIMS)
    assert off(seeded(answers=["No, it is paused right now."])) == []
    assert off(seeded(answers=["It's turned off; it has not failed."])) == []
    assert off(seeded(answers=["Yes, it's active."])) == [f"answer lacks /{b.PAUSED}/"]
    assert off(seeded(answers=["It is paused after 3 failures."])) != []
    assert off(seeded(answers=["Paused because the sign-in expired."])) != []
    why = b.explains(b.PAUSED, b.NOT_RECORDED, wrong=b.FAILED_CLAIMS)
    said = "It is paused by hand: no failures are recorded, so someone turned it off."
    assert why(seeded(answers=[said])) == []
    assert why(seeded(answers=["It is paused."])) == [f"answer lacks /{b.NOT_RECORDED}/"]


def test_a_quiet_condition_is_explained_from_the_figures_or_the_report_already_made() -> None:
    quiet = b.explains("1234", "300", wrong=b.FAILED_CLAIMS)
    assert quiet(seeded(answers=["Stock is 1,234, not under 300, so nothing to report."])) == []
    assert quiet(seeded(answers=["It is broken: it failed 2 times."])) != []
    told = b.explains(b.REPORTED_ONCE, wrong=b.FAILED_CLAIMS)
    assert told(seeded(answers=["It already told you; it stays quiet until it clears."])) == []
    assert told(seeded(answers=["It reports once while it holds, then again after."])) == []
    assert told(seeded(answers=["The run failed."])) != []


def test_what_a_run_found_is_read_from_its_record_not_fetched_again() -> None:
    found = b.from_record("1234", "SKU-1")
    assert found(seeded(answers=["Yesterday it found SKU-1 at 1,234 units."])) == []
    fetched = seeded(answers=["SKU-1: 1234."], calls=[[("stock_level", {}, "")]])
    assert found(fetched) == ["fetched stock_level again instead of reading the run"]
    assert found(seeded(answers=["It checked the stock."])) == [
        "answer lacks '1234'",
        "answer lacks 'SKU-1'",
    ]


def test_every_check_and_one_of_them() -> None:
    yes, no = (lambda chat: []), (lambda chat: ["no"])
    assert b.every(yes, no, no)(seeded()) == ["no", "no"]
    assert b.one_of(no, yes)(seeded()) == []
    assert b.one_of(no, no)(seeded()) == ["no"]


def test_paused_then_resumed_across_two_turns() -> None:
    s0, s1 = made("0 9 * * *", sid="s0"), made("0 8 * * 1", sid="s1")
    turns = [[("update_schedule", {"id": "s0", "enabled": False}, "")],
             [("update_schedule", {"id": "s0", "enabled": True}, "")]]  # fmt: skip
    both = seeded(s0, s1, answers=["Paused.", "Resumed."], calls=turns)
    assert b.paused_then_resumed(0, untouched=(1,))(both) == []
    never = seeded(s0, s1, answers=["Paused.", "Resumed."], calls=[[], turns[1]])
    assert b.paused_then_resumed(0)(never) == ["turn 1 did not pause it"]
    still = seeded(s0, s1, answers=["Paused.", "OK."], calls=[turns[0], []],
                   after=[s0.model_copy(update={"enabled": False}), s1])  # fmt: skip
    assert b.paused_then_resumed(0)(still) == ["turn 2 did not resume it", "enabled=False"]


def test_a_condition_set_to_report_every_time() -> None:
    s0 = made("0 9 * * *", condition="under 5", sid="s0")
    every = s0.model_copy(update={"metadata": {**s0.metadata, b.REPORT: "every_time"}})
    check = b.reports_as(0, "every_time")
    assert check(seeded(s0, after=[every])) == []
    assert check(seeded(s0)) == ["reports until_clears, not every_time"]
    assert check(seeded(s0, after=[])) == ["seed 0 is gone"]


def test_a_reference_two_fit_is_asked_and_then_one_moved() -> None:
    s0, s1 = made("0 8 * * 1", sid="s0"), made("0 18 * * *", sid="s1")
    target = {0: "0 7 * * 1", 1: "0 7 * * *"}
    moved = seeded(s0, s1, answers=["Which one?", "Moved."], asked=[["A", "B"], None],
                   after=[s0, s1.model_copy(update={"cadence": "0 7 * * *"})])  # fmt: skip
    assert b.asked_which_moved(target)(moved) == []
    both = seeded(s0, s1, answers=["Moved both.", "OK."],
                  calls=[[("update_schedule", {}, "")] * 2, []],
                  after=[s0.model_copy(update={"cadence": "0 7 * * 1"}),
                         s1.model_copy(update={"cadence": "0 7 * * *"})])  # fmt: skip
    assert b.asked_which_moved(target)(both) == [
        "did not ask which",
        "wrote before asking",
        "2 moved, not one",
    ]


def test_an_end_date_is_where_the_last_run_falls() -> None:
    until = made("0 9 * * *")
    until.metadata[TASK]["until"] = "2026-12-31"
    dec31 = b.ended_by(lambda now: (date(2026, 12, 31), date(2026, 12, 31)))
    assert dec31(chat(until)) == []
    assert dec31(chat(made("0 9 * * *"))) == ["no end date kept"]
    early = made("0 9 * * *")
    early.metadata[TASK]["until"] = "2026-12-30"
    assert dec31(chat(early)) == ["last run on 2026-12-30, not 2026-12-31 to 2026-12-31"]
    s0 = made("0 9 * * *", sid="s0")
    s0.metadata[TASK]["until"] = "2026-11-30"
    march = s0.model_copy(update={"metadata": {**s0.metadata, TASK: {**s0.metadata[TASK],
                                                                       "until": "2027-03-31"}}})  # fmt: skip
    to_march = b.ended_by(lambda now: (date(2027, 3, 1), date(2027, 3, 31)), seed=0)
    assert to_march(seeded(s0, after=[march])) == []
    assert to_march(seeded(s0)) == ["last run on 2026-11-30, not 2027-03-01 to 2027-03-31"]
    open_ended = s0.model_copy(update={"metadata": {**s0.metadata, TASK: {"task": "x"}}})
    assert b.no_end(0)(seeded(s0, after=[open_ended])) == []
    assert b.no_end(0)(seeded(s0)) == ["an end is still kept"]


def test_a_limit_is_said_plainly() -> None:
    cap = b.plain_limit("2", which=r"schedules", action=r"cancel|delete")
    good = "You have 2 of 2 schedules, the most you can have. Cancel one to add this."
    assert cap(chat(answer=good)) == []
    assert cap(chat(answer="Error schedule_limit: 0 17 * * 5 refused.")) == [
        "answer lacks '2'",
        "answer does not name the limit (/schedules/)",
        "answer says no way on (/cancel|delete/)",
        "answer shows a cron",
        "answer shows an error code",
    ]
    assert cap(chat(made("0 9 * * *"), answer=good)) == ["made a schedule"]
    tries = [("schedule_task", {}, "")] * 3
    assert cap(chat(answer=good, calls=tries)) == ["schedule_task tried 3 times"]
    org = b.plain_limit("50", which="organi[sz]ation", action="admin", codes=("QUOTA_X",))
    assert org(chat(answer="Your organisation has 50 of 50 (QUOTA_X); ask an admin.")) == [
        "answer shows an error code"
    ]


def test_a_day_of_the_month_by_its_weekday() -> None:
    second = b.nth_weekday(2, 1, 10)  # the 2nd Tuesday, 10:00
    assert second(chat(made("0 10 * * 2#2"))) == []
    assert second(chat(made("0 10 * * 2", condition="only the second Tuesday (day 8-14)"))) == []
    assert second(chat(made("0 10 8-14 * 2"))) != []  # cron ORs the two: every Tuesday too
    assert second(chat(made("0 10 * * 2"))) != []
    assert second(chat()) == ["0 schedules made, not 1"]


def test_times_a_day_and_each_said() -> None:
    twice = b.times_a_day(2)
    assert twice(chat(made("0 9,18 * * *"), answer="At 9:00 and 18:00 every day.")) == []
    assert twice(chat(made("0 9,18 * * *"), answer="Twice a day.")) == [
        "answer does not say 09:00",
        "answer does not say 18:00",
    ]
    assert twice(chat(made("0 9 * * *"), answer="At 9:00.")) == ["runs 1 times a day, not 2"]
    assert twice(chat(made("0 9 * * *"), answer="9 am")) != []


def test_a_date_that_does_not_exist() -> None:
    no_29 = b.impossible_date((date(2027, 2, 28), date(2027, 3, 1)))
    said = "2027 is not a leap year, so there is no 29 February. 28 February or 1 March?"
    assert no_29(chat(answer=said)) == []
    assert no_29(chat(made("0 9 28 2 *", once=True), answer="Set for 28 Feb 2027.")) == []
    assert no_29(chat(made("0 9 29 2 *", once=True), answer="Set for 29 Feb.")) == [
        "first run 2028-02-29, not one of 2027-02-28, 2027-03-01"
    ]
    assert no_29(chat(answer="Done.")) == ["neither scheduled near it nor said the date is not"]


def test_the_shortest_repeat_is_the_deployments() -> None:
    floor = b.shortest_said()
    five = b.Chat(now=NOW, answers=["No: the shortest is every 5 minutes."], calls=[[]],
                  min_interval=300)  # fmt: skip
    assert floor(five) == []
    hour = b.Chat(now=NOW, answers=["Sure, every minute works."], calls=[[]], min_interval=3600)
    assert floor(hour) == ["answer lacks '1 hour'", "answer does not say no"]


def test_the_facts_of_this_harness_and_made_up_ones() -> None:
    off = b.explains(b.RUNS_WITHOUT_YOU, b.IN_CHAT, wrong=b.NEEDS_DEVICE)
    good = "It still runs on our servers; the reply waits in this chat for you."
    assert off(chat(answer=good)) == []
    assert off(chat(answer="It doesn't need your laptop on: it runs on the server; see chat."))\
        == []  # fmt: skip
    assert off(chat(answer="Keep your laptop on, or it runs on the server later in chat.")) != []
    holidays = b.explains(b.RUNS_ON_HOLIDAYS, wrong=b.SKIPS_HOLIDAYS)
    assert holidays(chat(answer="Yes, it still runs: it doesn't skip holidays.")) == []
    assert holidays(chat(answer="No, it skips public holidays.")) != []
    who = b.explains(b.ONLY_YOU, wrong=b.OTHERS_SEE)
    assert who(chat(answer="Only you: other users cannot see them.")) == []
    assert who(chat(answer="Only you; not known whether admins can see them.")) == []
    assert who(chat(answer="Only you, and your admins can see them too.")) != []
    zone = b.explains(b.KOLKATA, wrong=b.IN_NEW_YORK)
    assert zone(chat(answer="It runs in Asia/Kolkata (IST), not New York time.")) == []
    assert zone(chat(answer="It is in New York time.")) != []


def test_pending_cases_name_their_feature_and_are_reported_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = {c.id for c in b.cases()}
    assert set(b.PENDING) <= ids and all(b.PENDING.values())
    pending = "until-dec31"  # a case whose feature is pending is reported apart
    monkeypatch.setitem(b.PENDING, pending, "end_at")
    rows = [{"case": pending, "category": "ends", "passed": False, "problems": ["x"],
             "model_calls": 1, "input_tokens": 1, "seconds": 1, "usd": 0},
            {"case": "daily-9", "category": "schedule", "passed": True, "problems": [],
             "model_calls": 1, "input_tokens": 1, "seconds": 1, "usd": 0}]  # fmt: skip
    report = b.summary(rows)
    assert (report["passed"], report["cases"]) == (1, 1)
    assert report["pending"] == {pending: {"feature": b.PENDING[pending], "passed": False}}
    assert "ends" not in report["by_category"] and report["failed"] == {}


def test_every_group_is_covered() -> None:
    by = {c: [x for x in b.cases() if x.category == c] for c in b.CATEGORIES}
    assert all(by.values())
    assert sum(len(v) for v in by.values()) == len(b.cases()) >= 140


def test_a_schedule_never_run_paused_next_and_where_its_reply_goes() -> None:
    never = b.explains(b.NOT_RUN_YET, wrong=b.RAN_CLAIMS)
    assert never(chat(answer="It hasn't run yet: its first run is Monday at 8:00.")) == []
    assert never(chat(answer="It last ran on Monday at 8:00.")) != []
    resumed = b.explains(b.PAUSED, b.RESUME)
    assert resumed(chat(answer="Not until you resume it: it is paused.")) == []
    assert resumed(chat(answer="Tomorrow at 9:00.")) != []
    chat_only = b.explains(b.IN_CHAT, wrong=b.ELSEWHERE)
    assert chat_only(chat(answer="Here in this chat, not by email.")) == []
    assert chat_only(chat(answer="In this chat, and by email too.")) != []
    most = b.most_said()
    assert most(b.Chat(now=NOW, answers=["Up to 20."], calls=[[]], max_per_user=20)) == []
    assert most(b.Chat(now=NOW, answers=["Unlimited."], calls=[[]], max_per_user=20)) == [
        "answer lacks '20'"
    ]
