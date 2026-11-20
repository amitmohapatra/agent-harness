"""Chat schedules at their edges: a clock change, a delay before now or past a year, an end
before the first run, a limit met exactly or set per user, another tenant's or user's
schedule, a one-time schedule already gone, a pause after failures resumed, the zone the
model names over the app's, and a per-user expiry. The helpers are ``test_schedules``'."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from croniter import croniter

from tests.integration.test_schedules import (
    WATCH,
    Chat,
    Failing,
    asked,
    fire,
    local,
    only,
    planned,
    said,
    seeded,
)
from tests.integration.test_write_asks import THREAD as THREAD_
from tests.integration.test_write_asks import WAIT, fired, pay, refund, wire
from tests.integration.test_write_asks import Chat as Asks
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings
from trellis.contracts import RunStatus, ScheduleSpec
from trellis.harness import runs, schedules, snapshots
from trellis.harness.schedules import FAILURES, TOUCHED, ScheduleLimits
from trellis.runs import ScheduleUpdate


def fires_at(timed: Any, start: datetime) -> datetime:
    """The instant ``_in_minutes``' cron first fires after ``start``, in the zone it is for."""
    cron, _, zone = timed
    return croniter(cron, start.astimezone(ZoneInfo(zone))).get_next(datetime).astimezone(UTC)


def weekday_cron(days_on: int) -> str:
    """A weekly cron at 09:00 on the weekday ``days_on`` days from today (UTC)."""
    day = (datetime.now(UTC) + timedelta(days=days_on)).weekday()
    return f"0 9 * * {(day + 1) % 7}"


@pytest.mark.parametrize(
    ("start", "minutes", "at"),
    [
        # Berlin springs forward at 02:00 on 29 March 2026: 00:30 CET + 2 hours is 03:30 CEST
        (datetime(2026, 3, 28, 23, 30, tzinfo=UTC), 120, datetime(2026, 3, 29, 1, 30, tzinfo=UTC)),
        # and falls back at 03:00 on 25 October: 02:50 CEST + 30 minutes is 02:20 CET
        (datetime(2026, 10, 25, 0, 50, tzinfo=UTC), 30, datetime(2026, 10, 25, 1, 20, tzinfo=UTC)),
        # 01:50 CEST + 90 minutes is the second 02:20, not the first
        (datetime(2026, 10, 24, 23, 50, tzinfo=UTC), 90, datetime(2026, 10, 25, 1, 20, tzinfo=UTC)),
    ],
)
def test_in_minutes_across_a_clock_change_is_that_many_real_minutes(
    start: datetime, minutes: float, at: datetime
) -> None:
    """ "In 2 hours" is two hours of real time, whatever the user's clock does meanwhile:
    never an hour early, and never next year for a wall-clock time that comes twice."""
    assert fires_at(schedules._in_minutes(minutes, "Europe/Berlin", start), start) == at


def test_in_minutes_before_now_or_past_a_year_is_said_back() -> None:
    """A delay already gone, or one a cron cannot name (it has no year), is said back: never
    a run next year, or this year in the place of the next."""
    start = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    for minutes in (-30, 60 * 24 * 400, 1e15):
        said_ = schedules._in_minutes(minutes, "UTC", start)
        assert isinstance(said_, str) and "cannot be scheduled" in said_, said_
        assert "a delay ahead of at most a year" in said_
    assert fires_at(schedules._in_minutes(60 * 24 * 300, "UTC", start), start) == (
        start + timedelta(days=300)
    )


async def test_a_delay_said_back_makes_no_schedule() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        for minutes in (-30, 60 * 24 * 400):
            args = {"task": "call back", "in_minutes": minutes}
            answer = await said(agent, chat, "ada", ("schedule_task", args))
            assert "Scheduled" not in answer and not local(h)._schedules, answer


async def test_an_end_before_the_first_run_is_said_back_and_nothing_is_made() -> None:
    """ "Every <day in three days> at 9 until tomorrow" would never run: said back, never a
    schedule made that sits paused for good; a change to such an end is said back too."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        tomorrow = f"{datetime.now(UTC) + timedelta(days=1):%Y-%m-%d}"
        args = {"task": "stock", "when": weekday_cron(3), "end_at": tomorrow}
        answer = await said(agent, chat, "ada", ("schedule_task", args))
        assert "never run" in answer and not local(h)._schedules, answer
        await said(agent, chat, "ada", ("schedule_task", {**args, "end_at": ""}))
        sid = only(h).schedule_id
        change = {"id": sid, "end_at": tomorrow}
        answer = await said(agent, chat, "ada", ("update_schedule", change))
        assert "never run" in answer and only(h).end_at is None, answer
        in_a_week = f"{datetime.now(UTC) + timedelta(days=7):%Y-%m-%d}"
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "end_at": in_a_week}))
        month = datetime.now(UTC) + timedelta(days=30)
        later = {"id": sid, "when": f"0 9 {month.day} {month.month} *"}  # past the end
        answer = await said(agent, chat, "ada", ("update_schedule", later))
        assert "never run" in answer and only(h).cadence == weekday_cron(3), answer


@pytest.mark.parametrize(("least", "made"), [(3600, True), (3601, False)])
async def test_the_least_gap_allowed_is_allowed_exactly(least: float, made: bool) -> None:
    chat = Chat()
    async with Harness(config=Settings(schedule_min_interval=least)) as h:
        agent = h.wrap(chat, id="desk")
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 * * * *"})
        )
        assert bool(local(h)._schedules) is made, answer


async def test_the_per_user_hook_overrides_the_settings_either_way() -> None:
    """A user's own limits beat the environment's, looser or tighter: a shorter gap for one,
    no schedules at all for another (zero is a limit, not "unset"), the settings for the rest."""
    chat = Chat()

    async def plans(user: str, tenant: str) -> ScheduleLimits | None:
        return {
            "pro": ScheduleLimits(min_interval=900),
            "none": ScheduleLimits(max_per_user=0),
        }.get(user)

    async with Harness(config=Settings(schedule_min_interval=3600), schedule_limits=plans) as h:
        agent = h.wrap(chat, id="desk")
        often = {"task": "a", "when": "*/15 * * * *"}
        assert "Scheduled" in await said(agent, chat, "pro", ("schedule_task", often))
        assert "more often than every 1 hour" in await said(
            agent, chat, "ada", ("schedule_task", often)
        )
        daily = {"task": "a", "when": "0 9 * * *"}
        assert "You have 0 of 0 schedules" in await said(
            agent, chat, "none", ("schedule_task", daily)
        )
        assert [s.on_behalf_of for s in local(h)._schedules.values()] == ["pro"]


async def test_the_same_user_in_another_tenant_sees_and_changes_none_of_it() -> None:
    """A platform key names each run's tenant: "ada" of another tenant is another person, who
    lists, changes and cancels none of the first one's schedules."""
    chat = Chat()
    platform = FakeMemoryService(tenant=None)  # a platform key: each run names its tenant
    async with Harness(config=Settings(), memory=platform.client()) as h:
        agent = h.wrap(chat, id="desk")
        chat.calls.append(("schedule_task", {"task": "t1's", "when": "0 8 * * *"}))
        await agent.run("schedule it", user="ada", thread="c-1", tenant="t1")
        sid = only(h).schedule_id
        for name, args in [
            ("list_schedules", {}),
            ("update_schedule", {"id": sid, "enabled": False}),
            ("cancel_schedule", {"id": sid}),
            ("cancel_schedule", {"id": "all"}),
        ]:
            chat.calls.append((name, args))
            result = await agent.run("x", user="ada", thread="c-2", tenant="t2")
            assert "no schedule" in str(result.answer).lower(), (name, result.answer)
        assert only(h).enabled is True and only(h).tenant_id == "t1"


async def test_ids_naming_another_users_schedule_offer_only_the_users_own() -> None:
    """The model passing ids that fit the words, one of them someone else's: that one is
    neither offered nor changed; the user's one left is asked about (the words fit several),
    and changed once picked."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "stock", "when": "0 8 * * *"}))
        mine = only(h).schedule_id
        await said(agent, chat, "bob", ("schedule_task", {"task": "stock", "when": "0 7 * * *"}))
        [theirs] = [s for s in local(h)._schedules if s != mine]
        for name in ("update_schedule", "cancel_schedule"):
            chat.calls.append((name, {"id": theirs, "ids": [theirs, mine], "enabled": False}))
            result = await agent.run("change it", user="ada", thread="c-1")
            assert result.interrupt is not None, result
            assert result.interrupt.option_values == [mine]
            done = await agent.resume(
                result.interrupt.interrupt_id, "answer", answer=mine, reviewer="ada"
            )
            assert done.status is RunStatus.SUCCESS, done
            assert local(h)._schedules[theirs].enabled is True
        assert mine not in local(h)._schedules


async def test_a_one_time_schedule_that_ran_cannot_be_changed_or_cancelled() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "remind me", "when": "0 9 4 11 *", "once": True}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        chat.fired.append("reminder")
        await fire(h, agent, sid)
        for name, change in [
            ("update_schedule", {"id": sid, "when": "0 10 4 11 *"}),
            ("cancel_schedule", {"id": sid}),
        ]:
            assert f"No schedule {sid} of yours" in await said(agent, chat, "ada", (name, change))
        assert not local(h)._schedules


async def test_a_schedule_resumed_after_its_failures_counts_them_again_from_none() -> None:
    """Resumed by the user, a schedule paused for its failures starts its count over: its
    next failure alone does not pause it again, and list_schedules says one in a row."""
    chat = Failing()
    async with Harness(config=Settings(schedule_pause_after_failures=2)) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "sync", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        for _ in range(2):
            chat.fired.appendleft("x")
            await fire(h, agent, sid)
        assert only(h).enabled is False
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": True}))
        assert only(h).metadata[FAILURES] == 0
        assert "failed_runs_in_a_row" not in await said(agent, chat, "ada", ("list_schedules", {}))
        chat.fired.appendleft("x")
        record = await fire(h, agent, sid)
        assert record.error is not None and "paused" not in record.error.message
        assert only(h).enabled is True
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"failed_runs_in_a_row": 1' in listed, listed


async def test_the_zone_the_model_names_beats_the_apps_and_the_app_beats_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app says Berlin; "9 Tokyo time" is Tokyo's, the next one without a zone is the
    app's, and each fired run is told its own schedule's zone."""
    chat = Chat()
    app = snapshots.zone

    def berlin(runtime: Any) -> str | None:
        return "Europe/Berlin" if runtime.scheduled is None else app(runtime)

    monkeypatch.setattr(snapshots, "zone", berlin)
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        tokyo = {"task": "a", "when": "0 9 * * *", "timezone": "Asia/Tokyo"}
        await said(agent, chat, "ada", ("schedule_task", tokyo))
        await said(agent, chat, "ada", ("schedule_task", {"task": "b", "when": "0 9 * * *"}))
        zones = {s.name: s.timezone for s in local(h)._schedules.values()}
        assert zones == {"a": "Asia/Tokyo", "b": "Europe/Berlin"}
        for schedule in list(local(h)._schedules.values()):
            chat.fired.append("done")
            await fire(h, agent, schedule.schedule_id)
        assert chat.zones == ["Asia/Tokyo", "Europe/Berlin"]


async def test_a_users_own_days_before_asking_beat_the_settings() -> None:
    chat = Chat()

    async def plans(user: str, tenant: str) -> ScheduleLimits | None:
        return ScheduleLimits(expire_days=1) if user == "ada" else None

    async with Harness(config=Settings(), schedule_limits=plans) as h:
        agent = h.wrap(chat, id="desk")
        two_days = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        for user in ("ada", "bob"):
            args = {"task": f"{user}'s", "when": "0 8 * * *"}
            await said(agent, chat, user, ("schedule_task", args))
        for schedule in list(local(h)._schedules.values()):
            await local(h).schedules.update(
                schedule.schedule_id, ScheduleUpdate(metadata={TOUCHED: two_days})
            )
            chat.fired.append("done")
            await fire(h, agent, schedule.schedule_id)
        enabled = {s.on_behalf_of: s.enabled for s in local(h)._schedules.values()}
        assert enabled == {"ada": False, "bob": True}
        assert "still need" in chat.inputs[-2] and "still need" not in chat.inputs[-1]


@pytest.mark.parametrize(
    ("cadence", "times"),
    [("30 2 * * *", ["02:30 CEST"]), ("0 * * * *", ["02:00 CEST", "02:00 CET", "03:00 CET"])],
)
def test_a_time_the_autumn_change_repeats_fires_once_and_hourly_every_real_hour(
    cadence: str, times: list[str]
) -> None:
    """02:00-03:00 comes twice in Berlin on 25 October 2026: a schedule at that time of day
    fires on its first pass only, as cron and agent-runs do; an hourly one, a rate, fires on
    both."""
    spec = ScheduleSpec(
        tenant_id="t", agent_id="a", name="n", cadence=cadence, timezone="Europe/Berlin",
        on_behalf_of="ada", input=1,
    )  # fmt: skip
    fires, after = [], datetime(2026, 10, 24, 23, 30, tzinfo=UTC)  # 01:30 CEST
    while (due := runs._next_fire(spec, after)) is not None and due < datetime(
        2026, 10, 25, 2, 30, tzinfo=UTC
    ):
        fires.append(f"{due.astimezone(ZoneInfo('Europe/Berlin')):%H:%M %Z}")
        after = due
    assert fires == times


async def test_a_pick_whose_schedule_went_meanwhile_asks_again_never_acts_on_another() -> None:
    """The user's words fit two and they pick one; before the run resumes, that one is gone
    (cancelled elsewhere, or a one-time run ended it). The one left is not what they picked:
    they are asked again about it, and nothing is changed meanwhile."""
    async with Harness(config=Settings()) as h:
        for name in ("update_schedule", "cancel_schedule"):
            a, b = await seeded(
                h, name, WATCH, {"task": "check stock of SKU-2", "when": "0 6 * * *"}
            )
            args = {"id": a, "ids": [a, b], "enabled": False}
            agent = await planned(h, (name, args), name=name)
            interrupt = await asked(agent)
            assert [o.value for o in interrupt.options] == [a, b]
            await local(h).schedules.delete(a)
            again = await agent.resume(interrupt.interrupt_id, "answer", answer=a, reviewer="ada")
            assert again.status is RunStatus.PAUSED and again.interrupt is not None, again
            assert [o.value for o in again.interrupt.options] == [b]
            kept = local(h)._schedules[b]
            assert kept.enabled is True


async def test_runs_nobody_answered_count_as_failures_in_a_row_and_pause_the_schedule() -> None:
    """A "wait for me" schedule whose runs each wait a day for an answer nobody gives (the
    store ends them ``TIMEOUT``, never the harness): each counts as a failure in a row, found
    by the next run, and the one that makes them as many as pause it pauses the schedule; that
    run says so in the chat and does nothing else. An answered wait is no failure."""
    async with Harness(config=Settings(schedule_pause_after_failures=2)) as h:
        agent = arm(h.wrap(Asks(("refund", 2)), id="desk"), [pay, refund, wire])
        asked_ = await agent.run("pay my rent monthly", user="ada", thread=THREAD_)
        assert asked_.interrupt is not None
        await agent.resume(asked_.interrupt.interrupt_id, "answer", answer=WAIT, reviewer="ada")
        runs_ = local(h)
        first = await fired(h, agent)
        assert first.status is RunStatus.PAUSED and first.awaiting is not None
        await agent.resume(first.awaiting.interrupt_id, "approve", reviewer="ada")
        assert await h.worker([agent]).run_once() is True
        answered = await h.runs.get(first.run_id)
        assert answered is not None and answered.status is RunStatus.SUCCESS  # no failure
        for n in (1, 2):
            waits = await fired(h, agent)
            assert waits.status is RunStatus.PAUSED and waits.awaiting is not None, waits
            assert only(h).metadata.get(FAILURES, 0) == n - 1
            runs_._expire_interrupts(waits.awaiting.deadline + timedelta(minutes=1))
        told = await fired(h, agent)
        assert told.status is RunStatus.SUCCESS and told.awaiting is None, told
        assert "paused" in str(told.output) and "pay the rent" in str(told.output), told.output
        schedule = only(h)
        assert schedule.enabled is False and schedule.metadata[FAILURES] == 2
        listed = await agent.run("list", user="ada", thread=THREAD_)
        assert '"paused_because": "its runs failed 2 times in a row"' in str(listed.answer)
