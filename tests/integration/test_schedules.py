"""Schedules set in chat: ``schedule_task``, ``list_schedules``, ``update_schedule`` and
``cancel_schedule``, the built-in tools every adapter gets while the run store can schedule
(off: ``without=["schedules"]``), against the in-process run store. A fired run is the same
agent, user and thread; a run whose condition does not hold answers ``NO_REPORT`` and is not
delivered; a condition that still holds is reported once."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import AgentError, ErrorCategory, RunRecord, RunStatus, ScheduleSpec
from trellis.harness import pipeline, schedules, snapshots
from trellis.harness.budget import BUDGET, Budget
from trellis.harness.frontend import Frontend
from trellis.harness.notifications import CHAT, ChatChannel, Kind, Notification
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import (
    ALLOWED,
    BOTH,
    CONFIRM,
    FAILURES,
    MAX_FAILURES,
    NEITHER,
    NO_REPORT,
    PAUSED_BY,
    REPORT,
    REPORTED,
    SCHEDULE_TOOLS,
    STALE,
    TASK,
    TOUCHED,
    UNAPPROVED,
    UPDATE,
    ScheduleLimits,
    runs_said,
    schedule_tools,
)
from trellis.harness.tools.convert import text_of
from trellis.runs import (
    CadenceTooFrequentError,
    ConflictError,
    ScheduleLimitError,
    ScheduleUpdate,
)

THREAD = "chat-1"


class Chat:
    """A function target: in chat it calls the tools it is told to; a fired run answers what
    ``fired`` says next (``NO_REPORT`` or a report)."""

    def __init__(self) -> None:
        self.calls: deque[tuple[str, dict[str, Any]]] = deque()
        self.fired: deque[str | Exception] = deque()
        self.inputs: list[str] = []
        self.toolboxes: list[set[str]] = []
        self.said: list[str] = []
        self.described: list[str] = []
        self.zones: list[str | None] = []

    async def __call__(self, input: str, agent: Runtime) -> str:
        self.inputs.append(input)
        self.toolboxes.append(set(agent.toolbox))
        found = agent.toolbox.get("schedule_task")
        self.described.append(found.spec.description if found is not None else "")
        if input.startswith("Scheduled task:"):
            self.zones.append(snapshots.zone(agent))
            answer = self.fired.popleft()
            if isinstance(answer, Exception):
                raise answer
            return answer
        said = [text_of(await agent.tools.call(n, **a)) for n, a in self.calls]
        self.calls.clear()
        self.said = said
        return said[-1] if said else "ok"


async def said(agent: Any, chat: Chat, user: str, *calls: tuple[str, dict[str, Any]]) -> str:
    chat.calls.extend(calls)
    result = await agent.run("schedule it", user=user, thread=THREAD)
    assert result.status is RunStatus.SUCCESS, result.error
    return str(result.answer)


def local(h: Harness) -> LocalRuns:
    assert isinstance(h.runs, LocalRuns)
    return h.runs


async def fire(h: Harness, agent: Any, schedule_id: str) -> RunRecord:
    fired = await h.runs.schedules.fire(schedule_id)
    assert await h.worker([agent]).run_once() is True
    record = await h.runs.get(fired.run_id)
    assert record is not None
    return record


def only(h: Harness) -> Any:
    [schedule] = local(h)._schedules.values()
    return schedule


async def test_every_run_gets_the_tools_and_without_schedules_turns_them_off() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await agent.run("hi", user="ada")
        assert set(SCHEDULE_TOOLS) <= chat.toolboxes[-1]
        await agent.run("hi", user="ada", without={"schedules"})
        assert not set(SCHEDULE_TOOLS) & chat.toolboxes[-1]
        off = h.wrap(chat, id="off", without={"schedules"})
        await off.run("hi", user="ada")
        assert not set(SCHEDULE_TOOLS) & chat.toolboxes[-1]


async def test_a_fired_run_is_offered_no_schedule_tools_and_a_chat_the_day_and_hour() -> None:
    """A fired run does its task: it is offered no schedule tool (it made schedules of its
    own when it was). In chat ``schedule_task`` says the day and the hour, in UTC where the app
    gave no zone; ``in_minutes`` is once, that many minutes from now."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(
            agent, chat, "ada", ("schedule_task", {"task": "check stock", "when": "0 8 * * *"})
        )
        assert set(SCHEDULE_TOOLS) <= chat.toolboxes[-1]
        now = datetime.now(UTC)
        assert f"It is now {now:%a %Y-%m-%d}, {now:%H}:00-{now:%H}:59 (UTC;" in chat.described[-1]
        assert only(h).timezone == "UTC"
        chat.fired.append("12 left")
        await fire(h, agent, only(h).schedule_id)
        assert not set(SCHEDULE_TOOLS) & chat.toolboxes[-1]
        local(h)._schedules.clear()
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "call SUP-3", "in_minutes": 120})
        )
        later = only(h)
        # the minute it counted from, where the model reads the result (the tool's text says
        # only the hour): a time two hours on from 01:53 is 03:53, not a mistake to correct
        start = re.search(r"120 minutes from (\d\d:\d\d) \(UTC\), as asked\.$", answer)
        assert start is not None, answer
        assert start[1] in {f"{now:%H:%M}", f"{now + timedelta(minutes=1):%H:%M}"}
        due = later.next_fire_at
        assert later.metadata[TASK]["once"] is True and due is not None
        assert timedelta(minutes=119) <= due - now <= timedelta(minutes=121)


async def test_a_task_set_in_chat_is_the_users_in_its_thread_with_the_agents_limits() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk", run_timeout=90)
        args = {"task": "check stock of A-1", "when": "0 8 * * 1", "condition": "under 5"}
        answer = await said(agent, chat, "ada", ("schedule_task", args))
        schedule = only(h)
        assert schedule.on_behalf_of == "ada" and schedule.agent_id == "desk"
        assert schedule.cadence == "0 8 * * 1" and schedule.timeout_seconds == 90
        assert schedule.metadata["thread_id"] == THREAD
        assert schedule.metadata[TASK] == {
            "task": "check stock of A-1",
            "condition": "under 5",
            "once": False,
        }
        assert schedule.schedule_id in answer and "check stock of A-1" in answer


#: A schedule's next runs, as the model and the user read them: the day, the date, the hour.
NEXT_RUNS = re.compile(
    r"next Mon \d{1,2} \w{3}, 09:00, then Mon \d{1,2} \w{3}, 09:00 \(Europe/Berlin\)"
)


async def test_a_schedule_is_said_by_its_next_runs_in_the_users_zone_never_its_cron() -> None:
    """The user never reads a cron (handoff 2.4): what the model is told it scheduled, and what
    it lists, say the next two runs as dates in the schedule's zone; the cron stays inside."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "weekly stock report", "when": "0 9 * * 1", "timezone": "Europe/Berlin"}
        answer = await said(agent, chat, "ada", ("schedule_task", args))
        assert "0 9 * * 1" not in answer and NEXT_RUNS.search(answer), answer
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert "0 9 * * 1" not in listed and NEXT_RUNS.search(listed), listed
        assert len(local(h)._schedules) == 1  # said once, made once
        assert only(h).cadence == "0 9 * * 1"  # the cron is kept, only not said


def test_a_one_time_run_says_its_one_date_and_its_year_when_not_this_one() -> None:
    after = datetime(2026, 12, 30, 12, tzinfo=UTC)
    assert (
        runs_said("0 9 4 1 *", "UTC", once=True, after=after) == "next Mon 4 Jan 2027, 09:00 (UTC)"
    )
    assert runs_said("0 9 * * *", "Asia/Kolkata", after=after) == (
        "next Thu 31 Dec, 09:00, then Fri 1 Jan 2027, 09:00 (Asia/Kolkata)"
    )
    assert runs_said("not a cron", "UTC") == "'not a cron' (UTC)"


@pytest.mark.parametrize(
    ("when", "said_"),
    [
        ("every monday", "is not a cron expression"),
        ("0 8 * * 1 2026", "is not a cron expression"),
        ("*/2 * * * *", "more often than every 1 hour"),
    ],
)
async def test_a_bad_or_too_frequent_time_is_said_back(when: str, said_: str) -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        answer = await said(agent, chat, "ada", ("schedule_task", {"task": "x", "when": when}))
        assert said_ in answer and not local(h)._schedules


async def test_the_env_limit_and_the_per_user_hook() -> None:
    chat = Chat()
    settings = Settings(schedule_max_per_user=2, schedule_min_interval=3600)
    async with Harness(config=settings) as h:
        agent = h.wrap(chat, id="desk")
        for task in ("check stock", "send revenue"):
            await said(agent, chat, "ada", ("schedule_task", {"task": task, "when": "0 9 * * *"}))
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "list suppliers", "when": "0 9 * * *"})
        )
        assert "You have 2 of 2 schedules" in answer
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "t", "when": "*/30 * * * *"})
        )
        assert "more often than every 1 hour" in answer
        # another user has their own count
        await said(agent, chat, "bob", ("schedule_task", {"task": "t", "when": "0 9 * * *"}))

    seen: list[tuple[str, str]] = []

    async def plans(user: str, tenant: str) -> ScheduleLimits | None:
        seen.append((user, tenant))
        return ScheduleLimits(max_per_user=1) if user == "free" else None

    # every 5 minutes is the env's here, for the users the hook leaves to it
    async with Harness(config=Settings(schedule_min_interval=300), schedule_limits=plans) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "free", ("schedule_task", {"task": "a", "when": "0 9 * * *"}))
        answer = await said(
            agent, chat, "free", ("schedule_task", {"task": "b", "when": "0 9 * * *"})
        )
        assert "You have 1 of 1 schedules" in answer
        await said(agent, chat, "paid", ("schedule_task", {"task": "a", "when": "*/5 * * * *"}))
        await said(agent, chat, "paid", ("schedule_task", {"task": "b", "when": "0 9 * * *"}))
        assert len(local(h)._schedules) == 3 and seen[0][0] == "free"


async def test_each_user_lists_and_changes_only_their_own() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "ada's", "when": "0 8 * * 1"}))
        ada = only(h).schedule_id
        await said(agent, chat, "bob", ("schedule_task", {"task": "bob's", "when": "0 9 * * *"}))
        # a schedule made in code is not a chat task: not listed
        await agent.schedule("0 7 * * *", user="ada", input="digest")
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert "ada's" in listed and "bob's" not in listed and "digest" not in listed
        for name, args in [
            ("update_schedule", {"id": ada, "when": "0 9 * * 1"}),
            ("cancel_schedule", {"id": ada}),
        ]:
            refused = await said(agent, chat, "bob", (name, args))
            assert f"No schedule {ada} of yours" in refused
        assert local(h)._schedules[ada].cadence == "0 8 * * 1"
        changed = await said(
            agent,
            chat,
            "ada",
            ("update_schedule", {"id": ada, "when": "0 9 * * 1", "enabled": False}),
        )
        kept = local(h)._schedules[ada]
        assert kept.cadence == "0 9 * * 1" and kept.enabled is False
        assert '"status": "paused"' in changed and "0 9 * * 1" not in changed  # never its cron
        assert "paused" in await said(agent, chat, "ada", ("list_schedules", {}))
        await said(agent, chat, "ada", ("update_schedule", {"id": ada, "task": "ada's new"}))
        assert local(h)._schedules[ada].metadata[TASK]["task"] == "ada's new"
        assert "Cancelled" in await said(agent, chat, "ada", ("cancel_schedule", {"id": ada}))
        assert ada not in local(h)._schedules
        assert "no schedules" in await said(agent, chat, "ada", ("list_schedules", {}))


async def test_a_fired_run_is_the_same_user_and_thread_and_no_report_is_not_delivered() -> None:
    chat = Chat()
    service = FakeMemoryService()
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "check stock of A-1", "when": "0 8 * * 1", "condition": "under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        await h.writes.drain()
        [turn] = [c for c in service.calls if c.name == "messages"]
        service.calls.clear()
        chat.fired.append("NO_REPORT")
        quiet = await fire(h, agent, sid)
        # nobody is present at a fire: the run acts for the person the schedule is for
        assert (quiet.thread_id, quiet.user_id, quiet.on_behalf_of) == (THREAD, None, "ada")
        assert quiet.status is RunStatus.SUCCESS and quiet.output == NO_REPORT
        assert "under 5" in chat.inputs[-1] and "check stock of A-1" in chat.inputs[-1]
        await h.writes.drain()
        assert not [c for c in service.calls if c.name == "messages"]  # not in the history
        chat.fired.append("Only 3 of A-1 left.")
        told = await fire(h, agent, sid)
        assert told.output == "Only 3 of A-1 left."
        await h.writes.drain()
        [written] = [c for c in service.calls if c.name == "messages"]
        # in the chat's own conversation: what the user sees when they open it
        assert "Only 3 of A-1 left." in str(written.body)
        assert written.scope["thread_id"] == turn.scope["thread_id"]


async def test_a_condition_that_still_holds_is_reported_once() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "check stock", "when": "0 8 * * *", "condition": "under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        outputs = []
        for answer in ["3 left", "2 left", "NO_REPORT.", "4 left"]:
            chat.fired.append(answer)
            outputs.append((await fire(h, agent, sid)).output)
        assert outputs == ["3 left", NO_REPORT, NO_REPORT, "4 left"]
        assert local(h)._schedules[sid].metadata[REPORTED] is True


async def test_without_a_condition_every_run_reports_and_once_ends_the_schedule() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "say hi", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        for _ in range(2):
            chat.fired.append("hi")
            assert (await fire(h, agent, sid)).output == "hi"
        args = {"task": "remind me", "when": "0 9 4 11 *", "once": True}
        await said(agent, chat, "ada", ("schedule_task", args))
        [once] = [s for s in local(h)._schedules.values() if s.schedule_id != sid]
        chat.fired.append("reminder")
        assert (await fire(h, agent, once.schedule_id)).output == "reminder"
        assert once.schedule_id not in local(h)._schedules


async def test_local_schedules_list_update_delete_by_tenant() -> None:
    from trellis.contracts import ScheduleSpec
    from trellis.runs import NotFoundError, ScheduleUpdate

    runs = LocalRuns()
    spec = ScheduleSpec(
        tenant_id="t1", agent_id="a", name="n", cadence="daily", on_behalf_of="ada", input=1
    )
    made = await runs.schedules.create(spec)
    other = await runs.schedules.create(spec.model_copy(update={"on_behalf_of": "bob"}))
    page = await runs.schedules.list(agent_id="a", on_behalf_of="ada", tenant="t1")
    assert [s.schedule_id for s in page.items] == [made.schedule_id] and page.next_cursor is None
    assert [s.schedule_id for s in (await runs.schedules.list(tenant="t2")).items] == []
    assert len((await runs.schedules.list(enabled=True)).items) == 2
    assert await runs.schedules.get(made.schedule_id, tenant="t2") is None
    changed = await runs.schedules.update(
        made.schedule_id,
        ScheduleUpdate(cadence="0 9 * * *", enabled=False, metadata={"a": 1}),
        tenant="t1",
    )
    assert changed.cadence == "0 9 * * *" and not changed.enabled and changed.metadata == {"a": 1}
    assert changed.next_fire_at is not None and changed.next_fire_at > datetime.now(UTC)
    again = await runs.schedules.update(made.schedule_id, ScheduleUpdate(metadata={"b": 2}))
    assert again.metadata == {"a": 1, "b": 2}
    assert (await runs.schedules.list(enabled=False)).items == [again]
    with pytest.raises(NotFoundError):
        await runs.schedules.update(other.schedule_id, ScheduleUpdate(enabled=False), tenant="t2")
    await runs.schedules.delete(made.schedule_id, tenant="t1")
    with pytest.raises(NotFoundError):
        await runs.schedules.delete(made.schedule_id)
    assert await runs.schedules.get(made.schedule_id) is None
    # a schedule from a chat fires into its thread
    chat = await runs.schedules.create(
        spec.model_copy(update={"input": 2, "metadata": {"thread_id": "c-1"}})
    )
    fired = await runs.schedules.fire(chat.schedule_id, at=datetime.now(UTC) - timedelta(1))
    record = await runs.get(fired.run_id)
    assert record is not None and record.thread_id == "c-1"


async def test_refusals_a_threadless_run_and_a_long_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.contracts import ScheduleSpec
    from trellis.runs import ConflictError

    chat = Chat()
    async with Harness(config=Settings(schedule_max_per_user=100)) as h:
        agent = h.wrap(chat, id="desk")
        bad_zone = {"task": "x", "when": "0 9 * * *", "timezone": "Mars/Base"}
        refused = await said(agent, chat, "ada", ("schedule_task", bad_zone))
        assert refused.startswith("That schedule was refused") and not local(h)._schedules
        chat.calls.append(("schedule_task", {"task": "no chat", "when": "0 9 * * *"}))
        await agent.run("schedule it", user="ada")  # a run with no thread of its own
        sid = only(h).schedule_id
        assert "thread_id" not in only(h).metadata
        too_often = await said(
            agent, chat, "ada", ("update_schedule", {"id": sid, "when": "* * * * *"})
        )
        assert "more often" in too_often
        changed = await said(
            agent, chat, "ada", ("update_schedule", {"id": sid, "condition": "low"})
        )
        assert (
            '"condition": "low"' in changed and local(h)._schedules[sid].metadata[REPORTED] is False
        )

        async def conflict(*args: Any, **kwargs: Any) -> Any:
            raise ConflictError("another schedule is the same", code="CONFLICT", status=409)

        with monkeypatch.context() as m:
            m.setattr(h.runs.schedules, "update", conflict)
            answer = await said(
                agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": False})
            )
        assert answer.startswith("That change was refused")
        tenant = only(h).tenant_id
        for n in range(60):  # past one page of the store's listing
            spec = ScheduleSpec(
                tenant_id=tenant,
                agent_id="desk",
                name=f"t{n}",
                cadence="daily",
                on_behalf_of="ada",
                input=n,
                metadata={TASK: {"task": f"t{n}", "condition": None, "once": False}},
            )
            await h.runs.schedules.create(spec)
        assert (await said(agent, chat, "ada", ("list_schedules", {}))).count('"id"') == 61


async def test_a_schedule_gone_or_unwritable_after_its_run_leaves_the_run_as_it_ended(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trellis.runs import NotFoundError

    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "check", "when": "0 9 * * *", "condition": "low", "once": True}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id

        async def gone(*args: Any, **kwargs: Any) -> Any:
            raise NotFoundError("no schedule", code="NOT_FOUND", status=404)

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

        monkeypatch.setattr(h.runs.schedules, "delete", gone)
        chat.fired.append("low: 2")
        assert (await fire(h, agent, sid)).output == "low: 2"
        await said(agent, chat, "ada", ("schedule_task", {**args, "task": "again", "once": False}))
        [again] = [s for s in local(h)._schedules.values() if s.schedule_id != sid]
        monkeypatch.setattr(h.runs.schedules, "update", broken)
        chat.fired.append("low: 1")
        assert (await fire(h, agent, again.schedule_id)).output == "low: 1"
        assert "not updated after its run" in caplog.text


async def test_a_graph_built_with_its_tools_takes_them_from_the_harness() -> None:
    async with Harness(config=Settings()) as h:
        assert [t.name for t in h.schedule_tools()] == list(SCHEDULE_TOOLS)


# --------------------------------------------------------------------------- asked with options


@tool(side_effects="write")
def send_report(to: str = "me") -> str:
    """Email the report.

    Args:
        to: Who it goes to.
    """
    return f"sent to {to}"


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock now.

    Args:
        sku: The SKU, such as SKU-1.
    """
    return 12


WATCH = {"task": "check stock of SKU-1", "when": "0 9 * * *", "condition": "SKU-1 under 5"}


async def planned(
    h: Harness,
    *calls: tuple[str, dict[str, Any]],
    name: str = "desk",
    tools: Sequence[Any] = (),
) -> Any:
    """``ReAct`` on a planned model making ``calls`` (the stock check first), with ``stock``
    and ``tools``."""
    model = PlannedChat([("stock", {"sku": "SKU-1"}), *calls])
    return arm(h.wrap(ReAct(system_prompt="Help.", model=model), id=name), [stock, *tools])


async def seeded(h: Harness, agent: str, *abouts: dict[str, Any]) -> list[str]:
    """Schedules ``ada`` has with the agent ``agent``, as ``schedule_task`` makes them."""
    made = []
    for about in abouts:
        spec = ScheduleSpec(
            tenant_id=await h._tenant(None), agent_id=agent, name=about["task"],
            cadence=about["when"], on_behalf_of="ada",
            metadata={TASK: {"task": about["task"], "condition": about.get("condition"),
                             "once": False}},
        )  # fmt: skip
        made.append((await h.runs.schedules.create(spec)).schedule_id)
    return made


async def asked(agent: Any) -> Any:
    result = await agent.run("schedule it", user="ada")
    assert result.status is RunStatus.PAUSED and result.interrupt is not None, result.answer
    return result.interrupt


async def picked(agent: Any, interrupt: Any, value: str) -> str:
    done = await agent.resume(interrupt.interrupt_id, "answer", answer=value, reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    return str(done.answer)


@pytest.mark.parametrize(("value", "conditions"), [
    (CONFIRM, ["SKU-1 is 3 or less"]),  # the recommended answer: update it
    ({"alike": UPDATE}, ["SKU-1 is 3 or less"]),
    ({"alike": BOTH}, ["SKU-1 is 3 or less", "SKU-1 under 5"]),
    (NEITHER, ["SKU-1 under 5"]),
])  # fmt: skip
async def test_a_schedule_the_model_judges_alike_is_asked_about_update_first(
    value: Any, conditions: list[str]
) -> None:
    """The same thing watched with another value, judged so by the model (``alike``, from
    list_schedules, handoff 2.5): in the schedule's one question the user picks — update it
    (recommended: the default), keep both, or cancel; the very same again is said to be set
    already."""
    async with Harness(config=Settings()) as h:
        [had] = await seeded(h, "desk", WATCH)
        lower = {**WATCH, "condition": "SKU-1 is 3 or less", "alike": had}
        agent = await planned(h, ("schedule_task", lower))
        interrupt = await asked(agent)
        assert [o.value for o in interrupt.options] == [CONFIRM, NEITHER]
        assert [o.recommended for o in interrupt.options] == [True, None]
        form = (interrupt.expects or {}).get("anyOf", [{}])[-1].get("properties", {})
        assert form["alike"]["enum"] == [UPDATE, BOTH] and form["alike"]["default"] == UPDATE
        await picked_value(agent, interrupt, value)
        found = [s.metadata[TASK]["condition"] for s in local(h)._schedules.values()]
        assert sorted(found) == sorted(conditions)
        same = {**WATCH, "condition": "SKU-1 below 5"}  # no alike: the very same is caught
        again = await planned(h, ("schedule_task", same), name="again")
        await seeded(h, "again", WATCH)
        result = await again.run("schedule it", user="ada")
        assert "Already set, nothing changed" in str(result.answer)


async def test_a_task_the_model_judges_alike_at_another_time_is_asked_about() -> None:
    async with Harness(config=Settings()) as h:
        [had] = await seeded(h, "desk", {"task": "Send the sales summary", "when": "0 8 * * 1"})
        later = {"task": "send me the sales summary", "when": "0 9 * * 1", "alike": had}
        agent = await planned(h, ("schedule_task", later))
        await picked_value(agent, await asked(agent), {"alike": UPDATE})
        assert [s.cadence for s in local(h)._schedules.values()] == ["0 9 * * 1"]


async def test_alike_words_the_model_did_not_judge_alike_are_made_with_no_rule_of_code() -> None:
    """No word-overlap rule decides (handoff 2.5: the model judges): without alike, a task in
    words like one the user has is made, and an alike that names no schedule of theirs asks
    nothing."""
    async with Harness(config=Settings()) as h:
        await seeded(h, "desk", {"task": "Send the sales summary", "when": "0 8 * * 1"})
        later = {"task": "send me the sales summary", "when": "0 9 * * 1"}
        agent = await planned(h, ("schedule_task", later))
        assert "Scheduled" in str((await agent.run("schedule it", user="ada")).answer)
        other = {"task": "check returns", "when": "0 7 * * 1", "alike": "sch_none"}
        made = await planned(h, ("schedule_task", other), name="other")
        assert "Scheduled" in str((await made.run("schedule it", user="ada")).answer)


async def test_cancel_all_is_confirmed_and_several_that_fit_are_picked_from() -> None:
    async with Harness(config=Settings()) as h:
        keep = await planned(h, ("cancel_schedule", {"id": "all"}), name="keep")
        await seeded(h, "keep", WATCH, {"task": "sales summary", "when": "0 8 * * 1"})
        interrupt = await asked(keep)
        assert interrupt.question == "Cancel all 2 of your schedules?"
        assert "Nothing cancelled" in await picked(keep, interrupt, NEITHER)
        assert len(local(h)._schedules) == 2
        gone = await planned(h, ("cancel_schedule", {"id": "all"}), name="gone")
        await seeded(h, "gone", WATCH)
        assert "Cancelled all 1" in await picked(gone, await asked(gone), UPDATE)
        assert len(local(h)._schedules) == 2  # the other agent's
        empty = await planned(h, ("cancel_schedule", {"id": "all"}), name="empty")
        assert "You have no schedules" in str((await empty.run("x", user="ada")).answer)
        # the user's words fit two: they pick which
        a, b = await seeded(
            h, "which", WATCH, {"task": "check stock of SKU-2", "when": "0 6 * * *"}
        )
        cancel = {"id": a, "ids": [a, b, "nope"]}
        which = await planned(h, ("cancel_schedule", cancel), name="which")
        interrupt = await asked(which)
        assert [o.value for o in interrupt.options] == [a, b]
        await picked(which, interrupt, b)
        assert a in local(h)._schedules and b not in local(h)._schedules


async def test_a_fired_run_is_told_the_schedules_zone_not_utc() -> None:
    """A fired run has no app to say the user's zone: it is the schedule's own (handoff 2.2:
    never silently UTC), so "every Monday at 8" in Kolkata reads Monday there."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "sales summary", "when": "0 8 * * 1", "timezone": "Asia/Kolkata"}
        await said(agent, chat, "ada", ("schedule_task", args))
        chat.fired.append("done")
        await fire(h, agent, only(h).schedule_id)
        assert chat.zones == ["Asia/Kolkata"]


async def test_list_schedules_says_the_last_run_and_its_failures() -> None:
    """Asked why a report stopped, the model reads it: the last run, failures in a row, and
    for a condition whether it was reported (the facts the schedule keeps)."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "stock of A-1", "when": "0 8 * * 1", "condition": "under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        chat.fired.append(RuntimeError("Gmail: sign-in expired"))
        assert (await fire(h, agent, only(h).schedule_id)).status is RunStatus.ERROR
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"failed_runs_in_a_row": 1' in listed and '"last_run": "' in listed, listed
        assert '"reported": false' in listed


async def test_list_schedules_says_a_schedule_has_not_run_yet_and_whose_it_is() -> None:
    """"When did my sales summary last run?" of one that never has: the listing said nothing
    of a last run, so every model answered that it could not see one. It says "not yet"; and
    the tool says a user's schedules are theirs ("Who else can see my schedules?")."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "sales summary", "when": "0 8 * * 1"}
        await said(agent, chat, "ada", ("schedule_task", args))
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"last_run": "not yet"' in listed, listed
    described = next(t for t in schedule_tools() if t.name == "list_schedules").spec.description
    assert "no other user" in described


async def test_a_failed_one_time_run_is_paused_and_said_not_deleted() -> None:
    """A one-time schedule whose run fails stays, paused, and the error says so: "remind me
    tomorrow at 9" is not lost to one tool's failure. A run that succeeds still ends it."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "remind me", "when": "0 9 4 11 *", "once": True}
        await said(agent, chat, "ada", ("schedule_task", args))
        chat.fired.append(RuntimeError("the reminder tool is down"))
        record = await fire(h, agent, only(h).schedule_id)
        assert record.status is RunStatus.ERROR and record.error is not None
        assert "paused" in record.error.message and "remind me" in record.error.message
        kept = only(h)
        assert kept.enabled is False
        await local(h).schedules.update(kept.schedule_id, ScheduleUpdate(enabled=True))
        chat.fired.append("done")
        await fire(h, agent, kept.schedule_id)
        assert not local(h)._schedules  # ran: ended


async def test_in_minutes_rounds_up_to_the_next_whole_minute() -> None:
    """A delay that ends inside a minute fires in the next one, never next year."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "check again", "in_minutes": 0.2}
        await said(agent, chat, "ada", ("schedule_task", args))
        due = only(h).next_fire_at
        assert due is not None and due - datetime.now(UTC) < timedelta(minutes=2), due


@pytest.mark.parametrize(
    ("report", "outputs", "left"),
    [
        ("until_clears", ["low: 2", "low: 1"], ["low: 2", NO_REPORT]),  # today's, the default
        ("every_time", ["low: 2", "low: 1"], ["low: 2", "low: 1"]),
        ("once", ["low: 2"], ["low: 2"]),  # told, then it ends
    ],
)
async def test_a_condition_reports_as_the_user_asked(
    report: str, outputs: list[str], left: list[str]
) -> None:
    """How often a watch reports is the model's choice from the user's words, said in the
    tool: until it clears (once, then quiet until it stops holding: the default), every time
    it holds, or once and then the schedule ends ("let me know when it is under 5")."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "stock of A-1", "when": "0 8 * * *", "condition": "under 5",
                "report": report}  # fmt: skip
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        got = []
        for answer in outputs:
            chat.fired.append(answer)
            got.append((await fire(h, agent, sid)).output)
        assert got == left
        assert bool(local(h)._schedules) == (report != "once")


def test_the_schedule_tool_says_its_rules_as_facts_from_the_settings() -> None:
    """What schedule_task's description says is true of this deployment, as facts the model
    decides by (handoff 2.x): the default time and the shortest repeat from the settings, the
    last day of a month, several times in one schedule, the report rule, and times in words."""
    settings = Settings(schedule_min_interval=3600, schedule_default_time="08:30")
    [schedule] = [t for t in schedule_tools(settings=settings) if t.name == "schedule_task"]
    said = schedule.spec.description
    when = str((schedule.spec.input_schema or {})["properties"]["when"]["description"])
    assert (
        "08:30" in said and "One schedule a request" not in said and "No time said: 9" not in said
    )
    assert "every 1 hour" in said  # the shortest repeat in force
    assert "0 9,17 * * *" in said and "' L " not in said and "L * *" in when  # last day: L
    assert "A condition reports as report says" in said
    assert "never the cron" in said
    default = next(t for t in schedule_tools() if t.name == "schedule_task").spec.description
    assert "09:00" in default and "every 1 hour" in default  # the defaults, said
    odd = Settings(schedule_min_interval=90)
    assert "every 90 seconds" in next(iter(schedule_tools(settings=odd))).spec.description


async def test_a_schedule_question_takes_any_time_the_user_gives_not_only_its_suggestions() -> None:
    """The question says what an answer may be (how often, the days, the time, a date, an end)
    as a JSON Schema the app renders from the user's own settings; the suggestions are
    shortcuts. An answer that is none of them, in words or the app's fields, goes back to the
    model as the user's own: nothing is made in its place, nothing fails."""
    choices = [{"task": "revenue yesterday", "when": "0 9 * * *"}]
    async with Harness(config=Settings()) as h:
        agent = await planned(h, ("schedule_task", {"task": "watch sales", "choices": choices}))
        interrupt = await asked(agent)
        when = (interrupt.expects or {}).get("anyOf", [{}])[-1].get("properties", {})
        assert {"repeat", "days", "time", "date", "ends"} <= set(when), interrupt.expects
        assert when["time"].get("format") == "time" and when["date"].get("format") == "date"
        said = await picked(agent, interrupt, "Tuesdays at 7 pm")
        assert "Tuesdays at 7 pm" in said and not local(h)._schedules  # the model's to make
        again = await planned(h, ("schedule_task", {"task": "x", "choices": choices}), name="b")
        fields = {"repeat": "weekly", "days": ["tue"], "time": "19:00"}
        told = await picked_value(again, await asked(again), fields)
        assert '"days": ["tue"]' in told and '"time": "19:00"' in told
        assert not local(h)._schedules


async def picked_value(agent: Any, interrupt: Any, value: Any) -> str:
    done = await agent.resume(interrupt.interrupt_id, "answer", answer=value, reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    return str(done.answer)


async def test_a_vague_ask_is_choices_the_user_picks_one_of() -> None:
    choices = [
        {"task": "revenue yesterday", "when": "0 9 * * *"},
        {"task": "top 5 SKUs", "when": "0 9 * * 1"},
        {"when": "0 9 * * *"},  # no task: not offered
    ]
    async with Harness(config=Settings()) as h:
        agent = await planned(h, ("schedule_task", {"task": "watch sales", "choices": choices}))
        interrupt = await asked(agent)
        assert interrupt.question == "Which should I set up?"
        assert [o.value for o in interrupt.options] == ["0", "1", NEITHER]
        # each button says when it would run, never its cron
        labels = [str(o.label) for o in interrupt.options[:2]]
        assert not [label for label in labels if "* *" in label], labels
        assert all(re.search(r"next \w{3} \d{1,2} \w{3}, 09:00", label) for label in labels)
        assert "Scheduled" in await picked(agent, interrupt, "1")
        assert [s.cadence for s in local(h)._schedules.values()] == ["0 9 * * 1"]
        none = await planned(h, ("schedule_task", {"task": "x", "choices": choices}), name="no")
        assert "none of these" in await picked(none, await asked(none), NEITHER)
        bad = await planned(
            h, ("schedule_task", {"task": "x", "choices": [{"when": "x"}]}), name="b"
        )
        assert "Each choice needs" in str((await bad.run("x", user="ada")).answer)
        four = [{"task": f"report {n}", "when": "0 9 * * *"} for n in range(4)]
        many = await planned(h, ("schedule_task", {"task": "x", "choices": four}), name="many")
        offered = await asked(many)
        assert [o.value for o in offered.options] == ["0", "1", "2", NEITHER]  # 3 at most


class Failing(Chat):
    """A fired run whose next answer is ``x`` fails."""

    async def __call__(self, input: str, agent: Runtime) -> str:
        if input.startswith("Scheduled task:") and self.fired[0] == "x":
            self.fired.popleft()
            raise RuntimeError("upstream down")
        return await super().__call__(input, agent)


async def test_three_failed_runs_in_a_row_pause_the_schedule_and_say_so_in_the_thread() -> None:
    """A success between failures resets the count; the third failure in a row pauses the
    schedule, and its error (the thread's reply, the run's record and its webhook) says so."""
    chat = Failing()
    async with Harness(config=Settings()) as h:
        chat_agent = h.wrap(chat, id="desk")
        args = {"task": "sync", "when": "0 8 * * *"}
        await said(chat_agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        chat.fired.extend(["x", "fine"])  # "x": the run fails
        assert (await fire(h, chat_agent, sid)).status is RunStatus.ERROR
        assert local(h)._schedules[sid].metadata[FAILURES] == 1
        assert (await fire(h, chat_agent, sid)).status is RunStatus.SUCCESS
        assert local(h)._schedules[sid].metadata[FAILURES] == 0
        records = []
        for _ in range(MAX_FAILURES):
            chat.fired.appendleft("x")
            records.append(await fire(h, chat_agent, sid))
        assert [r.status for r in records] == [RunStatus.ERROR] * MAX_FAILURES
        assert local(h)._schedules[sid].enabled is False
        assert "paused" not in records[0].error.message
        assert "failed 3 times in a row" in records[-1].error.message
        assert "sync" in records[-1].error.message


async def test_a_schedule_untouched_for_90_days_is_paused_and_its_run_asks_in_the_models_words() -> (
    None
):
    """Handoff 2.6: the question asked of a schedule left unchanged for its days
    (``TRELLIS_SCHEDULE_EXPIRE_DAYS``) is the run's model's, in the user's language: the run
    still does its task (its report not lost), is told to end by asking whether the user still
    needs it, and is delivered even when its condition has nothing to report; the schedule is
    paused (``paused_because``) until the user keeps it in chat, which renews it, or cancels it.
    """
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "sales", "when": "0 8 * * *", "condition": "sales under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        assert TOUCHED in local(h)._schedules[sid].metadata  # set when made
        old = (datetime.now(UTC) - STALE - timedelta(days=1)).isoformat()
        await local(h).schedules.update(sid, ScheduleUpdate(metadata={TOUCHED: old}))
        chat.fired.append("Sales: 12, nothing under 5. Do you still need this schedule?")
        done = await fire(h, agent, sid)
        assert done.status is RunStatus.SUCCESS and done.awaiting is None
        assert "still need" in chat.inputs[-1] and "language" in chat.inputs[-1]
        assert "Do you still need" in str(done.output)  # delivered, though not met
        schedule = local(h)._schedules[sid]
        assert schedule.enabled is False and schedule.metadata[PAUSED_BY]
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"paused_because": "it waits for the user to keep it"' in listed
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": True}))
        kept = local(h)._schedules[sid]
        assert kept.enabled is True and kept.metadata[TOUCHED] > old
        assert not kept.metadata.get(PAUSED_BY)
        chat.fired.append(NO_REPORT)  # renewed: an ordinary run again, quiet when not met
        again = await fire(h, agent, sid)
        assert "still need" not in chat.inputs[-1] and again.output == NO_REPORT


async def test_a_failure_not_counted_leaves_the_runs_error_as_it_was(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trellis.runs import NotFoundError

    chat = Failing()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "sync", "when": "0 8 * * *"}))
        sid = only(h).schedule_id

        async def gone(*args: Any, **kwargs: Any) -> Any:
            raise NotFoundError("no schedule", code="NOT_FOUND", status=404)

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

        for update in (gone, broken):
            monkeypatch.setattr(h.runs.schedules, "update", update)
            chat.fired.append("x")
            record = await fire(h, agent, sid)
            assert record.status is RunStatus.ERROR and record.error is not None
            assert "upstream down" in record.error.message
        assert "failure not counted" in caplog.text


async def test_a_too_frequent_time_is_said_in_words_never_its_cron() -> None:
    """The model tells the user what the tool says: the gap and the limit in words."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "x", "when": "*/2 * * * *"}
        answer = await said(agent, chat, "ada", ("schedule_task", args))
        assert "every 2 minutes" in answer and "*/2" not in answer, answer


async def test_list_schedules_says_why_a_schedule_is_paused() -> None:
    """Asked "why is it paused?", the model reads the reason: its runs failed in a row, its
    one run failed, or the user paused it; resumed, there is none."""
    chat = Failing()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "sync", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        for _ in range(MAX_FAILURES):
            chat.fired.appendleft("x")
            await fire(h, agent, sid)
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert f'"paused_because": "its runs failed {MAX_FAILURES} times in a row"' in listed
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": True}))
        assert "paused_because" not in await said(agent, chat, "ada", ("list_schedules", {}))
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": False}))
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"paused_because": "the user paused it"' in listed, listed

        once = {"task": "remind me", "when": "0 9 4 11 *", "once": True}
        await said(agent, chat, "ada", ("schedule_task", once))
        [one] = [s for s in local(h)._schedules.values() if s.schedule_id != sid]
        chat.fired.appendleft("x")
        await fire(h, agent, one.schedule_id)
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"paused_because": "its one run failed"' in listed, listed


async def test_how_a_condition_reports_can_be_changed_later() -> None:
    """ "Tell me every time instead": update_schedule takes report, as schedule_task does."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "stock of A-1", "when": "0 8 * * *", "condition": "under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        change = {"id": sid, "report": "every_time"}
        await said(agent, chat, "ada", ("update_schedule", change))
        assert local(h)._schedules[sid].metadata[REPORT] == "every_time"
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"reports": "every_time"' in listed, listed
        bad = await said(agent, chat, "ada", ("update_schedule", {"id": sid, "report": "never"}))
        assert "until_clears" in bad and local(h)._schedules[sid].metadata[REPORT] == "every_time"


async def test_a_schedules_runs_are_held_to_the_run_limits_of_the_settings_and_the_hook() -> None:
    """Each fired run at most 10 minutes and 30 tool calls (handoff phase 3), as its schedule
    keeps them; an agent's own tighter limit stays; the per-user hook sets a user's own."""
    chat = Chat()

    async def limits(user: str, tenant: str) -> ScheduleLimits | None:
        return ScheduleLimits(run_max_tool_calls=10) if user == "bo" else None

    async with Harness(config=Settings(), schedule_limits=limits) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"}))
        await said(agent, chat, "bo", ("schedule_task", {"task": "b", "when": "0 8 * * *"}))
        made = {s.on_behalf_of: s for s in local(h)._schedules.values()}
        assert made["ada"].timeout_seconds == 600
        assert made["ada"].metadata[BUDGET]["max_tool_calls"] == 30
        assert made["bo"].metadata[BUDGET]["max_tool_calls"] == 10
        tight = h.wrap(chat, id="tight", run_timeout=60, budget=Budget(max_tool_calls=4))
        await said(tight, chat, "ada", ("schedule_task", {"task": "c", "when": "0 9 * * *"}))
        [own] = [s for s in local(h)._schedules.values() if s.agent_id == "tight"]
        assert own.timeout_seconds == 60 and own.metadata[BUDGET]["max_tool_calls"] == 4


async def test_failures_before_a_pause_and_days_before_asking_are_settings() -> None:
    chat = Failing()
    async with Harness(
        config=Settings(schedule_pause_after_failures=2, schedule_expire_days=30)
    ) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "sync", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        chat.fired.appendleft("x")
        await fire(h, agent, sid)
        chat.fired.appendleft("x")
        last = await fire(h, agent, sid)
        assert local(h)._schedules[sid].enabled is False
        assert last.error is not None and "failed 2 times in a row" in last.error.message
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "enabled": True}))
        old = (datetime.now(UTC) - timedelta(days=31)).isoformat()
        await local(h).schedules.update(sid, ScheduleUpdate(metadata={TOUCHED: old}))
        chat.fired.append("sales: 12. Still need it?")
        await fire(h, agent, sid)
        assert "still need" in chat.inputs[-1] and local(h)._schedules[sid].enabled is False


async def test_a_schedule_ends_on_the_day_the_user_said_in_their_zone() -> None:
    """ "Every day at 9 until 31 December": end_at is the user's last day, in their zone, kept
    as the instant that day ends; its runs that day still happen, none after. "Extend it to
    March" moves it, "remove the end date" clears it; a past day is said back."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {
            "task": "stock of A-1",
            "when": "0 9 * * *",
            "timezone": "Asia/Kolkata",
            "end_at": "2026-12-31",
        }
        made = await said(agent, chat, "ada", ("schedule_task", args))
        schedule = only(h)
        assert schedule.end_at == datetime(2026, 12, 31, 18, 30, tzinfo=UTC), (
            made
        )  # 1 Jan 00:00 IST
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert '"ends": "after Thu 31 Dec 2026 (Asia/Kolkata)"' in listed, listed
        sid = schedule.schedule_id
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "end_at": "2027-03-31"}))
        assert local(h)._schedules[sid].end_at == datetime(2027, 3, 31, 18, 30, tzinfo=UTC)
        await said(agent, chat, "ada", ("update_schedule", {"id": sid, "end_at": ""}))
        assert local(h)._schedules[sid].end_at is None
        assert "ends" not in await said(agent, chat, "ada", ("list_schedules", {}))
        past = {**args, "task": "old", "end_at": "2020-01-01"}
        answer = await said(agent, chat, "ada", ("schedule_task", past))
        assert "already past" in answer and len(local(h)._schedules) == 1


async def test_a_schedule_past_its_end_fires_no_more() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "a", "when": "0 9 * * *", "end_at": "2030-01-01"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        gone = datetime.now(UTC) - timedelta(minutes=1)
        await local(h).schedules.update(sid, ScheduleUpdate(end_at=gone))
        schedule = local(h)._schedules[sid]
        assert schedule.next_fire_at is None
        with pytest.raises(ConflictError, match="ended"):
            await local(h).schedules.fire(sid)


@pytest.mark.parametrize(
    ("raised", "plain"),
    [
        (
            ScheduleLimitError(
                "CONFLICT: the tenant has 50 schedules",
                code="SCHEDULE_LIMIT",
                status=409,
                details={"max_per_tenant": 50, "schedule_count": 50},
            ),
            "Your organisation has 50 of 50 schedules, the most it allows",
        ),
        (
            CadenceTooFrequentError(
                "VALIDATION: cadence too frequent",
                code="CADENCE_TOO_FREQUENT",
                status=422,
                details={"min_interval_seconds": 3600, "interval_seconds": 1800},
            ),
            "That runs every 30 minutes, more often than every 1 hour",
        ),
    ],
)
async def test_a_limit_agent_runs_holds_is_said_in_plain_words_with_its_numbers(
    raised: Exception, plain: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service's own limits (``RUNS__SCHEDULES__*``) reach the user as the harness's do:
    which limit, its number, what to do; never an error code."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")

        async def refused(spec: ScheduleSpec) -> None:
            raise raised

        monkeypatch.setattr(local(h).schedules, "create", refused)
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"})
        )
        assert plain in answer and "SCHEDULE_LIMIT" not in answer, answer
        assert "CADENCE_TOO_FREQUENT" not in answer and "VALIDATION" not in answer


async def test_a_fire_refused_for_the_days_limit_is_listed_in_plain_words() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        spent = AgentError(
            code="FIRES_PER_DAY",
            category=ErrorCategory.RATE_LIMIT,
            message="FIRES_PER_DAY: 500 of 500",
            details={"max_fires_per_day": 500, "fires_today": 500, "day": "2026-11-20"},
        )
        local(h)._schedules[sid] = (
            local(h)
            ._schedules[sid]
            .model_copy(update={"consecutive_failures": 1, "last_error": spent})
        )
        listed = await said(agent, chat, "ada", ("list_schedules", {}))
        assert "500 scheduled runs" in listed and "FIRES_PER_DAY" not in listed, listed


async def test_a_keep_question_the_model_did_not_ask_leaves_the_schedule_on_to_ask_again() -> None:
    """A run told to ask that answered only NO_REPORT asked nothing: not delivered, and the
    schedule is not left paused unsaid; its next run asks again."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "sales", "when": "0 8 * * *", "condition": "sales under 5"}
        await said(agent, chat, "ada", ("schedule_task", args))
        sid = only(h).schedule_id
        old = (datetime.now(UTC) - STALE - timedelta(days=1)).isoformat()
        await local(h).schedules.update(sid, ScheduleUpdate(metadata={TOUCHED: old}))
        chat.fired.append(NO_REPORT)
        await fire(h, agent, sid)
        schedule = local(h)._schedules[sid]
        assert schedule.enabled is True and not schedule.metadata.get(PAUSED_BY)
        chat.fired.append("Nothing under 5. Still need it?")
        await fire(h, agent, sid)
        assert "still need" in chat.inputs[-1] and local(h)._schedules[sid].enabled is False


class Recorded:
    """A channel that keeps what it was given."""

    name = "recorded"

    def __init__(self) -> None:
        self.got: list[Notification] = []

    async def deliver(self, notification: Notification, harness: Harness) -> None:
        self.got.append(notification)


async def test_a_fired_runs_outcome_is_told_on_the_users_channels_once() -> None:
    """Point 31: a fired run's result, its failure and the pause it caused go to the user's
    channels (``Harness(channels=, user_channels=)``) once each; a quiet run tells nothing."""
    chat, recorded = Failing(), Recorded()

    async def routed(user: str, tenant: str, notification: Notification) -> list[str]:
        return [CHAT, "recorded"]

    async with Harness(
        config=Settings(schedule_pause_after_failures=2),
        channels=[ChatChannel(), recorded],
        user_channels=routed,
    ) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "sync", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        chat.fired.extend(["synced 4", NO_REPORT, "x", "x"])
        for _ in range(4):
            await fire(h, agent, sid)
        assert [n.kind for n in recorded.got] == [Kind.RESULT, Kind.FAILURE, Kind.PAUSED]
        result, failure, paused = recorded.got
        assert "synced 4" in result.text and result.schedule_id == sid
        assert "upstream down" in failure.text and "paused" in paused.text
        assert all(n.user == "ada" and n.thread for n in recorded.got)
        assert len({n.key for n in recorded.got}) == 3


async def test_with_no_zone_known_a_schedule_says_utc_never_silently() -> None:
    """Point 14 (handoff 2.2): no app zone and none named, the reply says the zone it used, and
    the model is told it is not the user's, so it says so or asks."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        made = await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 9 * * *"}))
        assert "(UTC)" in made and only(h).timezone == "UTC"
        assert "(UTC; the user's zone is not known)" in chat.described[-1]


async def test_wait_then_continue_fires_once_in_the_chat_and_ends_itself() -> None:
    """Point 27: "wait 20 minutes, then continue" is one run in the same conversation, the
    schedule gone after it; a repeat of its fire answers the same run, never a second."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "continue the report", "in_minutes": 20}
        await said(agent, chat, "ada", ("schedule_task", args))
        schedule = only(h)
        assert schedule.metadata[TASK]["once"] is True
        due = schedule.next_fire_at
        assert due is not None
        first = await local(h).schedules.fire(schedule.schedule_id, at=due)
        again = await local(h).schedules.fire(schedule.schedule_id, at=due)
        assert again.run_id == first.run_id  # the same tick: one run
        chat.fired.append("continued")
        assert await h.worker([agent]).run_once() is True
        record = await h.runs.get(first.run_id)
        assert record is not None and record.status is RunStatus.SUCCESS
        assert record.thread_id == THREAD and not local(h)._schedules


async def test_agent_runs_own_floor_is_the_least_when_it_is_the_longer_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortest repeat in force is the longer of the harness's and agent-runs' own
    (``GET /v1/schedules/policy``), read once a minute: never a schedule the harness takes and
    the service refuses."""
    chat, read = Chat(), []
    async with Harness(config=Settings(schedule_min_interval=900)) as h:
        agent = h.wrap(chat, id="desk")

        async def policy(*, tenant: str | None = None) -> Any:
            read.append(tenant)
            return SimpleNamespace(min_interval_seconds=7200)

        monkeypatch.setattr(local(h).schedules, "policy", policy, raising=False)
        hourly = {"task": "a", "when": "0 * * * *"}
        answer = await said(agent, chat, "ada", ("schedule_task", hourly))
        assert "more often than every 2 hours" in answer, answer
        answer = await said(agent, chat, "ada", ("schedule_task", {**hourly, "task": "b"}))
        assert "more often than every 2 hours" in answer and len(read) == 1  # read once a minute


def test_in_minutes_on_a_whole_minute_keeps_it() -> None:
    at = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    assert schedules._in_minutes(20, "UTC", at)[0] == "20 9 20 11 *"
    assert schedules._in_minutes(1.5, "UTC", at)[0] == "2 9 20 11 *"  # 09:01:30: on to 09:02


def test_an_end_is_a_day_or_an_exact_time_and_anything_else_is_said_back() -> None:
    zone = "Asia/Kolkata"
    exact = schedules._ending("2030-12-31T17:00", zone)
    assert isinstance(exact, datetime) and exact == datetime(2030, 12, 31, 11, 30, tzinfo=UTC)
    assert schedules._ends_said(exact, zone) == "at Tue 31 Dec 2030, 17:00 (Asia/Kolkata)"
    assert "is not a day" in str(schedules._ending("next spring", zone))
    assert schedules._ending("", zone) is None


def test_a_limit_without_its_numbers_is_said_as_the_service_said_it() -> None:
    bare = ScheduleLimitError("CONFLICT: at the limit", code="SCHEDULE_LIMIT", status=409)
    assert schedules._limit_said(bare) is None
    assert schedules._refusal(bare, "Refused") == "Refused: CONFLICT: at the limit"


async def test_a_bad_end_in_a_change_is_said_back_and_changes_nothing() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"}))
        sid = only(h).schedule_id
        answer = await said(agent, chat, "ada", ("update_schedule", {"id": sid, "end_at": "soon"}))
        assert "is not a day" in answer and only(h).end_at is None


async def test_a_policy_agent_runs_cannot_say_now_leaves_the_harness_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """agent-runs' policy unread (down, a key without the right): the harness's own floor holds,
    and the service's own refusal still applies."""
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")

        async def down(*, tenant: str | None = None) -> Any:
            raise RuntimeError("agent-runs unreachable")

        monkeypatch.setattr(local(h).schedules, "policy", down, raising=False)
        made = await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"}))
        assert "Scheduled" in made and h.schedule_floors == {}


async def test_a_channel_that_fails_never_fails_the_fired_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        await said(agent, chat, "ada", ("schedule_task", {"task": "a", "when": "0 8 * * *"}))

        async def broken(notification: Notification) -> list[str]:
            raise RuntimeError("the notification service is down")

        monkeypatch.setattr(h, "notify", broken)
        chat.fired.append("done")
        record = await fire(h, agent, only(h).schedule_id)
        assert record.status is RunStatus.SUCCESS and record.output == "done"


async def test_a_time_the_model_assumed_is_confirmed_once_and_may_be_changed() -> None:
    """Item 7: no time said, the model uses the default and says so (``inferred``); the user
    confirms in one question offering that time (recommended), a time picker, or their words.
    A time they pick is the schedule's."""
    async with Harness(config=Settings()) as h:
        args = {"task": "sales summary", "when": "0 9 * * 1", "inferred": ["time"]}
        agent = await planned(h, ("schedule_task", args))
        interrupt = await asked(agent)
        assert interrupt.options[0].value == CONFIRM
        assert interrupt.options[0].recommended is True
        form = (interrupt.expects or {}).get("anyOf", [{}])[-1].get("properties", {})
        assert form["time"]["default"] == "09:00" and form["time"]["format"] == "time"
        await picked_value(agent, interrupt, {"time": "07:30"})
        assert only(h).cadence == "30 7 * * 1"


async def test_everything_open_is_asked_in_one_question_never_two() -> None:
    """Item 8: an assumed zone, a schedule the model judged alike and a write its runs make are
    one question, with every open detail in it; confirmed with the recommended answers, the
    alike one is updated, the write allowed, and an unapproved step skipped and told."""
    async with Harness(config=Settings()) as h:
        [had] = await seeded(h, "desk", {"task": "Send the sales summary", "when": "0 8 * * 1"})
        args = {
            "task": "send the sales summary",
            "when": "0 9 * * 1",
            "alike": had,
            "tools": ["send_report"],
            "inferred": ["zone"],
        }
        agent = await planned(h, ("schedule_task", args), tools=[send_report])
        interrupt = await asked(agent)
        form = (interrupt.expects or {}).get("anyOf", [{}])[-1].get("properties", {})
        assert {"alike", "allow", "unapproved"} <= set(form) and "time" not in form
        assert "send_report" in interrupt.question and "already have" in interrupt.question
        result = await picked(agent, interrupt, CONFIRM)
        assert "Scheduled" in result or "Updated" in result or "sch_" in result, result
        [schedule] = local(h)._schedules.values()  # the alike one, updated: not a second
        assert schedule.schedule_id == had and schedule.cadence == "0 9 * * 1"
        assert schedule.metadata[ALLOWED] == ["send_report"]
        assert schedule.metadata[UNAPPROVED] == "skip"


async def in_app(agent: Any, text: str, zone: str = "Asia/Kolkata") -> Any:
    """A turn as a front end sends it: with the user's app, which says their zone."""
    record = await agent._opened(
        text, user="ada", thread=None, tenant=None, app=Frontend(timezone=zone)
    )
    return await pipeline.attempt(agent, record, text)


@pytest.mark.parametrize(
    ("asked_for", "when"),
    [
        ("Every day at 9am send me yesterday's sales summary.", "0 9 * * *"),
        ("Check the stock every morning at 7:30 and tell me.", "30 7 * * *"),
        ("On weekdays at 6 pm give me the day's revenue.", "0 18 * * 1-5"),
        ("Weekly on Friday at 17:00, send me the top SKUs.", "0 17 * * 5"),
        ("At noon every day, the sales summary.", "0 12 * * *"),
    ],
)
async def test_a_time_the_user_said_is_not_one_the_model_assumed(asked_for: str, when: str) -> None:
    """A model marks the time as assumed when the user said it ("every day at 9am", marked
    ``inferred: ["time"]``), and the zone when it is the user's own app's: the schedule then
    waited for a confirmation of what was said in full — fifty of the schedule benchmark's
    hundred and forty chats, on every adapter. The claim is read against the user's own words
    and their app's zone: said, it is not asked again."""
    async with Harness(config=Settings()) as h:
        args = {"task": "the report", "when": when, "inferred": ["time", "zone"]}
        agent = await planned(h, ("schedule_task", args))
        result = await in_app(agent, asked_for)
        assert result.status is RunStatus.SUCCESS, result.interrupt
        assert only(h).cadence == when and only(h).timezone == "Asia/Kolkata"


async def test_a_time_or_zone_the_user_did_not_say_is_still_confirmed() -> None:
    """Another time than the one said, a time with no morning or evening to it, and a zone no
    app gives are still the model's own."""
    for asked_for, when in (
        ("Every day at 9am send me the summary.", "0 21 * * *"),  # not the time said
        ("Every day at 9 send me the summary.", "0 9 * * *"),  # 9 in the morning, or at night
    ):
        async with Harness(config=Settings()) as h:
            args = {"task": "the report", "when": when, "inferred": ["time"]}
            agent = await planned(h, ("schedule_task", args))
            done = await in_app(agent, asked_for)
            assert done.status is RunStatus.PAUSED, asked_for
    async with Harness(config=Settings()) as h:
        args = {"task": "the report", "when": "0 9 * * *", "inferred": ["zone"]}
        agent = await planned(h, ("schedule_task", args))
        done = await agent.run("Every day at 9am send me the summary.", user="ada")
        assert done.status is RunStatus.PAUSED  # no app says the user's zone


async def test_a_schedule_said_in_full_is_made_with_no_question() -> None:
    chat = Chat()
    async with Harness(config=Settings()) as h:
        agent = h.wrap(chat, id="desk")
        args = {"task": "sales summary", "when": "0 9 * * 1", "timezone": "Asia/Kolkata"}
        made = await said(agent, chat, "ada", ("schedule_task", args))
        assert made.startswith("Scheduled") and len(local(h)._schedules) == 1


async def test_a_confirmation_approved_makes_the_schedule_and_rejected_makes_none() -> None:
    """The confirmation answered with a plain approve (an approvals client, a reviewer, the
    benchmark's turn that approves what it asks) is its recommended answer, Confirm; a reject
    is Cancel. An approve read as the user's own words "true" made nothing and told the model
    the user had answered "true"."""
    async with Harness(config=Settings()) as h:
        args = {"task": "sales summary", "when": "0 9 * * 1", "inferred": ["time"]}
        agent = await planned(h, ("schedule_task", args))
        interrupt = await asked(agent)
        done = await agent.resume(interrupt.interrupt_id, "approve", reviewer="ada")
        assert done.status is RunStatus.SUCCESS, done.error
        assert only(h).cadence == "0 9 * * 1"
        other = await planned(h, ("schedule_task", args), name="other")
        interrupt = await asked(other)
        done = await other.resume(interrupt.interrupt_id, "reject", reviewer="ada")
        assert "Nothing scheduled" in str(done.answer) and len(local(h)._schedules) == 1


async def test_a_confirmation_cancelled_or_answered_in_words_makes_nothing() -> None:
    async with Harness(config=Settings()) as h:
        args = {"task": "sales summary", "when": "0 9 * * 1", "inferred": ["time"]}
        agent = await planned(h, ("schedule_task", args))
        assert "Nothing scheduled" in await picked(agent, await asked(agent), NEITHER)
        other = await planned(h, ("schedule_task", args), name="other")
        said_ = await picked_value(other, await asked(other), "make it Tuesdays at 7 pm")
        assert "Tuesdays at 7 pm" in said_ and not local(h)._schedules
