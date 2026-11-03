"""Schedules set in chat: ``schedule_task``, ``list_schedules``, ``update_schedule`` and
``cancel_schedule``, the built-in tools every adapter gets while the run store can schedule
(off: ``without=["schedules"]``), against the in-process run store. A fired run is the same
agent, user and thread; a run whose condition does not hold answers ``NO_REPORT`` and is not
delivered; a condition that still holds is reported once."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.support.memory import FakeMemoryService
from trellis import Harness, Runtime, Settings
from trellis.contracts import RunRecord, RunStatus
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import (
    NO_REPORT,
    REPORTED,
    SCHEDULE_TOOLS,
    TASK,
    ScheduleLimits,
)
from trellis.harness.tools.convert import text_of

THREAD = "chat-1"


class Chat:
    """A function target: in chat it calls the tools it is told to; a fired run answers what
    ``fired`` says next (``NO_REPORT`` or a report)."""

    def __init__(self) -> None:
        self.calls: deque[tuple[str, dict[str, Any]]] = deque()
        self.fired: deque[str] = deque()
        self.inputs: list[str] = []
        self.toolboxes: list[set[str]] = []
        self.said: list[str] = []

    async def __call__(self, input: str, agent: Runtime) -> str:
        self.inputs.append(input)
        self.toolboxes.append(set(agent.toolbox))
        if input.startswith("Scheduled task:"):
            return self.fired.popleft()
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


@pytest.mark.parametrize(
    ("when", "said_"),
    [
        ("every monday", "is not a cron expression"),
        ("0 8 * * 1 2026", "is not a cron expression"),
        ("*/2 * * * *", "more often than every 5 minutes"),
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
        for n in range(2):
            await said(
                agent, chat, "ada", ("schedule_task", {"task": f"t{n}", "when": "0 9 * * *"})
            )
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "t3", "when": "0 9 * * *"})
        )
        assert "You have 2 of 2 schedules" in answer
        answer = await said(
            agent, chat, "ada", ("schedule_task", {"task": "t", "when": "*/30 * * * *"})
        )
        assert "more often than every 60 minutes" in answer
        # another user has their own count
        await said(agent, chat, "bob", ("schedule_task", {"task": "t", "when": "0 9 * * *"}))

    seen: list[tuple[str, str]] = []

    async def plans(user: str, tenant: str) -> ScheduleLimits | None:
        seen.append((user, tenant))
        return ScheduleLimits(max_per_user=1) if user == "free" else None

    async with Harness(config=Settings(), schedule_limits=plans) as h:
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
        assert kept.cadence == "0 9 * * 1" and kept.enabled is False and "0 9 * * 1" in changed
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
        assert quiet.thread_id == THREAD and quiet.user_id == "ada"
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
