"""SCH against the real agent-runs and its ticker, through the harness: a schedule a user sets
in chat is fired by the ticker into that user's conversation and executed by a worker.

* cron: ``schedule_task`` makes a schedule acting for the run's user, in its thread; a fire
  queues a run carrying the schedule's options, a worker executes it, its answer lands in the
  same conversation;
* missed fires: a schedule whose fire time passed hours ago (an outage) fires once, not once
  per missed tick; overlap: no new run while the last one is still queued; jitter: a fire
  lands within the minute after its tick;
* once: a ``once`` schedule is gone after its run; condition: a run whose condition does not
  hold answers ``NO_REPORT`` and is not delivered to the conversation;
* the breaker: three failed runs in a row pause the schedule, and the third run's error says
  so; expiry: a schedule untouched for 90 days asks "Still need this?" before it works;
* a user who left: an ``on_run_start`` hook ends the fired run before any tool runs.

A fire is triggered with ``POST /v1/schedules/{id}/fire`` or by moving the schedule's next fire
time back in the database (the ticker then fires it); the run is executed by the harness's own
worker (``h.worker``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.features.conftest import (
    RUNS_DSN,
    eventually,
    harness,
    memory_sql,
    move_clock,
    runs_http,
    runs_sql,
)
from tests.features.kit import approve_all, armed, ledger, lookup, record
from tests.support.arm import arm
from trellis import Hooks, Runtime
from trellis.contracts import RunStatus
from trellis.harness import schedules as chat_schedules

pytestmark = pytest.mark.timeout(300)


async def schedule_in_chat(tmp: Path, agent_id: str, user: str, thread: str, **args: Any) -> dict:
    """The user asks in chat; the agent's ``schedule_task`` makes it. The schedule, as kept."""
    async with harness() as h:
        plan = [("schedule_task", args)]
        agent, _ = await armed("react", h, tmp, plan, id=agent_id)
        done = await approve_all(
            agent, await agent.run("Schedule it, please.", user=user, thread=thread),
            reviewer=user,
        )
        assert done.status is RunStatus.SUCCESS, done
    rows = runs_sql(
        "select * from agent_schedules where agent_id = :1 and on_behalf_of = :2",
        agent_id,
        user,  # a bare id is a user's (agent-runs' principals)
    )
    assert len(rows) == 1, rows
    return rows[0]


async def fire(schedule_id: str) -> str:
    async with runs_http() as c:
        got = await c.post(f"/v1/schedules/{schedule_id}/fire", json={})
        assert got.status_code == 200, got.text
        return got.json()["run_id"]


async def work(tmp: Path, agent_id: str, plan: list[Any], *, hooks: Any = (),
               target: Any = None) -> None:
    """One worker claims and executes the agent's queued runs until none is left."""
    async with harness() as h:
        if target is None:
            agent, _ = await armed("react", h, tmp, plan, id=agent_id, hooks=hooks)
        else:
            agent = arm(h.wrap(target, id=agent_id, hooks=hooks), [lookup])
        worker = h.worker([agent], concurrency=1)
        while await worker.run_once():
            pass
        await h.writes.drain()


@pytest.mark.plan("SCH", "F95", "F06")
async def test_a_schedule_set_in_chat_fires_into_the_users_conversation(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user, thread = f"sch-{uid}", f"una-{uid}", "chat-1"
    row = await schedule_in_chat(tmp_path, agent_id, user, thread,
                                 task="check stock of A-1", when="0 9 * * *",
                                 timezone="Europe/Lisbon")
    assert row["cadence"] == "0 9 * * *" and row["timezone"] == "Europe/Lisbon"
    assert row["enabled"] is True
    meta = row["schedule_metadata"]
    assert meta[chat_schedules.TASK]["task"] == "check stock of A-1"
    next_fire = datetime.fromisoformat(row["next_fire_at"])
    local = next_fire.astimezone(__import__("zoneinfo").ZoneInfo("Europe/Lisbon"))
    assert (local.hour, local.minute) == (9, 0) and 0 <= local.second < 60  # jitter: in the minute
    run_id = await fire(row["schedule_id"])
    queued = await record(run_id)
    assert queued.status is RunStatus.QUEUED
    assert queued.on_behalf_of == user and queued.metadata["schedule_id"] == row["schedule_id"]
    async def check(input: Any, agent: Runtime) -> str:
        # a function: a planned model would count the chat's earlier turn (schedule_task's
        # call, replayed by memory's window) as one of its own steps
        return str(await agent.tools.call("lookup", sku="A-1"))

    await work(tmp_path, agent_id, [], target=check)
    done = await record(run_id)
    assert done.status is RunStatus.SUCCESS, done
    async with harness() as h:
        conversation = await h.conversation(user=user, thread=thread)
    assert done.thread_id == thread  # the same chat (agent-runs keeps the caller's thread name)
    assert "lookup:A-1" in ledger()

    async def delivered() -> list[dict]:
        return memory_sql(
            "select role, content from messages where agent_run_id = :1 and role = 'ASSISTANT'",
            run_id,
        )

    rows = await eventually(delivered, within=60)
    assert rows and "stock of A-1 is 3" in rows[-1]["content"], rows
    where = memory_sql("select distinct thread_id from messages where agent_run_id = :1", run_id)
    assert [w["thread_id"] for w in where] == [conversation]  # in the user's conversation
    # the fired run was offered none of the schedule tools (it does its task)
    async with runs_http() as c:
        events = (await c.get(f"/v1/runs/{run_id}/events", params={"limit": 500})).json()
    assert not any("schedule_task" in json.dumps(e["event"].get("data", {})) for e in events
                   if e["event"]["type"] == "TOOL_CALL_START")
    async with runs_http() as c:
        await c.delete(f"/v1/schedules/{row['schedule_id']}")


@pytest.mark.plan("SCH")
async def test_missed_fires_fire_once_and_an_overlapping_fire_is_skipped(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user = f"miss-{uid}", f"val-{uid}"
    row = await schedule_in_chat(tmp_path, agent_id, user, "c", task="daily report",
                                 when="0 9 * * *")
    sid = row["schedule_id"]
    # an outage: the fire was due three days ago
    move_clock(RUNS_DSN, "update agent_schedules set next_fire_at = now() - interval '3 days'"
               " where schedule_id = :1", sid)

    async def fired() -> list[dict]:
        return runs_sql("select run_id, status from agent_runs where agent_id = :1", agent_id + "")

    async def runs_of_schedule() -> list[dict]:
        rows = runs_sql(
            "select run_id, status from agent_runs where run_metadata->>'schedule_id' = :1", sid
        )
        return rows

    first = await eventually(runs_of_schedule, within=90)
    assert len(first) == 1, first  # once, not once per missed day
    after = runs_sql("select next_fire_at from agent_schedules where schedule_id = :1", sid)[0]
    assert datetime.fromisoformat(after["next_fire_at"]) > datetime.now(UTC)
    # the run still queued, the next fire due again: skipped, nothing queued beside it
    move_clock(RUNS_DSN, "update agent_schedules set next_fire_at = now() - interval '1 minute'"
               " where schedule_id = :1", sid)
    import asyncio

    await asyncio.sleep(15)  # three ticks
    assert len(await runs_of_schedule()) == 1, await runs_of_schedule()
    await work(tmp_path, agent_id, [])
    assert (await record(first[0]["run_id"])).status is RunStatus.SUCCESS
    async with runs_http() as c:
        await c.delete(f"/v1/schedules/{sid}")


@pytest.mark.plan("SCH")
async def test_a_once_schedule_is_gone_after_its_run_and_a_condition_that_fails_is_quiet(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user = f"once-{uid}", f"wes-{uid}"
    once = await schedule_in_chat(tmp_path, agent_id, user, "c1", task="remind me to call",
                                  in_minutes=120, once=True)
    run_id = await fire(once["schedule_id"])
    await work(tmp_path, agent_id, [])
    assert (await record(run_id)).status is RunStatus.SUCCESS
    assert not runs_sql("select 1 from agent_schedules where schedule_id = :1",
                        once["schedule_id"])  # gone

    watch = await schedule_in_chat(tmp_path, f"{agent_id}-w", user, "c2",
                                   task="tell me when A-1 stock is under 2", when="0 8 * * *",
                                   condition="stock of A-1 under 2")
    quiet_id = await fire(watch["schedule_id"])

    async def nothing(input: Any, agent: Runtime) -> str:
        await agent.tools.call("lookup", sku="A-1")
        return chat_schedules.NO_REPORT  # 3 is not under 2

    await work(tmp_path, f"{agent_id}-w", [], target=nothing)
    quiet = await record(quiet_id)
    assert quiet.status is RunStatus.SUCCESS and quiet.output == chat_schedules.NO_REPORT
    await __import__("asyncio").sleep(5)
    rows = memory_sql("select role from messages where agent_run_id = :1", quiet_id)
    assert not [r for r in rows if r["role"] == "ASSISTANT"], rows  # not delivered
    async with runs_http() as c:
        await c.delete(f"/v1/schedules/{watch['schedule_id']}")


class Broken(Hooks):
    async def on_run_start(self, run: Runtime) -> None:
        raise RuntimeError("the warehouse system is down")


@pytest.mark.plan("SCH", "RUN-04")
async def test_three_failed_runs_in_a_row_pause_the_schedule_and_say_so(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user = f"brk-{uid}", f"xia-{uid}"
    row = await schedule_in_chat(tmp_path, agent_id, user, "c", task="sync stock",
                                 when="0 7 * * *")
    sid, runs = row["schedule_id"], []
    for _ in range(3):
        runs.append(await fire(sid))
        await work(tmp_path, agent_id, [], hooks=[Broken()])
    ended = [await record(r) for r in runs]
    assert [e.status for e in ended] == [RunStatus.ERROR] * 3, ended
    assert "paused" in ended[-1].error.message and "sync stock" in ended[-1].error.message
    assert "paused" not in ended[0].error.message
    kept = runs_sql("select enabled, schedule_metadata from agent_schedules where schedule_id = :1",
                    sid)[0]
    assert kept["enabled"] is False
    assert kept["schedule_metadata"][chat_schedules.FAILURES] == 3
    async with runs_http() as c:
        refused = await c.post(f"/v1/schedules/{sid}/fire", json={})
        assert refused.status_code == 409  # a paused schedule does not fire
        await c.delete(f"/v1/schedules/{sid}")


@pytest.mark.plan("SCH")
async def test_a_schedule_untouched_for_90_days_asks_still_need_this(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user = f"old-{uid}", f"yan-{uid}"
    row = await schedule_in_chat(tmp_path, agent_id, user, "c", task="weekly audit",
                                 when="0 6 * * 1")
    sid = row["schedule_id"]
    old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    async with runs_http() as c:
        patched = await c.patch(f"/v1/schedules/{sid}",
                                json={"metadata": {chat_schedules.TOUCHED: old}})
        assert patched.status_code == 200, patched.text
    run_id = await fire(sid)
    await work(tmp_path, agent_id, [("lookup", {"sku": "A-1"})])
    paused = await record(run_id)
    assert paused.status is RunStatus.PAUSED, paused
    assert paused.awaiting and "Still need this?" in paused.awaiting.question
    assert "lookup:A-1" not in ledger()  # nothing done before the answer
    assert runs_sql("select enabled from agent_schedules where schedule_id = :1",
                    sid)[0]["enabled"] is False
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, [], id=agent_id)
        await agent.resume(paused.awaiting.interrupt_id, "answer", answer="none", reviewer=user)
    await work(tmp_path, agent_id, [])
    assert (await record(run_id)).status is RunStatus.SUCCESS
    assert not runs_sql("select 1 from agent_schedules where schedule_id = :1", sid)  # cancelled


class Left(Hooks):
    async def on_run_start(self, run: Runtime) -> None:
        if run.scheduled is not None:
            raise PermissionError(f"{run.user} has left: their schedules stop")


@pytest.mark.plan("SCH")
async def test_a_fired_run_of_a_user_who_left_ends_before_any_tool(
    tmp_path: Path, uid: str
) -> None:
    agent_id, user = f"left-{uid}", f"zed-{uid}"
    row = await schedule_in_chat(tmp_path, agent_id, user, "c", task="check A-1", when="0 5 * * *")
    run_id = await fire(row["schedule_id"])
    await work(tmp_path, agent_id, [("lookup", {"sku": "A-1"})], hooks=[Left()])
    ended = await record(run_id)
    assert ended.status is RunStatus.ERROR and "has left" in ended.error.message
    assert "lookup:A-1" not in ledger()
    async with runs_http() as c:
        await c.delete(f"/v1/schedules/{row['schedule_id']}")


@pytest.mark.plan("SCH")
@pytest.mark.skip(
    reason="DST: the next fire across a daylight-saving change is computed from the service's "
    "clock, which an end-to-end test cannot move; proven at unit level by agent-runs "
    "tests/test_cadence.py (DST forward and back) and harness unit tests of schedule_task"
)
async def test_dst() -> None:  # pragma: no cover
    ...


@pytest.mark.plan("SCH", "RUN-12")
async def test_schedules_are_upserted_listed_patched_and_deleted(tmp_path: Path, uid: str) -> None:
    """``agent.schedule`` from code: the same schedule again is the one that exists (a
    redeploy), its run options travel to every fired run, and the routes list, patch and
    delete it for its tenant."""
    agent_id = f"code-{uid}"
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, [], id=agent_id, run_timeout=120)
        first = await agent.schedule("0 4 * * *", user=f"amy-{uid}", input="nightly",
                                     priority=7, concurrency_key=f"k-{uid}",
                                     without=["judges"])
        again = await agent.schedule("0 4 * * *", user=f"amy-{uid}", input="nightly",
                                     priority=7, concurrency_key=f"k-{uid}",
                                     without=["judges"])
        assert again.schedule_id == first.schedule_id
    async with runs_http() as c:
        listed = await c.get("/v1/schedules", params={"agent_id": agent_id})
        assert [s["schedule_id"] for s in listed.json()] == [first.schedule_id]
        one = await c.get(f"/v1/schedules/{first.schedule_id}")
        assert one.json()["priority"] == 7
        run_id = await fire(first.schedule_id)
        rec = await record(run_id)
        assert (rec.priority, rec.concurrency_key, rec.timeout_seconds) == (7, f"k-{uid}", 120)
        assert rec.metadata["without"] == ["judges"]
        await c.post(f"/v1/runs/{run_id}/cancel", json={"reason": "test over"})
        off = await c.patch(f"/v1/schedules/{first.schedule_id}", json={"enabled": False})
        assert off.json()["enabled"] is False
        assert (await c.delete(f"/v1/schedules/{first.schedule_id}")).status_code == 204
        assert (await c.get(f"/v1/schedules/{first.schedule_id}")).status_code == 404
