"""Scenario 13 — long-running work (LR-01..LR-05, BG-05, BG-07).

* A turn that works for minutes: its hold in agent-runs is renewed and never lapses while its
  process lives, the user chats in another chat meanwhile, and the result is delivered. One case
  runs at real length with the product's own hold (60 s, renewed every 20 s): 200 s. The other
  compresses time with a short hold in a server process of the test's own (6 s).
* A 200-step task: every step runs once, the run's spending counts them all, a step limit and a
  budget stop it where they say, and a small window keeps every request fitted.
* A queued long task: its worker's short lease (6 s) is extended by its heartbeats and progress,
  never lapsing while the worker lives; a worker stopped gracefully hands the run back and
  another finishes it; a worker killed mid-task leaves it to the next, which finishes it without
  running a finished write again; a deadline ends it ``TIMEOUT`` even while it waits queued.
* A Claude subagent working in the background when its server is killed: its run ends
  (``holder_lost``), never left ``RUNNING``.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tests.scenarios.brains import SYSTEM, BrainChat, command, parse, token, wrapped
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS, Served, eventually, free_port, record
from tests.support.arm import arm
from trellis import Budget, Harness, ReAct
from trellis.contracts import RunEventType, RunStatus

pytestmark = [pytest.mark.timeout(1200), pytest.mark.usefixtures("writes_unasked")]


async def _watch(h: Harness, run_id: str, seen: list[str], stop: asyncio.Event) -> None:
    while not stop.is_set():
        got = await h.runs.get(run_id)
        if got is not None:
            seen.append(got.status.value)
        await asyncio.sleep(2)


@pytest.mark.plan("LR-01")
async def test_a_turn_working_for_minutes_at_real_length(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    """Real length: 200 s with the product's own hold (60 s, renewed every 20 s)."""
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s13r-{uid}")
    user = f"ada-{uid}"
    long = asyncio.create_task(
        agent.run(
            f"{token(uid, 1)} {command('wait', seconds=200)}", user=user, thread=f"long-{uid}"
        )
    )
    await asyncio.sleep(3)
    [summary] = [r async for r in h.runs.iterate(agent_id=agent.id)]
    seen: list[str] = []
    stop = asyncio.Event()
    watching = asyncio.create_task(_watch(h, summary.run_id, seen, stop))
    for n in range(3):  # the user chats in another chat meanwhile
        other = await agent.run(
            f"{token('o' + uid[:4], n)} hello", user=user, thread=f"other-{uid}"
        )
        assert other.status is RunStatus.SUCCESS, other.error
        await asyncio.sleep(30)
    done = await long
    stop.set()
    await watching
    assert done.status is RunStatus.SUCCESS and parse(done.answer).mine == token(uid, 1), done.error
    assert set(seen[:-1]) == {"RUNNING"}, seen  # never lapsed while it worked


@pytest.mark.plan("LR-01")
@pytest.mark.parametrize(
    "adapter", ["react", "langgraph", "openai_agents", "claude_agent_sdk", "kimi", "function"]
)
async def test_a_turn_working_well_past_its_hold(adapter: str, uid: str, tmp_path: Path) -> None:
    """Compressed: a 6 s hold (renewed every 2 s) and a turn of 40 s, over AG-UI."""
    agent_id = f"s13h-{adapter}-{uid}"
    claude = [{"tool": "wait", "args": {"seconds": 40}}]
    port = free_port()
    served = await Served(
        {
            "mode": "agui",
            "port": port,
            "hold": 6,
            "agents": [{"adapter": adapter, "id": agent_id, "claude": claude}],
        },
        tmp_path,
        "srv",
    ).started()
    run_id = f"run_{uid}long"
    try:
        async with (
            harness() as h,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", headers={"x-user": f"ada-{uid}"}, timeout=300
            ) as c,
        ):
            body = {
                "threadId": f"t-{uid}",
                "runId": run_id,
                "messages": [
                    {
                        "id": "m1",
                        "role": "user",
                        "content": f"{token(uid, 1)} {command('wait', seconds=40)}",
                    }
                ],
            }
            posting = asyncio.create_task(c.post(f"/agui/{agent_id}/run", json=body))
            seen: list[str] = []
            stop = asyncio.Event()
            await asyncio.sleep(2)
            watching = asyncio.create_task(_watch(h, run_id, seen, stop))
            chat = await c.post(
                f"/agui/{agent_id}/run",
                json={
                    "threadId": f"o-{uid}",
                    "messages": [
                        {"id": "m2", "role": "user", "content": f"{token('o' + uid[:4], 1)} hi"}
                    ],
                },
            )
            from trellis.harness.agui.sse import decode

            assert decode(chat.text)[-1][1].get("outcome") == {
                "type": "success"
            }  # chatting meanwhile
            sent = await posting
            stop.set()
            await watching
            finished = decode(sent.text)[-1][1]
            assert finished.get("outcome") == {"type": "success"}, finished
            assert "ERROR" not in seen and seen.count("RUNNING") >= 10, seen
    finally:
        served.stop()


@pytest.mark.plan("LR-02")
@pytest.mark.parametrize("adapter", ["react", "react_fitted", "middleware"])
async def test_a_two_hundred_step_task(h: Harness, adapter: str, uid: str, tmp_path: Path) -> None:
    from langchain.agents import create_agent

    from tests.scenarios.brains import BrainChatModel
    from trellis import middleware
    from trellis.harness.middleware import RunCheckpointer

    if adapter == "middleware":
        target = create_agent(
            BrainChatModel(),
            system_prompt=SYSTEM,
            middleware=middleware(max_steps=230),
            checkpointer=RunCheckpointer(),
        )
    else:
        target = ReAct(
            SYSTEM,
            BrainChat(),
            max_steps=230,
            context_window=8_000 if adapter == "react_fitted" else None,
        )
    agent = arm(h.wrap(target, id=f"s13s-{adapter}-{uid}"), TOOLS)
    done = await agent.run(f"{token(uid, 1)} loop:200:lookup", user=f"ada-{uid}", thread=f"t-{uid}")
    assert done.status is RunStatus.SUCCESS, (done.status, done.error)
    assert (
        parse(done.answer).mine == token(uid, 1) and parse(done.answer).last == "value-of-k199"
    ), done.answer[-200:]
    events = [e async for e in agent.events(done.run_id)]
    starts = [e for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert len(starts) == 200, len(starts)
    spent = [e.data["spent"] for e in events if e.type is RunEventType.RUN_FINISHED][-1]
    assert int(spent["tool_calls"]) == 200, spent
    if adapter == "react_fitted":
        biggest = max(len(json.dumps(r["messages"])) for r in target.model.requests)  # type: ignore[union-attr]
        assert biggest / 4 <= 8_000 * 1.25, biggest


@pytest.mark.plan("LR-03")
async def test_a_budget_stops_a_long_task_where_it_says(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s13b-{uid}")
    done = await agent.run(
        f"{token(uid, 1)} loop:60:lookup",
        user=f"ada-{uid}",
        thread=f"t-{uid}",
        budget=Budget(max_tool_calls=25),
    )
    assert done.status is RunStatus.PARTIAL, (done.status, done.error)
    events = [e async for e in agent.events(done.run_id)]
    assert len([e for e in events if e.type is RunEventType.TOOL_CALL_START]) <= 25


@pytest.mark.plan("LR-04", "LR-05")
@pytest.mark.parametrize("how", ["graceful", "killed"])
@pytest.mark.parametrize(
    "adapter", ["react", "langgraph", "openai_agents", "claude_agent_sdk", "function"]
)
async def test_a_queued_long_task_across_workers(
    adapter: str, how: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s13w-{adapter}-{uid}"
    steps = [
        command("note", text=f"first {uid}"),
        *[command("wait", seconds=1.0)] * 30,
        command("note", text=f"last {uid}"),
    ]
    claude = [
        {"tool": "note", "args": {"text": f"first {uid}"}},
        *[{"tool": "wait", "args": {"seconds": 1.0}}] * 30,
        {"tool": "note", "args": {"text": f"last {uid}"}},
    ]
    spec = {
        "mode": "worker",
        "agents": [{"adapter": adapter, "id": agent_id, "claude": claude}],
        "lease": 6,
        "hold": 6,
    }
    async with harness() as h:
        agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=agent_id, claude=claude)
        handle = await agent.start(
            f"{token(uid, 1)} {' '.join(steps)}", user=f"ada-{uid}", thread=f"t-{uid}"
        )
        one = await Served(spec, tmp_path, "w1").started()

        async def begun() -> bool:
            return book.exists() and f"note:first {uid}" in book.read_text()

        assert await eventually(begun, within=180), one.log.read_text()[-3000:]
        await asyncio.sleep(12)  # two leases' worth: extended by heartbeats and progress
        assert (await record(h, handle.id)).attempt == 1  # never lapsed while the worker lived
        if how == "graceful":
            one.stop()  # SIGTERM: stop claiming, hand the run back
        else:
            one.kill()
        two = await Served(spec, tmp_path, "w2").started()
        try:
            done = await handle.result(wait=400)
        finally:
            two.stop()
            one.stop()
        assert done.status is RunStatus.SUCCESS, (
            how,
            done.status,
            done.error,
            two.log.read_text()[-3000:],
        )
        lines = book.read_text().splitlines()
        assert lines.count(f"note:first {uid}") == 1, ("a finished write ran again", lines)
        assert lines.count(f"note:last {uid}") == 1, lines


@pytest.mark.plan("LR-03")
async def test_a_deadline_ends_a_queued_run_and_a_working_one(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s13d-{uid}")
    soon = datetime.now(UTC) + timedelta(seconds=8)
    queued = await agent.start(
        f"{token(uid, 1)} later", user=f"ada-{uid}", thread=f"q-{uid}", deadline=soon
    )

    async def ended() -> bool:  # agent-runs ends it while it waits in the queue (no worker)
        return (await record(h, queued.id)).status is RunStatus.TIMEOUT

    assert await eventually(ended, within=60)
    began = time.monotonic()
    working = await agent.run(
        f"{token(uid, 2)} {command('wait', seconds=120)}",
        user=f"ada-{uid}",
        thread=f"w-{uid}",
        deadline=datetime.now(UTC) + timedelta(seconds=6),
    )
    assert working.status is RunStatus.TIMEOUT, (working.status, working.error)
    assert time.monotonic() - began < 60


@pytest.mark.plan("BG-05", "BG-07", "LR-05")
async def test_a_background_subagent_when_its_server_is_killed(
    uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Claude subagent's run ends once its server is gone — never left ``RUNNING``; the
    conversation goes on on another server."""
    agent_id = f"s13g-{uid}"
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    background = [{"background": {"id": "t1", "agent": "researcher", "prompt": "research"}}]
    port = free_port()
    spec = {
        "mode": "agui",
        "port": port,
        "hold": 6,
        "agents": [{"adapter": "claude_agent_sdk", "id": agent_id, "claude": background}],
    }
    monkeypatch.setenv("FAKE_CLAUDE_TASKS", str(tasks))  # the server's CLI reads its tasks here
    one = await Served(spec, tmp_path, "one").started()
    from trellis.harness.agui.sse import decode

    async with (
        harness() as h,
        httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", headers={"x-user": f"ada-{uid}"}, timeout=300
        ) as c,
    ):
        first = await c.post(
            f"/agui/{agent_id}/run",
            json={
                "threadId": f"t-{uid}",
                "messages": [
                    {
                        "id": "m1",
                        "role": "user",
                        "content": f"{token(uid, 1)} research in the background",
                    }
                ],
            },
        )
        finished = decode(first.text)[-1][1]
        assert finished.get("outcome") == {"type": "success"}, finished
        run_id = decode(first.text)[0][1]["runId"]
        [child] = [r async for r in h.runs.iterate(parent_run_id=run_id)]
        assert (await record(h, child.run_id)).status is RunStatus.RUNNING
        one.kill()

        async def ended() -> bool:
            return (await record(h, child.run_id)).status is not RunStatus.RUNNING

        assert await eventually(ended, within=60)
        ending = await record(h, child.run_id)
        assert (
            ending.status is RunStatus.ERROR
            and ending.error is not None
            and ending.error.code == "holder_lost"
        ), ending
