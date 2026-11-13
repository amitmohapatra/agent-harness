"""The children an attempt has open when it ends out of time or cancelled — a sub-agent's child
run working inside one of its calls, a helper a framework spawned in the foreground — end with
it, as it ends: ``TIMEOUT`` with its error, ``CANCELLED`` with its reason; none is left
``RUNNING`` with nothing behind it. A Claude subagent working in the background of its
conversation's session outlives the turn that started it, and keeps doing so."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from tests.support.arm import arm
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunRecord, RunStatus
from trellis.harness.runs import LocalRuns

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
#: a subagent in the foreground whose tool call never ends
FOREGROUND = {
    "agent": "researcher",
    "prompt": "research",
    "steps": [{"tool": "look_up", "args": {}}, {"text": "found"}],
}
BACKGROUND = {"background": {"id": "t1", "agent": "researcher", "prompt": "research"}}


@tool(side_effects="read")
async def look_up() -> str:
    """Look something up (it takes for ever)."""
    await asyncio.Event().wait()
    return "never"


async def forever(input: str, agent: Runtime) -> str:
    await asyncio.Event().wait()
    return "never"


def claude(folder: Path, *scripts: list[dict[str, Any]]) -> ClaudeAgentOptions:
    (folder / "tasks").mkdir(exist_ok=True)
    env = {
        "FAKE_CLAUDE_TURNS": json.dumps(list(scripts)),
        "FAKE_CLAUDE_SESSIONS": str(folder),
        "FAKE_CLAUDE_TASKS": str(folder / "tasks"),
        "FAKE_CLAUDE_RECORD": str(folder / "cli.json"),
    }
    return ClaudeAgentOptions(cli_path=CLI, env=env)


async def children(store: LocalRuns, run_id: str) -> list[RunRecord]:
    found = [c async for c in store.iterate(parent_run_id=run_id)]
    return [r for r in [await store.get(c.run_id) for c in found] if r is not None]


async def until(check: Any, seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not await check():  # noqa: ASYNC110 - the run store's state
            await asyncio.sleep(0.02)


async def test_a_sub_agents_run_ends_out_of_time_with_the_run_whose_call_it_works_in() -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        worker = h.wrap(forever, id="worker")

        async def boss(input: str, agent: Runtime) -> str:
            return str(await worker.run("dig", user="u1"))

        ended = await h.wrap(boss, id="boss", run_timeout=0.5).run("go", user="u1")
        assert ended.status is RunStatus.TIMEOUT and ended.error is not None
        [child] = await children(store, ended.run_id)
    assert child.status is RunStatus.TIMEOUT and child.error is not None
    assert child.error.code == ended.error.code


async def test_a_foreground_helper_ends_out_of_time_with_its_run(tmp_path: Path) -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        agent = arm(h.wrap(claude(tmp_path, [FOREGROUND]), id="helped", run_timeout=1.5), [look_up])
        ended = await agent.run("research", user="u1")
        assert ended.status is RunStatus.TIMEOUT and ended.error is not None
        [helper] = await children(store, ended.run_id)
    assert helper.status is RunStatus.TIMEOUT and helper.error is not None
    assert helper.error.code == ended.error.code


async def test_a_foreground_helper_and_a_sub_agents_run_are_cancelled_with_their_run(
    tmp_path: Path,
) -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        worker = h.wrap(forever, id="worker")

        async def boss(input: str, agent: Runtime) -> str:
            return str(await worker.run("dig", user="u1"))

        helped = arm(h.wrap(claude(tmp_path, [FOREGROUND]), id="helped"), [look_up])
        for agent in (h.wrap(boss, id="boss"), helped):
            running = asyncio.create_task(agent.run("go", user="u1"))

            async def started() -> bool:
                for run_id in list(agent.running):  # noqa: B023 - read at once
                    if any(c.status is RunStatus.RUNNING for c in await children(store, run_id)):
                        return True
                return False

            await until(started)
            [parent] = list(agent.running)
            await agent.cancel(parent, reason="enough")
            ended = await running
            assert ended.status is RunStatus.CANCELLED
            [child] = await children(store, ended.run_id)
            assert child.status is RunStatus.CANCELLED, (agent.id, child)


async def test_a_worker_stopping_its_attempt_ends_its_open_sub_agent_run_as_cancelled() -> None:
    """A run a worker was asked to stop (its heartbeat says ``cancel_requested``) is cancelled
    there, not by ``agent.cancel`` in this process: its open children end with it too."""
    from trellis.runs import Job

    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        worker = h.wrap(forever, id="worker")

        async def boss(input: str, agent: Runtime) -> str:
            return str(await worker.run("dig", user="u1"))

        agent = h.wrap(boss, id="boss")
        handle = await agent.start("go", user="u1")
        claimed = await store.claim("w1", [agent.id], lease_seconds=30)
        assert claimed is not None
        job = Job(record=claimed.run, worker_id="w1", lease_seconds=30, store=store)
        attempt = asyncio.create_task(agent.execute(job))

        async def started() -> bool:
            return len(await children(store, handle.id)) == 1

        await until(started)
        await store.cancel(handle.id, reason="stop it")  # asked of its worker
        attempt.cancel()  # what the worker does on its next heartbeat
        ended = await asyncio.gather(attempt, return_exceptions=True)
        [child] = await children(store, handle.id)
    assert child.status is RunStatus.CANCELLED, (ended, child)


async def test_a_background_subagent_outlives_a_later_turn_that_runs_out_of_time(
    tmp_path: Path,
) -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        scripts = [[BACKGROUND, {"text": "started it"}], [{"stall": 3}, {"text": "late"}]]
        agent = h.wrap(claude(tmp_path, *scripts), id="bg")
        first = await agent.run("research", user="u1", thread="t1")
        assert first.status is RunStatus.SUCCESS
        [child] = await children(store, first.run_id)
        assert child.status is RunStatus.RUNNING
        second = await agent.run("and?", user="u1", thread="t1", run_timeout=1.5)
        assert second.status is RunStatus.TIMEOUT
        kept = await store.get(child.run_id)
        assert kept is not None and kept.status is RunStatus.RUNNING
        assert await children(store, second.run_id) == []
        (tmp_path / "tasks" / "t1.done").write_text("the market is growing")

        async def done() -> bool:
            record = await store.get(child.run_id)
            return record is not None and record.status is RunStatus.SUCCESS

        await until(done, seconds=30)


async def test_a_background_subagent_outlives_the_turn_that_started_it_running_out_of_time(
    tmp_path: Path,
) -> None:
    """The turn starts a subagent in the background, then runs out of time: the subagent's
    run is not the turn's to end (it is handed on to its session), and goes on to its end."""
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        scripts = [[BACKGROUND, {"stall": 3}, {"text": "late"}]]
        agent = h.wrap(claude(tmp_path, *scripts), id="bg")
        first = await agent.run("research", user="u1", thread="t1", run_timeout=1.5)
        assert first.status is RunStatus.TIMEOUT
        [child] = await children(store, first.run_id)
        assert child.status is RunStatus.RUNNING
        (tmp_path / "tasks" / "t1.done").write_text("the market is growing")

        async def done() -> bool:
            record = await store.get(child.run_id)
            return record is not None and record.status is RunStatus.SUCCESS

        await until(done, seconds=30)
