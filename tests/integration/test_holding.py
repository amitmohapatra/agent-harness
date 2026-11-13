"""Every run the harness executes in its own process is held in the run store while its attempt
is under way (``holding``, from ``pipeline.attempt``): renewed however long the attempt takes;
let go of by a pause, and taken again by whichever server continues the run after the answer;
never taken for a run a worker claimed from the queue, which its lease keeps. What a killed
server leaves of them: ``test_killed_server``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

from tests.support.arm import arm
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness import holding
from trellis.harness.runs import LocalRuns
from trellis.runs import Job, Lease

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")


class Heard(LocalRuns):
    """The in-process store, hearing every heartbeat: (run, who)."""

    def __init__(self) -> None:
        super().__init__()
        self.beats: list[tuple[str, str]] = []

    async def heartbeat(self, run_id: str, worker_id: str, **given: Any) -> Lease:
        self.beats.append((run_id, worker_id))
        return await super().heartbeat(run_id, worker_id, **given)


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(holding, "HOLD_SECONDS", 1)  # renewed every third of a second


async def until(check: Any, seconds: float = 5.0) -> None:
    async with asyncio.timeout(seconds):
        while not check():  # noqa: ASYNC110 - the holds' own task
            await asyncio.sleep(0.02)


async def test_an_attempt_that_takes_long_keeps_its_run_held() -> None:
    """A long tool call, a slow model: the hold is renewed while the attempt is under way,
    whatever it does, so the store's sweeps leave the run however long it takes."""
    store = Heard()
    async with Harness(config=Settings(), runs=store) as h:

        async def slow(input: str, agent: Runtime) -> str:
            await asyncio.sleep(2.5)  # past two holds
            return "done"

        running = asyncio.create_task(h.wrap(slow, id="slow").run("go", user="u1"))
        await until(lambda: store.beats)
        [(run_id, _)] = store.beats
        for _ in range(2):
            await asyncio.sleep(1.2)
            assert await store.claim("w", ["other"]) is None  # a sweep: the hold is renewed
            record = await store.get(run_id)
            assert record is not None and record.status is RunStatus.RUNNING
        result = await running
        assert result.status is RunStatus.SUCCESS and result.answer == "done"
    mine = [who for held, who in store.beats if held == run_id]
    assert len(mine) >= 6 and set(mine) == {h.holds.name}


async def test_a_run_resumed_on_another_server_is_held_by_that_one() -> None:
    """A pause lets go of the hold (the run waits for a person, not for this server); the
    server the answer reaches continues the run and holds it under its own name."""
    store = Heard()
    go = asyncio.Event()

    async def deploy(input: str, agent: Runtime) -> str:
        region = await agent.ask("Which region?")
        await go.wait()
        return f"deployed to {region}"

    async with (
        Harness(config=Settings(), runs=store) as one,
        Harness(config=Settings(), runs=store) as two,
    ):
        paused = await one.wrap(deploy, id="deploy").run("deploy", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert not one.holds.holds(paused.run_id)
        store.beats.clear()
        resumed = asyncio.create_task(
            two.wrap(deploy, id="deploy").resume(
                paused.interrupt.interrupt_id, "answer", answer="eu", reviewer="u1"
            )
        )
        await until(lambda: (paused.run_id, two.holds.name) in store.beats)
        assert two.holds.holds(paused.run_id) and not one.holds.holds(paused.run_id)
        go.set()
        finished = await resumed
    assert finished.status is RunStatus.SUCCESS and finished.answer == "deployed to eu"
    assert {who for run_id, who in store.beats if run_id == paused.run_id} == {two.holds.name}


async def test_a_run_a_worker_claimed_is_kept_by_its_lease_not_held() -> None:
    store = Heard()
    async with Harness(config=Settings(), runs=store) as h:

        async def quick_job(input: str, agent: Runtime) -> str:
            await asyncio.sleep(0.5)  # a heartbeat's rhythm and more
            return "done"

        agent = h.wrap(quick_job, id="job")
        handle = await agent.start("go", user="u1")
        claimed = await store.claim("w1", [agent.id], lease_seconds=30)
        assert claimed is not None
        job = Job(record=claimed.run, worker_id="w1", lease_seconds=30, store=store)
        result = await agent.execute(job)
        assert result.status is RunStatus.SUCCESS and not h.holds.holds(handle.id)
    assert (handle.id, h.holds.name) not in store.beats


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


async def test_a_helper_whose_run_waits_for_a_person_is_let_go_of_not_left_to_lapse(
    tmp_path: Path,
) -> None:
    """A Claude subagent at work in the foreground whose call asks a person: the run pauses
    and the helper waits with it, ``RUNNING``. Its hold is let go of in the store (not left to
    lapse, which would end a helper that is only waiting), and the attempt after the answer
    takes it up again, holds it, and ends it."""
    store = Heard()
    asks = {
        "agent": "researcher",
        "prompt": "refund",
        "steps": [{"tool": "refund", "args": {"order": "o1"}}, {"text": "refunded"}],
    }
    (tmp_path / "tasks").mkdir()
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps([asks, {"text": "done"}]),
        "FAKE_CLAUDE_SESSIONS": str(tmp_path),
        "FAKE_CLAUDE_TASKS": str(tmp_path / "tasks"),
    }
    researcher = AgentDefinition(description="Researches.", prompt="Research it.")
    options = ClaudeAgentOptions(cli_path=CLI, env=env, agents={"researcher": researcher})
    async with Harness(config=Settings(), runs=store) as h:
        agent = arm(h.wrap(options, id="helped"), [refund])
        paused = await agent.run("refund o1", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        [helper] = [r async for r in store.iterate(parent_run_id=paused.run_id)]
        assert helper.status is RunStatus.RUNNING and not h.holds.holds(helper.run_id)
        await asyncio.sleep(holding.HOLD_SECONDS + 0.3)
        assert await store.claim("w", ["other"]) is None  # a sweep: nothing lapsed
        waiting = await store.get(helper.run_id)
        assert waiting is not None and waiting.status is RunStatus.RUNNING
        store.beats.clear()
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS
    assert (helper.run_id, h.holds.name) in store.beats  # taken up again, and held
    ended = await store.get(helper.run_id)
    assert ended is not None and ended.status is RunStatus.SUCCESS
