"""How the work a turn started that outlives its attempt — a Claude subagent working in the
background of its live session (``bridges.live``), a Deep Agents async task on its server
(``bridges.async_tasks``), a harness background task at work in this process (``background``)
— stops, or goes on, when the turn is stopped in a way that is not ``agent.cancel``:

* **interrupted from another process** (``Harness._renewed`` → ``Agent.stop``): the turn ends
  ``CANCELLED`` here, and its kept work stops with it;
* **cancelled, or ended, on another server** (the run store ended the turn first; its hold
  here is then refused: ``Harness._lost``): the turn writes nothing — an in-process store
  takes a second ``CANCELLED``, and refuses one over another ending — and its kept work stops
  all the same;
* **its caller went away** (its stream closed, its task cancelled: no ``Runtime.cancelled``):
  what it handed on past its attempt goes on, and the background task it started ends.

Each framework is its test's stand-in (the scripted Claude CLI, the stand-in Agent Protocol
server, a scripted model): no model is called.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from tests.integration import test_claude_background_session as claude
from tests.integration import test_react_async_subagent as remote
from tests.integration import test_react_background as local
from trellis import Harness
from trellis.contracts import AgentError, RunStatus
from trellis.harness import admission
from trellis.harness.agent import Agent
from trellis.harness.result import Result

#: How a turn's attempt is stopped here, given its harness, its agent and its run.
Stop = Callable[[Harness, Agent, str], Awaitable[None]]


async def interrupted(harness: Harness, agent: Agent, run_id: str) -> None:
    """A newer message of the conversation, sent to another process with ``mode="interrupt"``,
    asks for the turn: seen here when the run's hold is next renewed."""

    async def asked(asked_of: str) -> bool:
        return asked_of == run_id

    agent.turns.interrupted = asked  # type: ignore[method-assign]
    await harness._renewed(run_id, "default")


async def cancelled_elsewhere(harness: Harness, agent: Agent, run_id: str) -> None:
    """The run store ended the turn first (a cancel on another server), and its hold here is
    then refused."""
    await harness.runs.cancel(run_id, reason="stopped on another server")
    await harness._lost(run_id, "default")


async def ended_elsewhere(harness: Harness, agent: Agent, run_id: str) -> None:
    """The run store ended the turn first with an ending of its own (its sweep: the hold
    lapsed), and its hold here is then refused: the turn's own ending is refused too."""
    lapsed = AgentError(code="HOLDER_LOST", message="its holder stopped heartbeating")
    await harness.runs.finish(run_id, RunStatus.ERROR, error=lapsed)
    await harness._lost(run_id, "default")


STOPS: dict[str, Stop] = {
    "interrupted": interrupted,
    "cancelled elsewhere": cancelled_elsewhere,
    "ended elsewhere": ended_elsewhere,
}


@pytest.fixture
def server() -> Iterator[remote.StandIn]:
    found = remote.StandIn()
    found.start()
    yield found
    found.stop()


async def background_subagent(
    harness: Harness, tmp_path: Path, id: str
) -> tuple[Agent, asyncio.Task[Result], str, str]:
    """A Claude turn under way whose subagent works in the background of its live session:
    the agent, the turn, its run and the subagent's."""
    scripts = [[claude.BACKGROUND, {"stall": 2}, {"text": "never said"}], [{"text": "after"}]]
    agent = harness.wrap(claude.options(tmp_path, scripts), id=id)
    turn = asyncio.create_task(agent.run("research the market", user="u1", thread="t1"))
    await claude.until(lambda: len(agent.running) == 2)
    live = claude.live_of(agent)
    assert live is not None
    await claude.until(lambda: all(found.turn is not None for found in live.background.values()))
    started = claude.turn_running(agent)
    child = await claude.child_of(harness, started)
    return agent, turn, started, child.run_id


@pytest.mark.host_execution
@pytest.mark.parametrize("how", list(STOPS))
async def test_a_turn_stopped_without_agent_cancel_stops_its_live_background_subagent(
    harness: Harness, tmp_path: Path, how: str
) -> None:
    """The subagent the stopped turn started ends ``CANCELLED`` and its task is stopped in the
    session — which is kept, and answers the conversation's next turn."""
    agent, turn, started, child = await background_subagent(harness, tmp_path, "kept-stops")
    await STOPS[how](harness, agent, started)
    assert (await turn).status is RunStatus.CANCELLED
    await claude.when(harness, child, RunStatus.CANCELLED)
    await claude.until((tmp_path / "tasks" / "t1.stopped").exists)
    after = await agent.run("and now?", user="u1", thread="t1")
    assert after.status is RunStatus.SUCCESS and after.answer == "after"


@pytest.mark.parametrize("how", list(STOPS))
async def test_a_turn_stopped_without_agent_cancel_stops_the_async_task_it_started(
    harness: Harness, server: remote.StandIn, how: str
) -> None:
    """The Deep Agents async task the stopped turn started is cancelled on its server, and its
    run ends ``CANCELLED``, held no more."""
    model = remote.Model(turns=[remote.START, "never said"], hold_from=1)
    agent = remote.chat(harness, server, model)
    turn = asyncio.create_task(remote.say(agent, "run the optimization"))
    await remote.stops_answering(model)  # the start returned; the model's next call hangs
    [run_id] = agent.running
    await STOPS[how](harness, agent, run_id)
    assert (await turn).status is RunStatus.CANCELLED
    await claude.until(lambda: server.status(server.task_id) == "cancelled")
    [task] = await remote.task_runs(harness)
    assert task.parent_run_id == run_id and task.status is RunStatus.CANCELLED
    assert not harness.holds.holds(task.run_id)


@pytest.mark.host_execution
async def test_a_turn_whose_caller_went_away_leaves_its_live_background_subagent_working(
    harness: Harness, tmp_path: Path
) -> None:
    """Nobody cancelled the turn (its caller stopped waiting for it): the subagent it started
    in the background was handed on past its attempt, and goes on in the session."""
    agent, turn, started, child = await background_subagent(harness, tmp_path, "kept-goes-on")
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)
    await claude.when(harness, started, RunStatus.CANCELLED)
    live = claude.live_of(agent)
    assert live is not None and list(live.working) == ["t1"]
    found = await harness.runs.get(child, tenant="default")
    assert found is not None and found.status is RunStatus.RUNNING
    assert not (tmp_path / "tasks" / "t1.stopped").exists()


async def test_a_turn_whose_caller_went_away_ends_the_background_task_it_started(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody cancelled the turn (its caller stopped waiting for it): the harness background
    task it started is no helper handed on — it ends ``CANCELLED`` with the turn, and its
    attempt here stops."""
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission())  # reads no machine
    gate = local.Gate()
    model = remote.Model(turns=[local.START, "never said"], hold_from=1)
    agent = local.chat(harness, gate, model)
    turn = asyncio.create_task(local.chatted(agent, "research lighthouses"))
    await local.until(lambda: gate.started)  # the task works; the turn's next model call hangs
    [run_id] = agent.running
    task = await local.task_run(harness, run_id)
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)
    await claude.when(harness, run_id, RunStatus.CANCELLED)
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED
    await local.until(lambda: not harness.background.at_work(task.run_id))
