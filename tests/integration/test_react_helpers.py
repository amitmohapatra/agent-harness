"""``ReAct``'s helpers (``adaptive.Helpers``: Deep Agents' ``task`` tool): offered on a task of
several parts only; each ``task`` call a child run of the harness — its tools governed by the
catalogue's rules, its spending capped by its parent's budget, continued after a crash — at
most ``HELPERS_AT_ONCE`` at once, only its answer coming back."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.support.catalog import FakeCatalog
from tests.support.models import ScriptedChat
from trellis import Budget, Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunRecord, RunStatus
from trellis.harness import adaptive
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

#: A request of three parts: the helpers' signal.
THREE = "Do three things:\n1. look up a\n2. look up b\n3. look up c\nThen combine them."
WINDOW = 32_000
done: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


@tool(side_effects="read")
def lookup(topic: str) -> str:
    """Look a topic up.

    Args:
        topic: The topic.
    """
    done.append(topic)
    return f"facts about {topic}"


def task(description: str, kind: str = adaptive.GENERAL) -> tuple[str, dict[str, Any]]:
    return ("task", {"description": description, "subagent_type": kind})


def offered(model: ScriptedChat, n: int) -> dict[str, str]:
    return {t["function"]["name"]: t["function"]["description"] for t in model.requests[n]["tools"]}


async def children(h: Harness, run_id: str) -> list[RunRecord]:
    found = [s async for s in h.runs.iterate(parent_run_id=run_id)]
    return [r for s in found if (r := await h.runs.get(s.run_id)) is not None]


# --------------------------------------------------------------------------- the signal
@pytest.mark.parametrize(
    ("asked", "window", "without", "wanted"),
    [
        (THREE, WINDOW, set(), True),
        ("look up a", WINDOW, set(), False),  # one part
        (THREE, 8_192, set(), False),  # no room
        (THREE, WINDOW, {"subagents"}, False),
    ],
)
async def test_task_is_offered_on_a_task_of_several_parts_only(
    harness: Harness, asked: str, window: int, without: set[str], wanted: bool
) -> None:
    model = ScriptedChat(["fine"])
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=window),
        id="signal",
        tools=[lookup],
        without=without,  # type: ignore[arg-type]
    )
    await agent.run(asked, user="u")
    assert ("task" in offered(model, 0)) is wanted


async def test_a_task_call_without_the_feature_is_refused(harness: Harness) -> None:
    model = ScriptedChat([task("look up a"), "done"])
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW),
        id="off",
        tools=[lookup],
        without={"subagents"},
    )
    assert (await agent.run(THREE, user="u")).answer == "done"
    assert model.requests[1]["messages"][-1]["content"] == "task is not available in this run"


# --------------------------------------------------------------------------- a helper's run
async def test_the_general_helper_is_a_child_run_and_only_its_answer_returns(
    harness: Harness,
) -> None:
    model = ScriptedChat([task("look up a"), ("lookup", {"topic": "a"}), "a: facts", "all done"])
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW), id="lead", tools=[lookup]
    )
    result = await agent.run(THREE, user="u")
    assert result.answer == "all done", result.error
    assert done == ["a"]
    [kid] = await children(harness, result.run_id)
    assert kid.agent_id == "lead-helper" and kid.parent_run_id == result.run_id
    back = model.requests[3]["messages"][-1]
    assert back["content"] == "a: facts"  # the helper's answer, not its transcript
    assert "facts about a" not in json.dumps(model.requests[3]["messages"])


async def test_the_runs_agents_are_reached_through_task_only(harness: Harness) -> None:
    async def research(input: str, agent: Runtime) -> str:
        """Research a topic."""
        return f"researched {input}"

    child = harness.wrap(research, id="research")
    model = ScriptedChat([task("topic a", "research"), "combined"])
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW), id="lead2", tools=[lookup, child]
    )
    result = await agent.run(THREE, user="u")
    assert result.answer == "combined", result.error
    tools = offered(model, 0)
    assert "research" not in tools and "- research: Research a topic." in tools["task"]
    assert model.requests[1]["messages"][-1]["content"] == "researched topic a"
    [kid] = await children(harness, result.run_id)
    assert kid.agent_id == "research"


async def test_helpers_of_one_step_work_at_once(harness: Harness) -> None:
    started: list[str] = []
    both = asyncio.Event()

    async def research(input: str, agent: Runtime) -> str:
        """Research a topic."""
        started.append(input)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5)  # each waits for the other: only at once
        return f"researched {input}"

    child = harness.wrap(research, id="research")
    model = ScriptedChat([[task("a", "research"), task("b", "research")], "combined"])
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW), id="lead3", tools=[child]
    )
    result = await agent.run(THREE, user="u")
    assert result.answer == "combined", result.error
    assert sorted(started) == ["a", "b"]


async def test_a_helpers_tool_is_held_by_the_catalogues_rule() -> None:
    governance = Governance(FakeCatalog({"lookup": Rule(risk="irreversible")}))
    async with Harness(config=Settings(), governance=governance) as h:
        model = ScriptedChat(
            [task("look up a"), ("lookup", {"topic": "a"}), "could not look", "said so"]
        )
        agent = h.wrap(
            ReAct(system="s", model=model, context_window=WINDOW), id="ruled", tools=[lookup]
        )
        paused = await agent.run(THREE, user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert paused.interrupt.payload["subagent"]["agent_id"] == "ruled-helper"
        result = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="r")
        assert result.answer == "said so", result.error
    assert done == []  # rejected: never run


async def test_a_parents_budget_caps_its_helpers(harness: Harness) -> None:
    model = ScriptedChat(
        [task("look up a"), ("lookup", {"topic": "a"}), ("lookup", {"topic": "b"}), "x", "y"]
    )
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW),
        id="capped",
        tools=[lookup],
        budget=Budget(max_tokens=40),
    )
    result = await agent.run(THREE, user="u")
    # the parent's step (15 tokens) and the helper's first (15): the helper's second call is
    # refused, spent; the parent ends PARTIAL
    assert result.status is RunStatus.PARTIAL, result.error
    assert done == ["a"]


class Crash(BaseException):
    """The worker dies."""


async def test_a_worker_killed_inside_a_helper_continues_it(harness: Harness) -> None:
    crashes = [Crash()]

    @tool(side_effects="write")
    def file(note: str) -> str:
        """File a note.

        Args:
            note: The note.
        """
        done.append(f"file {note}")
        return f"filed {note}"

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    model = ScriptedChat(
        [task("file n1"), ("file", {"note": "n1"}), ("boom", {}), "filed n1", "done"]
    )
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=WINDOW), id="clerk", tools=[file, boom]
    )
    store = harness.runs
    assert isinstance(store, LocalRuns)
    handle = await agent.start(THREE, user="u")
    worker = harness.worker([agent])
    claimed = await store.claim(worker.worker_id, [agent.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.run_id]
    store._leases[handle.run_id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await harness.worker([agent]).run_once()
    result = await handle.result(timeout=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    assert done == ["file n1"]  # filed once, before the crash
