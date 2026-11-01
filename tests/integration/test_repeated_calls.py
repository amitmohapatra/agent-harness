"""A model looping on one call, whatever its framework: the harness's tool bridge, where every
adapter's calls meet, does not run the same call (the same tool, the same arguments) a third
time in a row and tells the model to answer with what it has (``bridge.STALLED``) (ADV-9).
``ReAct`` and a graph with ``middleware()`` stop their loop earlier, in ``StallGuard``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.planned import FINAL
from trellis import Harness, Runtime, tool
from trellis.contracts import RunEvent, RunEventType
from trellis.harness.tools.bridge import MAX_REPEATS, STALLED

LOOPS = 8
#: the adapters whose loop only the bridge stops (``ReAct`` and ``middleware()``: StallGuard)
BRIDGED = sorted(set(BUILDERS) - {"react", "middleware"})


def lookup(ran: list[str]) -> Any:
    @tool(side_effects="read")
    def look(topic: str) -> str:
        """Facts about a topic."""
        ran.append(topic)
        return f"facts about {topic}"

    return look


@pytest.mark.parametrize("framework", BRIDGED)
async def test_the_same_call_made_over_and_over_runs_twice(
    harness: Harness, tmp_path: Path, framework: str
) -> None:
    ran: list[str] = []
    plan = [("look", {"topic": "tides"})] * LOOPS
    target, tools = await BUILDERS[framework](harness, [lookup(ran)], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"loop-{framework}"), tools)
    events = [e async for e in agent.stream("tides?", user="u")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert ran == ["tides"] * (MAX_REPEATS - 1)
    assert finished.data["result"], "the run ended with no answer"
    assert _warned(events).count("stalled") == 1, _warned(events)


async def test_openai_agents_answers_on_the_call_after_the_stall(
    harness: Harness, tmp_path: Path
) -> None:
    """OpenAI Agents offers its tools per turn (the SDK's ``FunctionTool.is_enabled``): the
    call after the stall is offered none, so the model's reply to ``STALLED`` is the run's
    answer — not the SDK's ``max_turns`` (10) seven calls later, answered about the limit
    (ADV-9)."""
    from agents.run_config import DEFAULT_MAX_TURNS

    ran: list[str] = []
    seen: list[Any] = []
    plan = [("look", {"topic": "tides"})] * (DEFAULT_MAX_TURNS * 3)
    target, tools = await BUILDERS["openai_agents"](
        harness, [lookup(ran)], tmp_path, plan, seen=seen
    )
    agent = arm(harness.wrap(target, id="loop-oa"), tools)
    events = [e async for e in agent.stream("tides?", user="u")]
    finished, model = events[-1], seen[0]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert ran == ["tides"] * (MAX_REPEATS - 1)
    assert [bool(offered) for offered in model.tools] == [True] * MAX_REPEATS + [False]
    stalled = STALLED.format(tool="look", times=MAX_REPEATS)
    assert finished.data["result"] == FINAL.format(last=stalled)
    assert _warned(events) == ["stalled"]


def _warned(events: list[RunEvent]) -> list[str]:
    """The codes of the run's warnings, in order."""
    return [
        e.data["code"]
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "warning"
    ]


async def test_a_different_call_between_two_starts_the_count_again(harness: Harness) -> None:
    ran: list[str] = []

    async def caller(input: str, agent: Runtime) -> list[Any]:
        topics = ["a", "a", "b", "a", "a", "a", "a"]
        return [await agent.tools.call("look", topic=t) for t in topics]

    agent = arm(harness.wrap(caller, id="calls"), [lookup(ran)])
    said = (await agent.run("x", user="u")).answer
    assert ran == ["a", "a", "b", "a", "a"]
    assert said[-2:] == [STALLED.format(tool="look", times=n) for n in (3, 4)]
