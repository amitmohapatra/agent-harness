"""The goal loop, opt-in: Deep Agents' ``RubricMiddleware`` given to ``ReAct(middleware=)`` — a
run whose input names a ``rubric`` has its answer graded by a model of its own and revised
until the grader is satisfied, within the run's step limit and budget; a run that names none
is sent what it always was."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.rubric import RubricMiddleware

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType

ASKED = [{"role": "user", "content": "how high is the mast?"}]


@tool(side_effects="read")
def mast() -> str:
    """The mast's height."""
    return "7"


def verdict(passed: bool) -> tuple[str, dict[str, Any]]:
    criterion = {"name": "Says the unit", "passed": passed, **({} if passed else {"gap": "m?"})}
    result = "satisfied" if passed else "needs_revision"
    return "GraderResponse", {"result": result, "explanation": "units", "criteria": [criterion]}


def graded(harness: Harness, turns: list[Any], verdicts: list[bool]) -> tuple[Any, Any, Any]:
    grader = ScriptedChatModel(turns=[verdict(v) for v in verdicts], spends=(100, 20))
    model = ScriptedChat(turns)
    graph = ReAct("You answer with units.", model, middleware=[RubricMiddleware(model=grader)])
    return arm(harness.wrap(graph, id="surveyor"), [mast]), model, grader


async def test_a_rubric_has_the_answer_graded_and_revised(harness: Harness) -> None:
    agent, model, grader = graded(harness, [("mast", {}), "7", "7 metres"], [False, True])
    given = {"messages": ASKED, "rubric": "Says the unit"}
    events = [e async for e in agent.stream(given, user="u")]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == "7 metres"
    assert len(grader.seen) == 2 and len(model.requests) == 3
    assert done.data["spent"]["model_calls"] == 5  # the grader's calls are the run's spend too


async def test_without_a_rubric_the_run_is_what_it_always_was(harness: Harness) -> None:
    agent, model, grader = graded(harness, [("mast", {}), "7"], [])
    result = await agent.run("how high is the mast?", user="u")
    assert result.answer == "7" and grader.seen == [] and len(model.requests) == 2
