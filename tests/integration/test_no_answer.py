"""A run never ends ``SUCCESS`` with no answer, whatever its framework.

A model may fill its reply with working (rows summed by hand) and reach its output limit before
it answers: no text, no tool call. LangChain then ends its loop, and the run used to end
``SUCCESS`` with an empty answer (the break-and-fix gate: two LangGraph turns at 4,096 output
tokens). Where the framework has no way back of its own (LangChain's ``create_agent`` family,
``ReAct`` among them), the model is asked once more, told so (``budget.CUT_OFF``) and offered no
tool, as at the step limit. Kimi Code asks again itself. Anywhere else the run ends ``ERROR``
saying why (``budget.NO_ANSWER``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.planned import CUT, FINAL, cut_message
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEvent, RunEventType, RunStatus
from trellis.harness.budget import CUT_OFF, NO_ANSWER
from trellis.harness.middleware import CUT_STOPPED

#: What each framework does with a reply cut at the output limit: asked once more by the
#: harness, asked again by the framework itself, or ended ``ERROR`` saying why.
ENDS: Final = {
    "react": "asked",
    "langgraph": "asked",  # create_agent without middleware(): the harness's narrowing copy
    "middleware": "asked",
    "deepagents": "asked",
    "graph": "error",  # a hand-built graph binds its model itself: no hook to ask again in
    "openai_agents": "error",  # the SDK's own ModelBehaviorError on a real chat model
    "claude_agent_sdk": "error",  # past Claude Code's own recovery
    "kimi": "itself",
}


@tool(side_effects="read")
def rows(month: str) -> str:
    """The sales rows of a month."""
    return f"{month}: 2,061 rows"


@pytest.mark.parametrize("framework", sorted(ENDS))
async def test_a_reply_cut_at_the_output_limit_never_ends_a_run_silently(
    harness: Harness, tmp_path: Path, framework: str
) -> None:
    target, tools = await BUILDERS[framework](harness, [rows], tmp_path, [CUT])
    agent = arm(harness.wrap(target, id=f"cut-{framework}"), tools)
    events = [e async for e in agent.stream("total the rows", user="u")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED
    if ENDS[framework] == "error":
        assert finished.outcome is not None and finished.outcome.value == "error", finished
        assert finished.error is not None and finished.error.code == NO_ANSWER
        cut = framework != "openai_agents"  # its scripted model says no finish reason
        why = "reached its output limit" if cut else "had no text"
        assert why in finished.error.message, finished.error.message
        return
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert finished.data["result"] == FINAL.format(last="")
    asked = ENDS[framework] == "asked"
    assert _warned(events) == (["output_limit"] if asked else [])


async def test_react_is_told_its_reply_was_cut_and_offered_no_tool(harness: Harness) -> None:
    """The one more call is the cut call's request, told so, with no tool to call; its answer is
    the run's, said as an ``output_limit`` warning."""
    from tests.support.models import ScriptedChat
    from tests.support.planned import CUT_REPLY

    model = ScriptedChat([CUT_REPLY, "Knitwear made 31.2% in Q3."])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="told"), [rows])
    events = [e async for e in agent.stream("margin?", user="u")]
    assert events[-1].data["result"] == "Knitwear made 31.2% in Q3."
    first, last = model.requests
    assert last["messages"][:-1] == first["messages"]
    assert last["messages"][-1] == {"role": "user", "content": CUT_OFF}
    assert last.get("tool_choice") == "none"
    assert [e.data["message"] for e in events if e.data.get("code") == "output_limit"] == [
        CUT_STOPPED
    ]


async def test_a_reply_cut_again_ends_the_run_saying_so(harness: Harness) -> None:
    from langchain.agents import create_agent

    model = ScriptedChatModel(turns=[cut_message(), cut_message()])
    result = await harness.wrap(create_agent(model, tools=[]), id="twice").run("margin?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == CUT_STOPPED


async def test_an_empty_answer_with_no_cut_ends_the_run_saying_its_reply_had_no_text(
    harness: Harness,
) -> None:
    """A model that answers nothing, its reply not cut, has nothing asked of it again: its run
    ends ``ERROR``, not ``SUCCESS`` with an empty answer."""
    from langchain.agents import create_agent

    model = ScriptedChatModel(turns=["  "])
    result = await harness.wrap(create_agent(model, tools=[]), id="blank").run("margin?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == NO_ANSWER and "had no text" in result.error.message


async def test_a_function_that_returns_nothing_still_succeeds(harness: Harness) -> None:
    """A function target has no model of its own: what it returns, nothing included, is its
    answer."""

    async def work(input: Any, agent: Any) -> None:
        return None

    result = await harness.wrap(work, id="quiet").run("do it", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer is None


def _warned(events: list[RunEvent]) -> list[str]:
    """The codes of the run's warnings, in order."""
    return [
        e.data["code"]
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "warning"
    ]
