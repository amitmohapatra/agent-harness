"""An answer stating figures nothing its model was given holds is sent back once
(``middleware.figures_checked``): the model computes them with a tool, or answers without them.
A follow-up's figure worked out in the model's head is the commonest wrong one."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import sandbox as sandbox_module
from trellis.harness.middleware import FIGURES


@pytest.fixture
def boxes(monkeypatch: pytest.MonkeyPatch) -> FakeSandboxes:
    """The shell the harness gives an agent by itself, on a fake provider: a tool that computes."""
    provider = FakeSandboxes()
    monkeypatch.setitem(sandbox_module._DEFAULT, "default", sandbox(provider))
    return provider


@tool(side_effects="read")
def sales(region: str) -> str:
    """A region's Q3 revenue."""
    return '{"east": 157733.89, "west": 216700}'


def told(model: ScriptedChatModel, call: int) -> str:
    last = model.seen[call][-1]
    return str(last.content) if isinstance(last, HumanMessage) else ""


def warned(events: list[Any]) -> list[str]:
    return [
        e.data["code"]
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "warning"
    ]


async def test_a_figure_worked_out_by_hand_is_sent_back_once_and_computed(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    turns: list[Any] = [
        ("sales", {"region": "all"}),
        "East would need about £188,000 to match, and 1.0355 times its margin.",
        ("execute", {"command": "python3 -c 'print(157733.89 + 216700)'"}),
        "Together east and west made £374,433.89, and more: £999,999.",
    ]
    model = ScriptedChatModel(turns=turns)
    agent = harness.wrap(
        ReAct(model=model, tools=[sales], system_prompt="You analyse sales."), id="figures"
    )
    events = [e async for e in agent.stream("what would east need?", user="ada")]
    done = events[-1]
    assert done.outcome is not None and done.outcome.value == "success", done
    # told which figures, with the tool that computes them: its draft answer before it
    said = FIGURES.format(figures="188000, 1.0355", verb="are", them="them", tools="execute")
    assert told(model, 2) == said and "188,000" in str(model.seen[2][-2].content)
    assert boxes.commands[-1].startswith("python3")
    # once a run: its next answer is the run's, whatever it states
    assert done.data["result"].endswith("£999,999.") and len(model.seen) == 4
    assert warned(events).count("figures") == 1


async def test_an_answer_whose_figures_are_backed_is_the_runs_as_it_is(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    turns: list[Any] = [("sales", {"region": "all"}), "East made £157,733.89, about £158k."]
    model = ScriptedChatModel(turns=turns)
    agent = harness.wrap(
        ReAct(model=model, tools=[sales], system_prompt="You analyse sales."), id="figures-backed"
    )
    done = await agent.run("how did east do?", user="ada")
    assert done.status is RunStatus.SUCCESS and len(model.seen) == 2


async def test_no_tool_to_compute_with_or_no_check_asked_for_leaves_the_answer(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    """Sent back only where the model can do something about it (a tool that computes is
    offered) and the run checks its answers (``without=["grounding"]``: not)."""
    answer = "East would need about £188,000."
    for name, without in (("no-code", ["code"]), ("no-check", ["grounding"])):
        model = ScriptedChatModel(turns=[("sales", {"region": "all"}), answer])
        agent = harness.wrap(
            ReAct(model=model, tools=[sales], system_prompt="You analyse sales."),
            id=f"figures-{name}",
        )
        done = await agent.run("what would east need?", user="ada", without=without)
        assert done.status is RunStatus.SUCCESS and done.answer == answer
        assert len(model.seen) == 2, name


async def test_a_reply_with_neither_answer_nor_call_leaves_the_first_answer(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    answer = "East would need about £188,000."
    model = ScriptedChatModel(turns=[("sales", {"region": "all"}), answer, ""])
    agent = harness.wrap(
        ReAct(model=model, tools=[sales], system_prompt="You analyse sales."), id="figures-empty"
    )
    done = await agent.run("what would east need?", user="ada")
    assert done.status is RunStatus.SUCCESS and done.answer == answer
