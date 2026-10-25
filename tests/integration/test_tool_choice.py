"""Tool choice, on a scripted model: the guidance in every call that offers tools, the answer
checked before it is final (the draft not streamed, the tools still needed called, a change
reaching its approval, a run narrowed by the tool hints widened), the native tool selector
among the run's tools, and ``tool_coverage`` scoring each item — a paused one too."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from trellis import Harness, ReAct, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolCall, ToolOutcome
from trellis.contracts import ToolStatus as Status
from trellis.harness.evals import EvalCase, exact_match, tool_coverage
from trellis.harness.middleware import (
    ANSWER_CHECK,
    TOOL_GUIDANCE,
    AnswerCheck,
    HarnessTools,
    ModelHooks,
    RunCheckpointer,
    StepLimit,
    ToolGuidance,
)

CHECK = ANSWER_CHECK.splitlines()[0]


@tool(side_effects="read")
def look(item: str) -> str:
    """Look an item up."""
    return f"{item} found"


@tool(side_effects="write")
def note(text: str) -> str:
    """Write a note."""
    return f"noted {text}"


@tool(side_effects="irreversible")
def rule(item: str, low: int, high: int) -> str:
    """Create a stock rule for an item."""
    return f"rule for {item}: {low}-{high}"


@tool(side_effects="read")
def broken(item: str) -> str:
    """A lookup whose backend is down."""
    raise RuntimeError("backend down")


def graph(model: Any, *middleware: Any, **kwargs: Any) -> Any:
    return create_agent(
        model, middleware=list(middleware), checkpointer=RunCheckpointer(), **kwargs
    )


def plain(model: Any, *middleware: Any, **kwargs: Any) -> Any:
    """A graph run outside the harness (no checkpointer)."""
    return create_agent(model, middleware=list(middleware), **kwargs)


def checked(model: ScriptedChatModel) -> list[int]:
    """The model calls that were asked the check."""
    return [n for n, seen in enumerate(model.seen) if CHECK in str(seen[-1].content)]


# --------------------------------------------------------------------------- guidance
async def test_the_guidance_is_in_every_call_offering_tools_and_names_the_writes(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"}), "done"])
    target = graph(model, HarnessTools(), ToolGuidance(), ModelHooks(), system_prompt="Base.")
    result = await arm(harness.wrap(target, id="g"), [look, note]).run("x", user="u")
    assert result.answer == "done"
    for seen in model.seen:
        assert isinstance(seen[0], SystemMessage)
        assert seen[0].content == f"Base.\n\n{TOOL_GUIDANCE}\nThese tools change data: note."


async def test_the_guidance_without_tools_without_a_prompt_and_as_content_blocks() -> None:
    bare = ScriptedChatModel(turns=["hi"])
    await plain(bare, ToolGuidance()).ainvoke({"messages": [HumanMessage("x")]})
    assert not isinstance(bare.seen[0][0], SystemMessage)  # no tools: nothing added
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def own(x: str) -> str:
        """The graph's own tool."""
        return x

    outside = ScriptedChatModel(turns=["hi"])
    await plain(outside, ToolGuidance("Pick well."), tools=[own]).ainvoke(
        {"messages": [HumanMessage("x")]}
    )
    assert outside.seen[0][0].content == "Pick well."  # outside a run: no tools named
    blocks = ScriptedChatModel(turns=["hi"])
    system = SystemMessage(content=[{"type": "text", "text": "Base."}])
    await plain(blocks, ToolGuidance("Pick well."), tools=[own], system_prompt=system).ainvoke(
        {"messages": [HumanMessage("x")]}
    )
    assert blocks.seen[0][0].content == [
        {"type": "text", "text": "Base."},
        {"type": "text", "text": "Pick well."},
    ]
    with pytest.raises(ConfigurationError):
        ToolGuidance(" ")


# --------------------------------------------------------------------------- the check
async def test_a_draft_that_skips_a_part_is_checked_and_the_missing_tool_called(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[("look", {"item": "a"}), "a is found.", ("note", {"text": "a"}), "a found, noted."]
    )
    target = graph(model, AnswerCheck(), HarnessTools(), ModelHooks())
    agent = arm(harness.wrap(target, id="c"), [look, note])
    events = [e async for e in agent.stream("look a up and note it", user="u")]
    assert events[-1].data["result"] == "a found, noted."
    assert checked(model) == [2]
    draft, asked = model.seen[2][-2:]
    assert isinstance(draft, AIMessage) and draft.content == "a is found."
    assert isinstance(asked, HumanMessage) and "that does it (note)" in str(asked.content)
    assert len(model.seen) == 4  # one check: the final answer is not checked again
    said = "".join(
        str(e.data.get("delta", "")) for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT
    )
    assert "a is found." not in said and "a found, noted." in said  # the draft is not streamed


async def test_a_change_claimed_in_text_is_called_and_reaches_its_approval(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=["Done: the rule is set.", ("rule", {"item": "a", "low": 2, "high": 6})]
    )
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="w"), [look, rule])
    result = await agent.run("create a rule for a, min 2 max 6", user="u")
    assert result.status is RunStatus.PAUSED and result.interrupt is not None
    assert result.interrupt.tool_call is not None and result.interrupt.tool_call.tool == "rule"
    assert "(rule)" in str(model.seen[1][-1].content)


async def test_checks_are_counted_and_an_empty_reply_keeps_the_draft(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["first", ("look", {"item": "a"}), "second", "third"])
    target = graph(model, AnswerCheck(checks=2), HarnessTools(), ModelHooks())
    result = await arm(harness.wrap(target, id="n"), [look]).run("x", user="u")
    assert result.answer == "third" and checked(model) == [1, 3]
    empty = ScriptedChatModel(turns=["the draft", ""])
    target = graph(empty, AnswerCheck(), HarnessTools(), ModelHooks())
    assert (await arm(harness.wrap(target, id="e"), [look]).run("x", user="u")).answer == (
        "the draft"
    )
    with pytest.raises(ConfigurationError):
        AnswerCheck(0)
    with pytest.raises(ConfigurationError, match="placeholder"):
        AnswerCheck(text="Check it.")


async def test_outside_a_run_the_check_names_no_tools() -> None:
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def own(x: str) -> str:
        """The graph's own tool."""
        return x

    model = ScriptedChatModel(turns=["draft", "final"])
    state = await plain(model, AnswerCheck(), tools=[own]).ainvoke(
        {"messages": [HumanMessage("x")]}
    )
    assert state["messages"][-1].content == "final"
    assert "that does it;" in str(model.seen[1][-1].content)


async def test_no_check_without_tools_and_a_structured_draft_is_shown_as_json(
    harness: Harness,
) -> None:
    bare = ScriptedChatModel(turns=["hi"])
    await plain(bare, AnswerCheck()).ainvoke({"messages": [HumanMessage("x")]})
    assert len(bare.seen) == 1

    class Answer(BaseModel):
        value: str

    model = ScriptedChatModel(turns=[("Answer", {"value": "a"}), ("Answer", {"value": "b"})])
    target = graph(model, AnswerCheck(), HarnessTools(), ModelHooks(), response_format=Answer)
    result = await arm(harness.wrap(target, id="s"), [look]).run("x", user="u")
    assert result.answer == Answer(value="b")
    assert model.seen[1][-2].content == '{"value": "a"}'


async def test_after_the_step_limit_the_check_only_corrects_the_text(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"}), "partial", "partial, checked"])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True, max_steps=1), id="l"), [look])
    assert (await agent.run("x", user="u")).answer == "partial, checked"
    assert [r.get("tool_choice") for r in model.requests[1:]] == ["none", "none"]


async def test_a_run_narrowed_by_the_hints_is_offered_every_tool_from_the_check(
    memory_service: FakeMemoryService,
) -> None:
    from trellis import Settings

    memory_service.candidates = ["look"]
    tools = [look, note, rule, broken, tool(lambda: "x", name="extra", side_effects="read")]
    model = ScriptedChatModel(turns=["a draft", ("note", {"text": "a"}), "noted"])
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(ReAct("s", model, answer_check=True), id="h"), tools)
        assert (await agent.run("note a", user="u")).answer == "noted"
    memory = sorted(MEMORY_TOOLS)
    assert model.requests[0]["tools"] == sorted(["look", *memory])
    assert model.requests[1]["tools"] == sorted([t.spec.name for t in tools] + memory)


# --------------------------------------------------------------------------- native selector
async def test_the_native_tool_selector_picks_among_the_runs_tools(harness: Harness) -> None:
    # it picks before every model call
    selector = ScriptedChatModel(turns=[("ToolSelectionResponse", {"tools": ["note"]})] * 2)
    model = ScriptedChatModel(turns=[("note", {"text": "a"}), "noted"])
    picked = LLMToolSelectorMiddleware(model=selector, max_tools=1, on_parsing_failure="all")
    target = ReAct("s", model, middleware=[picked])
    result = await arm(harness.wrap(target, id="sel"), [look, note, rule]).run("x", user="u")
    assert result.answer == "noted"
    assert [r["tools"] for r in model.requests] == [["note"], ["note"]]
    assert len(selector.seen) == 2


# --------------------------------------------------------------------------- coverage
def case(*calls: tuple[str, Status], **metadata: Any) -> EvalCase:
    trajectory = [(ToolCall(tool=t, args={}), ToolOutcome(tool=t, status=s)) for t, s in calls]
    return EvalCase(input="x", output="y", metadata=metadata, trajectory=trajectory)


async def test_tool_coverage_is_the_share_of_the_needed_tools_called() -> None:
    ok, error = Status.OK, Status.ERROR
    score = await tool_coverage()(
        case(
            ("a", ok),
            ("overview", ok),
            ("c", error),
            needed_tools=["a", "b", "c", "a"],
            avoid_tools="overview",
        )
    )
    assert score is not None and score.value == round(2 / 3, 4)
    assert score.comment == (
        "not called: b; every call failed: c; called though not needed: overview"
    )
    fixed = await tool_coverage(["a"], avoid=[], name="cov")(case(("a", ok)))
    assert fixed is not None and (fixed.name, fixed.value, fixed.comment) == ("cov", 1.0, None)
    assert await tool_coverage()(case(("a", ok))) is None  # nothing needed
    assert await tool_coverage(["a"])(EvalCase(input="x", output="y")) is None  # no trajectory


async def test_an_item_paused_for_approval_is_scored_by_tool_coverage_only(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[("rule", {"item": "a", "low": 2, "high": 6}), ("look", {"item": "a"}), "found"]
    )
    agent = arm(harness.wrap(ReAct("s", model), id="p"), [look, rule])
    dataset = [
        {"input": "rule for a", "expected": "x", "metadata": {"needed_tools": ["rule"]}},
        {"input": "look a up", "expected": "found", "metadata": {"needed_tools": ["look"]}},
    ]
    report = await harness.evaluate(agent, dataset, [tool_coverage(), exact_match()], concurrency=1)
    paused, answered = report.items
    assert paused.status == "interrupted" and paused.output is None
    assert [(s.name, s.value) for s in paused.scores] == [("tool_coverage", 1.0)]
    assert {s.name: s.value for s in answered.scores} == {"tool_coverage": 1.0, "exact_match": True}
    assert report.summary["tool_coverage"].count == 2


async def test_an_item_paused_on_a_question_is_not_scored(harness: Harness) -> None:
    from trellis import current

    @tool(side_effects="read")
    async def ask_person(question: str) -> str:
        """Ask a person."""
        run = current()
        assert run is not None
        return str(await run.ask(question))

    model = ScriptedChatModel(turns=[("ask_person", {"question": "which?"})])
    agent = arm(harness.wrap(ReAct("s", model), id="q"), [ask_person])
    report = await harness.evaluate(
        agent, [{"input": "x", "metadata": {"needed_tools": ["ask_person"]}}], [tool_coverage()]
    )
    [item] = report.items
    assert item.status == "interrupted" and item.scores == []


async def test_a_call_told_to_call_no_tool_is_not_checked(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"}), "partial"])
    target = graph(model, HarnessTools(), StepLimit(1), AnswerCheck(), ModelHooks())
    assert (await arm(harness.wrap(target, id="none"), [look]).run("x", user="u")).answer == (
        "partial"
    )
    assert checked(model) == [] and model.requests[-1]["tool_choice"] == "none"


async def test_the_measured_selector_picks_without_hints_past_its_threshold_keeping_used(
    harness: Harness, memory_service: FakeMemoryService
) -> None:
    from trellis.harness.automatic import ToolSelection

    found = [tool(lambda q: q, name=f"find_{n}", side_effects="read") for n in range(6)]
    model = ScriptedChatModel(
        turns=[
            ("ToolSelectionResponse", {"tools": ["find_1"]}),
            ("find_1", {"q": "a"}),
            ("ToolSelectionResponse", {"tools": ["find_2"]}),
            "found",
        ]
    )
    target = ReAct("s", model, middleware=[ToolSelection(above=2, keep=1)])
    result = await arm(harness.wrap(target, id="picks"), found).run("find a", user="u")
    assert result.answer == "found", result.error
    offered = [
        sorted(r["tools"]) for r in model.requests if r["tools"] != ["ToolSelectionResponse"]
    ]
    assert offered == [["find_1"], ["find_1", "find_2"]]  # the used one kept
    # with hints, the hints narrow: no selector call
    memory_service.candidates = ["find_0"]
    hinted = ScriptedChatModel(turns=["found"])
    from trellis import Settings

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        target = ReAct("s", hinted, middleware=[ToolSelection(above=0, keep=1)])
        assert (await arm(h.wrap(target, id="h"), found).run("x", user="u")).answer == "found"
    assert sorted(hinted.requests[0]["tools"]) == sorted(["find_0", *MEMORY_TOOLS])
