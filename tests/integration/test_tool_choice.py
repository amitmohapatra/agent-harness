"""Tool choice, on a scripted model: the native tool selector among the run's tools (the
team's own opt-in), no harness selector without hints, and ``tool_coverage`` scoring each
item — a paused one too."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import LLMToolSelectorMiddleware

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, tool
from trellis.contracts import ToolCall, ToolOutcome
from trellis.contracts import ToolStatus as Status
from trellis.harness.evals import EvalCase, exact_match, tool_coverage


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


async def test_without_hints_no_harness_selector_narrows_the_tools(harness: Harness) -> None:
    """One owner of tool narrowing (ADR 0003 §6): memory's hints. Without them every tool is
    offered and no selector of the harness's makes a model call of its own: there is no
    fallback chain (``LLMToolSelectorMiddleware`` stays the team's own opt-in, above)."""
    from trellis.harness import automatic, react

    assert not hasattr(automatic, "ToolSelection") and not hasattr(react, "defaults_on")
    found = [tool(lambda q: q, name=f"find_{n}", side_effects="read") for n in range(25)]
    model = ScriptedChatModel(turns=["found"])
    result = await arm(harness.wrap(ReAct("s", model), id="all"), found).run("find a", user="u")
    assert result.answer == "found", result.error
    [request] = model.requests  # one model call: the answer, no selector call before it
    assert sorted(request["tools"]) == sorted(f"find_{n}" for n in range(25))
