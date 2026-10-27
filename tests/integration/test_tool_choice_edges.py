"""Tool choice at its edges, on a scripted model: no tools, one, hundreds; names that clash or a
model API refuses; empty descriptions and huge schemas; an MCP server that is down, slow,
answers garbage or far too much, or a tool that raises; calls made at once; a change approved
or rejected after the check asked for it; every combination of the switches; and every
adapter's trajectory scored by ``tool_coverage``, a run paused for approval included."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from tests.support.adapters import BUILDERS
from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.unit.test_toolbox import READ, Def
from tests.unit.test_toolbox import Gateway as StubGateway
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ConfigurationError, RunStatus, ToolSpec
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.evals import tool_coverage
from trellis.harness.middleware import (
    ANSWER_CHECK,
    TOOL_GUIDANCE,
    AnswerCheck,
    ToolGuidance,
)
from trellis.harness.tools.convert import langchain as lc
from trellis.harness.tools.lint import MOST_TOOLS, lint

CHECK = ANSWER_CHECK.splitlines()[0]
WRITE_FIRST = "Before it is made, and before anyone is asked to approve it"
GUIDED = TOOL_GUIDANCE.splitlines()[0]
RULE = {"item": "a", "low": 2, "high": 6}


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


def checks(model: ScriptedChatModel) -> int:
    return sum(CHECK in str(seen[-1].content) for seen in model.seen)


def writes_checked(model: ScriptedChatModel) -> int:
    return sum(WRITE_FIRST in str(seen[-1].content) for seen in model.seen)


def guided(model: ScriptedChatModel) -> list[bool]:
    return [GUIDED in str(seen[0].content) for seen in model.seen]


def said_to_model(model: ScriptedChatModel) -> list[str]:
    """The tool results in the last call's conversation."""
    return [str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- how many tools
async def test_with_no_tools_neither_guidance_nor_a_check(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["hello"])
    target = ReAct("s", model, tool_guidance=True, answer_check=True)
    assert (await harness.wrap(target, id="none").run("hi", user="u")).answer == "hello"
    assert len(model.seen) == 1 and guided(model) == [False]


async def test_with_one_tool_it_is_named_when_it_changes_data(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["noted?", ("note", {"text": "a"}), "noted"])
    target = ReAct("s", model, tool_guidance=True, answer_check=True)
    assert (await arm(harness.wrap(target, id="one"), [note]).run("x", user="u")).answer == (
        "noted"
    )
    assert "These tools change data: note." in str(model.seen[0][0].content)
    assert "that does it (note)" in str(model.seen[1][-1].content)


async def test_hundreds_of_tools_narrowed_by_hints_widen_to_the_writes_only(
    memory_service: FakeMemoryService,
) -> None:
    reads = [
        tool(lambda: "x", name=f"r{n:03}", side_effects="read", description=f"Report {n}.")
        for n in range(MOST_TOOLS + 10)
    ]
    memory_service.candidates = ["r000"]
    model = ScriptedChatModel(turns=["draft", ("note", {"text": "a"}), "noted"])
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(ReAct("s", model, answer_check=True), id="many"), [*reads, note])
        assert (await agent.run("note a", user="u")).answer == "noted"
    memory = sorted(MEMORY_TOOLS)
    assert model.requests[0]["tools"] == sorted(["r000", *memory])
    assert model.requests[1]["tools"] == [*sorted(["r000", *memory]), "note"]  # appended
    assert [f.code for f in lint([t.spec for t in [*reads, note]])][-1] == "too_many_tools"


async def test_without_tool_search_the_check_offers_what_was_offered(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.candidates = ["look"]
    tools = [look, note, rule, *(tool(lambda: 1, name=f"x{n}") for n in range(3))]
    model = ScriptedChatModel(turns=["draft", "final"])
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(ReAct("s", model, answer_check=True), id="nh"), tools)
        await agent.run("x", user="u", without={"tool_search"})
    assert model.requests[0]["tools"] == model.requests[1]["tools"]  # every tool, both times


# --------------------------------------------------------------------------- names, schemas
async def test_two_functions_of_one_name_are_refused(harness: Harness) -> None:
    other = tool(lambda item: item, name="look", description="Another look.")
    agent = arm(harness.wrap(ReAct("s", ScriptedChatModel(turns=["x"])), id="dup"), [look, other])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "two tools are named look" in result.error.message


def test_an_empty_description_and_a_unicode_name_are_kept_and_flagged() -> None:
    bare = tool(lambda: 1, name="préstamo", description="")
    sent = convert_to_openai_tool(lc.convert([bare.tool])[0])["function"]
    assert (sent["name"], sent["description"]) == ("préstamo", "préstamo")  # the name stands in
    assert [f.code for f in lint([bare.spec])] == ["invalid_name", "no_description"]


def test_a_huge_schema_reaches_the_model_whole() -> None:
    properties = {
        f"field_{n}": {"type": "string", "description": f"Field {n}."} for n in range(500)
    }
    schema = {"type": "object", "properties": properties, "required": ["field_0"]}
    spec = ToolSpec(
        name="wide", description="A tool with five hundred fields.", input_schema=schema
    )
    from trellis.harness.tools.base import Tool

    async def run(args: dict[str, Any]) -> str:
        return "ok"

    sent = convert_to_openai_tool(lc.convert([Tool(spec, run)])[0])["function"]
    assert sent["parameters"] == schema
    assert lint([spec]) == []


# --------------------------------------------------------------------------- MCP servers misbehaving
class Misbehaving(StubGateway):
    """A gateway whose MCP server is down, slow, malformed or huge, by tool."""

    def __init__(self, defs: list[Def], *, down: bool = False) -> None:
        super().__init__(defs)
        self.down = down

    async def tools(self) -> list[Def]:
        if self.down:
            raise ConnectionError("the MCP server is down")
        return await super().tools()

    async def execute(self, name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        if name.endswith("-raises"):
            raise RuntimeError("the server failed")
        if name.endswith("-slow"):
            raise TimeoutError("no answer in time")
        if name.endswith("-garbage"):
            return object()
        return "x" * 100_000  # -huge


def defs(*names: str) -> list[Def]:
    made = [Def(f"erp-{n}", "erp", annotations=READ, code_mode=False) for n in names]
    for d in made:
        d.description = f"The erp's {d.name} report, by store."
    return made


async def test_an_mcp_server_down_at_the_first_listing_fails_the_run_saying_so() -> None:
    gateway = cast("Gateway", Misbehaving(defs("raises"), down=True))
    async with Harness(config=Settings(), gateway=gateway) as h:
        agent = arm(h.wrap(ReAct("s", ScriptedChatModel(turns=["x"])), id="down"), [look])
        result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "the MCP server is down" in result.error.message


async def test_failing_slow_malformed_and_huge_mcp_results_reach_the_model() -> None:
    gateway = cast("Gateway", Misbehaving(defs("raises", "slow", "garbage", "huge")))
    calls = [(f"erp-{n}", {}) for n in ("raises", "slow", "garbage", "huge")]
    model = ScriptedChatModel(turns=[calls, "draft", "final"])
    async with Harness(config=Settings(), gateway=gateway) as h:
        agent = h.wrap(ReAct("s", model, answer_check=True), id="bad")
        dataset = [{"input": "x", "metadata": {"needed_tools": [n for n, _ in calls]}}]
        report = await h.evaluate(agent, dataset, [tool_coverage()])
    [item] = report.items
    [score] = item.scores
    assert score.value == 1.0  # each was called...
    assert score.comment == "every call failed: erp-raises, erp-slow"  # ...two failed
    said = said_to_model(model)
    assert any("erp-raises failed: the server failed" in s for s in said)
    assert any("erp-slow timed out" in s for s in said)
    assert any("<object object at" in s for s in said)  # garbage: its text
    assert any("x" * 1_000 in s for s in said)  # huge: it reaches the model (no window known)
    assert checks(model) == 1


async def test_a_local_tool_that_raises_is_a_failure_the_check_sees(harness: Harness) -> None:
    @tool(side_effects="read")
    def broken(item: str) -> str:
        """A lookup whose backend is down."""
        raise RuntimeError("backend down")

    model = ScriptedChatModel(turns=[("broken", {"item": "a"}), "I could not look a up.", "same"])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="raise"), [broken])
    assert (await agent.run("look a up", user="u")).answer == "same"
    assert "broken failed: backend down" in str(model.seen[1])
    assert checks(model) == 1


# --------------------------------------------------------------------------- calls, approvals
async def test_calls_made_at_once_then_the_check(harness: Harness) -> None:
    both = [("look", {"item": "a"}), ("note", {"text": "a"})]
    model = ScriptedChatModel(turns=[both, both, "draft", "final"])  # the change checked first
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="par"), [look, note])
    assert (await agent.run("x", user="u")).answer == "final"
    assert sorted(said_to_model(model)[:2]) == ["a found", "noted a"]
    assert checks(model) == 1 and writes_checked(model) == 1


@pytest.mark.parametrize(
    ("decision", "said"),
    [("approve", "rule for a: 2-6"), ("reject", "rule was not run: the approver rejected it")],
)
async def test_a_change_the_check_asked_for_is_approved_or_rejected(
    harness: Harness, decision: str, said: str
) -> None:
    model = ScriptedChatModel(turns=["It is set.", ("rule", RULE), "the outcome"])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="ap"), [rule])
    paused = await agent.run("make sure a stays between 2 and 6", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, decision, reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "the outcome"
    assert said in said_to_model(model)[-1]
    assert checks(model) == 1  # the resumed graph is not checked again


# --------------------------------------------------------------------------- the switches
@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("check", [False, True])
async def test_every_combination_of_the_switches(
    harness: Harness, guidance: bool, check: bool
) -> None:
    turns: list[Any] = [("look", {"item": "a"}), "draft"] + (["final"] if check else [])
    model = ScriptedChatModel(turns=turns)
    target = ReAct("s", model, tool_guidance=guidance, answer_check=check)
    result = await arm(harness.wrap(target, id="combo"), [look]).run("x", user="u")
    assert result.answer == ("final" if check else "draft")
    assert guided(model) == [guidance] * len(model.seen)
    assert checks(model) == int(check)


async def test_the_middleware_in_a_deep_agents_graph(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["draft", ("look", {"item": "a"}), "final"])
    built = await native_tools(harness, look, framework="deepagents", without={"mcp"})
    graph = create_deep_agent(model=model, tools=built, middleware=[AnswerCheck(), ToolGuidance()])
    result = await harness.wrap(graph, id="deep").run("x", user="u")
    assert result.answer == "final"
    assert checks(model) == 1 and GUIDED in str(model.seen[0][0].content)


# --------------------------------------------------------------------------- every adapter
@pytest.mark.parametrize("adapter", list(BUILDERS))
async def test_every_adapter_is_scored_by_tool_coverage_paused_or_not(
    harness: Harness, adapter: str, tmp_path: Path
) -> None:
    for plan, status in (([("look", {"item": "a"})], "success"), ([("rule", RULE)], "interrupted")):
        target, tools = await BUILDERS[adapter](harness, [look, rule], tmp_path, plan)
        agent = arm(harness.wrap(target, id=f"cov-{adapter}-{status}"), tools)
        needed = plan[0][0]
        report = await harness.evaluate(
            agent,
            [{"input": "x", "metadata": {"needed_tools": [needed, "never"]}}],
            [tool_coverage()],
        )
        [item] = report.items
        assert item.status == status, item.error
        assert [(s.value, s.comment) for s in item.scores] == [(0.5, "not called: never")]


# --------------------------------------------------------------------------- a change, checked
RULED = "Standing rule from memory: never create a stock rule for item a."


async def test_a_change_a_rule_forbids_is_refused_before_anyone_is_asked(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("rule", RULE), "A standing rule forbids a rule for a."])
    agent = arm(harness.wrap(ReAct(RULED, model, answer_check=True), id="rr"), [rule])
    result = await agent.run("create a rule for a, min 2 max 6", user="u")
    assert result.status is RunStatus.SUCCESS and result.interrupt is None  # nobody asked
    assert result.answer == "A standing rule forbids a rule for a."
    asked = str(model.seen[1][-1].content)
    assert 'rule({"item": "a", "low": 2, "high": 6})' in asked and WRITE_FIRST in asked
    assert str(model.seen[1][-2].content).startswith("(about to call rule(")
    assert said_to_model(model) == []  # the tool never ran


async def test_a_change_no_rule_forbids_goes_on_to_its_approval(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("rule", RULE), ("rule", RULE)])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="ok"), [rule])
    result = await agent.run("create a rule for a", user="u")
    assert result.status is RunStatus.PAUSED and result.interrupt is not None
    assert writes_checked(model) == 1


async def test_the_change_check_is_once_per_invocation_off_or_kept_on_an_empty_reply(
    harness: Harness,
) -> None:
    once = ScriptedChatModel(
        turns=[
            ("note", {"text": "a"}),
            ("note", {"text": "a"}),
            ("note", {"text": "b"}),
            "done",
            "done",
        ]
    )
    agent = arm(harness.wrap(ReAct("s", once, answer_check=True), id="1x"), [note])
    assert (await agent.run("x", user="u")).answer == "done"
    assert writes_checked(once) == 1 and said_to_model(once) == ["noted a", "noted b"]
    off = ScriptedChatModel(turns=[("note", {"text": "a"}), "done", "done"])
    target = plain_check(off, AnswerCheck(writes=False))
    assert (await arm(harness.wrap(target, id="off"), [note]).run("x", user="u")).answer == ("done")
    assert writes_checked(off) == 0
    empty = ScriptedChatModel(turns=[("note", {"text": "a"}), "", "done", "done"])
    agent = arm(harness.wrap(ReAct("s", empty, answer_check=True), id="em"), [note])
    assert (await agent.run("x", user="u")).answer == "done"
    assert said_to_model(empty) == ["noted a"]  # the drafted call stood


def plain_check(model: ScriptedChatModel, check: AnswerCheck) -> Any:
    from langchain.agents import create_agent

    from trellis.harness.middleware import HarnessTools, ModelHooks, RunCheckpointer

    return create_agent(
        model, middleware=[check, HarnessTools(), ModelHooks()], checkpointer=RunCheckpointer()
    )


# --------------------------------------------------------------------------- no record
@tool(side_effects="read")
def find(style: str) -> Any:
    """A style's record, by its id."""
    return {"empty": [], "missing": "style ZZ999999 not found", "blank": ""}.get(style, "ST-1: 4")


@pytest.mark.parametrize("style", ["empty", "missing", "blank"])
async def test_nothing_found_is_said_plainly(harness: Harness, style: str) -> None:
    model = ScriptedChatModel(turns=[("find", {"style": style}), "maybe 4?", "No record."])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id=f"nr-{style}"), [find])
    assert (await agent.run(f"stock of {style}?", user="u")).answer == "No record."
    asked = str(model.seen[2][-1].content)
    assert "Nothing the tools returned is about what was asked" in asked
    assert "there is no record of it" in asked


async def test_a_found_record_is_not_told_there_is_none(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("find", {"style": "ST-1"}), "4", "4"])
    agent = arm(harness.wrap(ReAct("s", model, answer_check=True), id="found"), [find])
    await agent.run("stock of ST-1?", user="u")
    assert "no record" not in str(model.seen[2][-1].content).replace("If there is no record", "")


@pytest.mark.parametrize(("called", "source"), [(False, "in memory"), (True, "in memory or")])
async def test_memory_holding_nothing_is_said_plainly(
    memory_service: FakeMemoryService, called: bool, source: str
) -> None:
    memory_service.evidence_status = "INSUFFICIENT"
    turns: list[Any] = [("find", {"style": "empty"})] if called else []
    model = ScriptedChatModel(turns=[*turns, "a guess", "No record of it."])
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(ReAct("s", model, answer_check=True), id="ins"), [find])
        assert (await agent.run("what did I say about ZZ?", user="u")).answer == (
            "No record of it."
        )
    assert f"Nothing {source}" in str(model.seen[-1][-1].content)


# --------------------------------------------------------------------------- what it can do
async def test_what_the_agent_can_do_is_said_in_its_instructions_on_every_call(
    harness: Harness,
) -> None:
    said = "You cannot accept an image upload. You can read a PDF."
    model = ScriptedChatModel(turns=[("look", {"item": "x"}), "No."])
    target = ReAct(f"s\n{said}", model, tool_guidance=True)
    agent = arm(harness.wrap(target, id="cap"), [look])
    assert (await agent.run("Can you accept an image upload?", user="u")).answer == "No."
    assert len(model.seen) == 2
    assert all(said in str(seen[0].content) for seen in model.seen)
    assert GUIDED in str(model.seen[0][0].content)  # the guidance after it, as before
    with pytest.raises(ConfigurationError):
        ToolGuidance("")
    with pytest.raises(TypeError, match=r"ToolGuidance\(capabilities=\) was removed"):
        ToolGuidance(capabilities={"read a PDF": True})  # pyright: ignore[reportCallIssue]


def test_only_the_requests_own_results_count_as_nothing_found() -> None:
    from langchain_core.messages import AIMessage

    from trellis.harness.middleware import _no_record

    earlier = [HumanMessage("first"), ToolMessage("ST-1: 4", tool_call_id="1"), AIMessage("4")]
    now = [HumanMessage("and ZZ?"), ToolMessage("not found", tool_call_id="2")]
    assert "no record" in _no_record([*earlier, *now], None)
    assert _no_record([*now[:1], ToolMessage("ST-1: 4", tool_call_id="3")], None) == ""
    # no request in the conversation at all: every result counts
    assert "no record" in _no_record([ToolMessage("[]", tool_call_id="4")], None)


# --------------------------------------------------------------------------- automatic (None)
def lookers(count: int) -> list[Any]:
    def look_up(item: str) -> str:
        """Look an item up in one of the shelves."""
        return item

    return [tool(look_up, name=f"look_{n}", side_effects="read") for n in range(count)]


async def test_automatic_is_off_as_measured_many_tools_and_a_rule_alike(
    harness: Harness,
) -> None:
    """``None`` (the default) adds neither the guidance nor the check — measured on a small
    model, the guidance made it call changes nobody asked for and the check broke a rule."""
    many = lookers(30)
    model = ScriptedChatModel(turns=["made it"])
    agent = arm(harness.wrap(ReAct("s", model, context_window=64_000), id="auto"), [*many, rule])
    said = [
        {"role": "system", "content": "Remembered: never make a rule for a-1."},
        {"role": "user", "content": "make a rule for a-1"},
    ]
    assert (await agent.run(said, user="u")).answer == "made it"
    assert guided(model) == [False] and checks(model) == 0
