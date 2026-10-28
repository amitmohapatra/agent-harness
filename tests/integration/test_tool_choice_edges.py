"""Tool choice at its edges, on a scripted model: names that clash or a model API refuses;
empty descriptions and huge schemas; an MCP server that is down, slow, answers garbage or far
too much, or a tool that raises; calls made at once; a change approved or rejected; what the
agent can do said in its instructions; and every adapter's trajectory scored by
``tool_coverage``, a run paused for approval included."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.unit.test_toolbox import READ, Def
from tests.unit.test_toolbox import Gateway as StubGateway
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import RunStatus, ToolSpec
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.evals import tool_coverage
from trellis.harness.tools.convert import langchain as lc
from trellis.harness.tools.lint import lint

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


def said_to_model(model: ScriptedChatModel) -> list[str]:
    """The tool results in the last call's conversation."""
    return [str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage)]


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
    model = ScriptedChatModel(turns=[calls, "final"])
    async with Harness(config=Settings(), gateway=gateway) as h:
        agent = h.wrap(ReAct("s", model), id="bad")
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


async def test_a_local_tool_that_raises_is_a_failure_the_model_sees(harness: Harness) -> None:
    @tool(side_effects="read")
    def broken(item: str) -> str:
        """A lookup whose backend is down."""
        raise RuntimeError("backend down")

    model = ScriptedChatModel(turns=[("broken", {"item": "a"}), "I could not look a up."])
    agent = arm(harness.wrap(ReAct("s", model), id="raise"), [broken])
    assert (await agent.run("look a up", user="u")).answer == "I could not look a up."
    assert "broken failed: backend down" in str(model.seen[1])


# --------------------------------------------------------------------------- calls, approvals
async def test_calls_made_at_once_each_reach_the_model(harness: Harness) -> None:
    both = [("look", {"item": "a"}), ("note", {"text": "a"})]
    model = ScriptedChatModel(turns=[both, "final"])
    agent = arm(harness.wrap(ReAct("s", model), id="par"), [look, note])
    assert (await agent.run("x", user="u")).answer == "final"
    assert sorted(said_to_model(model)) == ["a found", "noted a"]


@pytest.mark.parametrize(
    ("decision", "said"),
    [("approve", "rule for a: 2-6"), ("reject", "rule was not run: the approver rejected it")],
)
async def test_a_change_is_approved_or_rejected(harness: Harness, decision: str, said: str) -> None:
    model = ScriptedChatModel(turns=[("rule", RULE), "the outcome"])
    agent = arm(harness.wrap(ReAct("s", model), id="ap"), [rule])
    paused = await agent.run("make sure a stays between 2 and 6", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, decision, reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "the outcome"
    assert said in said_to_model(model)[-1]


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


# --------------------------------------------------------------------------- what it can do
async def test_what_the_agent_can_do_is_said_in_its_instructions_on_every_call(
    harness: Harness,
) -> None:
    said = "You cannot accept an image upload. You can read a PDF."
    model = ScriptedChatModel(turns=[("look", {"item": "x"}), "No."])
    target = ReAct(f"s\n{said}", model)
    agent = arm(harness.wrap(target, id="cap"), [look])
    assert (await agent.run("Can you accept an image upload?", user="u")).answer == "No."
    assert len(model.seen) == 2
    assert all(said in str(seen[0].content) for seen in model.seen)
