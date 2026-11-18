"""``h.harness_tools()``: the tools the harness gives an agent by itself — the memory tools,
the schedule tools and ``ask_user`` — as LangChain tools for a graph built by hand, bound to
its model and put in its ``ToolNode`` when it is built. Each call is the run's that makes it:
``ask_user`` pauses the run and its answer is the call's result, ``schedule_task`` makes the
run's user a schedule, a part the run is without answers that it is off, a run a schedule fired
makes no schedule, and outside a run a call is refused."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tests.support.chat_model import ScriptedChatModel
from tests.support.gateway import URL, FakeGateway
from tests.support.memory import MEMORY_TOOLS
from trellis import Harness, Settings
from trellis.contracts import RunStatus, ToolError
from trellis.harness import pipeline
from trellis.harness.frontend import Frontend
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import SCHEDULE_TOOLS
from trellis.harness.tools.ask import APPS, ASK_USER
from trellis.harness.tools.bridge import FIRED

WATCH = {"task": "check stock of SKU-1", "when": "0 9 * * *"}


def by_hand(model: ScriptedChatModel, tools: list[Any], *, saver: Any = None) -> Any:
    """A ``StateGraph`` as a LangGraph team writes it: its model bound to ``tools`` in its own
    node, a ``ToolNode`` running them."""
    bound = model.bind_tools(tools)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=saver)


def results(model: ScriptedChatModel) -> list[str]:
    """What the model read back from its tool calls, by its last call."""
    return [str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage)]


async def test_they_are_the_memory_tools_the_schedule_tools_and_ask_user(
    harness: Harness, memory_harness: Harness
) -> None:
    named = [*SCHEDULE_TOOLS, ASK_USER]
    assert [t.name for t in await harness.harness_tools()] == named
    assert [t.name for t in await memory_harness.harness_tools()] == [*MEMORY_TOOLS, *named]


@pytest.mark.parametrize("saver", [InMemorySaver, None], ids=["checkpointed", "re-run"])
async def test_ask_user_in_a_tool_node_pauses_the_run_and_its_answer_is_the_result(
    harness: Harness, saver: Any
) -> None:
    """Paused as the run's question (LangGraph's ``interrupt`` with a checkpointer, a re-run
    from the journal without one), and resumed with the person's answer as the call's
    result. (Re-run, the graph's model asks again: the journal answers.)"""
    asks = (ASK_USER, {"question": "Which region?", "options": ["north", "south"]})
    model = ScriptedChatModel(turns=[asks, *([] if saver else [asks]), "north"])
    graph = by_hand(model, await harness.harness_tools(), saver=saver and saver())
    agent = harness.wrap(graph, id="asker")
    paused = await agent.run("stock?", user="ada", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question.startswith("Which region?")
    assert paused.interrupt.options == ["north", "south"]
    done = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="north", reviewer="ada"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "north"
    assert results(model) == ["north"]


async def test_schedule_task_makes_the_runs_user_a_schedule(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("schedule_task", WATCH), "scheduled"])
    agent = harness.wrap(by_hand(model, await harness.harness_tools()), id="desk")
    done = await agent.run("check SKU-1 every day at 9", user="ada", thread="t1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert isinstance(harness.runs, LocalRuns)
    [made] = harness.runs._schedules.values()
    assert made.agent_id == "desk" and made.cadence == "0 9 * * *"
    assert "check stock of SKU-1" in str(made.input)


@pytest.mark.parametrize(
    ("without", "call"),
    [("schedules", ("schedule_task", WATCH)), ("ask", (ASK_USER, {"question": "Which?"}))],
)
async def test_a_part_the_run_is_without_answers_that_it_is_off(
    harness: Harness, without: str, call: tuple[str, dict[str, Any]]
) -> None:
    model = ScriptedChatModel(turns=[call, "done"])
    agent = harness.wrap(by_hand(model, await harness.harness_tools()), id="off")
    done = await agent.run("go", user="ada", without={without})
    assert done.status is RunStatus.SUCCESS
    assert results(model) == [f"{call[0]} is off in this run (without {without})"]
    assert not harness.runs._schedules  # type: ignore[attr-defined]


async def test_a_run_a_schedule_fired_makes_no_schedule(harness: Harness) -> None:
    """A fired run does its task, no more: the graph's bound ``schedule_task`` is refused in
    it, as every other target is not offered it there."""
    model = ScriptedChatModel(
        turns=[("schedule_task", WATCH), "scheduled", ("schedule_task", WATCH), "checked"]
    )
    agent = harness.wrap(by_hand(model, await harness.harness_tools()), id="desk")
    await agent.run("check SKU-1 every day at 9", user="ada", thread="t1")
    [made] = harness.runs._schedules.values()  # type: ignore[attr-defined]
    fired = await harness.runs.schedules.fire(made.schedule_id)
    assert await harness.worker([agent]).run_once() is True
    record = await harness.runs.get(fired.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS
    assert results(model) == [FIRED.format(tool="schedule_task")]
    assert len(harness.runs._schedules) == 1  # type: ignore[attr-defined]


async def test_outside_a_run_a_call_is_refused(memory_harness: Harness) -> None:
    for found in await memory_harness.harness_tools():
        with pytest.raises(ToolError, match="runs inside a Harness run"):
            await found.ainvoke({"question": "Which?", **WATCH})


async def test_the_gateways_mcp_tools_come_first_and_are_off_without_mcp() -> None:
    fake = FakeGateway(tools=["crm-get"])
    settings = Settings(llm_gateway="bifrost", llm_base_url=URL, llm_api_key="vk")
    async with Harness(config=settings, gateway=fake.gateway()) as h:
        tools = await h.harness_tools()
        assert [t.name for t in tools] == ["crm-get", *SCHEDULE_TOOLS, ASK_USER]
        model = ScriptedChatModel(turns=[("crm-get", {}), "done"])
        done = await h.wrap(by_hand(model, tools), id="crm").run("go", user="ada", without={"mcp"})
        assert done.status is RunStatus.SUCCESS
        assert results(model) == ["crm-get is off in this run (without mcp)"]


async def test_a_create_agent_graph_is_offered_them_in_the_runs_words(harness: Harness) -> None:
    """Bound when the graph was built, they say what they say to every other target at each
    model call: ``schedule_task`` when it is in the user's zone, ``ask_user`` the app's own
    controls (the run's words, as ``Agent.tools_for`` gives them)."""
    model = ScriptedChatModel(turns=["hi"])
    agent = harness.wrap(create_agent(model, tools=await harness.harness_tools()), id="said")
    picker = {"name": "pick_date", "description": "A date picker."}
    app = Frontend(tools=(picker,), timezone="Asia/Kolkata")
    record = await agent._opened("hi", user="ada", thread="t1", tenant=None, app=app)
    assert (await pipeline.attempt(agent, record, "hi")).answer == "hi"
    said = model.described[-1]
    assert "(Asia/Kolkata, the user's zone)" in said["schedule_task"]
    assert said[ASK_USER].startswith(APPS.format(tools="pick_date"))
