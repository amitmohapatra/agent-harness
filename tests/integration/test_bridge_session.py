"""The OpenAI Agents bridges: the agent's conversation read from the memory service's thread
through the SDK's own ``Session`` (``bridges.session.MemorySession``), used only when the run
has no thread of its own; the SDK's saved ``RunState`` kept with the run, so a resume on another
worker continues it; and an agent called as a tool (``Agent.as_tool``) a governed child run."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from agents import Agent, SQLiteSession, Usage, function_tool

from tests.support.catalog import FakeCatalog
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Budget, Harness, Settings
from trellis.contracts import RunStatus
from trellis.harness import budget
from trellis.harness.adapters.openai_agents import _said_inside
from trellis.harness.bridges.session import HISTORY_LIMIT, MemorySession, ThreadReadOnly
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.runs import LocalRuns
from trellis.memory.models import MessageInfo


def said_to_model(model: ScriptedModel, turn: int) -> list[tuple[str, str]]:
    """The messages the model was given on ``turn``, as ``(role, text)``."""
    return [
        (item["role"], item["content"])
        for item in model.inputs[turn]
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    ]


async def test_the_thread_in_memory_is_the_agents_session(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel(["noted", "you said tides"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="sess")
    await agent.run("remember tides", user="u1", thread="t1")
    await memory_harness.writes.drain()  # the turn recorded (the harness writes in the background)
    await agent.run("what did I say?", user="u1", thread="t1")
    await memory_harness.writes.drain()
    # the earlier turn came from the memory service's thread, through the SDK's session
    told = said_to_model(model, 1)
    assert ("user", "remember tides") in told and ("assistant", "noted") in told
    assert told[-1] == ("user", "what did I say?")
    # the session owns the history: memory's window is not sent beside it
    assert [c.body.get("window") for c in memory_service.named("context")] == [False, False]
    # every turn is recorded once, by the harness (the session adds nothing of its own)
    texts = [m["content"] for m in memory_service.threads["t1"]]
    assert texts.count("remember tides") == 1 and texts.count("what did I say?") == 1


async def test_a_session_of_the_teams_own_is_kept(memory_harness: Harness) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="own")
    own = SQLiteSession("mine")
    for _ in range(2):
        await agent.run("hi", user="u1", thread="t1", framework_options={"session": own})
    asked = [i["content"] for i in await own.get_items() if i.get("role") == "user"]
    assert asked == ["hi", "hi"]  # the SDK kept its own session, as it was given


async def test_without_memory_the_sdk_has_no_session(memory_harness: Harness) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="off", without={"memory"})
    await agent.run("remember tides", user="u1", thread="t1")
    await agent.run("what did I say?", user="u1", thread="t1")
    assert said_to_model(model, 1) == [("user", "what did I say?")]


async def test_the_run_does_not_read_back_its_own_turns_and_the_thread_only_grows() -> None:
    def message(role: str, content: str, run: str) -> MessageInfo:
        return MessageInfo.model_validate(
            {
                "message_id": content,
                "role": role,
                "kind": "VISIBLE",
                "sequence": 1,
                "content": content,
                "agent_run_id": run,
            }
        )

    thread = [
        message("USER", "earlier", "run_0"),
        message("TOOL", "a tool's output", "run_0"),
        message("ASSISTANT", "said earlier", "run_0"),
        message("USER", "this run's question", "run_1"),  # an earlier attempt recorded it
    ]

    async def history(*, limit: int) -> list[MessageInfo]:
        assert limit == HISTORY_LIMIT
        return thread

    memory = SimpleNamespace(ctx=SimpleNamespace(history=history))
    runtime = SimpleNamespace(run_memory=memory, thread="t1", run_id="run_1")
    session = MemorySession(runtime)  # type: ignore[arg-type]
    assert session.session_id == "t1"
    assert await session.get_items() == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "said earlier"},
    ]
    assert await session.get_items(limit=1) == [{"role": "assistant", "content": "said earlier"}]
    assert await session.get_items(limit=0) == []
    await session.add_items([{"role": "user", "content": "x"}])  # the harness records it
    assert len(await session.get_items()) == 2
    with pytest.raises(ThreadReadOnly):
        await session.pop_item()
    with pytest.raises(ThreadReadOnly):
        await session.clear_session()


async def test_the_sdks_saved_run_state_resumes_on_another_worker() -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return "sent"

    runs = LocalRuns()  # agent-runs, as two workers share it
    model = ScriptedModel([("send", {"order": "o9"}), "sent it"])
    async with Harness(config=Settings(), runs=runs) as first:
        paused = await first.wrap(Agent(name="s", model=model, tools=[send]), id="s").run(
            "send o9", user="u1"
        )
    assert paused.interrupt is not None
    record = await runs.get(paused.run_id)
    assert record is not None and record.status is RunStatus.PAUSED
    async with Harness(config=Settings(), runs=runs) as second:
        agent = second.wrap(Agent(name="s", model=model, tools=[send]), id="s")
        finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "sent it"
    assert shipped == ["o9"] and len(model.inputs) == 2  # the saved state, not a re-run


# ---------------------------------------------------------------------- agent tools
async def test_an_agent_tools_spending_counts_toward_the_runs_budget(harness: Harness) -> None:
    inner = ScriptedModel(["o1 is fine"], usage=Usage(input_tokens=40, output_tokens=10))
    checker = Agent(name="checker", model=inner)
    model = ScriptedModel([("check", {"input": "check o1"}), ("check", {"input": "again"}), "ok"])
    target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Check.")])
    agent = harness.wrap(target, id="desk", budget=Budget(max_tokens=30))
    result = await agent.run("check o1", user="u")
    assert result.status is RunStatus.PARTIAL  # only the agent tool's own run spent
    assert result.error is not None and result.error.code == budget.BUDGET_EXCEEDED
    assert "50 spent" in result.error.message


async def test_an_agent_tools_own_ungoverned_tools_are_said_once(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    _said_inside.cache_clear()

    @function_tool
    def peek(order: str) -> str:
        return "fine"

    checker = Agent(name="checker", model=ScriptedModel(["ok"] * 2), tools=[peek])
    model = ScriptedModel(["done", "done"])
    target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Check.")])
    agent = harness.wrap(target, id="desk")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for _ in range(2):
            assert (await agent.run("hi", user="u")).status is RunStatus.SUCCESS
    [warning] = [r for r in caplog.records if "agent called as a tool" in r.getMessage()]
    assert warning.__dict__["tools"] == "peek" and warning.__dict__["agent_tool"] == "check"


async def test_an_agent_tools_call_asks_as_the_catalog_rules() -> None:
    checker = Agent(name="checker", model=ScriptedModel(["fine"], ids="checker_"))
    model = ScriptedModel([("check", {"input": "o1"}), "ok"])
    governance = Governance(FakeCatalog({"check": Rule(risk="irreversible")}))
    async with Harness(config=Settings(), governance=governance) as h:
        target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Check.")])
        paused = await h.wrap(target, id="desk").run("check o1", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
