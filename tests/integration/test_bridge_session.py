"""The OpenAI Agents bridges: the agent's conversation read from the memory service's thread
through the SDK's own ``Session`` (``bridges.session.MemorySession``), used only when the run
has no thread of its own; the SDK's saved ``RunState`` kept with the run, so a resume on another
worker continues it; and an agent called as a tool (``Agent.as_tool``) a governed child run."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from agents import Agent, SQLiteSession, Usage, function_tool

from tests.support.arm import with_governance
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
    kept = await memory_harness.conversation(user="u1", thread="t1")
    texts = [m["content"] for m in memory_service.threads[kept]]
    assert texts.count("remember tides") == 1 and texts.count("what did I say?") == 1


async def test_a_session_of_the_teams_own_is_kept(memory_harness: Harness) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="own")
    own = SQLiteSession("mine")
    for _ in range(2):
        await agent.run("hi", user="u1", thread="t1", framework_options={"session": own})
    items: list[dict[str, Any]] = [dict(i) for i in await own.get_items()]
    asked = [i["content"] for i in items if i.get("role") == "user"]
    assert asked == ["hi", "hi"]  # the SDK kept its own session, as it was given


async def test_without_memory_the_sdk_has_no_session_and_reads_the_run_stores_turns(
    memory_harness: Harness,
) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="off", without={"memory"})
    await agent.run("remember tides", user="u1", thread="t1")
    await agent.run("what did I say?", user="u1", thread="t1")
    # no session: the earlier turn is the run store's question and answer, as input items
    assert said_to_model(model, 1) == [
        ("user", "remember tides"),
        ("assistant", "one"),
        ("user", "what did I say?"),
    ]


async def test_the_run_does_not_read_back_its_own_turns_and_the_thread_only_grows() -> None:
    def message(role: str, content: str, run: str, **extra: Any) -> MessageInfo:
        return MessageInfo.model_validate(
            {
                "message_id": content,
                "role": role,
                "kind": extra.pop("kind", "VISIBLE"),
                "sequence": 1,
                "content": content,
                "agent_run_id": run,
                **extra,
            }
        )

    call = {"id": "c1", "name": "lookup", "arguments": {"q": "a"}}
    thread = [
        message("USER", "earlier", "run_0"),
        message("ASSISTANT", "", "run_0", tool_calls=[call]),
        message("TOOL", "found a", "run_0", kind="INTERNAL", tool_call_id="c1"),
        message("TOOL", "a result of no call", "run_0"),
        message("AGENT", "an internal step", "run_0", kind="INTERNAL"),
        message("ASSISTANT", "said earlier", "run_0"),
        message("USER", "this run's question", "run_1"),  # an earlier attempt recorded it
    ]

    async def history(*, limit: int, include_internal: bool) -> list[MessageInfo]:
        assert limit == HISTORY_LIMIT and include_internal
        return thread

    memory = SimpleNamespace(ctx=SimpleNamespace(history=history))
    runtime = SimpleNamespace(run_memory=memory, conversation="conv_t1", run_id="run_1")
    session = MemorySession(runtime)  # type: ignore[arg-type]
    assert session.session_id == "conv_t1"  # the run's conversation (Runtime.conversation)
    assert await session.get_items() == [
        {"role": "user", "content": "earlier"},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"q": "a"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "found a"},
        {"role": "assistant", "content": "said earlier"},
    ]
    assert await session.get_items(limit=1) == [{"role": "assistant", "content": "said earlier"}]
    assert await session.get_items(limit=0) == []
    await session.add_items([{"role": "user", "content": "x"}])  # the harness records it
    assert len(await session.get_items()) == 4
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


async def test_an_agent_tools_own_tools_are_governed_as_the_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The agent an agent tool runs (``Agent.as_tool``) runs as a governed copy of its own: its
    tools' calls are the run's — the catalog's rule asks a person about ``peek`` — and nothing
    is said about tools the harness cannot reach."""
    _said_inside.cache_clear()
    peeked: list[str] = []

    @function_tool
    def peek(order: str) -> str:
        peeked.append(order)
        return "fine"

    checker = Agent(
        name="checker",
        model=ScriptedModel([("peek", {"order": "o1"}), "checked"], ids="checker_"),
        tools=[peek],
    )
    model = ScriptedModel([("check", {"input": "o1"}), "done"])
    target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Check.")])
    governance = Governance(FakeCatalog({"peek": Rule(risk="irreversible")}))
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        async with with_governance(Harness(config=Settings()), governance) as h:
            paused = await h.wrap(target, id="desk").run("check o1", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "peek"
    assert peeked == []  # not run before a person approves it
    assert not [r for r in caplog.records if "agent called as a tool" in r.getMessage()]


async def test_an_agent_tools_call_asks_as_the_catalog_rules() -> None:
    checker = Agent(name="checker", model=ScriptedModel(["fine"], ids="checker_"))
    model = ScriptedModel([("check", {"input": "o1"}), "ok"])
    governance = Governance(FakeCatalog({"check": Rule(risk="irreversible")}))
    async with with_governance(Harness(config=Settings()), governance) as h:
        target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Check.")])
        paused = await h.wrap(target, id="desk").run("check o1", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None


#: a result the SDK's trimmer cuts in an older turn (over its 500 characters)
REPORT = "row " * 300


@function_tool
def report(name: str) -> str:
    """A long report."""
    return f"{name}: {REPORT}"


def outputs(model: ScriptedModel, call: int) -> list[str]:
    """The tool results the model was given on its ``call``-th call."""
    return [
        str(item["output"])
        for item in model.inputs[call]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


async def test_older_turns_large_results_are_trimmed_by_the_sdks_own_trimmer(
    memory_harness: Harness,
) -> None:
    """Where the harness gives the SDK the conversation (memory's thread as its session), the
    turns before the last two reach the model with their large tool results cut to the SDK's
    preview (its ``ToolOutputTrimmer``, the run's ``call_model_input_filter``): a long chat does
    not resend every old result whole on every call."""
    model = ScriptedModel([("report", {"name": "q3"}), "q3 done", "two", "three"])
    agent = memory_harness.wrap(
        Agent(name="s", model=model, tools=[report]),
        id="trim",
    )
    for asked in ("the q3 report", "and then?", "and now?"):
        await agent.run(asked, user="u1", thread="t1")
        await memory_harness.writes.drain()
    assert outputs(model, 1) == [f"q3: {REPORT}"]  # its own turn: whole
    assert outputs(model, 2) == [f"q3: {REPORT}"]  # one turn back: whole
    (older,) = outputs(model, 3)  # two turns back: the SDK's preview
    assert older.startswith("[Trimmed: report output") and len(older) < 400, older


@function_tool
def ledger(name: str) -> str:
    """A ledger: a table of many rows."""
    return "name,n,units\n" + "\n".join(f"{name},{n},{n * 3}" for n in range(500))


async def test_a_kept_result_cut_in_an_older_turn_still_says_where_it_is_kept(
    memory_harness: Harness,
) -> None:
    """A table the run keeps (its path said after its rows) reaches later turns cut to the
    SDK's preview, still naming its path: ``read_file`` and ``query_result`` read it whole."""
    model = ScriptedModel([("ledger", {"name": "q3"}), "q3 done", "two", "three"])
    agent = memory_harness.wrap(Agent(name="s", model=model, tools=[ledger]), id="kept")
    for asked in ("the q3 ledger", "and then?", "and now?"):
        await agent.run(asked, user="u1", thread="t1")
        await memory_harness.writes.drain()
    (whole,) = outputs(model, 1)
    path = whole.split("also saved at ", 1)[1].split()[0].rstrip(".")
    (older,) = outputs(model, 3)
    assert older.startswith("[Trimmed: ledger output") and len(older) < len(whole), older
    assert f"[kept whole at {path}: read_file or query_result reads it]" in older, older


async def test_a_session_or_filter_of_the_teams_own_gets_no_trimmer(
    memory_harness: Harness,
) -> None:
    """A team's own session keeps its own context, and a team's own input filter is theirs:
    the harness adds no trimmer to either."""
    from agents import RunConfig

    model = ScriptedModel([("report", {"name": "q3"}), "q3 done", "two", "three"])
    agent = memory_harness.wrap(Agent(name="s", model=model, tools=[report]), id="theirs")
    own = SQLiteSession("mine-trim")
    for asked in ("the q3 report", "and then?", "and now?"):
        await agent.run(asked, user="u1", thread="t2", framework_options={"session": own})
    assert outputs(model, 3) == [f"q3: {REPORT}"]
    model = ScriptedModel([("report", {"name": "q3"}), "q3 done", "two", "three"])
    agent = memory_harness.wrap(Agent(name="s", model=model, tools=[report]), id="filter")
    config = RunConfig(call_model_input_filter=lambda data: data.model_data)
    for asked in ("the q3 report", "and then?", "and now?"):
        await agent.run(asked, user="u1", thread="t3", framework_options={"run_config": config})
        await memory_harness.writes.drain()
    assert outputs(model, 3) == [f"q3: {REPORT}"]
