"""``framework_options=``: the framework's own options for its run call, passed through unchanged
— the agent's (``h.wrap``), a run's own over them (``run``/``stream``/``start``), kept with the
run's record for its later attempts; refused where there is no framework run call, or where
the harness sets the option itself."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultDeny
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph

from tests.support.adapters import BUILDERS, CLI
from tests.support.planned import Call
from trellis import Harness, tool
from trellis.contracts import ConfigurationError, RunOutcome, RunStatus
from trellis.harness import agent as agent_module
from trellis.harness.runs import LocalRuns

#: Each framework's own limit on a run's steps, below what one tool call needs, and what its
#: error says; and one with room enough.
TIGHT: Final[dict[str, tuple[dict[str, Any], str]]] = {
    "langgraph": ({"recursion_limit": 3}, "Recursion limit of 3 reached"),
    "deepagents": ({"recursion_limit": 3}, "Recursion limit of 3 reached"),
    "openai_agents": ({"max_turns": 1}, "Max turns (1) exceeded"),
    "claude_agent_sdk": ({"max_turns": 1}, "Reached maximum number of turns"),
}
ROOMY: Final[dict[str, dict[str, Any]]] = {
    "langgraph": {"recursion_limit": 50},
    "deepagents": {"recursion_limit": 50},
    "openai_agents": {"max_turns": 20},
    "claude_agent_sdk": {"max_turns": 20},
}
PLAN: Final[list[Call]] = [("lookup", {"topic": "tides"})]


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"facts about {topic}"


@tool(side_effects="irreversible")
async def confirm(step: str) -> str:
    """Confirm a step."""
    return f"confirmed {step}"


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
@pytest.mark.parametrize("framework", list(TIGHT))
async def test_the_options_reach_the_frameworks_run_call_a_runs_over_its_agents(
    harness: Harness, framework: str, streaming: bool, tmp_path: Path
) -> None:
    tight, said = TIGHT[framework]
    target, tools = await BUILDERS[framework](harness, [lookup], tmp_path, PLAN)
    agent = harness.wrap(target, id="limited", tools=tools, framework_options=tight)
    failed = await agent.run("tides?", user="u")  # the agent's own limit
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert said in failed.error.message
    roomy = ROOMY[framework]
    if streaming:
        events = [e async for e in agent.stream("tides?", user="u", framework_options=roomy)]
        assert events[-1].outcome is RunOutcome.SUCCESS, events[-1]
        run_id = events[-1].run_id
    else:
        result = await agent.run("tides?", user="u", framework_options=roomy)
        assert result.answer == "Done. facts about tides", result
        run_id = result.run_id
    record = await harness.runs.get(run_id)
    assert record is not None and record.metadata == {"framework_options": roomy}
    first = await harness.runs.get(failed.run_id)
    assert first is not None and first.metadata == {}  # the agent's are not the run's


async def test_a_queued_runs_options_are_kept_for_its_worker_and_its_resume(
    harness: Harness, tmp_path: Path
) -> None:
    """One turn: the first attempt pauses on the approval in it; the resumed attempt, which
    re-runs the turn and needs a second, ends where the framework's limit says."""
    plan: list[Call] = [("confirm", {"step": "go"}), ("lookup", {"topic": "tides"})]
    target, tools = await BUILDERS["openai_agents"](harness, [confirm, lookup], tmp_path, plan)
    agent = harness.wrap(target, id="queued", tools=tools)
    with pytest.raises(ConfigurationError, match=r"as JSON, and 'context' is not"):
        await agent.start("tides?", user="u", framework_options={"context": object()})
    handle = await agent.start("tides?", user="u", framework_options={"max_turns": 1})
    worker = harness.worker([agent])
    assert await worker.run_once()
    paused = await handle.status()
    assert paused.status is RunStatus.PAUSED and paused.awaiting is not None
    assert paused.metadata == {"framework_options": {"max_turns": 1}}
    await agent.resume(paused.awaiting.interrupt_id, "approve", reviewer="ops")
    assert await worker.run_once()
    ended = await handle.status()
    assert ended.status is RunStatus.ERROR and ended.error is not None
    assert "Max turns (1) exceeded" in ended.error.message


async def test_an_object_reaches_the_first_attempt_and_the_record_keeps_the_json(
    harness: Harness, tmp_path: Path
) -> None:
    class Seen(BaseCallbackHandler):
        def __init__(self) -> None:
            self.tags: list[str] = []

        def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
            self.tags.extend(kwargs.get("tags") or [])

    seen = Seen()
    target, _ = await BUILDERS["langgraph"](harness, [lookup], tmp_path, PLAN)
    agent = harness.wrap(target, id="graph")
    given = {"callbacks": [seen], "tags": ["nightly"]}
    result = await agent.run("tides?", user="u", framework_options=given)
    assert result.status is RunStatus.SUCCESS and "nightly" in seen.tags
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.metadata == {"framework_options": {"tags": ["nightly"]}}


async def test_a_graphs_configurable_keys_pass_and_the_harnesss_thread_wins(
    harness: Harness,
) -> None:
    def answer(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        given = config.get("configurable") or {}
        return {"messages": [AIMessage(content=f"{given['shop']} on {given['thread_id']}")]}

    graph = StateGraph(MessagesState).add_node("answer", answer)
    agent = harness.wrap(graph.add_edge(START, "answer").compile(), id="shop")
    configurable = {"shop": "north", "thread_id": "theirs"}
    result = await agent.run(
        "hi", user="u", thread="t1", framework_options={"configurable": configurable}
    )
    assert result.answer == "north on t1"


async def test_claudes_options_merge_with_what_the_harness_sets(
    harness: Harness, tmp_path: Path
) -> None:
    """The run's ``system_prompt`` and ``mcp_servers`` go to the CLI beside the harness's own
    server; its ``can_use_tool`` is asked after governance, as the target's would be."""
    asked: list[str] = []

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        asked.append(name)
        return PermissionResultDeny(message="not today")

    script = [
        {"builtin": "Read", "args": {"file_path": "a.txt"}},
        {"tool": "lookup", "args": {"topic": "tides"}},
        {"text": "read: {last}"},
    ]
    record = tmp_path / "cli.json"
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps(script),
        "FAKE_CLAUDE_RECORD": str(record),
        "FAKE_CLAUDE_SESSIONS": str(tmp_path),
    }
    agent = harness.wrap(ClaudeAgentOptions(cli_path=CLI, env=env), id="cc", tools=[lookup])
    erp = {"type": "stdio", "command": "erp-mcp", "args": []}
    given = {"system_prompt": "Be brief.", "can_use_tool": own, "mcp_servers": {"erp": erp}}
    result = await agent.run("tides?", user="u", framework_options=given)
    assert result.answer == "read: facts about tides" and asked == ["Read"]
    cli = json.loads(record.read_text())
    assert cli["system_prompt"] == "Be brief."
    servers = json.loads(cli["mcp_config"])["mcpServers"]
    assert servers["erp"] == erp and servers["trellis"]["type"] == "sdk"


@pytest.mark.parametrize(
    ("framework", "options", "said"),
    [
        ("function", {"max_turns": 1}, "which a function target does not have"),
        ("react", {"max_turns": 1}, "RunnableConfig keys: 'max_turns' is none of"),
        ("langgraph", {"max_turns": 1}, "RunnableConfig keys: 'max_turns' is none of"),
        ("langgraph", {"configurable": "north"}, "configurable is a dict"),
        ("openai_agents", {"hooks": None}, "'hooks': the harness gives Runner.run"),
        ("openai_agents", {"recursion_limit": 3}, "Runner.run takes no such argument"),
        ("claude_agent_sdk", {"recursion_limit": 3}, "no ClaudeAgentOptions field"),
        ("claude_agent_sdk", {"resume": "s1"}, "'resume' \\(the harness resumes"),
        ("claude_agent_sdk", {"mcp_servers": {"trellis": {}}}, "the harness's own server"),
    ],
)
async def test_options_a_framework_cannot_take_are_refused_before_any_run(
    harness: Harness, framework: str, options: dict[str, Any], said: str, tmp_path: Path
) -> None:
    target, tools = await BUILDERS[framework](harness, [lookup], tmp_path, PLAN)
    with pytest.raises(ConfigurationError, match=said):
        harness.wrap(target, id="refused", tools=tools, framework_options=options)
    agent = harness.wrap(target, id="plain", tools=tools)
    with pytest.raises(ConfigurationError, match=said):
        await agent.run("tides?", user="u", framework_options=options)
    with pytest.raises(ConfigurationError, match=said):
        await agent.start("tides?", user="u", framework_options=options)


async def test_an_openai_sandbox_agent_runs_wrapped_with_its_run_config_as_an_option(
    harness: Harness,
) -> None:
    """``SandboxAgent`` needs ``RunConfig(sandbox=...)`` on ``Runner.run``: an object given on
    ``h.wrap`` (it lives in code, rebuilt in every process) — and its shell runs in it."""
    from agents import RunConfig
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    from tests.support.planned import PlannedModel

    plan: list[Call] = [("lookup", {"topic": "tides"}), ("exec_command", {"cmd": "echo boxed"})]
    target = SandboxAgent(name="box", instructions="Work.", model=PlannedModel(plan))
    refused = harness.wrap(target, id="bare", tools=[lookup])
    failed = await refused.run("tides?", user="u")
    assert failed.error is not None and "RunConfig(sandbox=...)" in failed.error.message
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    agent = harness.wrap(
        target, id="boxed", tools=[lookup], framework_options={"run_config": config}
    )
    result = await agent.run("tides?", user="u")
    assert result.status is RunStatus.SUCCESS and "boxed" in str(result.answer), result


async def test_a_schedule_takes_what_start_takes_and_every_fired_run_carries_it(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, tools = await BUILDERS["openai_agents"](harness, [lookup], tmp_path, PLAN)
    agent = harness.wrap(target, id="nightly", tools=tools, timeout=60)
    with pytest.raises(ConfigurationError, match="as JSON"):
        await agent.schedule("manual", "x", on_behalf_of="ada", framework_options={"c": object()})
    schedule = await agent.schedule(
        "0 0 1 1 *",
        "tides?",
        on_behalf_of="ada",
        timeout=30,
        without={"judges"},
        framework_options={"max_turns": 1},
        priority=5,
        concurrency_key="nightly",
    )
    assert schedule.metadata == {"without": ["judges"], "framework_options": {"max_turns": 1}}
    assert (schedule.timeout_seconds, schedule.priority, schedule.concurrency_key) == (
        30,
        5,
        "nightly",
    )
    store = harness.runs
    assert isinstance(store, LocalRuns)
    store._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    assert await harness.worker([agent]).run_once()
    fired = [r for r in store._runs.values() if r.metadata.get("schedule_id")]
    assert len(fired) == 1
    run = fired[0]
    assert run.metadata == {**schedule.metadata, "schedule_id": schedule.schedule_id}
    assert (run.timeout_seconds, run.priority, run.concurrency_key) == (30, 5, "nightly")
    assert run.status is RunStatus.ERROR and run.error is not None  # its options reached it
    assert "Max turns (1) exceeded" in run.error.message
    monkeypatch.setattr(agent_module, "SCHEDULES_QUEUE", False)  # contracts before 0.6.1
    with pytest.raises(ConfigurationError, match=r"schedule\(priority=\) needs trellis-contracts"):
        await agent.schedule("manual", "y", on_behalf_of="ada", priority=1)
    assert (await agent.schedule("manual", "y", on_behalf_of="ada")).priority == 0
