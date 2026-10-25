"""Native helpers — the agents a framework spawns itself (a Deep Agent's ``task`` sub-agents, an
OpenAI agent handed off to, Claude's ``agents=`` subagents) — each a governed child of the run:
its tool calls under the catalog's rules, its tokens on the run's budget, its own run record
naming the run (lineage), its turns in memory; and after a crash, the next worker ends the same
record (``bridges.helpers``)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agents import Agent
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions
from deepagents import create_deep_agent

from tests.integration.test_bridge_claude import machine
from tests.integration.test_restarts import Crash, crash_then_finish, workers  # noqa: F401
from tests.support.arm import arm, native_tools, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.chat_model import ScriptedChatModel
from tests.support.durable import SharedSaver
from tests.support.fake_claude_cli import STEP_TOKENS
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Budget, Harness, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness import budget
from trellis.harness.bridges.helpers import HELPER, TASK_HEADING
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.identity import AGENT_GROUP
from trellis.harness.runs import LocalRuns
from trellis.runs import Job, RunSummary

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
done: list[str] = []
crashes: list[BaseException] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()
    crashes[:] = [Crash()]


@tool(side_effects="write")
def refund(order: str) -> str:
    """Refund an order."""
    done.append(f"refund {order}")
    return f"refunded {order}"


@tool(side_effects="read")
def die() -> str:
    """The worker dies here, once."""
    if crashes:
        raise crashes.pop()
    return "fine"


#: the catalog's rule: a refund asks a person first
ASKS = Governance(FakeCatalog({"refund": Rule(risk="irreversible")}))


async def children(h: Harness, run_id: str) -> list[RunSummary]:
    return [r async for r in h.runs.iterate(parent_run_id=run_id)]


def told(service: FakeMemoryService, run_id: str) -> list[tuple[str, str]]:
    """What the memory service was told in ``run_id``'s own thread."""
    return [(m["role"], str(m["content"])) for m in service.threads.get(run_id, [])]


def a_full_child(service: FakeMemoryService, run_id: str, child_id: str, tools: list[str]) -> None:
    """``child_id`` is a full child run of ``run_id`` in memory: its context asked for in its
    own scope, its tool calls (``tools``) recorded under its run, its outcome recorded."""
    [asked] = [c for c in service.named("context") if c.scope.get("agent_run_id") == child_id]
    assert asked.scope["parent_agent_run_id"] == run_id
    recorded = [c for c in service.named("record_tool") if c.body["tool"] in tools]
    assert [c.scope["agent_run_id"] for c in recorded] == [child_id] * len(tools)
    [outcome] = [
        c.body
        for c in service.named("feedback")
        if c.body.get("source") == "system" and c.body["target_id"] == child_id
    ]
    assert outcome["verdict"] == "confirm"


@pytest.fixture
async def asks() -> AsyncIterator[Harness]:
    async with with_governance(Harness(config=Settings()), ASKS) as h:
        yield h


# ---------------------------------------------------------------------- Deep Agents' task
TASK = ("task", {"description": "refund o1", "subagent_type": "checker"})


async def deep(h: Harness, main: Any, sub: Any, *tools: Any, saver: Any = None) -> Any:
    given = await native_tools(h, *(tools or (refund,)), framework="deepagents")
    return create_deep_agent(
        model=main,
        tools=given,
        system_prompt="You hand refunds to the checker.",
        subagents=[
            {
                "name": "checker",
                "description": "Checks and refunds an order.",
                "system_prompt": "Refund the order.",
                "tools": given,
                "model": sub,
            }
        ],
        checkpointer=saver or SharedSaver(uuid.uuid4().hex),
    )


async def test_a_deep_agents_sub_agent_asks_as_the_catalog_rules_and_a_rejection_holds(
    asks: Harness,
) -> None:
    main = ScriptedChatModel(turns=[TASK, "the checker could not"])
    sub = ScriptedChatModel(turns=[("refund", {"order": "o1"}), "refund refused"])
    agent = asks.wrap(await deep(asks, main, sub), id="desk")
    paused = await agent.run("refund o1", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question.startswith("Approve refund?")
    ended = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="r")
    assert ended.status is RunStatus.SUCCESS and done == []  # denied: never run
    [child] = await children(asks, ended.run_id)
    assert child.status is RunStatus.SUCCESS


async def test_a_deep_agents_sub_agent_is_a_child_run_its_turns_in_memory(
    memory_service: FakeMemoryService,
) -> None:
    main = ScriptedChatModel(turns=[TASK, "o1 refunded"])
    sub = ScriptedChatModel(turns=[("refund", {"order": "o1"}), "refunded o1"])
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        result = await h.wrap(await deep(h, main, sub), id="desk").run("refund o1", user="u")
        assert result.status is RunStatus.SUCCESS and done == ["refund o1"]
        await h.writes.drain()
        [child] = await children(h, result.run_id)
        record = await h.runs.get(child.run_id)
    assert record is not None and record.parent_run_id == result.run_id
    assert record.metadata[HELPER] == {"framework": "langgraph", "kind": "task", "name": "checker"}
    assert record.input == "refund o1" and record.output == "refunded o1"
    assert told(memory_service, child.run_id) == [
        ("USER", "refund o1"),
        ("ASSISTANT", "refunded o1"),
    ]
    # the helper's scope is a sub-agent's: its parent the run, the run's tree its group
    [scope] = [
        c.scope for c in memory_service.named("messages") if c.scope["thread_id"] == child.run_id
    ]
    assert scope["agent_run_id"] == child.run_id
    assert scope["parent_agent_run_id"] == scope["agent_group_id"] == result.run_id
    assert record.metadata[AGENT_GROUP] == result.run_id
    # a full child run: given its own memory context before its task, its call recorded under
    # its run, its outcome recorded
    given = [m.content for m in sub.seen[0] if m.type == "human"]
    assert given == [f"{memory_service.context_text}\n\n{TASK_HEADING}refund o1"]
    a_full_child(memory_service, result.run_id, child.run_id, ["refund"])


async def test_a_deep_agents_sub_agent_that_fails_is_a_failed_child(
    memory_service: FakeMemoryService,
) -> None:
    main = ScriptedChatModel(turns=[TASK, "the checker failed"])
    sub = ScriptedChatModel(turns=[])  # it has no reply to give: its model call fails
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        result = await h.wrap(await deep(h, main, sub), id="desk").run("refund o1", user="u")
        [child] = await children(h, result.run_id)
        await h.writes.drain()
    assert child.status is RunStatus.ERROR
    [outcome] = [
        c.body
        for c in memory_service.named("feedback")
        if c.body.get("source") == "system" and c.body["target_id"] == child.run_id
    ]
    assert outcome["verdict"] == "reject" and "IndexError" in outcome["comment"]


async def test_a_deep_agents_sub_agents_tokens_count_toward_the_runs_budget(
    harness: Harness,
) -> None:
    main = ScriptedChatModel(turns=[TASK, ("refund", {"order": "o2"}), "done"])
    sub = ScriptedChatModel(turns=["checked o1"], spends=(50, 10))  # only the sub-agent spends
    graph = await deep(harness, main, sub)
    agent = harness.wrap(graph, id="desk", budget=Budget(max_tokens=30))
    result = await agent.run("refund o1, then o2", user="u")
    assert result.status is RunStatus.PARTIAL and done == []  # o2's refund was not started
    assert result.error is not None and result.error.code == budget.BUDGET_EXCEEDED
    assert "max_tokens 30 reached (60 spent)" in result.error.message


# ---------------------------------------------------------------------- Claude's subagents
def claude(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_SESSIONS": str(tmp_path),
            "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        },
        agents={"checker": AgentDefinition(description="Checks orders.", prompt="Check it.")},
        **fields,
    )


def built_ins(tmp_path: Path) -> list[str]:
    log = tmp_path / "builtins.jsonl"
    return (
        [json.loads(line)["tool"] for line in log.read_text().splitlines()] if log.exists() else []
    )


def subagent(builtin: str) -> dict[str, Any]:
    """A subagent's step: it calls ``builtin``, then answers."""
    return {
        "agent": "checker",
        "prompt": "check o1",
        "steps": [{"builtin": builtin, "args": {"file_path": "o1"}}, {"text": "o1 is fine"}],
    }


async def test_a_claude_subagents_built_in_asks_as_the_catalog_rules_and_a_rejection_holds(
    tmp_path: Path,
) -> None:
    governance = Governance(FakeCatalog({"Bash": Rule(risk="irreversible")}))
    async with with_governance(Harness(config=Settings()), governance) as h:
        agent = h.wrap(claude(tmp_path, [subagent("Bash"), {"text": "checked"}]), id="desk")
        paused = await agent.run("check o1", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "Bash"
        ended = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="r")
    assert ended.status is RunStatus.SUCCESS and built_ins(tmp_path) == []


async def test_a_claude_subagent_is_a_child_run_its_turns_in_memory(
    tmp_path: Path, memory_service: FakeMemoryService
) -> None:
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        options = claude(tmp_path, [subagent("Read"), {"text": "checked"}])
        options.env["FAKE_CLAUDE_BRIEFS"] = str(tmp_path / "briefs.jsonl")
        result = await h.wrap(options, id="desk").run("check o1", user="u")
        assert result.status is RunStatus.SUCCESS and result.answer == "checked"
        [child] = await children(h, result.run_id)
        record = await h.runs.get(child.run_id)
        await h.writes.drain()
    assert record is not None and record.status is RunStatus.SUCCESS
    assert record.metadata[HELPER] == {
        "framework": "claude_agent_sdk",
        "kind": "subagent",
        "name": "checker",
    }
    assert told(memory_service, child.run_id) == [
        ("USER", "check o1"),
        ("ASSISTANT", "o1 is fine"),
    ]
    # a full child run: the subagent's prompt carries its memory context (the PreToolUse
    # hook's update), its Read is recorded under its run (the CLI names the subagent)
    [brief] = [json.loads(line) for line in (tmp_path / "briefs.jsonl").read_text().splitlines()]
    assert brief["prompt"] == f"{memory_service.context_text}\n\n{TASK_HEADING}check o1"
    a_full_child(memory_service, result.run_id, child.run_id, ["Read"])


async def test_a_claude_subagents_tokens_count_toward_the_runs_budget(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounted: list[dict[str, Any]] = []
    account = budget.account

    def counted(runtime: Any, **spent: Any) -> None:
        accounted.append(spent)
        account(runtime, **spent)

    monkeypatch.setattr(budget, "account", counted)
    options = claude(tmp_path, [subagent("Read"), {"text": "checked"}])
    options.env["FAKE_CLAUDE_COST"] = "0.01"
    result = await harness.wrap(options, id="desk").run("check o1", user="u")
    assert result.status is RunStatus.SUCCESS
    # the query's ResultMessage, the subagent's steps among them, is what the run spent
    [spent] = accounted
    steps = round(spent["cost"] / 0.01)
    assert steps >= 3  # the Agent call, the subagent's Read, the answer
    tokens = STEP_TOKENS[0] * steps, STEP_TOKENS[1] * steps
    assert (spent["input_tokens"], spent["output_tokens"]) == tokens


async def test_a_claude_subagent_cut_by_a_crash_is_ended_by_the_next_worker(
    tmp_path: Path,
) -> None:
    script = [
        {
            "agent": "checker",
            "prompt": "check o1",
            "steps": [
                {"builtin": "Write", "args": {"file_path": "o1.txt"}},
                {"tool": "die", "args": {}},
                {"text": "o1 is fine"},
            ],
        },
        {"text": "checked"},
    ]
    store = LocalRuns()
    first, second = (Harness(config=Settings(), runs=store) for _ in range(2))
    try:
        dying = arm(first.wrap(machine(tmp_path, "one", script), id="desk"), [die])
        taking_over = arm(second.wrap(machine(tmp_path, "two", script), id="desk"), [die])
        handle = await dying.start("check o1", user="u")
        worker = first.worker([dying])
        claimed = await store.claim(worker.worker_id, [dying.id])
        assert claimed is not None
        job = Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        with pytest.RaisesGroup(Crash, flatten_subgroups=True):
            await dying.execute(job)
        held, _ = store._leases[handle.id]
        store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
        assert await second.worker([taking_over]).run_once()
        result = await handle.result(wait=10)
        [child] = await children(second, result.run_id)
    finally:
        await first.aclose()
        await second.aclose()
    assert result.status is RunStatus.SUCCESS and result.answer == "checked", result.error
    assert child.status is RunStatus.SUCCESS  # the same record, ended by the next worker


# ---------------------------------------------------------------------- OpenAI agent tools
async def test_an_agent_called_as_a_tool_is_a_child_run_under_the_catalogs_rules(
    memory_service: FakeMemoryService,
) -> None:
    inner = ScriptedModel(["o1 is fine"], ids="checker_")
    checker = Agent(name="checker", model=inner)
    tool_ = checker.as_tool(tool_name="check", tool_description="Check an order.")
    model = ScriptedModel(
        [("check", {"input": "check o1"}), ("check", {"input": "check o1"}), "ok"]
    )
    governance = Governance(FakeCatalog({"check": Rule(risk="irreversible")}))
    async with with_governance(
        Harness(config=Settings(), memory=memory_service.client()), governance
    ) as h:
        agent = h.wrap(Agent(name="desk", model=model, tools=[tool_]), id="desk")
        paused = await agent.run("check o1", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.question.startswith("Approve check?")  # the catalog's rule
        result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
        assert result.status is RunStatus.SUCCESS and result.answer == "ok"
        [child] = await children(h, result.run_id)
        record = await h.runs.get(child.run_id)
        await h.writes.drain()
    assert record is not None and record.status is RunStatus.SUCCESS
    assert record.metadata[HELPER] == {
        "framework": "openai_agents",
        "kind": "agent_tool",
        "name": "check",
    }
    assert told(memory_service, child.run_id) == [("USER", "check o1"), ("ASSISTANT", "o1 is fine")]
    # a full child run: the agent is given its memory context before its task
    assert inner.inputs[0][-1]["content"] == (
        f"{memory_service.context_text}\n\n{TASK_HEADING}check o1"
    )
    a_full_child(memory_service, result.run_id, child.run_id, [])


async def test_an_agent_tool_paused_inside_is_the_same_child_after_the_resume(
    asks: Harness,
) -> None:
    inner = ScriptedModel([*[("refund", {"order": "o1"})] * 2, "refunded o1"], ids="checker_")
    checker = Agent(
        name="checker",
        model=inner,
        tools=await native_tools(asks, refund, framework="openai_agents"),
    )
    model = ScriptedModel([*[("check", {"input": "refund o1"})] * 2, "ok"])
    target = Agent(name="desk", model=model, tools=[checker.as_tool("check", "Refund.")])
    agent = asks.wrap(target, id="desk")
    paused = await agent.run("refund o1", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    [waiting] = await children(asks, paused.run_id)
    assert waiting.status is RunStatus.RUNNING  # it waits with the run
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.status is RunStatus.SUCCESS and done == ["refund o1"]
    [child] = await children(asks, result.run_id)
    assert child.run_id == waiting.run_id and child.status is RunStatus.SUCCESS


async def test_an_agent_tool_that_fails_is_a_failed_child(harness: Harness) -> None:
    broken = Agent(name="checker", model=ScriptedModel([], ids="checker_"))  # no reply to give
    tool_ = broken.as_tool("check", "Check.", failure_error_function=None)
    model = ScriptedModel([("check", {"input": "check o1"}), "it failed"])
    result = await harness.wrap(Agent(name="desk", model=model, tools=[tool_]), id="d").run(
        "check o1", user="u"
    )
    assert result.status is RunStatus.SUCCESS and result.answer == "it failed"
    [child] = await children(harness, result.run_id)
    assert child.status is RunStatus.ERROR
