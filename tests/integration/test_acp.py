"""ACP agents (``adapters.acp``): the real ACP SDK on both sides — the harness's client, and a
scripted agent (``tests/support/fake_acp_agent.py``) in a process of its own, speaking ACP over
stdio as ``kimi acp`` does — permissions, sessions, the harness's MCP server, sub-agents, the
usage log and the sandbox included. ``tests/integration/test_kimi.py`` runs Kimi Code itself."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.sandbox import FakeSandboxes
from trellis import Deny, Harness, Hooks, Rewrite, current, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolCall
from trellis.harness import admission, compat
from trellis.harness.adapters import acp
from trellis.harness.adapters.acp import (
    EDITED,
    KIMI,
    KIMI_BUILTINS,
    REJECTED,
    ACPAgent,
    UsageLog,
    kimi,
    native_spec,
)
from trellis.harness.adapters.claude import RESUMED, RESUMED_CALL, RESUMED_DENIED
from trellis.harness.agent import NARROW_FROM
from trellis.harness.budget import Budget
from trellis.harness.history import ASKED_WITH
from trellis.harness.sandbox import sandbox

#: What Kimi reads its swarm's concurrency from.
SWARM_LIMIT = "KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"

FAKE = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


@tool(side_effects="read")
def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"facts about {topic}"


def fake(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ACPAgent:
    """The scripted agent as Kimi: its home and its record in ``tmp_path``."""
    env = {
        "FAKE_ACP_SCRIPT": json.dumps(script),
        "FAKE_ACP_RECORD": str(tmp_path / "record.json"),
        "KIMI_CODE_HOME": str(tmp_path / "home"),
        **fields.pop("env", {}),
    }
    return ACPAgent(
        command=(sys.executable, FAKE),
        env=env,
        cwd=fields.pop("cwd", str(tmp_path)),
        profile=fields.pop("profile", KIMI),
        **fields,
    )


async def children(h: Harness, parent_run_id: str) -> list[Any]:
    found = [s async for s in h.runs.iterate(parent_run_id=parent_run_id)]
    records = [await h.runs.get(s.run_id) for s in found]
    return sorted((r for r in records if r is not None), key=lambda r: r.run_id)


def recorded(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "record.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------- tools and answers
async def test_a_turn_answers_and_calls_harness_tools(harness: Harness, tmp_path: Path) -> None:
    refunds.clear()
    script = [{"tool": "lookup", "args": {"topic": "tea"}}, {"text": "found: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="lookups"), [lookup])
    result = await agent.run("tell me about tea", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "found: facts about tea"
    record = recorded(tmp_path)
    # read_file is listed from the start: the agent lists a server's tools once
    assert record["listed"] == [["lookup", "read_file", "query_result"]]
    [session] = record["sessions"]
    [server] = session["servers"]
    assert server["name"] == "trellis" and server["url"].startswith("http://127.0.0.1:")
    assert server["headers"][0]["name"] == "Authorization"
    # the harness tool needs no person: Kimi asks, the harness lets it through
    assert record["permissions"] == [{"title": "mcp__trellis__lookup", "answer": "approve_once"}]
    assert record["prompts"] == ["tell me about tea"]
    assert record["initialize"][0]["terminal"] is False  # no sandbox: no terminal of ours


async def test_the_harness_mcp_server_refuses_a_request_without_its_token() -> None:
    import httpx

    from trellis.harness.tools.convert import mcp as convert

    server = convert.convert([])
    async with convert.served(server) as served, httpx.AsyncClient() as client:
        refused = await client.post(served.url, json={})
        assert refused.status_code == 401


async def test_an_approval_stops_the_turn_and_a_resume_continues_the_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o2"}}, {"text": "done: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="refunds"), [refund])
    paused = await agent.run("refund o2", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and refunds == []
    assert recorded(tmp_path)["cancels"]  # the turn was stopped there
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: refunded o2"
    assert refunds == ["o2"]
    record = recorded(tmp_path)
    assert [s["how"] for s in record["sessions"]] == ["new", "resume"]
    assert record["prompts"][-1] == RESUMED_CALL.format(tool="mcp__trellis__refund")


class Died(BaseException):
    """What a worker process's death looks like to the code it ran (not an ``Exception``)."""


@tool(side_effects="read")
def dies(order: str) -> str:
    """A tool whose worker dies while it runs."""
    raise Died


async def test_a_worker_that_dies_in_a_harness_call_dies_in_the_run_not_in_its_server(
    harness: Harness, tmp_path: Path
) -> None:
    """A harness tool reaches an ACP agent through an MCP server the attempt serves: a
    ``BaseException`` that is not an error (the worker dying, ``SystemExit``) raised in its
    call ended the server's own task group, and the run went on as if the call had failed (a
    matrix cell, ``X15-kimi-…``: "the worker did not die"). It ends the attempt, as an
    in-process tool's does."""
    script = [{"tool": "dies", "args": {"order": "o5"}}, {"text": "went on: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="dies"), [dies])
    with pytest.raises(BaseException) as raised:
        await agent.run("go", user="u1")
    assert any(isinstance(e, Died) for e in _leaves(raised.value)), raised.value


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for e in error.exceptions for leaf in _leaves(e)]
    return [error]


async def test_a_denied_call_resumes_the_session_told_it_will_not_run(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o3"}}, {"text": "said: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="refunds-denied"), [refund])
    paused = await agent.run("refund o3", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and refunds == []
    told = RESUMED_DENIED.format(tool="mcp__trellis__refund")
    assert recorded(tmp_path)["prompts"][-1] == told and done.answer == f"said: {told}"


@tool(side_effects="read")
async def pick_warehouse() -> str:
    """Ask which warehouse ships."""
    runtime = current()
    assert runtime is not None
    return f"ships from {await runtime.ask('Which warehouse?')}"


async def test_a_question_a_tool_asked_resumes_the_session_without_naming_a_call(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"tool": "pick_warehouse", "args": {}}, {"text": "{last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="shipping"), [pick_warehouse])
    asked = await agent.run("where does it ship from?", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is None
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="Leeds", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "ships from Leeds"
    assert recorded(tmp_path)["prompts"][-1] == RESUMED


async def test_a_harness_tool_error_reaches_the_agent(harness: Harness, tmp_path: Path) -> None:
    @tool(side_effects="read")
    def broken(why: str) -> str:
        """Fail."""
        raise ValueError(why)

    script = [{"tool": "broken", "args": {"why": "no stock"}}, {"text": "read: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="broken"), [broken])
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.SUCCESS and "no stock" in str(result.answer)


# --------------------------------------------------------------------------- its own tools
async def test_its_own_tools_are_governed_and_recorded(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """``Bash`` asks a person (irreversible); approved, it runs once, after a pause — the reads
    run without asking; each call it ran is recorded with its result."""
    script = [
        {"builtin": "Read", "args": {"path": "orders.csv"}, "output": "o1,o2"},
        {"builtin": "Bash", "args": {"command": "ls"}, "ask": True, "output": "a.txt"},
        {"builtin": "Grep", "args": {"pattern": "x"}, "fails": "bad pattern"},
        {"text": "done: {last}"},
    ]
    agent = memory_harness.wrap(fake(tmp_path, script), id="ops")
    asked = await agent.run("tidy up", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.tool == "Bash"
    assert asked.interrupt.tool_call.args == {"command": "ls"}
    assert asked.interrupt.question == "Approve Bash? Bash is irreversible."
    done = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: bad pattern"
    assert recorded(tmp_path)["prompts"][-1] == RESUMED_CALL.format(tool="Bash")
    await memory_harness.writes.drain()
    records = [
        (c.body["tool"], c.body["status"]) for c in memory_service.calls if c.name == "record_tool"
    ]
    assert records == [("Read", "ok"), ("Bash", "ok"), ("Grep", "error")]  # each once


async def test_a_rejected_call_ends_the_turn_and_the_agent_reads_why(
    harness: Harness, tmp_path: Path
) -> None:
    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            return Deny("no shell on Fridays") if call.tool == "Bash" else None

    script = [
        {"builtin": "Bash", "args": {"command": "rm -rf /"}, "ask": True},
        {"text": "said: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="guarded", hooks=[Guard()])
    result = await agent.run("clean", user="u1")
    assert result.status is RunStatus.SUCCESS
    told = recorded(tmp_path)["prompts"][-1]
    assert told.startswith(REJECTED.format(tool="Bash", why="")[:20])
    assert "no shell on Fridays" in told and result.answer == f"said: {told}"


async def test_an_edited_call_is_made_again_with_its_arguments_and_runs(
    harness: Harness, tmp_path: Path
) -> None:
    class Narrow(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            if call.tool == "Edit" and call.args["path"] != "b.txt":
                return Rewrite({"path": "b.txt"})
            return None

    script = [{"builtin": "Edit", "args": {"path": "a.txt"}, "ask": True}, {"text": "{last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="narrowed", hooks=[Narrow()])
    result = await agent.run("fix", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "Edit ran"
    record = recorded(tmp_path)
    assert record["prompts"][-1] == EDITED.format(tool="Edit", args='{"path": "b.txt"}')
    assert [p["answer"] for p in record["permissions"]] == ["reject", "approve_once"]


async def test_its_question_is_the_runs_question(harness: Harness, tmp_path: Path) -> None:
    script = [
        {"question": "Which colour?", "options": ["red", "blue"]},
        {"text": "picked {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="asker")
    asked = await agent.run("paint", user="u1")
    assert asked.interrupt is not None and asked.interrupt.question == "Which colour?"
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="blue", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "picked blue"


async def test_a_question_answered_with_no_option_it_offers_is_skipped(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"question": "Which?", "options": ["a"]}, {"text": "{last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="skipper")
    asked = await agent.run("x", user="u1")
    assert asked.interrupt is not None
    answer = acp._chosen([], "reject_once")
    assert answer.outcome.outcome == "cancelled"  # no option of that kind: the question dismissed


# --------------------------------------------------------------------------- helpers
async def test_a_sub_agent_is_a_child_run_and_its_calls_are_governed(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {
            "agent": {"description": "look", "prompt": "find tea", "subagent_type": "explore"},
            "asks": {"builtin": "Bash", "args": {"command": "ls"}},
            "result": "agent_id: agent-0\nstatus: completed\n\n[summary]\ntea found",
        },
        {"text": "done"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="lead")
    asked = await agent.run("find tea", user="u1")
    # the sub-agent's own Bash call asks: governed, though ACP shows no update of it
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.tool == "Bash"
    done = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    kids = await children(harness, done.run_id)
    assert [c.metadata["native_helper"]["name"] for c in kids] == ["explore"]
    assert {c.status for c in kids} == {RunStatus.SUCCESS}


async def test_a_failed_sub_agent_ends_its_child_run_failed(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"agent": {"prompt": "find"}, "result": "", "fails": True},
        {"text": "gave up"},
    ]
    done = await harness.wrap(fake(tmp_path, script), id="lead2").run("find", user="u1")
    assert done.status is RunStatus.SUCCESS
    [child] = await children(harness, done.run_id)
    assert child.status is RunStatus.ERROR and child.metadata["native_helper"]["name"] == "coder"


async def test_each_swarm_member_is_a_child_run(harness: Harness, tmp_path: Path) -> None:
    """Kimi's ``AgentSwarm`` passes through as it is: each item a child run, ended as its
    member's result says (a member Kimi reports nothing of fails)."""
    result = (
        "<agent_swarm_result>\n<summary>completed: 1</summary>\n"
        '<subagent agent_id="agent-0" item="a.ts" outcome="completed">fine</subagent>\n'
        '<subagent agent_id="agent-1" item="b.ts" outcome="failed">broke</subagent>\n'
        "</agent_swarm_result>"
    )
    swarm = {
        "description": "review",
        "prompt_template": "Review {{item}}",
        "items": ["a.ts", "b.ts", "c.ts"],
    }
    script = [{"swarm": swarm, "result": result}, {"text": "reviewed"}]
    done = await harness.wrap(fake(tmp_path, script), id="swarm").run("review", user="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "reviewed"
    kids = sorted(await children(harness, done.run_id), key=lambda c: c.input)
    assert [(c.input, c.status) for c in kids] == [
        ("Review a.ts", RunStatus.SUCCESS),
        ("Review b.ts", RunStatus.ERROR),
        ("Review c.ts", RunStatus.ERROR),
    ]
    assert "helper" in native_spec("AgentSwarm", KIMI).tags


async def test_a_swarm_runs_as_many_at_once_as_the_harness_admits_and_is_on_the_tree(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi is told the adaptive limit of child runs as its swarm's concurrency (the agent's own
    ``env`` wins); each member is on the run's task tree."""
    monkeypatch.delenv(SWARM_LIMIT, raising=False)
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(start=3, machine=None))
    result = (
        "<agent_swarm_result>\n"
        '<subagent agent_id="agent-0" item="a.ts" outcome="completed">fine</subagent>\n'
        "</agent_swarm_result>"
    )
    swarm = {"description": "review", "prompt_template": "Review {{item}}", "items": ["a.ts"]}
    script = [{"swarm": swarm, "result": result}, {"text": "reviewed"}]
    agent = harness.wrap(fake(tmp_path, script), id="swarm")
    events = [e async for e in agent.stream("review", user="u1")]
    assert recorded(tmp_path)["started"][0]["env"][SWARM_LIMIT] == "3"
    tree = [e.data for e in events if e.type is RunEventType.CUSTOM]
    tree = [d for d in tree if d["name"].startswith("helper_")]
    assert [(d["name"], d.get("status")) for d in tree] == [
        ("helper_started", None),
        ("helper_finished", "success"),
    ]
    assert tree[0]["framework"] == "acp" and tree[0]["kind"] == "subagent"
    own = harness.wrap(fake(tmp_path, [{"text": "ok"}], env={SWARM_LIMIT: "7"}), id="own")
    await own.run("x", user="u1")
    assert recorded(tmp_path)["started"][-1]["env"][SWARM_LIMIT] == "7"


# --------------------------------------------------------------------------- sessions
async def test_a_session_the_agent_no_longer_holds_is_run_again_from_its_prompt(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o4"}}, {"text": "done"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="moved"), [refund])
    paused = await agent.run("refund o4", user="u1")
    assert paused.interrupt is not None
    for kept in (tmp_path / "home").rglob("state.json"):
        kept.unlink()  # resumed on a machine whose agent never held the session
    events: list[Any] = []
    record = await harness.runs.get(paused.run_id)
    assert record is not None
    resumed, resolution = await agent._resolution(
        paused.interrupt.interrupt_id, "approve", None, "u1", tenant=record.tenant_id
    )
    done = await agent._continue(resumed, resolution, events.append)
    assert done.status is RunStatus.SUCCESS and refunds == ["o4"]
    [warning] = [e for e in events if e.data.get("code") == "acp_session"]
    assert "was not resumed" in warning.data["message"]
    assert recorded(tmp_path)["prompts"][-1] == "refund o4"  # from the prompt, the journal's


async def test_an_agent_without_session_resume_loads_the_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [
        {"text": "looking"},
        {"tool": "refund", "args": {"order": "o5"}},
        {"text": "done"},
    ]
    target = fake(tmp_path, script, env={"FAKE_ACP_RESUME": "0"})
    agent = arm(harness.wrap(target, id="loader"), [refund])
    paused = await agent.run("refund o5", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    # the loaded session's history replayed while it loads is not this turn's answer
    assert done.status is RunStatus.SUCCESS and done.answer == "done"
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new", "load"]


async def test_the_session_mode_is_set_and_one_that_asks_nothing_refused(
    harness: Harness, tmp_path: Path
) -> None:
    await harness.wrap(fake(tmp_path, [{"text": "ok"}], mode="plan"), id="planner").run(
        "x", user="u1"
    )
    assert recorded(tmp_path)["modes"] == ["plan"]
    with pytest.raises(ConfigurationError, match="a mode that approves on its own"):
        kimi(mode="yolo")


async def test_kimis_built_in_tools_do_not_count_toward_narrowing_the_runs_own(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Kimi's built-ins are the framework's, not the run's: the tools the hints narrow from
    (``NARROW_FROM``) are the run's own. Counted with them, a run of a few tools of its own
    narrowed, and the ``trellis`` server did not list the tool a question needed."""

    def make(i: int) -> Any:
        def look(key: str) -> str:
            return f"t{i}"

        return tool(look, name=f"t{i}", side_effects="read")

    memory_service.candidates = ["t1"]
    own = NARROW_FROM - len(KIMI_BUILTINS) + 1  # with the built-ins, past the threshold
    script = [{"tool": "t0", "args": {"key": "a"}}, {"text": "{last}"}]
    agent = arm(
        memory_harness.wrap(fake(tmp_path, script), id="own"), [make(i) for i in range(own)]
    )
    result = await agent.run("look a up", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "t0"
    [listed] = recorded(tmp_path)["listed"]
    assert {f"t{i}" for i in range(own)} <= set(listed)


# --------------------------------------------------------------------------- the context
async def test_the_context_goes_before_the_prompt_marked_off_from_it(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """ACP's prompt has no place for background but its blocks: the context goes first, marked
    off as every adapter's is where it shares the user's message (``history.ASKED_WITH``).
    Unmarked, a model read the context as the user's message and the question as a note."""
    await memory_harness.wrap(fake(tmp_path, [{"text": "ok"}]), id="ctx").run(
        "reach me?", user="u1"
    )
    [prompt] = recorded(tmp_path)["prompts"]
    opened, closed = ASKED_WITH.split("{context}")
    assert prompt.startswith(opened + memory_service.context_text)
    assert prompt.endswith(f"{closed}\nreach me?")
    [request] = memory_service.named("context")
    available = request.body["tools"]["available"]
    assert {"Bash", "Read", "AgentSwarm"} <= set(available)  # Kimi's own tools: its tool set


async def test_streaming_carries_the_agents_text(harness: Harness, tmp_path: Path) -> None:
    agent = harness.wrap(fake(tmp_path, [{"plan": True}, {"text": "hello there"}]), id="s")
    events = [e async for e in agent.stream("hi", user="u1")]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert deltas == ["hello ", "there"]
    assert events[-1].type is RunEventType.RUN_FINISHED


# --------------------------------------------------------------------------- limits
async def test_usage_is_read_from_the_agents_log_and_the_budget_stops_the_turn(
    harness: Harness, tmp_path: Path
) -> None:
    """Every model request Kimi logs — its sub-agents' too — counts; past the budget the turn is
    cancelled and the run ends PARTIAL."""
    script = [
        {"usage": {"inputOther": 100, "output": 50, "inputCacheRead": 10}},
        {"usage": {"inputOther": 100, "output": 50}, "agent": "agent-0"},
        {"usage": {"garbled": True}},
        {"stall": 2.0},
        {"text": "never"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="spender", budget=Budget(max_tokens=200))
    result = await agent.run("spend", user="u1")
    assert result.status is RunStatus.PARTIAL and result.error is not None
    assert result.error.code == "BUDGET_EXCEEDED"
    assert recorded(tmp_path)["cancels"]


async def test_acps_own_usage_counts_when_the_agent_gives_it(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACP agent the harness keeps no usage log of counts the usage ACP's answer to a prompt
    carries; Kimi's own log, read as the turn runs, is its count (the answer's is not added)."""
    from trellis.harness import budget

    counted: list[dict[str, Any]] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (counted.append(kw), account(run, **kw))
    )
    usage = {"totalTokens": 37, "inputTokens": 20, "outputTokens": 10, "cachedReadTokens": 7}
    script = [{"prompt_usage": usage}, {"text": "ok"}]
    agent = harness.wrap(fake(tmp_path, script, profile=None), id="u")
    assert (await agent.run("x", user="u1")).status is RunStatus.SUCCESS
    assert counted == [{"model": "acp", "input_tokens": 20, "output_tokens": 10, "cached": 7}]
    counted.clear()
    (tmp_path / "k").mkdir()
    kimi_like = harness.wrap(fake(tmp_path / "k", script), id="k")
    assert (await kimi_like.run("x", user="u1")).status is RunStatus.SUCCESS
    assert counted == []  # Kimi's count is its log's


async def test_a_usage_line_still_being_written_is_read_next_time(tmp_path: Path) -> None:
    folder = tmp_path / "sessions" / "wd" / "s1" / "agents" / "main"
    folder.mkdir(parents=True)
    log = folder / "wire.jsonl"
    log.write_bytes(b'{"type": "usage.record", "usage": {"output": 1}}\n{"type": "usage.re')
    usage = UsageLog(tmp_path, "s1")
    assert len(usage.records()) == 1
    with log.open("ab") as file:
        file.write(b'cord", "usage": {"output": 2}}\n[]\n')
    assert [r["usage"]["output"] for r in usage.records()] == [2]


async def test_a_stalled_step_ends_the_run(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness import runtime

    monkeypatch.setattr(runtime, "STALL_SECONDS", 0.5)
    agent = harness.wrap(fake(tmp_path, [{"stall": 30}, {"text": "late"}]), id="stalled")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == "MODEL_TIMEOUT"


async def test_the_run_time_limit_stops_the_agent(harness: Harness, tmp_path: Path) -> None:
    script = [{"builtin": "Bash", "args": {"command": "sleep"}, "wait": 30}, {"text": "late"}]
    agent = harness.wrap(fake(tmp_path, script), id="slow", run_timeout=1.0)
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.TIMEOUT


async def test_a_cancelled_run_stops_the_agent(harness: Harness, tmp_path: Path) -> None:
    script = [{"builtin": "Bash", "args": {"command": "sleep"}, "wait": 30}, {"text": "late"}]
    agent = harness.wrap(fake(tmp_path, script), id="cancelled")
    handle = asyncio.create_task(agent.run("x", user="u1"))
    while "sessions" not in recorded(tmp_path):  # noqa: ASYNC110 - another process says so
        await asyncio.sleep(0.05)
    [run] = [s async for s in harness.runs.iterate(agent_id="cancelled")]
    await agent.cancel(run.run_id, reason="enough")
    result = await handle
    assert result.status is RunStatus.CANCELLED


waited: list[str] = []
waiting = asyncio.Event()


@tool(side_effects="read")
async def wait_long(seconds: int) -> str:
    """Wait a number of seconds, then say so."""
    waiting.set()
    await asyncio.sleep(seconds)
    waited.append("done")
    return "waited"


async def test_a_cancelled_run_cancels_the_harness_call_under_way(
    harness: Harness, tmp_path: Path
) -> None:
    waited.clear()
    waiting.clear()
    # under the second the server is given to stop in: the call is cancelled, not waited for
    script = [{"tool": "wait_long", "args": {"seconds": 1}}, {"text": "late"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="waits"), [wait_long])
    handle = asyncio.create_task(agent.run("x", user="u1"))
    await asyncio.wait_for(waiting.wait(), 30)
    [run] = [s async for s in harness.runs.iterate(agent_id="waits")]
    await agent.cancel(run.run_id, reason="enough")
    result = await handle
    assert result.status is RunStatus.CANCELLED
    await asyncio.sleep(1.5)  # past when the call would have ended
    assert waited == []  # cut short with the run, never finished after it


# --------------------------------------------------------------------------- failures
async def test_a_process_that_ends_mid_turn_fails_the_run_with_its_stderr(
    harness: Harness, tmp_path: Path
) -> None:
    agent = harness.wrap(fake(tmp_path, [{"crash": "out of memory"}]), id="crash")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "out of memory" in result.error.message and "exit 3" in result.error.message


async def test_a_turn_ended_at_the_step_limit_is_asked_once_more(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi ends a turn at its step limit with no answer, the reason only in its log: the run
    asks once more in the same session, and says so on the stream."""
    script = [{"step_limit": 1}, {"text": "the answer at the limit"}]
    agent = harness.wrap(fake(tmp_path, script), id="limited")
    events = [e async for e in agent.stream("work", user="u1")]
    warned = [e.data["message"] for e in events if e.data.get("code") == "step_limit"]
    assert warned == ["stopped: the step limit (1) was reached"]
    assert events[-1].data.get("result") == "the answer at the limit"


async def test_a_refusal_fails_the_run(harness: Harness, tmp_path: Path) -> None:
    result = await harness.wrap(fake(tmp_path, [{"stop": "refusal"}]), id="no").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "refusal" in result.error.message


async def test_an_agent_that_needs_a_login_fails_saying_so(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [{"text": "x"}], env={"FAKE_ACP_AUTH": "missing"})
    result = await harness.wrap(target, id="login").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Authentication required" in result.error.message


@pytest.mark.parametrize("version", ["2.3.3", "2.0.9", ""])
async def test_a_kimi_outside_the_tested_range_is_refused(
    harness: Harness, tmp_path: Path, version: str
) -> None:
    target = fake(tmp_path, [{"text": "x"}], env={"FAKE_ACP_VERSION": version})
    result = await harness.wrap(target, id=f"v{version}").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    low, high = compat.BINARIES["kimi-code"]
    assert f"tested with kimi-code>={low},<{high}" in result.error.message
    assert compat.INSTALL["kimi-code"] in result.error.message


async def test_an_acp_agent_without_a_profile_runs_unchecked(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [{"text": "plain"}], profile=None, env={"FAKE_ACP_VERSION": "9"})
    result = await harness.wrap(target, id="plain").run("x", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "plain"


# --------------------------------------------------------------------------- options
async def test_framework_options_add_env_and_servers_and_replace_the_rest(
    harness: Harness, tmp_path: Path
) -> None:
    server = {"name": "notes", "command": sys.executable, "args": ["-c", "pass"]}
    agent = harness.wrap(
        fake(tmp_path, [{"text": "ok"}], env={"KIMI_ONE": "1"}),
        id="opts",
        framework_options={"env": {"KIMI_TWO": "2"}, "mcp_servers": [server]},
    )
    await agent.run("x", user="u1", framework_options={"mode": "default"})
    record = recorded(tmp_path)
    env = record["started"][0]["env"]
    assert env["KIMI_ONE"] == "1" and env["KIMI_TWO"] == "2"
    assert [s["name"] for s in record["sessions"][0]["servers"]] == ["notes"]
    assert record["modes"] == ["default"]
    with pytest.raises(ConfigurationError, match="an ACP agent's run takes"):
        harness.wrap(fake(tmp_path, []), id="bad", framework_options={"command": ["x"]})
    with pytest.raises(ConfigurationError, match="the harness's own server"):
        harness.wrap(
            fake(tmp_path, []), id="bad2", framework_options={"mcp_servers": [{"name": "trellis"}]}
        )
    with pytest.raises(ConfigurationError, match="the command that starts it"):
        ACPAgent(command=())


async def test_the_teams_http_and_sse_servers_go_to_the_agent(
    harness: Harness, tmp_path: Path
) -> None:
    servers = (
        {"type": "http", "name": "web", "url": "http://127.0.0.1:9/mcp", "headers": []},
        {"type": "sse", "name": "old", "url": "http://127.0.0.1:9/sse", "headers": []},
    )
    await harness.wrap(fake(tmp_path, [{"text": "ok"}], mcp_servers=servers), id="srv").run(
        "x", user="u1"
    )
    assert [s["name"] for s in recorded(tmp_path)["sessions"][0]["servers"]] == ["web", "old"]


# --------------------------------------------------------------------------- the sandbox
async def test_a_sandboxed_run_works_in_the_sandbox_and_says_reads_are_not_isolated(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandboxes = FakeSandboxes()
    monkeypatch.setenv("FAKE_SECRET_TOKEN", "do-not-pass")
    script = [
        {"builtin": "Write", "args": {"path": "/w/notes.txt", "content": "a\nb\nc\n"}},
        {"builtin": "Read", "args": {"path": "/w/notes.txt", "line": 2, "limit": 1}},
        {"builtin": "Read", "args": {"path": "/w/notes.txt"}},
        {"builtin": "Bash", "args": {"command": "echo hello"}, "ask": True, "limit": 3},
        {"builtin": "Bash", "args": {"command": "sleep 9"}, "kill": True},
        {"text": "{last}"},
    ]
    target = fake(tmp_path, script, sandbox=sandbox(sandboxes), env={"FAKE_ACP_RAW": "1"})
    agent = harness.wrap(target, id="boxed")
    events = [e async for e in agent.stream("work", user="u1")]
    [paused] = [e for e in events if e.type is RunEventType.INTERRUPT]
    warned = [e for e in events if e.data.get("code") == "acp_sandbox"]
    assert warned and "reads are not isolated" in warned[0].data["message"]
    done = await agent.resume(paused.data["interrupt_id"], "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    [made] = sandboxes.made()
    assert made not in sandboxes.boxes  # deleted with its run
    assert any(c.startswith("mkdir -p ") for c in sandboxes.commands)
    assert "echo hello" in sandboxes.commands and "sleep 9" in sandboxes.killed
    record = recorded(tmp_path)
    assert record["initialize"][0]["terminal"] is True
    started = record["started"][0]
    assert "FAKE_SECRET_TOKEN" not in started["env"]  # no secret of the host's
    assert started["env"]["HOME"].endswith("/home")
    assert started["env"]["KIMI_CODE_HOME"].endswith("/home/.agent")
    assert started["cwd"].endswith("/work")
    ran = [t for t in record["terminal"] if "exit" in t]
    # the scripted sandbox echoes the env it was given (FAKE=1); its tail, cut to the limit
    assert ran[0] == {"exit": 0, "signal": None, "output": "E=1", "truncated": True}
    assert ran[1]["signal"] == "SIGKILL"
    assert not Path(started["cwd"]).exists()  # noqa: ASYNC240  # the run's directories go with it


async def test_without_a_sandbox_the_agents_file_requests_are_refused(tmp_path: Path) -> None:
    from acp import RequestError

    attempt = acp.Attempt(acp.ACPAdapter(), None, fake(tmp_path, []))  # type: ignore[arg-type]
    with pytest.raises(RequestError):
        await attempt.read_text_file("s", "/etc/passwd")


async def test_a_process_that_ends_before_it_answers_fails_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [], env={"FAKE_ACP_CRASH": "initialize"})
    result = await harness.wrap(target, id="dead").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "cannot start" in result.error.message and "exit 4" in result.error.message


async def test_an_error_the_agent_answers_the_prompt_with_fails_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    result = await harness.wrap(fake(tmp_path, [{"error": "bad state"}]), id="err").run(
        "x", user="u1"
    )
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Internal error" in result.error.message


async def test_calls_made_together_and_updates_of_unknown_calls(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Reads made together are each recorded (their structured output as JSON); an update of a
    call the agent never started is passed over; the team's own MCP tool joins the tool set."""
    script = [
        {"orphan": True},
        {"parallel": ["Read", "Glob"]},
        {"builtin": "mcp__crm__find", "args": {"q": "ada"}},
        {"text": "{last}"},
    ]
    agent = memory_harness.wrap(fake(tmp_path, script), id="together")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.SUCCESS
    await memory_harness.writes.drain()
    outputs = {
        c.body["tool"]: c.body["output"] for c in memory_service.calls if c.name == "record_tool"
    }
    assert outputs["Read"] == '{"read": "Read"}' and "Glob" in outputs
    assert "mcp__crm__find" in {s.name for s in await agent.adapter.native(agent.target)}


async def test_a_kept_result_is_read_through_the_harness_server(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"tool": "gone", "args": {}},
        {"tool": "read_file", "args": {"file_path": "/large_tool_results/x"}},
        {"text": "{last}"},
    ]
    result = await arm(harness.wrap(fake(tmp_path, script), id="pager"), [lookup]).run(
        "x", user="u1"
    )
    assert result.status is RunStatus.SUCCESS and "/large_tool_results/x" in str(result.answer)


async def test_an_acp_agent_harness_knows_nothing_of_runs_its_sub_agent_unmapped(
    harness: Harness, tmp_path: Path
) -> None:
    """Without a profile the harness knows no sub-agent tool: the call is the agent's own."""
    script = [{"agent": {"prompt": "go"}, "result": "went"}, {"text": "{last}"}]
    target = fake(tmp_path, script, profile=None)
    done = await harness.wrap(target, id="generic").run("x", user="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "went"
    assert await children(harness, done.run_id) == []


async def test_a_sandboxed_command_that_fails_or_outlives_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    from trellis.contracts import ToolError

    sandboxes = FakeSandboxes()
    sandboxes.crash = ToolError("the sandbox is gone", source="tools")
    script = [
        {"builtin": "Read", "args": {"path": "/w/none.txt"}},
        {"builtin": "Bash", "args": {"command": "crash"}},
        {"builtin": "Bash", "args": {"command": "sleep 30"}},
        {"text": "never"},
    ]
    target = fake(
        tmp_path, script, sandbox=sandbox(sandboxes), profile=None, env={"FAKE_ACP_RAW": "1"}
    )
    result = await harness.wrap(target, id="boxed2", run_timeout=3.0).run("x", user="u1")
    assert result.status is RunStatus.TIMEOUT
    record = recorded(tmp_path)
    [crashed] = [t for t in record["terminal"] if "exit" in t]
    assert crashed["signal"] == "SIGKILL" and "the sandbox is gone" in crashed["output"]
    assert "HOME" in record["started"][0]["env"] and "KIMI_CODE_HOME" in record["started"][0]["env"]
    assert "sleep 30" in sandboxes.killed  # the command still running when the run ended


# --------------------------------------------------------------------------- skills
async def test_the_runs_skills_are_in_kimis_skills_folder_and_its_login_is_linked(
    harness: Harness, tmp_path: Path
) -> None:
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    store = SkillStore()
    store.put(
        "sql-review",
        "Check every JOIN.",
        description="Reviews SQL.",
        files={"rules.md": "No SELECT *."},
    )
    usual = tmp_path / "home"
    usual.mkdir()
    (usual / "config.toml").write_text("default_model = 'k2'\n")
    target = fake(tmp_path, [{"skills": True}, {"text": "ok"}], **combine(store).kimi())
    result = await harness.wrap(target, id="skilled").run("review", user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    [seen] = recorded(tmp_path)["skills"]
    assert "Check every JOIN." in seen["files"]["sql-review/SKILL.md"]
    assert seen["files"]["sql-review/rules.md"] == "No SELECT *."
    assert seen["linked"] == ["config.toml"]  # its own configuration; no credentials kept here
    started = recorded(tmp_path)["started"][0]["env"]["KIMI_CODE_HOME"]
    gone = not Path(started).exists()  # noqa: ASYNC240
    assert started != str(usual) and gone  # the run's, removed with it
