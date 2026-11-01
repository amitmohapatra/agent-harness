"""Kimi Code itself (``kimi acp``, the version ``compat.BINARIES`` pins) under the harness, its
model a planned endpoint on this machine (``tests.support.kimi``): no Moonshot key, nothing
sent anywhere. These pin what the adapter reads of Kimi — its tool names, its permission
requests, its sub-agents' and swarms' results, its usage log — so a Kimi that changes them
fails here. CI installs the pinned ``@moonshot-ai/kimi-code``; without it they are skipped
(``CI`` set: they fail)."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.kimi import (
    KIMI_BIN,
    PlannedEndpoint,
    kimi_target,
    needs_kimi,
    result_text,
    text_of,
)
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import compat
from trellis.harness.adapters.acp import KIMI_BUILTINS
from trellis.harness.adapters.claude import RESUMED_CALL
from trellis.harness.budget import Budget
from trellis.harness.sandbox import sandbox

pytestmark = needs_kimi
lookups: list[str] = []


@tool(side_effects="read")
def lookup(topic: str) -> str:
    """Look a topic up in the team's notes."""
    lookups.append(topic)
    return f"facts about {topic}"


@pytest.fixture
async def endpoint() -> AsyncIterator[PlannedEndpoint]:
    async with PlannedEndpoint().serving() as served:
        yield served


async def test_kimi_is_the_pinned_version() -> None:
    import asyncio

    assert KIMI_BIN is not None
    process = await asyncio.create_subprocess_exec(
        KIMI_BIN, "--version", stdout=asyncio.subprocess.PIPE
    )
    out, _ = await process.communicate()
    compat.require_binary("kimi-code", out.decode().strip(), command=KIMI_BIN)


def test_a_call_kimi_aborted_when_its_turn_was_stopped_did_not_run() -> None:
    """A run that pauses on a harness call stops Kimi's turn (``session/cancel``). When the
    cancel reaches Kimi before the call's "waiting for a person's approval", Kimi aborts the
    call itself: its result is 'The user manually interrupted "<tool>" …' (Kimi's
    ``abortedToolOutput``). That call did not run either: the planned model calls it again
    after the resume, as it does after the waiting result — counting it as run made the
    ``SEL-kimi-…`` matrix cell answer with the abort text (a timing-dependent failure)."""
    endpoint = PlannedEndpoint([("refund", {"order": "o1"})])
    aborted = (
        'The user manually interrupted "mcp__trellis__refund" (and anything else running at the '
        "same time). This was a deliberate user action, not a system error, timeout, or "
        "capacity limit. Do not retry automatically or guess at the cause — wait for the "
        "user's next instruction."
    )
    body = {
        "messages": [
            {"role": "user", "content": "refund o1"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": aborted},
            {"role": "user", "content": "This task was paused at your call to refund"},
        ]
    }
    assert endpoint.next(body) == ("mcp__trellis__refund", {"order": "o1"})


async def test_kimi_calls_a_harness_tool_and_answers(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    lookups.clear()
    endpoint.plan = [("lookup", {"topic": "tea"})]
    agent = arm(harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-tools"), [lookup])
    result = await agent.run("what about tea?", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == "Done. facts about tea" and lookups == ["tea"]
    offered = endpoint.offered()[0]
    # the harness's tools through its MCP server, beside Kimi's own — its swarm included
    assert "mcp__trellis__lookup" in offered and "mcp__trellis__read_file" in offered
    assert {"Bash", "Agent", "AgentSwarm", "Skill"} <= set(offered)
    assert set(offered) - {t for t in offered if t.startswith("mcp__")} <= set(KIMI_BUILTINS)
    assert "what about tea?" in endpoint.said()


async def test_bash_asks_a_person_and_the_resume_continues_kimis_session(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [("Bash", {"command": "echo resumed"})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-bash")
    paused = await agent.run("say it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "Bash"
    assert paused.interrupt.tool_call.args == {"command": "echo resumed"}
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == "Done. resumed"
    # the same session, told to call again: what came before is in it
    last = endpoint.requests[-1]["messages"]
    assert any(RESUMED_CALL.format(tool="Bash") in text_of(m.get("content")) for m in last)
    assert any("say it" in text_of(m.get("content")) for m in last)


async def test_a_rejected_bash_call_is_not_run_and_kimi_reads_why(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [("Bash", {"command": "rm -rf build"})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-reject")
    paused = await agent.run("clean", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", reviewer="u1", comment="not on main"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert "not on main" in endpoint.said() or "denied" in str(done.answer)
    assert not (tmp_path / "kimi-work" / "build").exists()


async def test_kimis_usage_log_is_the_contract_the_budget_reads(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi writes each model request's usage as a ``usage.record`` line in its session's
    ``agents/<agent>/wire.jsonl``: the shape the adapter reads (a change fails here). Every
    request counts; past the budget the turn stops and the run ends ``PARTIAL``."""
    plan = [("lookup", {"topic": "a"}), ("lookup", {"topic": "b"}), ("lookup", {"topic": "c"})]
    # each reply takes a moment: the log of one request is read before the next call
    async with PlannedEndpoint(plan, stall=1.0).serving() as endpoint:
        target = kimi_target(endpoint, tmp_path)
        agent = arm(harness.wrap(target, id="kimi-spend", budget=Budget(max_tokens=20)), [lookup])
        result = await agent.run("look three times", user="u1")
    [log] = list((tmp_path / "kimi-home").glob("sessions/*/*/agents/main/wire.jsonl"))
    records = [json.loads(line) for line in log.read_text().splitlines() if "usage.record" in line]
    assert records, "Kimi logs a usage.record per model request"
    for record in records:
        assert record["type"] == "usage.record" and record["agentId"] == "main"
        assert set(record["usage"]) >= {
            "inputOther",
            "output",
            "inputCacheRead",
            "inputCacheCreation",
        }
        assert record["usage"]["inputOther"] == 10 and record["usage"]["output"] == 5
    # each turn's end is logged too (a step limit's reason in it): the turn settles on it
    ends = [json.loads(line) for line in log.read_text().splitlines() if "turn.ended" in line]
    assert ends and all(e["type"] == "turn.ended" and e["agentId"] == "main" for e in ends)
    # 15 tokens a request: past 20 after the second, and the run stops there
    assert result.status is RunStatus.PARTIAL and result.error is not None
    assert result.error.code == "BUDGET_EXCEEDED"
    assert len(endpoint.requests) < 4  # never the answer after the third call


async def test_the_usage_kimi_logs_last_is_counted(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi writes its log a moment behind: the last request's usage (the answer's) counts."""
    from trellis.harness import budget

    counted: list[int] = []
    account = budget.account

    def counting(runtime: Any, **usage: Any) -> None:
        counted.append(usage["input_tokens"] + usage["output_tokens"])
        account(runtime, **usage)

    monkeypatch.setattr(budget, "account", counting)
    endpoint.plan = [("lookup", {"topic": "a"})]
    agent = arm(harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-count"), [lookup])
    result = await agent.run("look", user="u1")
    assert result.status is RunStatus.SUCCESS
    assert counted == [15] * len(endpoint.requests)


def _swarm_turns(body: dict[str, Any]) -> Any:
    """The main agent starts a swarm of two; each member answers its item; then the main
    agent answers with what the swarm said."""
    messages = body["messages"]
    tools = {t["function"]["name"] for t in body.get("tools") or ()}
    if "AgentSwarm" not in tools:  # a member: its task names its item
        said = " ".join(text_of(m.get("content")) for m in messages if m.get("role") == "user")
        return f"reviewed {re.search(r'Review (\S+)', said)[1]}"  # type: ignore[index]
    results = [m for m in messages if m.get("role") == "tool"]
    if not results:
        swarm = {
            "description": "review",
            "prompt_template": "Review {{item}}",
            "items": ["a.ts", "b.ts"],
            "subagent_type": "explore",
        }
        return ("AgentSwarm", swarm)
    return result_text(results[-1]["content"])


async def test_a_swarm_kimi_starts_runs_unchanged_each_member_a_child_run(
    harness: Harness, tmp_path: Path
) -> None:
    async with PlannedEndpoint(turns=_swarm_turns).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-swarm")
        done = await agent.run("review both", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert 'item="a.ts" outcome="completed">reviewed a.ts' in str(done.answer)
    found = [s async for s in harness.runs.iterate(parent_run_id=done.run_id)]
    kids = [await harness.runs.get(s.run_id) for s in found]
    assert sorted((k.input, k.status) for k in kids if k is not None) == [
        ("Review a.ts", RunStatus.SUCCESS),
        ("Review b.ts", RunStatus.SUCCESS),
    ]


def _subagent_turns(body: dict[str, Any]) -> Any:
    tools = {t["function"]["name"] for t in body.get("tools") or ()}
    results = [m for m in body["messages"] if m.get("role") == "tool"]
    if "Agent" not in tools:
        return "the sub-agent found tea"
    if not results:
        return ("Agent", {"description": "look", "prompt": "find tea", "subagent_type": "explore"})
    return "lead says: " + result_text(results[-1]["content"]).split("[summary]")[-1].strip()


async def test_a_sub_agent_kimi_starts_is_a_child_run(harness: Harness, tmp_path: Path) -> None:
    async with PlannedEndpoint(turns=_subagent_turns).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-lead")
        done = await agent.run("find tea", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert "the sub-agent found tea" in str(done.answer)
    [kid] = [s async for s in harness.runs.iterate(parent_run_id=done.run_id)]
    record = await harness.runs.get(kid.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS
    assert record.metadata["native_helper"]["name"] == "explore"


async def test_kimis_answer_streams(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-stream")
    events = [e async for e in agent.stream("hi", user="u1")]
    text = "".join(e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT)
    assert text == "Done. " and events[-1].type is RunEventType.RUN_FINISHED


async def test_kimi_through_bifrost_carries_the_runs_headers(tmp_path: Path) -> None:
    """With Bifrost serving the harness and a model named, Kimi's model calls go to Bifrost's
    OpenAI route with the virtual key, the deny-all MCP scope and the run's dimensions."""
    async with PlannedEndpoint().serving() as endpoint:
        settings = Settings(llm_gateway="bifrost", llm_base_url=endpoint.url, llm_api_key="vk-test")
        async with Harness(config=settings, gateway=False) as h:
            target = kimi_target(endpoint, tmp_path, model="moonshot/kimi-k2")
            env = {k: v for k, v in target.env.items() if not k.startswith("KIMI_MODEL")}
            from dataclasses import replace

            done = await h.wrap(replace(target, env=env), id="kimi-routed").run("x", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    [headers] = endpoint.headers
    assert headers["authorization"] == "Bearer vk-test"
    assert headers["x-bf-mcp-include-clients"] == "" and headers["x-bf-dim-agent"] == "kimi-routed"
    assert headers["x-bf-dim-user"]  # the run's user, as the gateway's dimensions keep it
    assert endpoint.requests[0]["model"] == "moonshot/kimi-k2"


async def test_a_sandboxed_kimi_runs_bash_and_writes_in_the_sandbox(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    sandboxes = FakeSandboxes()
    endpoint.plan = [
        ("Write", {"path": "/workspace/notes.txt", "content": "hello"}),
        ("Bash", {"command": "echo boxed"}),
    ]
    target = kimi_target(endpoint, tmp_path, sandbox=sandbox(sandboxes))
    agent = harness.wrap(target, id="kimi-boxed")
    paused = await agent.run("note it", user="u1")
    assert paused.interrupt is not None  # Bash asks, in the sandbox too
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    [made] = sandboxes.made()
    assert made not in sandboxes.boxes  # deleted with the run
    assert any("echo boxed" in c for c in sandboxes.commands)
    assert not (tmp_path / "kimi-work" / "notes.txt").exists()  # not on this machine


async def test_kimi_reads_the_runs_skills(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    store = SkillStore()
    store.put("sql-review", "Check every JOIN.", description="Reviews SQL queries for joins.")
    target = kimi_target(endpoint, tmp_path, **combine(store).kimi())
    done = await harness.wrap(target, id="kimi-skills").run("review", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    sent = json.dumps(endpoint.requests[0])
    assert "sql-review" in sent and "Reviews SQL queries for joins." in sent


async def test_a_kept_result_stays_out_of_the_workspace_and_pages_smaller_than_kimis_read(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi has its own ``Read``; the harness's ``read_file`` (its MCP server, over the run's
    journal) is kept for a large result, by measurement: the result is not written into the
    agent's workspace (the user's working directory, which another worker resuming the run
    does not have, and a second store beside the journal), and a page is at most
    ``PAGE_CHARS`` where Kimi's ``Read`` returns up to 100,000 characters at once by default."""
    from trellis.harness.journal import content_key
    from trellis.harness.tools import results

    big = "sku,units\n" + "\n".join(f"s{n},{n % 7}" for n in range(12_000))
    assert len(big) > results.MAX_CHARS

    @tool(side_effects="read")
    def rows() -> str:
        """Every row."""
        return big

    path = f"{results.PREFIX}/{content_key('result', big)}"
    work = tmp_path / "kimi-work"
    endpoint.plan = [("rows", {}), ("read_file", {"file_path": path})]
    agent = arm(harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-big"), [rows])
    result = await agent.run("count the rows", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    told = [text_of(m.get("content")) for m in endpoint.requests[-1]["messages"]]
    stub, page = next(t for t in told if path in t), told[-1]
    assert "Tool result too large" in stub  # the stub names the journal's path ...
    assert sorted(p.name for p in work.iterdir()) == []  # ... and nothing is in the workspace
    assert "  1  sku,units" in page and "more lines: read_file(" in page
    assert len(page) <= results.PAGE_CHARS + 200
    # the native path, for the record: the same rows as a workspace file, read by Kimi's Read
    (work / "rows.csv").write_text(big)
    endpoint.plan = [("Read", {"path": "rows.csv"})]
    native = await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-native").run(
        "count the rows", user="u1"
    )
    assert native.status is RunStatus.SUCCESS, native.error
    read = text_of(endpoint.requests[-1]["messages"][-1].get("content"))
    assert len(read) >= 4 * results.PAGE_CHARS  # one call: over four of the harness's pages


async def test_at_its_step_limit_kimi_is_asked_once_for_a_last_answer(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi ends a turn at its step limit (``loop_control.max_steps_per_turn``) as a finished
    turn with no answer: ACP's ``end_turn``, the reason only in its own log
    (``loop.max_steps_exceeded``). As a ``ReAct`` run at ``max_steps`` does, the run asks once
    more in the same session, told the limit is reached (``budget.LIMIT``): its answer is the
    run's, said as a ``step_limit`` warning."""
    from trellis.harness.budget import TURN_LIMIT

    lookups.clear()
    endpoint.plan = [("lookup", {"topic": "tea"}), ("lookup", {"topic": "coffee"})]
    target = kimi_target(endpoint, tmp_path, env={"KIMI_LOOP_MAX_STEPS_PER_TURN": "1"})
    agent = arm(harness.wrap(target, id="kimi-limit"), [lookup])
    events = [e async for e in agent.stream("tea and coffee?", user="u1")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    said = "".join(e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT)
    assert said.endswith("Done. facts about tea") and lookups == ["tea"]
    warned = [e.data["message"] for e in events if e.data.get("code") == "step_limit"]
    assert warned == ["stopped: the step limit (1) was reached"]
    # Kimi's session keeps it: told it holds for this turn only (a later turn calls tools again)
    assert TURN_LIMIT.format(limit=1) in endpoint.said()
