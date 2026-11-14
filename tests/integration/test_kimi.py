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

from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.kimi import (
    KIMI_BIN,
    PlannedEndpoint,
    kimi_target,
    needs_kimi,
    result_text,
    text_of,
)
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, Hooks, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import compat
from trellis.harness.adapters.acp import KIMI_BUILTINS, KIMI_OUTLIVES_THE_TURN
from trellis.harness.adapters.claude import RESUMED_CALL
from trellis.harness.budget import Budget
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.sandbox import sandbox
from trellis.harness.tools.bridge import TURN_STALLED

pytestmark = needs_kimi
lookups: list[str] = []
#: A command that writes (a file in Kimi's working directory): Kimi asks, and so a person.
WRITES = "touch made.txt && echo resumed"


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
    own = set(KIMI_BUILTINS) | set(KIMI_OUTLIVES_THE_TURN)  # listed by Kimi, turned off
    assert set(offered) - {t for t in offered if t.startswith("mcp__")} <= own
    assert "what about tea?" in endpoint.said()


async def test_a_schedule_kimi_could_not_keep_is_refused_by_kimi_itself(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's scheduled prompt (``CronCreate``) fires only while Kimi runs, and the harness ends
    Kimi with each attempt: nothing would ever fire it. Kimi's own switch turns scheduling off
    (``KIMI_DISABLE_CRON``): the call says so and the model can tell the user, where Kimi
    scheduled it and answered as if it would run (the break-and-fix gate: bf2-cantdo-02). The
    cron tools are not of the run's tool set (no hint offers them)."""
    schedule = {"cron": "0 9 * * 1", "prompt": "send the top 5 SKUs", "recurring": True}
    endpoint.plan = [("CronCreate", schedule)]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-cron")
    result = await agent.run("every Monday, send me the top 5 SKUs", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert "Cron scheduling is disabled" in str(result.answer)
    assert "nextFireAt" not in str(result.answer)  # nothing was scheduled
    own = {t.name for t in await agent.adapter.native(agent.target)}
    assert not own & set(KIMI_OUTLIVES_THE_TURN)


async def test_kimis_tower_mode_is_off_whatever_the_agents_own_env_says(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's tower mode starts its workers in its ``auto`` permission mode (they ask about
    nothing: the harness could govern none of their calls). Turned on in the agent's ``env``,
    Kimi offers its ``Tower*`` tools even over ACP; under the harness its own switch keeps the
    mode off (Kimi reads its environment's word first), and the run says why."""
    target = kimi_target(endpoint, tmp_path, env={"KIMI_CODE_EXPERIMENTAL_TOWER": "1"})
    events = [e async for e in harness.wrap(target, id="kimi-tower").stream("hi", user="u1")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    assert not [t for t in endpoint.offered()[0] if t.startswith("Tower")]
    assert [e.data.get("code") for e in events if e.data.get("name") == "warning"] == ["acp_env"]


async def test_bash_asks_a_person_and_the_resume_continues_kimis_session(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [("Bash", {"command": WRITES})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-bash")
    paused = await agent.run("say it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "Bash"
    assert paused.interrupt.tool_call.args == {"command": WRITES}
    assert paused.interrupt.question == "Approve Bash? Kimi Code asks before it runs this call."
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == "Done. resumed"
    assert (tmp_path / "kimi-work" / "made.txt").exists()
    # the same session, told to call again: what came before is in it
    last = endpoint.requests[-1]["messages"]
    assert any(RESUMED_CALL.format(tool="Bash") in text_of(m.get("content")) for m in last)
    assert any("say it" in text_of(m.get("content")) for m in last)


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
async def test_a_command_that_only_reads_runs_without_asking_a_person(
    harness: Harness, tmp_path: Path, streamed: bool
) -> None:
    """Kimi asks before every ``Bash`` command; one that only reads, the harness lets run as
    Kimi's own read tools run: no pause."""
    async with PlannedEndpoint(streamed=streamed).serving() as endpoint:
        endpoint.plan = [("Bash", {"command": "echo resumed | wc -l; which kimi 2>&1"})]
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-reads")
        done = await agent.run("count", user="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer is not None and done.answer.startswith("Done. 1")


async def test_kimi_still_asks_about_a_command_its_config_allows_and_so_a_person_is_asked(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi Code 2.1.1 reads ``[permission]`` rules in its ``config.toml`` but, over ACP, its
    agent never loads them into the rules its policy checks: a command a team's rule allows is
    still asked about, so the harness asks a person. Pinned: a Kimi that applies them passes
    the team's allowlist through, and this fails."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(
        '[[permission.rules]]\ndecision = "allow"\npattern = "Bash(touch *)"\n'
    )
    endpoint.plan = [("Bash", {"command": "touch made.txt"})]
    paused = await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-allowed").run(
        "make it", user="u1"
    )
    assert paused.status is RunStatus.PAUSED
    assert not (tmp_path / "kimi-work" / "made.txt").exists()


@pytest.mark.parametrize(
    "rule", [Rule(risk="irreversible"), Rule(risk="write", approve_when="true")], ids=str
)
async def test_a_team_that_wants_every_shell_command_approved_says_so_in_the_catalog(
    endpoint: PlannedEndpoint, tmp_path: Path, rule: Rule
) -> None:
    endpoint.plan = [("Bash", {"command": "ls"})]
    async with with_governance(Harness(), Governance(FakeCatalog({"Bash": rule}))) as h:
        paused = await h.wrap(kimi_target(endpoint, tmp_path), id="kimi-strict").run(
            "look", user="u1"
        )
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        call = paused.interrupt.tool_call
        assert call is not None and (call.tool, call.args) == ("Bash", {"command": "ls"})


async def test_a_file_kimi_saves_in_its_outputs_folder_is_kept_for_its_user(
    endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi is told that ``outputs/`` of its working folder is the user's; a file it writes
    there (its own ``Bash``, asked about and approved) moves to the file store on success."""
    from trellis.harness.sandbox import DELIVERS_HERE

    save = "mkdir -p outputs && printf 'a,b\\n1,2\\n' > outputs/sales.csv"
    endpoint.plan = [("Bash", {"command": save})]
    folders: list[str] = []

    class Ended(Hooks):
        async def on_run_end(self, run: Any, result: Any) -> None:
            folders.append(run.files_folder())

    h = Harness(config=Settings(files_url=f"memory://kimi-out-{id(tmp_path)}"))
    agent = h.wrap(kimi_target(endpoint, tmp_path), id="kimi-saver", hooks=[Ended()])
    paused = await agent.run("save the sales", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert await h.files.read(folders[-1], "sales.csv") == b"a,b\n1,2\n"
    assert not (tmp_path / "kimi-work" / "outputs" / "sales.csv").exists()
    assert DELIVERS_HERE in endpoint.said()
    await h.aclose()


async def test_a_bash_call_whose_arguments_stream_pauses_with_its_command(
    harness: Harness, tmp_path: Path
) -> None:
    """A provider streams a call's arguments after its name (Anthropic's, OpenAI's): Kimi asks
    permission before it starts the call, so the arguments are only in the call's updates."""
    async with PlannedEndpoint(streamed=True).serving() as endpoint:
        endpoint.plan = [("Bash", {"command": WRITES})]
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-bash-streamed")
        paused = await agent.run("say it", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "Bash"
        assert paused.interrupt.tool_call.args == {"command": WRITES}
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "Done. resumed"


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
    # every turn's end is logged too (Kimi's durable ``TurnEnded``; a step limit is reason
    # "failed" with error.code loop.max_steps_exceeded): the turn settles on it. Its
    # ``agent.turn.ended`` lines are another record (``outcome``), not read
    ends = [
        entry
        for entry in map(json.loads, log.read_text().splitlines())
        if entry.get("type") == "turn.ended"
    ]
    assert ends, "Kimi logs each turn's end as a turn.ended line"
    for end in ends:
        assert end["agentId"] == "main" and isinstance(end["turnId"], int)
        assert end["reason"] in {"completed", "cancelled", "failed", "blocked"}, end
    assert ends[-1]["reason"] == "cancelled"  # the budget stopped the turn
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


def _resuming_turns(body: dict[str, Any]) -> Any:
    """A swarm of two ``explore`` members; then the main agent goes on with the first in a
    swarm of its resumed member alone (``resume_agent_ids``), then with the second through
    ``Agent(resume=...)``; then it answers."""
    messages = body["messages"]
    tools = {t["function"]["name"] for t in body.get("tools") or ()}
    users = [text_of(m.get("content")) for m in messages if m.get("role") == "user"]
    if "AgentSwarm" not in tools:  # a member, by the last thing it was told (not a reminder)
        last = [u for u in users if not u.startswith("<system-reminder>")][-1]
        return {"continue a": "went on", "carry on b": "carried on"}.get(last) or (
            f"reviewed {re.search(r'Review (\S+)', last)[1]}"  # type: ignore[index]
        )
    results = [result_text(m["content"]) for m in messages if m.get("role") == "tool"]
    if not results:
        swarm = {
            "description": "review",
            "prompt_template": "Review {{item}}",
            "items": ["a.ts", "b.ts"],
            "subagent_type": "explore",
        }
        return ("AgentSwarm", swarm)
    ids = dict(re.findall(r'agent_id="([^"]+)" item="([^"]+)"', results[0]))
    by_item = {item: agent for agent, item in ids.items()}
    if len(results) == 1:
        again = {"description": "again", "resume_agent_ids": {by_item["a.ts"]: "continue a"}}
        return ("AgentSwarm", again)
    if len(results) == 2:
        more = {"description": "more", "prompt": "carry on b", "resume": by_item["b.ts"]}
        return ("Agent", more)
    return "done"


async def test_resumed_members_and_agents_kimi_goes_on_with_are_child_runs_by_name(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi goes on with sub-agents it ran: a swarm's ``resume_agent_ids`` and
    ``Agent(resume=...)``. Each is a child run of its own, named as Kimi started it (its
    ``explore`` type, which a resumed agent keeps — not ``coder``, the name of none), its task
    the prompt it goes on with, ended with what it said."""
    async with PlannedEndpoint(turns=_resuming_turns).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-resumes")
        done = await agent.run("review both, then go on", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    found = [s async for s in harness.runs.iterate(parent_run_id=done.run_id)]
    kids = [await harness.runs.get(s.run_id) for s in found]
    named = sorted(
        (k.input, k.metadata["native_helper"]["name"], k.status) for k in kids if k is not None
    )
    assert named == [
        ("Review a.ts", "explore", RunStatus.SUCCESS),
        ("Review b.ts", "explore", RunStatus.SUCCESS),
        ("carry on b", "explore", RunStatus.SUCCESS),
        ("continue a", "explore", RunStatus.SUCCESS),
    ]
    outputs = {k.input: str(k.output) for k in kids if k is not None}
    assert outputs["continue a"] == "went on" and "carried on" in outputs["carry on b"]


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


#: How long each of a turn's two sub-agents takes to answer (its one model reply).
MEMBER_SECONDS = 1.5
spans: list[tuple[float, float]] = []


@tool(side_effects="read")
async def slow_lookup(topic: str) -> str:
    """Look a topic up, slowly."""
    import asyncio
    import time

    began = time.monotonic()
    await asyncio.sleep(0.5)
    spans.append((began, time.monotonic()))
    return f"facts about {topic}"


def _together_turns(body: dict[str, Any]) -> Any:
    """Two ``Agent`` calls in one message; then two of the harness's reads in one message;
    then the answer."""
    tools = {t["function"]["name"] for t in body.get("tools") or ()}
    if "Agent" not in tools:
        return "found"
    done = len([m for m in body["messages"] if m.get("role") == "tool"])
    if done == 0:
        return [
            (
                "Agent",
                {"description": f"look {x}", "prompt": f"find {x}", "subagent_type": "explore"},
            )
            for x in ("a", "b")
        ]
    if done == 2:
        return [("slow_lookup", {"topic": x}) for x in ("a", "b")]
    return "done"


def _member(body: dict[str, Any]) -> bool:
    return "Agent" not in {t["function"]["name"] for t in body.get("tools") or ()}


async def test_kimi_runs_its_own_sub_agents_together_and_the_harness_tools_one_by_one(
    harness: Harness, tmp_path: Path
) -> None:
    """What the matrix's F25r reason says (N3): Kimi runs a turn's calls itself, by what each
    touches (its ``ToolScheduler``): its two ``Agent`` calls of one message run together (both
    sub-agents ask the model before either has its answer), while the harness's tools — MCP
    tools, which declare nothing they touch — run one after the other."""
    spans.clear()
    endpoint = PlannedEndpoint(turns=_together_turns, stall=lambda b: MEMBER_SECONDS * _member(b))
    async with endpoint.serving():
        agent = arm(
            harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-together"), [slow_lookup]
        )
        done = await agent.run("look both ways", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    first, second = [at for at, body in endpoint.arrived if _member(body)]
    assert second - first < MEMBER_SECONDS  # the second asked before the first was answered
    (_, ended), (began, _) = sorted(spans)
    assert began >= ended  # the harness's second read began after its first ended


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


async def test_a_conversations_next_turn_continues_kimis_session_when_its_runs_have_skills(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """A run given skills works in a home of its own, removed with the run; Kimi's sessions
    stay in its usual home (linked into the run's), so the conversation's next turn resumes
    the session the last one ended in: the earlier turn is in what the model is sent, and no
    session is reported lost."""
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    store = SkillStore()
    store.put("sql-review", "Check every JOIN.", description="Reviews SQL queries for joins.")
    target = kimi_target(endpoint, tmp_path, **combine(store).kimi())
    agent = harness.wrap(target, id="kimi-chat")
    first = await agent.run("My order is o-17.", user="ada", thread="chat")
    assert first.status is RunStatus.SUCCESS, first.error
    events = [e async for e in agent.stream("Which order was it?", user="ada", thread="chat")]
    lost = [e.data for e in events if e.data.get("code") == "acp_session"]
    assert lost == [] and events[-1].error is None
    assert "My order is o-17." in json.dumps(endpoint.requests[-1])
    assert (tmp_path / "kimi-home" / "sessions").is_dir()  # kept where Kimi keeps them


async def test_a_sandboxed_conversations_next_turn_is_given_its_earlier_turns_by_the_harness(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """A sandboxed run's home links nothing, so its session is removed with it: the session is
    not kept for the conversation's next turn, which starts a new one with the earlier turns
    in its context (no session reported lost, no turn without its history)."""
    target = kimi_target(endpoint, tmp_path, sandbox=sandbox(FakeSandboxes()))
    agent = harness.wrap(target, id="kimi-boxed-chat")
    first = await agent.run("My order is o-17.", user="ada", thread="chat")
    assert first.status is RunStatus.SUCCESS, first.error
    events = [e async for e in agent.stream("Which order was it?", user="ada", thread="chat")]
    lost = [e.data for e in events if e.data.get("code") == "acp_session"]
    assert lost == [] and events[-1].error is None
    assert "My order is o-17." in json.dumps(endpoint.requests[-1])
    assert not (tmp_path / "kimi-home" / "sessions").exists()  # nothing of it on this machine


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
    # re-checked (#92): Kimi's own Read on the kept result's path finds nothing there — the
    # journal holds it, not the disk — so read_file stays its reader
    endpoint.plan = [("Read", {"path": path})]
    native = await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-kept").run(
        "read the kept rows", user="u1"
    )
    told = text_of(endpoint.requests[-1]["messages"][-1].get("content"))
    assert "sku,units" not in told and told.startswith("<system>ERROR")


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


#: More times than Kimi's own repeat breaker lets one call repeat before it ends the turn (12).
LOOPS = 20


async def test_a_loop_kimis_repeat_breaker_ends_is_asked_once_for_an_answer(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """A model looping on one harness call (ADV-9): the bridge refuses the third (``STALLED``)
    and Kimi's own repeat breaker ends the turn at the twelfth with no answer (its log's
    ``stopReason`` ``repeat_breaker``; ACP's ``end_turn``). As at its step limit, the run asks
    once more in the same session, told the turn stalled: its answer is the run's."""
    lookups.clear()
    endpoint.plan = [("lookup", {"topic": "tea"})] * LOOPS
    agent = arm(harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-loop"), [lookup])
    events = [e async for e in agent.stream("tea?", user="u1")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    said = "".join(e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT)
    assert said.startswith("Done."), said[:200]  # the planned model's answer: what it read last
    assert lookups == ["tea", "tea"]
    assert "You called lookup with the same arguments 3 times in a row" in endpoint.said()
    warned = [e.data.get("code") for e in events if e.data.get("name") == "warning"]
    assert warned == ["stalled"]  # the bridge's, once
    assert TURN_STALLED in endpoint.said()
    assert len(endpoint.requests) < LOOPS  # told so, the model answered


async def test_a_loop_on_kimis_own_tool_is_asked_once_for_an_answer(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's own tools do not go through the bridge's refusal: its repeat breaker alone ends
    the loop, with no answer — asked once more the same way, said as a ``stalled`` warning."""
    endpoint.plan = [("Glob", {"pattern": "*.md"})] * LOOPS
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-own-loop")
    events = [e async for e in agent.stream("which notes?", user="u1")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    said = "".join(e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT)
    assert said.startswith("Done."), said[:200]
    warned = [e.data for e in events if e.data.get("name") == "warning"]
    assert [w["code"] for w in warned] == ["stalled"], warned
    assert TURN_STALLED in endpoint.said()
    assert len(endpoint.requests) < LOOPS  # told so, the model answered
