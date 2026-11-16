"""Kimi Code itself (``kimi acp``, the version ``compat.BINARIES`` pins) under the harness, its
model a planned endpoint on this machine (``tests.support.kimi``): no Moonshot key, nothing
sent anywhere. These pin what the adapter reads of Kimi — its tool names, its permission
requests, its sub-agents' and swarms' results, its usage log — so a Kimi that changes them
fails here. CI installs the pinned ``@moonshot-ai/kimi-code``; without it they are skipped
(``CI`` set: they fail)."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.integration.test_mcp_servers import http_server  # noqa: F401 - a fixture
from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.invariants import descendants
from tests.support.kimi import (
    KIMI_BIN,
    PlannedEndpoint,
    kimi_home_of,
    kimi_target,
    needs_kimi,
    result_text,
    text_of,
    unreminded,
)
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, Hooks, Settings, tool
from trellis.contracts import RunEventType, RunOutcome, RunStatus
from trellis.harness import compat
from trellis.harness.adapters.acp import KIMI_BUILTINS, KIMI_OUTLIVES_THE_TURN, ACPAdapter
from trellis.harness.bridges import live
from trellis.harness.bridges.acp_live import ACPSession
from trellis.harness.bridges.resumed import RESUMED_CALL
from trellis.harness.budget import Budget
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.identity import conversation_key
from trellis.harness.sandbox import sandbox
from trellis.harness.tools import bridge
from trellis.harness.tools.bridge import TURN_STALLED

#: Kimi Code runs its shell in a folder of this machine: the tests' deployment allows it
#: (TRELLIS_HOST_EXECUTION=allow, the harness fixtures).
pytestmark = [needs_kimi, pytest.mark.host_execution]
#: the deployment of the tests that make their own harness
HOST_ALLOWED = Settings(host_execution="allow")
lookups: list[str] = []
#: A command that writes (a file in Kimi's working directory): Kimi asks, and so a person.
WRITES = "touch made.txt && echo resumed"
#: A write of the user's own file, which Kimi asks about outside a git working tree: the
#: user's to decide (the one rule), where a command is the deployment's.
WRITE = ("Write", {"path": "made.txt", "content": "resumed"})


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


async def test_kimi_offers_its_model_an_mcp_server_over_http_in_its_first_request(
    harness: Harness,
    endpoint: PlannedEndpoint,
    tmp_path: Path,
    http_server: str,  # noqa: F811 - the fixture imported above
) -> None:
    """KIMI-MCP-FIRST (kimi.md, Limits): an MCP server over HTTP given in ``mcp_servers`` is in
    the tools of Kimi's first model request — the way to give Kimi 2.6.2 an MCP server of your
    own, whose stdio servers it lists but leaves out of its requests."""
    server = {"type": "http", "name": "notes", "url": http_server}
    headers = [{"name": "Authorization", "value": "Bearer t0k"}]
    target = kimi_target(endpoint, tmp_path, mcp_servers=[{**server, "headers": headers}])
    assert (await harness.wrap(target, id="kimi-mcp").run("hi", user="u1")).answer is not None
    assert "mcp__notes__read_note" in endpoint.offered()[0]


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


@pytest.mark.parametrize(
    ("told", "warned"),
    [
        (({"KIMI_CODE_EXPERIMENTAL_TOWER": "1"}, ""), ["acp_env"]),
        (({"KIMI_CODE_EXPERIMENTAL_FLAG": "1"}, ""), []),
        (({}, "[experimental]\ntower = true\n"), []),
    ],
    ids=["its-own-switch", "the-master-switch", "its-config"],
)
async def test_kimis_tower_mode_is_off_however_kimi_is_told_to_turn_it_on(
    harness: Harness,
    endpoint: PlannedEndpoint,
    tmp_path: Path,
    told: tuple[dict[str, str], str],
    warned: list[str],
) -> None:
    """L47. Kimi's tower mode starts its workers in its ``auto`` permission mode (they ask
    about nothing: the harness could govern none of their calls). Turned on — by its own
    switch in the agent's ``env``, by the switch for every experiment
    (``KIMI_CODE_EXPERIMENTAL_FLAG``), or in Kimi's ``config.toml`` — Kimi offers its
    ``Tower*`` tools even over ACP; under the harness its own switch keeps the mode off (Kimi
    reads that variable before its config and the master switch), and a run whose ``env``
    asked for it says why."""
    env, config = told
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config)
    target = kimi_target(endpoint, tmp_path, env=env)
    events = [e async for e in harness.wrap(target, id="kimi-tower").stream("hi", user="u1")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    assert not [t for t in endpoint.offered()[0] if t.startswith("Tower")]
    assert [e.data.get("code") for e in events if e.data.get("name") == "warning"] == warned


async def test_a_write_asks_a_person_and_the_resume_continues_kimis_session(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [WRITE]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-write")
    paused = await agent.run("say it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "Write"
    assert paused.interrupt.tool_call.args == WRITE[1]
    assert paused.interrupt.question == (
        "Approve Write? Kimi Code asks before it runs this call. "
        "Allow once, allow for this conversation, or deny."
    )
    assert not (tmp_path / "kimi-work" / "made.txt").exists()
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer is not None and done.answer.startswith("Done.")
    assert (tmp_path / "kimi-work" / "made.txt").read_text() == "resumed"
    # the same session, told to call again: what came before is in it
    last = endpoint.requests[-1]["messages"]
    assert any(RESUMED_CALL.format(tool="Write") in text_of(m.get("content")) for m in last)
    assert any("say it" in text_of(m.get("content")) for m in last)


async def test_a_command_on_a_host_the_deployment_allows_runs_unasked_and_announced(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's ``Bash`` on this machine, where the administrator allows commands here: the
    command is the deployment's, never the user's question — it runs, announced."""
    endpoint.plan = [("Bash", {"command": WRITES})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-bash")
    events = [e async for e in agent.stream("say it", user="u1")]
    assert events[-1].error is None and not any(e.type is RunEventType.INTERRUPT for e in events)
    assert [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"] == ["Bash"]
    assert (tmp_path / "kimi-work" / "made.txt").exists()


async def test_a_pause_kimi_stops_before_it_reads_the_answer_resumes_the_same_call(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause answers Kimi's permission request (dismissed) and stops the turn
    (``session/cancel``); Kimi may take the cancel first — on a loaded machine it does — and
    its ``Bash`` then ends the waiting call itself, its result ``Aborted before command
    started`` instead of "not run because the approval request was cancelled"; its ``Write``
    likewise. Either way the call did not run: the resume tells Kimi to call it again, and it
    runs once approved."""
    from trellis.harness.adapters import acp

    stop = acp.Attempt._stop

    async def cancelled_first(self: Any) -> None:
        """The turn stopped before the answer goes out: the answer waits until Kimi has ended
        the call itself (its log says so)."""
        await self.cancel()
        async with asyncio.timeout(30):
            while not any(  # noqa: ASYNC110 - a file Kimi writes
                '"tool.result"' in line
                for log in kimi_home_of("u1").glob("sessions/*/*/agents/main/wire.jsonl")
                for line in log.read_text().splitlines()
            ):
                await asyncio.sleep(0.01)

    monkeypatch.setattr(acp.Attempt, "_stop", cancelled_first)
    endpoint.plan = [WRITE]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-write-cancelled-first")
    paused = await agent.run("say it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert not (tmp_path / "kimi-work" / "made.txt").exists()
    monkeypatch.setattr(acp.Attempt, "_stop", stop)
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer is not None and done.answer.startswith("Done.")
    assert (tmp_path / "kimi-work" / "made.txt").read_text() == "resumed"


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
async def test_a_command_that_only_reads_runs_without_asking_a_person(
    harness: Harness, tmp_path: Path, streamed: bool
) -> None:
    """Kimi asks before every ``Bash`` command; the harness lets each run, one that only reads
    as any: the shell is the deployment's, as Kimi's own read tools run: no pause."""
    async with PlannedEndpoint(streamed=streamed).serving() as endpoint:
        endpoint.plan = [("Bash", {"command": "echo resumed | wc -l; which kimi 2>&1"})]
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-reads")
        done = await agent.run("count", user="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer is not None and done.answer.startswith("Done. 1")


@pytest.mark.parametrize(
    "rules",
    [
        '[[permission.rules]]\ndecision = "allow"\npattern = "Write"\n',
        '[[permission.allow]]\ntool = "Write"\n',
    ],
    ids=["a-rule", "an-allow-entry"],
)
async def test_a_call_kimis_own_config_allows_is_still_asked_about_once(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, rules: str
) -> None:
    """APR-01. Since Kimi Code 2.6 the ``allow`` rules of its ``config.toml`` apply over ACP: a
    call one covers would run with nobody asked, and ACP gives a client no way to be asked
    first. A run's Kimi works in its user's own home, whose config is the team's with an
    ``ask`` rule for each such ``allow`` rule (Kimi reads those first): Kimi asks, governance
    decides, a person is asked once, and the call runs once approved."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text("default_thinking = false\n" + rules)
    endpoint.plan = [WRITE]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-allowed")
    paused = await agent.run("make it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.args == WRITE[1]
    assert not (tmp_path / "kimi-work" / "made.txt").exists()
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done.error
    assert (tmp_path / "kimi-work" / "made.txt").exists()
    # the team's config is as it was; the user's has it whole, and the rule that asks
    assert (home / "config.toml").read_text() == "default_thinking = false\n" + rules
    own = (kimi_home_of("u1") / "config.toml").read_text()
    assert own.startswith("default_thinking = false\n" + rules) and 'decision = "ask"' in own


async def test_a_config_whose_rules_are_inline_is_still_asked_about(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """APR-01. A team's config that writes its ``[permission]`` rules inline takes no
    ``[[permission.rules]]`` after them (TOML), but it takes Kimi's other form of an ``ask``
    rule (``[[permission.ask]]``, which Kimi reads too): its ``allow`` rule is asked over —
    Kimi asks, and a person is asked before the call runs."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    inline = '[permission]\nrules = [{ decision = "allow", pattern = "Write" }]\n'
    (home / "config.toml").write_text(inline)
    endpoint.plan = [WRITE]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-inline")
    paused = await agent.run("make it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    call = paused.interrupt.tool_call
    assert call is not None and call.args == WRITE[1]
    assert not (tmp_path / "kimi-work" / "made.txt").exists()
    own = (kimi_home_of("u1") / "config.toml").read_text()
    assert own.startswith(inline) and '[[permission.ask]]\npattern = "Write"' in own


async def test_a_config_whose_whole_permission_is_inline_stops_the_run_before_its_prompt(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """APR-01. A ``permission`` written as one inline table takes no rule after it in any form
    (TOML): its ``allow`` rule would let the call run unasked, so the run gets no prompt and
    fails naming the rule."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    inline = 'permission = { rules = [{ decision = "allow", pattern = "Bash(touch *)" }] }\n'
    (home / "config.toml").write_text(inline)
    endpoint.plan = [("Bash", {"command": "touch made.txt"})]
    done = await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-inline-all").run(
        "make it", user="u1"
    )
    assert done.status is RunStatus.ERROR and done.error is not None
    assert "Bash(touch *)" in done.error.message and "config.toml" in done.error.message
    assert not endpoint.requests and not (tmp_path / "kimi-work" / "made.txt").exists()


@pytest.mark.parametrize("asks", [False, True], ids=["a-read", "the-catalog-asks"])
async def test_a_skill_kimi_loads_is_decided_by_governance(
    endpoint: PlannedEndpoint, tmp_path: Path, asks: bool
) -> None:
    """KIMI-SKILL-1. Kimi runs its ``Skill`` tool unasked by default; a run's Kimi asks about
    it (an ``ask`` rule in its user's config), so governance decides the call: a read runs at
    once and is the run's call; a catalog that says more has a person asked first."""
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    store = SkillStore()
    store.put("sql-review", "Check every JOIN.", description="Reviews SQL queries for joins.")
    endpoint.plan = [("Skill", {"skill": "sql-review"})]
    rules = {"Skill": Rule(risk="irreversible")} if asks else {}
    async with with_governance(Harness(config=HOST_ALLOWED), Governance(FakeCatalog(rules))) as h:
        target = kimi_target(endpoint, tmp_path, **combine(store).kimi())
        agent = h.wrap(target, id=f"kimi-skill-{asks}")
        events = [e async for e in agent.stream("review my SQL", user="u1")]
        finished = events[-1]
        if asks:
            interrupt = finished.data.get("interrupt") or {}
            call = interrupt.get("tool_call") or {}
            assert call.get("tool") == "Skill", finished.data
            assert call.get("args") == {"skill": "sql-review"}
        else:
            assert finished.data.get("interrupt") is None, finished.data
            assert finished.error is None, finished.error
            ended = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
            assert [(r["tool"], r["status"]) for r in ended] == [("Skill", "ok")]
    assert 'pattern = "Skill"' in (kimi_home_of("u1") / "config.toml").read_text()


async def test_two_users_never_share_a_kimi_home(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """SEC (DECISIONS A2, for Kimi). Each tenant's user has a Kimi home of their own on a
    worker: their sessions and Kimi's logs and caches are in it, never in another user's or
    in the team's; the team's login and setup are linked into both."""
    team = tmp_path / "kimi-home"
    team.mkdir(parents=True)
    (team / "AGENTS.md").write_text("Answer briefly.\n")
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-homes")
    for user in ("ada", "bob"):
        done = await agent.run("hi", user=user)
        assert done.status is RunStatus.SUCCESS, done.error
    ada, bob = kimi_home_of("ada"), kimi_home_of("bob")
    assert ada != bob and not (team / "sessions").exists()
    # each home holds its user's one session (its folder, and Kimi's own files beside it)
    held = [
        {p.name.partition(".")[0] for p in home.glob("sessions/*/session_*")} for home in (ada, bob)
    ]
    assert [len(ids) for ids in held] == [1, 1] and held[0] != held[1]
    assert all((home / "AGENTS.md").resolve() == team / "AGENTS.md" for home in (ada, bob))
    assert all(not (home / "config.toml").is_symlink() for home in (ada, bob))
    assert ada.stat().st_mode & 0o777 == 0o700


async def test_a_write_inside_a_git_working_tree_is_kimis_own_to_approve(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """L47. Kimi approves ``Write`` and ``Edit`` inside a git working tree itself (its
    ``git-cwd-write-approve`` policy: no setting turns it off, and ACP has no hook before a
    call), so the file is written with nobody asked and the call is recorded after; the same
    write outside a git tree asks. Pinned: a Kimi that asks here pauses the first run."""
    work = tmp_path / "kimi-work"
    endpoint.plan = [("Write", {"path": "new.txt", "content": "hi"})]
    target = kimi_target(endpoint, tmp_path)
    made = await asyncio.create_subprocess_exec("git", "init", "-q", str(work))
    assert await made.wait() == 0
    inside = await harness.wrap(target, id="kimi-git").run("write it", user="u1")
    assert inside.status is RunStatus.SUCCESS and (work / "new.txt").read_text() == "hi"
    outside = await harness.wrap(kimi_target(endpoint, tmp_path / "plain"), id="kimi-plain").run(
        "write it", user="u1"
    )
    assert outside.status is RunStatus.PAUSED and outside.interrupt is not None
    assert outside.interrupt.tool_call is not None and outside.interrupt.tool_call.tool == "Write"


async def test_kimis_schedule_tools_leave_the_models_list_when_kimis_config_disables_them(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's scheduling is off under the harness (``KIMI_DISABLE_CRON``), but Kimi still
    lists ``CronCreate``, ``CronList`` and ``CronDelete`` to its model: no environment switch
    and nothing in ACP takes a built-in off. Kimi's own ``[tools] disabled`` does."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-cron-listed").run("hi", user="u1")
    assert {"CronCreate", "CronList", "CronDelete"} <= set(endpoint.offered()[0])
    (home / "config.toml").write_text(
        '[tools]\ndisabled = ["CronCreate", "CronList", "CronDelete"]\n'
    )
    await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-cron-hidden").run("hi", user="u1")
    assert not [name for name in endpoint.offered()[-1] if name.startswith("Cron")]


async def test_a_write_allowed_for_the_conversation_is_approved_for_kimis_session(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """APR-01 (DECISIONS B4). "Allow for this conversation" on a write Kimi asks about is
    Kimi's own "Approve for this session" (its log says so), and the tool's next write asks
    nobody: one question."""
    endpoint.plan = [WRITE, ("Write", {"path": "too.txt", "content": "too"})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-keeps")
    paused = await agent.run("make them", user="u1", thread="chat")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert [o.label for o in paused.interrupt.options if not isinstance(o, str)] == [
        "Allow once",
        "Allow for this conversation",
        "Deny",
    ]
    done = await agent.resume(
        paused.interrupt.interrupt_id, "approve", reviewer="u1", remember="run"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    work = tmp_path / "kimi-work"
    assert (work / "made.txt").exists() and (work / "too.txt").exists()
    [log] = kimi_home_of("u1").glob("sessions/*/*/agents/main/wire.jsonl")
    resolved = [
        json.loads(line) for line in log.read_text().splitlines() if "interaction.resolved" in line
    ]
    # the request the run paused on is resolved {"decision": "cancelled"} by the harness — or,
    # when the process ends before that answer is applied (a loaded machine's order), by Kimi
    # itself, {"cancelled": true, "reason": "agent_closed"}, with no decision; every approval
    # is for the session
    assert [
        r["response"].get("scope") for r in resolved if r["response"].get("decision") == "approved"
    ] == [
        "session",
        "session",
    ]


@pytest.mark.parametrize("starts_in", ["auto", "yolo"])
async def test_a_config_that_approves_on_its_own_does_not_turn_the_approvals_off(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, starts_in: str
) -> None:
    """APR-01, L46. ``default_permission_mode`` in Kimi's ``config.toml`` starts every session
    approving on its own, and Kimi's ACP server still says ``default``. The harness sets the
    mode that asks on each session — the first, and the one each resume continues in a new
    process — so each write asks a person and runs only once approved."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(f'default_permission_mode = "{starts_in}"\n')
    endpoint.plan = [WRITE, ("Write", {"path": "too.txt", "content": "too"})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-config-mode")
    work = tmp_path / "kimi-work"
    paused = await agent.run("make them", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert not (work / "made.txt").exists()
    again = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert again.status is RunStatus.PAUSED and again.interrupt is not None
    assert (work / "made.txt").exists() and not (work / "too.txt").exists()
    done = await agent.resume(again.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert (work / "too.txt").exists()


def _asks_two(body: dict[str, Any]) -> Any:
    """A model that asks its user two questions in one call — again when the call was
    dismissed (the run paused on it) — and then says what it was answered."""
    results = [result_text(m["content"]) for m in body["messages"] if m.get("role") == "tool"]
    asked_again = not results or '"answers":{}' in results[-1] or '"answers"' not in results[-1]
    if asked_again:  # not answered: its question was cut short by the pause
        pick = [{"label": "pg", "description": "Postgres"}, {"label": "mysql", "description": ""}]
        clouds = [{"label": "aws", "description": ""}, {"label": "gcp", "description": ""}]
        questions = [
            {"question": "Which db?", "header": "DB", "options": pick},
            {
                "question": "Which cloud?",
                "header": "Cloud",
                "options": clouds,
                "multi_select": True,
            },
        ]
        return ("AskUserQuestion", {"questions": questions})
    return "answered: " + results[-1]


async def test_every_question_kimi_asks_in_one_call_reaches_the_person(
    harness: Harness, tmp_path: Path
) -> None:
    """L47. Kimi sends a client that takes forms every question of an ``AskUserQuestion``
    call (ACP's ``elicitation/create``); a client that does not is asked the first only, and
    the others are dropped. The harness takes forms: one question to the person holds both,
    the second taking several picks, and Kimi reads both answers."""
    async with PlannedEndpoint(turns=_asks_two).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-asks-two")
        asked = await agent.run("set it up", user="u1")
        assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
        assert asked.interrupt.question == "Which db?\nWhich cloud?"
        assert asked.interrupt.expects is not None
        fields = asked.interrupt.expects["properties"]
        assert [o["const"] for o in fields["q0"]["oneOf"]] == ["pg", "mysql"]
        assert fields["q0"]["oneOf"][0]["description"] == "Postgres"
        assert [o["const"] for o in fields["q1"]["items"]["anyOf"]] == ["aws", "gcp"]
        picked = {"q0": "mysql", "q1": ["aws", "gcp"]}
        done = await agent.resume(
            asked.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1"
        )
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer is not None
        assert '"Which db?":"mysql"' in done.answer
        assert '"Which cloud?":"aws, gcp"' in done.answer


#: The plan file Kimi names in its plan mode's reminder.
_PLAN_FILE = re.compile(r"(/[^\s\"'`<>]+/plans/[^\s\"'`<>]+\.md)")
PLAN = "# Plan\n\nOption A: fast.\nOption B: safe.\n"


def _plans(body: dict[str, Any]) -> Any:
    """A model in Kimi's plan mode: it writes the plan file Kimi names, presents the plan with
    two approaches — again when its review was cut short (the run paused on it) — and then
    says what the review answered."""
    messages = body["messages"]
    results = [result_text(m["content"]) for m in messages if m.get("role") == "tool"]
    if not results:
        said = "\n".join(text_of(m.get("content")) for m in messages)
        return ("Write", {"path": _PLAN_FILE.findall(said)[-1], "content": PLAN})
    reviewed = ("Exited plan mode", "User requested revisions", "Plan rejected")
    if len(results) == 1 or not any(said in results[-1] for said in reviewed):
        # not reviewed yet: the plan was only written, or its review was cut short by the
        # pause (Kimi's result then says it was dismissed, or interrupted)
        approaches = [{"label": "Fast", "description": "A"}, {"label": "Safe", "description": "B"}]
        return ("ExitPlanMode", {"options": approaches})
    return "reviewed: " + results[-1]


@pytest.mark.parametrize(
    ("planning", "picked", "result"),
    [
        ("mode", "Safe", "Selected approach: Safe"),
        ("mode", "Revise", "User requested revisions. Plan mode remains active."),
        ("config", "Fast", "Selected approach: Fast"),
    ],
    ids=["mode-approach", "mode-revise", "config-approach"],
)
async def test_a_plan_kimi_presents_is_reviewed_by_a_person_with_every_option(
    harness: Harness, tmp_path: Path, planning: str, picked: str, result: str
) -> None:
    """APR-01, L47 (DECISIONS B3). Kimi asks its user to review every plan (``ExitPlanMode``):
    the plan, an option an approach, ``Revise`` and ``Reject and Exit``. The run pauses on that
    one question, and the person's pick is Kimi's answer — in a session the run put in plan
    mode (``mode="plan"``) and in one Kimi's own ``default_plan_mode`` did."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    if planning == "config":
        (home / "config.toml").write_text("default_plan_mode = true\n")
    fields = {"mode": "plan"} if planning == "mode" else {}
    async with PlannedEndpoint(turns=_plans).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path, **fields), id="kimi-plans")
        asked = await agent.run("plan it", user="u1")
        assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
        assert PLAN.strip() in asked.interrupt.question
        assert asked.interrupt.options == ["Fast", "Safe", "Revise", "Reject and Exit"]
        done = await agent.resume(
            asked.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1"
        )
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer is not None and result in done.answer


@pytest.mark.parametrize(
    "rule", [Rule(risk="irreversible"), Rule(risk="write", approve_when="true")], ids=str
)
async def test_a_team_that_wants_every_shell_command_approved_says_so_in_the_catalog(
    endpoint: PlannedEndpoint, tmp_path: Path, rule: Rule
) -> None:
    endpoint.plan = [("Bash", {"command": "ls"})]
    async with with_governance(
        Harness(config=HOST_ALLOWED), Governance(FakeCatalog({"Bash": rule}))
    ) as h:
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
    there (its own ``Bash``, unasked where the deployment allows commands on this machine)
    moves to the file store on success."""
    from trellis.harness.sandbox import DELIVERS_HERE

    save = "mkdir -p outputs && printf 'a,b\\n1,2\\n' > outputs/sales.csv"
    endpoint.plan = [("Bash", {"command": save})]
    folders: list[str] = []

    class Ended(Hooks):
        async def on_run_end(self, run: Any, result: Any) -> None:
            folders.append(run.files_folder())

    allowed = Settings(host_execution="allow", files_url=f"memory://kimi-out-{id(tmp_path)}")
    h = Harness(config=allowed)
    agent = h.wrap(kimi_target(endpoint, tmp_path), id="kimi-saver", hooks=[Ended()])
    done = await agent.run("save the sales", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done.error
    assert await h.files.read(folders[-1], "sales.csv") == b"a,b\n1,2\n"
    assert not (tmp_path / "kimi-work" / "outputs" / "sales.csv").exists()
    assert DELIVERS_HERE in endpoint.said()
    await h.aclose()


async def test_a_write_whose_arguments_stream_pauses_with_them(
    harness: Harness, tmp_path: Path
) -> None:
    """A provider streams a call's arguments after its name (Anthropic's, OpenAI's): Kimi asks
    permission before it starts the call, so the arguments are only in the call's updates."""
    async with PlannedEndpoint(streamed=True).serving() as endpoint:
        endpoint.plan = [WRITE]
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-write-streamed")
        paused = await agent.run("say it", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "Write"
        assert paused.interrupt.tool_call.args == WRITE[1]
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer is not None and done.answer.startswith("Done.")


async def test_a_rejected_write_is_not_run_and_kimi_reads_why(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [("Write", {"path": "build", "content": "x"})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-reject")
    paused = await agent.run("clean", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", reviewer="u1", comment="not on main"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert "not on main" in endpoint.said() or "denied" in str(done.answer)
    assert not (tmp_path / "kimi-work" / "build").exists()


async def test_kimis_own_tools_that_only_read_run_unasked_and_are_recorded(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi asks about none of its tools that only read (``Glob``, ``Grep``, ``Read``,
    ``TaskList``): each runs with nobody asked, and is recorded with its result once it ran."""
    ran: list[tuple[str, Any]] = []
    observed = bridge.observed

    async def recording(runtime: Any, spec: Any, args: Any, outcome: Any, **more: Any) -> None:
        ran.append((spec.name, outcome.status.value))
        await observed(runtime, spec, args, outcome, **more)

    monkeypatch.setattr(bridge, "observed", recording)
    target = kimi_target(endpoint, tmp_path)
    (tmp_path / "kimi-work" / "notes.txt").write_text("tea is in the cupboard\n")
    endpoint.plan = [
        ("Glob", {"pattern": "*.txt"}),
        ("Grep", {"pattern": "cupboard"}),
        ("Read", {"path": "notes.txt"}),
        ("TaskList", {}),
    ]
    done = await harness.wrap(target, id="kimi-reads-all").run("where is the tea?", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done.error
    assert ran == [("Glob", "ok"), ("Grep", "ok"), ("Read", "ok"), ("TaskList", "ok")]


async def test_kimis_todo_list_is_the_runs_plan_event(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's ``TodoList`` is its plan over ACP: each state of it is a ``plan`` event of the
    run, its entries as Kimi tells them."""
    todos = [
        {"title": "read the code", "status": "in_progress"},
        {"title": "fix it", "status": "pending"},
    ]
    endpoint.plan = [("TodoList", {"todos": todos})]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-todos")
    events = [e async for e in agent.stream("plan your work", user="u1")]
    assert events[-1].error is None
    [plan] = [e.data["entries"] for e in events if e.data.get("name") == "plan"]
    assert [(entry["content"], entry["status"]) for entry in plan] == [
        ("read the code", "in_progress"),
        ("fix it", "pending"),
    ]


def _provider_down(body: dict[str, Any]) -> Any:
    raise RuntimeError("the provider is down")


async def test_a_turn_kimi_could_not_finish_fails_the_run_with_kimis_reason(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi's provider answers every request with an error: past its own retries Kimi ends the
    turn ``failed`` in its log and answers the prompt ``end_turn`` with nothing. The run fails
    with Kimi's reason (its error's code and the provider's status), not ``SUCCESS``."""
    env = {"KIMI_LOOP_MAX_RETRIES_PER_STEP": "0", "KIMI_LOOP_MAX_ATTEMPTS_PER_STEP": "1"}
    async with PlannedEndpoint(turns=_provider_down).serving() as endpoint:
        target = kimi_target(endpoint, tmp_path, env=env)
        done = await harness.wrap(target, id="kimi-provider-down").run("go", user="u1")
    assert done.status is RunStatus.ERROR and done.error is not None
    assert "provider.api_error" in done.error.message and "500" in done.error.message


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
    [log] = list(kimi_home_of("u1").glob("sessions/*/*/agents/main/wire.jsonl"))
    records = [json.loads(line) for line in log.read_text().splitlines() if "usage.record" in line]
    assert records, "Kimi logs a usage.record per model request"
    for record in records:
        assert record["type"] == "usage.record" and record["agentId"] == "main"
        # the counts the adapter reads; Kimi 2.6.2 leaves a cache count out when it is none
        assert {"inputOther", "output"} <= set(record["usage"])
        assert set(record["usage"]) <= {
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


#: What Kimi answers an ``AgentSwarm`` call outside its swarm mode.
SWARM_OFF = "Swarm mode is off, so AgentSwarm did not run and no subagents were started."


async def test_a_swarm_is_refused_by_kimi_over_acp_and_its_members_runs_end_saying_so(
    harness: Harness, tmp_path: Path
) -> None:
    """Since Kimi Code 2.3.2 ``AgentSwarm`` runs only in Kimi's swarm mode, which only its user
    turns on in its own terminal (``/swarm``): no ACP client can. Kimi still offers the tool;
    called, it starts nobody and says so. The run goes on with Kimi's answer, and the child run
    opened for each member ends ``ERROR`` with Kimi's words — none left ``RUNNING``."""
    async with PlannedEndpoint(turns=_swarm_turns).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-swarm")
        done = await agent.run("review both", user="u1")
        assert "AgentSwarm" in endpoint.offered()[0]
    assert done.status is RunStatus.SUCCESS, done.error
    assert str(done.answer).startswith(SWARM_OFF)
    found = [s async for s in harness.runs.iterate(parent_run_id=done.run_id)]
    kids = [await harness.runs.get(s.run_id) for s in found]
    assert sorted((k.input, k.status) for k in kids if k is not None) == [
        ("Review a.ts", RunStatus.ERROR),
        ("Review b.ts", RunStatus.ERROR),
    ]
    assert all(k.error is not None and k.error.message.startswith(SWARM_OFF) for k in kids if k)


def _resuming_turns(body: dict[str, Any]) -> Any:
    """The main agent runs an ``explore`` sub-agent, goes on with it (``Agent(resume=...)``,
    its id in the first call's answer), then answers."""
    messages = body["messages"]
    tools = {t["function"]["name"] for t in body.get("tools") or ()}
    users = [text_of(m.get("content")) for m in messages if m.get("role") == "user"]
    if "Agent" not in tools:  # the sub-agent, by the last thing it was told (not a reminder)
        last = [u for u in users if not u.startswith("<system-reminder>")][-1]
        return "went on" if "continue a" in last else "reviewed a.ts"
    results = [result_text(m["content"]) for m in messages if m.get("role") == "tool"]
    if not results:
        first = {"description": "review", "prompt": "Review a.ts", "subagent_type": "explore"}
        return ("Agent", first)
    if len(results) == 1:
        ran = re.search(r"^agent_id: (\S+)$", results[0], re.MULTILINE)
        assert ran is not None, results[0]
        return ("Agent", {"description": "more", "prompt": "continue a", "resume": ran[1]})
    return "done"


async def test_a_sub_agent_kimi_goes_on_with_is_a_child_run_by_the_name_it_was_started_with(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi goes on with a sub-agent it ran (``Agent(resume=...)``): a child run of its own,
    named as Kimi started it (its ``explore`` type, which a resumed agent keeps — not
    ``coder``, the name of none), its task the prompt it goes on with, ended with what it
    said."""
    async with PlannedEndpoint(turns=_resuming_turns).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-resumes")
        done = await agent.run("review, then go on", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    found = [s async for s in harness.runs.iterate(parent_run_id=done.run_id)]
    kids = [await harness.runs.get(s.run_id) for s in found]
    named = sorted(
        (k.input, k.metadata["native_helper"]["name"], k.status) for k in kids if k is not None
    )
    assert named == [
        ("Review a.ts", "explore", RunStatus.SUCCESS),
        ("continue a", "explore", RunStatus.SUCCESS),
    ]
    outputs = {k.input: str(k.output) for k in kids if k is not None}
    assert "went on" in outputs["continue a"]


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


def _said_last(body: dict[str, Any]) -> str:
    """What the request's last user message says (not a reminder Kimi put before it)."""
    users = [text_of(m.get("content")) for m in body["messages"] if m.get("role") == "user"]
    return [u for u in users if not u.startswith("<system-reminder>")][-1]


def _backgrounded(sub: Any) -> Any:
    """The main agent runs a sub-agent in the background (``sub``: what that sub-agent does)
    and says it started; Kimi's own turn for its end (its notification) notes it; the next
    turn answers with what the conversation heard of it."""

    def turn(body: dict[str, Any]) -> Any:
        if not _member(body):
            said, last = _said_last(body), body["messages"][-1]
            if said.startswith("<notification"):
                return "noted"
            if "look for tea" in said:
                if last.get("role") == "tool":
                    return "started it"
                look = {"description": "look", "prompt": "find tea", "run_in_background": True}
                return ("Agent", look)
            notified = "task.completed" in json.dumps(body["messages"])  # its notification
            return f"heard: {'it ended' if notified else 'nothing'}"
        return sub(body)

    return turn


async def _until(check: Any, seconds: float = 30.0) -> Any:
    """``check`` (a coroutine function) awaited until what it gives holds: that."""
    async with asyncio.timeout(seconds):
        while not (found := await check()):  # noqa: ASYNC110 - what Kimi's process did
            await asyncio.sleep(0.1)
    return found


def _finds_tea(body: dict[str, Any]) -> Any:
    return "the sub-agent found tea"


async def test_a_sub_agent_kimi_runs_in_the_background_finishes_after_its_turn(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi's ``Agent(run_in_background=true)`` answers at once; its turn ends. The
    conversation's Kimi process is kept while the sub-agent works: its child run stays
    ``RUNNING`` (not ``SUCCESS`` with the launch's words), then ends with what the sub-agent
    said; Kimi's own turn for its end runs (ACP shows the client none of it), and the
    conversation's next turn — served by the same process — has its notification."""
    stall = lambda body: 1.5 * _member(body)  # noqa: E731 - the sub-agent takes a while
    async with PlannedEndpoint(turns=_backgrounded(_finds_tea), stall=stall).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-background")
        first = await agent.run("look for tea in the background", user="u1", thread="t1")
        assert first.status is RunStatus.SUCCESS and first.answer == "started it", first.error
        [summary] = [s async for s in harness.runs.iterate(parent_run_id=first.run_id)]
        child = await harness.runs.get(summary.run_id)
        assert child is not None and child.status is RunStatus.RUNNING
        child = await _until(lambda: _ended(harness, summary.run_id))
        assert child.status is RunStatus.SUCCESS and child.output == "the sub-agent found tea"

        async def noted() -> bool:  # Kimi's own turn for its end: it noted the result
            return any(_said_last(b).startswith("<notification") for b in endpoint.requests)

        await _until(noted)
        second = await agent.run("what did it find?", user="u1", thread="t1")
    assert second.status is RunStatus.SUCCESS and second.answer == "heard: it ended", second.error
    sessions = {json.dumps(b["messages"][0]) for b in endpoint.requests if not _member(b)}
    assert len(sessions) == 1  # one conversation in Kimi: the next turn was sent into it


async def _ended(harness: Harness, run_id: str) -> Any:
    """The run's record once it ended (``None`` before)."""
    record = await harness.runs.get(run_id)
    return record if record is not None and record.status.final else None


async def _paused(harness: Harness, run_id: str) -> bool:
    record = await harness.runs.get(run_id)
    return record is not None and record.status is RunStatus.PAUSED


def _makes_it(body: dict[str, Any]) -> Any:
    """A sub-agent whose work writes a file (its ``Write``: Kimi asks first)."""
    if body["messages"][-1].get("role") == "tool":
        return "made it"
    return WRITE


async def test_a_background_sub_agents_call_kimi_asks_about_is_governed_in_its_own_run(
    harness: Harness, tmp_path: Path
) -> None:
    """The background sub-agent's ``Write`` asks permission after its turn ended (Kimi's
    approvals are its session's, not a turn's): the call is decided in the sub-agent's own run
    — a person is asked, so that run pauses, and the conversation's next turn asks it. Approved,
    the write runs, the task ends and its run with it; then the turn's own prompt goes."""
    stall = lambda body: 1.0 * _member(body)  # noqa: E731 - the sub-agent takes a while
    async with PlannedEndpoint(turns=_backgrounded(_makes_it), stall=stall).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-background-asks")
        first = await agent.run("look for tea in the background", user="u1", thread="t1")
        assert first.status is RunStatus.SUCCESS, first.error
        [summary] = [s async for s in harness.runs.iterate(parent_run_id=first.run_id)]
        await _until(lambda: _paused(harness, summary.run_id))
        assert not (tmp_path / "kimi-work" / "made.txt").exists()
        second = await agent.run("how is it going?", user="u1", thread="t1")
        assert second.status is RunStatus.PAUSED and second.interrupt is not None
        call = second.interrupt.tool_call
        assert call is not None and call.tool == "Write"
        done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        child = await _until(lambda: _ended(harness, summary.run_id))
    assert child.status is RunStatus.SUCCESS and child.output == "made it"
    assert (tmp_path / "kimi-work" / "made.txt").exists()


def _looks_up(body: dict[str, Any]) -> Any:
    """A sub-agent that calls the harness's ``lookup`` (through its MCP server)."""
    if body["messages"][-1].get("role") == "tool":
        return "looked: " + result_text(body["messages"][-1]["content"])
    return ("mcp__trellis__lookup", {"topic": "tea"})


async def test_a_background_sub_agents_harness_tool_call_is_made_in_its_own_run(
    harness: Harness, tmp_path: Path
) -> None:
    """The background sub-agent calls a harness tool after its turn ended: Kimi asks
    permission for it first (whose it is: its log's ``interaction.request``), and the call
    runs for the sub-agent's own run."""
    lookups.clear()
    stall = lambda body: 1.0 * _member(body)  # noqa: E731 - the sub-agent takes a while
    async with PlannedEndpoint(turns=_backgrounded(_looks_up), stall=stall).serving() as endpoint:
        target = kimi_target(endpoint, tmp_path)
        agent = arm(harness.wrap(target, id="kimi-background-tool"), [lookup])
        first = await agent.run("look for tea in the background", user="u1", thread="t1")
        [summary] = [s async for s in harness.runs.iterate(parent_run_id=first.run_id)]
        child = await _until(lambda: _ended(harness, summary.run_id))
    assert child.status is RunStatus.SUCCESS and child.output == "looked: facts about tea"
    assert lookups == ["tea"]


async def test_a_command_kimi_runs_in_the_background_finishes_after_its_turn(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's ``Bash(run_in_background=true)`` answers at once with its task; the process is
    kept while the command works, so the command finishes after the turn ended."""
    command = {
        "command": "sleep 2 && echo done > done.txt",
        "run_in_background": True,
        "description": "wait",
    }
    endpoint.plan = [("Bash", command)]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-background-bash")
    # Kimi asks about a command that writes; the deployment's to allow, never the user's
    done = await agent.run("wait a bit", user="u1", thread="t1")
    assert done.interrupt is None
    assert done.status is RunStatus.SUCCESS and "task_id:" in str(done.answer), done.error
    made = tmp_path / "kimi-work" / "done.txt"
    assert not made.exists()  # still at work when the turn ended
    async with asyncio.timeout(20):
        while not made.exists():  # noqa: ASYNC110 - a file the command writes
            await asyncio.sleep(0.1)


#: What the scripted model says in the turn Kimi runs for a background command's end.
NOTIFIED = "The background command said BG-77."


def _waits_for_its_command(gate: Path) -> Any:
    """Turn one starts a command in the background that ends once ``gate`` is there; turn two
    waits for it (``WaitFor``), the gate made as the model asks to wait — so the command ends
    while that turn is under way; the turn Kimi runs for the command's end (its notification)
    reads what the command wrote and answers :data:`NOTIFIED`; any other turn answers
    ``noted``. Each answers the latest of
    those prompts, by the results that came after it."""

    def turn(body: dict[str, Any]) -> Any:
        messages = body["messages"]
        said = [unreminded(text_of(m.get("content"))) for m in messages]
        asked = [
            i for i, m in enumerate(messages) if m["role"] == "user" and _PROMPT.search(said[i])
        ]
        prompt = _PROMPT.search(said[asked[-1]])[0]  # type: ignore[index]
        ran = [
            said[i] for i in range(asked[-1] + 1, len(messages)) if messages[i]["role"] == "tool"
        ]
        if prompt == "<notification":  # it reads what the command wrote, then answers
            return NOTIFIED if ran else ("Read", {"path": str(gate.parent / "out.txt")})
        if prompt == "start the command" and not any("task_id:" in r for r in ran):
            command = {
                "command": f"while [ ! -f {gate.name} ]; do sleep 0.1; done; echo BG-77 > out.txt",
                "run_in_background": True,
                "description": "wait for the gate",
            }
            return ("Bash", command)
        if prompt == "wait for the command" and not ran:
            [task] = re.findall(r"task_id: (\S+)", " ".join(said))[-1:]
            gate.touch()
            return ("WaitFor", {"task_id": task, "timeout": 30})
        return "noted"

    return turn


#: The prompts :func:`_waits_for_its_command` answers.
_PROMPT = re.compile(r"start the command|wait for the command|<notification|thanks")


async def test_a_background_commands_end_read_in_a_later_turn_is_that_turns_answer(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIMI-BG-ANSWER, real Kimi: a turn that waits for a command it ran in the background
    (``WaitFor``) while the command ends — Kimi steers the command's notification into that
    turn, ends the turn right after the wait, and answers in a turn of its own that ACP does
    not forward — answers with what that turn said, as ``kimi -p`` does; its model request is
    counted once, on that turn's run, and the next turn counts only its own."""
    from trellis.harness import budget

    counted: dict[str, int] = {}
    account = budget.account

    def counting(runtime: Any, **usage: Any) -> None:
        tokens = usage["input_tokens"] + usage["output_tokens"]
        counted[runtime.run_id] = counted.get(runtime.run_id, 0) + tokens
        account(runtime, **usage)

    monkeypatch.setattr(budget, "account", counting)
    gate = tmp_path / "kimi-work" / "go"
    async with PlannedEndpoint(turns=_waits_for_its_command(gate)).serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-waits")
        # Kimi asks before a command; the command is the deployment's: it runs, unasked
        started = await agent.run("start the command", user="u1", thread="t1")
        assert started.status is RunStatus.SUCCESS and started.interrupt is None, started.error
        before = len(endpoint.requests)
        waited = await agent.run(
            "wait for the command and say what it printed", user="u1", thread="t1"
        )
        assert waited.status is RunStatus.SUCCESS, waited.error
        assert waited.answer == NOTIFIED
        # the wait, then the notification's turn: its read, its answer
        assert len(endpoint.requests) - before == 3
        after = len(endpoint.requests)
        thanked = await agent.run("thanks", user="u1", thread="t1")
        assert thanked.status is RunStatus.SUCCESS and thanked.answer == "noted"
    assert counted[waited.run_id] == 15 * 3
    assert counted[thanked.run_id] == 15 * (len(endpoint.requests) - after)
    assert sum(counted.values()) == 15 * len(endpoint.requests)


async def test_a_kept_kimi_whose_hold_lapses_ends_with_its_command_and_its_sessions_files(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIMI-LIVE-1, real Kimi: the process kept for a command it runs in the background ends
    when the conversation's hold of it lapses (kept as long as a session is) — and the command
    with it, no process of the test's left — no run is left ``RUNNING``, and the session's own
    folder leaves its user's Kimi home, whose config stays."""
    monkeypatch.setattr(live, "WORKING_SECONDS", 1.0)
    command = {
        "command": "sleep 60 && touch late.txt",
        "run_in_background": True,
        "description": "wait",
    }
    endpoint.plan = [("Bash", command)]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-lapse")
    before = set(descendants(os.getpid()))
    done = await agent.run("wait a minute", user="u1", thread="t1")  # the command: unasked
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done.error
    adapter = agent.adapter
    assert isinstance(adapter, ACPAdapter)
    session = adapter.live.get(conversation_key("default", "u1", "t1"))
    assert isinstance(session, ACPSession) and session.working  # kept: the command works
    home = kimi_home_of("u1")
    [folder] = home.glob(f"sessions/*/{session.session_id}")
    async with asyncio.timeout(30):
        while not session.closed or folder.exists() or set(descendants(os.getpid())) - before:  # noqa: ASYNC110 - processes and files
            await asyncio.sleep(0.1)
    assert not [s async for s in harness.runs.iterate() if s.status is RunStatus.RUNNING]
    assert (home / "config.toml").is_file() and (home / "sessions").is_dir()
    assert not (tmp_path / "kimi-work" / "late.txt").exists()


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
        settings = Settings(
            llm_gateway="bifrost",
            llm_base_url=endpoint.url,
            llm_api_key="vk-test",
            host_execution="allow",  # Kimi works in a folder of this machine
        )
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
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its command runs in the client's terminal, so Kimi's last update of the call names the
    terminal and carries no output: what the call is recorded with is that terminal's output,
    the one Kimi's model read."""
    recorded: dict[str, Any] = {}
    observed = bridge.observed

    async def recording(runtime: Any, spec: Any, args: Any, outcome: Any, **more: Any) -> None:
        recorded[spec.name] = outcome.output
        await observed(runtime, spec, args, outcome, **more)

    monkeypatch.setattr(bridge, "observed", recording)
    sandboxes = FakeSandboxes()
    endpoint.plan = [
        ("Write", {"path": "/workspace/notes.txt", "content": "hello"}),
        ("Bash", {"command": "echo boxed"}),
    ]
    target = kimi_target(endpoint, tmp_path, sandbox=sandbox(sandboxes))
    agent = harness.wrap(target, id="kimi-boxed")
    paused = await agent.run("note it", user="u1")
    assert paused.interrupt is not None  # the write asks; the command, in the sandbox, does not
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "Write"
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    [made] = sandboxes.made()
    assert made not in sandboxes.boxes  # deleted with the run
    assert any("echo boxed" in c for c in sandboxes.commands)
    assert not (tmp_path / "kimi-work" / "notes.txt").exists()  # not on this machine
    assert recorded["Bash"] and str(recorded["Bash"]).strip() in str(done.answer)


class Internal(http.server.BaseHTTPRequestHandler):
    """An address inside the deployment (a metadata server, an internal API): who reached it."""

    reached: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        Internal.reached.append(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"internal")

    def log_message(self, format: str, *args: Any) -> None:
        pass


async def test_a_sandboxed_kimis_fetch_is_closed_as_any_kimis(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi's ``FetchURL`` runs in its own process, on this machine's network, whether or not
    the run has a sandbox: with ``sandbox=`` too, the tool catalog's closed default holds — no
    request to an address inside the deployment."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Internal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/latest/meta-data/"
    Internal.reached.clear()
    endpoint.plan = [("FetchURL", {"url": url})]
    try:
        target = kimi_target(endpoint, tmp_path, sandbox=sandbox(FakeSandboxes()))
        result = await harness.wrap(target, id="kimi-fetch").run("go", user="u1")
    finally:
        server.shutdown()
    assert Internal.reached == [], result.answer
    assert "the tool catalog does not allow it" in str(result.answer)


async def test_a_sandboxed_kimis_own_searches_reach_no_file_of_the_service(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """A sandboxed Kimi's ``Grep`` (and ``Glob``, ``ReadMediaFile``) run in its own process on
    this machine, as the service's user — not in the sandbox, where its commands and ``Read``
    go: a path outside the run's folders is refused, its files never read."""
    secret = tmp_path / "service" / "secret.env"
    secret.parent.mkdir()
    secret.write_text("sk-service-kimi-0123456789")
    endpoint.plan = [("Grep", {"pattern": "sk-service", "path": str(secret.parent)})]
    target = kimi_target(endpoint, tmp_path, sandbox=sandbox(FakeSandboxes()))
    result = await harness.wrap(target, id="kimi-grep").run("go", user="u1")
    assert "sk-service-kimi" not in str(result.answer), result.answer
    assert "outside the folders this run works in" in str(result.answer)


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


async def test_kimis_own_skills_and_instructions_stay_beside_the_runs_skills(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """A run given skills works in a home of its own; what Kimi keeps in its usual home for
    every session — its own skills (``skills/``, where the run's are written too), its
    ``AGENTS.md`` — is still Kimi's there: the model is told both its own skill and the run's,
    and the instructions."""
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    own = tmp_path / "kimi-home" / "skills" / "tone"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("---\nname: tone\ndescription: How we write.\n---\nBe brief.\n")
    (tmp_path / "kimi-home" / "AGENTS.md").write_text("Answer in metric units.\n")
    store = SkillStore()
    store.put("sql-review", "Check every JOIN.", description="Reviews SQL queries for joins.")
    target = kimi_target(endpoint, tmp_path, **combine(store).kimi())
    done = await harness.wrap(target, id="kimi-own-skills").run("review", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    sent = json.dumps(endpoint.requests[0])
    assert "Reviews SQL queries for joins." in sent and "How we write." in sent
    assert "Answer in metric units." in sent
    assert (own / "SKILL.md").is_file()  # the run's home removed, Kimi's own skill kept


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
    assert (kimi_home_of("ada") / "sessions").is_dir()  # kept in its user's own Kimi home


@pytest.mark.parametrize("boxed", [False, True], ids=["on-this-machine", "sandboxed"])
async def test_a_users_message_that_starts_with_a_slash_is_words_for_the_model(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, boxed: bool
) -> None:
    """SEC. Over ACP a prompt whose first block starts with ``/`` is a command of the agent's
    (Kimi's ``/compact``, ``/status``, ``/skill:<name>``; any other: "Unknown ACP command"),
    run with no model turn. A user's message is words, not a command, in every run: the model
    is asked, and reads it."""
    fields = {"sandbox": sandbox(FakeSandboxes())} if boxed else {}
    agent = harness.wrap(kimi_target(endpoint, tmp_path, **fields), id="kimi-slash")
    done = await agent.run("/status of my order?", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    [request] = endpoint.requests
    assert "/status of my order?" in json.dumps(request["messages"])


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


async def test_at_its_step_limit_kimi_ends_partial_with_what_it_said(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """Kimi ends a turn at its step limit (``loop_control.max_steps_per_turn``) as a finished
    turn with no answer: ACP's ``end_turn``, the reason only in its own log
    (``loop.max_steps_exceeded``). As Claude's turn limit does (decision D1), the run ends
    ``PARTIAL`` with what Kimi said, saying so as a ``step_limit`` warning — no paid last
    answer: nothing more is asked of the model."""
    from trellis.harness.budget import TURN_LIMIT

    lookups.clear()
    endpoint.plan = [("lookup", {"topic": "tea"}), ("lookup", {"topic": "coffee"})]
    target = kimi_target(endpoint, tmp_path, env={"KIMI_LOOP_MAX_STEPS_PER_TURN": "1"})
    agent = arm(harness.wrap(target, id="kimi-limit"), [lookup])
    asked = len(endpoint.requests)
    events = [e async for e in agent.stream("tea and coffee?", user="u1")]
    done = events[-1]
    assert done.type is RunEventType.RUN_FINISHED and done.outcome is RunOutcome.PARTIAL
    assert lookups == ["tea"]
    warned = [e.data["message"] for e in events if e.data.get("code") == "step_limit"]
    assert warned == ["stopped: the step limit (1) was reached"]
    assert TURN_LIMIT.format(limit=1) not in endpoint.said()  # no last answer asked for
    assert len(endpoint.requests) - asked == 1  # the one step, nothing after it


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
