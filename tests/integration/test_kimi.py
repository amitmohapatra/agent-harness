"""Kimi Code itself (``kimi acp``, the version ``compat.BINARIES`` pins) under the harness, its
model a planned endpoint on this machine (``tests.support.kimi``): no Moonshot key, nothing
sent anywhere. These pin what the adapter reads of Kimi — its tool names, its permission
requests, its sub-agents' and swarms' results, its usage log — so a Kimi that changes them
fails here. CI installs the pinned ``@moonshot-ai/kimi-code``; without it they are skipped
(``CI`` set: they fail)."""

from __future__ import annotations

import asyncio
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
    assert paused.interrupt.question == (
        "Approve Bash? Kimi Code asks before it runs this call. "
        "Allow once, allow for this conversation, or deny."
    )
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


@pytest.mark.parametrize(
    "rules",
    [
        '[[permission.rules]]\ndecision = "allow"\npattern = "Bash(touch *)"\n',
        '[[permission.allow]]\ntool = "Bash"\nmatch = "touch *"\n',
    ],
    ids=["a-rule", "an-allow-entry"],
)
async def test_a_rule_of_kimis_own_config_that_allows_a_call_stops_the_run_before_it(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path, rules: str
) -> None:
    """APR-01. Since Kimi Code 2.6 the ``[permission]`` rules of its ``config.toml`` apply
    over ACP: a call an ``allow`` rule covers runs without Kimi asking, and ACP gives a client
    no way to be asked first. The harness governs only what Kimi asks about, so a run whose
    Kimi holds such a rule gets no prompt: it fails naming the rule, and nothing ran."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(rules)
    endpoint.plan = [("Bash", {"command": "touch made.txt"})]
    done = await harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-allowed").run(
        "make it", user="u1"
    )
    assert done.status is RunStatus.ERROR and done.error is not None
    assert "Bash(touch *)" in done.error.message and "config.toml" in done.error.message
    assert not endpoint.requests and not (tmp_path / "kimi-work" / "made.txt").exists()


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


async def test_a_command_allowed_for_the_conversation_is_approved_for_kimis_session(
    harness: Harness, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    """APR-01 (DECISIONS B4). "Allow for this conversation" on a command Kimi asks about is
    Kimi's own "Approve for this session" (its log says so), and the tool's next command asks
    nobody: one question."""
    endpoint.plan = [
        ("Bash", {"command": "touch made.txt"}),
        ("Bash", {"command": "touch too.txt"}),
    ]
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
    [log] = (tmp_path / "kimi-home").glob("sessions/*/*/agents/main/wire.jsonl")
    resolved = [
        json.loads(line) for line in log.read_text().splitlines() if "interaction.resolved" in line
    ]
    assert [
        r["response"].get("scope") for r in resolved if r["response"]["decision"] == "approved"
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
    process — so each command asks a person and runs only once approved."""
    home = tmp_path / "kimi-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(f'default_permission_mode = "{starts_in}"\n')
    endpoint.plan = [
        ("Bash", {"command": "touch made.txt"}),
        ("Bash", {"command": "touch too.txt"}),
    ]
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
    if not results or "dismissed" in results[-1]:
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
    two approaches — again when its review was dismissed (the run paused on it) — and then
    says what the review answered."""
    messages = body["messages"]
    results = [result_text(m["content"]) for m in messages if m.get("role") == "tool"]
    if not results:
        said = "\n".join(text_of(m.get("content")) for m in messages)
        return ("Write", {"path": _PLAN_FILE.findall(said)[-1], "content": PLAN})
    if len(results) == 1 or "dismissed" in results[-1]:
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
    """A sub-agent whose work writes a file (its ``Bash``: Kimi asks first)."""
    if body["messages"][-1].get("role") == "tool":
        return "made it"
    return ("Bash", {"command": "touch made.txt"})


async def test_a_background_sub_agents_call_kimi_asks_about_is_governed_in_its_own_run(
    harness: Harness, tmp_path: Path
) -> None:
    """The background sub-agent's ``Bash`` asks permission after its turn ended (Kimi's
    approvals are its session's, not a turn's): the call is decided in the sub-agent's own run
    — a person is asked, so that run pauses, and the conversation's next turn asks it. Approved,
    the command runs, the task ends and its run with it; then the turn's own prompt goes."""
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
        assert call is not None and call.tool == "Bash"
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
    paused = await agent.run("wait a bit", user="u1", thread="t1")
    assert paused.interrupt is not None  # Kimi asks about a command that writes
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and "task_id:" in str(done.answer), done.error
    made = tmp_path / "kimi-work" / "done.txt"
    assert not made.exists()  # still at work when the turn ended
    async with asyncio.timeout(20):
        while not made.exists():  # noqa: ASYNC110 - a file the command writes
            await asyncio.sleep(0.1)


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
