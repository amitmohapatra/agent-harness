"""Claude Code's own shell runs through the harness's isolation layer (``sandbox.relay``): the
CLI the SDK bundles (2.1.325) drives a scripted model (``tests/support/claude_code.py``), and
every ``Bash`` command it would run itself, as the service's user, runs in the run's workspace
as the run's uid instead — what Claude reads of each call is what Claude Code makes of the
command run itself. Each test tries to reach what a run's command must not — another run's
workspace, the service's environment and files, other processes — or checks the limits, the
clean-up, the audit, a background command, the CLI's own timeout, Claude Code's own rules, the
files Claude writes itself, a machine that cannot isolate, and the working files a
conversation carries. Run as root (Claude Code works in the run's workspace)."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import secrets
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.claude_code import ScriptedClaude, claude_target
from tests.support.isolation import fresh_root, processes_of, root_helper
from trellis import Harness, Settings
from trellis.contracts import RunEvent, RunEventType
from trellis.harness.sandbox.base import RUN
from trellis.harness.sandbox.isolation import AUDIT, HostSandbox, Limits, use_host

SECRET = "sk-service-claude-0123456789"


@pytest.fixture
def root() -> Iterator[str]:
    helper = root_helper()
    if helper:
        pytest.skip("Claude Code works in the run's workspace only where the service is root")
    yield from fresh_root(helper)


@pytest.fixture
def layer(root: str) -> Iterator[HostSandbox]:
    # uids of its own: another test's layer (another root) may still be ending its runs
    base = 1_500_000_000 + secrets.randbelow(100_000) * 1_000
    made = HostSandbox(root, limits=Limits(processes=64), uid_base=base, tenants=1)
    replaced = use_host(made)
    try:
        yield made
    finally:
        use_host(replaced)


@pytest.fixture
async def model() -> AsyncIterator[ScriptedClaude]:
    found = ScriptedClaude([])
    async with found.serving():
        yield found


async def ran(
    harness: Harness,
    model: ScriptedClaude,
    tmp_path: Path,
    script: list[dict[str, Any]],
    *,
    thread: str | None = None,
    user: str = "u1",
    **fields: Any,
) -> list[RunEvent]:
    """The run's events, its script played; what the model read is ``said(model)``."""
    model.script = script
    agent = harness.agents.get("coder") or harness.wrap(
        claude_target(model, tmp_path, **fields), id="coder"
    )
    return [e async for e in agent.stream("go", user=user, thread=thread)]


def bash(command: str, **more: Any) -> dict[str, Any]:
    return {"tool": "Bash", "input": {"command": command, "description": "run it", **more}}


def said(model: ScriptedClaude) -> list[str]:
    return [r["text"] for r in model.results]


def customs(events: list[RunEvent], name: str) -> list[dict[str, Any]]:
    return [e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == name]


def succeeded(events: list[RunEvent]) -> bool:
    return events[-1].type is RunEventType.RUN_FINISHED and events[-1].outcome == "success"


async def test_a_command_runs_in_the_runs_workspace_as_its_uid_and_claude_reads_it_as_its_own(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    relays = set(glob.glob(f"{tempfile.gettempdir()}/trellis-relay-*"))
    events = await ran(
        harness,
        model,
        tmp_path,
        [
            bash("id -u; pwd; echo err >&2; exit 3"),
            bash("mkdir -p sub && cd sub"),
            bash("pwd; ls -ld ."),
            {"text": "done"},
        ],
    )
    assert succeeded(events)
    failed, empty, where = said(model)
    code, uid, home, err = failed.splitlines()
    # what Claude Code makes of a command that failed: its exit code, its stdout, its stderr
    assert code == "Exit code 3" and err == "err"
    assert int(uid) >= layer.isolation.uid_base
    assert home.startswith(f"{layer.isolation.root}/t-")
    assert empty == "(Bash completed with no output)"
    # a cd holds for the next command, as in Claude Code's own shell
    assert where.splitlines()[0] == f"{home}/sub"
    assert f" {uid} " in where.splitlines()[1]
    # the run's end removed its workspace (the folder Claude worked in) and its relay
    assert not await asyncio.to_thread(os.path.exists, home)
    assert await layer.labelled({RUN: events[-1].run_id}) == []
    assert processes_of(int(uid)) == []
    assert set(glob.glob(f"{tempfile.gettempdir()}/trellis-relay-*")) <= relays
    audited = customs(events, "sandbox_command")
    assert [a["command"] for a in audited][:3] == [
        "id -u; pwd; echo err >&2; exit 3",
        "mkdir -p sub && cd sub",
        "pwd; ls -ld .",
    ]
    assert audited[0]["exit_code"] == 3 and audited[0]["uid"] == int(uid)


async def test_the_services_environment_files_and_processes_are_out_of_reach(
    harness: Harness,
    layer: HostSandbox,
    model: ScriptedClaude,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRELLIS_TEST_SECRET", SECRET)
    secret_file = Path(os.getcwd()) / ".claude-isolation-secret"
    secret_file.write_text(SECRET)
    try:
        await ran(
            harness,
            model,
            tmp_path,
            [
                bash("env"),
                bash(f"cat /proc/{os.getpid()}/environ; cat {secret_file}; ls /root"),
                bash("ls /proc | grep -c '^[0-9]'"),
                bash(f"kill -0 {os.getpid()}"),
                {"text": "done"},
            ],
        )
    finally:
        secret_file.unlink()
    env, reads, processes, signal = said(model)
    assert SECRET not in env and "ANTHROPIC_API_KEY" not in env
    assert SECRET not in reads
    assert int(processes) <= 4  # its own shell and what it ran: no other process is seen
    assert signal.startswith("Exit code 1")


async def test_claudes_own_file_tools_reach_no_file_of_the_service(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    """Claude Code's ``Read`` and ``Write`` run in its own process, as the service's user, not
    in the workspace: a path outside the run's folders — a secret file of the service's — is
    refused (Claude Code asks about it, and the harness says no), never read or written; the
    run's own folder is read as ever."""
    secret_file = Path(os.getcwd()) / ".claude-read-secret"
    secret_file.write_text(SECRET)
    planted = Path(os.getcwd()) / ".claude-planted"
    try:
        events = await ran(
            harness,
            model,
            tmp_path,
            [
                {"tool": "Read", "input": {"file_path": str(secret_file)}},
                {"tool": "Write", "input": {"file_path": str(planted), "content": "x"}},
                bash("echo mine > notes.md"),
                {"tool": "Read", "input": {"file_path": "notes.md"}},
                {"text": "done"},
            ],
        )
        assert not planted.exists()
    finally:
        secret_file.unlink()
        planted.unlink(missing_ok=True)
    assert succeeded(events)
    read, wrote, _, again = said(model)
    assert SECRET not in read and read.startswith("Read was not run: ")
    assert "outside the folders this run works in" in read
    assert wrote.startswith("Write was not run: ")
    assert "mine" in again  # the run's own folder: read and written as ever


async def test_another_runs_workspace_is_out_of_reach(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    other = await layer.workspace("default", "u2", "another-run")
    await other.write("notes.txt", b"the other run's")
    command = f"cat {other.home}/notes.txt; ls {other.home}; touch {other.home}/x"
    try:
        await ran(harness, model, tmp_path, [bash(command), {"text": "x"}])
    finally:
        await layer.delete(other.ref)
    [read] = said(model)
    assert read.startswith("Exit code 1")
    assert "the other run's" not in read and read.count("Permission denied") == 3


async def test_the_layers_limits_hold_and_the_run_goes_on(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    script = [bash("ulimit -u; ulimit -t"), bash("echo still here"), {"text": "x"}]
    await ran(harness, model, tmp_path, script)
    limits, after = said(model)
    assert limits.split() == ["64", "120"] and after == "still here"


async def test_a_background_command_goes_on_in_the_workspace_after_its_call_returns(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    await ran(
        harness,
        model,
        tmp_path,
        [
            bash("sleep 1; id -u > bg.txt", run_in_background=True),
            bash("cat bg.txt || echo not yet"),
            bash("sleep 2; cat bg.txt"),
            {"text": "x"},
        ],
    )
    started, early, late = said(model)
    assert "running in background" in started
    assert "not yet" in early
    assert int(late) >= layer.isolation.uid_base


async def test_a_command_past_claude_codes_timeout_is_killed_with_the_run(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    """Claude Code moves a command past its call's timeout to the background (2.1.325); the
    query's end kills it in the workspace, with everything it started."""
    script = [bash("id -u; sleep 300", timeout=1500), bash("echo next"), {"text": "x"}]
    started = time.monotonic()
    events = await ran(harness, model, tmp_path, script)
    moved, after = said(model)
    assert "moved to the background" in moved and after == "next"
    assert succeeded(events) and time.monotonic() - started < 60
    uids = {a["uid"] for a in customs(events, "sandbox_command")}
    assert all(processes_of(uid) == [] for uid in uids)


async def test_claude_codes_own_rules_see_the_command_the_model_made(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    rules = json.dumps({"permissions": {"deny": ["Bash(rm:*)"]}})
    script = [bash("rm -rf notes"), bash("echo allowed"), {"text": "x"}]
    await ran(harness, model, tmp_path, script, settings=rules)
    denied, allowed = said(model)
    assert "denied" in denied and "rm -rf notes" in denied
    assert allowed == "allowed"


async def test_a_file_claude_writes_itself_in_the_workspace_is_the_runs(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    await ran(
        harness,
        model,
        tmp_path,
        [
            bash("pwd"),
            {"tool": "Write", "input": {"file_path": "{last}/notes/a.txt", "content": "one\n"}},
            bash("echo two >> notes/a.txt && touch notes/b.txt && cat notes/a.txt"),
            {"text": "x"},
        ],
        permission_mode="acceptEdits",
    )
    *_, appended = said(model)
    assert appended == "one\ntwo"


async def test_where_commands_cannot_be_isolated_claude_codes_shell_is_off(
    harness: Harness, root: str, model: ScriptedClaude, tmp_path: Path, caplog: Any
) -> None:
    from trellis.harness import host

    host._unisolated.cache_clear()
    unable = HostSandbox(root, helper=("false",))
    replaced = use_host(unable)
    try:
        events = await ran(harness, model, tmp_path, [bash("id -u"), {"text": "x"}])
    finally:
        use_host(replaced)
    [refused] = said(model)
    assert "id -u" not in refused and ("No such tool" in refused or "not" in refused)
    [warned] = [
        r.getMessage() for r in caplog.records if "shell commands are off" in r.getMessage()
    ]
    assert "cannot be isolated" in warned and "helper false" in warned
    assert succeeded(events)


async def test_a_conversations_working_files_carry_to_its_next_turn_and_to_no_one_else(
    layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    made = (
        "mkdir -p data outputs __pycache__ && echo kept > data/a.txt && echo out > outputs/r.txt"
        " && echo c > __pycache__/x"
    )
    async with Harness(config=Settings()) as first:
        await ran(first, model, tmp_path, [bash(made), {"text": "x"}], thread="t1")
    async with Harness(config=Settings()) as second:  # another worker, the same file store
        events = await ran(
            second, model, tmp_path, [bash("cat data/a.txt; ls"), {"text": "x"}], thread="t1"
        )
        [listed] = said(model)
        told = model.sent[-1]
        await ran(second, model, tmp_path, [bash("cat data/a.txt"), {"text": "x"}], thread="t2")
        [other_thread] = said(model)
        script = [bash("cat data/a.txt"), {"text": "x"}]
        await ran(second, model, tmp_path, script, thread="t1", user="u2")
        [other_user] = said(model)
    assert succeeded(events)
    assert listed.splitlines() == ["kept", "data"]  # not the outputs, nor the cache
    restored = "Restored from this conversation's earlier turns" in told and "data/a.txt" in told
    assert restored  # (the prompt itself is not shown: it holds the CLI's own context)
    assert other_thread.startswith("Exit code 1") and other_user.startswith("Exit code 1")


async def test_the_audit_log_has_each_command_with_its_user_and_run(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path, caplog: Any
) -> None:
    caplog.set_level(logging.INFO, logger=AUDIT)
    events = await ran(harness, model, tmp_path, [bash("echo audited"), {"text": "x"}])
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == AUDIT]
    mine = [r for r in records if r["command"] == "echo audited"]
    assert mine and mine[0]["user"] == "u1" and mine[0]["run"] == events[-1].run_id


async def test_the_teams_own_permission_check_decides_first_and_its_rewrite_is_what_runs(
    harness: Harness, layer: HostSandbox, model: ScriptedClaude, tmp_path: Path
) -> None:
    """The options' own ``can_use_tool`` is asked before a command runs in the workspace: what
    it denies does not run; the command it rewrites is the one that runs there."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        if "secret" in str(args.get("command")):
            return PermissionResultDeny(message="not that one")
        return PermissionResultAllow(updated_input={**args, "command": "echo rewritten"})

    script = [bash("cat secret.txt"), bash("echo as made"), {"text": "x"}]
    await ran(harness, model, tmp_path, script, can_use_tool=own)
    denied, rewritten = said(model)
    assert "not that one" in denied and rewritten == "rewritten"
