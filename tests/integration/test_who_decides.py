"""Who decides what, for the frameworks that run their own tools (Claude Code, Kimi Code): the
run's user decides only the actions taken in their name; infrastructure is the
administrator's (``docs/governance.md#who-decides-what``).

| Tool | Decided by |
|---|---|
| ``Read``, ``Grep``, ``Glob`` (Kimi's reads) in the user's workspace | nobody: they run unasked |
| ``Write``, ``Edit`` of the user's own files | the user, under the one rule: once, for this conversation, always |
| the shell in the sandbox | nobody: it runs unasked, announced and recorded |
| the shell on the host | runs as the framework runs it, unasked and announced, unless the administrator says ``TRELLIS_HOST_EXECUTION=deny`` |
| the network out (``WebFetch``, ``WebSearch``; Kimi's ``FetchURL``, ``WebSearch``) | the administrator, in the tool catalog: refused until it gives the tool a risk or an ``approve_when`` |
| MCP servers, permission modes | the deployment: on ``h.wrap``, never a run's own ``framework_options`` |
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow

from tests.support.adapters import CLI
from tests.support.memory import FakeMemoryService
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, Settings, sandbox
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness.adapters.acp import KIMI, ACPAgent
from trellis.harness.events import NOTICE
from trellis.harness.governance.decision import ALWAYS, CONVERSATION, DENY, ONCE

FAKE_ACP = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
#: Claude Code's sandbox with no command let out of it: no host execution to allow.
BOXED = {"enabled": True, "allowUnsandboxedCommands": False}
#: What the model reads of a network call the catalog says nothing of.
UNLISTED = "the tool catalog does not allow it"
#: What the model reads of an infrastructure call a person would have been asked about.
INFRASTRUCTURE_DENIED = "is the platform's infrastructure, and a call of it that needs approval"


def claude(tmp: Path, *steps: dict[str, Any], **fields: Any) -> ClaudeAgentOptions:
    """Claude Code (the scripted CLI) making ``steps`` in its sandbox, unless ``fields`` say
    otherwise."""
    fields.setdefault("sandbox", BOXED)
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps([*steps, {"text": "done: {last}"}]),
            "FAKE_CLAUDE_SESSIONS": str(tmp),
            "FAKE_CLAUDE_BUILTINS": str(tmp / "builtins.jsonl"),
        },
        **fields,
    )


def ran(tmp: Path) -> list[str]:
    """The built-ins Claude Code (or the scripted ACP agent) ran, in order."""
    log = tmp / "builtins.jsonl"
    return (
        [json.loads(line)["tool"] for line in log.read_text().splitlines()] if log.exists() else []
    )


def kimi_fake(tmp: Path, *steps: dict[str, Any], **fields: Any) -> ACPAgent:
    """The scripted ACP agent as Kimi Code, its home and record in ``tmp``."""
    env = {
        "FAKE_ACP_SCRIPT": json.dumps([*steps, {"text": "done: {last}"}]),
        "FAKE_ACP_RECORD": str(tmp / "record.json"),
        "KIMI_CODE_HOME": str(tmp / "home"),
    }
    return ACPAgent(
        command=(sys.executable, FAKE_ACP), env=env, cwd=str(tmp), profile=KIMI, **fields
    )


def notices(events: list[Any]) -> list[str]:
    return [e.data["tool"] for e in events if e.data.get("name") == NOTICE]


def allowing_host(memory: FakeMemoryService | None = None) -> Harness:
    config = Settings(host_execution="allow")
    return Harness(config=config, memory=memory.client() if memory else False)


# --------------------------------------------------------------------------- Claude Code
async def test_claude_reads_in_the_workspace_run_unasked(harness: Harness, tmp_path: Path) -> None:
    reads = [{"builtin": name, "args": {"pattern": "x"}} for name in ("Read", "Grep", "Glob")]
    done = await harness.wrap(claude(tmp_path, *reads), id="reads").run("look", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
    assert ran(tmp_path) == ["Read", "Grep", "Glob"]


async def test_claude_writes_are_the_users_to_decide_under_the_one_rule(
    memory_harness: Harness, tmp_path: Path
) -> None:
    for n, name in enumerate(("Write", "Edit")):
        step = {"builtin": name, "args": {"file_path": f"notes-{n}.md"}}
        paused = await memory_harness.wrap(claude(tmp_path, step), id=name).run("w", user="ada")
        assert paused.interrupt is not None, paused
        assert list(paused.interrupt.options) == [ONCE, CONVERSATION, ALWAYS, DENY]
    assert ran(tmp_path) == []


async def test_claude_shell_in_its_sandbox_runs_unasked_announced_and_recorded(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Claude Code's own permission question about a command in its sandbox is not the user's:
    the command runs, announced and recorded."""
    step = {"builtin": "Bash", "args": {"command": "rm -rf build"}}
    agent = memory_harness.wrap(claude(tmp_path, step), id="boxed")
    events = [e async for e in agent.stream("clean", user="ada")]
    assert events[-1].error is None, events[-1].error
    assert ran(tmp_path) == ["Bash"] and notices(events) == ["Bash"]
    assert not any(e.type is RunEventType.INTERRUPT for e in events)
    await memory_harness.writes.drain()
    assert [c.body["tool"] for c in memory_service.calls if c.name == "record_tool"] == ["Bash"]


async def test_claude_shell_on_the_host_is_the_administrators_never_the_users(
    tmp_path: Path,
) -> None:
    """Refused where the administrator says ``deny``; by default it runs as Claude Code runs
    it, as the service's user: announced, never asked of the user."""
    step = {"builtin": "Bash", "args": {"command": "make deploy"}}
    deny = Settings(host_execution="deny")
    with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION=deny"):
        Harness(config=deny, memory=False).wrap(claude(tmp_path, step, sandbox=None), id="h")
    agent = allowing_host().wrap(claude(tmp_path, step, sandbox=None), id="host")
    events = [e async for e in agent.stream("deploy", user="ada")]
    assert events[-1].error is None, events[-1].error
    assert ran(tmp_path) == ["Bash"] and notices(events) == ["Bash"]


@pytest.mark.parametrize("name", ["WebFetch", "WebSearch"])
async def test_claude_network_out_is_refused_until_the_catalog_allows_it(
    tmp_path: Path, name: str
) -> None:
    """The network out is the administrator's policy in the tool catalog, not a question to
    the user: with no word there (no catalog, or no risk and no rule for the tool) the call is
    refused and the model reads why; the risk ``read`` lets it run unasked; an
    ``approve_when`` that holds would ask a person, and the network is the platform's
    infrastructure: the call is denied, asked of nobody."""
    step = {"builtin": name, "args": {"url": "https://example.com/x"}}
    async with Harness(config=Settings(), memory=False) as h:
        done = await h.wrap(claude(tmp_path, step), id="off").run("fetch", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
    assert UNLISTED in str(done.answer) and ran(tmp_path) == []

    service = FakeMemoryService()
    async with Harness(config=Settings(), memory=service.client()) as h:
        unlisted = await h.wrap(claude(tmp_path, step), id="unlisted").run("fetch", user="ada")
        assert UNLISTED in str(unlisted.answer) and ran(tmp_path) == [], unlisted
        service.catalog[name] = {"name": name, "side_effects": "read"}
        allowed = await h.wrap(claude(tmp_path, step), id="allowed").run("fetch", user="ada")
        assert allowed.status is RunStatus.SUCCESS and allowed.interrupt is None, allowed
        assert ran(tmp_path) == [name]
        service.catalog[name] = {"name": name, "approve_when": "url != ''"}
        ruled = await h.wrap(claude(tmp_path, step), id="ruled").run("fetch", user="ada")
        assert ruled.status is RunStatus.SUCCESS and ruled.interrupt is None, ruled
        assert INFRASTRUCTURE_DENIED in str(ruled.answer) and ran(tmp_path) == [name], ruled


async def test_claude_mcp_servers_and_permission_modes_are_the_deployments(
    harness: Harness, tmp_path: Path
) -> None:
    """On ``h.wrap`` (the deployment's code) the options may name MCP servers, pre-approved
    tools and a permission mode that lets calls through; a run's own ``framework_options``
    may not."""
    servers = {"erp": {"type": "stdio", "command": "erp-mcp", "args": []}}
    agent = harness.wrap(
        claude(tmp_path, mcp_servers=servers, permission_mode="acceptEdits"), id="deployed"
    )
    for given in (
        {"permission_mode": "bypassPermissions"},
        {"permission_mode": "acceptEdits"},
        {"mcp_servers": servers},
        {"allowed_tools": ["Write"]},
    ):
        with pytest.raises(ConfigurationError, match=r"h\.wrap"):
            await agent.run("x", user="ada", framework_options=given)
    planned = await agent.run("plan it", user="ada", framework_options={"permission_mode": "plan"})
    assert planned.status is RunStatus.SUCCESS, planned


async def allow_everything(*_: Any) -> PermissionResultAllow:
    return PermissionResultAllow()


@pytest.mark.parametrize(
    "given",
    [
        {"extra_args": {"permission-mode": "bypassPermissions"}},
        {"extra_args": {"allowedTools": "Write"}},
        {"settings": json.dumps({"permissions": {"allow": ["Write"]}})},
    ],
    ids=["flag-mode", "flag-allowed", "settings"],
)
async def test_claudes_permissions_are_loosened_by_no_other_run_option(
    memory_harness: Harness, tmp_path: Path, given: dict[str, Any]
) -> None:
    """What lets Claude Code run a call without asking — the CLI's own flags, its settings'
    permission rules — is the deployment's as its permission mode is: a run's own
    ``framework_options`` saying it is refused, and the user decides the write under the one
    rule."""
    step = {"builtin": "Write", "args": {"file_path": "notes.md"}}
    agent = memory_harness.wrap(claude(tmp_path, step), id="loosened")
    with pytest.raises(ConfigurationError, match=r"h\.wrap"):
        await agent.run("w", user="ada", framework_options=given)
    paused = await agent.run("w", user="ada")
    assert paused.interrupt is not None and ran(tmp_path) == [], paused


async def test_a_runs_own_permission_check_never_approves_for_the_user(
    memory_harness: Harness, tmp_path: Path
) -> None:
    """A run's own ``can_use_tool`` is asked when Claude Code would ask its user, and may
    refuse the call; one that lets it through does not answer for the user, who is asked as
    without it. (The deployment's own, on the options it wraps, answers as it is.)"""
    step = {"builtin": "Write", "args": {"file_path": "notes.md"}}
    agent = memory_harness.wrap(claude(tmp_path, step), id="checked")
    given = {"can_use_tool": allow_everything}
    paused = await agent.run("w", user="ada", framework_options=given)
    assert paused.interrupt is not None and ran(tmp_path) == [], paused
    deployed = memory_harness.wrap(claude(tmp_path, step, can_use_tool=allow_everything), id="d")
    allowed = await deployed.run("w", user="ada")
    assert allowed.status is RunStatus.SUCCESS and ran(tmp_path) == ["Write"], allowed


# --------------------------------------------------------------------------- Kimi Code
async def test_kimi_reads_run_unasked(tmp_path: Path) -> None:
    steps = [
        {"builtin": name, "args": {"pattern": "x"}, "ask": True, "output": "1"}
        for name in ("Read", "Grep", "Glob")
    ]
    done = await allowing_host().wrap(kimi_fake(tmp_path, *steps), id="r").run("look", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done


async def test_kimi_writes_are_the_users_to_decide_under_the_one_rule(
    memory_harness: Harness, tmp_path: Path
) -> None:
    step = {"builtin": "Write", "args": {"path": "notes.md", "content": "x"}, "ask": True}
    agent = memory_harness.wrap(kimi_fake(tmp_path, step, sandbox=sandbox(FakeSandboxes())), id="w")
    paused = await agent.run("write", user="ada")
    assert paused.interrupt is not None, paused
    assert list(paused.interrupt.options) == [ONCE, CONVERSATION, ALWAYS, DENY]


async def test_kimi_shell_in_its_sandbox_runs_unasked_and_announced(
    harness: Harness, tmp_path: Path
) -> None:
    boxes = FakeSandboxes()
    step = {"builtin": "Bash", "args": {"command": "echo built"}, "ask": True}
    agent = harness.wrap(kimi_fake(tmp_path, step, sandbox=sandbox(boxes)), id="boxed")
    events = [e async for e in agent.stream("build", user="ada")]
    assert events[-1].error is None, events[-1].error
    assert notices(events) == ["Bash"]
    assert any(c.endswith("&& echo built'") for c in boxes.commands)


async def test_kimi_shell_on_the_host_is_the_administrators_never_the_users(
    tmp_path: Path,
) -> None:
    step = {"builtin": "Bash", "args": {"command": "rm a.txt"}, "ask": True, "output": ""}
    deny = Settings(host_execution="deny")
    with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION=deny"):
        Harness(config=deny, memory=False).wrap(kimi_fake(tmp_path, step), id="h")
    agent = allowing_host().wrap(kimi_fake(tmp_path, step), id="host")
    events = [e async for e in agent.stream("tidy", user="ada")]
    assert events[-1].error is None, events[-1].error
    assert notices(events) == ["Bash"]
    assert not any(e.type is RunEventType.INTERRUPT for e in events)


@pytest.mark.parametrize("name", ["FetchURL", "WebSearch"])
async def test_kimi_network_out_is_refused_until_the_catalog_allows_it(
    tmp_path: Path, name: str
) -> None:
    """Kimi is configured to ask about its network tools (its own policy runs them unasked),
    so the catalog decides: refused with no word there; a risk lets them run unasked."""
    step = {"builtin": name, "args": {"url": "https://example.com/x"}, "ask": True, "output": "p"}
    service = FakeMemoryService()
    async with allowing_host(service) as h:
        refused = await h.wrap(kimi_fake(tmp_path, step), id="unlisted").run("f", user="ada")
        assert refused.status is RunStatus.SUCCESS and refused.interrupt is None, refused
        assert UNLISTED in json.loads((tmp_path / "record.json").read_text())["prompts"][-1]
        service.catalog[name] = {"name": name, "side_effects": "read"}
        allowed = await h.wrap(kimi_fake(tmp_path, step), id="allowed").run("f", user="ada")
        assert allowed.status is RunStatus.SUCCESS and allowed.answer == "done: p", allowed
    config = next((tmp_path / "home").parent.glob("**/trellis/kimi/*/config.toml"), None)
    assert config is None or f'"{name}"' in config.read_text()


async def test_kimi_mcp_servers_are_the_deployments(harness: Harness, tmp_path: Path) -> None:
    server = {"name": "notes", "command": sys.executable, "args": ["-c", "pass"]}
    agent = harness.wrap(
        kimi_fake(tmp_path, sandbox=sandbox(FakeSandboxes())),
        id="deployed",
        framework_options={"mcp_servers": [server]},
    )
    other = {**server, "name": "shell-anything"}
    with pytest.raises(ConfigurationError, match=r"h\.wrap"):
        await agent.run("x", user="ada", framework_options={"mcp_servers": [server, other]})
    with pytest.raises(ConfigurationError, match="yolo"):
        await agent.run("x", user="ada", framework_options={"mode": "yolo"})
