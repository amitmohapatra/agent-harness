"""Shell commands are allowed by default, and isolation is the guard
(``Settings.host_execution``: ``allow`` unless an administrator sets ``deny``).

* Claude Code: the harness turns its own sandbox on when the team gave none; its shell runs
  only where that sandbox can confine it (bubblewrap and socat on Linux, ``sandbox-exec`` on
  macOS) — otherwise each command is refused, and the model reads why (the real CLI, missing
  them, runs the command unsandboxed with a warning: never left to it). A team's own sandbox
  that lets commands out is refused at ``h.wrap``.
* Kimi Code: its own ``Bash`` on the host cannot go through the isolation layer, so with no
  ``sandbox=`` each command is refused, Kimi told how to give it one (``kimi(sandbox=sandbox())``,
  which falls back to the host's isolation layer).
* Deep Agents' ``LocalShellBackend``, LangChain's ``ShellToolMiddleware`` on the host and the
  OpenAI SDK's Unix-local sandbox run commands unisolated: refused at ``h.wrap`` whatever the
  setting, naming ``sandbox()``.
* ``sandbox()`` with no provider: Docker when its daemon answers, else the host's isolation
  layer (``HostSandbox``) when it can isolate, else no shell — each command refused with the
  reason, said once.
* ``TRELLIS_HOST_EXECUTION=deny`` switches host shells off (the administrator's)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from tests.support.adapters import CLI
from tests.support.sandbox import FakeSandboxes
from tests.unit.test_host_execution import HOST_PATHS, deep_local, react_local, shell_middleware
from trellis import Harness, Settings, sandbox
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import host
from trellis.harness import sandbox as sandbox_module
from trellis.harness.adapters.acp import KIMI, ACPAgent
from trellis.harness.sandbox import HostSandbox
from trellis.harness.sandbox.shared import SharedDockerSandbox

FAKE_ACP = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")


def claude(tmp: Path, *steps: dict[str, Any], **fields: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps([*steps, {"text": "said: {last}"}]),
            "FAKE_CLAUDE_SESSIONS": str(tmp),
            "FAKE_CLAUDE_RECORD": str(tmp / "cli.json"),
            "FAKE_CLAUDE_BUILTINS": str(tmp / "builtins.jsonl"),
        },
        **fields,
    )


def settings_sent(tmp: Path) -> dict[str, Any]:
    """The ``--settings`` the CLI was started with."""
    argv = json.loads((tmp / "cli.json").read_text())["argv"]
    return json.loads(argv[argv.index("--settings") + 1])


def ran(tmp: Path) -> list[str]:
    log = tmp / "builtins.jsonl"
    return (
        [json.loads(line)["tool"] for line in log.read_text().splitlines()] if log.exists() else []
    )


def kimi_fake(tmp: Path, *steps: dict[str, Any], **fields: Any) -> ACPAgent:
    env = {
        "FAKE_ACP_SCRIPT": json.dumps([*steps, {"text": "done: {last}"}]),
        "FAKE_ACP_RECORD": str(tmp / "record.json"),
        "KIMI_CODE_HOME": str(tmp / "home"),
    }
    return ACPAgent(
        command=(sys.executable, FAKE_ACP), env=env, cwd=str(tmp), profile=KIMI, **fields
    )


@pytest.fixture
def confined(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine whose Claude Code sandbox can confine its shell."""
    monkeypatch.setattr(host, "claude_unconfined", lambda: None)


# --------------------------------------------------------------------------- the setting
def test_shell_commands_are_allowed_by_default_and_deny_switches_them_off() -> None:
    assert Settings().host_execution == "allow"
    assert Settings.from_env({"TRELLIS_HOST_EXECUTION": "deny"}).host_execution == "deny"
    with pytest.raises(ValueError, match="host_execution"):
        Settings.from_env({"TRELLIS_HOST_EXECUTION": "maybe"})
    h = Harness(config=Settings(), memory=False)
    for way in ("claude-unsandboxed", "kimi-unsandboxed", "openai-shell-tool"):
        target, options, _ = HOST_PATHS[way]
        h.wrap(target(), id=way, framework_options=options())
    off = Harness(config=Settings(host_execution="deny"), memory=False)
    for way in ("claude-unsandboxed", "kimi-unsandboxed", "openai-shell-tool"):
        target, options, _ = HOST_PATHS[way]
        with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION"):
            off.wrap(target(), id=way, framework_options=options())


@pytest.mark.parametrize(
    "build", [deep_local, react_local, shell_middleware], ids=["deepagents", "react", "langchain"]
)
def test_an_unisolated_shell_is_refused_whatever_the_setting(build: Any) -> None:
    """``LocalShellBackend``, LangChain's host shell: no isolation the harness can give them —
    refused at ``h.wrap``, naming ``sandbox()``, allowed or not."""
    for value in ("allow", "deny"):
        h = Harness(config=Settings(host_execution=value), memory=False)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ConfigurationError, match=r"sandbox\(\)"):
            h.wrap(build(), id="unisolated")


# --------------------------------------------------------------------------- Claude Code
async def test_claude_gets_its_own_sandbox_when_the_team_gave_none(
    tmp_path: Path, confined: None
) -> None:
    h = Harness(config=Settings(), memory=False)
    step = {"builtin": "Bash", "args": {"command": "make test"}}
    done = await h.wrap(claude(tmp_path, step), id="boxed").run("test", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
    assert settings_sent(tmp_path)["sandbox"] == {
        "enabled": True,
        "allowUnsandboxedCommands": False,
    }
    assert ran(tmp_path) == ["Bash"]


async def test_claude_shell_is_refused_with_the_reason_where_its_sandbox_cannot_confine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without bubblewrap and socat the CLI would run the command unsandboxed (it says so on
    its stderr, and goes on): the harness refuses each command instead, and says why once."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(host.shutil, "which", lambda name: None)
    host.claude_unconfined.cache_clear()
    try:
        assert host.claude_unconfined() == (
            "Claude Code's sandbox needs bubblewrap (bwrap) and socat on Linux, and bwrap, "
            "socat are missing"
        )
        h = Harness(config=Settings(), memory=False)
        step = {"builtin": "Bash", "args": {"command": "make test"}}
        with caplog.at_level(logging.WARNING, logger="trellis.host"):
            done = await h.wrap(claude(tmp_path, step), id="unboxed").run("test", user="ada")
        assert done.status is RunStatus.SUCCESS, done
        assert "Bash was not run: commands cannot be confined here" in str(done.answer)
        assert "bubblewrap" in str(done.answer) and ran(tmp_path) == []
        assert any("bubblewrap" in r.getMessage() for r in caplog.records)
    finally:
        host.claude_unconfined.cache_clear()


@pytest.mark.parametrize(
    "platform, found, why",
    [
        ("darwin", {"sandbox-exec"}, None),
        ("darwin", set(), "Claude Code's sandbox needs sandbox-exec on macOS, and it is missing"),
        ("linux", {"bwrap", "socat"}, None),
        ("win32", set(), "Claude Code's sandbox runs on Linux and macOS, and this is win32"),
    ],
)
def test_where_claude_codes_sandbox_can_confine(
    monkeypatch: pytest.MonkeyPatch, platform: str, found: set[str], why: str | None
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(host.shutil, "which", lambda name: name if name in found else None)
    host.claude_unconfined.cache_clear()
    try:
        assert host.claude_unconfined() == why
    finally:
        host.claude_unconfined.cache_clear()


@pytest.mark.parametrize(
    "box",
    [
        {"enabled": False},
        {"enabled": True},
        {"enabled": True, "allowUnsandboxedCommands": False, "excludedCommands": ["git"]},
    ],
    ids=["off", "lets-commands-out", "excluded-commands"],
)
def test_a_teams_sandbox_that_lets_commands_out_is_refused(tmp_path: Path, box: Any) -> None:
    with pytest.raises(ConfigurationError, match="allowUnsandboxedCommands"):
        Harness(config=Settings(), memory=False).wrap(claude(tmp_path, sandbox=box), id="out")


def test_a_claude_with_no_shell_gets_no_sandbox(tmp_path: Path) -> None:
    assert host.claude_sandbox(ClaudeAgentOptions(tools=["Read"])) is None
    assert host.claude_sandbox(ClaudeAgentOptions()) == {
        "enabled": True,
        "allowUnsandboxedCommands": False,
    }
    given: Any = {"enabled": True, "allowUnsandboxedCommands": False, "network": {}}
    assert host.claude_sandbox(ClaudeAgentOptions(sandbox=given)) == given


# --------------------------------------------------------------------------- Kimi Code
async def test_kimi_bash_on_the_host_is_refused_naming_a_sandbox(tmp_path: Path) -> None:
    """Kimi's own ``Bash`` cannot go through the isolation layer: with no sandbox each command
    is refused and Kimi told how to have one; with ``sandbox()`` it runs, unasked."""
    step = {"builtin": "Bash", "args": {"command": "rm a.txt"}, "ask": True, "output": ""}
    h = Harness(config=Settings(), memory=False)
    done = await h.wrap(kimi_fake(tmp_path, step), id="host").run("tidy", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
    told = json.loads((tmp_path / "record.json").read_text())["prompts"][-1]
    assert "kimi(sandbox=sandbox())" in told
    boxed = h.wrap(kimi_fake(tmp_path, step, sandbox=sandbox(FakeSandboxes())), id="boxed")
    assert (await boxed.run("tidy", user="ada")).interrupt is None


# --------------------------------------------------------------------------- sandbox()
def test_a_sandbox_with_no_provider_falls_back_from_docker_to_the_hosts_isolation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sandbox_module, "docker_answers", lambda: True)
    assert isinstance(sandbox().provider_for(), SharedDockerSandbox)
    monkeypatch.setattr(sandbox_module, "docker_answers", lambda: False)
    monkeypatch.setattr(HostSandbox, "cannot_isolate", lambda self: None)
    assert isinstance(sandbox().provider_for(), HostSandbox)
    monkeypatch.setattr(HostSandbox, "cannot_isolate", lambda self: "no root here")
    with caplog.at_level(logging.WARNING, logger="trellis.sandbox"):
        none = sandbox().provider_for()
        sandbox().provider_for()
    assert isinstance(none, sandbox_module.NoShell) and none.why == "no root here"
    assert [r.getMessage() for r in caplog.records if "no root here" in r.getMessage()][:1]
    given = FakeSandboxes()
    assert sandbox(given).provider_for() is given


async def test_with_no_shell_each_command_is_refused_with_the_reason() -> None:
    from trellis.harness.sandbox import SandboxRef

    none = sandbox_module.NoShell("no Docker daemon answers, and no root here")
    ref = SandboxRef(provider=none.name, id="trellis-r1", labels={})
    with pytest.raises(ConfigurationError, match="no Docker daemon answers"):
        await none.create(ref)
    with pytest.raises(ConfigurationError, match="no Docker daemon answers"):
        await none.attach(ref)
    await none.delete(ref)
    assert await none.labelled({}) == []
