"""Shell commands are allowed by default; ``TRELLIS_HOST_EXECUTION=deny`` is the
administrator's switch that turns off every shell outside a sandbox (``trellis.harness.host``).

* The harness's own commands on this machine go through its isolation layer: ``sandbox()``
  with no provider is Docker when its daemon answers, else ``HostSandbox`` when it can isolate,
  else no shell, each command refused with the reason (said once).
* Claude Code's and Kimi Code's own shells run as they do, announced and recorded; a team's own
  Claude ``SandboxSettings`` is honoured as given.
* Deep Agents' ``LocalShellBackend``, LangChain's host shell and the OpenAI SDK's Unix-local
  sandbox run with no isolation: refused at ``h.wrap`` whatever the setting, naming
  ``sandbox()``.
* ``deny`` refuses, per adapter, Claude Code's shell (``Bash``, ``PowerShell``, ``REPL``,
  ``Monitor``) and Kimi's ``Bash`` on the host, an OpenAI local shell tool, and the isolation
  layer's own commands; commands in a sandbox are not affected."""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from tests.support.adapters import CLI
from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes
from tests.unit.test_host_execution import HOST_PATHS, UNISOLATED_PATHS
from trellis import Harness, ReAct, Settings, sandbox
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import sandbox as sandbox_module
from trellis.harness.adapters.acp import kimi
from trellis.harness.sandbox import HostSandbox, SandboxRef
from trellis.harness.sandbox.shared import SharedDockerSandbox


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


def deny() -> Harness:
    return Harness(config=Settings(host_execution="deny"), memory=False)


# --------------------------------------------------------------------------- the setting
def test_shell_commands_are_allowed_by_default_and_deny_is_the_only_other_value() -> None:
    assert Settings().host_execution == "allow"
    assert Settings.from_env({}).host_execution == "allow"
    assert Settings.from_env({"TRELLIS_HOST_EXECUTION": "deny"}).host_execution == "deny"
    with pytest.raises(ValueError, match="host_execution"):
        Settings.from_env({"TRELLIS_HOST_EXECUTION": "maybe"})


@pytest.mark.parametrize("way", sorted(HOST_PATHS))
def test_deny_switches_each_adapters_shell_off_and_the_default_wraps_it(way: str) -> None:
    target, options, fix = HOST_PATHS[way]
    with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION=deny") as refused:
        deny().wrap(target(), id=way, framework_options=options())
    assert fix in str(refused.value)
    allowed = Harness(config=Settings(), memory=False)
    allowed.wrap(target(), id=way, framework_options=options())
    assert way in allowed.agents


@pytest.mark.parametrize("way", sorted(UNISOLATED_PATHS))
def test_an_unisolated_shell_is_refused_whatever_the_setting(way: str) -> None:
    target, options, fix = UNISOLATED_PATHS[way]
    for value in ("allow", "deny"):
        h = Harness(config=Settings(host_execution=value), memory=False)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ConfigurationError, match=r"sandbox\(\)") as refused:
            h.wrap(target(), id=way, framework_options=options())
        assert fix in str(refused.value)


def test_deny_leaves_commands_in_a_sandbox_alone() -> None:
    h = deny()
    boxed: Any = {"enabled": True, "allowUnsandboxedCommands": False}
    h.wrap(ClaudeAgentOptions(sandbox=boxed), id="claude-boxed")
    h.wrap(kimi(sandbox=sandbox(FakeSandboxes())), id="kimi-boxed")
    h.wrap(ReAct("s", ScriptedChatModel(turns=[]), backend=sandbox(FakeSandboxes())), id="react")
    assert set(h.agents) == {"claude-boxed", "kimi-boxed", "react"}


# --------------------------------------------------------------------------- Claude Code
async def test_a_teams_own_claude_sandbox_is_honoured_as_given(tmp_path: Path) -> None:
    """The harness turns nothing of Claude Code's sandbox on or off: the team's settings go to
    the CLI as given, and none when it gave none."""
    given: Any = {"enabled": True, "excludedCommands": ["git"]}
    h = Harness(config=Settings(), memory=False)
    await h.wrap(claude(tmp_path, sandbox=given), id="given").run("hi", user="ada")
    argv = json.loads((tmp_path / "cli.json").read_text())["argv"]
    assert json.loads(argv[argv.index("--settings") + 1])["sandbox"] == given
    await h.wrap(claude(tmp_path), id="none").run("hi", user="ada")
    argv = json.loads((tmp_path / "cli.json").read_text())["argv"]
    assert "--settings" not in argv or "sandbox" not in json.loads(
        argv[argv.index("--settings") + 1]
    )


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
    assert len([r for r in caplog.records if "no root here" in r.getMessage()]) == 1
    given = FakeSandboxes()
    assert sandbox(given).provider_for() is given


async def test_with_no_shell_each_command_is_refused_with_the_reason() -> None:
    none = sandbox_module.NoShell("no unshare here")
    ref = SandboxRef(provider=none.name, id="trellis-r1", labels={})
    for call in (none.create(ref), none.attach(ref)):
        with pytest.raises(ConfigurationError, match=r"no Docker daemon answers.*no unshare here"):
            await call
    await none.delete(ref)
    assert await none.labelled({}) == []


async def test_deny_switches_the_isolation_layers_commands_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sandbox on this machine (``HostSandbox``) is a host command: under ``deny`` its first
    call is refused, naming the setting, and its card is not read."""
    monkeypatch.setattr(HostSandbox, "cannot_isolate", lambda self: None)
    box = sandbox(HostSandbox())
    turns = [("execute", {"command": "ls"}), "done"]
    scripted = ScriptedChatModel(turns=turns)
    agent = deny().wrap(ReAct("s", scripted, backend=box), id="host-box")
    done = await agent.run("ls", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert "TRELLIS_HOST_EXECUTION=deny" in str(scripted.seen[-1][-1].content)


@pytest.fixture
def daemon(tmp_path: Path) -> Iterator[str]:
    """A Docker daemon's ``/_ping`` on a unix socket: ``OK``."""
    path = str(tmp_path / "docker.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()

    def answer() -> None:
        conn, _ = server.accept()
        with conn:
            conn.recv(256)
            conn.sendall(b"HTTP/1.0 200 OK\r\n\r\nOK")

    threading.Thread(target=answer, daemon=True).start()
    yield path
    server.close()


def test_docker_answers_where_its_daemon_does(
    monkeypatch: pytest.MonkeyPatch, daemon: str, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox_module, "docker_host", lambda: f"unix://{daemon}")
    assert sandbox_module.docker_answers()
    monkeypatch.setattr(sandbox_module, "docker_host", lambda: f"unix://{tmp_path}/none.sock")
    assert not sandbox_module.docker_answers()
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    monkeypatch.setattr(sandbox_module, "docker_host", lambda: f"tcp://127.0.0.1:{port}")
    assert not sandbox_module.docker_answers()
    monkeypatch.setattr(sandbox_module, "docker_host", lambda: "npipe:////./pipe/docker")
    assert not sandbox_module.docker_answers()
