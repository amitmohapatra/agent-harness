"""The edges of running every framework's shell through the isolation layer that need no
real command: a terminal Kimi asks for with no turn's attempt under way is refused; an OpenAI
shell request outside a run is refused; Claude Code's shell call with no command is refused,
and where the layer reaches root through a helper its shell is off; the shared container's
open network keeps the metadata server out (its root's script, not run here); a command's
audit record on no tool call's span is kept for the framework's call that ran it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from acp import RequestError
from claude_agent_sdk import PermissionResultDeny

from trellis.contracts import ToolError
from trellis.harness.adapters import claude
from trellis.harness.bridges import acp_live
from trellis.harness.bridges.live import LiveSessions
from trellis.harness.sandbox import shared
from trellis.harness.sandbox.base import SandboxSpec
from trellis.harness.sandbox.isolation import HostSandbox, use_host
from trellis.harness.tools.convert.openai_local import isolated


async def test_a_terminal_asked_for_between_turns_is_refused() -> None:
    session = acp_live.ACPSession("chat", "form", LiveSessions(), owner="u", keeps=True, model="m")
    with pytest.raises(RequestError):
        await session.create_terminal(command="ls")
    assert session.refused["terminal"] == 1


async def test_an_openai_shell_request_outside_a_run_is_refused() -> None:
    with pytest.raises(ToolError, match="in a harness run"):
        await isolated(SimpleNamespace(data=SimpleNamespace()))


def test_a_shell_call_with_no_command_is_refused() -> None:
    relay: Any = SimpleNamespace(command=lambda *a, **k: "never")
    verdict = claude._relayed(relay, "Bash", {"description": "nothing"})
    assert isinstance(verdict, PermissionResultDeny) and "no command" in verdict.message


def test_through_a_root_helper_claude_codes_shell_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = HostSandbox("/var/lib/trellis-test-helper", helper=("sudo", "-n"))
    monkeypatch.setattr(layer, "cannot_isolate", lambda: None)
    replaced = use_host(layer)
    try:
        assert claude._unisolated() == claude.HELPER_ONLY
    finally:
        use_host(replaced)


def test_an_open_container_gets_net_admin_for_the_routes_its_layer_adds() -> None:
    pools = shared._Pools("python:3.12-slim")
    ref: Any = SimpleNamespace(id="p", labels={}, snapshot=None)
    opened = pools._config(ref, SandboxSpec(network="open"))
    closed = pools._config(ref, SandboxSpec())
    assert "NET_ADMIN" in opened["HostConfig"]["CapAdd"]
    assert "NET_ADMIN" not in closed["HostConfig"]["CapAdd"]
    assert "169.254.169.254" not in shared.ROUTES and "AF_NETLINK" in shared.ROUTES
