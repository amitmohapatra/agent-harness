"""The ACP adapter's parts that need no agent: its options, its model routing through Bifrost,
the version check, and how it reads a call's arguments and output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from trellis.contracts import ConfigurationError
from trellis.harness import compat
from trellis.harness.adapters import acp
from trellis.harness.adapters.acp import KIMI, KIMI_HEADERS, ACPAdapter, given, kimi
from trellis.harness.gateways import Bifrost, ProviderSDK


def runtime(gateway: Any) -> Any:
    harness = SimpleNamespace(model_gateway=gateway)
    return SimpleNamespace(agent=SimpleNamespace(harness=harness), agent_id="k")


def test_kimi_is_routed_through_bifrosts_openai_route() -> None:
    bifrost = Bifrost("http://bf/v1", "vk")
    env = {KIMI_HEADERS: "x-bf-prompt-id: p1\nx-bf-mcp-include-tools: all"}
    acp._kimi_routed(env, runtime(bifrost), "moonshot/kimi-k2")
    assert env["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
    assert env["KIMI_MODEL_BASE_URL"] == "http://bf/v1" and env["KIMI_MODEL_API_KEY"] == "vk"
    assert env["KIMI_MODEL_NAME"] == "moonshot/kimi-k2"
    lines = env[KIMI_HEADERS].splitlines()
    assert "x-bf-prompt-id: p1" in lines  # theirs kept
    assert "x-bf-mcp-include-tools: " in lines and "x-bf-mcp-include-tools: all" not in lines


def test_kimi_keeps_its_own_endpoint_and_is_said_once_as_not_routed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    acp._unrouted.cache_clear()
    bifrost = Bifrost("http://bf/v1", None)
    theirs = {"KIMI_MODEL_BASE_URL": "https://api.moonshot.ai/v1"}
    with caplog.at_level("INFO", logger="trellis.harness"):
        acp._kimi_routed(theirs, runtime(bifrost), "kimi-k2")
        acp._kimi_routed({}, runtime(ProviderSDK()), None)
    assert theirs == {
        "KIMI_MODEL_BASE_URL": "https://api.moonshot.ai/v1",
        "KIMI_MODEL_NAME": "kimi-k2",
    }
    assert len([r for r in caplog.records if "do not go through Bifrost" in r.message]) == 1
    keyless = {}
    acp._kimi_routed(keyless, runtime(bifrost), "kimi-k2")
    assert keyless["KIMI_MODEL_API_KEY"] == "no-key"


def test_options_add_to_env_and_servers_and_replace_the_rest() -> None:
    target = kimi(env={"A": "1"}, mcp_servers=[{"name": "one", "command": "x"}])
    assert given(target, {}) is target
    changed = given(target, {"env": {"B": "2"}, "mode": "plan"})
    assert changed.env == {"A": "1", "B": "2"} and changed.mode == "plan"
    assert changed.mcp_servers == target.mcp_servers
    more = given(target, {"mcp_servers": [{"name": "two", "command": "y"}]})
    assert [s["name"] for s in more.mcp_servers] == ["one", "two"] and more.env == {"A": "1"}
    ACPAdapter().check_options({"cwd": "/tmp", "model": "m"})
    with pytest.raises(ConfigurationError, match="a mode that approves on its own"):
        ACPAdapter().check_options({"mode": "auto"})


def test_a_version_outside_the_range_is_refused_saying_how_to_install() -> None:
    compat.require_binary("kimi-code", "2.1.1", command="kimi")
    compat.require_binary("kimi-code", "2.1.9", command="kimi")
    with pytest.raises(ConfigurationError, match=r"kimi-code 2\.2\.0 answers at 'kimi'"):
        compat.require_binary("kimi-code", "2.2.0", command="kimi")
    with pytest.raises(ConfigurationError, match="reports no version"):
        compat.require_binary("kimi-code", "", command="kimi")


def test_a_calls_arguments_and_output_as_the_agent_shows_them() -> None:
    block = SimpleNamespace(content=SimpleNamespace(text='{"a": 1}'))
    assert acp._args({"x": 1}, None) == {"x": 1}
    assert acp._args(None, [block]) == {"a": 1}
    assert acp._args(None, []) == {}
    listed = SimpleNamespace(content=SimpleNamespace(text="[1]"))
    assert acp._args(None, [listed]) == {"request": "[1]"}
    prose = SimpleNamespace(content=SimpleNamespace(text="Running: ls"))
    assert acp._args(None, [prose]) == {"request": "Running: ls"}
    terminal = SimpleNamespace(terminal_id="t1")  # content with no text
    assert acp._text([terminal, block]) == '{"a": 1}'
    assert acp._output(SimpleNamespace(raw_output=None, content=[block])) == '{"a": 1}'
    assert acp._output(SimpleNamespace(raw_output=[1], content=None)) == "[1]"


def test_the_native_tools_are_kimis_own_by_their_risk() -> None:
    assert acp.native_spec("Bash", KIMI).side_effects == "irreversible"
    assert acp.native_spec("Agent", KIMI).tags == ["helper"]
    assert acp.native_spec("mcp__crm__find", KIMI).server == "crm"
    assert acp.native_spec("Other", None).side_effects == "write"


def test_a_usage_log_reads_from_where_the_attempt_began(tmp_path: Any) -> None:
    folder = tmp_path / "sessions" / "wd" / "s1" / "agents" / "main"
    folder.mkdir(parents=True)
    log = folder / "wire.jsonl"
    log.write_text('{"type": "usage.record", "usage": {"output": 1}}\n')
    usage = acp.UsageLog(tmp_path, "s1")
    usage.start()  # an earlier attempt read that one: it is in the journal's spending
    with log.open("a") as file:
        file.write('{"type": "usage.record", "usage": {"output": 2}}\n')
    assert [r["usage"]["output"] for r in usage.records()] == [2]


def test_a_run_home_of_an_agent_with_no_profile_keeps_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_SETTING", "kept")
    target = acp.ACPAgent(command=("agent",), env={"OWN": "1"}, skills=object())
    home = acp.RunHome(target, SimpleNamespace(run_id="run_home_test"))  # type: ignore[arg-type]
    env = home.env()
    assert env["SOME_SETTING"] == "kept" and env["OWN"] == "1"
    assert "KIMI_CODE_HOME" not in env or env["KIMI_CODE_HOME"] != str(home.agent_home)
    import shutil

    shutil.rmtree(home.root)


async def test_the_usage_log_is_read_until_it_stops_growing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acp, "SETTLE_STEP", 0.0)
    attempt = acp.Attempt(ACPAdapter(), SimpleNamespace(), kimi())  # type: ignore[arg-type]
    attempt.usage = SimpleNamespace()  # type: ignore[assignment]
    reads = iter([2, 1, 1, 0, 5])
    monkeypatch.setattr(attempt, "_accounted", lambda: next(reads))
    await attempt._settled()
    assert next(reads) == 5  # read until a read found nothing new
