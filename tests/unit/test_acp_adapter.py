"""The ACP adapter's parts that need no agent: its options, its model routing through Bifrost,
the version check, and how it reads a call's arguments and output."""

from __future__ import annotations

from pathlib import Path
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


def test_the_allow_rules_of_kimis_own_configuration_are_found_wherever_kimi_reads_them(
    tmp_path: Path,
) -> None:
    """Kimi reads ``allow`` rules from its ``config.toml`` (``[[permission.rules]]`` with the
    decision, ``[[permission.allow]]`` entries: a pattern, or a tool with what it matches) and
    from a project's ``.kimi-code/local.toml``, at the working folder or above it. Its ``ask``
    and ``deny`` rules let nothing run unasked."""
    home, work = tmp_path / "home", tmp_path / "repo" / "pkg"
    home.mkdir()
    work.mkdir(parents=True)
    env = {"KIMI_CODE_HOME": str(home)}
    assert acp.kimi_allows(env, str(work)) == []
    (home / "config.toml").write_text(
        '[[permission.rules]]\ndecision = "allow"\npattern = "Bash(ls *)"\n'
        '[[permission.rules]]\ndecision = "ask"\npattern = "Write"\n'
        '[[permission.rules]]\ndecision = "deny"\npattern = "Bash(rm *)"\n'
        '[[permission.allow]]\ntool = "Bash"\nmatch = "git status"\n'
        '[[permission.allow]]\ntool = "Edit"\n'
    )
    assert acp.kimi_allows(env, str(work)) == ["Bash(ls *)", "Bash(git status)", "Edit"]
    local = tmp_path / "repo" / ".kimi-code"
    local.mkdir()
    (local / "local.toml").write_text(
        '[[permission.rules]]\ndecision = "allow"\npattern = "Write"\n'
        '[[permission.rules]]\ndecision = "ask"\npattern = "Edit"\n'
    )
    # an ``ask`` rule of the same pattern stands before an ``allow`` rule: that call asks
    assert acp.kimi_allows(env, str(work)) == ["Bash(ls *)", "Bash(git status)", "Write"]
    (home / "config.toml").write_text("not = toml =")  # Kimi itself refuses to start on it
    assert acp.kimi_allows(env, str(work)) == ["Write"]


async def test_a_form_the_agent_sends_while_no_turn_is_served_is_left_unfilled() -> None:
    """Nothing is there to ask a person: the agent's form is answered ``cancel``."""
    from trellis.harness.bridges import acp_live
    from trellis.harness.bridges.live import LiveSessions

    session = acp_live.ACPSession("chat", "form", LiveSessions(), keeps=False, model="m")
    answer = await session.create_elicitation("Which?", SimpleNamespace(tool_call_id="0:x"))
    assert answer.action == "cancel"


def test_a_version_outside_the_range_is_refused_saying_how_to_install() -> None:
    compat.require_binary("kimi-code", "2.6.2", command="kimi")
    compat.require_binary("kimi-code", "2.6.9", command="kimi")
    for outside in ("2.1.1", "2.6.1", "2.7.0"):  # below the tested range, and past it
        with pytest.raises(ConfigurationError, match=f"kimi-code {outside} answers at 'kimi'"):
            compat.require_binary("kimi-code", outside, command="kimi")
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
    assert acp.native_spec("Bash", KIMI).side_effects == "write"  # Kimi's own policy asks first
    assert acp.native_spec("Agent", KIMI).tags == ["builtin", "helper"]
    assert acp.native_spec("Read", KIMI).tags == ["builtin"]  # the framework's: not counted
    assert acp.native_spec("mcp__crm__find", KIMI).server == "crm"
    assert acp.native_spec("mcp__crm__find", KIMI).tags == []  # its servers' tools: the run's
    assert acp.native_spec("Other", None).side_effects == "unknown"  # nothing said: it asks


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
    attempt.usage = SimpleNamespace(ended=True)  # type: ignore[assignment]  # its end is read
    reads = iter([2, 1, 1, 0, 5])
    monkeypatch.setattr(attempt, "_accounted", lambda: next(reads))
    await attempt._settled()
    assert next(reads) == 5  # read until a read found nothing new


async def test_a_turns_end_logged_late_is_still_read(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi logs a turn's end (``turn.ended``, a step limit's reason in it) a moment after the
    turn's last usage: the turn is settled only once that line is read."""
    import asyncio
    import json

    monkeypatch.setattr(acp, "SETTLE_STEP", 0.01)
    folder = tmp_path / "sessions" / "wd" / "s1" / "agents" / "main"
    folder.mkdir(parents=True)
    log = folder / "wire.jsonl"
    log.write_text('{"type": "usage.record", "usage": {"output": 1}}\n')
    attempt = acp.Attempt(ACPAdapter(), SimpleNamespace(), kimi())  # type: ignore[arg-type]
    attempt.runtime = SimpleNamespace()  # type: ignore[assignment]
    attempt.target = SimpleNamespace(model=None)  # type: ignore[assignment]
    attempt.usage = acp.UsageLog(tmp_path, "s1")
    monkeypatch.setattr(acp.budget, "account", lambda *args, **kwargs: None)
    ended = {
        "type": acp.TURN_ENDED,
        "agentId": "main",
        "error": {"code": acp.AT_STEP_LIMIT, "details": {"maxSteps": 1}},
    }

    async def late() -> None:
        await asyncio.sleep(0.1)  # past several settle steps with nothing new
        with log.open("a") as file:
            file.write(json.dumps(ended) + "\n")

    writing = asyncio.create_task(late())
    await attempt._settled()
    await writing
    assert attempt.usage.limited == 1  # the step limit was read: the run asks once more
    assert attempt.usage.ended


async def test_a_prompt_waits_for_the_agent_to_be_free_and_fails_when_it_never_is() -> None:
    """A prompt is sent once the agent runs no turn of its own (``Keeper.settled``): one that
    never ends fails the turn — a prompt sent then would be queued behind it, unseen."""
    from trellis.harness.bridges.acp_live import QUEUED

    async def never() -> bool:
        return False

    attempt = acp.Attempt(ACPAdapter(), SimpleNamespace(), kimi())  # type: ignore[arg-type]
    attempt.session = SimpleNamespace(settled=never)  # type: ignore[assignment]
    with pytest.raises(acp.ACPRunError, match="queued the prompt"):
        async for _ in attempt._turn("hi"):
            pass
    assert "queued the prompt" in QUEUED
