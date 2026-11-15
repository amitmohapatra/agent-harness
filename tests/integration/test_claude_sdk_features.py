"""The Claude Agent SDK's own features, each used through the harness: what the SDK does with
an option, a hook, a permission answer or a control still happens when the options are
wrapped, on both paths the adapter has — a query in a process of its own (a run with no
conversation) and a conversation's live session (``bridges.claude_live``) — and the harness's
own part (governance, records, memory, spending, a pause) holds beside it.

The CLI is the scripted stand-in (``tests/support/fake_claude_cli.py``) driven by the real SDK
(0.2.160): what the stand-in does with a flag or a frame is what the bundled CLI (2.1.283) was
seen doing, driven against a scripted model on this machine. No real model was called.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    ProcessError,
    ResultMessage,
    StreamEvent,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool
from claude_agent_sdk.types import PermissionRuleValue
from mcp.types import ToolAnnotations

from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.memory import FakeMemoryService
from trellis import Ask, Harness, Hooks, Rewrite, Settings, current, tool
from trellis.contracts import RunEventType, RunStatus, ToolCall
from trellis.harness.adapters.claude import (
    BUILTIN_SIDE_EFFECTS,
    OUTLIVES_THE_QUERY,
    UNHELD_AGENTS,
    ClaudeAdapter,
)
from trellis.harness.bridges.helpers import HELPER
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.host import CLAUDE_SHELLS
from trellis.harness.sandbox import DELIVERS_HERE

#: Claude Code and Kimi Code run their shells in a folder of this machine: the tests' deployment
#: allows it (TRELLIS_HOST_EXECUTION=allow, the harness fixtures).
pytestmark = pytest.mark.host_execution
#: the deployment of the tests that make their own harness
HOST_ALLOWED = Settings(host_execution="allow")

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
#: The adapter's two paths: a query in a process of its own, and a conversation's live session.
PATHS = ("query", "live")
#: Every hook event the SDK exposes (its ``HookEvent``).
EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "Notification",
    "SubagentStart",
    "PermissionRequest",
)


def options(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ClaudeAgentOptions:
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps(script),
        "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
        "FAKE_CLAUDE_SESSIONS": str(tmp_path),
        "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        **fields.pop("env", {}),
    }
    return ClaudeAgentOptions(cli_path=CLI, env=env, **fields)


def conversation(path: str) -> str | None:
    """The thread that makes a run a conversation's turn (served by a live session), or none."""
    return "t1" if path == "live" else None


def started_with(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "cli.json").read_text())


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def flag(argv: list[str], name: str) -> str | None:
    """The value the CLI was given for ``name`` (``--name value`` or ``--name=value``)."""
    joined = [a.split("=", 1)[1] for a in argv if a.startswith(f"{name}=")]
    return joined[0] if joined else argv[argv.index(name) + 1] if name in argv else None


def live_sessions(agent: Any) -> int:
    adapter = agent.adapter
    assert isinstance(adapter, ClaudeAdapter)
    return len(adapter.live)


def tool_results(events: list[Any]) -> list[str]:
    return [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]


# --------------------------------------------------------------------------- options
@pytest.mark.parametrize("path", PATHS)
async def test_the_options_reach_the_cli_as_the_sdk_sends_them(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """Every option the harness has no part in goes to the CLI as the SDK sends it — its
    flag, its place in the ``initialize`` request, the process's folder, environment and
    user — and those it adds to keep what the team gave: its tools after ``disallowed_tools``,
    its line after the preset's ``append``."""
    said: list[str] = []
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    work = tmp_path / "work"
    work.mkdir()
    checker = AgentDefinition(
        description="Checks orders.", prompt="Check it.", tools=["Read"], model="haiku"
    )
    target = options(
        tmp_path,
        [{"text": "ok"}],
        env={"TEAM_REGION": "eu", "FAKE_CLAUDE_STDERR": "the CLI's own warning"},
        model="claude-team-model",
        fallback_model="claude-team-fallback",
        betas=["context-1m-2025-08-07"],
        permission_mode="acceptEdits",
        max_turns=7,
        tools=["Read", "Bash", "Agent"],
        allowed_tools=["Read"],
        disallowed_tools=["WebFetch"],
        system_prompt={"type": "preset", "preset": "claude_code", "append": "Be brief."},
        add_dirs=[tmp_path / "shared"],
        settings='{"includeCoAuthoredBy": false}',
        setting_sources=["user", "project"],
        extra_args={"debug-to-stderr": None, "team-flag": "on"},
        thinking={"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
        effort="high",
        output_format={"type": "json_schema", "schema": schema},
        include_partial_messages=True,
        include_hook_events=True,
        forward_subagent_text=True,
        strict_mcp_config=True,
        task_budget={"total": 5000},
        agents={"checker": checker},
        mcp_servers={"erp": {"type": "http", "url": "http://127.0.0.1:9/mcp"}},
        cwd=work,
        user=pwd.getpwuid(os.getuid()).pw_name,
        stderr=said.append,
    )
    agent = harness.wrap(target, id="options")
    result = await agent.run("hi", user="u1", thread=conversation(path))
    assert result.status is RunStatus.SUCCESS and result.answer == "ok"
    assert live_sessions(agent) == (1 if path == "live" else 0)
    cli = started_with(tmp_path)
    argv = cli["argv"]
    given = {
        "--model": "claude-team-model",
        "--fallback-model": "claude-team-fallback",
        "--betas": "context-1m-2025-08-07",
        "--permission-mode": "acceptEdits",
        "--max-turns": "7",
        "--tools": "Read,Bash,Agent",
        "--allowedTools": "Read",
        "--add-dir": str(tmp_path / "shared"),
        # the team's, and other Claude sessions' messages refused
        "--settings": '{"includeCoAuthoredBy": false, "crossSessionInbound": "refuse"}',
        "--setting-sources": "user,project",
        "--team-flag": "on",
        "--max-thinking-tokens": "2048",
        "--thinking-display": "summarized",
        "--effort": "high",
        "--json-schema": json.dumps(schema),
        "--task-budget": "5000",
        "--append-system-prompt": f"Be brief.\n\n{DELIVERS_HERE}",
    }
    assert {name: flag(argv, name) for name in given} == given
    switches = ("--debug-to-stderr", "--include-partial-messages", "--include-hook-events")
    assert all(name in argv for name in (*switches, "--strict-mcp-config"))
    assert "--system-prompt" not in argv  # the preset stays Claude Code's own prompt
    # the team's, its shell (off: the tests' layer cannot isolate a command, conftest), then
    # the tools whose work outlives a query
    assert flag(argv, "--disallowedTools") == ",".join(
        ["WebFetch", *sorted(CLAUDE_SHELLS), *OUTLIVES_THE_QUERY, *UNHELD_AGENTS]
    )
    servers = json.loads(cli["mcp_config"])["mcpServers"]
    assert servers["erp"] == {"type": "http", "url": "http://127.0.0.1:9/mcp"}
    sent = cli["initialize"]
    assert sent["agents"] == {
        "checker": {
            "description": "Checks orders.",
            "prompt": "Check it.",
            "tools": ["Read"],
            "model": "haiku",
        }
    }
    assert sent["forwardSubagentText"] is True
    assert cli["cwd"] == str(work) and cli["team_env"] == {"TEAM_REGION": "eu"}
    assert said == ["the CLI's own warning"]  # the options' stderr callback


@pytest.mark.parametrize(
    ("given", "argv", "absent"),
    [
        (
            {"system_prompt": "You triage."},
            ["--system-prompt", f"You triage.\n\n{DELIVERS_HERE}"],
            "x",
        ),
        (
            {"system_prompt": {"type": "custom", "prompt": "You triage.", "snapshot": False}},
            ["--system-prompt", f"You triage.\n\n{DELIVERS_HERE}"],
            "--append-system-prompt",
        ),
        (
            {"system_prompt": {"type": "preset", "preset": "claude_code"}},
            ["--append-system-prompt", DELIVERS_HERE],
            "--system-prompt",
        ),
        (
            {"system_prompt": {"type": "file", "path": "/srv/prompt.md"}},
            ["--system-prompt-file", "/srv/prompt.md"],
            "--system-prompt",
        ),
        ({"tools": []}, ["--tools", ""], "x"),
        ({"tools": {"type": "preset", "preset": "claude_code"}}, ["--tools", "default"], "x"),
        ({"thinking": {"type": "adaptive"}}, ["--thinking", "adaptive"], "--max-thinking-tokens"),
        ({"thinking": {"type": "disabled"}}, ["--thinking", "disabled"], "--max-thinking-tokens"),
        ({"max_thinking_tokens": 512}, ["--max-thinking-tokens", "512"], "--thinking"),
        ({"max_budget_usd": 2.5}, ["--max-budget-usd", "2.5"], "x"),
        ({"skills": ["tidy"]}, ["--allowedTools", "Skill(tidy)"], "x"),
        ({"skills": "all"}, ["--allowedTools", "Skill"], "x"),
        (
            {"plugins": [{"type": "local", "path": "/srv/plugin"}]},
            ["--plugin-dir", "/srv/plugin"],
            "x",
        ),
        ({"verbatim_prompts": True}, [], "x"),
    ],
)
async def test_each_form_an_option_takes_reaches_the_cli(
    harness: Harness, tmp_path: Path, given: dict[str, Any], argv: list[str], absent: str
) -> None:
    result = await harness.wrap(options(tmp_path, [{"text": "ok"}], **given), id="forms").run(
        "hi", user="u1"
    )
    assert result.status is RunStatus.SUCCESS and result.answer == "ok"
    sent = started_with(tmp_path)["argv"]
    if argv:
        assert flag(sent, argv[0]) == argv[1]
    assert absent not in sent


async def test_a_sandbox_of_the_options_is_merged_into_the_settings_the_cli_gets(
    harness: Harness, tmp_path: Path
) -> None:
    target = options(
        tmp_path,
        [{"text": "ok"}],
        sandbox={"enabled": True, "autoAllowBashIfSandboxed": False},
        settings='{"includeCoAuthoredBy": false}',
    )
    assert (await harness.wrap(target, id="boxed").run("hi", user="u1")).answer == "ok"
    settings = json.loads(str(flag(started_with(tmp_path)["argv"], "--settings")))
    assert settings == {
        "includeCoAuthoredBy": False,
        "crossSessionInbound": "refuse",
        "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": False},
    }


# --------------------------------------------------------------------------- hooks
@pytest.mark.parametrize("path", PATHS)
async def test_every_hook_event_of_the_teams_own_runs_inside_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, path: str
) -> None:
    """The options' own hooks, one for every event the SDK exposes: each is registered with
    the CLI and called as the CLI calls it — inside the run it is called for (``current()``)
    — beside the harness's own (governance before a call, the record after it, memory's
    context with the prompt), which still do their part."""
    called: list[tuple[str, str | None, str | None]] = []

    def hook(event: str) -> list[HookMatcher]:
        async def own(given: Any, tool_use_id: str | None, context: Any) -> Any:
            runtime = current()
            called.append((event, given.get("tool_name"), runtime and runtime.run_id))
            if event == "UserPromptSubmit":
                added = {"hookEventName": event, "additionalContext": "The team's own note."}
                return {"hookSpecificOutput": added}
            return {}

        return [HookMatcher(hooks=[own])]

    async def allow(name: str, args: dict[str, Any], context: Any) -> Any:
        return PermissionResultAllow()

    script = [
        {"builtin": "Read", "args": {"file_path": "notes.txt"}},
        {"builtin": "Bash", "args": {"command": "make build"}, "fails": "no rule"},
        {
            "agent": "checker",
            "prompt": "check o1",
            "steps": [{"builtin": "Grep", "args": {"pattern": "o1"}}, {"text": "o1 is fine"}],
        },
        {"text": "{context}"},
    ]
    target = options(
        tmp_path,
        script,
        hooks={event: hook(event) for event in EVENTS},
        can_use_tool=allow,
        agents={"checker": AgentDefinition(description="Checks orders.", prompt="Check it.")},
    )
    agent = memory_harness.wrap(target, id="hooked")
    events = [e async for e in agent.stream("check o1", user="u1", thread=conversation(path))]
    run_id = events[-1].run_id
    assert set(started_with(tmp_path)["initialize"]["hooks"]) == set(EVENTS)
    assert {run for _, _, run in called} == {run_id}  # each inside the run it was called for
    fired = [(event, name) for event, name, _ in called]
    assert fired == [
        ("UserPromptSubmit", None),
        ("PreToolUse", "Read"),
        ("PermissionRequest", "Read"),
        ("PostToolUse", "Read"),
        ("PreToolUse", "Bash"),
        ("PermissionRequest", "Bash"),
        ("PostToolUseFailure", "Bash"),
        ("PreToolUse", "Agent"),
        ("SubagentStart", None),
        ("PreToolUse", "Grep"),
        ("PermissionRequest", "Grep"),
        ("PostToolUse", "Grep"),
        ("SubagentStop", None),
        ("PostToolUse", "Agent"),
        ("Stop", None),
    ]
    # the team's context and memory's both went with the prompt
    answer = events[-1].data["result"]
    assert answer.startswith(f"The team's own note.\n{memory_service.context_text}")
    # the harness's own hooks did their part: every call recorded, the failed one as failed
    assert tool_results(events) == ["Read", "Bash", "Grep", "Agent"]
    failed = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT][1]
    assert failed["status"] == "error"


@pytest.mark.parametrize("path", PATHS)
async def test_a_slash_command_no_model_replied_in_is_a_turn_done_with_an_empty_answer(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """A slash command the app sends on purpose (the run's ``verbatim_prompts=False``: a
    user's own text is delivered as written) runs as with the SDK alone (``/compact``: the
    options' ``PreCompact`` hook is called, inside the run; no prompt hook). The CLI answers it itself — its result says ``success`` with no text, and no
    reply of the model's is in its turn — so the run ends ``SUCCESS`` with an empty answer,
    as a run of any framework in which no model replied does. The conversation goes on."""
    called: list[tuple[str, str | None]] = []

    async def own(given: Any, tool_use_id: str | None, context: Any) -> Any:
        runtime = current()
        called.append((given["hook_event_name"], runtime and runtime.run_id))
        return {}

    hooks = {event: [HookMatcher(hooks=[own])] for event in ("PreCompact", "UserPromptSubmit")}
    agent = harness.wrap(options(tmp_path, [{"text": "ok"}], hooks=hooks), id="compact")
    explicit = {"verbatim_prompts": False}
    compacted = await agent.run(
        "/compact", user="u1", thread=conversation(path), framework_options=explicit
    )
    assert called == [("PreCompact", compacted.run_id)]
    assert compacted.status is RunStatus.SUCCESS and compacted.answer == "", compacted.error
    after = await agent.run("hello", user="u1", thread=conversation(path))
    assert after.status is RunStatus.SUCCESS and after.answer == "ok"


@pytest.mark.parametrize("path", PATHS)
async def test_a_turn_whose_model_replied_with_no_text_ends_saying_so(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """A turn the model did reply in, its last reply blank, is a run with no answer, as on
    every framework: ``ERROR``, saying its reply had no text — never a silent success."""
    from trellis.harness.budget import NO_ANSWER

    agent = harness.wrap(options(tmp_path, [{"text": "  "}]), id="blank")
    result = await agent.run("margin?", user="u1", thread=conversation(path))
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == NO_ANSWER and "had no text" in result.error.message


@pytest.mark.parametrize("path", PATHS)
async def test_a_hook_of_the_teams_own_that_refuses_a_call_is_obeyed(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """The options' own ``PreToolUse`` hook answers before the harness's: a call it denies
    does not run, and Claude reads its reason."""

    async def own(given: Any, tool_use_id: str | None, context: Any) -> Any:
        refused = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "the team's hook said no",
        }
        return {"hookSpecificOutput": refused}

    script = [{"builtin": "Grep", "args": {"pattern": "x"}}, {"text": "{last}"}]
    hooks = {"PreToolUse": [HookMatcher(matcher="Grep", hooks=[own])]}
    agent = harness.wrap(options(tmp_path, script, hooks=hooks), id="refusing")
    result = await agent.run("search", user="u1", thread=conversation(path))
    assert result.status is RunStatus.SUCCESS and result.answer == "the team's hook said no"
    assert lines(tmp_path / "builtins.jsonl") == []


# --------------------------------------------------------------------------- permissions
@pytest.mark.parametrize("path", PATHS)
async def test_the_teams_permission_answer_reaches_claude_code_as_given(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """The options' own ``can_use_tool`` is given what Claude Code asks with — the call's
    id, the subagent that makes it, Claude Code's suggestions — and what it answers reaches
    Claude Code as given: the input it rewrote, the permission updates it chose, its refusal
    and whether that ends the turn."""
    asked: list[tuple[str, str | None, str | None, list[dict[str, Any]]]] = []
    rule = PermissionUpdate(
        type="addRules",
        rules=[PermissionRuleValue(tool_name="Bash", rule_content="make *")],
        behavior="allow",
        destination="session",
    )

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        suggested = [update.to_dict() for update in context.suggestions]
        asked.append((name, context.tool_use_id, context.agent_id, suggested))
        if name == "Bash":
            rewritten = {"command": "make build --dry-run"}
            return PermissionResultAllow(updated_input=rewritten, updated_permissions=[rule])
        if name == "Write":
            return PermissionResultDeny(message="not in this folder")
        return PermissionResultAllow()

    script = [
        {"builtin": "Bash", "args": {"command": "make build"}},
        {"builtin": "Write", "args": {"file_path": "a.txt", "content": "x"}},
        {
            "agent": "checker",
            "prompt": "check o1",
            "steps": [{"builtin": "Edit", "args": {"file_path": "o1"}}, {"text": "o1 is fine"}],
        },
        {"text": "done"},
    ]
    target = options(
        tmp_path,
        script,
        can_use_tool=own,
        agents={"checker": AgentDefinition(description="Checks orders.", prompt="Check it.")},
    )
    agent = harness.wrap(target, id="asked")
    events = [e async for e in agent.stream("build", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "done"
    suggested = [
        {
            "type": "addRules",
            "destination": "session",
            "rules": [{"toolName": name, "ruleContent": None}],
            "behavior": "allow",
        }
        for name in ("Bash", "Write", "Edit")
    ]
    assert asked == [
        ("Bash", "toolu_0", None, [suggested[0]]),
        ("Write", "toolu_1", None, [suggested[1]]),
        ("Edit", "toolu_2_0", "agent_toolu_2", [suggested[2]]),  # the subagent's own call
    ]
    bash, write, edit = lines(tmp_path / "cli.json.permissions")
    assert bash["answer"] == {
        "behavior": "allow",
        "updatedInput": {"command": "make build --dry-run"},
        "updatedPermissions": [rule.to_dict()],
    }
    assert write["answer"] == {"behavior": "deny", "message": "not in this folder"}
    assert edit["answer"]["behavior"] == "allow"
    ran = lines(tmp_path / "builtins.jsonl")
    assert [(call["tool"], call["args"]) for call in ran] == [
        ("Bash", {"command": "make build --dry-run"}),  # as the team rewrote it
        ("Edit", {"file_path": "o1"}),
    ]
    assert tool_results(events) == ["Bash", "Edit", "Agent"]  # what ran is recorded


@pytest.mark.parametrize("path", PATHS)
async def test_what_claude_code_says_beside_its_replies_is_on_the_runs_stream(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """Claude Code says on its stream when it is close to (or at) its rate limit, retries a
    model call, compacts the session's context, falls back to another model, or could not
    mirror a step into the session store, and when a reply failed: each is a warning of the
    run, by its own code —
    a slow, shortened or rerouted turn says why. A rate limit that is fine says nothing."""
    limit = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}
    near = {"status": "allowed_warning", "rateLimitType": "five_hour", "utilization": 0.9}
    failed = {"id": "msg_failed", "role": "assistant", "model": "fake", "content": []}
    frames = [
        limit,
        {**limit, "rate_limit_info": near},
        {"type": "system", "subtype": "api_retry", "attempt": 2, "error": "overloaded"},
        {"type": "system", "subtype": "compact_boundary", "trigger": "auto"},
        {"type": "system", "subtype": "model_fallback", "to": "claude-haiku"},
        {"type": "system", "subtype": "mirror_error", "error": "the store timed out"},
        {
            "type": "assistant",
            "message": failed,
            "parent_tool_use_id": None,
            "error": "server_error",
        },
    ]
    script = [*({"frame": frame} for frame in frames), {"text": "done"}]
    agent = harness.wrap(options(tmp_path, script), id="noticed")
    events = [e async for e in agent.stream("go", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "done"
    warned = [
        (e.data["code"], e.data["message"]) for e in events if e.data.get("name") == "warning"
    ]
    codes = ["rate_limit", "model_retry", "compaction", "model_fallback", "session_store"]
    assert [code for code, _ in warned] == [*codes, "model_error"]
    said = dict(warned)
    assert "allowed_warning" in said["rate_limit"] and "five_hour" in said["rate_limit"]
    assert "overloaded" in said["model_retry"] and "claude-haiku" in said["model_fallback"]
    assert "the store timed out" in said["session_store"]
    assert "server_error" in said["model_error"]


@pytest.mark.parametrize("path", PATHS)
async def test_claudes_reasoning_and_the_tools_the_api_ran_are_on_the_stream_and_recorded(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    tmp_path: Path,
    path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """What the provider did inside a model step. Claude's reasoning (a ``thinking`` block)
    is on the run's stream — its text only: the block's signature is nowhere (no event, no
    log, no record), and none of it goes to memory. A tool the API ran itself (web search,
    the advisor) is a call of the run: recorded as a read, with its result where the stream
    carries one, and never put to a person — it already ran."""
    thought = {"type": "thinking", "thinking": "Weighing the refund.", "signature": "SIG-SECRET"}
    searched = {"type": "server_tool_use", "id": "srv_1", "name": "web_search"}
    advised = {"type": "server_tool_use", "id": "srv_2", "name": "advisor", "input": {}}
    advice = {"type": "advisor_tool_result", "tool_use_id": "srv_2", "content": {"text": "go"}}
    blocks = [thought, {**searched, "input": {"query": "refund policy"}}, advised, advice]
    said = {"id": "msg_blocks", "role": "assistant", "model": "fake", "content": blocks}
    frame = {"type": "assistant", "message": said, "parent_tool_use_id": None}
    agent = memory_harness.wrap(
        options(tmp_path, [{"frame": frame}, {"text": "done"}]), id="thinks"
    )
    with caplog.at_level("DEBUG"):
        events = [e async for e in agent.stream("go", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "done"
    reasoned = [e.data for e in events if e.data.get("name") == "reasoning"]
    assert reasoned == [{"name": "reasoning", "text": "Weighing the refund."}]
    results = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [(r["tool"], r["status"]) for r in results] == [("advisor", "ok"), ("web_search", "ok")]
    assert results[0]["output"] == {"text": "go"}  # the advisor's, as the stream carried it
    await memory_harness.writes.drain()
    kept = repr(memory_service.calls)
    assert "advisor" in kept and "web_search" in kept  # the calls are recorded
    everywhere = kept + repr([e.model_dump() for e in events]) + caplog.text
    assert "SIG-SECRET" not in everywhere and "Weighing the refund." not in kept + caplog.text


@pytest.mark.parametrize("path", PATHS)
async def test_a_subagents_reasoning_and_the_tools_the_api_ran_for_it_are_its_runs(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    tmp_path: Path,
    path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLAUDE-SUB-1. What the provider did inside a subagent's model step is the subagent's:
    its reasoning on the run's stream naming the subagent's child run (its text only: the
    signature nowhere, none of it in memory), and a tool the API ran for it (web search, the
    advisor) recorded as a read under the child run — never put to a person. The main agent's
    stay the run's own."""
    thought = {"type": "thinking", "thinking": "Weighing sources.", "signature": "SIG-SUB"}
    searched = {"type": "server_tool_use", "id": "srv_9", "name": "web_search"}
    advised = {"type": "server_tool_use", "id": "srv_8", "name": "advisor", "input": {}}
    advice = {"type": "advisor_tool_result", "tool_use_id": "srv_8", "content": {"text": "go"}}
    blocks = [thought, {**searched, "input": {"query": "refund law"}}, advised, advice]
    mine = {"type": "thinking", "thinking": "Delegating.", "signature": "SIG-MAIN"}
    said = {"id": "msg_main", "role": "assistant", "model": "fake", "content": [mine]}
    script = [
        {"frame": {"type": "assistant", "message": said, "parent_tool_use_id": None}},
        {"agent": "researcher", "prompt": "find it", "steps": [{"blocks": blocks}, {"text": "x"}]},
        {"text": "done"},
    ]
    researcher = AgentDefinition(description="Researches.", prompt="Research.")
    target = options(tmp_path, script, agents={"researcher": researcher})
    agent = memory_harness.wrap(target, id="sub-thinks")
    with caplog.at_level("DEBUG"):
        events = [e async for e in agent.stream("go", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "done", events[-1].data
    [child] = [e.data for e in events if e.data.get("name") == "helper_started"]
    reasoned = [e.data for e in events if e.data.get("name") == "reasoning"]
    by_child = {"run_id": child["run_id"], "name": "researcher"}
    assert reasoned == [
        {"name": "reasoning", "text": "Delegating."},
        {"name": "reasoning", "text": "Weighing sources.", HELPER: by_child},
    ]
    results = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [(r["tool"], r["status"]) for r in results if r["tool"] != "Agent"] == [
        ("advisor", "ok"),
        ("web_search", "ok"),
    ]
    assert not [e for e in events if e.type is RunEventType.INTERRUPT]
    await memory_harness.writes.drain()
    made = {c.body["tool"]: c.scope["agent_run_id"] for c in memory_service.named("record_tool")}
    assert made["advisor"] == made["web_search"] == child["run_id"]
    kept = repr(memory_service.calls)
    everywhere = kept + repr([e.model_dump() for e in events]) + caplog.text
    assert "SIG-SUB" not in everywhere and "SIG-MAIN" not in everywhere
    assert "Weighing sources." not in kept + caplog.text


@pytest.mark.parametrize("path", PATHS)
async def test_the_teams_hook_answers_before_governance_decides_what_it_left(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """Claude Code asks every ``PreToolUse`` hook at once, and the call runs with the
    arguments of the hook that answered last. The harness's hook answers after the team's
    own: the run's hooks and governance decide the call as the team's hook left it — its
    rewritten arguments, which are also the ones recorded — not the ones the model gave."""
    seen: list[dict[str, Any]] = []

    async def scoped(given: Any, tool_use_id: str | None, hook: Any) -> Any:
        await asyncio.sleep(0.05)  # the team's hook is still at work when the harness's is asked
        safe = {**given["tool_input"], "command": "make deploy --dry-run"}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": safe}}

    class Seeing(Hooks):
        async def before_tool(self, call: ToolCall) -> None:
            seen.append(dict(call.args))

    script = [{"builtin": "Bash", "args": {"command": "make deploy"}}, {"text": "{last}"}]
    own = {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[scoped])]}
    target = options(tmp_path, script, hooks=own, permission_mode="bypassPermissions")
    agent = harness.wrap(target, id="ordered", hooks=[Seeing()])
    events = [e async for e in agent.stream("deploy", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "Bash ran"
    rewritten = {"command": "make deploy --dry-run"}
    assert seen == [rewritten]  # what governance decided is what ran
    assert [c["args"] for c in lines(tmp_path / "builtins.jsonl")] == [rewritten]
    said = [e.data["args"] for e in events if e.type is RunEventType.TOOL_CALL_ARGS]
    assert said == [rewritten]  # and what the run recorded of the call


@pytest.mark.parametrize("path", PATHS)
async def test_a_call_the_teams_hook_refuses_is_not_a_persons_to_approve(
    tmp_path: Path, path: str
) -> None:
    """A call the team's own ``PreToolUse`` hook refuses does not run, whatever the harness
    says of it: nobody is asked to approve it first."""

    async def refusing(given: Any, tool_use_id: str | None, hook: Any) -> Any:
        await asyncio.sleep(0.05)
        refused = {"permissionDecision": "deny", "permissionDecisionReason": "not on Fridays"}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", **refused}}

    script = [{"builtin": "Bash", "args": {"command": "make deploy"}}, {"text": "{last}"}]
    own = {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[refusing])]}
    rules = {"Bash": Rule(risk="irreversible")}  # governance would ask a person
    async with with_governance(Harness(config=HOST_ALLOWED), Governance(FakeCatalog(rules))) as h:
        agent = h.wrap(options(tmp_path, script, hooks=own), id="refused")
        result = await agent.run("deploy", user="u1", thread=conversation(path))
    assert result.status is RunStatus.SUCCESS and result.answer == "not on Fridays", result
    assert lines(tmp_path / "builtins.jsonl") == []


@pytest.mark.parametrize("path", PATHS)
async def test_the_result_the_teams_hook_put_in_its_place_is_the_one_recorded(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """A team's ``PostToolUse`` hook replaces what a call returned before the model reads it
    (``updatedToolOutput``): the run's record of the call is what the model read, not what
    the hook took out."""

    async def redacting(given: Any, tool_use_id: str | None, hook: Any) -> Any:
        await asyncio.sleep(0.05)
        clean = [{"type": "text", "text": "card ending 4242"}]
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": clean}}

    script = [{"builtin": "Read", "args": {"file_path": "cards.csv"}}, {"text": "{last}"}]
    own = {"PostToolUse": [HookMatcher(matcher="Read", hooks=[redacting])]}
    agent = harness.wrap(options(tmp_path, script, hooks=own), id="redacted")
    events = [e async for e in agent.stream("read", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == "card ending 4242"  # what the model read
    [read] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert read["tool"] == "Read" and read["output"] == "card ending 4242"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name", ["FutureTool", "mcp__erp__post_invoice"])
async def test_a_tool_that_says_nothing_of_what_it_does_asks_a_person_once(
    harness: Harness, tmp_path: Path, path: str, name: str
) -> None:
    """A tool the harness has no word on — one of the CLI's not in ``BUILTIN_SIDE_EFFECTS``,
    a tool of an MCP server of the team's own the CLI reaches itself — is a person's to
    approve, on either path. Governance asks before the call (``PreToolUse``), and Claude
    Code asks its user about such a call too (``can_use_tool``): one call, one question —
    approved, it runs once without a second one, and is recorded."""
    assert name not in BUILTIN_SIDE_EFFECTS
    script = [{"builtin": name, "args": {"name": "fix"}}, {"text": "{last}"}]
    agent = harness.wrap(options(tmp_path, script), id="unknown")
    paused = await agent.run("go on", user="u1", thread=conversation(path))
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == name
    assert lines(tmp_path / "builtins.jsonl") == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == f"{name} ran", done.interrupt
    assert [call["tool"] for call in lines(tmp_path / "builtins.jsonl")] == [name]
    # Claude Code did ask (its own check), and was answered without a person
    asked = lines(tmp_path / "cli.json.permissions")
    assert [(a["tool"], a["answer"]["behavior"]) for a in asked] == [(name, "allow")]


@pytest.mark.parametrize("path", PATHS)
async def test_loading_a_deferred_tool_asks_nobody(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """Claude Code loads a team MCP server's deferred tools with ``ToolSearch`` (and waits for
    a server still connecting with ``WaitForMcpServers``): both only read, so they run."""
    script = [
        {"builtin": "ToolSearch", "args": {"query": "select:mcp__erp__post_invoice"}},
        {"builtin": "WaitForMcpServers", "args": {}},
        {"text": "{last}"},
    ]
    done = await harness.wrap(options(tmp_path, script), id="deferred").run(
        "go on", user="u1", thread=conversation(path)
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "WaitForMcpServers ran"
    assert [c["tool"] for c in lines(tmp_path / "builtins.jsonl")] == [
        "ToolSearch",
        "WaitForMcpServers",
    ]


@pytest.mark.parametrize("name", ["ExitPlanMode", "Skill"])
async def test_what_claude_code_asks_about_a_person_answers_unless_the_catalog_allows_it(
    tmp_path: Path, name: str
) -> None:
    """Claude Code asks its user before it leaves plan mode with a plan (``ExitPlanMode``) or
    runs a skill (``Skill``): a person answers as it would have, once — unless the tool
    catalog allows the tool, which then runs without a question. (The network, ``WebFetch``
    and ``WebSearch``, is the catalog's alone: ``test_who_decides``.)"""
    script = [{"builtin": name, "args": {"url": "https://example.com"}}, {"text": "{last}"}]
    async with with_governance(Harness(config=HOST_ALLOWED), Governance(FakeCatalog({}))) as h:
        agent = h.wrap(options(tmp_path, script), id="asks")
        paused = await agent.run("go on", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == name
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.answer == f"{name} ran", done.error
        assert [c["tool"] for c in lines(tmp_path / "builtins.jsonl")] == [name]  # once
    (tmp_path / "builtins.jsonl").unlink()
    allows = FakeCatalog({name: Rule(risk="read")})
    async with with_governance(Harness(config=HOST_ALLOWED), Governance(allows)) as h:
        done = await h.wrap(options(tmp_path, script), id="allowed").run("go on", user="u1")
        assert done.status is RunStatus.SUCCESS and done.answer == f"{name} ran", done.error


@pytest.mark.parametrize("path", PATHS)
async def test_a_write_claude_code_asks_about_is_one_question_and_the_same_call_again_another(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """A write governance lets run is asked about when Claude Code asks (its question): once
    a call. The same command made again is another call, asked about again."""
    again = {"builtin": "Edit", "args": {"file_path": "deploy.txt"}}
    agent = harness.wrap(options(tmp_path, [again, again, {"text": "{last}"}]), id="twice")
    paused = await agent.run("deploy twice", user="u1", thread=conversation(path))
    for _ in range(2):
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert "Claude Code asks" in paused.interrupt.question
        paused = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert paused.status is RunStatus.SUCCESS and paused.answer == "Edit ran"
    assert len(lines(tmp_path / "builtins.jsonl")) == 2


#: Why governance asks a person about a call of Claude's own: the catalog's rule for its
#: tool, a run hook's ``Ask``, the catalog's rule about the arguments a run hook rewrote.
WHY_ASKED = ("rule", "hook", "rewritten")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("inside", [False, True], ids=["main-agent", "subagent"])
@pytest.mark.parametrize("why", WHY_ASKED)
async def test_a_built_in_governance_asked_about_is_one_question_whatever_made_it_ask(
    tmp_path: Path, path: str, inside: bool, why: str
) -> None:
    """Governance asks a person about a call of Claude's own before it runs (``PreToolUse``),
    and Claude Code asks its user about the same call (``can_use_tool``: ``Bash`` is not in
    its allowed tools): one call, one question — whatever made governance ask, on either
    path, the main agent's call or a subagent's. Approved, it runs once, as decided."""
    command = {"command": "make deploy"}
    call = {"builtin": "Bash", "args": command}
    deploys = {"agent": "deployer", "prompt": "deploy", "steps": [call, {"text": "deployed"}]}
    script = [deploys if inside else call, {"text": "{last}"}]

    class Asking(Hooks):
        async def before_tool(self, call: ToolCall) -> Ask | Rewrite | None:
            if call.tool != "Bash" or why == "rule":
                return None
            return Ask("Deploy now?") if why == "hook" else Rewrite({"command": "make deploy -n"})

    rules = {} if why == "hook" else {"Bash": Rule(risk="irreversible")}
    agents = {"deployer": AgentDefinition(description="Deploys.", prompt="Deploy it.")}
    async with with_governance(Harness(config=HOST_ALLOWED), Governance(FakeCatalog(rules))) as h:
        target = options(tmp_path, script, agents=agents)
        agent = h.wrap(target, id="deploys", hooks=[Asking()])
        paused = await agent.run("deploy", user="u1", thread=conversation(path))
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "Bash"
        assert "Claude Code asks" not in paused.interrupt.question  # governance's question
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        again = None if done.interrupt is None else done.interrupt.question
        assert done.status is RunStatus.SUCCESS, again  # no second question
    ran = [(c["tool"], c["args"]) for c in lines(tmp_path / "builtins.jsonl")]
    assert ran == [("Bash", {"command": "make deploy -n"} if why == "rewritten" else command)]


async def test_an_approval_is_carried_for_the_call_it_answered_and_no_other(
    harness: Harness, tmp_path: Path
) -> None:
    """SEC-R-7: a person's answer spares the call it answered Claude Code's own question, by
    the call's id. From a CLI (or a relay) that names no id, nothing is carried: an approved
    edit of ``deploy.txt`` is no approval of the edit of ``build.txt`` that follows, which a person is
    asked about before it runs (the call it could not name is asked about again)."""
    deploy, wipe = {"file_path": "deploy.txt"}, {"file_path": "build.txt"}
    script = [{"builtin": "Edit", "args": deploy}, {"builtin": "Edit", "args": wipe}]

    class Asking(Hooks):
        async def before_tool(self, call: ToolCall) -> Ask | None:
            return Ask("Deploy now?") if call.args == deploy else None

    target = options(tmp_path, [*script, {"text": "{last}"}])
    target.env["FAKE_CLAUDE_NO_CALL_IDS"] = "1"
    agent = harness.wrap(target, id="carries", hooks=[Asking()])
    asked = await agent.run("deploy, then clean", user="u1")
    for _ in range(2):  # the hook's question, then Claude Code's own about the same call
        assert asked.status is RunStatus.PAUSED and asked.interrupt is not None, asked.error
        assert asked.interrupt.tool_call is not None
        if asked.interrupt.tool_call.args != deploy:
            break
        asked = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None, asked.answer
    assert asked.interrupt.tool_call is not None and asked.interrupt.tool_call.args == wipe
    ran = [(c["tool"], c["args"]) for c in lines(tmp_path / "builtins.jsonl")]
    assert ran == [("Edit", deploy)]


@pytest.mark.parametrize("path", PATHS)
async def test_a_question_a_hook_asks_while_a_call_is_decided_is_no_approval_of_that_call(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """Only a person's answer to the approval of the call itself spares it Claude Code's own
    question (one call, one question). A run hook that asks a person something else while
    the call is decided (``trellis.current().ask``: a ticket number) approves nothing: the
    write Claude Code asks about is still asked about, and does not run until approved."""

    class Ticket(Hooks):
        async def before_tool(self, call: ToolCall) -> None:
            runtime = current()
            if call.tool == "Edit" and runtime is not None:
                await runtime.ask("Which ticket is this change for?")

    touch = {"builtin": "Edit", "args": {"file_path": "notes.txt"}}
    agent = harness.wrap(options(tmp_path, [touch, {"text": "{last}"}]), id="ticketed",
                         hooks=[Ticket()])  # fmt: skip
    asked = await agent.run("touch it", user="u1", thread=conversation(path))
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.tool_call is None  # the hook's question
    approval = await agent.resume(asked.interrupt.interrupt_id, "approve", answer="T-1",
                                  reviewer="u1")  # fmt: skip
    assert lines(tmp_path / "builtins.jsonl") == [], "Edit ran without anyone approving it"
    assert approval.status is RunStatus.PAUSED and approval.interrupt is not None
    assert "Claude Code asks" in approval.interrupt.question  # its own question, asked
    done = await agent.resume(approval.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "Edit ran"
    assert [c["args"] for c in lines(tmp_path / "builtins.jsonl")] == [touch["args"]]


@pytest.mark.parametrize("path", PATHS)
async def test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """The tools Claude Code keeps its own work with — its task list (``TaskCreate``,
    ``TaskUpdate``, ``TaskList``, ``TaskGet``: ``TodoWrite``'s successors in the bundled
    CLI), entering plan mode — and the one its "model" hands a structured answer to
    (``output_format``: ``StructuredOutput``) change nothing outside the session: Claude
    Code runs them without asking its user (seen driving the bundled CLI), and so does the
    harness. The run's answer is the structured one."""
    kept = ("TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "EnterPlanMode", "TodoWrite")
    assert {BUILTIN_SIDE_EFFECTS[name] for name in (*kept, "StructuredOutput")} == {"read"}
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    script = [
        {"builtin": "TaskCreate", "args": {"subject": "find the city"}},
        {"builtin": "TaskUpdate", "args": {"taskId": "1", "status": "completed"}},
        {"builtin": "TaskList", "args": {}},
        {"builtin": "StructuredOutput", "args": {"city": "Paris"}},
    ]
    target = options(tmp_path, script, output_format={"type": "json_schema", "schema": schema})
    agent = harness.wrap(target, id="structured")
    events = [e async for e in agent.stream("where?", user="u1", thread=conversation(path))]
    assert events[-1].data["result"] == {"city": "Paris"}  # the structured output
    assert tool_results(events) == ["TaskCreate", "TaskUpdate", "TaskList", "StructuredOutput"]
    assert not lines(tmp_path / "cli.json.permissions") or all(
        asked["answer"]["behavior"] == "allow" for asked in lines(tmp_path / "cli.json.permissions")
    )


# --------------------------------------------------------------------------- messages
@pytest.mark.parametrize("path", PATHS)
async def test_partial_messages_and_other_kinds_of_message_leave_the_answer_as_it_is(
    harness: Harness, tmp_path: Path, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``include_partial_messages`` makes the CLI send the model's stream as it comes
    (``StreamEvent``) before each message: the SDK hands them to the adapter with the rest,
    the run's text is each assistant message's once, and the answer is the result's."""
    seen: list[Any] = []
    extract = ClaudeAdapter.extract

    def kept(self: ClaudeAdapter, target: Any, output: Any) -> Any:
        seen.extend(output)
        return extract(self, target, output)

    monkeypatch.setattr(ClaudeAdapter, "extract", kept)
    target = options(tmp_path, [{"text": "the tide is out"}], include_partial_messages=True)
    agent = harness.wrap(target, id="partial")
    events = [e async for e in agent.stream("tide?", user="u1", thread=conversation(path))]
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(said) == "the tide is out" == events[-1].data["result"]
    deltas = [
        m.event["delta"]["text"] for m in seen if isinstance(m, StreamEvent) and "delta" in m.event
    ]
    assert "".join(deltas) == "the tide is out"
    [result] = [m for m in seen if isinstance(m, ResultMessage)]
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.parametrize("path", PATHS)
async def test_partial_messages_stream_the_runs_text_as_the_model_writes_it(
    harness: Harness, tmp_path: Path, path: str
) -> None:
    """With ``include_partial_messages`` the SDK's own stream gives the model's text delta by
    delta (``StreamEvent``); the run's stream does the same — each delta a text event as it
    comes, the message's whole text not sent again — on the query and the live way (seen on
    Haiku: one event a message, docs/research/sdk-full-stack-2026-11-15)."""
    target = options(tmp_path, [{"text": "the tide is out"}], include_partial_messages=True)
    agent = harness.wrap(target, id="partial-deltas")
    events = [e async for e in agent.stream("tide?", user="u1", thread=conversation(path))]
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert said == ["the", " tide", " is", " out"]
    assert events[-1].data["result"] == "the tide is out"


async def test_without_partial_messages_the_runs_text_is_each_message_once(
    harness: Harness, tmp_path: Path
) -> None:
    """Without ``include_partial_messages`` the CLI sends no deltas: each assistant message's
    text is one text event."""
    agent = harness.wrap(options(tmp_path, [{"text": "the tide is out"}]), id="whole")
    events = [e async for e in agent.stream("tide?", user="u1")]
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert said == ["the tide is out"]


@pytest.mark.parametrize("named", [True, False], ids=["allowed", "unnamed"])
async def test_an_in_process_tools_annotations_decide_it_when_allowed_tools_names_it(
    harness: Harness, tmp_path: Path, named: bool
) -> None:
    """An in-process server's tool is governed by its annotations (``readOnlyHint``: a read,
    which runs unasked) — also when ``allowed_tools`` names it, as the SDK's docs pre-approve
    an MCP tool: naming it there said nothing of its risk, and each call asked a person (seen
    on Haiku, docs/research/sdk-full-stack-2026-11-15). APR-01"""
    ran: list[str] = []

    @sdk_tool("stock", "Units of a SKU.", {"sku": str},
              annotations=ToolAnnotations(read_only_hint=True))  # fmt: skip
    async def stock(args: dict[str, Any]) -> dict[str, Any]:
        ran.append(args["sku"])
        return {"content": [{"type": "text", "text": "7 units"}]}

    shop = create_sdk_mcp_server("shop", tools=[stock])
    script = [{"tool": "stock", "server": "shop", "args": {"sku": "S1"}}, {"text": "{last}"}]
    allowed = {"allowed_tools": ["mcp__shop__stock"]} if named else {}
    target = options(tmp_path, script, mcp_servers={"shop": shop}, **allowed)
    result = await harness.wrap(target, id="annotated").run("stock of S1?", user="u1")
    assert result.interrupt is None and result.status is RunStatus.SUCCESS, result.error
    assert ran == ["S1"] and result.answer == "7 units"


# --------------------------------------------------------------------------- sessions
async def test_a_session_id_of_the_options_names_the_session_and_a_pause_resumes_it(
    harness: Harness, tmp_path: Path
) -> None:
    """``session_id`` names the session a query starts. Claude Code takes it with
    ``--resume`` only to fork, so the attempt that resumes that session after a pause names
    it once, as the one to resume."""

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return f"refunded {order}"

    named = str(uuid.uuid4())
    script = [{"tool": "refund", "args": {"order": "o1"}}, {"text": "{last}"}]
    agent = arm(harness.wrap(options(tmp_path, script, session_id=named), id="named"), [refund])
    paused = await agent.run("refund o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    first = started_with(tmp_path)
    assert first["session"] == named and flag(first["argv"], "--session-id") == named
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "refunded o1", done.error
    again = started_with(tmp_path)
    assert again["session"] == named and again["resume"] == named
    assert flag(again["argv"], "--session-id") is None


async def test_where_the_options_cut_the_session_they_resume_holds_for_the_first_query_only(
    harness: Harness, tmp_path: Path
) -> None:
    """``resume_session_at`` (with ``resume_drops_turn``) cuts the session the options resume
    at an entry of theirs. The run's later attempt resumes the run's own session after a
    pause: cut there, it would lose the paused turn's work, so it is not."""

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return f"refunded {order}"

    began = str(uuid.uuid4())
    cut = {"resume_session_at": str(uuid.uuid4()), "resume_drops_turn": str(uuid.uuid4())}
    first = await harness.wrap(
        options(tmp_path, [{"text": "noted"}], session_id=began), id="a"
    ).run("the code is 4242", user="u1")
    assert first.answer == "noted"
    script = [{"tool": "refund", "args": {"order": "o1"}}, {"text": "{last}"}]
    target = options(tmp_path, script, resume=began, fork_session=True, **cut)
    agent = arm(harness.wrap(target, id="cut"), [refund])
    paused = await agent.run("refund o1", user="u1")
    assert paused.interrupt is not None, paused.error
    argv = started_with(tmp_path)["argv"]
    assert flag(argv, "--resume-session-at") == cut["resume_session_at"]  # the options' own
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.answer == "refunded o1", done.error
    argv = started_with(tmp_path)["argv"]
    assert flag(argv, "--resume-session-at") is None and flag(argv, "--resume-drops-turn") is None


async def test_the_sessions_the_options_choose_are_claude_codes_to_resume_fork_or_continue(
    harness: Harness, tmp_path: Path
) -> None:
    """The options' own ``resume``, ``fork_session`` and ``continue_conversation`` choose
    the session a run's first query starts from, as without the harness; the session the
    query then runs in is the run's."""
    first = await harness.wrap(options(tmp_path, [{"text": "noted"}]), id="one").run(
        "the code is 4242", user="u1"
    )
    assert first.answer == "noted"
    began = started_with(tmp_path)["session"]
    script = [{"text": "ok"}]
    resumed = options(tmp_path, script, resume=began)
    assert (await harness.wrap(resumed, id="two").run("what was it?", user="u1")).answer == "ok"
    again = started_with(tmp_path)
    assert again["session"] == began and again["resume"] == began
    forked = options(tmp_path, script, resume=began, fork_session=True)
    assert (await harness.wrap(forked, id="three").run("again?", user="u1")).status is (
        RunStatus.SUCCESS
    )
    fork = started_with(tmp_path)
    assert fork["session"] != began and "--fork-session" in fork["argv"]
    continued = options(tmp_path, script, continue_conversation=True)
    assert (await harness.wrap(continued, id="four").run("and now?", user="u1")).status is (
        RunStatus.SUCCESS
    )
    last = started_with(tmp_path)
    assert "--continue" in last["argv"] and last["session"] == fork["session"]


async def test_a_forked_session_paused_inside_goes_on_in_its_fork(
    harness: Harness, tmp_path: Path
) -> None:
    """Options that fork a session (``resume`` with ``fork_session``) and a run that pauses
    in the fork: the attempt after the answer goes on in the session the query ran in, and
    the call it paused on runs once."""
    first = await harness.wrap(options(tmp_path, [{"text": "noted"}]), id="one").run(
        "the code is 4242", user="u1"
    )
    assert first.answer == "noted"
    began = started_with(tmp_path)["session"]
    made: list[str] = []

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        made.append(order)
        return f"refunded {order}"

    script = [{"tool": "refund", "args": {"order": "o1"}}, {"text": "{last}"}]
    forked = options(tmp_path, script, resume=began, fork_session=True)
    agent = arm(harness.wrap(forked, id="forked"), [refund])
    paused = await agent.run("refund o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    fork = started_with(tmp_path)["session"]
    assert fork != began
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "refunded o1"
    assert made == ["o1"] and started_with(tmp_path)["resume"] == fork


# --------------------------------------------------------------------------- errors
async def test_a_cli_that_cannot_start_fails_the_run_with_the_sdks_own_error(
    harness: Harness, tmp_path: Path
) -> None:
    """The SDK's errors are the run's: a CLI that exits before it answers (here: a session id
    already in use, the bundled CLI's own refusal) is a ``ProcessError``; one that is not
    there, a ``CLINotFoundError``."""
    named = str(uuid.uuid4())
    target = options(tmp_path, [{"text": "ok"}], session_id=named)
    assert (await harness.wrap(target, id="first").run("hi", user="u1")).answer == "ok"
    again = await harness.wrap(target, id="second").run("hi", user="u1")
    assert again.status is RunStatus.ERROR and again.error is not None
    assert again.error.code == ProcessError.__name__
    missing = ClaudeAgentOptions(cli_path=str(tmp_path / "no-such-cli"))
    gone = await harness.wrap(missing, id="missing").run("hi", user="u1")
    assert gone.status is RunStatus.ERROR and gone.error is not None
    assert gone.error.code == "CLINotFoundError"
