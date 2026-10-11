"""Claude Agent SDK: the real SDK drives a scripted stand-in for the Claude Code CLI
(``tests/support/fake_claude_cli.py``) through ``cli_path`` — subprocess, control protocol and
in-process MCP tool calls included."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny

from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from trellis import Deny, Harness, Hooks, Rewrite, current, tool
from trellis.contracts import RunEventType, RunStatus, ToolCall, ToolSpec
from trellis.harness.adapters.claude import RESUMED, RESUMED_CALL
from trellis.harness.journal import content_key
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert.claude import _schema

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


def options(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(tmp_path),
            "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        },
        **fields,
    )


def built_ins(tmp_path: Path) -> list[dict[str, Any]]:
    """What Claude Code's own tools ran."""
    log = tmp_path / "builtins.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def started_with(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "cli.json").read_text())


async def test_a_query_answers_and_calls_harness_tools(harness: Harness, tmp_path: Path) -> None:
    refunds.clear()
    target = options(
        tmp_path,
        [{"tool": "refund", "args": {"order": "o1"}}, {"text": "refunded o1"}],
        system_prompt="You refund.",
    )
    agent = harness.wrap(target, id="refunds", tools=[tool(refund.fn, side_effects="write")])
    result = await agent.run("refund o1", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded o1"
    assert refunds == ["o1"]
    cli = started_with(tmp_path)
    # read_file is listed from the start: the CLI lists a server's tools once
    assert cli["tools"] == ["refund", "read_file"] and cli["allowed_tools"] is None
    assert cli["permission_prompt_tool"] == "stdio"  # the harness's can_use_tool decides
    assert cli["system_prompt"] == "You refund."
    assert target.mcp_servers == {}  # the team's options are untouched


asked: list[dict[str, Any]] = []


class NoArguments:
    """An MCP server's tool that takes no argument: its schema is ``{"type": "object"}``."""

    async def resolve(self) -> list[Tool]:
        spec = ToolSpec(name="now", description="The time.", input_schema={"type": "object"})

        async def run(args: dict[str, Any]) -> Any:
            asked.append(args)
            return "noon"

        return [Tool(spec, run, feature="mcp")]


async def test_a_tool_whose_schema_declares_no_properties_takes_no_argument(
    harness: Harness, tmp_path: Path
) -> None:
    """The SDK reads a schema with no ``properties`` as a map of argument names to types:
    the tool is offered as a JSON schema of an object without arguments."""
    target = options(tmp_path, [{"tool": "now", "args": {}}, {"text": "it is noon"}])
    asked.clear()
    result = await harness.wrap(target, id="clock", tools=[NoArguments()]).run("time?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "it is noon"
    assert asked == [{}]  # offered with no argument, called with none
    [now] = await NoArguments().resolve()
    assert _schema(now) == {"type": "object", "properties": {}}
    kept = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert _schema(replace(now, spec=now.spec.model_copy(update={"input_schema": kept}))) == kept


async def test_a_tool_result_over_a_mebibyte_reaches_claude_as_a_preview(
    harness: Harness, tmp_path: Path
) -> None:
    """A tool's result comes back from the CLI in one message, which the SDK reads into a 1 MiB
    buffer unless its options say more (the harness's say enough for a team's own MCP server's
    result, or a built-in's): a harness tool's large result is cut before, so even a team's own
    1 MiB buffer holds its preview."""

    @tool(side_effects="read")
    def export(rows: int) -> str:
        """Export a report."""
        return "x" * (1024 * 1024 + 1)

    script = [{"tool": "export", "args": {"rows": 3}}, {"text": "{last}"}]
    own = options(tmp_path, script, max_buffer_size=1024 * 1024)
    result = await harness.wrap(own, id="small", tools=[export]).run("export", user="u")
    assert result.status is RunStatus.SUCCESS and "/large_tool_results/" in result.answer
    assert len(result.answer) < 25_000


async def test_an_approval_stops_the_cli_and_a_resume_continues_its_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    agent = harness.wrap(
        options(tmp_path, [{"tool": "refund", "args": {"order": "o2"}}, {"text": "done"}]),
        id="refunds",
        tools=[refund],
    )
    paused = await agent.run("refund o2", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and refunds == []
    session = started_with(tmp_path)
    assert session["resume"] is None and session["prompt"] == "refund o2"
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "done"
    assert refunds == ["o2"]
    resumed = started_with(tmp_path)
    # told which call the person answered about, by the name Claude calls it: the session
    # holds that call's result as "waiting", which is not "no result yet"
    assert resumed["resume"] is not None
    assert resumed["prompt"] == RESUMED_CALL.format(tool="mcp__trellis__refund")


@tool(side_effects="read")
async def pick_warehouse() -> str:
    """Ask which warehouse ships."""
    runtime = current()
    assert runtime is not None
    return f"ships from {await runtime.ask('Which warehouse?')}"


async def test_a_question_a_tool_asked_resumes_the_session_without_naming_a_call(
    harness: Harness, tmp_path: Path
) -> None:
    """A pause that is not an approval names no call: the resumed session is told to make
    again the call that has no result yet."""
    script = [{"tool": "pick_warehouse", "args": {}}, {"text": "{last}"}]
    agent = harness.wrap(options(tmp_path, script), id="shipping", tools=[pick_warehouse])
    asked = await agent.run("where does it ship from?", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.tool_call is None
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="Leeds", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "ships from Leeds"
    assert started_with(tmp_path)["prompt"] == RESUMED


async def test_built_in_tools_are_governed_and_not_run_again_after_a_pause(
    harness: Harness, tmp_path: Path
) -> None:
    """``Bash`` asks a person (governance: irreversible); approved, it runs once — the resumed
    session does not run it again when the run pauses later, on a harness tool."""
    refunds.clear()
    script = [
        {"builtin": "Read", "args": {"file_path": "orders.csv"}},
        {"builtin": "Bash", "args": {"command": "ls"}},
        {"tool": "refund", "args": {"order": "o3"}},
        {"text": "done: {last}"},
    ]
    agent = harness.wrap(options(tmp_path, script), id="ops", tools=[refund])
    asked = await agent.run("tidy up, then refund o3", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.tool == "Bash" and asked.interrupt.question == (
        "Approve Bash? Bash is irreversible."
    )
    assert built_ins(tmp_path) == [{"tool": "Read", "args": {"file_path": "orders.csv"}}]
    again = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert started_with(tmp_path)["prompt"] == RESUMED_CALL.format(tool="Bash")  # a built-in
    assert again.interrupt is not None and again.interrupt.tool_call is not None
    assert again.interrupt.tool_call.tool == "refund"
    done = await agent.resume(again.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: refunded o3"
    assert [b["tool"] for b in built_ins(tmp_path)] == ["Read", "Bash"]  # each once
    assert refunds == ["o3"]


async def test_a_built_in_tool_goes_through_the_hooks_then_the_teams_own_callback(
    harness: Harness, tmp_path: Path
) -> None:
    asked: list[tuple[str, dict[str, Any]]] = []

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        asked.append((name, args))
        if name == "Write":
            return PermissionResultDeny(message="no writes on Fridays")
        return PermissionResultAllow()

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            if call.tool == "Grep":
                return Deny("no searching here")
            if call.tool == "Edit":
                return Rewrite({**call.args, "file_path": "safe.txt"})
            return None

    script = [
        {"builtin": "Grep", "args": {"pattern": "x"}},
        {"builtin": "Edit", "args": {"file_path": "/etc/hosts"}},
        {"builtin": "Write", "args": {"file_path": "a.txt"}},
        {"text": "said: {last}"},
    ]
    target = options(tmp_path, script, can_use_tool=own)
    agent = harness.wrap(target, id="careful", hooks=[Guard()])
    events = [e async for e in agent.stream("work", user="u1")]
    assert events[-1].data["result"] == "said: no writes on Fridays"
    assert built_ins(tmp_path) == [{"tool": "Edit", "args": {"file_path": "safe.txt"}}]
    assert asked == [("Edit", {"file_path": "safe.txt"}), ("Write", {"file_path": "a.txt"})]
    notices = [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"]
    assert notices == ["Edit", "Write"]  # writes are announced, as harness writes are


async def test_a_session_the_cli_no_longer_holds_is_run_again_from_its_prompt(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    agent = harness.wrap(
        options(tmp_path, [{"tool": "refund", "args": {"order": "o4"}}, {"text": "done"}]),
        id="moved",
        tools=[refund],
    )
    paused = await agent.run("refund o4", user="u1")
    assert paused.interrupt is not None
    for kept in (tmp_path / "fake-claude").glob("*.json"):
        kept.unlink()  # resumed on a machine whose CLI never held the session
    events: list[Any] = []
    record = await harness.runs.get(paused.run_id)
    assert record is not None
    resumed, resolution = await agent._resolution(
        paused.interrupt.interrupt_id, "approve", None, "u1", tenant=record.tenant_id
    )
    done = await agent._continue(resumed, resolution, events.append)
    assert done.status is RunStatus.SUCCESS and refunds == ["o4"]
    [warning] = [e for e in events if e.data.get("code") == "claude_session"]
    assert "No conversation found with session ID" in warning.data["message"]
    assert started_with(tmp_path)["prompt"] == "refund o4"  # from the prompt, against the journal


async def test_a_permission_prompt_tool_of_the_teams_own_is_refused(
    harness: Harness, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"text": "x"}], permission_prompt_tool_name="mcp__ops__ask")
    result = await harness.wrap(target, id="own").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "give your own permission check as can_use_tool" in result.error.message


async def test_the_context_is_appended_to_the_system_prompt(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"text": "email"}], system_prompt="You help.")
    await memory_harness.wrap(target, id="helper").run("reach me how?", user="u1")
    assert started_with(tmp_path)["system_prompt"] == f"You help.\n\n{memory_service.context_text}"


async def test_the_hints_narrow_the_tools_for_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}"

        return tool(lookup, name=f"t{i}", side_effects="read")

    memory_service.candidates = ["t2", "t5"]
    target = options(tmp_path, [{"text": "done"}])
    await memory_harness.wrap(target, id="n", tools=[make(i) for i in range(6)]).run("x", user="u1")
    assert started_with(tmp_path)["tools"] == ["t2", "t5", *MEMORY_TOOLS, "read_file"]


async def test_streaming_carries_the_assistant_text(harness: Harness, tmp_path: Path) -> None:
    agent = harness.wrap(options(tmp_path, [{"text": "hello there"}]), id="s")
    events = [e async for e in agent.stream("hi", user="u1")]
    assert [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT] == [
        "hello there"
    ]


async def test_mcp_servers_the_team_configured_as_a_file_or_json_stay(
    harness: Harness, tmp_path: Path
) -> None:
    """``mcp_servers`` may be a path or a JSON string (the CLI's ``--mcp-config``): the harness
    adds its server beside the team's, never in place of them."""
    erp = {"type": "stdio", "command": "erp-mcp", "args": []}
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"erp": erp}}))
    script = [{"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    for configured in (config, str(config), json.dumps({"mcpServers": {"erp": erp}})):
        target = options(tmp_path, script, mcp_servers=configured)
        agent = harness.wrap(target, id=f"mcp-{len(harness.agents)}", tools=[tool(refund.fn)])
        assert (await agent.run("refund o9", user="u1")).answer == "done"
        servers = json.loads(started_with(tmp_path)["mcp_config"])["mcpServers"]
        assert servers["erp"] == erp and servers["trellis"]["type"] == "sdk"


#: a result over ``tools.results.MAX_CHARS``, and where the run keeps it
LONG = "\n".join(f"row {n}" for n in range(1, 30_001))
KEPT = f"/large_tool_results/{content_key('result', LONG)}"


@tool(side_effects="read")
def dump() -> str:
    """Dump every row."""
    return LONG


async def test_a_large_result_is_previewed_naming_where_the_run_keeps_it(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"tool": "dump", "args": {}}, {"text": "{last}"}]
    agent = harness.wrap(options(tmp_path, script), id="d", tools=[dump])
    preview = (await agent.run("dump it", user="u1")).answer
    assert f"saved in the filesystem at this path: {KEPT}" in preview and len(preview) < 25_000
    assert "row 30000" in preview and "row 100" not in preview


async def test_a_kept_result_is_read_in_pages_after_a_pause(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [
        {"tool": "dump", "args": {}},
        {"tool": "refund", "args": {"order": "o1"}},
        {"tool": "read_file", "args": {"file_path": KEPT, "offset": 10, "limit": 2}},
        {"text": "{last}"},
    ]
    agent = harness.wrap(options(tmp_path, script), id="dumper", tools=[dump, refund])
    paused = await agent.run("dump it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert refunds == ["o1"]  # the session went on: read_file read the journal's copy
    assert finished.answer == (
        f'11  row 11\n12  row 12\n…[29988 more lines: read_file(file_path="{KEPT}", offset=12)]'
    )
