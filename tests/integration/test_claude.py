"""Claude Agent SDK: the real SDK drives a scripted stand-in for the Claude Code CLI
(``tests/support/fake_claude_cli.py``) through ``cli_path`` — subprocess, control protocol and
in-process MCP tool calls included."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from trellis import Harness, tool
from trellis.contracts import RunEventType, RunStatus

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
        },
        **fields,
    )


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
    assert cli["allowed_tools"] == "mcp__trellis__refund"
    assert cli["system_prompt"] == "You refund."
    assert target.mcp_servers == {}  # the team's options are untouched


async def test_an_approval_stops_the_cli_and_a_resume_reruns(
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
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "done"
    assert refunds == ["o2"]


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
    allowed = started_with(tmp_path)["allowed_tools"].split(",")
    assert allowed == [f"mcp__trellis__{n}" for n in ("t2", "t5", *MEMORY_TOOLS)]


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
