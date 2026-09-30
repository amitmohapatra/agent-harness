"""Claude Agent SDK: the real SDK drives a scripted stand-in for the Claude Code CLI
(``tests/support/fake_claude_cli.py``) through ``cli_path`` — subprocess, control protocol and
in-process MCP tool calls included."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from trellis.contracts import RunEventType, RunStatus

from tests.support.memory import FakeMemoryService
from trellis import Harness, tool

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
    agent = harness.wrap(target, id="refunds", tools=[refund], approve={"refund": False})
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
    await memory_harness.wrap(target, id="helper", memory="read").run("reach me how?", user="u1")
    assert started_with(tmp_path)["system_prompt"] == f"You help.\n\n{memory_service.context_text}"


async def test_streaming_carries_the_assistant_text(harness: Harness, tmp_path: Path) -> None:
    agent = harness.wrap(options(tmp_path, [{"text": "hello there"}]), id="s")
    events = [e async for e in agent.stream("hi", user="u1")]
    assert [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT] == [
        "hello there"
    ]
