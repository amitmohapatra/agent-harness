"""One approval rule for a tool its framework runs, as for the harness's own: a person asked about
a call Claude Code runs itself — a tool of its own MCP server, a built-in — picks from the
choices of the call's risk (allow once, for this conversation, always with a memory service,
deny; allow once or deny for an irreversible one), and what they allow for the conversation or
always is honoured as Claude Code's own "don't ask again" would be. The tool catalog's word on
such a tool decides before Claude Code's own question. Every adapter and way a tool arrives is
walked in ``tests/matrix/test_approval_rule.py``; these are the fast cases ``make test`` keeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from tests.support.adapters import CLI
from tests.support.memory import FakeMemoryService
from trellis import Harness
from trellis.contracts import RunStatus
from trellis.harness.governance.decision import ALWAYS, CONVERSATION, DENY, ONCE, SAID

#: What the ledger server's tool ran, by call.
ran: list[int] = []
FOUR = (ONCE, CONVERSATION, ALWAYS, DENY)


@sdk_tool("tidy", "Tidy an entry of the ledger.", {"n": int})
async def tidy(args: dict[str, Any]) -> dict[str, Any]:
    """A tool of Claude's own in-process server that says nothing: no annotations at all."""
    ran.append(args["n"])
    return {"content": [{"type": "text", "text": f"tidied {args['n']}"}]}


def claude(tmp: Path, *calls: dict[str, Any]) -> ClaudeAgentOptions:
    """Claude Code (the scripted CLI) making ``calls``, with its own ``ledger`` server."""
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps([*calls, {"text": "done"}]),
            "FAKE_CLAUDE_SESSIONS": str(tmp),
            "FAKE_CLAUDE_BUILTINS": str(tmp / "builtins.jsonl"),
        },
        mcp_servers={"ledger": create_sdk_mcp_server("ledger", tools=[tidy])},
    )


def tidying(n: int) -> dict[str, Any]:
    return {"tool": "tidy", "server": "ledger", "args": {"n": n}}


@pytest.mark.host_execution
async def test_a_claude_servers_tool_that_says_nothing_offers_every_choice_and_always_holds(
    memory_harness: Harness, tmp_path: Path
) -> None:
    ran.clear()
    agent = memory_harness.wrap(claude(tmp_path, tidying(1), tidying(2)), id="ledger")
    first = await agent.run("tidy", user="ada", thread="c1")
    asked = first.interrupt
    assert asked is not None and ran == [], first
    assert list(asked.options) == list(FOUR)
    assert asked.question == (
        "Approve mcp__ledger__tidy? mcp__ledger__tidy has not declared its side effects. "
        f"{SAID[FOUR]}"
    )
    done = await agent.resume(asked.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    assert done.status is RunStatus.SUCCESS and ran == [1, 2], done  # 2: not asked
    await memory_harness.writes.drain()
    other = memory_harness.wrap(claude(tmp_path, tidying(3)), id="ledger-2")
    again = await other.run("tidy", user="ada", thread="c2")  # the next conversation
    assert again.status is RunStatus.SUCCESS and ran == [1, 2, 3], again


@pytest.mark.host_execution
async def test_claude_codes_own_question_offers_the_choices_of_the_risk(
    harness: Harness, tmp_path: Path
) -> None:
    """A write Claude Code asks about itself (its ``Write``): the person picks as for a
    harness tool's write, and what they allow for the conversation covers the next call."""
    calls = [{"builtin": "Write", "args": {"n": n}} for n in (1, 2)]
    agent = harness.wrap(claude(tmp_path, *calls), id="writer")
    first = await agent.run("write", user="ada", thread="c1")
    asked = first.interrupt
    assert asked is not None and list(asked.options) == [ONCE, CONVERSATION, DENY]
    assert asked.question == (
        "Approve Write? Claude Code asks before it runs this call. "
        "Allow once, allow for this conversation, or deny."
    )
    done = await agent.resume(asked.interrupt_id, "answer", answer="run", reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done
    written = (tmp_path / "builtins.jsonl").read_text().splitlines()
    assert [json.loads(line)["args"]["n"] for line in written] == [1, 2]


@pytest.mark.host_execution
async def test_the_catalogs_risk_decides_before_claude_codes_own_question(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """An administrator's risk in the tool catalog wins over the framework's own question as
    over a tool's declaration: Claude Code's ``Write`` declared ``read`` runs, nobody asked."""
    memory_service.catalog["Write"] = {"name": "Write", "side_effects": "read"}
    agent = memory_harness.wrap(
        claude(tmp_path, {"builtin": "Write", "args": {"n": 1}}), id="catalogued"
    )
    done = await agent.run("write", user="ada")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
