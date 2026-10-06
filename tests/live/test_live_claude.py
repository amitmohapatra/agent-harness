"""The Claude Agent SDK with the real Claude Code CLI, through the gateway's Anthropic route to
the live model: the harness's permission callback and in-process tools reach the CLI, the
session it runs in is kept in the run's journal, and — when the model calls the harness tool
that asks for approval — the resumed run continues that session and the approved call runs
once. Skipped without the CLI. The local model's answers are not what is checked: whether it
calls the tool at all is the model's (the test says so when it does not)."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from tests.live.conftest import BIFROST_URL, MODEL, live_harness, needs_gateway, settings
from trellis import tool
from trellis.contracts import RunStatus
from trellis.harness.journal import Journal

pytestmark = [
    pytest.mark.live,
    needs_gateway,
    pytest.mark.skipif(shutil.which("claude") is None, reason="needs the Claude Code CLI"),
]


@pytest.fixture
def cli_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The CLI as a deployment runs it: none of this process's own Claude settings (a Claude
    Code session's variables point a child CLI elsewhere), its sessions in a home of its own."""
    for name in list(os.environ):
        if name.startswith(("CLAUDE", "ANTHROPIC")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.timeout(600)
async def test_claude_runs_through_the_gateway_and_resumes_its_session(cli_home: Path) -> None:
    assert BIFROST_URL is not None
    closed: list[str] = []

    @tool(side_effects="irreversible")
    def close_ticket(ticket: str) -> str:
        """Close a support ticket by its id."""
        closed.append(ticket)
        return f"ticket {ticket} is closed"

    key = settings().bifrost_virtual_key or "unused"
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt="You close support tickets with the close_ticket tool. Be brief.",
        tools=[],  # no built-in tools: the model sees the harness's tool only
        max_turns=4,
        env={
            "ANTHROPIC_BASE_URL": f"{BIFROST_URL.removesuffix('/v1')}/anthropic",
            "ANTHROPIC_API_KEY": key,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    )
    async with live_harness() as h:
        agent = h.wrap(options, id=f"live-claude-{uuid.uuid4().hex[:8]}", tools=[close_ticket])
        first = await agent.run("Close ticket T-1.", user="live-user")
        assert first.status in (RunStatus.SUCCESS, RunStatus.PAUSED), first.error
        if first.status is RunStatus.SUCCESS:
            assert closed == []  # irreversible: never without an approval
            pytest.skip("the live model answered without calling close_ticket")
        assert first.interrupt is not None and first.interrupt.tool_call is not None
        assert first.interrupt.tool_call.tool == "close_ticket" and closed == []
        record = await h.runs.get(first.run_id, tenant=await h.tenant())
        assert record is not None
        journal = await Journal.read(record.checkpoint, h.runs.artifacts, tenant=record.tenant_id)
        assert journal.session  # the CLI's session, kept with the pause
        [kept] = list((cli_home / ".claude").rglob(f"{journal.session}.jsonl"))
        before = len(kept.read_text().splitlines())  # the CLI holds it: the resume continues it
        done = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status in (RunStatus.SUCCESS, RunStatus.PAUSED), done.error
    assert len(kept.read_text().splitlines()) > before  # the same session went on
    assert len(closed) <= 1  # the approved call runs once, if the model makes it again
