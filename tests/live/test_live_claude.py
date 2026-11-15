"""The Claude Agent SDK with the real Claude Code CLI on the live model — through the gateway's
Anthropic route, or directly on Anthropic's API with the provider's key when there is no gateway: the harness's permission callback and in-process tools reach the CLI, the
session it runs in is kept in the run's journal, and — when the model calls the harness tool
that asks for approval — the resumed run continues that session and the approved call runs
once. Skipped without the CLI. The local model's answers are not what is checked: whether it
calls the tool at all is the model's (the test says so when it does not)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from tests.live.conftest import (
    MODEL,
    claude_cli_env,
    live_harness,
    needs_claude_cli,
    needs_claude_model,
)
from tests.support.arm import arm
from trellis import Hooks, Result, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness.journal import Journal

pytestmark = [pytest.mark.live, needs_claude_model, needs_claude_cli]


class Ended(Hooks):
    """The run's journal as its last attempt ended (a finished run's checkpoint keeps none)."""

    def __init__(self) -> None:
        self.journal: Journal | None = None

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        self.journal = run.replay.journal.model_copy(deep=True)


@pytest.mark.timeout(600)
async def test_claude_runs_on_the_live_model_and_resumes_its_session(cli_home: Path) -> None:
    closed: list[str] = []

    @tool(side_effects="irreversible")
    def close_ticket(ticket: str) -> str:
        """Close a support ticket by its id."""
        closed.append(ticket)
        return f"ticket {ticket} is closed"

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt="You close support tickets with the close_ticket tool. Be brief.",
        tools=[],  # no built-in tools: the model sees the harness's tool only
        max_turns=4,
        cwd=cli_home,
        env=claude_cli_env(),
    )
    async with live_harness() as h:
        ended = Ended()
        agent = arm(
            h.wrap(options, id=f"live-claude-{uuid.uuid4().hex[:8]}", hooks=[ended]), [close_ticket]
        )
        first = await agent.run("Close ticket T-1.", user="live-user")
        assert first.status in (RunStatus.SUCCESS, RunStatus.PAUSED), first.error
        if first.status is RunStatus.SUCCESS:
            assert closed == []  # irreversible: never without an approval
            pytest.skip("the live model answered without calling close_ticket")
        assert first.interrupt is not None and first.interrupt.tool_call is not None
        assert first.interrupt.tool_call.tool == "close_ticket" and closed == []
        record = await h.runs.get(first.run_id, tenant=await h._tenant())
        assert record is not None
        journal = await Journal.read(record.checkpoint, h.runs.artifacts, tenant=record.tenant_id)
        assert journal.session  # the CLI's session, kept with the pause
        # its transcript is kept in the run's journal (the SDK's session store,
        # ``bridges.claude.JournalSessions``): the resume continues it from there
        before = len(journal.transcripts.get(journal.session) or [])
        assert before
        done = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status in (RunStatus.SUCCESS, RunStatus.PAUSED), done.error
    after = ended.journal  # the resumed attempt's, as it ended
    assert after is not None and after.session == journal.session  # the same session went on
    assert len(after.transcripts[journal.session]) > before
    assert len(closed) <= 1  # the approved call runs once, if the model makes it again
