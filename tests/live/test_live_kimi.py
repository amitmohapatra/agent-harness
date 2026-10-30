"""Kimi Code (the real ``kimi acp``) through the gateway's OpenAI route to the live model: the
harness's tools reach it as an MCP server, the run's headers go with every model call, and —
when the model calls the harness tool that asks for approval — the resumed run continues
Kimi's session and the approved call runs once. Opt-in (``-m live``), like every live test; no
Moonshot key: the model is the gateway's (``TRELLIS_LIVE_MODEL``, or a Kimi model the gateway
serves). Skipped without the ``kimi`` program. Whether the model calls the tool at all is the
model's (the test says so when it does not)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests.live.conftest import GATEWAY_URL, MODEL, live_harness, needs_gateway
from tests.support.arm import arm
from tests.support.kimi import KIMI_BIN
from trellis import kimi, tool
from trellis.contracts import RunStatus

pytestmark = [
    pytest.mark.live,
    needs_gateway,
    pytest.mark.skipif(KIMI_BIN is None, reason="the kimi program is not installed"),
]


@pytest.mark.timeout(600)
async def test_kimi_runs_through_the_gateway_and_resumes_its_session(tmp_path: Path) -> None:
    assert GATEWAY_URL is not None
    closed: list[str] = []

    @tool(side_effects="irreversible")
    def close_ticket(ticket: str) -> str:
        """Close a support ticket by its id."""
        closed.append(ticket)
        return f"ticket {ticket} is closed"

    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    target = kimi(
        model=MODEL,
        cwd=str(work),
        env={"KIMI_CODE_HOME": str(home), "KIMI_DISABLE_TELEMETRY": "1"},
    )
    async with live_harness() as h:
        agent = arm(h.wrap(target, id=f"live-kimi-{uuid.uuid4().hex[:8]}"), [close_ticket])
        first = await agent.run(
            "Close ticket T-1 with the close_ticket tool, then say done.", user="live-user"
        )
        assert first.status in (RunStatus.SUCCESS, RunStatus.PAUSED), first.error
        if first.status is RunStatus.SUCCESS:
            assert closed == []  # irreversible: never without an approval
            pytest.skip("the live model answered without calling close_ticket")
        assert first.interrupt is not None and first.interrupt.tool_call is not None
        assert first.interrupt.tool_call.tool == "close_ticket" and closed == []
        done = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="live-user")
        assert done.status is RunStatus.SUCCESS, done.error
        assert closed == ["T-1"]  # once, in the session Kimi resumed
