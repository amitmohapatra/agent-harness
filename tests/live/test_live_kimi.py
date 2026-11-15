"""Kimi Code (the real ``kimi acp``) on the live model — through the gateway's OpenAI route, or,
with no gateway, Kimi's own configuration naming the model's endpoint (``KIMI_MODEL_*``: the
OpenAI-compatible ``LLM_BASE_URL`` and ``LLM_API_KEY``): the
harness's tools reach it as an MCP server, the run's headers go with every model call, and —
when the model calls the harness tool that asks for approval — the resumed run continues
Kimi's session and the approved call runs once. Opt-in (``-m live``), like every live test; no
Moonshot key: the model is the gateway's or the endpoint's (``TRELLIS_LIVE_MODEL``, or a Kimi
model the gateway serves). Skipped without the ``kimi`` program. Whether the model calls the tool at all is the
model's (the test says so when it does not)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests.live.conftest import DIRECT_URL, MODEL, live_harness, needs_model, settings
from tests.support.arm import arm
from tests.support.kimi import KIMI_BIN
from trellis import kimi, tool
from trellis.contracts import RunStatus

pytestmark = [
    pytest.mark.live,
    needs_model,
    pytest.mark.skipif(KIMI_BIN is None, reason="the kimi program is not installed"),
]


@pytest.mark.timeout(600)
async def test_kimi_runs_on_the_live_model_and_resumes_its_session(tmp_path: Path) -> None:
    closed: list[str] = []

    @tool(side_effects="irreversible")
    def close_ticket(ticket: str) -> str:
        """Close a support ticket by its id."""
        closed.append(ticket)
        return f"ticket {ticket} is closed"

    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = {"KIMI_CODE_HOME": str(home), "KIMI_DISABLE_TELEMETRY": "1"}
    if DIRECT_URL is not None:  # no gateway: Kimi's own configuration names the endpoint
        env |= {
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": DIRECT_URL,
            "KIMI_MODEL_API_KEY": settings().llm_api_key or "none",
            "KIMI_MODEL_NAME": MODEL,
        }
    target = kimi(model=MODEL, cwd=str(work), env=env)
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
