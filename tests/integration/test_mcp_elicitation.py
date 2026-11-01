"""A server that asks the person in the middle of a call (MCP elicitation), against a real server
on the official SDK (``tests/support/mcp_asking_server.py``) over stdio, on both protocol eras:
the question pauses the run as an ``ask`` does (one field of a few values a choice, else a form
with the server's schema), the answer resumes it and goes back to the server, and the call is
made once to its end."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from trellis import Harness, Runtime, mcp, stdio
from trellis.contracts import InterruptReason, RunStatus

SERVER = Path(__file__).parents[1] / "support" / "mcp_asking_server.py"

#: The protocol before 2026-07-28 (the question a request of the server's own, during the
#: call) and 2026-07-28 (the question in the call's result, the call made again with the
#: answer).
ERAS = pytest.mark.parametrize("legacy", [True, False], ids=["handshake", "2026-07-28"])


def shop(legacy: bool) -> Any:
    return mcp(stdio(sys.executable, str(SERVER), env={"LEGACY": "1"} if legacy else {}))


async def calling(tool: str) -> Any:
    async def agent_fn(input: str, agent: Runtime) -> Any:
        said = await agent.tools.call(f"shop-{tool}", item="shirt")
        return [said, await agent.tools.call("shop-served_calls")]

    return agent_fn


@ERAS
async def test_a_question_of_a_few_values_is_a_choice(harness: Harness, legacy: bool) -> None:
    agent = arm(harness.wrap(await calling("order"), id="orders"), [shop(legacy)])
    paused = await agent.run("order a shirt", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    asked = paused.interrupt
    assert asked.reason is InterruptReason.CHOICE and asked.ui == "choice"
    assert asked.question == "Which size of shirt?"
    assert asked.options == ["S", "L"]
    assert asked.assignee == "user:ada"
    done = await agent.resume(asked.interrupt_id, "answer", answer="L", reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    said, served = done.answer
    assert said == "ordered shirt in size L"
    assert served == ["order"]  # the tool's body ran once, with the answer


@ERAS
async def test_a_form_has_the_servers_schema(harness: Harness, legacy: bool) -> None:
    agent = arm(harness.wrap(await calling("book"), id="bookings"), [shop(legacy)])
    paused = await agent.run("book it", user="ada")
    asked = paused.interrupt
    assert asked is not None and asked.reason is InterruptReason.QUESTION and asked.ui == "form"
    assert asked.question == "When should shirt be delivered?"
    assert asked.expects is not None
    assert asked.expects["properties"]["day"] == {
        "format": "date",
        "title": "Day",
        "type": "string",
    }
    answer = {"day": "2026-11-02", "note": "ring twice"}
    done = await agent.resume(asked.interrupt_id, "answer", answer=answer, reviewer="ada")
    said, served = done.answer
    assert said == "shirt on 2026-11-02: ring twice" and served == ["book"]


@ERAS
async def test_a_rejected_question_is_declined(harness: Harness, legacy: bool) -> None:
    agent = arm(harness.wrap(await calling("order"), id="declines"), [shop(legacy)])
    paused = await agent.run("order a shirt", user="ada")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="ada")
    said, served = done.answer
    assert said == "not ordered (decline)" and served == ["order"]


async def test_a_cancelled_question_ends_the_run(harness: Harness) -> None:
    agent = arm(harness.wrap(await calling("order"), id="cancels"), [shop(legacy=False)])
    paused = await agent.run("order a shirt", user="ada")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="ada")
    assert done.status is RunStatus.CANCELLED
