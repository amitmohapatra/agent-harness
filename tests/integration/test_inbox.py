"""The inbox without a page of the harness's: ``h.inbox`` lists the paused runs (with the
interrupt each waits on), ``agent.resume`` answers them — checked first, as the reviewer named —
and a screen of your own serves both in a few lines. AG-UI's resume answers a chat's own runs
(``test_agui.py``)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

from trellis import Harness, Runtime, tool
from trellis.contracts import ConfigurationError, InterruptReason, RunStatus


@tool(side_effects="irreversible")
def refund(amount: int) -> str:
    """Refund an amount."""
    return f"refunded {amount}"


async def refunds(input: Any, agent: Runtime) -> Any:
    plan = await agent.ask("Which plan?", options=["a", "b"], assignee="role:sales")
    return f"{plan}: {await agent.tools.call('refund', amount=5)}"


async def test_paused_runs_are_listed_and_answered_with_h_inbox_and_agent_resume(
    harness: Harness,
) -> None:
    agent = harness.wrap(refunds, id="refunds", tools=[refund])
    paused = await agent.run("go", user="ada")
    [listed] = await harness.inbox("role:sales")
    assert (listed.run_id, listed.agent_id, listed.assignee) == (
        paused.run_id,
        "refunds",
        "role:sales",
    )
    assert listed.awaiting is not None and listed.awaiting.options == ["a", "b"]
    assert await harness.inbox("role:none") == []
    owner = harness.agents[listed.agent_id]
    with pytest.raises(ConfigurationError, match="not one of the options"):
        await owner.resume(listed.run_id, "answer", answer="z", reviewer="lee")
    record = await harness.runs.get(listed.run_id)
    assert record is not None and record.status is RunStatus.PAUSED and record.attempt == 1
    asked = await owner.resume(
        listed.awaiting.interrupt_id, "answer", answer="a", reviewer="lee", comment="the usual"
    )
    assert asked.status is RunStatus.PAUSED
    record = await harness.runs.get(listed.run_id)
    assert record is not None and record.last_resolution is not None
    assert (record.last_resolution.reviewer, record.last_resolution.comment) == ("lee", "the usual")
    [approval] = await harness.inbox()  # anyone's
    assert approval.awaiting is not None and approval.awaiting.reason is InterruptReason.APPROVAL
    assert approval.awaiting.tool_call is not None and approval.awaiting.tool_call.tool == "refund"
    done = await owner.resume(approval.run_id, "approve", reviewer="lee")
    assert done.status is RunStatus.SUCCESS and done.answer == "a: refunded 5"
    other = await agent.run("go", user="ada")
    cancelled = await agent.resume(other.run_id, "cancel", reviewer="lee")
    assert cancelled.status is RunStatus.CANCELLED
    record = await harness.runs.get(other.run_id)
    assert record is not None and record.last_resolution is not None
    assert record.last_resolution.reviewer == "lee"
    with pytest.raises(ConfigurationError, match="no run run_other"):
        await agent.resume("run_other.1.1", "answer", answer="a", reviewer="lee")


async def test_a_screen_of_your_own_lists_and_answers_with_the_same_two_calls(
    harness: Harness,
) -> None:
    """What a team's own inbox route needs: ``h.inbox`` and ``agent.resume``, a refused answer
    ``409`` with why, the reviewer named by the request."""
    app = FastAPI()

    @app.get("/inbox")
    async def listing(assignee: str | None = None) -> list[dict[str, Any]]:
        return [
            {"run_id": s.run_id, "interrupt": s.awaiting.awaiting()}
            for s in await harness.inbox(assignee)
            if s.agent_id in harness.agents and s.awaiting is not None
        ]

    @app.post("/inbox/{run_id}")
    async def answer(run_id: str, request: Request) -> dict[str, Any]:
        given = await request.json()
        record = await harness.runs.get(run_id)
        if record is None or record.agent_id not in harness.agents:
            raise HTTPException(404, f"no run {run_id}")
        try:
            result = await harness.agents[record.agent_id].resume(
                run_id,
                given["decision"],
                answer=given.get("answer"),
                comment=given.get("comment"),
                reviewer=request.headers["x-user"],
            )
        except ConfigurationError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": result.status.value}

    agent = harness.wrap(refunds, id="refunds", tools=[refund])
    paused = await agent.run("go", user="ada")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inbox", headers={"x-user": "lee"}
    ) as http:
        [listed] = (await http.get("/inbox", params={"assignee": "role:sales"})).json()
        assert listed["run_id"] == paused.run_id and listed["interrupt"]["options"] == ["a", "b"]
        url = f"/inbox/{paused.run_id}"
        refused = await http.post(url, json={"decision": "answer", "answer": "z"})
        assert refused.status_code == 409 and "not one of the options" in refused.text
        assert (await http.post("/inbox/run_none", json={"decision": "cancel"})).status_code == 404
        went_on = await http.post(url, json={"decision": "answer", "answer": "a"})
        assert went_on.json() == {"status": "PAUSED"}  # now on the refund's approval
        done = await http.post(url, json={"decision": "approve"})
        assert done.json() == {"status": "SUCCESS"}
    record = await harness.runs.get(paused.run_id)
    assert record is not None and record.last_resolution is not None
    assert record.last_resolution.reviewer == "lee"
