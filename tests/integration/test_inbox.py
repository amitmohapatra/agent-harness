"""The reference inbox (``h.serve_inbox``): its page, the paused runs it lists, and answers it
checks before the run goes on in the background."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunStatus


@tool(side_effects="irreversible")
def refund(amount: int) -> str:
    """Refund an amount."""
    return f"refunded {amount}"


async def refunds(input: Any, agent: Runtime) -> Any:
    plan = await agent.ask("Which plan?", options=["a", "b"], assignee="role:sales")
    return f"{plan}: {await agent.tools.call('refund', amount=5)}"


def reviewer(request: Request) -> str:
    return request.headers.get("x-user", "")


@pytest.fixture
async def inbox() -> AsyncIterator[tuple[Harness, httpx.AsyncClient]]:
    harness = Harness(config=Settings())
    harness.wrap(refunds, id="refunds", tools=[refund])
    app = FastAPI()
    harness.serve_inbox(app, identity=reviewer)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inbox", headers={"x-user": "lee"}
    ) as http:
        yield harness, http
    await harness.aclose()


async def settled(
    harness: Harness, run_id: str, status: RunStatus, *, attempt: int = 1
) -> RunStatus:
    """The run's status once it is ``status`` on ``attempt`` or later (it goes on in the
    background)."""
    for _ in range(200):
        record = await harness.runs.get(run_id)
        assert record is not None
        if record.status is status and record.attempt >= attempt:
            return record.status
        await asyncio.sleep(0.01)
    raise AssertionError(f"{run_id} never became {status}")


async def test_the_page_lists_paused_runs_and_answers_them(
    inbox: tuple[Harness, httpx.AsyncClient],
) -> None:
    harness, http = inbox
    page = await http.get("/inbox")
    assert page.status_code == 200 and "trellisComponents" in page.text
    paused = await harness.agents["refunds"].run("go", user="ada")
    assert paused.interrupt is not None
    [listed] = (await http.get("/inbox/runs", params={"assignee": "role:sales"})).json()
    assert listed["interrupt"]["options"] == ["a", "b"] and listed["agent_id"] == "refunds"
    assert (await http.get("/inbox/runs", params={"assignee": "role:none"})).json() == []
    url = f"/inbox/runs/{paused.run_id}/resume"
    answer = {"interrupt_id": paused.interrupt.interrupt_id, "decision": "answer"}
    refused = await http.post(url, json={**answer, "answer": "z"})
    assert refused.status_code == 409 and "BAD_RESUME" in refused.json()["detail"]
    accepted = await http.post(url, json={**answer, "answer": "a", "comment": "the usual"})
    assert accepted.status_code == 202
    assert await settled(harness, paused.run_id, RunStatus.PAUSED, attempt=2) is RunStatus.PAUSED
    record = await harness.runs.get(paused.run_id)
    assert record is not None and record.awaiting is not None and record.attempt == 2
    assert record.last_resolution is not None
    assert (record.last_resolution.reviewer, record.last_resolution.comment) == ("lee", "the usual")
    approval = {"interrupt_id": record.awaiting.interrupt_id, "decision": "approve"}
    assert (await http.post(url, json=approval)).status_code == 202
    assert await settled(harness, paused.run_id, RunStatus.SUCCESS) is RunStatus.SUCCESS


async def test_an_answer_to_a_run_not_served_here_is_not_found(
    inbox: tuple[Harness, httpx.AsyncClient],
) -> None:
    harness, http = inbox
    paused = await harness.agents["refunds"].run("go", user="ada")
    assert paused.interrupt is not None
    answer = {"interrupt_id": paused.interrupt.interrupt_id, "decision": "answer", "answer": "a"}
    assert (await http.post("/inbox/runs/run_none/resume", json=answer)).status_code == 404
    other = {**answer, "interrupt_id": "run_other.1.1"}
    response = await http.post(f"/inbox/runs/{paused.run_id}/resume", json=other)
    assert response.status_code == 404 and "is not a question of" in response.json()["detail"]
    nobody = await http.get("/inbox/runs", headers={"x-user": ""})
    assert nobody.status_code == 401


async def test_a_run_that_fails_to_go_on_is_logged(
    inbox: tuple[Harness, httpx.AsyncClient], caplog: pytest.LogCaptureFixture
) -> None:
    harness, http = inbox
    agent = harness.agents["refunds"]
    paused = await agent.run("go", user="ada")
    assert paused.interrupt is not None

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the store went away")

    agent._continue = broken  # type: ignore[method-assign]
    answer = {"interrupt_id": paused.interrupt.interrupt_id, "decision": "answer", "answer": "a"}
    with caplog.at_level("ERROR", logger="trellis.inbox"):
        response = await http.post(f"/inbox/runs/{paused.run_id}/resume", json=answer)
        assert response.status_code == 202
        for _ in range(100):
            if "failed to continue" in caplog.text:
                break
            await asyncio.sleep(0.01)
    assert "the store went away" in caplog.text


async def test_without_an_identity_every_answer_is_anonymous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with Harness(config=Settings()) as harness:
        agent = harness.wrap(refunds, id="refunds", tools=[refund])
        app = FastAPI()
        with caplog.at_level("WARNING", logger="trellis.inbox"):
            harness.serve_inbox(app, path="/review")
        assert "every answer is 'anonymous'" in caplog.text
        paused = await agent.run("go", user="ada")
        assert paused.interrupt is not None
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://inbox") as http:
            answer = {"interrupt_id": paused.run_id, "decision": "cancel"}
            response = await http.post(f"/review/runs/{paused.run_id}/resume", json=answer)
            assert response.status_code == 404  # an interrupt's own id, not the run's
            answer["interrupt_id"] = paused.interrupt.interrupt_id
            response = await http.post(f"/review/runs/{paused.run_id}/resume", json=answer)
        assert response.status_code == 202
        assert await settled(harness, paused.run_id, RunStatus.CANCELLED) is RunStatus.CANCELLED
        record = await harness.runs.get(paused.run_id)
        assert record is not None and record.last_resolution is not None
        assert record.last_resolution.reviewer == "anonymous"
