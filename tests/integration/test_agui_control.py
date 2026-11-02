"""The AG-UI surface's control of a run and its background tasks: a new instruction, a pause
and a resume (``POST {path}/runs/{id}/messages``, ``/pause``, ``/resume``), for the run's own
user only; a target with no step to take them at says so."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.integration.test_background import WAIT, Gate, ended, entered, researcher
from tests.integration.test_inbox import status, stepper, working
from tests.support.arm import arm
from trellis import Harness, Runtime, Settings, background
from trellis.contracts import InterruptReason, RunStatus

PATH = "/agui"


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


class Deployment:
    def __init__(self) -> None:
        self.harness = Harness(config=Settings())
        self.gate = Gate()
        self.work = stepper(self.gate, 2, "the tides")
        self.child = researcher(self.harness, self.gate, model=self.work)

        async def lead(input: Any, agent: Runtime) -> Any:
            if input == "status":
                return await agent.tools.call("list_async_tasks", status_filter="all")
            return await agent.tools.call(
                "start_async_task", description=str(input), subagent_type="research"
            )

        self.lead = arm(self.harness.wrap(lead, id="lead"), [background(self.child)])
        app = FastAPI()
        self.lead.serve_chat(app, identity=user_of)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://agui",
            headers={"x-user": "u1"},
        )

    async def task(self) -> str:
        said = await self.lead.run("the tides", user="u1", thread="t1")
        return str(said.answer).rsplit("task_id: ", 1)[1].strip()


@pytest.fixture
async def deployed() -> AsyncIterator[Deployment]:
    d = Deployment()
    try:
        yield d
    finally:
        await d.client.aclose()
        await d.harness.aclose()


async def test_the_user_pauses_instructs_and_resumes_their_task(deployed: Deployment) -> None:
    d, client = deployed, deployed.client
    async with working(d.harness, d.child):
        task = await d.task()
        await entered(d.gate)
        paused = await client.post(f"{PATH}/runs/{task}/pause")
        assert paused.status_code == 200, paused.text
        d.gate.released.set()
        record = await status(d.harness, task, RunStatus.PAUSED)
        assert record.awaiting is not None and record.awaiting.reason is InterruptReason.PAUSE
        told = await client.post(f"{PATH}/runs/{task}/messages", json={"text": "and the north"})
        assert told.status_code == 200 and told.json()["position"] == 2, told.text
        again = await client.post(f"{PATH}/runs/{task}/pause")
        assert again.status_code == 409  # it already waits
        resumed = await client.post(f"{PATH}/runs/{task}/resume")
        assert resumed.status_code == 200, resumed.text
        record = await ended(d.harness, task)
    assert record.status is RunStatus.SUCCESS
    assert "and the north" in str(d.work.requests[-2]["messages"])


async def test_nobody_else_reaches_a_users_task(deployed: Deployment) -> None:
    d, client = deployed, deployed.client
    async with working(d.harness, d.child):
        task = await d.task()
        await entered(d.gate)
        stranger = {"x-user": "u2"}
        for route, body in (("messages", {"text": "stop"}), ("pause", None), ("resume", None)):
            refused = await client.post(f"{PATH}/runs/{task}/{route}", json=body, headers=stranger)
            assert refused.status_code == 404, (route, refused.text)
        assert await d.harness.runs.messages(task) == []
        missing = await client.post(f"{PATH}/runs/run-nope/pause")
        assert missing.status_code == 404
        early = await client.post(f"{PATH}/runs/{task}/resume")
        assert early.status_code == 409  # not paused
        d.gate.released.set()


async def test_a_target_with_no_step_says_it_takes_no_instruction(deployed: Deployment) -> None:
    d = deployed
    held = asyncio.Event()

    async def slow(input: Any, agent: Runtime) -> str:
        await held.wait()
        return "done"

    fn = d.harness.wrap(slow, id="slow")
    run = asyncio.create_task(fn.run("x", user="u1"))
    try:
        async with asyncio.timeout(WAIT):
            while True:
                if fn.running:
                    break
                await asyncio.sleep(0.01)
        [run_id] = fn.running
        app = FastAPI()
        fn.serve_chat(app, identity=user_of)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
        ) as own:
            for route, body in (("messages", {"text": "faster"}), ("pause", None)):
                refused = await own.post(f"{PATH}/runs/{run_id}/{route}", json=body)
                assert refused.status_code == 409, refused.text
                assert "no step" in refused.json()["detail"]
    finally:
        held.set()
        await run


async def test_the_status_view_is_one_source_for_the_ui_and_the_chat(
    deployed: Deployment,
) -> None:
    """``GET {path}/tasks`` lists the user's tasks with their counts, read as the chat's
    ``list_async_tasks`` reads them: the same records, the same shape."""
    d, client = deployed, deployed.client
    async with working(d.harness, d.child):
        task = await d.task()
        d.gate.released.set()
        await ended(d.harness, task)
    view = await client.get(f"{PATH}/tasks")
    assert view.status_code == 200, view.text
    [shown] = view.json()["tasks"]
    assert (shown["task_id"], shown["status"]) == (task, "success")
    usage = shown["usage"]
    assert (usage["model_calls"], usage["tool_calls"], usage["commands"]) == (3, 2, 0), usage
    assert shown["elapsed_seconds"] >= 0
    said = await d.lead.run("status", user="u1", thread="t1")
    assert json.loads(str(said.answer)) == [shown]
    assert (await client.get(f"{PATH}/tasks", headers={"x-user": "u2"})).json() == {"tasks": []}
