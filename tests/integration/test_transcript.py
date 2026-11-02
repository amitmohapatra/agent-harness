"""The transcript of any run or sub-agent run (``GET {path}/runs/{id}/transcript``): its
messages and tool calls, each call with its arguments and its result, read from the run's log
(agent-runs' event log), a page at a time; for the run's own user only."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.integration.test_background import Gate, ended, researcher
from tests.integration.test_inbox import working
from tests.support.arm import arm
from tests.support.models import ScriptedChat
from tests.support.runlog import LoggedRuns
from trellis import Harness, Runtime, Settings, background
from trellis.harness.transcript import folded

PATH = "/agui"


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


class Deployment:
    def __init__(self, calls: int = 1) -> None:
        self.store = LoggedRuns()
        self.harness = Harness(config=Settings(), runs=self.store)
        self.gate = Gate()
        self.gate.released.set()
        plan: list[Any] = [
            [("hold", {"topic": f"t{k}"}) for k in range(calls)],  # at once, in one step
            ("hold", {"topic": "last"}),
            "tides",
        ]
        self.child = researcher(self.harness, self.gate, model=ScriptedChat(plan))

        async def lead(input: Any, agent: Runtime) -> Any:
            return await agent.tools.call(
                "start_async_task", description=str(input), subagent_type="research"
            )

        self.lead = arm(self.harness.wrap(lead, id="lead"), [background(self.child)])
        app = FastAPI()
        self.lead.serve_chat(app, identity=user_of)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
        )

    async def task(self) -> str:
        async with working(self.harness, self.child):
            said = await self.lead.run("the tides", user="u1", thread="t1")
            run_id = str(said.answer).rsplit("task_id: ", 1)[1].strip()
            await ended(self.harness, run_id)
        return run_id


@pytest.fixture
async def deployed() -> AsyncIterator[Deployment]:
    d = Deployment()
    try:
        yield d
    finally:
        await d.client.aclose()
        await d.harness.aclose()


async def test_a_sub_agents_transcript_has_its_messages_and_calls(deployed: Deployment) -> None:
    task = await deployed.task()
    got = await deployed.client.get(f"{PATH}/runs/{task}/transcript")
    assert got.status_code == 200, got.text
    body = got.json()
    assert (body["run_id"], body["agent_id"], body["status"]) == (task, "research", "SUCCESS")
    assert body["input"] == "the tides" and body["more"] is False
    calls = [i for i in body["items"] if i["type"] == "tool_call"]
    assert [(c["tool"], c["args"], c["result"]) for c in calls] == [
        ("hold", {"topic": "t0"}, "facts about t0"),
        ("hold", {"topic": "last"}, "facts about last"),
    ]
    said = [i["text"] for i in body["items"] if i["type"] == "message"]
    assert said[-1] == "tides"
    assert body["items"][-1] == {"type": "end", "outcome": "success"}


async def test_nobody_else_reads_a_transcript(deployed: Deployment) -> None:
    task = await deployed.task()
    other = await deployed.client.get(f"{PATH}/runs/{task}/transcript", headers={"x-user": "u2"})
    assert other.status_code == 404
    assert (await deployed.client.get(f"{PATH}/runs/run-nope/transcript")).status_code == 404


async def test_a_large_transcript_comes_a_page_at_a_time() -> None:
    d = Deployment(calls=60)
    try:
        task = await d.task()
        pages, after, more = [], 0, True
        while more:
            got = (
                await d.client.get(
                    f"{PATH}/runs/{task}/transcript", params={"after": after, "limit": 50}
                )
            ).json()
            pages.append(got["items"])
            after, more = got["after"], got["more"]
        assert len(pages) > 3
        whole = folded(d.store.logged(task))
        merged: dict[str, dict[str, Any]] = {}
        for item in (i for page in pages for i in page):
            merged.setdefault(item.get("id") or str(len(merged)), {}).update(item)
        calls = [i for i in merged.values() if i["type"] == "tool_call"]
        assert len(calls) == 61 == len([i for i in whole if i["type"] == "tool_call"])
        assert all("args" in c and "result" in c for c in calls)
    finally:
        await d.client.aclose()
        await d.harness.aclose()


async def test_without_a_run_log_it_says_so() -> None:
    async with Harness(config=Settings()) as h:

        async def fn(input: Any, agent: Runtime) -> str:
            return "done"

        agent = h.wrap(fn, id="fn")
        result = await agent.run("x", user="u1")
        app = FastAPI()
        agent.serve_chat(app, identity=user_of)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
        ) as client:
            got = await client.get(f"{PATH}/runs/{result.run_id}/transcript")
    assert got.status_code == 404 and "run log" in got.json()["detail"]


async def test_a_page_out_of_bounds_is_refused(deployed: Deployment) -> None:
    for params in ({"after": -1}, {"limit": 0}, {"limit": 501}):
        got = await deployed.client.get(f"{PATH}/runs/r/transcript", params=params)
        assert got.status_code == 422, params
