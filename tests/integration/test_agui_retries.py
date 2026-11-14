"""AG-UI requests a client sends twice or sends empty (found by the scenario suite):

* SCN-9: the same ``runId`` posted twice at once (a client's retry racing its first try) starts
  one run — the second is a ``422``, as a ``runId`` already used is — never the same run twice;
* SCN-10: a user message with no text at all (``content: ""``) opens no run: a ``422``, as a
  blank one is, rather than falling back to the request's ``state`` and failing inside the
  framework."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI, Request

from trellis import Harness, Runtime

ran: list[str] = []


async def echo(input: Any, agent: Runtime) -> str:
    ran.append(str(input))
    await asyncio.sleep(0.05)
    return f"you said {input}"


def _client(harness: Harness, agent_id: str) -> httpx.AsyncClient:
    app = FastAPI()

    def user_of(request: Request) -> str:
        return "ada"

    harness.wrap(echo, id=agent_id).serve_chat(app, identity=user_of)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agui")


async def test_scn9_the_same_run_id_posted_twice_at_once_is_one_run(harness: Harness) -> None:
    start = harness.runs.start

    async def slow(record: Any, *args: Any, **kwargs: Any) -> Any:  # a store over a network
        await asyncio.sleep(0.05)
        return await start(record, *args, **kwargs)

    harness.runs.start = slow  # type: ignore[method-assign]
    ran.clear()
    body = {
        "threadId": "t1",
        "runId": "run_retry1",
        "messages": [{"id": "m1", "role": "user", "content": "hello"}],
    }
    async with _client(harness, "scn9") as client:
        one, two = await asyncio.gather(
            client.post("/agui/run", json=body), client.post("/agui/run", json=body)
        )
    assert sorted([one.status_code, two.status_code]) == [200, 422], (
        one.text[:200],
        two.text[:200],
    )
    assert ran == ["hello"], ran  # the message ran once


async def test_scn10_an_empty_message_opens_no_run(harness: Harness) -> None:
    ran.clear()
    body = {"threadId": "t1", "messages": [{"id": "m1", "role": "user", "content": ""}]}
    async with _client(harness, "scn10") as client:
        sent = await client.post("/agui/run", json=body)
    assert sent.status_code == 422, sent.text[:300]
    assert ran == []
