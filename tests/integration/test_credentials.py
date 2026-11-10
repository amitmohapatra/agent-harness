"""``Harness(credentials=)``: the run's own headers on each call of an ``openapi(...)``
operation, an ``a2a(...)`` agent and an ``mcp(...)`` server over HTTP — per user and target,
never mixed, asked again each run, a missing connection asked for, a failing hook failing the
call, the token nowhere but on the request."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from tests.integration.test_a2a_server import URL as A2A_URL
from tests.integration.test_a2a_server import asgi, serve
from tests.integration.test_mcp_servers import SERVER, free_port
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from trellis import Harness, Runtime, Settings, a2a, mcp, openapi
from trellis.contracts import RunStatus
from trellis.harness.a2a import client as a2a_client
from trellis.harness.credentials import NeedsAuth, Target
from trellis.harness.tools import mcp as mcp_module
from trellis.harness.tools.convert import text_of

DOC = {
    "openapi": "3.1.0",
    "paths": {
        "/orders/{id}": {
            "get": {
                "operationId": "get_order",
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "string"}}],
            }
        }
    },
}


def erp(host: str = "erp.test", **kwargs: Any) -> Any:
    return openapi({**DOC, "servers": [{"url": f"http://{host}"}]}, **kwargs)


async def order(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("get_order", id="7")


def served(seen: list[httpx.Request]) -> respx.Route:
    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": 7})

    return respx.get(url__regex=r"http://\w+\.test/orders/7").mock(side_effect=answer)


@respx.mock
async def test_without_a_hook_or_with_one_answering_none_the_fixed_headers_go() -> None:
    seen: list[httpx.Request] = []
    served(seen)

    async def nothing(run: Runtime, target: Target) -> None:
        return None

    for hook in (None, nothing):
        async with Harness(config=Settings(), credentials=hook) as h:
            agent = arm(h.wrap(order, id="desk"), [erp(headers={"X-Key": "fixed"})])
            assert (await agent.run("go", user="ada")).answer == {"id": 7}
    assert [(r.headers["x-key"], "authorization" in r.headers) for r in seen] == [
        ("fixed", False),
        ("fixed", False),
    ]


@respx.mock
async def test_each_user_and_target_gets_its_own_and_each_run_asks_again() -> None:
    """Two sources, two users: nothing mixed; a second run (a scheduled one) asks the hook
    again, so a refreshed token is the one sent; within a run, once per target."""
    seen: list[httpx.Request] = []
    served(seen)
    asked: list[tuple[str, str, str]] = []

    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        asked.append((run.user, target.name, target.kind))
        return {"Authorization": f"Bearer {target.name}-{run.user}-{len(asked)}"}

    async def both(input: str, agent: Runtime) -> Any:
        return [await agent.tools.call(n, id="7") for n in ("get_order", "crm_get_order")]

    crm = openapi(
        {**DOC, "servers": [{"url": "http://crm.test"}], "paths": {"/orders/{id}": {"get": {
            **DOC["paths"]["/orders/{id}"]["get"], "operationId": "crm_get_order"}}}},
        headers={"Authorization": "Bearer fixed"},
    )  # fmt: skip
    async with Harness(config=Settings(), credentials=hook) as h:
        agent = arm(h.wrap(both, id="desk"), [erp(), crm])
        for user in ("ada", "bob", "ada"):
            assert (await agent.run("go", user=user)).status is RunStatus.SUCCESS
            await agent.run("go", user=user)
    tokens = [r.headers["authorization"] for r in seen]
    assert all(t.split("-")[1] == u for t, u in zip(tokens, ["ada"] * 4 + ["bob"] * 4 + ["ada"] * 4, strict=True))  # fmt: skip
    assert [t.split(" ")[1].split("-")[0] for t in tokens[:2]] == ["erp.test", "crm.test"]
    assert len(set(tokens)) == len(tokens)  # every run its own, the fixed one overridden
    assert {k for _, _, k in asked} == {"openapi"} and len(asked) == len(tokens)


@respx.mock
async def test_a_connection_asked_for_pauses_then_the_call_runs_once() -> None:
    seen: list[httpx.Request] = []
    served(seen)
    calls: list[str] = []

    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        calls.append(run.user)
        if len(calls) <= 2:  # the resumed attempt's first ask still finds it unconnected
            raise NeedsAuth("https://auth.test/connect?app=erp")
        return {"Authorization": "Bearer t-ada"}

    async with Harness(config=Settings(), credentials=hook) as h:
        agent = arm(h.wrap(order, id="desk"), [erp()])
        paused = await agent.run("go", user="ada")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert (
            paused.interrupt.question == "Connect your erp.test: https://auth.test/connect?app=erp"
        )
        assert seen == []
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="connected", reviewer="ada"
        )
        assert done.status is RunStatus.SUCCESS and done.answer == {"id": 7}
    assert [r.headers["authorization"] for r in seen] == ["Bearer t-ada"]
    assert len(calls) == 3  # asked, then on the answer once more, and again: no more pauses


@respx.mock
async def test_a_failing_hook_fails_the_call_without_its_words_or_another_token() -> None:
    seen: list[httpx.Request] = []
    served(seen)

    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        raise RuntimeError("vault said: secret-123")

    async def tried(input: str, agent: Runtime) -> Any:
        return text_of(await agent.tools.call("get_order", id="7"))

    async with Harness(config=Settings(), credentials=hook) as h:
        agent = arm(h.wrap(tried, id="desk"), [erp(headers={"Authorization": "Bearer fixed"})])
        said = (await agent.run("go", user="ada")).answer
    assert "no credentials for erp.test (RuntimeError)" in said and "secret" not in said
    assert seen == []


@respx.mock
async def test_the_token_is_on_the_request_only(spans: Any) -> None:
    seen: list[httpx.Request] = []
    served(seen)
    token = "tok-only-on-the-wire"
    service = FakeMemoryService()

    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async with Harness(config=Settings(), memory=service.client(), credentials=hook) as h:
        agent = arm(h.wrap(order, id="desk"), [erp()])
        events = [e async for e in agent.stream("go", user="ada")]
        await h.writes.drain()
        [record] = [r async for r in h.runs.iterate()]
        full = await h.runs.get(record.run_id)
    assert seen[0].headers["authorization"] == f"Bearer {token}"
    kept = [
        json.dumps([e.model_dump(mode="json") for e in events]),
        json.dumps(full.model_dump(mode="json") if full else None),
        json.dumps([c.body for c in service.calls], default=str),
        json.dumps([dict(s.attributes or {}) for s in spans.get_finished_spans()], default=str),
    ]
    assert all(token not in text for text in kept)


@pytest.mark.usefixtures("writes_unasked")  # the headers, not the call's approval
async def test_an_a2a_agent_gets_the_runs_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    app, remote, _ = serve()
    sent: list[httpx.Request] = []

    def spied(timeout: float) -> httpx.AsyncClient:
        client = asgi(app)

        async def spy(request: httpx.Request) -> None:
            sent.append(request)

        client.event_hooks["request"].append(spy)
        return client

    monkeypatch.setattr(a2a_client, "_http", spied)

    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        return {"Authorization": f"Bearer {target.kind}-{target.name}-{run.user}"}

    async def delegate(input: Any, agent: Runtime) -> Any:
        return await agent.tools.call("greeter", message=input)

    async with Harness(config=Settings(), credentials=hook) as h:
        agent = arm(h.wrap(delegate, id="caller"), [a2a(A2A_URL)])
        assert (await agent.run("world", user="ada")).answer == "hello world"
    calls = [r for r in sent if r.method == "POST"]
    assert calls and all(r.headers["authorization"] == "Bearer a2a-greeter-ada" for r in calls)
    await remote.aclose()


@pytest.fixture(scope="module")
def two_tokens() -> Any:
    import os
    import socket
    import subprocess
    import sys
    import time

    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)],
        env={**os.environ, "TOKEN": "t-ada,t-bob,shared"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait(timeout=10)


async def whoami(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("box-whoami")


async def test_mcp_sessions_are_per_user_with_the_hook_and_shared_without(
    two_tokens: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hook(run: Runtime, target: Target) -> dict[str, str]:
        assert (target.kind, target.url) == ("mcp", two_tokens)
        return {"Authorization": f"Bearer t-{run.user}"}

    box = mcp(two_tokens, name="box")
    async with Harness(config=Settings(), credentials=hook) as h:
        agent = arm(h.wrap(whoami, id="desk"), [box])
        said = [(await agent.run("go", user=u)).answer for u in ("ada", "bob", "ada")]
        assert said == ["t-ada", "t-bob", "t-ada"]
        assert sorted(user for _, user, _ in box._sessions) == ["ada", "bob"]
        assert box._connection._task is None  # the shared one never opened
        monkeypatch.setattr(mcp_module, "IDLE_SECONDS", -1.0)  # idle: closed at the next call
        assert (await agent.run("go", user="bob")).answer == "t-bob"
        assert [user for _, user, _ in box._sessions] == ["bob"]
        await box.aclose()
        assert box._sessions == {}

    shared = mcp(two_tokens, name="box", headers={"Authorization": "Bearer shared"})
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(whoami, id="desk"), [shared])
        said = [(await agent.run("go", user=u)).answer for u in ("ada", "bob")]
        assert said == ["shared", "shared"] and shared._sessions == {}
        await shared.aclose()
