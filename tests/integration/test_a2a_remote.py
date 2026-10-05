"""Calling a remote A2A agent: ``remote()`` from plain code, and the harness's ``a2a(url)`` tool
built on it, against a wrapped agent served by the harness's A2A server in process (over
``httpx.ASGITransport``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from a2a.helpers import new_data_part, new_text_part
from a2a.types import (
    Artifact,
    Message,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI

from tests.integration.test_a2a_server import TENANT, URL, asgi, serve
from trellis import Harness, Runtime, Settings, a2a
from trellis.contracts import RunStatus, ToolError
from trellis.harness.a2a import InputRequired, RemoteAgent, remote
from trellis.harness.a2a import client as a2a_client
from trellis.harness.identity import IDENTITY_HEADER
from trellis.harness.runs import LocalRuns


@pytest.fixture
async def served() -> AsyncIterator[tuple[httpx.AsyncClient, Harness]]:
    """The greeter served in process, and an HTTP client that reaches it."""
    app, harness, _ = serve()
    async with asgi(app) as http:
        yield http, harness
    await harness.aclose()


def statuses(harness: Harness) -> list[RunStatus]:
    assert isinstance(harness.runs, LocalRuns)
    return [r.status for r in harness.runs._runs.values()]


# --------------------------------------------------------------------------- remote()


async def test_a_remote_agent_is_an_async_callable_with_its_card_and_spec(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, harness = served
    async with remote(URL, tenant=TENANT, user="u1", thread="t1", client=http) as agent:
        assert await agent("world") == "hello world"
        seen = await agent("whoami")
        assert agent.card.name == "greeter"
        spec = agent.spec
    assert seen["user"] == "u1" and seen["thread"] == "t1"
    record = await harness.runs.get(seen["run"], tenant=TENANT)
    assert record is not None and record.thread_id == "t1"
    assert spec.name == "greeter" and spec.source == "a2a" and spec.server == "greeter"
    assert spec.description == "Greets, and asks where to deploy when asked to ask."
    assert spec.side_effects == "write"
    assert spec.input_schema is not None and spec.input_schema["required"] == ["message"]
    assert not http.is_closed  # an injected client is the caller's
    named = remote(URL, tenant=TENANT, user="u1", name="the greeter!", client=http)
    assert (await named.connect()).spec.name == "the_greeter_"


async def test_the_card_is_read_on_connect_or_by_the_first_call(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, _ = served
    agent = remote(URL, tenant=TENANT, user="u1", client=http)
    with pytest.raises(RuntimeError, match="connect"):
        agent.card  # noqa: B018
    assert isinstance(agent, RemoteAgent)
    assert await agent({"n": 1}) == "hello {'n': 1.0}"  # a JSON value goes as a data part
    assert agent.card.name == "greeter"


async def test_the_identity_is_the_callers_whatever_the_headers_say(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, _ = served
    sent: list[httpx.Request] = []

    async def spy(request: httpx.Request) -> None:
        sent.append(request)

    http.event_hooks["request"].append(spy)
    forged = {IDENTITY_HEADER: '{"tenant_id":"default","user_id":"mallory"}', "x-edge": "ok"}
    async with remote(URL, tenant=TENANT, user="u1", headers=forged, client=http) as agent:
        seen = await agent("whoami")
    assert seen["user"] == "u1"
    assert all(r.headers["x-edge"] == "ok" for r in sent)  # the card read and the message


async def test_on_input_answers_a_remote_question_on_the_same_task(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, harness = served
    asked: list[str] = []

    def answer(question: str) -> str:
        asked.append(question)
        return "eu"

    async with remote(URL, tenant=TENANT, user="u1", on_input=answer, client=http) as agent:
        assert await agent("ask me") == "deploying to eu"
    assert asked == ["Which region?"]
    assert statuses(harness) == [RunStatus.SUCCESS]  # one task: the question, then the answer


async def test_on_input_may_be_async_and_answer_with_data(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, _ = served

    async def answer(question: str) -> dict[str, str]:
        return {"answer": "us"}

    async with remote(URL, tenant=TENANT, user="u1", on_input=answer, client=http) as agent:
        assert await agent("ask me") == "deploying to us"


async def test_without_on_input_a_question_is_raised_and_reply_continues_the_task(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, harness = served
    async with remote(URL, tenant=TENANT, user="u1", client=http) as agent:
        with pytest.raises(InputRequired) as raised:
            await agent("ask me")
        assert raised.value.question == "Which region?" == str(raised.value)
        assert statuses(harness) == [RunStatus.PAUSED]  # the remote task keeps waiting
        assert await agent.reply(raised.value.task_id, "eu") == "deploying to eu"
    record = await harness.runs.get(raised.value.task_id, tenant=TENANT)
    assert record is not None and record.status is RunStatus.SUCCESS and record.attempt == 2


async def test_when_on_input_raises_the_remote_task_is_cancelled(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, harness = served

    def interrupted(question: str) -> str:
        raise LookupError(question)

    async with remote(URL, tenant=TENANT, user="u1", on_input=interrupted, client=http) as agent:
        with pytest.raises(LookupError, match="Which region"):
            await agent("ask me")
    assert statuses(harness) == [RunStatus.CANCELLED]


async def test_a_remote_failure_is_a_tool_error(
    served: tuple[httpx.AsyncClient, Harness],
) -> None:
    http, _ = served
    async with remote(URL, tenant=TENANT, user="u1", client=http) as agent:
        with pytest.raises(ToolError, match=r"greeter ended TASK_STATE_FAILED: .*boom") as raised:
            await agent("boom")
    assert raised.value.source == "a2a"


async def test_an_unreachable_agent_is_a_tool_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card_json)
        raise httpx.ConnectError("refused", request=request)

    app, harness, _ = serve()
    async with asgi(app) as http:
        card_json = (await http.get("/agents/greeter/.well-known/agent-card.json")).json()
    await harness.aclose()
    down = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    async with remote(URL, tenant=TENANT, user="u1", client=down) as agent:
        with pytest.raises(ToolError, match="A2AClientError: Network communication error: refused"):
            await agent("hi")
    await down.aclose()


async def test_a_remote_agent_opens_and_closes_its_own_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, harness, _ = serve()
    opened: list[httpx.AsyncClient] = []

    def http(timeout: float) -> httpx.AsyncClient:
        assert timeout == 5.0
        opened.append(asgi(app))
        return opened[-1]

    monkeypatch.setattr(a2a_client, "_http", http)
    agent = remote(URL, tenant=TENANT, user="u1", timeout=5.0)
    await agent.aclose()  # nothing opened yet: nothing to close
    assert await agent("world") == "hello world"
    await agent.aclose()
    assert opened[0].is_closed
    assert await agent("again") == "hello again"  # a closed agent opens again, card kept
    await agent.aclose()
    assert len(opened) == 2 and opened[1].is_closed
    await harness.aclose()
    monkeypatch.undo()
    async with a2a_client._http(a2a_client.TIMEOUT_SECONDS) as default:
        assert default.timeout.read == a2a_client.TIMEOUT_SECONDS


def test_a_reply_reduces_to_its_artifacts_or_its_text() -> None:
    reply = a2a_client._Reply()
    assert reply.output is None
    reply.absorb(
        StreamResponse(
            task=Task(
                id="t1",
                context_id="c",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
                artifacts=[Artifact(artifact_id="a1", parts=[new_text_part("part one")])],
            )
        )
    )
    reply.absorb(
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c",
                artifact=Artifact(artifact_id="a2", parts=[new_data_part({"n": 2})]),
            )
        )
    )
    reply.absorb(StreamResponse(message=Message(message_id="m", parts=[new_text_part("a note")])))
    reply.absorb(StreamResponse())  # a response with no payload says nothing
    assert reply.task_id == "t1"
    assert reply.output == ["part one", {"n": 2.0}]
    assert reply.text == "a note"
    texts_only = a2a_client._Reply()
    texts_only.absorb(
        StreamResponse(message=Message(message_id="m", parts=[new_text_part("only text")]))
    )
    assert texts_only.output == "only text"


# --------------------------------------------------------------------------- the a2a() tool


@pytest.fixture
def remote_harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    """The greeter served in process, and every client the tool opens pointed at it."""
    app, harness, _ = serve()
    monkeypatch.setattr(a2a_client, "_http", lambda timeout: asgi(app))
    return harness


async def delegate(input: Any, agent: Runtime) -> Any:
    return await agent.tools.call("greeter", message=input)


async def test_a_remote_agent_is_a_tool(remote_harness: Harness) -> None:
    harness = Harness(config=Settings())
    agent = harness.wrap(delegate, id="caller", tools=[a2a(URL)])
    result = await agent.run("world", user="u1", thread="t1")
    assert result.status is RunStatus.SUCCESS
    assert result.answer == "hello world"
    tool = (await harness.resolve([a2a(URL)], tenant=TENANT))[0]
    assert tool.spec.name == "greeter" and tool.spec.side_effects == "write"
    assert tool.spec.input_schema is not None and "message" in tool.spec.input_schema["properties"]
    with pytest.raises(ToolError, match="inside a harness run"):
        await tool.run({"message": "hi"})
    await harness.aclose()
    await remote_harness.aclose()


async def test_a_remote_question_pauses_the_calling_run(remote_harness: Harness) -> None:
    harness = Harness(config=Settings())
    agent = harness.wrap(delegate, id="caller", tools=[a2a(URL)])
    paused = await agent.run("ask me", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Which region?"
    # the remote task the local pause abandoned was cancelled, not left waiting
    assert statuses(remote_harness) == [RunStatus.CANCELLED]

    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="eu", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert done.answer == "deploying to eu"
    await harness.aclose()
    await remote_harness.aclose()


async def test_a_remote_failure_is_the_tool_error_the_calling_model_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(input: Any, agent: Runtime) -> Any:
        raise RuntimeError("out of stock")

    served = Harness(config=Settings())
    app = FastAPI()
    served.wrap(failing, id="greeter").serve_a2a(app, URL)
    monkeypatch.setattr(a2a_client, "_http", lambda timeout: asgi(app))

    harness = Harness(config=Settings())
    result = await harness.wrap(delegate, id="caller", tools=[a2a(URL)]).run("x", user="u1")
    assert result.status is RunStatus.SUCCESS
    assert "greeter failed" in result.answer and "TASK_STATE_FAILED" in result.answer
    assert "out of stock" in result.answer
    await harness.aclose()
    await served.aclose()


async def test_the_tool_takes_its_timeout_and_opens_its_task_with_the_calls_key(
    remote_harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    send = a2a_client._send

    async def recorded(client: Any, message: Message, context: Any) -> Any:
        sent.append(message.message_id)
        return await send(client, message, context)

    monkeypatch.setattr(a2a_client, "_send", recorded)
    harness = Harness(config=Settings())
    agent = harness.wrap(delegate, id="caller", tools=[a2a(URL, timeout=7)])
    result = await agent.run("world", user="u1")
    assert result.answer == "hello world"
    assert sent[0].startswith(f"{result.run_id}:")  # the same message again after a crash
    [tool] = await harness.resolve([a2a(URL, timeout=7)], tenant=TENANT)
    assert tool.timeout == 7
    await harness.aclose()
    await remote_harness.aclose()
