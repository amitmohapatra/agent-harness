"""The A2A server against the real a2a-sdk client, over ``httpx.ASGITransport``: the card, a
run streamed to completion, a pause answered by the next message, cancellation and signed push
notifications. Calling a remote agent is ``test_a2a_remote``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskPushNotificationConfig,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import A2AError
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict

from trellis import Harness, Settings
from trellis.contracts import RunStatus
from trellis.harness.a2a.identity import EXTENSION_URI, identity_headers
from trellis.harness.a2a.push import PushNotifier, TargetRefused, validate_url
from trellis.harness.agent import Agent
from trellis.harness.runtime import Runtime
from trellis.runs.webhooks import SIGNATURE_HEADER, verify_signature

URL = "http://a2a.test/agents/greeter"
TENANT = "default"


async def greeter(input: Any, agent: Runtime) -> Any:
    """Greets, and asks where to deploy when asked to ask."""
    if input == "whoami":
        return {"user": agent.user, "thread": agent.thread, "run": agent.run_id}
    if input == "boom":
        raise RuntimeError("boom")
    if isinstance(input, str) and input.startswith("ask"):
        region = await agent.ask("Which region?", expects={"type": "string"})
        return f"deploying to {region}"
    return f"hello {input}"


def serve() -> tuple[FastAPI, Harness, Agent]:
    harness = Harness(config=Settings())
    agent = harness.wrap(greeter, id="greeter")
    app = FastAPI()
    agent.serve_a2a(app, URL)
    return app, harness, agent


def asgi(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2a.test")


async def connect(http: httpx.AsyncClient) -> Client:
    card = await A2ACardResolver(http, URL).get_agent_card()
    return ClientFactory(ClientConfig(httpx_client=http)).create(card)


def caller(user: str = "u1") -> ClientCallContext:
    headers = {HTTP_EXTENSION_HEADER: EXTENSION_URI, **identity_headers(TENANT, user)}
    return ClientCallContext(service_parameters=headers)


async def send(
    client: Client,
    text: str = "",
    *,
    task_id: str = "",
    context_id: str | None = None,
    data: dict[str, Any] | None = None,
    user: str = "u1",
) -> list[StreamResponse]:
    parts = [new_text_part(text)] if text else []
    if data is not None:
        parts.append(new_data_part(data))
    message = new_message(parts, context_id=context_id, role=Role.ROLE_USER)
    if task_id:
        message.task_id = task_id
    request = SendMessageRequest(message=message)
    return [r async for r in client.send_message(request, context=caller(user))]


def states(responses: Sequence[StreamResponse]) -> list[Any]:
    found = []
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "task":
            found.append(response.task.status.state)
        elif which == "status_update":
            found.append(response.status_update.status.state)
    return found


def task_id_of(responses: Sequence[StreamResponse]) -> str:
    for response in responses:
        if response.WhichOneof("payload") == "task":
            return response.task.id
        if response.WhichOneof("payload") == "status_update":
            return response.status_update.task_id
    raise AssertionError("no task in the stream")


def artifacts(responses: Sequence[StreamResponse]) -> list[Any]:
    found: list[Any] = []
    for response in responses:
        if response.WhichOneof("payload") == "artifact_update":
            for part in response.artifact_update.artifact.parts:
                which = part.WhichOneof("content")
                found.append(part.text if which == "text" else MessageToDict(part.data))
    return found


def status_texts(responses: Sequence[StreamResponse]) -> list[str]:
    return [
        p.text
        for r in responses
        if r.WhichOneof("payload") == "status_update"
        for p in r.status_update.status.message.parts
        if p.WhichOneof("content") == "text"
    ]


@pytest.fixture
async def served() -> AsyncIterator[tuple[Client, Harness]]:
    app, harness, _ = serve()
    async with asgi(app) as http:
        yield await connect(http), harness
    await harness.aclose()


# --------------------------------------------------------------------------- serving


async def test_the_card_describes_the_agent_where_the_url_says() -> None:
    app, harness, _ = serve()
    async with asgi(app) as http:
        response = await http.get("/agents/greeter/.well-known/agent-card.json")
    card = response.json()
    assert card["name"] == "greeter"
    assert card["description"] == "Greets, and asks where to deploy when asked to ask."
    assert card["supportedInterfaces"][0]["url"] == URL
    assert card["capabilities"]["streaming"] and card["capabilities"]["pushNotifications"]
    assert card["capabilities"]["extensions"][0]["uri"] == EXTENSION_URI
    await harness.aclose()


async def test_a_run_streams_to_completion_and_the_task_is_the_run(
    served: tuple[Client, Harness],
) -> None:
    client, harness = served
    responses = await send(client, "world")
    assert states(responses)[0] == TaskState.TASK_STATE_SUBMITTED
    assert TaskState.TASK_STATE_WORKING in states(responses)
    assert states(responses)[-1] == TaskState.TASK_STATE_COMPLETED
    assert artifacts(responses) == ["hello world"]
    record = await harness.runs.get(task_id_of(responses))
    assert record is not None and record.status is RunStatus.SUCCESS and record.user_id == "u1"


async def test_identity_comes_from_the_header_and_the_thread_is_the_context(
    served: tuple[Client, Harness],
) -> None:
    client, _ = served
    responses = await send(client, "whoami", user="u7")
    seen = artifacts(responses)[0]
    assert seen["user"] == "u7" and seen["run"] == task_id_of(responses)
    assert seen["thread"] == responses[0].task.context_id


async def test_a_failing_run_fails_the_task(served: tuple[Client, Harness]) -> None:
    client, _ = served
    responses = await send(client, "boom")
    assert states(responses)[-1] == TaskState.TASK_STATE_FAILED
    assert "boom" in " ".join(status_texts(responses))


async def test_a_pause_is_input_required_and_the_next_message_resumes_it(
    served: tuple[Client, Harness],
) -> None:
    client, harness = served
    paused = await send(client, "ask me")
    assert states(paused)[-1] == TaskState.TASK_STATE_INPUT_REQUIRED
    assert "Which region?" in status_texts(paused)
    task_id = task_id_of(paused)

    with pytest.raises(A2AError):  # another user's task does not exist for them
        await send(client, "eu", task_id=task_id, user="mallory")
    record = await harness.runs.get(task_id)
    assert record is not None and record.status is RunStatus.PAUSED

    resumed = await send(client, "eu", task_id=task_id)
    assert states(resumed)[-1] == TaskState.TASK_STATE_COMPLETED
    assert artifacts(resumed) == ["deploying to eu"]
    record = await harness.runs.get(task_id)
    assert record is not None and record.status is RunStatus.SUCCESS and record.attempt == 2


async def test_another_replica_rebuilds_a_paused_task_from_the_run_store() -> None:
    app, harness, agent = serve()
    replica = FastAPI()
    agent.serve_a2a(replica, URL)  # same runs, a server that never saw the task
    async with asgi(app) as http:
        paused = await send(await connect(http), "ask me")
    task_id, context_id = task_id_of(paused), paused[0].task.context_id
    async with asgi(replica) as http:
        client = await connect(http)
        task = await client.get_task(GetTaskRequest(id=task_id), context=caller())
        assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        assert task.status.message.parts[0].text == "Which region?"
        resumed = await send(client, "eu", task_id=task_id, context_id=context_id)
    assert artifacts(resumed) == ["deploying to eu"]
    await harness.aclose()


async def test_a_paused_task_can_be_cancelled(served: tuple[Client, Harness]) -> None:
    client, harness = served
    task_id = task_id_of(await send(client, "ask me"))
    cancelled = await client.cancel_task(CancelTaskRequest(id=task_id), context=caller())
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED
    record = await harness.runs.get(task_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_push_url_that_is_not_public_https_is_refused_at_registration(
    served: tuple[Client, Harness],
) -> None:
    client, _ = served
    task_id = task_id_of(await send(client, "ask me"))
    with pytest.raises(A2AError):
        await client.create_task_push_notification_config(
            TaskPushNotificationConfig(task_id=task_id, url="http://localhost/hook", token="t"),
            context=caller(),
        )


# --------------------------------------------------------------------------- push


def test_push_targets_are_public_https_only() -> None:
    assert validate_url("https://93.184.216.34/hook") == "https://93.184.216.34/hook"
    for refused in (
        "http://93.184.216.34/hook",
        "https://localhost/hook",
        "https://10.0.0.1/hook",
        "https://user:pw@example.com/",
        "https://service.internal/",
    ):
        with pytest.raises(TargetRefused):
            validate_url(refused)


async def test_a_push_is_signed_with_the_registered_token() -> None:
    seen: list[httpx.Request] = []

    def receive(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryPushNotificationConfigStore

    configs = InMemoryPushNotificationConfigStore()
    await configs.set_info(
        "run-1",
        TaskPushNotificationConfig(task_id="run-1", url="https://93.184.216.34/hook", token="tok"),
        ServerCallContext(),
    )
    notifier = PushNotifier(
        configs, client=httpx.AsyncClient(transport=httpx.MockTransport(receive))
    )
    task = Task(
        id="run-1", context_id="c1", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)
    )
    await notifier.send_notification("run-1", task)
    assert len(seen) == 1
    request = seen[0]
    assert verify_signature("tok", request.headers[SIGNATURE_HEADER], request.content)
    assert request.headers["X-A2A-Notification-Token"] == "tok"
    assert not await notifier.deliver("https://93.184.216.34/hook", "", "run-1", b"{}")
    assert not await notifier.validate_url("https://localhost/hook")


# --------------------------------------------------------------------------- answers


def test_an_answer_is_read_as_the_decision_it_names() -> None:
    from trellis.contracts import InterruptDecision, InterruptReason
    from trellis.harness.a2a.executor import _decision

    def message(text: str = "", data: dict[str, Any] | None = None) -> Any:
        parts = [new_text_part(text)] if text else []
        if data is not None:
            parts.append(new_data_part(data))
        return new_message(parts, role=Role.ROLE_USER)

    approval, question = InterruptReason.APPROVAL, InterruptReason.QUESTION
    assert _decision(approval, message("yes")) == (InterruptDecision.APPROVE, "yes")
    assert _decision(approval, message("deny"))[0] is InterruptDecision.REJECT
    assert _decision(approval, message(data={"amount": 5})) == (
        InterruptDecision.EDIT,
        {"amount": 5},
    )
    assert _decision(question, message("cancel"))[0] is InterruptDecision.CANCEL
    assert _decision(question, message("eu")) == (InterruptDecision.ANSWER, "eu")
    assert (
        _decision(question, message(data={"decision": "approve"}))[0] is InterruptDecision.APPROVE
    )
    with pytest.raises(ValueError, match="approval"):
        _decision(approval, message("maybe"))


async def test_the_a2a_routes_are_in_the_openapi_document_and_still_served_by_the_sdk() -> None:
    from fastapi import HTTPException
    from starlette.applications import Starlette

    from trellis.harness.a2a.server import served_by_the_sdk

    harness = Harness(config=Settings())
    app = FastAPI()
    agent = harness.wrap(greeter, id="greeter")
    agent.serve_a2a(app, URL)
    harness.wrap(greeter, id="other").serve_a2a(app, "http://a2a.test/agents/other")  # one tag
    doc = app.openapi()
    assert [t["name"] for t in doc["tags"]] == ["a2a"]
    rpc = doc["paths"]["/agents/greeter"]["post"]
    methods = rpc["requestBody"]["content"]["application/json"]["schema"]["properties"]["method"]
    assert {"SendMessage", "SendStreamingMessage", "CancelTask"} <= set(methods["enum"])
    assert set(rpc["responses"]["200"]["content"]) == {"application/json", "text/event-stream"}
    assert "get" in doc["paths"]["/agents/greeter/.well-known/agent-card.json"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://a2a.test") as http:
        card = await http.get("/agents/greeter/.well-known/agent-card.json")
        assert card.status_code == 200 and card.json()["name"] == "greeter"  # the SDK's route
    with pytest.raises(HTTPException):
        await served_by_the_sdk()
    bare = Starlette()
    agent.serve_a2a(bare, URL)  # a plain Starlette app: served, nothing to document
    assert len(bare.router.routes) == 2
