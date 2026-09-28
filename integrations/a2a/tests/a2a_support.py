"""Shared fixtures for the A2A tests: a harness, an agent that pauses, and an in-process client.

There is no fake A2A server here. Every test drives the real ``a2a-sdk`` server through
``httpx.ASGITransport`` — the SDK ships no in-memory transport, and a JSON-RPC surface is exactly
the kind of thing that works in a unit test and fails on the wire.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from a2a.client import ClientCallContext, ClientConfig, ClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import Message, Role, SendMessageRequest, StreamResponse
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import MessageToDict
from trellis.contracts.a2a import AgentCard
from trellis.contracts.errors import AgentPaused

from trellis.harness import AgentHarness, CollectingEventSink
from trellis.harness.interrupts import ANSWER
from trellis.harness_a2a import A2AServer, identity_headers
from trellis.harness_a2a.identity import EXTENSION_URI

TENANT = "acme"
ACME = {"tenant_id": TENANT, "user_id": "u1", "workspace_id": "ws1"}
BASE_URL = "http://a2a.test"
AGENT_URL = f"{BASE_URL}/"


async def greeter(payload: Any, runtime: Any) -> Any:
    """An agent that answers, asks, remembers the answer, and can fail on demand."""
    answer = runtime.state.get("resolutions", {}).get(ANSWER)
    if isinstance(payload, str) and payload.startswith("ask") and answer is None:
        raise AgentPaused("Which region?", expects={"type": "string"})
    if payload == "whoami":
        return {
            "tenant": runtime.context.tenant_id,
            "user": runtime.context.user_id,
            "workspace": runtime.context.workspace_id,
            "thread": runtime.context.thread_id,
            "run": runtime.context.agent_run_id,
            "metadata": runtime.state["request"].metadata,
        }
    if payload == "boom":
        raise RuntimeError("boom")
    if answer is not None:
        return f"deploying to {answer.answer}"
    return f"hello {payload}"


def harness(**options: Any) -> AgentHarness:
    """A harness configured as the AG-UI tests configure theirs: no memory service, real pipeline."""
    options.setdefault("defaults", {"tenant_id": TENANT})
    options.setdefault("event_sinks", [CollectingEventSink()])
    options.setdefault("error_mode", "return")
    return AgentHarness(memory=None, **options)


def asgi_client(app: Any, *, base_url: str = BASE_URL) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


def sdk_client(server: A2AServer, http: httpx.AsyncClient, *, streaming: bool = True) -> Any:
    config = ClientConfig(
        httpx_client=http,
        streaming=streaming,
        supported_protocol_bindings=[TransportProtocol.JSONRPC],
    )
    return ClientFactory(config).create(server.sdk_card)


def caller(identity: dict[str, Any] | None = None, **extra: str) -> ClientCallContext:
    """A call context carrying the trusted identity header the platform's edge would set."""
    headers = {HTTP_EXTENSION_HEADER: EXTENSION_URI, **extra}
    if identity is not None:
        headers.update(identity_headers(identity))
    return ClientCallContext(service_parameters=headers)


def message(
    text: str = "", *, task_id: str = "", context_id: str = "", data: dict[str, Any] | None = None
) -> Message:
    parts = [new_text_part(text)] if text else []
    if data is not None:
        parts.append(new_data_part(data))
    built = new_message(parts, context_id=context_id or None, role=Role.ROLE_USER)
    if task_id:
        built.task_id = task_id
    return built


def request(
    text: str = "", *, tenant: str = "", data: dict[str, Any] | None = None, **fields: str
) -> SendMessageRequest:
    built = SendMessageRequest(message=message(text, data=data, **fields))
    if tenant:
        built.tenant = tenant
    return built


async def send(client: Any, req: SendMessageRequest, context: ClientCallContext) -> list[Any]:
    """Every stream response of one call, in order."""
    return [response async for response in client.send_message(req, context=context)]


def states(responses: Sequence[StreamResponse]) -> list[Any]:
    """The task states the stream reported, in order."""
    seen = []
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "task":
            seen.append(response.task.status.state)
        elif which == "status_update":
            seen.append(response.status_update.status.state)
    return seen


def final(responses: Sequence[StreamResponse]) -> Any:
    """The last task state reported."""
    return states(responses)[-1]


def task_id_of(responses: Sequence[StreamResponse]) -> str:
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "task" and response.task.id:
            return response.task.id
        if which == "status_update" and response.status_update.task_id:
            return response.status_update.task_id
    raise AssertionError("no task id in the stream")


def texts(responses: Sequence[StreamResponse]) -> list[str]:
    found: list[str] = []
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "status_update":
            found.extend(
                p.text for p in response.status_update.status.message.parts if p.HasField("text")
            )
    return found


def data_parts(responses: Sequence[StreamResponse]) -> list[dict[str, Any]]:
    """Every structured part carried on a status message, as plain dicts."""
    return [
        MessageToDict(part.data)
        for response in responses
        if response.WhichOneof("payload") == "status_update"
        for part in response.status_update.status.message.parts
        if part.HasField("data")
    ]


def artifacts(responses: Sequence[StreamResponse]) -> list[Any]:
    found: list[Any] = []
    for response in responses:
        if response.WhichOneof("payload") == "artifact_update":
            for part in response.artifact_update.artifact.parts:
                which = part.WhichOneof("content")
                if which == "text":
                    found.append(part.text)
                elif which == "data":
                    found.append(MessageToDict(part.data))
        elif response.WhichOneof("payload") == "task":
            for artifact in response.task.artifacts:
                for part in artifact.parts:
                    if part.HasField("text"):
                        found.append(part.text)
    return found


class StaticDirectory:
    """An ``AgentDirectory`` over a fixed list of cards, for the calling tests."""

    def __init__(self, cards: Sequence[AgentCard]) -> None:
        self.cards = list(cards)
        self.published: list[AgentCard] = []

    async def get(self, agent_id: str) -> AgentCard | None:
        return next((c for c in self.cards if c.name == agent_id), None)

    async def find(
        self, query: str | None = None, *, skill: str | None = None, limit: int = 20
    ) -> Sequence[AgentCard]:
        found = [
            c
            for c in self.cards
            if (skill is None or c.skill(skill) is not None)
            and (not query or query.lower() in f"{c.name} {c.description}".lower())
        ]
        return found[:limit]

    async def publish(self, card: AgentCard) -> None:
        self.published.append(card)
