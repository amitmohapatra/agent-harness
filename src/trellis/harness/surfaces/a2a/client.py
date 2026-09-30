"""A remote A2A agent as one harness tool (``a2a(url)``): a message in, its answer out.

The call carries the current run's identity on the trusted-identity header, and the run's
thread as the A2A context, so a conversation between two agents is one thread on both sides.
When the remote task asks a person something (``input-required``) the question becomes this
run's own pause — ``current().ask`` — and the answer is sent on the same remote task; if the
local run pauses instead, the remote task is cancelled, and a resumed run asks again.
"""

from __future__ import annotations

import re
from typing import Any, Final

import httpx
from trellis.contracts import ToolError, ToolSpec

from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.helpers import new_message, new_text_part
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    Message,
    Role,
    SendMessageRequest,
    StreamResponse,
    TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, TransportProtocol
from trellis.harness.runtime import Paused, current
from trellis.harness.surfaces.a2a.identity import EXTENSION_URI, identity_headers
from trellis.harness.surfaces.a2a.translate import TERMINAL_STATES, value_part, values
from trellis.harness.tools.base import Tool

#: How long one exchange with a remote agent may take.
TIMEOUT_SECONDS: Final = 120.0
#: A model's tool names: letters, digits, ``_`` and ``-``, at most 64 characters.
_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_-]")
MAX_NAME: Final = 64
MAX_DESCRIPTION: Final = 400
SCHEMA: Final = {
    "type": "object",
    "properties": {"message": {"type": "string", "description": "what the agent should do"}},
    "required": ["message"],
}


def _http() -> httpx.AsyncClient:
    """The HTTP client for one exchange (a seam tests replace with an ASGI transport)."""
    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS)


async def remote_agent_tool(url: str, *, name: str | None = None) -> Tool:
    """The agent whose card is at ``{url}/.well-known/agent-card.json``, as a tool."""
    async with _http() as http:
        card = await A2ACardResolver(
            http, url.rstrip("/"), AGENT_CARD_WELL_KNOWN_PATH
        ).get_agent_card()
    description = " ".join(card.description.split())[:MAX_DESCRIPTION] or f"the {card.name} agent"
    spec = ToolSpec(
        name=_UNSAFE.sub("_", name or card.name)[:MAX_NAME],
        description=description,
        input_schema=SCHEMA,
        source="a2a",
        server=card.name,
        side_effects="write",
    )

    async def run(args: dict[str, Any]) -> Any:
        return await _exchange(card, str(args.get("message") or ""))

    return Tool(spec, run)


async def _exchange(card: AgentCard, text: str) -> Any:
    runtime = current()
    if runtime is None:
        raise ToolError("an A2A call is made inside a harness run", source="a2a")
    headers = {
        HTTP_EXTENSION_HEADER: EXTENSION_URI,
        **identity_headers(runtime.tenant, runtime.user),
    }
    context = ClientCallContext(service_parameters=headers)
    context_id = runtime.thread or runtime.run_id
    async with _http() as http:
        client = ClientFactory(
            ClientConfig(httpx_client=http, supported_protocol_bindings=[TransportProtocol.JSONRPC])
        ).create(card)
        reply = await _send(client, _message(text, context_id, None), context)
        while reply.state == TaskState.TASK_STATE_INPUT_REQUIRED:
            try:
                answer = await runtime.ask(
                    reply.question or f"{card.name} is waiting for an answer"
                )
            except Paused:
                await client.cancel_task(CancelTaskRequest(id=reply.task_id), context=context)
                raise
            reply = await _send(client, _message(answer, context_id, reply.task_id), context)
    if reply.state in TERMINAL_STATES and reply.state != TaskState.TASK_STATE_COMPLETED:
        raise ToolError(
            f"{card.name} ended {TaskState.Name(reply.state)}: {reply.text}", source="a2a"
        )
    return reply.output


def _message(value: Any, context_id: str, task_id: str | None) -> Message:
    part = new_text_part(value) if isinstance(value, str) else value_part(value)
    message = new_message([part], context_id=context_id, role=Role.ROLE_USER)
    if task_id:
        message.task_id = task_id
    return message


async def _send(client: Client, message: Message, context: ClientCallContext) -> _Reply:
    reply = _Reply()
    try:
        async for response in client.send_message(
            SendMessageRequest(message=message), context=context
        ):
            reply.absorb(response)
    except Exception as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}", source="a2a") from exc
    return reply


class _Reply:
    """What a remote task said, reduced from its stream."""

    def __init__(self) -> None:
        self.state: Any = TaskState.TASK_STATE_UNSPECIFIED
        self.task_id = ""
        self.texts: list[str] = []
        self.artifacts: list[Any] = []
        self.question: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(t for t in self.texts if t)

    @property
    def output(self) -> Any:
        if len(self.artifacts) == 1:
            return self.artifacts[0]
        return self.artifacts or self.text or None

    def absorb(self, response: StreamResponse) -> None:
        payload = response.WhichOneof("payload")
        if payload == "task":
            task = response.task
            self.task_id = task.id or self.task_id
            self.state = task.status.state
            for artifact in task.artifacts:
                self.artifacts.extend(values(artifact.parts))
            self._status(task.status.message)
        elif payload == "status_update":
            update = response.status_update
            self.task_id = update.task_id or self.task_id
            self.state = update.status.state
            self._status(update.status.message)
        elif payload == "artifact_update":
            self.artifacts.extend(values(response.artifact_update.artifact.parts))
        elif payload == "message":
            self.texts.extend(v for v in values(response.message.parts) if isinstance(v, str))

    def _status(self, message: Message) -> None:
        texts = [v for v in values(message.parts) if isinstance(v, str)]
        if self.state == TaskState.TASK_STATE_INPUT_REQUIRED and texts:
            self.question = "\n".join(texts)
        elif self.state != TaskState.TASK_STATE_WORKING:
            self.texts.extend(texts)
