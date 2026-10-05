"""Calling a remote A2A agent from any code: ``remote(url, tenant=, user=)`` is a
:class:`RemoteAgent`, an async callable — a message in, the remote agent's answer out.

The call carries the caller's identity on the trusted-identity header, and ``thread`` as the A2A
context, so a conversation between two agents is one thread on both sides. When the remote task
asks something (``input-required``), ``on_input(question)`` answers it on the same task; without
``on_input`` the call raises :class:`InputRequired` and ``reply(task_id, answer)`` continues it.
If ``on_input`` raises instead (a run that pauses to ask a person), the remote task is cancelled
and the exception goes on.

The harness's ``a2a(url)`` tool is a ``RemoteAgent`` per call (:func:`remote_agent_tool`): the
calling run's identity and thread, ``on_input`` the run's own ``ask`` — the remote question
becomes this run's pause, and a resumed run calls again with the journal answering it — and the
call's idempotency key as the id of the message that opens the task.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Final, Self

import httpx
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

from trellis.contracts import ToolError, ToolSpec
from trellis.harness.a2a.identity import EXTENSION_URI
from trellis.harness.a2a.translate import TERMINAL_STATES, value_part, values
from trellis.harness.identity import identity_headers
from trellis.harness.runtime import current
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

#: Answers a remote question: ``on_input(question) -> answer``, sync or async.
OnInput = Callable[[str], Any]


class InputRequired(Exception):
    """The remote task waits for an answer and no ``on_input`` was given: answer it with
    ``reply(task_id, answer)``."""

    def __init__(self, question: str, task_id: str) -> None:
        super().__init__(question)
        self.question = question
        self.task_id = task_id


def remote(
    url: str,
    *,
    tenant: str,
    user: str,
    thread: str | None = None,
    on_input: OnInput | None = None,
    name: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> RemoteAgent:
    """The agent whose card is at ``{url}/.well-known/agent-card.json``, called as ``tenant`` /
    ``user`` on ``thread`` (the A2A context; each call its own when ``None``).

    ``on_input(question)`` answers a remote question (else :class:`InputRequired`); ``name``
    names its :attr:`~RemoteAgent.spec`; ``headers`` go on every request (an edge's
    ``Authorization``); ``client`` is an ``httpx.AsyncClient`` to use (never closed here),
    otherwise one with ``timeout`` is opened on connect and closed by ``aclose``.
    """
    return RemoteAgent(
        url,
        tenant=tenant,
        user=user,
        thread=thread,
        on_input=on_input,
        name=name,
        headers=headers,
        timeout=timeout,
        client=client,
    )


class RemoteAgent:
    """A remote A2A agent as an async callable: ``await agent(message)`` is its answer.

    ``connect()`` (or ``async with``) reads the card, which ``card`` and ``spec`` need; a call
    connects first when nothing has. A task that ends ``failed``, ``rejected`` or ``canceled``,
    or a remote agent that cannot be reached, is a :class:`~trellis.contracts.ToolError`.
    """

    def __init__(
        self,
        url: str,
        *,
        tenant: str,
        user: str,
        thread: str | None = None,
        on_input: OnInput | None = None,
        name: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        card: AgentCard | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.thread = thread
        self.on_input = on_input
        self.name = name
        #: Identity last: no header of the caller's can speak for someone else.
        self.headers = {
            **(headers or {}),
            HTTP_EXTENSION_HEADER: EXTENSION_URI,
            **identity_headers(tenant, user),
        }
        self._timeout = timeout
        self._http = client
        self._owns_http = client is None
        self._card = card
        self._client: Client | None = None

    @property
    def card(self) -> AgentCard:
        """The remote agent's card (read by ``connect``)."""
        if self._card is None:
            raise RuntimeError("the card is read by connect(): await it (or use async with)")
        return self._card

    @property
    def spec(self) -> ToolSpec:
        """The remote agent as a tool any framework can offer: ``{"message": string}`` in."""
        return _spec(self.card, self.name)

    async def connect(self) -> Self:
        """Read the card (unless it was given) and open the A2A client; once."""
        await self._connected()
        return self

    async def __call__(self, message: Any, *, message_id: str | None = None) -> Any:
        """Send ``message`` (text, or a JSON value as a data part) as a new task; its answer.
        ``message_id`` names the message (a fresh id otherwise): the same id again is the same
        message, to a server that deduplicates."""
        return await self._exchange(message, None, message_id)

    async def reply(self, task_id: str, answer: Any) -> Any:
        """Answer the question :class:`InputRequired` reported; the task's answer."""
        return await self._exchange(answer, task_id)

    async def aclose(self) -> None:
        """Close the HTTP client this agent opened (an injected one is the caller's)."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None
        self._client = None

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _connected(self) -> Client:
        if self._client is None:
            if self._http is None:
                self._http = _http(self._timeout)
            if self._card is None:
                self._card = await _read_card(self._http, self.url, self.headers)
            self._client = ClientFactory(
                ClientConfig(
                    httpx_client=self._http,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(self._card)
        return self._client

    async def _exchange(
        self, value: Any, task_id: str | None, message_id: str | None = None
    ) -> Any:
        client = await self._connected()
        context = ClientCallContext(service_parameters=self.headers)
        message = _message(value, self.thread, task_id)
        if message_id is not None:
            message.message_id = message_id
        reply = await _send(client, message, context)
        while reply.state == TaskState.TASK_STATE_INPUT_REQUIRED:
            question = reply.question or f"{self.card.name} is waiting for an answer"
            if self.on_input is None:
                raise InputRequired(question, reply.task_id)
            try:
                answer = self.on_input(question)
                if inspect.isawaitable(answer):
                    answer = await answer
            except Exception:
                await client.cancel_task(CancelTaskRequest(id=reply.task_id), context=context)
                raise
            reply = await _send(client, _message(answer, self.thread, reply.task_id), context)
        if reply.state in TERMINAL_STATES and reply.state != TaskState.TASK_STATE_COMPLETED:
            raise ToolError(
                f"{self.card.name} ended {TaskState.Name(reply.state)}: {reply.text}",
                source="a2a",
            )
        return reply.output


def _spec(card: AgentCard, name: str | None) -> ToolSpec:
    """A card as a tool: named after the agent (or ``name``), described by it."""
    description = " ".join(card.description.split())[:MAX_DESCRIPTION] or f"the {card.name} agent"
    return ToolSpec(
        name=_UNSAFE.sub("_", name or card.name)[:MAX_NAME],
        description=description,
        input_schema=SCHEMA,
        source="a2a",
        server=card.name,
        side_effects="write",
    )


async def remote_agent_tool(
    url: str,
    *,
    name: str | None = None,
    timeout: float | None = None,  # noqa: ASYNC109 - the tool's, for each of its calls
) -> Tool:
    """The ``a2a(url)`` tool: the card read once, then each call a :class:`RemoteAgent` as the
    calling run (its tenant, user and thread; ``on_input`` its ``ask``; the call's idempotency
    key the opening message's id), taking at most ``timeout`` (:data:`TIMEOUT_SECONDS` when
    ``None``)."""
    seconds = TIMEOUT_SECONDS if timeout is None else timeout
    async with _http(seconds) as http:
        card = await _read_card(http, url.rstrip("/"), {})

    async def run(args: dict[str, Any]) -> Any:
        runtime = current()
        if runtime is None:
            raise ToolError("an A2A call is made inside a harness run", source="a2a")
        async with RemoteAgent(
            url,
            tenant=runtime.tenant,
            user=runtime.user,
            thread=runtime.thread or runtime.run_id,
            on_input=runtime.ask,
            timeout=seconds,
            card=card,
        ) as agent:
            text = str(args.get("message") or "")
            return await agent(text, message_id=runtime.idempotency_key)

    return Tool(_spec(card, name), run, timeout=seconds)


async def _read_card(http: httpx.AsyncClient, url: str, headers: Mapping[str, str]) -> AgentCard:
    return await A2ACardResolver(http, url, AGENT_CARD_WELL_KNOWN_PATH).get_agent_card(
        http_kwargs={"headers": dict(headers)}
    )


def _http(timeout: float) -> httpx.AsyncClient:
    """The HTTP client a remote agent opens (a seam tests replace with an ASGI transport)."""
    return httpx.AsyncClient(timeout=timeout)


def _message(value: Any, context_id: str | None, task_id: str | None) -> Message:
    part = new_text_part(value) if isinstance(value, str) else value_part(value)
    return new_message([part], context_id=context_id, task_id=task_id, role=Role.ROLE_USER)


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
