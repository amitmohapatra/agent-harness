"""Registry agents as tools: the calling half of A2A (design §6 mode B, §9).

```mermaid
sequenceDiagram
  participant P as Planner (react / your loop)
  participant T as A2AAgentClient (ToolClient)
  participant R as Registry directory
  participant A as Remote agent
  P->>T: list_tools()
  T->>R: find() -> cards (name, skills, card URL)
  P->>T: call("a2a_billing_refund_agent", message="refund order 7")
  T->>A: GET card URL (verify the name it claims)
  T->>A: SendMessage(contextId = this run's thread, trusted identity header)
  A-->>T: working ... artifact ... completed
  T-->>P: ToolOutcome(output)
  alt the remote agent asked a person
    A-->>T: input-required(question)
    T-->>P: raise AgentPaused -> the local run pauses, one mechanism
  end
```

Why this is a ``ToolClient`` and not a new port: a remote agent is a tool with a long timeout. Put
it in a ``CompositeToolClient`` next to local functions and the gateway's MCP tools and policy,
instrumentation, idempotency keys and tool memory apply to it unchanged — the planner does not
learn a second way to call things, and an approval rule about "tools that spend money" covers a
remote agent that spends money.

Three rules worth stating:

* **Context and tasks.** The caller's thread is the A2A ``contextId``, so a conversation between
  two agents is one thread on both sides. A task id is kept per (context, agent) so a follow-up
  continues the same task; a task that reached a terminal state is forgotten, because A2A tasks
  are immutable once terminal and a refinement is a new task on the same context.
* **Identity travels, and is not asked for.** The local run's tenant, user and workspace go on the
  trusted-context header with the activated extension. A remote agent that trusts its edge gets
  the caller it actually has.
* **A card is data.** It is fetched from the location the Registry published, and a card whose
  ``name`` is not the agent that was asked for is refused rather than called.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
from a2a.client import (
    A2ACardResolver,
    AuthInterceptor,
    Client,
    ClientCallContext,
    ClientConfig,
    ClientFactory,
    CredentialService,
)
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import (
    AgentCard as SDKAgentCard,
)
from a2a.types import (
    Message,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
)
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import MessageToDict
from trellis.contracts import ToolStatus
from trellis.contracts.a2a import AgentCard
from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.errors import AgentPaused, ToolError
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec

from trellis.harness.events.targets import TargetRefused, resolved_addresses, validate_url
from trellis.harness.registry.directory import AGENT_CARD_PATH
from trellis.harness.runtime.logging import get_logger
from trellis.harness.runtime.propagation import current_context, trace_headers
from trellis.harness_a2a.card import DEFAULT_MODES
from trellis.harness_a2a.identity import EXTENSION_URI, IDENTITY_FIELDS, identity_headers
from trellis.harness_a2a.translate import TERMINAL_STATES

log = get_logger("trellis.harness_a2a.client")

SOURCE: Final = "a2a"
#: Tool names must be safe for a model's tool list, so an agent id becomes one deterministically.
TOOL_PREFIX: Final = "a2a_"
_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_-]")
#: How many (context, agent) task ids are remembered before the oldest are dropped.
MAX_TASKS: Final = 512
#: The argument the message text arrives in, and its accepted aliases.
MESSAGE_ARGS: Final = ("message", "text", "input", "objective")
#: How much catalogue text may reach a model's tool list, and on one line.
MAX_DESCRIPTION_CHARS: Final = 400
#: A skill id is an id. Anything else in that field is another system's prose, not a name.
_SKILL_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_STATUSES: Final[dict[Any, ToolStatus]] = {
    TaskState.TASK_STATE_COMPLETED: ToolStatus.OK,
    TaskState.TASK_STATE_FAILED: ToolStatus.ERROR,
    TaskState.TASK_STATE_REJECTED: ToolStatus.REJECTED,
    TaskState.TASK_STATE_CANCELED: ToolStatus.CANCELLED,
}
#: Why a call produced no result, when the state says more than "it did not work". A task still
#: submitted or working when the stream ended has not failed: it has not finished, and its id is
#: kept so the next call continues it.
_ERROR_CLASSES: Final[dict[Any, str]] = {
    TaskState.TASK_STATE_AUTH_REQUIRED: "A2AAuthRequired",
    TaskState.TASK_STATE_SUBMITTED: "A2ATaskIncomplete",
    TaskState.TASK_STATE_WORKING: "A2ATaskIncomplete",
}


def tool_name(agent_id: str) -> str:
    """The tool name a model sees for a registry agent.

    Agent ids carry a product prefix (``billing:refund-agent``), and a colon is not allowed in a
    tool name by most providers. The mapping is deterministic and the client keeps the reverse
    lookup, so nothing has to parse the name back.
    """
    return f"{TOOL_PREFIX}{_UNSAFE.sub('_', agent_id)}"


class MappingCredentials(CredentialService):
    """Credentials per security scheme, from a mapping or a callable.

    The value never comes from a card and is never logged. A deployment hands in a dict read from
    its own environment or secret store, or a callable that fetches one per call.
    """

    def __init__(self, source: Mapping[str, str] | Callable[[str], Any] | None = None) -> None:
        self._source = source or {}

    async def get_credentials(
        self, security_scheme_name: str, context: ClientCallContext | None = None
    ) -> str | None:
        source = self._source
        if isinstance(source, Mapping):
            value = source.get(security_scheme_name)
        else:
            value = source(security_scheme_name)
            if hasattr(value, "__await__"):
                value = await value
        return str(value) if value else None


class A2AAgentClient:
    """The harness ``ToolClient`` port over the agents a Registry lists."""

    name = SOURCE

    def __init__(
        self,
        directory: Any,
        *,
        credentials: CredentialService | Mapping[str, str] | Callable[[str], Any] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        context: Callable[[], AgentExecutionContext | None] = current_context,
        identity: Mapping[str, Any] | None = None,
        streaming: bool = True,
        timeout: float | None = None,
        limit: int = 50,
        transport: str = TransportProtocol.JSONRPC,
        card_path: str = AGENT_CARD_PATH,
        allow_local_targets: bool = False,
        verify_targets: bool = True,
    ) -> None:
        """``directory`` is an ``AgentDirectory`` (the Registry one, or a static list for tests).

        ``context`` answers which run is calling — by default the context bound in this async task,
        which is the run the planner is inside. ``identity`` is the fallback for a caller that is
        not a run (a script, a scheduled job): without either, a call is refused rather than sent
        with no caller, because an agent that cannot say who is asking should not be asking.

        ``allow_local_targets`` permits http and local card URLs (development only) and
        ``verify_targets=False`` skips name resolution (tests with a mocked transport); both are
        the same knobs the harness's webhook sink has, because a card URL is an outbound fetch of
        a URL somebody else wrote.
        """
        for method in ("get", "find"):
            if not callable(getattr(directory, method, None)):
                raise TypeError("A2AAgentClient needs an AgentDirectory")
        self._directory = directory
        self._credentials = (
            credentials
            if isinstance(credentials, CredentialService)
            else MappingCredentials(credentials)
        )
        self._client = httpx_client or httpx.AsyncClient(timeout=timeout or 60.0)
        self._owns_client = httpx_client is None
        self._context = context
        self.identity = dict(identity) if identity else None
        self.streaming = streaming
        self.timeout = timeout
        self.limit = limit
        self.transport = transport
        self.card_path = card_path
        self.allow_local_targets = allow_local_targets
        self.verify_targets = verify_targets
        self._specs: dict[str, ToolSpec] = {}
        self._cards: dict[str, AgentCard] = {}
        #: Keyed by (agent, card location): a card that moves is re-fetched, not re-used.
        self._remote: dict[tuple[str, str], tuple[SDKAgentCard, Client]] = {}
        self._tasks: dict[tuple[str, str, str], str] = {}

    # ------------------------------------------------------------------ the tool port
    async def list_tools(self) -> Sequence[ToolSpec]:
        """Every agent the caller may reach, as one tool each."""
        specs: dict[str, ToolSpec] = {}
        cards: dict[str, AgentCard] = {}
        for card in await self._directory.find(limit=self.limit):
            name = tool_name(card.name)
            specs[name] = _spec(name, card)
            cards[name] = card
        self._specs, self._cards = specs, cards
        return list(specs.values())

    def spec(self, tool: str) -> ToolSpec | None:
        return self._specs.get(tool)

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        """Send a message to a remote agent and return its result as a tool outcome.

        Raises :class:`AgentPaused` when the remote task needs an answer, so the local run pauses
        through the platform's one mechanism and the answer, once a person gives it, is sent as
        the next message on the same task.
        """
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        card = await self._card_for(call.tool)
        _remote_card, client = await self._connect(card)
        execution = self._context()
        identity = self._identity_of(execution)
        context_id = _context_id(execution, call)
        # Keyed by tenant as well as context: a thread id is caller-chosen material, so two tenants
        # that pick the same one must never continue each other's remote task (the same rule the
        # event sinks and the resolution registry state).
        key = (str(identity.get("tenant_id") or ""), context_id, card.name)
        task_id = self._task_to_continue(key, call)
        request = SendMessageRequest(
            message=_message(call.args, context_id=context_id, task_id=task_id),
            configuration=SendMessageConfiguration(
                accepted_output_modes=list(card.default_output_modes or DEFAULT_MODES)
            ),
        )
        result = await self._exchange(client, request, identity, execution, call.tool)
        if result.task_id and result.state not in TERMINAL_STATES:
            self._remember(key, result.task_id)
        else:
            self._tasks.pop(key, None)
        if result.state is TaskState.TASK_STATE_INPUT_REQUIRED:
            raise AgentPaused(
                result.question or f"{card.name} is waiting for an answer",
                expects=result.expects,
                payload={
                    SOURCE: {
                        "agent": card.name,
                        "tool": call.tool,
                        "task_id": result.task_id,
                        "context_id": context_id,
                    }
                },
            )
        status = ToolStatus.OK if result.answered else _STATUSES.get(result.state, ToolStatus.ERROR)
        error_class = _ERROR_CLASSES.get(result.state, "A2ATaskFailed")
        return ToolOutcome(
            tool=call.tool,
            status=status,
            output=result.output,
            error_class=error_class if status is ToolStatus.ERROR else None,
            metadata={"task_id": result.task_id, "context_id": context_id, "agent": card.name},
        )

    async def aclose(self) -> None:
        """Drop the per-agent clients and close the HTTP client, if this object opened it.

        Deliberately not ``Client.close()``: the SDK's client closes the *httpx* client it was
        given, and that one is shared by every agent here and possibly owned by the caller.
        """
        self._remote.clear()
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ internals
    async def _card_for(self, tool: str) -> AgentCard:
        card = self._cards.get(tool)
        if card is None:
            await self.list_tools()
            card = self._cards.get(tool)
        if card is None:
            raise ToolError(f"unknown agent tool {tool!r}", source=f"{SOURCE}.directory")
        return card

    async def _connect(self, card: AgentCard) -> tuple[SDKAgentCard, Client]:
        """The remote agent's own card and a client for it, fetched once per agent.

        The Registry publishes *where* a card is; the agent publishes the card. Fetching it is what
        makes transports, streaming support and security schemes the callee's own statement rather
        than a copy in a catalogue that may be stale.
        """
        base, path = await self._card_address(card)
        cached = self._remote.get((card.name, f"{base}{path}"))
        if cached is not None:
            return cached
        resolver = A2ACardResolver(self._client, base, agent_card_path=path)
        sdk_card = await resolver.get_agent_card()
        if sdk_card.name != card.name:
            raise ToolError(
                f"the card at {base}{path} claims to be {sdk_card.name!r}, not {card.name!r}",
                source=f"{SOURCE}.card",
            )
        await self._require_credential(sdk_card)
        config = ClientConfig(
            httpx_client=self._client,
            streaming=self.streaming and sdk_card.capabilities.streaming,
            supported_protocol_bindings=[self.transport],
        )
        client = ClientFactory(config).create(
            sdk_card, interceptors=[AuthInterceptor(self._credentials)]
        )
        self._remote[(card.name, f"{base}{path}")] = (sdk_card, client)
        return sdk_card, client

    async def _card_address(self, card: AgentCard) -> tuple[str, str]:
        """Where to fetch this agent's card from, once the URL passes the harness's target rules.

        A card URL is an outbound fetch of a URL *somebody else wrote* — the Registry holds it, and
        a hostile or mistaken entry pointing at ``169.254.169.254`` would otherwise be fetched with
        this process's network position. So it goes through exactly the checks a webhook target
        does: https, no credentials in the URL, a public address, re-resolved here.
        """
        base, path = _card_location(card, self.card_path)
        try:
            validate_url(f"{base}{path}", allow_local=self.allow_local_targets)
            if self.verify_targets:
                await resolved_addresses(base, allow_local=self.allow_local_targets)
        except TargetRefused as exc:
            raise ToolError(
                f"the card URL for {card.name} may not be fetched: {exc}", source=f"{SOURCE}.card"
            ) from exc
        return base, path

    async def _require_credential(self, sdk_card: SDKAgentCard) -> None:
        """Refuse to call an agent whose card requires authentication we cannot provide.

        The SDK's ``AuthInterceptor`` adds a credential when it finds one and silently adds nothing
        when it does not. Sending an unauthenticated request in that case wastes a round trip and
        reports the remote agent's 401 as the remote agent's problem, so it is refused here, named.
        """
        wanted = [
            name
            for requirement in sdk_card.security_requirements
            for name in requirement.schemes
            if name in sdk_card.security_schemes
        ]
        if not wanted:
            return
        for name in wanted:
            if await self._credentials.get_credentials(name, None):
                return
        raise ToolError(
            f"{sdk_card.name} requires one of the security schemes {sorted(set(wanted))} and no "
            "credential is configured for any of them",
            source=f"{SOURCE}.auth",
        )

    async def _exchange(
        self,
        client: Client,
        request: SendMessageRequest,
        identity: Mapping[str, Any],
        execution: AgentExecutionContext | None,
        tool: str,
    ) -> Outcome:
        """Send, follow the stream, and reduce it to what a tool call returns."""
        call_context = self._call_context(identity, execution)
        outcome = Outcome()
        try:
            async for response in client.send_message(request, context=call_context):
                outcome.absorb(response)
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}", source=f"{SOURCE}.{tool}") from exc
        return outcome

    def _call_context(
        self, identity: Mapping[str, Any], execution: AgentExecutionContext | None
    ) -> ClientCallContext:
        """The outbound headers: the trusted identity, the activated extension, trace context."""
        headers: dict[str, str] = {HTTP_EXTENSION_HEADER: EXTENSION_URI}
        headers.update(identity_headers(identity))
        if execution is not None:
            headers.update(trace_headers(execution))
        context = ClientCallContext(service_parameters=headers)
        if self.timeout is not None:
            context.timeout = self.timeout
        return context

    def _identity_of(self, execution: AgentExecutionContext | None) -> Mapping[str, Any]:
        """Who this call is from: the run that is calling, else the configured caller.

        Refusing here rather than sending an unidentified call is deliberate — the callee would
        answer with its own refusal, and the reason ("you sent no identity") is one this side
        already knows.
        """
        if execution is not None and execution.tenant_id:
            return {f: getattr(execution, f, None) for f in IDENTITY_FIELDS}
        if self.identity:
            return self.identity
        raise ToolError(
            "an A2A call needs the caller's identity: call it inside a harness run, or pass "
            "A2AAgentClient(identity={'tenant_id': ...})",
            source=f"{SOURCE}.identity",
        )

    def _task_to_continue(self, key: tuple[str, str, str], call: ToolCall) -> str:
        """The remote task this call continues: the one remembered for (tenant, context, agent).

        A caller may name it, and a planner is a model, so a named id is only honoured when it *is*
        the remembered one. Otherwise a prompt-injected plan could post a message into any task id
        it can guess; the callee's own ownership check would refuse a foreign one, but the local
        side already knows which task belongs to this conversation.
        """
        remembered = self._tasks.get(key, "")
        named = str(call.args.get("task_id") or "")
        if named and named != remembered:
            raise ToolError(
                f"task {named!r} is not this conversation's task with {key[2]}",
                source=f"{SOURCE}.task",
            )
        return remembered

    def _remember(self, key: tuple[str, str, str], task_id: str) -> None:
        self._tasks[key] = task_id
        while len(self._tasks) > MAX_TASKS:
            self._tasks.pop(next(iter(self._tasks)))


class Outcome:
    """What a remote task said, reduced from its stream."""

    __slots__ = ("artifacts", "expects", "question", "state", "task_id", "texts")

    def __init__(self) -> None:
        self.state: Any = TaskState.TASK_STATE_UNSPECIFIED
        self.task_id: str = ""
        self.texts: list[str] = []
        self.artifacts: list[Any] = []
        self.question: str | None = None
        self.expects: dict[str, Any] | None = None

    @property
    def output(self) -> Any:
        """The result: the artifacts the agent published, else what it said."""
        if len(self.artifacts) == 1:
            return self.artifacts[0]
        if self.artifacts:
            return self.artifacts
        return "\n".join(t for t in self.texts if t) or None

    @property
    def answered(self) -> bool:
        """Whether the agent answered outright, with a message and no task of its own.

        A2A allows it, and it is the reason ``absorb`` handles a ``message`` payload: a short
        question can be answered without opening a task. Reporting that as a failed tool call would
        teach tool memory a failure that never happened.
        """
        return (
            self.state == TaskState.TASK_STATE_UNSPECIFIED
            and not self.task_id
            and self.output is not None
        )

    def absorb(self, response: StreamResponse) -> None:
        payload = response.WhichOneof("payload")
        if payload == "task":
            self._task(response.task)
        elif payload == "status_update":
            update = response.status_update
            self.task_id = update.task_id or self.task_id
            self.state = update.status.state
            self._status_message(update.status.message)
        elif payload == "artifact_update":
            self.task_id = response.artifact_update.task_id or self.task_id
            self.artifacts.extend(_values(response.artifact_update.artifact.parts))
        elif payload == "message":
            self.texts.extend(_texts(response.message.parts))

    def _task(self, task: Task) -> None:
        self.task_id = task.id or self.task_id
        self.state = task.status.state
        for artifact in task.artifacts:
            self.artifacts.extend(_values(artifact.parts))
        self._status_message(task.status.message)

    def _status_message(self, message: Message) -> None:
        """A status message is prose for a person; on a pause it also carries what is asked."""
        if not message.parts:
            return
        texts = _texts(message.parts)
        if self.state is TaskState.TASK_STATE_INPUT_REQUIRED:
            self.question = "\n".join(texts) or self.question
            for value in _values(message.parts):
                if isinstance(value, Mapping) and value.get("expects"):
                    expects = value["expects"]
                    self.expects = dict(expects) if isinstance(expects, Mapping) else None
        self.texts.extend(texts)


# ---------------------------------------------------------------------------- helpers


def _spec(name: str, card: AgentCard) -> ToolSpec:
    """A remote agent as a tool the planner can choose.

    The description names the agent and its skills because that is what a model selects on, and the
    schema is deliberately small: a message, optionally structured data, optionally the task to
    continue. An agent is asked in words; it is not an RPC with a bespoke signature.

    The text is somebody else's: a Registry entity's description and a remote card's skills. It
    reaches a model, so it is flattened to one bounded line first — a description must not be able
    to carry an instruction block into the planner's tool list. What the agent may actually do is
    still decided by policy, not by what its card says about itself.
    """
    skills = ", ".join(_skill_id(skill.id) for skill in card.skills if _skill_id(skill.id))
    description = _one_line(card.description, MAX_DESCRIPTION_CHARS) or f"the {card.name} agent"
    return ToolSpec(
        name=name,
        description=f"{description}." + (f" Skills: {skills}." if skills else ""),
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "what this agent should do"},
                "data": {"type": "object", "description": "structured input, when it needs one"},
                "task_id": {
                    "type": "string",
                    "description": "continue this task of the same conversation",
                },
            },
            "required": ["message"],
        },
        tags=[skill.id for skill in card.skills],
        source=SOURCE,
        server=card.name,
        side_effects="unknown",  # another agent's side effects are its own to declare
    )


def _one_line(text: str, limit: int) -> str:
    """Somebody else's text, on one line and bounded: no control characters, no newlines."""
    flattened = " ".join(str(text or "").split())
    return flattened[:limit].rstrip()


def _skill_id(skill_id: str) -> str:
    """The id part of a skill id, and nothing after it.

    A skill id names a capability (``billing.refund``). Prose in that field — a space and then a
    sentence — is not a name, and this is the field a model reads next to the description, so only
    the leading id survives.
    """
    found = _SKILL_ID.match(str(skill_id or "").strip())
    return found.group(0) if found else ""


def _message(args: Mapping[str, Any], *, context_id: str, task_id: str) -> Message:
    text = next((str(args[k]) for k in MESSAGE_ARGS if args.get(k)), "")
    data = args.get("data")
    parts = [new_text_part(text)] if text else []
    if isinstance(data, Mapping) and data:
        parts.append(new_data_part(dict(data)))
    if not parts:
        raise ToolError("an A2A call needs a message", source=f"{SOURCE}.call")
    message = new_message(parts, context_id=context_id, role=Role.ROLE_USER)
    if task_id:
        message.task_id = task_id
    return message


def _context_id(execution: AgentExecutionContext | None, call: ToolCall) -> str:
    """The A2A conversation this call belongs to: the caller's thread, else its run."""
    if execution is not None:
        return execution.thread_id or execution.agent_run_id
    return str(call.args.get("context_id") or call.idempotency_key or "a2a-context")


def _card_location(card: AgentCard, default_path: str) -> tuple[str, str]:
    """The base URL and card path for the location the Registry published.

    The Registry may hold the card document itself (``.../.well-known/agent-card.json``) or the
    agent's endpoint; both are accepted, and both resolve to one address.
    """
    location = str(card.metadata.get("card_url") or card.url)
    split = urlsplit(location)
    if not split.scheme or not split.netloc:
        raise ToolError(f"{card.name} has no usable card URL", source=f"{SOURCE}.card")
    path = split.path or default_path
    if not path.endswith(".json"):
        path = f"{path.rstrip('/')}{default_path}"
    return f"{split.scheme}://{split.netloc}", path


def _texts(parts: Sequence[Any]) -> list[str]:
    return [p.text for p in parts if p.WhichOneof("content") == "text" and p.text]


def _values(parts: Sequence[Any]) -> list[Any]:
    """A part list as plain Python: text as strings, data parts as objects, file parts as URLs."""
    values: list[Any] = []
    for part in parts:
        which = part.WhichOneof("content")
        if which == "text":
            values.append(part.text)
        elif which == "data":
            values.append(MessageToDict(part.data))
        elif which == "url":
            values.append(part.url)
    return values


__all__ = [
    "MAX_TASKS",
    "SOURCE",
    "TOOL_PREFIX",
    "A2AAgentClient",
    "MappingCredentials",
    "tool_name",
]
