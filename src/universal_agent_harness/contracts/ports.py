"""Every outbound dependency of the harness, as a Protocol (§27, §61).

The core imports nothing but these protocols and the contracts. Concrete adapters (Memory
Service SDK, OpenTelemetry, Langfuse, local tools, LangGraph...) implement them and are
injected. That is what keeps the core framework- and vendor-neutral.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from universal_agent_harness.contracts.artifacts import ArtifactRef, MemoryObservation
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.descriptors import AgentDescriptor
from universal_agent_harness.contracts.errors import AgentError
from universal_agent_harness.contracts.events import AgentEvalEvent
from universal_agent_harness.contracts.messages import AgentRequest, AgentResult
from universal_agent_harness.contracts.model import ModelRequest, ModelResponse
from universal_agent_harness.contracts.tool import ToolCall, ToolOutcome, ToolSpec

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle at runtime
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


# --------------------------------------------------------------------------- model / tool


@runtime_checkable
class ModelClient(Protocol):
    """A model provider. Implementations are wrapped by the harness for instrumentation."""

    async def invoke(self, request: ModelRequest | str, /, **kwargs: Any) -> ModelResponse: ...

    async def structured(
        self, request: ModelRequest | str, /, schema: Any, **kwargs: Any
    ) -> ModelResponse: ...

    def stream(self, request: ModelRequest | str, /, **kwargs: Any) -> AsyncIterator[Any]: ...


@runtime_checkable
class ToolClient(Protocol):
    """A tool runtime: local callables, an MCP server, or a gateway."""

    async def list_tools(self) -> Sequence[ToolSpec]: ...

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome: ...


@runtime_checkable
class ArtifactClient(Protocol):
    """Where large payloads go so they never sit inside a result or a graph state (§43)."""

    async def put(
        self,
        content: bytes | str,
        *,
        type: str = "blob",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactRef: ...

    async def get(self, artifact_id: str) -> bytes | None: ...


# --------------------------------------------------------------------------- memory


@runtime_checkable
class MemoryPort(Protocol):
    """The harness's view of the Memory Service. Implemented by the SDK adapter and by a
    no-op. Retrieval returns whatever bundle type the backing service produces; the harness
    treats it as opaque except for the small facts it reads through :meth:`describe`."""

    enabled: bool

    async def retrieve(self, query: str, /, **options: Any) -> Any | None: ...

    async def observe(self, observation: MemoryObservation, /) -> Any | None: ...

    async def record_input(self, text: str, /, **metadata: Any) -> Any | None: ...

    async def record_output(self, text: str, /, **metadata: Any) -> Any | None: ...

    def describe(self, bundle: Any, /) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- telemetry


@runtime_checkable
class TelemetryProvider(Protocol):
    """Span/event/metric emission (§23). ``start_span`` is a context manager yielding a
    :class:`HarnessSpan`-shaped object."""

    def start_span(
        self, name: str, *, kind: str = "internal", attributes: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[Any]: ...

    def record_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None: ...

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None: ...

    def flush(self, timeout_seconds: float = 5.0) -> None: ...


@runtime_checkable
class TelemetryRedactor(Protocol):
    """What may leave the process (§27). Applied before any attribute reaches a backend."""

    def redact_attributes(self, attributes: Mapping[str, Any]) -> dict[str, Any]: ...

    def redact_input(self, value: Any) -> Any: ...

    def redact_output(self, value: Any) -> Any: ...


@runtime_checkable
class EvaluationProvider(Protocol):
    async def score(
        self,
        name: str,
        value: float | str,
        /,
        *,
        context: AgentExecutionContext | None = None,
        comment: str | None = None,
        **metadata: Any,
    ) -> None: ...

    async def submit_dataset_item(self, dataset: str, item: Mapping[str, Any]) -> None: ...

    async def submit_feedback(self, feedback: Mapping[str, Any]) -> None: ...


@runtime_checkable
class EvaluationSink(Protocol):
    """Where :class:`AgentEvalEvent` values go. Async, off the critical path (§29)."""

    async def emit(self, event: AgentEvalEvent) -> None: ...


@runtime_checkable
class PromptProvider(Protocol):
    """Prompt management (§31). Optional; agents work without one."""

    # ``name`` is positional-only: a prompt's own variables may legitimately be called
    # "name", and they arrive in ``**vars``.
    async def get_prompt(self, name: str, /, *, version: str | None = None, **vars: Any) -> Any: ...


# --------------------------------------------------------------------------- policy / registry


@runtime_checkable
class AgentPolicyProvider(Protocol):
    """Authorization decisions (§48). A denial raises ``PolicyDeniedError`` in the harness."""

    async def authorize_execution(self, request: AgentRequest) -> bool | str: ...

    async def authorize_tool(
        self, context: AgentExecutionContext, call: ToolCall
    ) -> bool | str: ...

    async def authorize_model(
        self, context: AgentExecutionContext, request: ModelRequest
    ) -> bool | str: ...


@runtime_checkable
class AgentRegistryClient(Protocol):
    """Future agent registry (§47). Default implementation is a no-op."""

    async def register(self, descriptor: AgentDescriptor) -> None: ...

    async def heartbeat(self, descriptor: AgentDescriptor, *, status: str = "healthy") -> None: ...


# --------------------------------------------------------------------------- pipeline


@runtime_checkable
class AgentInterceptor(Protocol):
    """One stage of the execution pipeline (§20). Deterministically ordered by ``order``."""

    name: str
    order: int

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest: ...

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult: ...

    async def on_error(
        self, error: AgentError, runtime: AgentRuntime
    ) -> AgentResult | None: ...


@runtime_checkable
class LifecycleListener(Protocol):
    """Observes lifecycle events. Must not raise; the harness swallows and logs if it does."""

    def on_event(self, event: str, payload: Mapping[str, Any]) -> Awaitable[None] | None: ...


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Framework-specific wrapping (§50). The only place a framework may be imported."""

    name: str

    def supports(self, target: object) -> bool: ...

    def wrap(self, target: object, config: Any) -> object: ...

    def extract_context(self, *args: Any, **kwargs: Any) -> AgentExecutionContext | None: ...

    def map_result(self, result: AgentResult, *args: Any, **kwargs: Any) -> object: ...
