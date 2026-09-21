"""``ExecutionCoordinator``: one agent execution, start to finish.

The order is fixed and is the whole point of the harness:

    sampling decision -> runtime construction -> ``agent.run`` span opened
      -> interceptor ``before`` chain (identity, policy, memory, telemetry, timeout)
        -> the developer's agent, inside a deadline and a cancellation scope
      -> result coercion -> interceptor ``after`` chain (validation, memory, evaluation)
    -> lifecycle events -> normalized result or the original exception

Everything below the span is a child of it, which is what produces the trace shape in §25.
The coordinator owns no framework knowledge: adapters give it a plain callable.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from universal_agent_contracts.context import AgentExecutionContext
from universal_agent_contracts.descriptors import AgentDescriptor
from universal_agent_contracts.errors import (
    AgentCancelledError,
    AgentError,
    AgentTimeoutError,
    ErrorCategory,
    HarnessError,
    is_pause_signal,
)
from universal_agent_contracts.events import LifecycleEvent
from universal_agent_contracts.messages import AgentRequest, AgentResponse, AgentStatus

from universal_agent_harness.execution.retry import RetryPolicy, with_retry
from universal_agent_harness.runtime.agent_runtime import AgentRuntime
from universal_agent_harness.runtime.cancellation import CancellationToken
from universal_agent_harness.runtime.logging import get_logger
from universal_agent_harness.runtime.propagation import bind
from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import AGENT_OVERHEAD


class ExecutionCoordinator:
    """Runs one agent under the harness. Constructed once per harness, reused per call."""

    def __init__(
        self,
        *,
        chain: Any,
        runtime_builder: Any,
        events: Any,
        sampler: Any,
        tracer: Any,
        error_mode: str = "raise",
    ) -> None:
        self.chain = chain
        self.runtime_builder = runtime_builder
        self.events = events
        self.sampler = sampler
        self.tracer = tracer
        self.error_mode = error_mode

    async def execute(
        self,
        agent: Callable[..., Any],
        request: AgentRequest,
        descriptor: AgentDescriptor,
        *,
        call: Callable[[AgentRuntime], Awaitable[Any]] | None = None,
        retry: RetryPolicy | None = None,
        error_mode: str | None = None,
        extra_interceptors: Any = (),
        memory_policy: Any = None,
    ) -> AgentResponse:
        """Execute ``agent`` (or ``call``) for ``request``. Returns a normalized result."""
        context = request.context
        decision = self.sampler.decide(agent_id=context.agent_id, run_id=context.agent_run_id)
        tracer = self.tracer.for_decision(decision)
        runtime = self.runtime_builder.build(
            context, descriptor, tracer=tracer, memory_policy=memory_policy
        )
        chain = self.chain.with_extra(extra_interceptors) if extra_interceptors else self.chain
        mode = error_mode or self.error_mode
        policy = retry or RetryPolicy()

        with tracer.agent_span(context, skills=descriptor.skill_ids or None) as span:
            runtime.state["span"] = span
            runtime.state["request"] = request
            runtime.state["sampling"] = decision
            with bind(context, runtime):
                self.events.emit(
                    LifecycleEvent.AGENT_START,
                    {"context": context, "descriptor": descriptor, "request": request},
                )
                try:
                    result = await self._run_pipeline(
                        agent, request, runtime, chain, policy, call=call
                    )
                except asyncio.CancelledError:
                    await self._on_cancel(runtime, chain)
                    raise
                except BaseException as exc:
                    if is_pause_signal(exc):
                        self._on_pause(exc, runtime, span)
                        raise
                    result, error = await self._on_error(exc, runtime, chain)
                    if result is None:
                        self._finish(runtime, None, error)
                        if mode == "raise":
                            raise
                        result = AgentResponse.failed(error, status=_status_for(error))
                        result = await self._safe_after(result, runtime, chain)
                    self._finish(runtime, result, error)
                    return result
                self._finish(runtime, result, None)
                return result

    # ------------------------------------------------------------------ pipeline
    async def _run_pipeline(
        self,
        agent: Callable[..., Any],
        request: AgentRequest,
        runtime: AgentRuntime,
        chain: Any,
        policy: RetryPolicy,
        *,
        call: Callable[[AgentRuntime], Awaitable[Any]] | None,
    ) -> AgentResponse:
        overhead = _Overhead()
        prepared = await chain.before(request, runtime)
        runtime.state["request"] = prepared
        overhead.mark_before()

        async def attempt(number: int) -> Any:
            if number > 1:
                runtime.state["span"].set(**{N.RETRY: number})
                runtime.logger.warning("agent.retry", attempt=number)
            return await self._invoke(agent, prepared, runtime, call=call)

        raw = await with_retry(
            attempt,
            policy,
            classify=lambda exc: AgentError.of(exc, trace_id=runtime.context.trace_id),
            on_retry=lambda error, n: runtime.tracer.metrics.count(
                "agent.retries.count", agent_id=runtime.agent_id, error_category=str(error.category)
            ),
        )
        overhead.mark_agent()
        result = AgentResponse.coerce(raw)
        result = await chain.after(result, runtime)
        overhead.record(runtime)
        return result

    async def _invoke(
        self,
        agent: Callable[..., Any],
        request: AgentRequest,
        runtime: AgentRuntime,
        *,
        call: Callable[[AgentRuntime], Awaitable[Any]] | None,
    ) -> Any:
        """Run the developer's agent inside the deadline and the cancellation scope."""
        timeout = _seconds_until(runtime.deadline)
        invoke = call(runtime) if call is not None else _call_agent(agent, request, runtime)
        if timeout is None:
            return await invoke
        scope = asyncio.timeout(timeout)
        try:
            async with scope:
                return await invoke
        except TimeoutError as exc:
            # An agent may raise TimeoutError of its own (a client library's, say). Only the
            # scope actually expiring means *the harness's* deadline was breached.
            if not scope.expired():
                raise
            runtime.cancellation.cancel("timeout")
            raise AgentTimeoutError(
                f"agent {runtime.agent_id!r} exceeded its {timeout:.3f}s deadline",
                details={"timeout_seconds": timeout},
                source="harness.timeout",
            ) from exc

    # ------------------------------------------------------------------ failure paths
    async def _on_error(
        self, exc: BaseException, runtime: AgentRuntime, chain: Any
    ) -> tuple[AgentResponse | None, AgentError]:
        error = AgentError.of(exc, trace_id=runtime.context.trace_id, source=runtime.agent_id)
        try:
            recovered = await chain.on_error(error, runtime)
        except Exception:
            runtime.logger.exception("interceptor on_error failed")
            recovered = None
        event = (
            LifecycleEvent.AGENT_TIMEOUT
            if error.category is ErrorCategory.TIMEOUT
            else LifecycleEvent.AGENT_ERROR
        )
        self.events.emit(event, {"context": runtime.context, "error": error})
        if not isinstance(exc, HarnessError) and not hasattr(exc, "agent_error"):
            # Attach the normalized error *without* changing the exception type (rule 3):
            # callers keep catching their own exceptions and also get the classification.
            with contextlib.suppress(AttributeError, TypeError):
                exc.agent_error = error  # type: ignore[attr-defined]
        return recovered, error

    def _on_pause(self, exc: BaseException, runtime: AgentRuntime, span: Any) -> None:
        """A suspended run is not a failed one.

        ``interrupt()`` in a LangGraph node raises to hand control back to the graph runtime,
        which persists the checkpoint and waits for a human. The exception is the mechanism,
        so the harness saw a run "fail" with an unclassifiable error every time a turn asked
        a person a question: an ERROR span, an on_agent_error interceptor pass, and an
        error-rate metric that counted the feature working as the feature breaking.

        So: mark the span OK — setting OK is final in OpenTelemetry, so the automatic
        set-status-on-exception that follows is ignored — say PAUSED, and re-raise unchanged
        so the graph suspends exactly as it would without the harness. ``after``
        interceptors do not run, because the turn is not over; they run on resume, when the
        node is re-entered and reaches its end.
        """
        span.set(**{N.STATUS: str(AgentStatus.PAUSED)})
        span.event("agent.paused", reason=type(exc).__name__)
        span.ok()
        runtime.logger.info("agent.paused", reason=type(exc).__name__)
        self.events.emit(LifecycleEvent.AGENT_PAUSE, {"context": runtime.context, "signal": exc})
        self.events.emit(
            LifecycleEvent.AGENT_FINISH,
            {"context": runtime.context, "status": str(AgentStatus.PAUSED)},
        )

    async def _on_cancel(self, runtime: AgentRuntime, chain: Any) -> None:
        runtime.cancellation.cancel("cancelled")
        error = AgentError(
            code=AgentCancelledError.code,
            category=ErrorCategory.CANCELLED,
            message="execution cancelled",
            trace_id=runtime.context.trace_id,
        )
        try:
            await chain.on_error(error, runtime)
        except Exception:  # pragma: no cover
            runtime.logger.debug("interceptor on_error failed during cancellation")
        self.events.emit(LifecycleEvent.AGENT_CANCEL, {"context": runtime.context})
        self.events.emit(
            LifecycleEvent.AGENT_FINISH,
            {"context": runtime.context, "status": "CANCELLED"},
        )

    async def _safe_after(
        self, result: AgentResponse, runtime: AgentRuntime, chain: Any
    ) -> AgentResponse:
        try:
            return await chain.after(result, runtime)
        except Exception:
            runtime.logger.exception("interceptor after() failed on the error path")
            return result

    def _finish(
        self, runtime: AgentRuntime, result: AgentResponse | None, error: AgentError | None
    ) -> None:
        if result is not None:
            status = str(result.status)
        else:
            status = str(error.category) if error else "ERROR"
        if result is not None and result.succeeded:
            self.events.emit(
                LifecycleEvent.AGENT_SUCCESS, {"context": runtime.context, "result": result}
            )
        self.events.emit(
            LifecycleEvent.AGENT_FINISH,
            {"context": runtime.context, "status": status, "result": result, "error": error},
        )


class _Overhead:
    """Measures harness-only time: pipeline work minus the agent's own execution (§78)."""

    __slots__ = ("agent_end", "before_end", "start")

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.before_end = self.start
        self.agent_end = self.start

    def mark_before(self) -> None:
        self.before_end = time.perf_counter()

    def mark_agent(self) -> None:
        self.agent_end = time.perf_counter()

    def record(self, runtime: AgentRuntime) -> None:
        total = time.perf_counter() - self.start
        agent_time = self.agent_end - self.before_end
        overhead_ms = max(0.0, (total - agent_time) * 1000.0)
        runtime.state["harness_overhead_ms"] = overhead_ms
        runtime.tracer.metrics.duration(AGENT_OVERHEAD, overhead_ms, agent_id=runtime.agent_id)


class RuntimeBuilder:
    """Assembles an :class:`AgentRuntime` per execution from the harness's providers."""

    def __init__(
        self,
        *,
        memory_factory: Any,
        model_client: Any,
        tool_client: Any,
        artifact_store: Any,
        policy: Any,
        events: Any,
        timeouts: Any,
        tools_config: Any,
        artifacts_config: Any,
    ) -> None:
        self.memory_factory = memory_factory
        self.model_client = model_client
        self.tool_client = tool_client
        self.artifact_store = artifact_store
        self.policy = policy
        self.events = events
        self.timeouts = timeouts
        self.tools_config = tools_config
        self.artifacts_config = artifacts_config

    def build(
        self,
        context: AgentExecutionContext,
        descriptor: AgentDescriptor,
        *,
        tracer: Any,
        memory_policy: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentRuntime:
        # Imported here, not at module level: these clients import the runtime, which
        # imports this module. Deferring the import is what keeps the cycle from forming.
        from universal_agent_harness.artifacts.client import ArtifactRuntime  # noqa: PLC0415
        from universal_agent_harness.models.client import InstrumentedModelClient  # noqa: PLC0415
        from universal_agent_harness.tools.client import InstrumentedToolClient  # noqa: PLC0415

        memory = self.memory_factory.create(context, tracer=tracer, policy=memory_policy)
        model = InstrumentedModelClient(
            self.model_client,
            timeout=self.timeouts.model_seconds,
            policy=self.policy,
            events=self.events,
        )
        tools = InstrumentedToolClient(
            self.tool_client,
            timeout=self.timeouts.tool_seconds,
            policy=self.policy,
            events=self.events,
            record_to_memory=self.tools_config.record_to_memory,
        )
        artifacts = ArtifactRuntime(
            self.artifact_store, inline_max_bytes=self.artifacts_config.inline_max_bytes
        )
        runtime = AgentRuntime(
            context=context,
            descriptor=descriptor,
            memory=memory,
            model=model,
            tools=tools,
            artifacts=artifacts,
            tracer=tracer,
            logger=get_logger(**context.log_fields()),
            cancellation=CancellationToken(),
            deadline=context.deadline,
            metadata=dict(metadata or {}),
        )
        model.attach(runtime)
        tools.attach(runtime)
        artifacts.attach(runtime)
        return runtime


async def _call_agent(
    agent: Callable[..., Any], request: AgentRequest, runtime: AgentRuntime
) -> Any:
    """Call the developer's agent with the arguments its signature actually asks for.

    A runtime-aware agent takes ``(input, runtime)``; an existing agent usually takes just
    its own input. Both work unchanged — that is the "minimally invasive" requirement.
    """
    arity = _arity(agent)
    if arity >= 2:
        value = agent(request.input, runtime)
    elif arity == 1:
        value = agent(request.input)
    else:
        value = agent()
    if inspect.isawaitable(value):
        return await value
    return value


_ARITY_CACHE: dict[Any, int] = {}


def _arity(fn: Callable[..., Any]) -> int:
    """How many positional arguments the target accepts (2+ means runtime-aware). Cached:
    signature inspection is not free and this runs on every execution (§64)."""
    key = getattr(fn, "__code__", None) or fn
    try:
        cached = _ARITY_CACHE.get(key)
    except TypeError:  # pragma: no cover - unhashable callables
        cached = None
        key = None
    if cached is not None:
        return cached
    target = fn
    bound_self = False
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
        # Not a callability test: a callable object's signature lives on its type's
        # ``__call__``, whose first parameter is ``self`` and is not passed by us.
        call = getattr(type(fn), "__call__", None)  # noqa: B004
        if call is not None:
            target, bound_self = call, True
    try:
        parameters = list(inspect.signature(target).parameters.values())
        if bound_self and parameters:
            parameters = parameters[1:]
        if any(p.kind is p.VAR_POSITIONAL for p in parameters):
            count = 2
        else:
            count = len(
                [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            )
    except (TypeError, ValueError):  # pragma: no cover - builtins
        count = 1
    if key is not None:
        _ARITY_CACHE[key] = count
    return count


def _seconds_until(deadline: datetime | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


def _status_for(error: AgentError) -> AgentStatus:
    if error.category is ErrorCategory.TIMEOUT:
        return AgentStatus.TIMEOUT
    if error.category is ErrorCategory.CANCELLED:
        return AgentStatus.CANCELLED
    if error.category is ErrorCategory.POLICY:
        return AgentStatus.REJECTED
    return AgentStatus.ERROR
