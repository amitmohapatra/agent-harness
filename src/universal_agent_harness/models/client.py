"""``InstrumentedModelClient`` — what ``runtime.model`` actually is (§15).

Around every model call it adds, in this order: policy authorization, a deadline derived
from the execution's remaining time, a ``agent.model.invoke`` span with GenAI semantic
attributes, lifecycle events, token/cost metrics, and a compact call summary kept on the
runtime for the evaluation event. Payloads are attached only when the capture policy allows.

Streaming is instrumented without buffering (§73): the wrapper times the first chunk,
counts chunks, and closes the span when the stream ends, is cancelled, or raises.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.errors import HarnessError, ModelError, PolicyDeniedError
from universal_agent_contracts.events import LifecycleEvent
from universal_agent_contracts.model import ModelRequest, ModelResponse

from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import (
    MODEL_CALLS,
    MODEL_COST,
    MODEL_LATENCY,
    MODEL_TOKENS,
    MODEL_TTFT,
)
from universal_agent_harness.telemetry.tracer import Stopwatch

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class InstrumentedModelClient:
    """Wraps a :class:`ModelClient` for one execution. Created per run; never shared."""

    def __init__(
        self,
        client: Any,
        *,
        runtime_ref: Any = None,
        timeout: float | None = None,
        policy: Any = None,
        events: Any = None,
    ) -> None:
        self._client = client
        self._runtime: AgentRuntime | None = runtime_ref
        self.timeout = timeout
        self._policy = policy
        self._events = events

    def attach(self, runtime: AgentRuntime) -> None:
        """Called once by the execution coordinator when the runtime exists."""
        self._runtime = runtime

    @property
    def inner(self) -> Any:
        """The wrapped client, for callers who need the provider object itself."""
        return self._client

    # -- calls -----------------------------------------------------------------------
    async def invoke(self, request: ModelRequest | str, /, **kwargs: Any) -> ModelResponse:
        return await self._call("invoke", request, kwargs)

    async def structured(
        self, request: ModelRequest | str, /, schema: Any, **kwargs: Any
    ) -> ModelResponse:
        return await self._call("structured", request, {"schema": schema, **kwargs})

    async def _call(self, method: str, request: ModelRequest | str, kwargs: dict[str, Any]) -> Any:
        runtime = self._runtime
        req = request if isinstance(request, ModelRequest) else ModelRequest(prompt=request)
        await self._authorize(req)
        tracer = runtime.tracer if runtime else None
        watch = Stopwatch()
        if tracer is None:  # no runtime bound: still correct, just uninstrumented
            return await self._invoke_inner(method, request, kwargs)
        self._emit(LifecycleEvent.MODEL_START, {"model": req.model, "provider": req.provider})
        with tracer.model_span(req, **{N.MODEL_STREAMING: False}) as span:
            span.set_input(req.messages or req.prompt, category="prompt")
            try:
                response = await self._bounded(self._invoke_inner(method, request, kwargs))
            except asyncio.CancelledError:
                span.error("cancelled", **{N.STATUS: "cancelled"})
                raise
            except Exception as exc:
                span.error(exc, **{N.STATUS: "error"})
                self._metrics(req, None, watch.ms, status="error")
                self._emit(LifecycleEvent.MODEL_END, {"status": "error", "error": str(exc)})
                # A harness error (configuration, policy, timeout) already says what went
                # wrong; only provider failures become MODEL errors.
                if isinstance(exc, HarnessError):
                    raise
                raise ModelError(str(exc), source=req.provider or "model") from exc
            normalized = ModelResponse.coerce(response, request=req)
            normalized = normalized.model_copy(update={"latency_ms": watch.ms})
            span.set_attributes(_response_attributes(normalized))
            span.set_output(normalized.text or normalized.data, category="model")
            span.ok()
        self._metrics(req, normalized, watch.ms, status="ok")
        self._record(req, normalized, watch.ms, status="ok")
        self._emit(
            LifecycleEvent.MODEL_END,
            {"status": "ok", "model": normalized.model, "latency_ms": watch.ms},
        )
        return normalized

    async def _invoke_inner(self, method: str, request: Any, kwargs: dict[str, Any]) -> Any:
        target = getattr(self._client, method)
        if method == "structured":
            schema = kwargs.pop("schema")
            return await target(request, schema, **kwargs)
        return await target(request, **kwargs)

    # -- streaming ---------------------------------------------------------------------
    async def stream(self, request: ModelRequest | str, /, **kwargs: Any) -> AsyncIterator[Any]:
        """Yield chunks as they arrive; record start, first token, completion, error, cancel."""
        runtime = self._runtime
        req = request if isinstance(request, ModelRequest) else ModelRequest(prompt=request)
        await self._authorize(req)
        if runtime is None:
            async for chunk in _aiter(self._client.stream(request, **kwargs)):
                yield chunk
            return
        watch = Stopwatch()
        chunks = 0
        first_token_ms: float | None = None
        self._emit(LifecycleEvent.MODEL_START, {"model": req.model, "streaming": True})
        with runtime.tracer.model_span(req, **{N.MODEL_STREAMING: True}) as span:
            span.set_input(req.messages or req.prompt, category="prompt")
            try:
                async for chunk in _aiter(self._client.stream(request, **kwargs)):
                    if first_token_ms is None:
                        first_token_ms = watch.ms
                        span.event("first_token", **{N.MODEL_TTFT_MS: first_token_ms})
                    chunks += 1
                    yield chunk
            except asyncio.CancelledError:
                span.error("cancelled", **{N.STATUS: "cancelled", "chunks": chunks})
                self._metrics(req, None, watch.ms, status="cancelled", streaming=True)
                raise
            except Exception as exc:
                span.error(exc, **{N.STATUS: "error", "chunks": chunks})
                self._metrics(req, None, watch.ms, status="error", streaming=True)
                if isinstance(exc, HarnessError):
                    raise
                raise ModelError(str(exc), source=req.provider or "model") from exc
            span.set(**{"chunks": chunks, N.MODEL_TTFT_MS: first_token_ms})
            span.ok()
        if first_token_ms is not None:
            runtime.tracer.metrics.duration(
                MODEL_TTFT, first_token_ms, model=req.model, provider=req.provider
            )
        self._metrics(req, None, watch.ms, status="ok", streaming=True)
        self._record(req, None, watch.ms, status="ok", streaming=True, chunks=chunks)
        self._emit(LifecycleEvent.MODEL_END, {"status": "ok", "streaming": True, "chunks": chunks})

    # -- plumbing -----------------------------------------------------------------------
    async def _authorize(self, request: ModelRequest) -> None:
        if self._policy is None or self._runtime is None:
            return
        decision = await self._policy.authorize_model(self._runtime.context, request)
        if decision is not True:
            raise PolicyDeniedError(
                decision if isinstance(decision, str) else "model call denied by policy",
                source="policy.model",
            )

    async def _bounded(self, awaitable: Any) -> Any:
        budget = _budget(self.timeout, self._runtime.remaining_seconds if self._runtime else None)
        if budget is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, budget)

    def _metrics(
        self,
        request: ModelRequest,
        response: ModelResponse | None,
        ms: float,
        *,
        status: str,
        streaming: bool = False,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        metrics = runtime.tracer.metrics
        labels = {
            "model": response.model if response and response.model else request.model,
            "provider": request.provider,
            "status": status,
            "streaming": streaming,
        }
        metrics.count(MODEL_CALLS, **labels)
        metrics.duration(MODEL_LATENCY, ms, **labels)
        usage = response.usage if response else None
        if usage is None:
            return
        if usage.input_tokens:
            metrics.value(MODEL_TOKENS, float(usage.input_tokens), **labels, outcome="input")
        if usage.output_tokens:
            metrics.value(MODEL_TOKENS, float(usage.output_tokens), **labels, outcome="output")
        if usage.cost_usd:
            metrics.value(MODEL_COST, float(usage.cost_usd), unit="USD", **labels)

    def _record(
        self,
        request: ModelRequest,
        response: ModelResponse | None,
        ms: float,
        *,
        status: str,
        **extra: Any,
    ) -> None:
        if self._runtime is None:
            return
        usage = response.usage if response else None
        self._runtime.record_model_call(
            {
                "model": (response.model if response else None) or request.model,
                "provider": request.provider,
                "prompt_id": request.prompt_id,
                "status": status,
                "latency_ms": round(ms, 3),
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "cost_usd": usage.cost_usd if usage else None,
                "fallback_used": response.fallback_used if response else None,
                **extra,
            }
        )

    def _emit(self, event: LifecycleEvent, payload: dict[str, Any]) -> None:
        if self._events is not None and self._runtime is not None:
            self._events.emit(event, {"context": self._runtime.context, **payload})


def _response_attributes(response: ModelResponse) -> dict[str, Any]:
    usage = response.usage
    return {
        N.MODEL_RESPONSE_MODEL: response.model,
        N.MODEL_FINISH_REASON: response.finish_reason,
        N.MODEL_FALLBACK: response.fallback_used,
        N.MODEL_INPUT_TOKENS: usage.input_tokens if usage else None,
        N.MODEL_OUTPUT_TOKENS: usage.output_tokens if usage else None,
        N.MODEL_TOTAL_TOKENS: usage.tokens if usage else None,
        N.MODEL_COST: usage.cost_usd if usage else None,
        N.DURATION_MS: response.latency_ms,
    }


def _budget(configured: float | None, remaining: float | None) -> float | None:
    """A model call never outlives the agent's deadline (§38)."""
    values = [v for v in (configured, remaining) if v is not None]
    return min(values) if values else None


def _aiter(value: Any) -> AsyncIterator[Any]:
    return value.__aiter__() if hasattr(value, "__aiter__") else value
