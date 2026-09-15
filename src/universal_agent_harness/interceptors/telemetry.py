"""Span enrichment and execution metrics.

The ``agent.run`` span is opened by the coordinator (so everything else nests inside it);
this interceptor decorates it with request and result facts, and records the execution
counters and latencies.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from universal_agent_harness.contracts.errors import AgentError
from universal_agent_harness.contracts.messages import AgentRequest, AgentResult
from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import AGENT_EXECUTIONS, AGENT_LATENCY

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class TelemetryInterceptor(BaseInterceptor):
    name = "telemetry"
    order = Order.TELEMETRY

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        span = runtime.state["span"]
        span.set_input(request.input if request.input is not None else request.objective)
        span.set(**{"objective": request.objective} if request.objective else {})
        runtime.state["started_at"] = time.perf_counter()
        return request

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult:
        span = runtime.state["span"]
        elapsed = _elapsed_ms(runtime)
        span.set(
            **{
                N.STATUS: str(result.status),
                N.DURATION_MS: round(elapsed, 3),
                "result.claims": len(result.claims) or None,
                "result.artifacts": len(result.artifacts) or None,
                "result.warnings": len(result.warnings) or None,
                "result.confidence": result.confidence,
                "model.calls": len(runtime.model_calls) or None,
                "tool.calls": len(runtime.tool_calls) or None,
            }
        )
        span.set_output(result.data)
        if result.succeeded:
            span.ok()
        runtime.tracer.metrics.count(
            AGENT_EXECUTIONS,
            agent_id=runtime.agent_id,
            status=str(result.status),
            framework=runtime.descriptor.framework,
        )
        runtime.tracer.metrics.duration(
            AGENT_LATENCY, elapsed, agent_id=runtime.agent_id, status=str(result.status)
        )
        runtime.logger.info(
            "agent.finish",
            event_name="agent.finish",
            status=str(result.status),
            duration_ms=round(elapsed, 3),
            model_calls=len(runtime.model_calls),
            tool_calls=len(runtime.tool_calls),
        )
        return result

    async def on_error(self, error: AgentError, runtime: AgentRuntime) -> None:
        span = runtime.state["span"]
        span.error(
            error.message,
            **{
                N.ERROR_CODE: error.code,
                N.ERROR_CATEGORY: str(error.category),
                N.STATUS: "error",
                N.DURATION_MS: round(_elapsed_ms(runtime), 3),
            },
        )
        runtime.tracer.metrics.count(
            AGENT_EXECUTIONS,
            agent_id=runtime.agent_id,
            status="ERROR",
            error_category=str(error.category),
        )
        runtime.logger.error(
            "agent.error",
            event_name="agent.error",
            error_code=error.code,
            error_category=str(error.category),
            error_message=error.message,
            retryable=error.retryable,
        )


def _elapsed_ms(runtime: AgentRuntime) -> float:
    started = runtime.state.get("started_at")
    return (time.perf_counter() - started) * 1000.0 if started else 0.0
