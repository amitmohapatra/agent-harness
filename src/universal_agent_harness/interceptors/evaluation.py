"""Evaluation event emission after every execution (§49)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.events import AgentEvalEvent
from universal_agent_contracts.messages import AgentRequest, AgentResponse

from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.memory.writeback import WritebackQueue
from universal_agent_harness.telemetry.sampling import roll

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class EvaluationEventInterceptor(BaseInterceptor):
    """Builds the event from references only and hands it to the sink asynchronously."""

    name = "evaluation"
    order = Order.EVALUATION

    def __init__(
        self,
        sink: Any,
        *,
        synchronous: bool = False,
        sample_rate: float = 1.0,
        queue: WritebackQueue | None = None,
    ) -> None:
        self.sink = sink
        self.synchronous = synchronous
        self.sample_rate = sample_rate
        self.queue = queue or WritebackQueue()

    async def after(self, result: AgentResponse, runtime: AgentRuntime) -> AgentResponse:
        await self._emit(runtime, result)
        return result

    async def _emit(self, runtime: AgentRuntime, result: AgentResponse) -> None:
        if not self._sampled(runtime):
            return
        request: AgentRequest | None = runtime.state.get("request")
        started = runtime.state.get("started_at")
        event = AgentEvalEvent(
            agent_id=runtime.agent_id,
            agent_run_id=runtime.run_id,
            tenant_id=runtime.context.tenant_id,
            trace_id=runtime.context.trace_id,
            skills=list(
                runtime.descriptor.skill_ids or (request.skills_requested if request else [])
            ),
            request_ref=runtime.context.request_id,
            result_ref=runtime.idempotency_key("result"),
            evidence_refs=list(result.evidence),
            model_metadata=list(runtime.model_calls),
            tool_calls=list(runtime.tool_calls),
            status=str(result.status),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3) if started else 0.0,
            metrics=dict(result.metrics),
            metadata={"bundle": runtime.state.get("memory_facts") or {}},
        )
        if self.synchronous:
            await self.sink.emit(event)
            return
        if self.queue.submit(self.sink.emit(event), name=f"eval:{runtime.run_id}") is None:
            runtime.logger.debug("evaluation queue saturated; dropping event")

    def _sampled(self, runtime: AgentRuntime) -> bool:
        if self.sample_rate >= 1.0:
            return True
        return bool(roll(runtime.run_id, self.sample_rate, "evaluation"))
