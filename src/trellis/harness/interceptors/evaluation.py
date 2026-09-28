"""Evaluation event emission after every execution (§49)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trellis.contracts.messages import AgentResponse

from trellis.harness.evaluation.events import build_eval_event
from trellis.harness.interceptors.base import BaseInterceptor, Order
from trellis.harness.memory.writeback import WritebackQueue
from trellis.harness.telemetry.sampling import roll

if TYPE_CHECKING:  # pragma: no cover
    from trellis.harness.runtime.agent_runtime import AgentRuntime


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
        event = build_eval_event(runtime, result)
        if self.synchronous:
            await self.sink.emit(event)
            return
        if self.queue.submit(self.sink.emit(event), name=f"eval:{runtime.run_id}") is None:
            runtime.logger.debug("evaluation queue saturated; dropping event")

    def _sampled(self, runtime: AgentRuntime) -> bool:
        if self.sample_rate >= 1.0:
            return True
        return bool(roll(runtime.run_id, self.sample_rate, "evaluation"))
