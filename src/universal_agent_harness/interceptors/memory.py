"""Memory around the execution (§10).

``before``: fetch the bundle and hand it to the agent on the runtime.
``after``: schedule observations — *after* the result has been produced, so the turn never
waits for consolidation, KG updates, summaries or eval processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from universal_agent_harness.contracts.artifacts import MemoryObservation
from universal_agent_harness.contracts.errors import AgentError, ErrorCategory
from universal_agent_harness.contracts.events import LifecycleEvent
from universal_agent_harness.contracts.messages import AgentRequest, AgentResult
from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.memory.writeback import WritebackQueue

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.evaluation.events import LifecycleDispatcher
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class MemoryContextInterceptor(BaseInterceptor):
    """Pre-execution retrieval."""

    name = "memory_context"
    order = Order.MEMORY_CONTEXT

    def __init__(self, events: LifecycleDispatcher | None = None) -> None:
        self.events = events

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        memory = runtime.memory
        if not memory.enabled or not memory.policy.retrieve_before:
            return request
        query = request.query
        if not query:
            return request
        bundle = await memory.retrieve(query)
        runtime.memory_context = bundle
        facts = memory.describe(bundle)
        if bundle is None and memory.last_error is not None:
            runtime.state.setdefault("warnings", []).append(
                ("MEMORY_DEGRADED", "memory retrieval failed; running without context")
            )
        if self.events is not None:
            self.events.emit(
                LifecycleEvent.CONTEXT_LOADED,
                {"context": runtime.context, "bundle": facts, "has_context": bundle is not None},
            )
        runtime.state["memory_facts"] = facts
        return request


class MemoryObservationInterceptor(BaseInterceptor):
    """Post-execution writes, off the critical path."""

    name = "memory_observation"
    order = Order.MEMORY_OBSERVATION

    def __init__(
        self, writeback: WritebackQueue | None = None, *, fail_closed: bool = False
    ) -> None:
        self.writeback = writeback or WritebackQueue()
        #: With ``fail_closed`` a failed write fails the execution; otherwise the result is
        #: returned with an explicit warning. Either way the failure is never silent (§77).
        self.fail_closed = fail_closed

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult:
        memory = runtime.memory
        if not memory.enabled or not result.status.ok:
            return result
        policy = memory.policy
        observations = list(result.memory_observations)
        request: AgentRequest | None = runtime.state.get("request")

        if policy.observe_input and request is not None and request.query:
            observations.append(
                MemoryObservation(
                    content=request.query,
                    kind="AGENT_INPUT",
                    metadata={"agent_id": runtime.agent_id},
                    idempotency_key=runtime.idempotency_key("obs", "input", request.query),
                )
            )
        if policy.observe_output:
            summary = _summarize(result)
            if summary:
                observations.append(
                    MemoryObservation(
                        content=summary,
                        kind="AGENT_RESULT",
                        metadata={"agent_id": runtime.agent_id, "status": str(result.status)},
                        idempotency_key=runtime.idempotency_key("obs", "output", summary),
                    )
                )
        if policy.observe_claims:
            for claim in result.claims:
                observations.append(
                    MemoryObservation(
                        content=claim.text,
                        kind="CLAIM",
                        hints={"memory_type": "SEMANTIC"},
                        metadata={"claim_id": claim.claim_id, "confidence": claim.confidence},
                        idempotency_key=runtime.idempotency_key("obs", "claim", claim.claim_id),
                    )
                )
        if not observations:
            return result

        if policy.writeback:
            scheduled = self.writeback.submit(
                self._write(runtime, observations, result),
                name=f"memory-writeback:{runtime.run_id}",
            )
            if scheduled is not None:
                return result
            runtime.logger.warning("memory writeback saturated; writing inline")
        try:
            await self._write(runtime, observations, result)
        except Exception as exc:
            if self.fail_closed:
                raise
            return result.add_warning(
                "MEMORY_WRITE_FAILED",
                "memory observations could not be written",
                error=str(exc),
            )
        return result

    async def _write(
        self, runtime: AgentRuntime, observations: list[MemoryObservation], result: AgentResult
    ) -> None:
        memory = runtime.memory
        policy = memory.policy
        try:
            if policy.record_messages:
                request: AgentRequest | None = runtime.state.get("request")
                if request is not None and request.query:
                    await memory.record_input(request.query)
                summary = _summarize(result)
                if summary:
                    await memory.record_output(summary)
            for observation in observations:
                await memory.observe(observation)
        except Exception as exc:
            error = AgentError.of(exc, category=ErrorCategory.MEMORY, source="memory.observe")
            runtime.logger.warning(
                "memory.writeback.failed", error_code=error.code, error_message=error.message
            )
            raise


def _summarize(result: AgentResult) -> str | None:
    """What the harness writes back as the agent's output.

    Text data is written verbatim; structured data is *not* dumped wholesale into memory —
    an agent that wants structured memory should return explicit ``memory_observations``.
    """
    if isinstance(result.data, str) and result.data.strip():
        return result.data.strip()
    if result.claims:
        return "\n".join(c.text for c in result.claims)
    return None
