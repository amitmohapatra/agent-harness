"""Result normalization and the artifact/size guard (§26, §43)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from universal_agent_harness.contracts.artifacts import AgentWarning
from universal_agent_harness.contracts.messages import AgentResult, AgentStatus
from universal_agent_harness.interceptors.base import BaseInterceptor, Order

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class ResultValidationInterceptor(BaseInterceptor):
    """Attaches what the runtime collected and keeps oversized payloads out of the result.

    Artifacts created during the run are registered on the result even if the agent forgot
    to; warnings raised by other interceptors are merged; a data payload larger than the
    configured inline limit is moved to the artifact store and replaced by a reference.
    """

    name = "result_validation"
    order = Order.RESULT_VALIDATION

    def __init__(self, *, offload_large_payloads: bool = True) -> None:
        self.offload_large_payloads = offload_large_payloads

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult:
        updates: dict[str, Any] = {}

        created = [a for a in runtime.artifacts.created if a not in result.artifacts]
        if created:
            updates["artifacts"] = [*result.artifacts, *created]

        pending = runtime.state.get("warnings") or []
        if pending:
            updates["warnings"] = [
                *result.warnings,
                *[AgentWarning(code=c, message=m) for c, m in pending],
            ]

        metrics = dict(result.metrics)
        if runtime.model_calls:
            metrics.setdefault("model_calls", float(len(runtime.model_calls)))
            tokens = sum(
                (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
                for c in runtime.model_calls
            )
            if tokens:
                metrics.setdefault("total_tokens", float(tokens))
            cost = sum(c.get("cost_usd") or 0 for c in runtime.model_calls)
            if cost:
                metrics.setdefault("cost_usd", float(cost))
        if runtime.tool_calls:
            metrics.setdefault("tool_calls", float(len(runtime.tool_calls)))
        if metrics != result.metrics:
            updates["metrics"] = metrics

        if self.offload_large_payloads and runtime.artifacts.should_offload(result.data):
            ref = await runtime.artifacts.put(
                result.data,
                type="agent_result",
                mime_type="text/plain" if isinstance(result.data, str) else None,
                metadata={"agent_id": runtime.agent_id, "run_id": runtime.run_id},
            )
            updates["data"] = None
            updates["artifacts"] = [*updates.get("artifacts", result.artifacts), ref]
            updates["warnings"] = [
                *updates.get("warnings", result.warnings),
                AgentWarning(
                    code="RESULT_OFFLOADED",
                    message="result data exceeded the inline limit and was stored as an artifact",
                    details={"artifact_id": ref.artifact_id},
                ),
            ]

        if result.error is not None and result.status == AgentStatus.SUCCESS:
            updates["status"] = AgentStatus.ERROR

        return result.with_fields(**updates) if updates else result
