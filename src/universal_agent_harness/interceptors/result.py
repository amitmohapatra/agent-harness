"""Result normalization, the artifact/size guard, and the declared-output check (§26, §43)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.artifacts import AgentWarning
from universal_agent_contracts.errors import ConfigurationError, ResultValidationError
from universal_agent_contracts.messages import AgentResponse, AgentStatus

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

    def __init__(
        self,
        *,
        offload_large_payloads: bool = True,
        output_schema: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
    ) -> None:
        self.offload_large_payloads = offload_large_payloads
        #: Resolves an agent's declared output schema, or ``None`` when it declared none.
        #: Supplied by the harness when the deployment is bound to a registry; absent
        #: otherwise, which is why an unbound deployment pays nothing for this.
        self.output_schema = output_schema

    async def after(self, result: AgentResponse, runtime: AgentRuntime) -> AgentResponse:
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

        # Checked before offloading: once an oversized payload is replaced by an artifact
        # reference there is nothing left to validate, and an agent would silently escape
        # its own contract by returning too much.
        await self._check_declared_output(result, runtime)

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

    async def _check_declared_output(self, result: AgentResponse, runtime: AgentRuntime) -> None:
        """Hold the agent to the contract it published in the registry.

        The schema describes ``AgentResponse.data``. Only a *successful* result is checked:
        a run that already failed has no data to speak of, and reporting a schema violation
        on top of the real error would bury it.

        Raises rather than warning, deliberately. The schema exists so that callers can rely
        on the shape without defending against it; a payload that quietly violates it is
        worse downstream than a loud failure here, and the harness's ``error_mode`` already
        decides whether a raised error propagates or is returned.
        """
        if self.output_schema is None or result.status != AgentStatus.SUCCESS:
            return
        schema = await self.output_schema(runtime.descriptor.agent_id)
        if not schema:
            return
        try:
            import jsonschema  # noqa: PLC0415 - only needed when a schema was declared
        except ImportError as exc:  # pragma: no cover - documented degradation
            raise ConfigurationError(
                "an agent declared an output_schema but jsonschema is not installed: "
                "pip install 'universal-agent-harness[registry]'"
            ) from exc
        try:
            jsonschema.validate(result.data, schema)
        except jsonschema.ValidationError as exc:
            raise ResultValidationError(
                f"result does not match the output_schema declared in the registry: {exc.message}",
                details={
                    "agent_id": runtime.descriptor.agent_id,
                    "path": "/".join(str(p) for p in exc.absolute_path) or "data",
                },
            ) from exc
