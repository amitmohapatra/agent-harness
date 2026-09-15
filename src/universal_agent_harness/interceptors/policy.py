"""Authorization before execution (§48)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from universal_agent_harness.contracts.errors import PolicyDeniedError
from universal_agent_harness.contracts.messages import AgentRequest
from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.telemetry import names as N

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class PolicyInterceptor(BaseInterceptor):
    """Denials become a ``POLICY`` error; a provider outage follows ``failure_mode``."""

    name = "policy"
    order = Order.POLICY

    def __init__(self, provider: Any, *, fail_closed: bool = False) -> None:
        self.provider = provider
        #: A provider *outage* allows by default (its denials always deny). A provider that
        #: must fail closed should raise ``PolicyDeniedError`` itself.
        self.fail_closed = fail_closed

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        with runtime.tracer.span(N.POLICY_CHECK, category="agent") as span:
            try:
                decision = await self.provider.authorize_execution(request)
            except PolicyDeniedError:
                raise
            except Exception as exc:
                span.error(exc)
                if self.fail_closed:
                    raise PolicyDeniedError(
                        f"policy provider unavailable: {exc}", source="policy"
                    ) from exc
                runtime.logger.warning("policy provider unavailable; allowing", error=str(exc))
                return request
            if decision is not True:
                reason = decision if isinstance(decision, str) else "execution denied by policy"
                span.set(**{N.STATUS: "denied"})
                raise PolicyDeniedError(reason, source="policy.execution")
            span.ok()
        return request
