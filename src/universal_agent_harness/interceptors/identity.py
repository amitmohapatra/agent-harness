"""Identity: make sure every execution is attributable before anything else happens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from universal_agent_contracts.messages import AgentRequest

from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.telemetry import names as N

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class IdentityInterceptor(BaseInterceptor):
    """Stamps the descriptor's identity onto the span and the log context.

    The context itself is built before the pipeline runs (it is what the pipeline is *for*),
    so this interceptor's job is the metadata around it: agent version, framework and
    harness versions, requested skills.
    """

    name = "identity"
    order = Order.IDENTITY

    def __init__(self, harness_version: str, framework: str | None = None) -> None:
        self.harness_version = harness_version
        self.framework = framework

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        descriptor = runtime.descriptor
        runtime.state["span"].set(
            **{
                N.AGENT_VERSION: descriptor.version,
                N.AGENT_FRAMEWORK: descriptor.framework or self.framework,
                N.AGENT_FRAMEWORK_VERSION: descriptor.framework_version,
                N.HARNESS_VERSION: self.harness_version,
                N.AGENT_SKILL: request.skills_requested or descriptor.skill_ids or None,
            }
        )
        runtime.logger.debug(
            "agent.start",
            event_name="agent.start",
            agent_version=descriptor.version,
            framework=descriptor.framework or self.framework,
        )
        return request
