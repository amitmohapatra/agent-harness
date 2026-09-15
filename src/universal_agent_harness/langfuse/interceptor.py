"""The Langfuse interceptor: trace-level mapping for one agent execution (§24).

Sessions, users and tags are *trace* concepts in Langfuse, so they are set once, on the
root agent span. The mapping is the documented one: thread -> session, user -> user (only
when the capture policy allows it), agent run -> observation, skills -> tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from universal_agent_harness.config.settings import CaptureConfig
from universal_agent_harness.contracts.messages import AgentRequest, AgentResult
from universal_agent_harness.interceptors.base import BaseInterceptor, Order

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.langfuse.provider import LangfuseTelemetryProvider
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class LangfuseInterceptor(BaseInterceptor):
    name = "langfuse"
    order = Order.OBSERVABILITY

    def __init__(
        self,
        provider: LangfuseTelemetryProvider,
        capture: CaptureConfig,
        *,
        fail_closed: bool = False,
    ) -> None:
        self.provider = provider
        #: The same capture policy every other backend obeys.
        self.capture = capture
        self.fail_closed = fail_closed

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        capture = self.capture
        context = runtime.context
        tags = [f"agent:{context.agent_id}", *[f"skill:{s}" for s in runtime.descriptor.skill_ids]]
        metadata = {
            "agent_run_id": context.agent_run_id,
            "parent_agent_run_id": context.parent_agent_run_id,
            "turn_id": context.turn_id,
            "task_id": context.task_id,
            "tenant_id": context.tenant_id,
            "harness": "universal-agent-harness",
        }
        try:
            self.provider.decorate_trace(
                runtime.state["span"].span,
                trace_name=context.agent_id,
                session_id=(context.thread_id or context.session_id) if capture.thread_id else None,
                user_id=context.user_id if capture.user_id else None,
                tags=tags,
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        except Exception as exc:
            # Observability is not a business dependency (§32): in the default
            # non-blocking mode a Langfuse problem is logged and the agent still runs.
            if self.fail_closed:
                raise
            runtime.logger.warning("langfuse.decorate_failed", error=str(exc))
        return request

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult:
        return result
