"""Deadline derivation (§38).

Enforcement is structural — the coordinator runs the agent inside ``asyncio.timeout`` —
because an interceptor cannot interrupt a call it merely wraps. What lives here is the
*derivation*: the effective deadline for this execution, never later than the parent's.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from universal_agent_harness.contracts.messages import AgentRequest
from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.runtime.cancellation import tightest

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class TimeoutInterceptor(BaseInterceptor):
    name = "timeout"
    order = Order.TIMEOUT

    def __init__(self, default_seconds: float | None = 30.0) -> None:
        self.default_seconds = default_seconds

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        configured = request.constraints.get("timeout_seconds", self.default_seconds)
        own = (
            datetime.now(UTC) + timedelta(seconds=float(configured))
            if configured is not None
            else None
        )
        deadline = tightest(runtime.context.deadline, own)
        runtime.deadline = deadline
        runtime.state["deadline"] = deadline
        if deadline is not None:
            runtime.state["span"].set(deadline=deadline.isoformat())
        return request
