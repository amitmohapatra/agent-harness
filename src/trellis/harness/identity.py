"""Who a run is for: tenant, user, thread, agent and run — derived once, used everywhere."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from trellis.contracts import AgentExecutionContext


class Identity(BaseModel):
    """The scope of one run. ``user`` is required: memory, approvals and the inbox are all
    somebody's, and a run for nobody would be scoped to everybody."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str
    user: str
    agent_id: str
    run_id: str
    thread: str | None = None
    workspace: str | None = None

    def scope(self) -> dict[str, Any]:
        """The memory service scope."""
        fields = {
            "tenant_id": self.tenant,
            "user_id": self.user,
            "agent_id": self.agent_id,
            "agent_run_id": self.run_id,
            "thread_id": self.thread,
            "workspace_id": self.workspace,
        }
        return {k: v for k, v in fields.items() if v is not None}

    def context(self) -> AgentExecutionContext:
        """The contracts execution context (events, interrupts and feedback are built on it)."""
        return AgentExecutionContext.create(
            tenant_id=self.tenant,
            agent_id=self.agent_id,
            agent_run_id=self.run_id,
            user_id=self.user,
            thread_id=self.thread,
            workspace_id=self.workspace,
        )
