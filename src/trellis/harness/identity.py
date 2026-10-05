"""Who a run is for: tenant, user, thread, agent and run — derived once, used everywhere.

What leaves the process for a run says who it is for in one trusted header,
:data:`IDENTITY_HEADER` (:func:`identity_headers`): the calls a run makes to other agents
(A2A), and its MCP tool calls through the gateway, which forwards it to the MCP servers whose
``allowed_extra_headers`` name it.
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from trellis.contracts import AgentExecutionContext

#: The trusted identity header (lower case: the A2A SDK hands a server lower-cased headers).
IDENTITY_HEADER: Final = "x-trellis-identity"


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


def identity_headers(tenant: str, user: str) -> dict[str, str]:
    """The outbound trusted identity, for a call this process makes for a run."""
    fields = {"tenant_id": tenant, "user_id": user}
    return {IDENTITY_HEADER: json.dumps(fields, separators=(",", ":"), sort_keys=True)}
