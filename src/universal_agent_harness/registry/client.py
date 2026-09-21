"""Agent registry hook (§47). No registry service is built here — only the seam for one."""

from __future__ import annotations

from universal_agent_contracts.descriptors import AgentDescriptor

from universal_agent_harness.runtime.logging import get_logger

log = get_logger("universal_agent_harness.registry")


class NoOpAgentRegistry:
    """The default. Records nothing, costs nothing, keeps the contract honest."""

    name = "noop"

    async def register(self, descriptor: AgentDescriptor) -> None:
        return None

    async def heartbeat(self, descriptor: AgentDescriptor, *, status: str = "healthy") -> None:
        return None


class InMemoryAgentRegistry:
    """Keeps descriptors in-process: useful for tests and for exposing "what is deployed
    here" without a service."""

    name = "memory"

    def __init__(self) -> None:
        self.agents: dict[str, AgentDescriptor] = {}
        self.heartbeats: dict[str, str] = {}

    async def register(self, descriptor: AgentDescriptor) -> None:
        self.agents[descriptor.agent_id] = descriptor

    async def heartbeat(self, descriptor: AgentDescriptor, *, status: str = "healthy") -> None:
        self.heartbeats[descriptor.agent_id] = status

    def get(self, agent_id: str) -> AgentDescriptor | None:
        return self.agents.get(agent_id)
