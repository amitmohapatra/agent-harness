"""Memory policy (§11): what the harness reads and writes around one agent execution.

Defaults come from :class:`MemoryConfig`; an agent can narrow or widen them at wrap time
(``harness.wrap(agent, memory_policy=MemoryPolicy(observe_output=False))``) without
touching global configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from universal_agent_harness.config.settings import MemoryConfig


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    retrieve_before: bool = True
    observe_input: bool = True
    observe_output: bool = True
    observe_tool_results: bool = False
    observe_claims: bool = True
    private_by_default: bool = False
    record_messages: bool = False
    token_budget: int | None = None
    writeback: bool = True

    @classmethod
    def from_config(cls, config: MemoryConfig) -> MemoryPolicy:
        return cls(
            retrieve_before=config.retrieve_before,
            observe_input=config.observe_input,
            observe_output=config.observe_output,
            observe_tool_results=config.observe_tool_results,
            observe_claims=config.observe_claims,
            private_by_default=config.private_by_default,
            record_messages=config.record_messages,
            token_budget=config.token_budget,
            writeback=config.writeback,
        )

    def merged(self, override: MemoryPolicy | dict[str, Any] | None) -> MemoryPolicy:
        if override is None:
            return self
        if isinstance(override, MemoryPolicy):
            return override
        return replace(self, **override)

    @property
    def writes_anything(self) -> bool:
        return (
            self.observe_input
            or self.observe_output
            or self.observe_claims
            or self.record_messages
        )

    def visibility_hints(self) -> dict[str, Any]:
        """Hints attached to every observation this policy produces."""
        return {"visibility": "RUN"} if self.private_by_default else {}
