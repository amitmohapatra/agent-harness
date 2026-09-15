"""Binding the Memory Service SDK to one agent execution.

``MemoryClient`` is process-wide; a ``MemoryContext`` is per execution and immutable. This
module is the only place the harness touches the SDK's construction API, so swapping in a
different memory backend means implementing ``MemoryPort`` and passing it in — no core
changes.
"""

from __future__ import annotations

from typing import Any

from universal_agent_harness.config.settings import MemoryConfig
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.errors import ConfigurationError
from universal_agent_harness.memory.policy import MemoryPolicy
from universal_agent_harness.memory.runtime import MemoryRuntime, NoOpMemoryRuntime
from universal_agent_harness.telemetry.tracer import HarnessTracer


class MemoryFactory:
    """Creates a :class:`MemoryRuntime` per execution from a shared client."""

    def __init__(self, client: Any | None, config: MemoryConfig) -> None:
        self.client = client
        self.config = config
        self.default_policy = MemoryPolicy.from_config(config)

    @property
    def enabled(self) -> bool:
        return bool(self.client) and self.config.enabled

    def create(
        self,
        context: AgentExecutionContext,
        *,
        tracer: HarnessTracer,
        policy: MemoryPolicy | dict[str, Any] | None = None,
    ) -> Any:
        resolved = self.default_policy.merged(policy)
        if not self.enabled:
            return NoOpMemoryRuntime(context)
        memory_context = self.bind(context)
        return MemoryRuntime(
            memory_context,
            context=context,
            policy=resolved,
            tracer=tracer,
            retrieval_timeout=self.config.retrieval_timeout_seconds,
            observation_timeout=self.config.observation_timeout_seconds,
            fail_closed=self.config.failure_mode == "fail_closed",
        )

    def bind(self, context: AgentExecutionContext) -> Any:
        """Bind the SDK context for this execution's scope.

        A caller may pass an object that is already a bound ``MemoryContext`` (it has
        ``derive``): the harness then derives from it so application-level scope fields the
        harness does not know about (work ids, custom metadata) are preserved.
        """
        if self.client is None:
            raise ConfigurationError("no memory client is configured")
        scope = context.scope_fields()
        if hasattr(self.client, "derive") and hasattr(self.client, "scope"):
            return self.client.derive(**scope)
        return self.client.bind(**scope)
