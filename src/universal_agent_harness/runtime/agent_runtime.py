"""``AgentRuntime`` (§9): everything a runtime-aware agent is handed.

A Level-2 agent receives one of these as its second argument. Using it is what turns
"wrapped" into "instrumented": ``runtime.model``/``runtime.tools``/``runtime.memory`` are
already traced, metered, policy-checked, deadline-bounded and idempotent. An agent that
bypasses them still gets execution-level instrumentation — the harness is explicit that it
cannot instrument calls it never sees (§13).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.descriptors import AgentDescriptor
from universal_agent_harness.runtime.cancellation import CancellationToken, remaining_seconds
from universal_agent_harness.runtime.logging import HarnessLogger, get_logger
from universal_agent_harness.telemetry.tracer import HarnessTracer

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.artifacts.client import ArtifactRuntime
    from universal_agent_harness.memory.runtime import MemoryRuntime
    from universal_agent_harness.models.client import InstrumentedModelClient
    from universal_agent_harness.tools.client import InstrumentedToolClient


@dataclass(slots=True)
class AgentRuntime:
    """Per-execution runtime. Created by the harness; never constructed by application code.

    Mutable only in the fields the harness itself fills in during execution
    (``memory_context``, the recorded model/tool calls); the *context* stays immutable.
    """

    context: AgentExecutionContext
    descriptor: AgentDescriptor

    memory: MemoryRuntime
    model: InstrumentedModelClient
    tools: InstrumentedToolClient
    artifacts: ArtifactRuntime

    tracer: HarnessTracer
    logger: HarnessLogger
    cancellation: CancellationToken

    deadline: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    #: The bundle fetched by the memory interceptor before the agent ran (§10).
    memory_context: Any = None
    #: Model/tool call summaries collected for evaluation events and the result metrics.
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Free-form scratch space for interceptors to pass values to each other.
    state: dict[str, Any] = field(default_factory=dict)

    # -- conveniences ----------------------------------------------------------------
    @property
    def agent_id(self) -> str:
        return self.context.agent_id

    @property
    def run_id(self) -> str:
        return self.context.agent_run_id

    @property
    def remaining_seconds(self) -> float | None:
        return remaining_seconds(self.deadline)

    @property
    def cancelled(self) -> bool:
        return self.cancellation.cancelled

    def check_cancelled(self) -> None:
        """Cooperative cancellation point for long agents."""
        self.cancellation.raise_if_cancelled()

    def idempotency_key(self, *parts: object) -> str:
        """A key stable across retries of this execution (§42)."""
        return self.context.idempotency_key(*parts)

    def log(self, event: str, **fields: Any) -> None:
        self.logger.info(event, **fields)

    def record_model_call(self, summary: Mapping[str, Any]) -> None:
        self.model_calls.append(dict(summary))

    def record_tool_call(self, summary: Mapping[str, Any]) -> None:
        self.tool_calls.append(dict(summary))

    def child_context(self, agent_id: str, **fields: Any) -> AgentExecutionContext:
        """Context for a nested agent run started from inside this one."""
        return self.context.for_agent(agent_id, deadline=self.deadline, **fields)


def bare_logger(context: AgentExecutionContext) -> HarnessLogger:
    return get_logger(**context.log_fields())
