"""Metrics with deliberately bounded label sets (§35).

Tenant, user, thread and run ids are *never* labels: they are unbounded and would blow up
any Prometheus-style backend. They live on spans and log lines, where high cardinality is
fine. What is labelled here is agent id, status, model, tool and outcome — all bounded by
what an application actually deploys.
"""

from __future__ import annotations

from typing import Any

# -- instrument names ---------------------------------------------------------------
AGENT_EXECUTIONS = "agent.executions.count"
AGENT_LATENCY = "agent.duration_ms"
AGENT_OVERHEAD = "agent.harness_overhead_ms"
MODEL_CALLS = "agent.model.calls.count"
MODEL_LATENCY = "agent.model.duration_ms"
MODEL_TOKENS = "agent.model.tokens.count"
MODEL_COST = "agent.model.cost.total"
MODEL_TTFT = "agent.model.time_to_first_token_ms"
TOOL_CALLS = "agent.tool.calls.count"
TOOL_LATENCY = "agent.tool.duration_ms"
MEMORY_OPERATIONS = "agent.memory.operations.count"
MEMORY_LATENCY = "agent.memory.duration_ms"
MEMORY_CONTEXT_TOKENS = "agent.memory.context_tokens"

#: The only labels the harness ever attaches to a metric.
ALLOWED_LABELS = frozenset(
    {
        "agent_id",
        "skill",
        "status",
        "error_category",
        "model",
        "provider",
        "tool",
        "operation",
        "framework",
        "outcome",
        "cached",
        "retry",
        "streaming",
    }
)


def labels(**fields: Any) -> dict[str, Any]:
    """Keep only bounded labels, drop empties. Unknown keys are dropped, not raised on:
    a metric label typo must not fail an execution."""
    return {
        k: (v if isinstance(v, bool | int | float) else str(v))
        for k, v in fields.items()
        if k in ALLOWED_LABELS and v is not None and v != ""
    }


class MetricsRecorder:
    """Thin facade over a telemetry provider so call sites read like the metric they mean."""

    __slots__ = ("_provider", "enabled")

    def __init__(self, provider: Any, *, enabled: bool = True) -> None:
        self._provider = provider
        self.enabled = enabled

    def count(self, name: str, value: float = 1.0, **label_fields: Any) -> None:
        if self.enabled:
            self._provider.record_metric(name, value, attributes=labels(**label_fields))

    def duration(self, name: str, milliseconds: float, **label_fields: Any) -> None:
        if self.enabled:
            self._provider.record_metric(
                name, milliseconds, unit="ms", attributes=labels(**label_fields)
            )

    def value(self, name: str, amount: float, *, unit: str = "", **label_fields: Any) -> None:
        if self.enabled:
            self._provider.record_metric(name, amount, unit=unit, attributes=labels(**label_fields))
