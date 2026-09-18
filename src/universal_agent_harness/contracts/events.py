"""Lifecycle events (§18) and evaluation events (§49).

Listeners observe; they never change the execution. A listener raising is logged and
swallowed — an observer must not be able to fail a business execution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from universal_agent_harness.contracts.artifacts import EvidenceRef


class LifecycleEvent(StrEnum):
    AGENT_START = "on_agent_start"
    CONTEXT_LOADED = "on_context_loaded"
    MODEL_START = "on_model_start"
    MODEL_END = "on_model_end"
    TOOL_START = "on_tool_start"
    TOOL_END = "on_tool_end"
    AGENT_SUCCESS = "on_agent_success"
    AGENT_ERROR = "on_agent_error"
    AGENT_CANCEL = "on_agent_cancel"
    AGENT_PAUSE = "on_agent_pause"
    AGENT_TIMEOUT = "on_agent_timeout"
    AGENT_FINISH = "on_agent_finish"


class AgentEvalEvent(BaseModel):
    """Emitted after every execution when evaluation events are enabled.

    Carries *references*, not payloads: an evaluator (Langfuse, DeepEval, a custom job)
    resolves them out-of-band, so enabling evaluation never widens what the harness holds
    in memory or sends over the wire.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    agent_id: str
    agent_run_id: str
    tenant_id: str
    trace_id: str | None = None
    skills: list[str] = Field(default_factory=list)
    request_ref: str | None = None
    result_ref: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    model_metadata: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "SUCCESS"
    latency_ms: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
