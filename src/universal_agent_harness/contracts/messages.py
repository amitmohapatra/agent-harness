"""``AgentRequest`` / ``AgentResult`` (§6, §7): the harness's stable, serializable boundary.

Both are plain Pydantic models with no framework types in them, so they can be persisted,
queued, or (later) mapped onto A2A without touching agent code (§90).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from universal_agent_harness.contracts.artifacts import (
    AgentWarning,
    ArtifactRef,
    Claim,
    EvidenceRef,
    MemoryObservation,
    RecommendedAction,
)
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.errors import AgentError


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    #: suspended waiting for something outside the run — a human, typically — and expected
    #: to be resumed. Not a failure, and not a finished turn either.
    PAUSED = "PAUSED"

    @property
    def ok(self) -> bool:
        return self in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)


class AgentRequest(BaseModel):
    """What an agent was asked to do."""

    model_config = ConfigDict(frozen=True, extra="allow")

    request_id: str
    objective: str | None = None
    input: Any = None

    context: AgentExecutionContext

    skills_requested: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, context: AgentExecutionContext, input: Any = None, **fields: Any) -> Self:
        return cls(request_id=context.request_id, context=context, input=input, **fields)

    def with_fields(self, **changes: Any) -> Self:
        """Interceptors return modified requests through this (the model is frozen)."""
        return self.model_copy(update=changes)

    @property
    def query(self) -> str | None:
        """The text a memory retrieval should use: the objective, else a string input."""
        if self.objective:
            return self.objective
        if isinstance(self.input, str):
            return self.input
        if isinstance(self.input, dict):
            for key in ("query", "question", "objective", "prompt", "input", "text"):
                value = self.input.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None


class AgentResult(BaseModel):
    """What an agent produced. Adapters map this onto framework state — never the reverse."""

    model_config = ConfigDict(extra="allow")

    status: AgentStatus = AgentStatus.SUCCESS

    data: Any = None

    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    memory_observations: list[MemoryObservation] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)

    confidence: float | None = None
    warnings: list[AgentWarning] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    error: AgentError | None = None

    @classmethod
    def ok(cls, data: Any = None, **fields: Any) -> Self:
        return cls(status=AgentStatus.SUCCESS, data=data, **fields)

    @classmethod
    def failed(
        cls, error: AgentError, *, status: AgentStatus = AgentStatus.ERROR, **fields: Any
    ) -> Self:
        return cls(status=status, error=error, **fields)

    @classmethod
    def coerce(cls, value: Any) -> Self:
        """Normalize whatever a developer's agent returned into an ``AgentResult`` (§26).

        An ``AgentResult`` passes through; anything else becomes ``data``. This is what lets
        an existing agent be wrapped without changing its return type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, AgentResult):  # subclass/superclass mix
            return cls.model_validate(value.model_dump())
        return cls(status=AgentStatus.SUCCESS, data=value)

    @property
    def succeeded(self) -> bool:
        return self.status.ok and self.error is None

    def with_fields(self, **changes: Any) -> Self:
        return self.model_copy(update=changes)

    def add_warning(self, code: str, message: str, **details: Any) -> Self:
        return self.model_copy(
            update={
                "warnings": [
                    *self.warnings,
                    AgentWarning(code=code, message=message, details=details),
                ]
            }
        )
