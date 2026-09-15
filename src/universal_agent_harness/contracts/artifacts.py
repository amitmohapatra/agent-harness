"""References, not payloads (§43). Large data leaves the result and becomes an ``ArtifactRef``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactRef(BaseModel):
    """A pointer to content stored outside the result/state."""

    model_config = ConfigDict(frozen=True, extra="allow")

    artifact_id: str
    type: str = "blob"
    uri: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    size_bytes: int | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRef(BaseModel):
    """Where a claim came from. Mirrors the Memory Service evidence shape."""

    model_config = ConfigDict(frozen=True, extra="allow")

    source_type: str = "memory"
    source_id: str
    message_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    citation: str | None = None
    observed_at: datetime | None = None

    @property
    def evidence_id(self) -> str:
        return self.source_id


class Claim(BaseModel):
    """One assertion an agent made, with the evidence that supports it (§44)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class RecommendedAction(BaseModel):
    """A next step the agent proposes. Concise rationale only — never chain-of-thought (§45)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    action_type: str
    description: str
    reason_summary: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


#: The observation kinds the Memory Service accepts (its ``ObservationKind`` enum). Sending
#: anything else is rejected with a 422, so the harness validates before the wire rather
#: than letting a typo become a runtime failure in the writeback path.
OBSERVATION_KINDS: frozenset[str] = frozenset(
    {"MESSAGE", "FILE", "AGENT_RESULT", "TOOL_RESULT", "DECISION", "FEEDBACK", "EVENT", "IMPORT"}
)


class MemoryObservation(BaseModel):
    """Something the agent wants remembered. Written after the result is returned."""

    model_config = ConfigDict(frozen=True, extra="allow")

    content: str
    kind: str = "AGENT_RESULT"
    hints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in OBSERVATION_KINDS:
            raise ValueError(
                f"unknown observation kind {value!r}; the Memory Service accepts "
                f"{', '.join(sorted(OBSERVATION_KINDS))}"
            )
        return value


class AgentWarning(BaseModel):
    """A non-fatal problem the caller should see (degraded memory, partial tool failure...)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
