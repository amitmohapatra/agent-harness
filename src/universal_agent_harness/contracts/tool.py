"""Tool contracts (§14, §89). Shaped so an MCP or gateway-backed client fits without change:
identity, schema, call, result, streaming, idempotency and authorization metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from universal_agent_harness.contracts.artifacts import ArtifactRef


class ToolSpec(BaseModel):
    """What a tool is. ``server``/``source`` allow MCP servers and gateways to be identified."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    version: str = "1"
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "local"
    server: str | None = None
    idempotent: bool = False
    side_effects: str = "unknown"
    authorization: dict[str, Any] = Field(default_factory=dict)

    def descriptor(self) -> dict[str, Any]:
        """The shape the Memory Service tool APIs accept for ``available_tools``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "tags": list(self.tags),
        }


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    task: str = ""
    step: int | None = None
    idempotency_key: str | None = None


class ToolOutcome(BaseModel):
    """The normalized result of a tool call."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tool: str
    status: str = "ok"
    output: Any = None
    output_summary: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    cached: bool = False
    attempts: int = 1
    latency_ms: float | None = None
    error_class: str | None = None
    invocation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"
