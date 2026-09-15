"""Agent and skill identity (§46). Present from V1 so a future registry has something to
register; the default registry client is a no-op."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    skill_id: str
    version: str = "1.0.0"
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)


class AgentDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    agent_id: str
    version: str = "0.1.0"
    description: str = ""
    agent_group_id: str | None = None
    skills: list[SkillDescriptor] = Field(default_factory=list)
    framework: str | None = None
    framework_version: str | None = None
    harness_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def skill_ids(self) -> list[str]:
        return [s.skill_id for s in self.skills]

    @classmethod
    def build(
        cls,
        agent_id: str,
        *,
        skills: list[str | SkillDescriptor] | None = None,
        **fields: Any,
    ) -> AgentDescriptor:
        """Accept plain skill ids (the common case) or full descriptors."""
        resolved = [SkillDescriptor(skill_id=s) if isinstance(s, str) else s for s in skills or []]
        return cls(agent_id=agent_id, skills=resolved, **fields)
