"""Skill sources (``trellis.harness.skills``): where a team keeps its skills, fed to each
framework's own skills slot.

    from trellis import skills

    src = skills.combine(skills.directory("skills"), skills.bifrost(), my_store)
    agent = h.wrap(ReAct("You review SQL.", model, skills=src), id="reviewer")
"""

from __future__ import annotations

from trellis.harness.skills import (
    Level,
    Scope,
    SkillDoc,
    SkillInfo,
    SkillSet,
    SkillSource,
    bifrost,
    combine,
    directory,
    langfuse,
)

__all__ = [
    "Level",
    "Scope",
    "SkillDoc",
    "SkillInfo",
    "SkillSet",
    "SkillSource",
    "bifrost",
    "combine",
    "directory",
    "langfuse",
]
