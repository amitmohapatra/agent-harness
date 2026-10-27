"""Skills into the OpenAI Agents SDK's own slot: its ``Skills`` capability of a ``SandboxAgent``,
lazily loaded (``Skills(lazy_from=...)``). The capability is the SDK's, its instructions and
``load_skill`` tool too; what the harness adds is the source: the run's pinned skills listed,
and a skill ``load_skill`` asks for copied into the sandbox, its ``SKILL.md`` and its files
(``trellis.harness.skills``).

The SDK lists a lazy source's skills synchronously, so the capability pins them first, in its
(asynchronous) instructions: the same per-run pin as every framework's.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from agents.sandbox.capabilities.skills import LazySkillSource, SkillMetadata, Skills
from agents.sandbox.errors import SkillsConfigError
from agents.sandbox.manifest import Manifest
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.types import User
from pydantic import ConfigDict, Field

from trellis.contracts import ToolError
from trellis.harness.skills import SKILL_MD, Scope, SkillSet


class TrellisSkillSource(LazySkillSource):
    """A lazy skill source over a :class:`SkillSet`: the run's pinned skills (or
    ``scope``'s)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: the set itself: shared, never copied or dumped (the SDK dumps a capability's fields
    #: to name the agent)
    skills: SkillSet = Field(exclude=True)
    scope: Scope | None = None

    async def prepare(self) -> None:
        """Pin the skills (the run's: its journal's on a resume) before they are listed."""
        await self.skills.pinned(self.scope)

    def list_skill_metadata(
        self, *, skills_path: str, source_grants: tuple[Any, ...] = ()
    ) -> list[SkillMetadata]:
        return [
            SkillMetadata(name=c.name, description=c.description, path=Path(skills_path) / c.name)
            for c in self.skills.pinned_now(self.scope).values()
        ]

    async def load_skill(
        self,
        *,
        skill_name: str,
        session: BaseSandboxSession,
        skills_path: str,
        user: str | User | None = None,
    ) -> dict[str, str]:
        skills = self.skills
        try:
            files = {SKILL_MD: await skills.full_text(skill_name, self.scope)}
            for path in (await skills.document(skill_name, self.scope)).files:
                files[path] = await skills.read_file(skill_name, path, self.scope)
        except ToolError as exc:
            raise SkillsConfigError(message=str(exc), context={"skill_name": skill_name}) from exc
        home = Path(session.state.manifest.root) / skills_path / skill_name
        for path, text in files.items():
            await session.write(home / path, io.BytesIO(text.encode()), user=user)
        return {"status": "loaded", "skill_name": skill_name, "path": f"{skills_path}/{skill_name}"}


class TrellisSkills(Skills):
    """The SDK's ``Skills`` capability whose lazy source is pinned before its instructions
    list it."""

    async def instructions(self, manifest: Manifest) -> str | None:
        source = self.lazy_from
        assert isinstance(source, TrellisSkillSource)
        await source.prepare()
        return await super().instructions(manifest)


def capability(skills: SkillSet, scope: Scope | None) -> TrellisSkills:
    """``skills`` as the SDK's ``Skills`` capability, loaded lazily."""
    return TrellisSkills(lazy_from=TrellisSkillSource(skills=skills, scope=scope))
