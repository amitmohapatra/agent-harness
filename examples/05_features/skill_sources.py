"""Skills from where a team keeps them, in each framework's own skills slot (``trellis.skills``):
a folder of ``SKILL.md`` skills on this machine combined with a store of the team's own — a
user's version of a skill wins over the tenant's — read by ``ReAct`` through Deep Agents' own
skills middleware, and by a plain function through ``list_skills``/``load_skill``.

Each run uses the skills of its own tenant, user and agent, at the versions it saw first (kept
in its journal: a resume reads the same text); an edit applies from the next run.

    python -m examples.05_features.skill_sources
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from examples._support.offline import react_model

from trellis import Harness, ReAct, Runtime, skills
from trellis.harness.repository import NotFound
from trellis.skills import Level, Scope, SkillDoc, SkillInfo

HERE = Path(__file__).resolve().parent


class TeamStore:
    """A team's own store (a database, a CMS...): the ``SkillSource`` protocol. Here a dict:
    ada's own version of ``tone``, the tenant's for everyone else."""

    label = "team-store"
    held: ClassVar[dict[tuple[str, Level, str], tuple[str, str]]] = {
        ("tone", "user", "ada"): ("Answer in three bullet points.", "1"),
        ("tone", "tenant", "default"): ("Answer in one paragraph.", "1"),
    }

    async def list(self, scope: Scope) -> list[SkillInfo]:
        mine: dict[str, str | None] = {"user": scope.user, "tenant": scope.tenant}
        return [
            SkillInfo(name, "How we write answers.", f"{level}-{version}", level=level)
            for (name, level, owner), (_, version) in self.held.items()
            if mine[level] == owner
        ]

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc:
        level, _, number = version.partition("-")
        owner = scope.user if level == "user" else scope.tenant
        body, kept = self.held[name, "user" if level == "user" else "tenant", owner or ""]
        if kept != number:
            raise NotFound(f"no {name} version {version}")
        return SkillDoc(body)

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str:
        raise NotFound("no files")


async def main() -> None:
    src = skills.combine(skills.directory(HERE / "skills"), TeamStore())
    model = react_model(
        [
            ("read_file", {"file_path": "/skills/sql-review/SKILL.md", "limit": 1000}),
            "Name the columns instead of SELECT *.",
        ]
    )
    async with Harness() as h:
        reviewer = h.wrap(ReAct("You review SQL.", model, skills=src), id="reviewer")
        result = await reviewer.run("Review: SELECT * FROM orders", user="ada")
        print("ReAct:", result.answer)

        async def writer(input: str, agent: Runtime) -> str:
            await agent.tools.use(src.tools())
            print(await agent.tools.call("list_skills"))
            return str(await agent.tools.call("load_skill", name="tone"))

        plain = h.wrap(writer, id="writer")
        for user in ("ada", "bob"):
            print(user, "->", (await plain.run("write", user=user)).answer.splitlines()[-1])


if __name__ == "__main__":
    asyncio.run(main())
