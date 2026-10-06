"""Agent Skills, from wherever a team keeps them, for any framework: ``skills(...)``.

    h = Harness(skills=[skills_dir("skills")])
    agent = h.wrap(graph_or_agent, id="analyst",
                   skills=["sql-review", "refunds@1.2.0", Skill("tone", "How we write.", body)])
    tools = await h.tools(skills("sql-review"), framework="langgraph")  # a graph binds its own

A source answers ``resolve(name, version) -> ResolvedSkill`` (:class:`SkillSource`): the
skill's name, version, description, ``SKILL.md`` body and the paths of its other files, which
``read(path)`` reads. ``version=None`` is the version the source serves now. Shipped:

* code — :class:`Skill` (``Skill("tone", "How we write.", body, files={"words.md": "..."})``);
* files — :func:`skills_dir`: the Agent Skills layout, ``<name>/SKILL.md`` with YAML front
  matter (``name``, ``description``; a ``version``, else the folder's content digest) and any
  other file of the folder readable by its relative path (a path out of the folder is refused);
* Bifrost — :class:`BifrostSkills`, the gateway's Skills Repository (``BIFROST_URL``).

Where a name is looked up (:class:`SkillSources`): a :class:`Skill` given is its own; a name is
asked of the sources the code passes (``Harness(skills=[...])``), then ``SKILLS_DIR``, then
Bifrost (``BIFROST_URL`` set) — the first that has it answers. Skills of every source mix in
one run: one context section, the same two tools.

Progressive disclosure: at the start of a run each skill is pinned — the version named
(``name@version``), else the version its source serves — and its name and description go
into the context pushed into the framework's input (a ``## Skills`` section, after the memory
context); two read-only tools read the rest when the model wants it: :data:`LOAD_SKILL`
(``SKILL.md``'s body and the skill's file list, as the pinned version reads) and
:data:`READ_SKILL_FILE` (one file). Both are harness tools: every call goes through the bridge
(journaled, so a resumed run reads the same text), and what was pinned is journaled too —
each skill's version, description, body and file list — so a resumed run keeps them, whatever
its source holds by then; the run says which versions it used (a ``skills`` event, and the
``trellis.skills`` attribute of its span).

A source serves a file only of the version it holds now: a file of another pinned version is
refused with a message saying so (its body still loads). A skill no source can give — its
source is down and never read it; there is no such skill — is a ``skills_unavailable``
warning, and the run goes on without it (a remote source's last copy stands while it is down).

Code that is not wrapped (Way 2) pins the same way: :meth:`SkillSources.pin` gives a
:class:`PinnedSkills` — the section for its prompt, ``load`` and ``read`` for its own tools,
and ``record()`` to keep with its checkpoint (``pin(..., recorded=)`` restores it).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.harness import telemetry
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.journal import content_key
from trellis.harness.repository import (
    BIFROST,
    PINNED,
    Chain,
    NotFound,
    digest,
    front_matter,
    inside,
    pinned,
)
from trellis.harness.runtime import Runtime, current
from trellis.harness.settings import Settings
from trellis.harness.tools.base import Tool

#: The tools, and the event a run's pinned versions are said with.
LOAD_SKILL: Final = "load_skill"
READ_SKILL_FILE: Final = "read_skill_file"
EVENT: Final = "skills"
#: What the section the skills go into the context with begins with.
SECTION: Final = (
    "## Skills\nInstructions for kinds of task. When one fits the task, read it first with "
    f"{LOAD_SKILL}(name); {READ_SKILL_FILE}(name, path) reads a file it lists."
)
#: An Agent Skill's instructions file, in its folder.
SKILL_MD: Final = "SKILL.md"


class SkillSource(Protocol):
    """Where skills come from: ``resolve`` raises a ``LookupError`` for a name — or a version —
    it does not have; ``read`` reads a file of a version it gave."""

    @property
    def label(self) -> str: ...

    async def resolve(self, name: str, version: str | None) -> ResolvedSkill: ...

    async def read(self, name: str, version: str, path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ResolvedSkill:
    """One version of a skill, as its source gave it."""

    name: str
    version: str
    description: str
    #: ``SKILL.md``'s body
    body: str
    #: the paths of its other files
    files: tuple[str, ...] = ()
    #: which source it came from (its ``label``)
    source: str = ""
    #: the source its files are read from (``None``: no source with that label now)
    origin: SkillSource | None = field(default=None, compare=False, repr=False)

    async def read(self, path: str) -> str:
        """One of its files, as its source holds it for this version: ``ToolError`` for a path
        it does not list, or a version its source no longer serves files of."""
        if path not in self.files:
            raise ToolError(
                f"skill {self.name} {self.version} has no file {path!r}", source="tools"
            )
        if self.origin is None:
            raise ToolError(
                f"{path} of skill {self.name} cannot be read: no source is {self.source} now",
                source="tools",
            )
        return await self.origin.read(self.name, self.version, path)

    def record(self) -> dict[str, Any]:
        """What the journal keeps of it."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "body": self.body,
            "files": list(self.files),
            "source": self.source,
        }


def _changed(name: str, version: str, path: str, holder: str, now: str | None) -> ToolError:
    return ToolError(
        f"{path} of skill {name} cannot be read: this run uses version {version}, and "
        f"{holder} has {now or 'none'} now",
        source="tools",
    )


# --------------------------------------------------------------------------- code


@dataclass(frozen=True, slots=True)
class Skill:
    """A skill in code: its description, its ``SKILL.md`` body and its files (path to text).
    It is its own source: given where skills are named (``skills=[Skill(...)]``), or in
    ``Harness(skills=[...])``."""

    name: str
    description: str
    body: str
    files: Mapping[str, str] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or PINNED in self.name or not self.version:
            raise ConfigurationError(f"Skill({self.name!r}): a name without {PINNED}, a version")

    @property
    def label(self) -> str:
        return "code"

    async def resolve(self, name: str, version: str | None) -> ResolvedSkill:
        if name != self.name or version not in (None, self.version):
            raise NotFound(f"holds {self.name}{PINNED}{self.version} only")
        files = tuple(sorted(self.files))
        return ResolvedSkill(
            self.name, self.version, self.description, self.body, files, self.label, self
        )

    async def read(self, name: str, version: str, path: str) -> str:
        if version != self.version:
            raise _changed(name, version, path, "the code", self.version)
        return self.files[path]


# --------------------------------------------------------------------------- files


class SkillsDir:
    """A folder of Agent Skills (:func:`skills_dir`), read at each resolve: what is on disk
    now."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise ConfigurationError(f"skills_dir({str(self.root)!r}): no such folder")

    @property
    def label(self) -> str:
        return f"skills_dir({self.root})"

    async def resolve(self, name: str, version: str | None) -> ResolvedSkill:
        found = await asyncio.to_thread(self._resolve, name)
        if version not in (None, found.version):
            raise NotFound(f"{name} is version {found.version}, not {version}")
        return found

    async def read(self, name: str, version: str, path: str) -> str:
        return await asyncio.to_thread(self._read, name, version, path)

    def _resolve(self, name: str) -> ResolvedSkill:
        folder = inside(self.root, name)
        if folder is None or "/" in name or "\\" in name:
            raise NotFound(f"{name!r} is not a folder name inside it (refused)")
        if not (folder / SKILL_MD).is_file():
            raise NotFound(f"it has no {name}/{SKILL_MD}")
        listed = (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        files = sorted(p for p in listed if p != SKILL_MD and inside(folder, p) is not None)
        raw = (folder / SKILL_MD).read_bytes()
        where = str(folder / SKILL_MD)
        meta, body = front_matter(raw.decode("utf-8", errors="replace"), where=where)
        if meta.get("name") != name:
            raise ConfigurationError(f"{where}: its front matter's name must be {name!r}")
        description = meta.get("description")
        if not isinstance(description, str) or not description:
            raise ConfigurationError(f"{where}: its front matter has no description")
        extra = meta.get("metadata")
        version = meta.get("version") or (extra.get("version") if isinstance(extra, dict) else None)
        if not version:
            version = digest(raw, *(p.encode() + (folder / p).read_bytes() for p in files))
        return ResolvedSkill(
            name, str(version), description, body.strip(), tuple(files), self.label, self
        )

    def _read(self, name: str, version: str, path: str) -> str:
        try:
            now: ResolvedSkill | None = self._resolve(name)
        except NotFound:
            now = None
        if now is None or now.version != version:
            raise _changed(name, version, path, self.label, now.version if now else None)
        file = inside(self.root / name, path)
        if file is None or path not in now.files:
            raise ToolError(f"skill {name} {version} has no file {path!r}", source="tools")
        return file.read_text("utf-8", errors="replace")


def skills_dir(path: str | os.PathLike[str]) -> SkillsDir:
    """A folder of Agent Skills: ``<name>/SKILL.md`` each, with YAML front matter (``name``,
    the folder's; ``description``; a ``version`` or ``metadata.version``, else the folder's
    content digest), and any other file of the folder read by its relative path. A path out
    of the folder (``..``, an absolute path, a link elsewhere) is refused."""
    return SkillsDir(path)


# --------------------------------------------------------------------------- Bifrost


class BifrostSkills:
    """The gateway's Skills Repository: a version once read is kept for good, the served one
    read again every ``repository.TTL_SECONDS`` (``Gateway.skill``). The gateway serves files
    only of the version it serves now."""

    @property
    def label(self) -> str:
        return BIFROST

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway

    async def resolve(self, name: str, version: str | None) -> ResolvedSkill:
        skill = await self.gateway.skill(name, version)
        files = tuple(f.path for f in skill.files)
        return ResolvedSkill(
            skill.name, skill.version, skill.description, skill.body, files, self.label, self
        )

    async def read(self, name: str, version: str, path: str) -> str:
        served = await self.gateway.served(name)
        if served != version:
            raise ToolError(
                f"{path} of skill {name} cannot be read: this run uses version {version}, and "
                f"the gateway serves files only of the version it serves ({served or 'none'})",
                source="tools",
            )
        return (await self.gateway.skill_file(name, path)).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- pinned


@dataclass(slots=True)
class PinnedSkills:
    """The skills one run uses, as pinned at its start; what could not be pinned, and why."""

    skills: dict[str, ResolvedSkill] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)

    @property
    def section(self) -> str:
        """The context section that discloses them (empty: none)."""
        if not self.skills:
            return ""
        lines = [f"- {name}: {skill.description}" for name, skill in self.skills.items()]
        return "\n".join([SECTION, *lines])

    @property
    def versions(self) -> dict[str, str]:
        return {name: skill.version for name, skill in self.skills.items()}

    def record(self) -> dict[str, Any]:
        """What to keep with the run's checkpoint: ``pin(..., recorded=)`` restores it."""
        return {name: skill.record() for name, skill in self.skills.items()}

    async def load(self, name: str) -> str:
        """``SKILL.md``'s body and the skill's file list (:data:`LOAD_SKILL`)."""
        skill = self._skill(name)
        files = [f"- {f}" for f in skill.files]
        listed = f"\n\nFiles ({READ_SKILL_FILE}):\n" + "\n".join(files) if files else ""
        return f"# {skill.name} (version {skill.version})\n\n{skill.body}{listed}"

    async def read(self, name: str, path: str) -> str:
        """One file of the skill (:data:`READ_SKILL_FILE`)."""
        return await self._skill(name).read(path)

    def _skill(self, name: str) -> ResolvedSkill:
        found = self.skills.get(name)
        if found is None:
            known = ", ".join(self.skills) or "none"
            raise ToolError(f"no skill {name!r} in this run (its skills: {known})", source="tools")
        return found


Ref = tuple[str, str | None, Skill | None]


def refs_of(refs: Sequence[str | Skill]) -> list[Ref]:
    """Each skill named — ``"name"``, ``"name@version"``, a :class:`Skill` — once."""
    parsed: list[Ref] = [
        (r.name, r.version, r) if isinstance(r, Skill) else (*pinned(r), None) for r in refs
    ]
    names = [name for name, _, _ in parsed]
    if not names or len(set(names)) < len(names):
        raise ConfigurationError("skills(...) names each skill once, at least one")
    return parsed


class SkillSources(Chain[ResolvedSkill]):
    """The skill sources, in the order they are asked: the code's, then ``SKILLS_DIR`` and
    Bifrost as the deployment names them (:meth:`of`)."""

    kind = "skill"
    hint = "pass Harness(skills=[...]) or Skill objects, or set SKILLS_DIR or BIFROST_URL"

    def __init__(self, sources: Sequence[SkillSource] = ()) -> None:
        super().__init__(sources)

    @classmethod
    def of(
        cls,
        settings: Settings,
        *,
        gateway: Gateway | None = None,
        given: Sequence[SkillSource] = (),
    ) -> SkillSources:
        """``given``, then the sources ``settings`` name: ``skills_dir``, the gateway."""
        sources = list(given)
        if settings.skills_dir:
            sources.append(skills_dir(settings.skills_dir))
        if gateway is not None:
            sources.append(BifrostSkills(gateway))
        return cls(sources)

    @classmethod
    def given(cls, sources: SkillSources | Sequence[SkillSource]) -> SkillSources:
        """The sources a block was given, as they are (a chain, or the sources in order)."""
        if isinstance(sources, SkillSources):
            return sources
        return cls(sources)

    async def pin(
        self, refs: Sequence[str | Skill], *, recorded: Mapping[str, Any] | None = None
    ) -> PinnedSkills:
        """The skills ``refs`` name, pinned: as ``recorded`` (an earlier pin's ``record()``)
        says, else as their sources give them now. A skill that cannot be pinned is one of the
        result's ``problems``, not an error; a name with no source at all to ask is a
        ``ConfigurationError``."""
        parsed = refs_of(refs)
        kept = dict(recorded or {})
        asked = [n for n, _, own in parsed if own is None and not isinstance(kept.get(n), Mapping)]
        if asked and not self.sources:
            raise ConfigurationError(f"no skill source: {self.hint}")
        found = PinnedSkills()
        for name, version, own in parsed:
            entry = kept.get(name)
            try:
                if isinstance(entry, Mapping):
                    skill = self._restored(entry, own)
                elif own is not None:
                    skill = await own.resolve(name, version)
                else:  # a version an earlier harness journaled, or none: resolve it now
                    skill = await self.find(name, entry if isinstance(entry, str) else version)
            except Exception as exc:
                found.problems[name] = str(exc)
                continue
            found.skills[name] = skill
        return found

    def _restored(self, kept: Mapping[str, Any], own: Skill | None) -> ResolvedSkill:
        """A skill as the journal kept it, its files read from the source it came from."""
        candidates: list[Any] = [own, *self.sources]
        origin = next(
            (
                s
                for s in candidates
                if s is not None
                and s.label == kept["source"]
                and getattr(s, "name", kept["name"]) == kept["name"]
            ),
            None,
        )
        return ResolvedSkill(
            name=kept["name"],
            version=kept["version"],
            description=kept["description"],
            body=kept["body"],
            files=tuple(kept["files"]),
            source=kept["source"],
            origin=origin,
        )


# --------------------------------------------------------------------------- in a run


class Skills:
    """The skills an agent uses, as a tool source (``tools=[...]``, ``h.tools(...)``): the
    two tools; the pinning and the context section are the run's (:func:`pin`)."""

    def __init__(self, refs: Sequence[str | Skill]) -> None:
        self.refs: list[str | Skill] = list(refs)
        parsed = refs_of(self.refs)
        self.names = [name for name, _, _ in parsed]
        #: what the run's pin is journaled under
        self.key = content_key(EVENT, [[name, version] for name, version, _ in parsed])

    async def resolve(self) -> list[Tool]:
        name = {"type": "string", "enum": self.names, "description": "the skill's name"}
        load = ToolSpec(
            name=LOAD_SKILL,
            description="Read a skill's instructions (its SKILL.md) and the files it has.",
            input_schema={"type": "object", "properties": {"name": name}, "required": ["name"]},
            side_effects="read",
        )
        read = ToolSpec(
            name=READ_SKILL_FILE,
            description="Read one file of a skill, by the path load_skill lists.",
            input_schema={
                "type": "object",
                "properties": {"name": name, "path": {"type": "string"}},
                "required": ["name", "path"],
            },
            side_effects="read",
        )
        return [Tool(load, _load, feature="skills"), Tool(read, _read, feature="skills")]


def skills(*refs: str | Skill) -> Skills:
    """Skills by name, each optionally pinned to a version (``"refunds@1.2.0"``; else the
    version its source serves when a run starts), or given (:class:`Skill`), as a tool
    source."""
    return Skills(refs)


async def pin(runtime: Runtime, sources: Sequence[Any]) -> None:
    """The run's skills (from its agent's :class:`Skills` source, when it has one) pinned, as
    the journal says an earlier attempt pinned them or as their sources give them now; their
    section appended to the run's context, and their tools offered."""
    source = next((s for s in sources if isinstance(s, Skills)), None)
    if source is None:
        return
    replayed, recorded = runtime.replay.call(source.key)
    found = await runtime.agent.harness.skills.pin(
        source.refs, recorded=recorded if replayed else None
    )
    for name, problem in found.problems.items():
        runtime.events.warning("skills_unavailable", f"skill {name}: {problem}")
    if not replayed:
        runtime.replay.record_call(source.key, found.record())
    runtime.skills = found.skills
    if not found.skills:
        return
    section = found.section
    runtime.context = f"{runtime.context}\n\n{section}" if runtime.context else section
    runtime.offer([LOAD_SKILL, READ_SKILL_FILE])
    runtime.events.custom(EVENT, versions=found.versions)
    telemetry.attribute("trellis.skills", ",".join(f"{n}@{v}" for n, v in found.versions.items()))


def _run_skills() -> PinnedSkills:
    runtime = current()
    if runtime is None:
        raise ToolError("skills are read inside a Harness run", source="tools")
    return PinnedSkills(runtime.skills)


async def _load(args: dict[str, Any]) -> str:
    return await _run_skills().load(str(args.get("name")))


async def _read(args: dict[str, Any]) -> str:
    return await _run_skills().read(str(args.get("name")), str(args.get("path")))
