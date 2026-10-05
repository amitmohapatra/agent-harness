"""Agent Skills from the gateway's Skills Repository, for any framework: ``skills(...)``.

    agent = h.wrap(graph_or_agent, id="analyst", skills=["sql-review", "refunds@1.2.0"])
    tools = await h.tools(skills("sql-review"), framework="langgraph")  # a graph binds its own

Progressive disclosure: at the start of a run each skill is pinned — the version named
(``name@version``), else the version the gateway serves — and its name and description go
into the context pushed into the framework's input (a ``## Skills`` section, after the memory
context); two read-only tools read the rest when the model wants it: :data:`LOAD_SKILL`
(``SKILL.md``'s body and the skill's file list, as the pinned version reads) and
:data:`READ_SKILL_FILE` (one file). Both are harness tools: every call goes through the bridge
(journaled, so a resumed run reads the same text), and the pinned versions are journaled too,
so a resumed run keeps them; the run says which versions it used (a ``skills`` event, and the
``trellis.skills`` attribute of its span).

The gateway serves a file only of the version it serves now: a file of another pinned version
is refused with a message saying so (its body still loads). A skill the gateway cannot give —
it is down, and was never read; there is no such skill — is a ``skills_unavailable`` warning,
and the run goes on without it (the last copy read stands while the gateway is down).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.harness import telemetry
from trellis.harness.clients.bifrost import pinned
from trellis.harness.journal import content_key
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import Tool

if TYPE_CHECKING:
    from bifrost_sdk.admin import Skill

    from trellis.harness.clients.bifrost import Gateway

#: The tools, and the event a run's pinned versions are said with.
LOAD_SKILL: Final = "load_skill"
READ_SKILL_FILE: Final = "read_skill_file"
EVENT: Final = "skills"
#: What the section the skills go into the context with begins with.
SECTION: Final = (
    "## Skills\nInstructions for kinds of task. When one fits the task, read it first with "
    f"{LOAD_SKILL}(name); {READ_SKILL_FILE}(name, path) reads a file it lists."
)


class Skills:
    """The skills an agent uses, as a tool source (``tools=[...]``, ``h.tools(...)``): the
    two tools; the pinning and the context section are the run's (:func:`pin`)."""

    def __init__(self, refs: Sequence[str]) -> None:
        self.refs = [pinned(ref) for ref in refs]
        names = [name for name, _ in self.refs]
        if not names or len(set(names)) < len(names):
            raise ConfigurationError("skills(...) names each skill once, at least one")
        self.names = names

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
        return [Tool(load, _load), Tool(read, _read)]


def skills(*refs: str) -> Skills:
    """Skills of the gateway's Skills Repository, by name, each optionally pinned to a version
    (``"refunds@1.2.0"``; else the version served when a run starts), as a tool source."""
    return Skills(refs)


async def pin(runtime: Runtime, sources: Sequence[Any]) -> None:
    """The run's skills (from its agent's :class:`Skills` source, when it has one) pinned, as
    the journal says an earlier attempt pinned them or as the gateway serves them now; their
    section appended to the run's context, and their tools offered."""
    source = next((s for s in sources if isinstance(s, Skills)), None)
    if source is None:
        return
    gateway = runtime.agent.harness.gateway
    if gateway is None:
        raise ConfigurationError("skills come from the Bifrost gateway: set BIFROST_URL")
    key = content_key(EVENT, source.refs)
    replayed, recorded = runtime.replay.call(key)
    versions: dict[str, str] = dict(recorded) if replayed else {}
    for name, version in source.refs:
        try:
            runtime.skills[name] = await gateway.skill(name, version or versions.get(name))
        except Exception as exc:
            runtime.events.warning("skills_unavailable", f"skill {name}: {exc}")
    if not replayed:
        runtime.replay.record_call(key, {n: s.version for n, s in runtime.skills.items()})
    if not runtime.skills:
        return
    lines = [f"- {name}: {skill.description}" for name, skill in runtime.skills.items()]
    section = "\n".join([SECTION, *lines])
    runtime.context = f"{runtime.context}\n\n{section}" if runtime.context else section
    runtime.offer([LOAD_SKILL, READ_SKILL_FILE])
    used = {name: skill.version for name, skill in runtime.skills.items()}
    runtime.events.custom(EVENT, versions=used)
    telemetry.attribute("trellis.skills", ",".join(f"{n}@{v}" for n, v in used.items()))


def _pinned(name: str) -> tuple[Skill, Gateway]:
    runtime = current()
    if runtime is None:
        raise ToolError("skills are read inside a Harness run", source="tools")
    found = runtime.skills.get(name)
    gateway = runtime.agent.harness.gateway
    if found is None or gateway is None:
        known = ", ".join(runtime.skills) or "none"
        raise ToolError(f"no skill {name!r} in this run (its skills: {known})", source="tools")
    return found, gateway


async def _load(args: dict[str, Any]) -> str:
    skill, _ = _pinned(str(args.get("name")))
    files = [f"- {f.path}" for f in skill.files]
    listed = f"\n\nFiles ({READ_SKILL_FILE}):\n" + "\n".join(files) if files else ""
    return f"# {skill.name} (version {skill.version})\n\n{skill.body}{listed}"


async def _read(args: dict[str, Any]) -> str:
    skill, gateway = _pinned(str(args.get("name")))
    path = str(args.get("path"))
    if path not in {f.path for f in skill.files}:
        raise ToolError(f"skill {skill.name} {skill.version} has no file {path!r}", source="tools")
    served = await gateway.skill(skill.name)
    if served.version != skill.version:
        raise ToolError(
            f"{path} of skill {skill.name} cannot be read: this run uses version "
            f"{skill.version}, and the gateway serves files only of the version it serves "
            f"({served.version})",
            source="tools",
        )
    return (await gateway.skill_file(skill.name, path)).decode("utf-8", errors="replace")
