"""Skills from the stores teams keep them in — edited at any time, per tenant, workspace, user
and agent — fed to each framework's own skills slot:

    src = skills.combine(skills.directory("skills"), my_store)   # trellis.skills
    ReAct("You review SQL.", model, skills=src)                  # ReAct
    create_deep_agent(model, **src.deep_agent())                 # Deep Agents: skills= and backend=
    ClaudeAgentOptions(**src.claude())                           # Claude: a local plugin of skills
    SandboxAgent(..., capabilities=[*Capabilities.default(), src.openai()])  # OpenAI: Skills
    await agent.tools.use(src.tools())                           # no skills slot: list/load tools

A source (:class:`SkillSource`, a customer's store implements it) answers three questions for a
:class:`Scope`: ``list`` — the skills it has there (:class:`SkillInfo`: name, description,
version, the level it is kept at, enabled or not); ``load`` — one version's ``SKILL.md`` body
and the paths of its other files; ``read_file`` — one of those files. ``changed_since`` is
optional: a cheap check that spares a listing. Built in (``trellis.skills``):
:func:`directory` (``SKILL.md`` folders), :func:`bifrost` (the gateway's skills repository),
:func:`langfuse` (Langfuse prompts tagged as skills), and :func:`combine`.

Which skill a name means (:meth:`SkillSet.pinned`): every source's skills of the run's scope,
the one at the most specific level winning — agent over user over workspace over tenant over
the platform-wide sources — then the first source in the order given; a winner that is
disabled hides the name. Per run:

* **the scope** is the run's (its tenant, workspace, user and agent); code that is not wrapped
  gives one (``scope=``), else the default tenant's;
* **the versions are pinned** at the run's first use and kept in its journal
  (``repository.journaled``, the stored prompts' pin): a resumed run uses the versions it saw;
  an edit applies from the next run, never mid-run;
* **a skill deleted** (or a version its source no longer serves) fails soft: the model reads
  why, the run goes on;
* **a source that is down** stands at what it last listed for that scope (said once per run as
  a ``skills_unavailable`` warning); one never read lists nothing, and the run goes on;
* **skill text is data**, capped: a description at :data:`MAX_DESCRIPTION` characters on one
  line, a body at :data:`MAX_BODY`, a file at :data:`MAX_FILE`; a name is the Agent Skills
  spec's (lowercase letters, digits and single hyphens, at most 64), any other skipped.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import logging
import os
import re
import shutil
import tempfile
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.harness import telemetry
from trellis.harness.bounded import LRU
from trellis.harness.journal import content_key
from trellis.harness.repository import (
    BIFROST,
    RETRY_SECONDS,
    TTL_SECONDS,
    Kept,
    NotFound,
    digest,
    front_matter,
    inside,
    journaled,
)
from trellis.harness.runtime import Runtime, current
from trellis.harness.sandbox import SandboxSource
from trellis.harness.tools.base import Tool

if TYPE_CHECKING:
    from bifrost_sdk.admin import Admin

log = logging.getLogger("trellis.skills")

#: The levels a skill is kept at, from the least specific: a name at a later level wins.
LEVELS: Final = ("platform", "tenant", "workspace", "user", "agent")
Level = Literal["platform", "tenant", "workspace", "user", "agent"]
#: The tenant of code that is not wrapped and names none.
DEFAULT_TENANT: Final = "default"
#: Where a Deep Agent finds the skills (its ``skills=`` source, a route of its backend).
SKILLS_ROOT: Final = "/skills/"
SKILL_MD: Final = "SKILL.md"
#: The Claude plugin the skills are a local plugin of (``trellis:<skill>``).
PLUGIN: Final = "trellis"
#: The event, and the span attribute ``trellis.skills``, a run's pinned versions are said with;
#: the code of the warning a source that is down, or a skill that is gone, is said with.
EVENT: Final = "skills"
UNAVAILABLE: Final = "skills_unavailable"
#: The tools of an engine with no skills slot (:meth:`SkillSet.tools`).
LIST_SKILLS: Final = "list_skills"
LOAD_SKILL: Final = "load_skill"
#: What a skill's text may hold: it is data, never the system's own instructions.
MAX_DESCRIPTION: Final = 1024
MAX_BODY: Final = 32_000
MAX_FILE: Final = 256 * 1024
MAX_NAME: Final = 64
NAME: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: How many skills are loaded at once when a framework reads them all (Deep Agents' listing).
LOADS_AT_ONCE: Final = 16
#: How long code that is not wrapped keeps a listing (a wrapped run lists at its start).
OUTSIDE_TTL_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class Scope:
    """Whose skills: a tenant's, and within it a workspace's, a user's and an agent's."""

    tenant: str = DEFAULT_TENANT
    workspace: str | None = None
    user: str | None = None
    agent: str | None = None

    @classmethod
    def of(cls, runtime: Runtime) -> Scope:
        identity = runtime.identity
        return cls(identity.tenant, identity.workspace, identity.user, identity.agent_id)

    def key(self) -> tuple[str, str | None, str | None, str | None]:
        return (self.tenant, self.workspace, self.user, self.agent)


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """One skill a source has in a scope: its version (any string the source can serve again
    by :meth:`SkillSource.load`), the level it is kept at, and whether it is on."""

    name: str
    description: str
    version: str
    updated_at: datetime | None = None
    level: Level = "platform"
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class SkillDoc:
    """One version of a skill: its ``SKILL.md`` body (without front matter) and the paths of
    its other files."""

    body: str
    files: tuple[str, ...] = ()


@runtime_checkable
class SkillSource(Protocol):
    """Where skills come from. ``load`` and ``read_file`` raise a ``LookupError`` for a name,
    version or file the source does not have (now). Optional: ``async def
    changed_since(scope, cursor: datetime) -> bool`` — whether anything of ``scope`` changed
    since ``cursor`` (``False`` keeps the last listing)."""

    @property
    def label(self) -> str: ...

    async def list(self, scope: Scope) -> Sequence[SkillInfo]: ...

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc: ...

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str: ...


@dataclass(frozen=True, slots=True)
class Chosen:
    """A skill as a run pinned it: which version, from which source, kept at which level."""

    name: str
    description: str
    version: str
    source: str
    level: Level

    def record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def of_record(cls, data: Mapping[str, Any]) -> Chosen:
        return cls(**data)

    @property
    def skill_md(self) -> str:
        """The front matter of its ``SKILL.md`` (the description a JSON string, valid YAML)."""
        return f"---\nname: {self.name}\ndescription: {json.dumps(self.description)}\n---\n"


def capped(text: str, most: int, *, between: str = "\n\n") -> str:
    """``text`` cut at ``most`` characters, saying so (after ``between``)."""
    if len(text) <= most:
        return text
    return f"{text[:most]}{between}[cut: {len(text) - most} more characters; at most {most}]"


def _described(text: str) -> str:
    """A description on one line, capped: what the skills listing shows."""
    return capped(" ".join(text.split()), MAX_DESCRIPTION, between=" ")


def _relative(path: str) -> bool:
    """Whether a skill's file path stays in its folder: relative, no ``..``."""
    parts = PurePosixPath(path).parts
    return bool(parts) and not path.startswith("/") and ".." not in parts


def valid_name(name: str) -> bool:
    return len(name) <= MAX_NAME and NAME.fullmatch(name) is not None


def allowed(meta: Mapping[str, Any], scope: Scope) -> bool:
    """A shared skill's allow-list (``tenants:``, ``users:`` in its front matter): none, or
    one that names the scope's."""
    for key, value in (("tenants", scope.tenant), ("users", scope.user)):
        listed = meta.get(key)
        if listed is None:
            continue
        names = [listed] if isinstance(listed, str) else [str(n) for n in listed]
        if value not in names:
            return False
    return True


# --------------------------------------------------------------------------- the set
@dataclass(slots=True)
class _Listed:
    """A source's last listing of a scope, and when it was read."""

    at: datetime
    skills: list[SkillInfo]


class SkillSet:
    """Skill sources, combined (:func:`combine`), and the adapters into each framework's own
    skills slot. One shared object, never copied (a framework that copies what it is given
    per run — the OpenAI Agents SDK's capabilities — keeps this one)."""

    __slots__ = ("_by_label", "_docs", "_listed", "_outside", "_runs", "sources")

    def __copy__(self) -> SkillSet:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> SkillSet:
        return self

    def __init__(self, sources: Sequence[SkillSource]) -> None:
        if not sources:
            raise ConfigurationError("skills.combine(...) takes at least one source")
        for source in sources:
            if not isinstance(source, SkillSource):
                raise ConfigurationError(f"not a skill source: {source!r} (list, load, read_file)")
        labels = [s.label for s in sources]
        doubled = sorted({label for label in labels if labels.count(label) > 1})
        if doubled:
            raise ConfigurationError(f"two skill sources are labelled {', '.join(doubled)}")
        self.sources = list(sources)
        self._by_label = {s.label: s for s in self.sources}
        #: each source's last listing per scope: what stands while it is down
        self._listed: LRU[tuple[Any, ...], _Listed] = LRU()
        #: one version of a skill of a scope, once loaded
        self._docs: LRU[tuple[Any, ...], SkillDoc] = LRU()
        #: what one attempt of a run pinned, and which sources it said were down
        self._runs: weakref.WeakKeyDictionary[Runtime, dict[str, Chosen]] = (
            weakref.WeakKeyDictionary()
        )
        #: what code that is not wrapped pinned, per scope, for a while
        self._outside: LRU[tuple[Any, ...], tuple[float, dict[str, Chosen]]] = LRU()

    @property
    def label(self) -> str:
        return " + ".join(self._by_label)

    # ------------------------------------------------------------------ which skills
    async def pinned(self, scope: Scope | None = None) -> dict[str, Chosen]:
        """The skills by name, sorted: inside a run its scope's, pinned at its first use (the
        journal's on a resume); otherwise ``scope``'s (else the default tenant's), kept for
        :data:`OUTSIDE_TTL_SECONDS`."""
        runtime = current()
        if runtime is None:
            return await self._outside_pinned(scope or Scope())
        found = self._runs.get(runtime)
        if found is not None:
            return found
        async with runtime.replay.exclusive(EVENT):
            found = self._runs.get(runtime)
            if found is not None:
                return found
            listed = await journaled(
                runtime,
                content_key(EVENT, self.label),
                lambda: self._choose(Scope.of(runtime), runtime),
                dump=lambda chosen: [c.record() for c in chosen.values()],
                load=lambda kept: {c["name"]: Chosen.of_record(c) for c in kept},
            )
            self._runs[runtime] = listed
        versions = {c.name: c.version for c in listed.values()}
        runtime.events.custom(EVENT, versions=versions)
        telemetry.attribute("trellis.skills", ",".join(f"{n}@{v}" for n, v in versions.items()))
        return listed

    def pinned_now(self, scope: Scope | None = None) -> dict[str, Chosen]:
        """What :meth:`pinned` gave last for the run (or ``scope``), without asking: for a
        framework that reads the listing synchronously, after it was pinned."""
        runtime = current()
        if runtime is not None:
            return self._runs.get(runtime) or {}
        kept = self._outside.get((scope or Scope()).key())
        return kept[1] if kept is not None else {}

    async def _outside_pinned(self, scope: Scope) -> dict[str, Chosen]:
        now = asyncio.get_running_loop().time()
        kept = self._outside.get(scope.key())
        if kept is not None and now - kept[0] < OUTSIDE_TTL_SECONDS:
            return kept[1]
        chosen = await self._choose(scope, None)
        self._outside[scope.key()] = (now, chosen)
        return chosen

    async def _choose(self, scope: Scope, runtime: Runtime | None) -> dict[str, Chosen]:
        """Every source's listing of ``scope``, merged: per name the most specific level wins,
        then the source given first; a disabled winner hides the name."""
        best: dict[str, tuple[tuple[int, int], SkillInfo, SkillSource]] = {}
        listings = await asyncio.gather(*(self._listing(s, scope, runtime) for s in self.sources))
        for order, (source, listing) in enumerate(zip(self.sources, listings, strict=True)):
            for info in listing:
                if not valid_name(info.name):
                    log.warning(
                        "skill %r of %s: not a skill name; skipped", info.name, source.label
                    )
                    continue
                rank = (LEVELS.index(info.level), -order)
                held = best.get(info.name)
                if held is None or rank > held[0]:
                    best[info.name] = (rank, info, source)
        return {
            name: Chosen(name, _described(info.description), info.version, source.label, info.level)
            for name, (_, info, source) in sorted(best.items())
            if info.enabled
        }

    async def _listing(
        self, source: SkillSource, scope: Scope, runtime: Runtime | None
    ) -> list[SkillInfo]:
        """``source``'s skills in ``scope``: asked again (or, with ``changed_since``, only when
        something changed); while it is down, its last listing of the scope."""
        key = (source.label, *scope.key())
        last = self._listed.get(key)
        try:
            changed = getattr(source, "changed_since", None)
            if last is not None and changed is not None and not await changed(scope, last.at):
                return last.skills
            now = datetime.now(UTC)
            skills = list(await source.list(scope))
        except Exception as exc:
            standing = "its last listing stands" if last is not None else "it lists nothing"
            said = (
                f"skill source {source.label} could not be read ({type(exc).__name__}: {exc}): "
                f"{standing}"
            )
            if runtime is not None:
                runtime.events.warning(UNAVAILABLE, said)
            log.warning("%s", said)
            return last.skills if last is not None else []
        self._listed[key] = _Listed(now, skills)
        return skills

    # ------------------------------------------------------------------ what a skill says
    async def document(self, name: str, scope: Scope | None = None) -> SkillDoc:
        """The pinned version of ``name``: its body capped, its files. A skill not pinned, or
        one its source no longer serves, is a :class:`ToolError` saying why (the model reads
        it, the run goes on)."""
        chosen, given = await self._chosen(name, scope)
        key = (chosen.source, *given.key(), name, chosen.version)
        found = self._docs.get(key)
        if found is not None:
            return found
        source = self._by_label[chosen.source]
        try:
            loaded = await source.load(name, chosen.version, given)
        except Exception as exc:
            raise self._gone(chosen, exc) from exc
        files = tuple(f for f in loaded.files if _relative(f))  # never a path out of it
        found = self._docs[key] = SkillDoc(capped(loaded.body, MAX_BODY), files)
        return found

    async def read_file(self, name: str, path: str, scope: Scope | None = None) -> str:
        """One file of the pinned version of ``name``, capped at :data:`MAX_FILE`."""
        chosen, given = await self._chosen(name, scope)
        doc = await self.document(name, scope)
        if path not in doc.files:
            raise ToolError(f"skill {name} has no file {path!r}", source="tools")
        try:
            text = await self._by_label[chosen.source].read_file(name, chosen.version, path, given)
        except Exception as exc:
            raise self._gone(chosen, exc, path) from exc
        return capped(text, MAX_FILE)

    async def full_text(self, name: str, scope: Scope | None = None) -> str:
        """``SKILL.md`` as a framework reads it: its front matter, then its body."""
        chosen, _ = await self._chosen(name, scope)
        return f"{chosen.skill_md}\n{(await self.document(name, scope)).body}\n"

    async def _chosen(self, name: str, scope: Scope | None) -> tuple[Chosen, Scope]:
        runtime = current()
        given = Scope.of(runtime) if runtime is not None else (scope or Scope())
        chosen = (await self.pinned(given if runtime is None else None)).get(name)
        if chosen is None:
            raise ToolError(f"no skill {name!r} for this run", source="tools")
        return chosen, given

    def _gone(self, chosen: Chosen, exc: Exception, path: str | None = None) -> ToolError:
        what = f"{path} of skill {chosen.name}" if path else f"skill {chosen.name}"
        said = (
            f"{what} cannot be read: this run uses version {chosen.version} of "
            f"{chosen.source}, which does not serve it now ({type(exc).__name__}: {exc})"
        )
        runtime = current()
        if runtime is not None:
            runtime.events.warning(UNAVAILABLE, said)
        return ToolError(said, source="tools")

    # ------------------------------------------------------------------ the adapters
    def deep_agent(self, backend: Any = None, *, scope: Scope | None = None) -> dict[str, Any]:
        """``create_deep_agent``'s arguments for these skills: ``skills=[SKILLS_ROOT]`` and a
        ``backend`` whose :data:`SKILLS_ROOT` route serves them (Deep Agents'
        ``CompositeBackend``) and whose default is ``backend`` — a Deep Agents backend, or
        ``sandbox()`` — else the graph's state. ``scope``: whose, outside a harness run."""
        from deepagents.backends import CompositeBackend, StateBackend  # noqa: PLC0415 - extra

        if isinstance(backend, SandboxSource):
            backend = backend.deep_agent()["backend"]
        routes = {SKILLS_ROOT: SkillsBackend(self, scope)}
        default = backend if backend is not None else StateBackend()
        return {
            "backend": CompositeBackend(default=default, routes=routes),
            "skills": [SKILLS_ROOT],
        }

    def claude(self) -> dict[str, Any]:
        """``ClaudeAgentOptions``' arguments for these skills: a local plugin (:data:`PLUGIN`)
        holding the run's pinned skills — written for each run by the harness, removed after
        it — and ``skills="all"``. Code that is not wrapped writes one itself:
        :meth:`claude_dir`."""
        return {"plugins": [ClaudeSkills(self)], "skills": "all"}

    async def claude_dir(
        self, path: str | os.PathLike[str], scope: Scope | None = None
    ) -> dict[str, Any]:
        """The skills of ``scope`` written as a Claude plugin at ``path``, and
        ``ClaudeAgentOptions``' arguments for it — for code that runs the SDK itself."""
        await write_plugin(self, Path(path), scope)
        return {"plugins": [{"type": "local", "path": str(path)}], "skills": "all"}

    def openai(self, scope: Scope | None = None) -> Any:
        """The OpenAI Agents SDK's own ``Skills`` capability for a ``SandboxAgent``: the run's
        pinned skills listed, each copied into the sandbox by its ``load_skill``."""
        from trellis.harness.skills_openai import capability  # noqa: PLC0415 - its extra

        return capability(self, scope)

    def tools(self, scope: Scope | None = None) -> SkillTools:
        """:data:`LIST_SKILLS` and :data:`LOAD_SKILL`, harness tools, for an engine with no
        skills slot of its own (a function, a LangGraph graph, an OpenAI ``Agent``)."""
        return SkillTools(self, scope)


def combine(*sources: SkillSource | SkillSet) -> SkillSet:
    """Skill sources as one: a name kept at a more specific level wins (agent, user,
    workspace, tenant, then the platform-wide sources), then the source given first."""
    flat: list[SkillSource] = []
    for given in sources:
        flat.extend(given.sources if isinstance(given, SkillSet) else [given])
    return SkillSet(flat)


def skill_set(given: Any) -> SkillSet | None:
    """``given`` as a :class:`SkillSet` when it is skills (a set, or one source), else
    ``None``."""
    if isinstance(given, SkillSet):
        return given
    return SkillSet([given]) if isinstance(given, SkillSource) else None


def deep_agent(sandbox: Any = None, skills: SkillSet | SkillSource | None = None) -> dict[str, Any]:
    """``create_deep_agent``'s arguments for a sandbox and skills in its one ``backend=``:
    ``create_deep_agent(model, **deep_agent(sandbox(), src))`` — the sandbox the default of a
    ``CompositeBackend``, the skills its :data:`SKILLS_ROOT` route (``skills=``). ``sandbox``
    may be any Deep Agents backend."""
    found = skill_set(skills) if skills is not None else None
    if skills is not None and found is None:
        raise ConfigurationError(f"not skills: {skills!r}")
    if found is not None:
        return found.deep_agent(sandbox)
    if sandbox is None:
        raise ConfigurationError("deep_agent(sandbox, skills): give a sandbox, skills or both")
    return sandbox.deep_agent() if isinstance(sandbox, SandboxSource) else {"backend": sandbox}


# --------------------------------------------------------------------------- Deep Agents
#: The files one call of :class:`SkillsBackend` serves (Deep Agents' ``StateBackend`` reads them).
_FILES: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "trellis_skill_files", default=None
)
#: Said by a synchronous call (a graph run with ``invoke``).
ASYNC_ONLY: Final = "the skills backend runs in a graph run asynchronously (ainvoke)"
READ_ONLY: Final = "the skills are read-only: write your files elsewhere"


def _file(content: str) -> dict[str, Any]:
    from deepagents.backends.utils import create_file_data  # noqa: PLC0415 - its extra

    return dict(create_file_data(content))


def _make_backend_class() -> type:
    from deepagents.backends import StateBackend  # noqa: PLC0415 - its extra
    from deepagents.backends.protocol import (  # noqa: PLC0415
        DeleteResult,
        EditResult,
        FileUploadResponse,
        WriteResult,
    )

    class _SkillsBackend(StateBackend):
        """The pinned skills as Deep Agents files, read-only: ``/<name>/SKILL.md`` (front
        matter and body) and ``/<name>/<file>`` — Deep Agents' ``StateBackend`` reading them,
        one call's files at a time."""

        def __init__(self, skills: SkillSet, scope: Scope | None = None) -> None:
            super().__init__()
            self.skills = skills
            self.scope = scope

        def _read_files(self) -> dict[str, Any]:
            files = _FILES.get()
            if files is None:
                raise ConfigurationError(ASYNC_ONLY)
            return files

        def _send_files_update(self, update: dict[str, Any]) -> None:
            raise ToolError(READ_ONLY, source="tools")

        async def _files(self, names: Sequence[str] | None, *, contents: bool) -> dict[str, Any]:
            """The files of ``names`` (``None``: every skill's): their paths, and with
            ``contents`` their text; a skill that cannot be read is left out."""
            pinned = await self.skills.pinned(self.scope)
            chosen = list(pinned) if names is None else [n for n in names if n in pinned]
            semaphore = asyncio.Semaphore(LOADS_AT_ONCE)

            async def one(name: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        doc = await self.skills.document(name, self.scope)
                        found = {
                            f"/{name}/{SKILL_MD}": await self.skills.full_text(name, self.scope)
                        }
                        for path in doc.files:
                            found[f"/{name}/{path}"] = (
                                await self.skills.read_file(name, path, self.scope)
                                if contents
                                else ""
                            )
                    except ToolError:
                        return {}
                return {path: _file(text) for path, text in found.items()}

            return {
                k: v for part in await asyncio.gather(*map(one, chosen)) for k, v in part.items()
            }

        @contextmanager
        def _serving(self, files: dict[str, Any]) -> Iterator[None]:
            token = _FILES.set(files)
            try:
                yield
            finally:
                _FILES.reset(token)

        async def als(self, path: str) -> Any:
            names = _names_in(path)
            if names is None:  # the root: one folder per skill
                pinned = await self.skills.pinned(self.scope)
                files = {f"/{n}/{SKILL_MD}": _file("") for n in pinned}
            else:
                files = await self._files(names, contents=False)
            with self._serving(files):
                return self.ls(path)

        async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
            files = await self._files(_names_in(file_path) or [], contents=True)
            with self._serving(files):
                return self.read(file_path, offset, limit)

        async def adownload_files(self, paths: list[str]) -> Any:
            names = sorted({n for p in paths for n in _names_in(p) or []})
            with self._serving(await self._files(names, contents=True)):
                return self.download_files(paths)

        async def agrep(
            self,
            pattern: str,
            path: str | None = None,
            glob: str | None = None,
            *,
            max_count: int | None = None,
        ) -> Any:
            with self._serving(await self._files(None, contents=True)):
                return self.grep(pattern, path, glob, max_count=max_count)

        async def aglob(self, pattern: str, path: str | None = None) -> Any:
            with self._serving(await self._files(None, contents=False)):
                return self.glob(pattern, path)

        async def awrite(self, file_path: str, content: str) -> Any:
            return WriteResult(error=READ_ONLY)

        async def aedit(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            return EditResult(error=READ_ONLY)

        async def adelete(self, file_path: str) -> Any:
            return DeleteResult(error=READ_ONLY)

        async def aupload_files(self, files: list[tuple[str, bytes]]) -> Any:
            return [FileUploadResponse(path=p, error="permission_denied") for p, _ in files]

    return _SkillsBackend


def _names_in(path: str) -> list[str] | None:
    """The skill a path is in (``None``: the root, every skill)."""
    parts = [p for p in PurePosixPath("/" + path.lstrip("/")).parts[1:] if p]
    return [parts[0]] if parts else None


_BACKEND: type | None = None


def SkillsBackend(skills: SkillSet, scope: Scope | None = None) -> Any:
    """A Deep Agents backend serving ``skills`` (read-only), built on its ``StateBackend``."""
    global _BACKEND  # noqa: PLW0603 - the class is built once Deep Agents is imported
    if _BACKEND is None:
        _BACKEND = _make_backend_class()
    return _BACKEND(skills, scope)


# --------------------------------------------------------------------------- Claude
class ClaudeSkills(dict[str, str]):
    """A Claude local plugin (``{"type": "local", "path": ...}``) the harness fills with the
    run's pinned skills before each query (:func:`for_run`): until then its path is empty."""

    def __init__(self, skills: SkillSet) -> None:
        super().__init__(type="local", path="")
        self.skills = skills


async def write_plugin(skills: SkillSet, root: Path, scope: Scope | None = None) -> None:
    """``skills`` of the run (or ``scope``) as a Claude plugin at ``root``:
    ``.claude-plugin/plugin.json`` and ``skills/<name>/SKILL.md`` with its files. A skill that
    cannot be read is left out (said as a warning)."""
    pinned = await skills.pinned(scope if current() is None else None)
    manifest = {"name": PLUGIN, "description": "Skills served by Trellis", "version": "1.0.0"}
    written = {".claude-plugin/plugin.json": json.dumps(manifest)}
    for name in pinned:
        try:
            files = {SKILL_MD: await skills.full_text(name, scope)}
            for path in (await skills.document(name, scope)).files:
                files[path] = await skills.read_file(name, path, scope)
        except ToolError:
            continue
        written |= {f"skills/{name}/{path}": text for path, text in files.items()}
    await asyncio.to_thread(_written, root, written)


def _written(root: Path, files: Mapping[str, str]) -> None:
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")


async def for_run(plugins: Sequence[Any]) -> tuple[list[Any], list[Path]]:
    """``plugins`` with each :class:`ClaudeSkills` written for the run in a folder of its own,
    and the folders to remove when the query ends."""
    made: list[Path] = []
    given: list[Any] = []
    for plugin in plugins:
        if not isinstance(plugin, ClaudeSkills):
            given.append(plugin)
            continue
        root = Path(tempfile.mkdtemp(prefix="trellis-skills-"))
        made.append(root)
        await write_plugin(plugin.skills, root)
        given.append({"type": "local", "path": str(root)})
    return given, made


def removed(folders: Sequence[Path]) -> None:
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- tools
class SkillTools:
    """:data:`LIST_SKILLS` and :data:`LOAD_SKILL` as a tool source (:meth:`SkillSet.tools`)."""

    def __init__(self, skills: SkillSet, scope: Scope | None) -> None:
        self.skills = skills
        self.scope = scope

    async def resolve(self) -> list[Tool]:
        listing = ToolSpec(
            name=LIST_SKILLS,
            description="List the skills you can load: each one's name and what it is for.",
            input_schema={"type": "object", "properties": {}},
            side_effects="read",
        )
        loading = ToolSpec(
            name=LOAD_SKILL,
            description=(
                "Read a skill's instructions (its SKILL.md) and the files it has; with path, "
                "one of its files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "the skill's name"},
                    "path": {"type": "string", "description": "a file it lists (optional)"},
                },
                "required": ["name"],
            },
            side_effects="read",
        )
        return [
            Tool(listing, self._list),
            Tool(loading, self._load),
        ]

    async def _list(self, args: dict[str, Any]) -> str:
        pinned = await self.skills.pinned(self.scope)
        return "\n".join(f"- {c.name}: {c.description}" for c in pinned.values()) or "no skills"

    async def _load(self, args: dict[str, Any]) -> str:
        name, path = str(args["name"]), args.get("path")
        if path:
            return await self.skills.read_file(name, str(path), self.scope)
        doc = await self.skills.document(name, self.scope)
        listed = "".join(f"\n- {f}" for f in doc.files)
        files = f"\n\nFiles ({LOAD_SKILL}(name, path)):{listed}" if listed else ""
        return f"{await self.skills.full_text(name, self.scope)}{files}"


# --------------------------------------------------------------------------- built-in sources
class DirectorySource:
    """``SKILL.md`` folders on this machine (:func:`directory`), read at each listing: what is
    on disk now. A version is the front matter's (``version``, ``metadata.version``), else the
    folder's content digest; an allow-list (``tenants:``, ``users:``) limits who has it."""

    def __init__(self, path: str | os.PathLike[str], *, level: Level = "platform") -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise ConfigurationError(f"skills.directory({str(self.root)!r}): no such folder")
        self.level: Level = level

    @property
    def label(self) -> str:
        return f"directory({self.root})"

    async def list(self, scope: Scope) -> list[SkillInfo]:
        return await asyncio.to_thread(self._list, scope)

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc:
        meta, doc = await asyncio.to_thread(self._read, name)
        if meta["version"] != version:
            raise NotFound(f"{name} is version {meta['version']} on disk now, not {version}")
        return doc

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str:
        meta, doc = await asyncio.to_thread(self._read, name)
        file = inside(self.root / name, path)
        if meta["version"] != version or file is None or path not in doc.files:
            raise NotFound(f"{name} {version} has no file {path!r} on disk now")
        return await asyncio.to_thread(file.read_text, "utf-8", "replace")

    def _list(self, scope: Scope) -> list[SkillInfo]:
        found: list[SkillInfo] = []
        for folder in sorted(p for p in self.root.iterdir() if (p / SKILL_MD).is_file()):
            try:
                meta, _ = self._read(folder.name)
            except ConfigurationError as exc:
                log.warning("%s: skipped (%s)", folder / SKILL_MD, exc)
                continue
            if allowed(meta, scope):
                found.append(
                    SkillInfo(folder.name, meta["description"], meta["version"], level=self.level)
                )
        return found

    def _read(self, name: str) -> tuple[dict[str, Any], SkillDoc]:
        folder = inside(self.root, name)
        if folder is None or not (folder / SKILL_MD).is_file():
            raise NotFound(f"no skill folder {name!r}")
        raw = (folder / SKILL_MD).read_bytes()
        where = str(folder / SKILL_MD)
        meta, body = front_matter(raw.decode("utf-8", errors="replace"), where=where)
        if meta.get("name") != name:
            raise ConfigurationError(f"{where}: its front matter's name must be {name!r}")
        if not isinstance(meta.get("description"), str) or not meta["description"]:
            raise ConfigurationError(f"{where}: its front matter has no description")
        listed = (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        files = sorted(p for p in listed if p != SKILL_MD and inside(folder, p) is not None)
        extra = meta.get("metadata")
        version = meta.get("version") or (extra.get("version") if isinstance(extra, dict) else None)
        meta["version"] = str(
            version or digest(raw, *(p.encode() + (folder / p).read_bytes() for p in files))
        )
        return meta, SkillDoc(body.strip(), tuple(files))


def directory(path: str | os.PathLike[str], *, level: Level = "platform") -> SkillSet:
    """``<name>/SKILL.md`` folders (front matter ``name``, ``description``; optionally
    ``version``, ``tenants``, ``users``) and their files, as skills kept at ``level``."""
    return SkillSet([DirectorySource(path, level=level)])


class BifrostSource:
    """The gateway's skills repository (``bifrost_sdk.admin.skills``): platform-wide, one
    version served at a time, any published version loaded; a file only of the served version
    (the gateway serves no other). An allow-list in a skill's extra front matter (``tenants``,
    ``users``) limits who has it."""

    def __init__(self, admin: Admin | None = None) -> None:
        self.given = admin

    @property
    def label(self) -> str:
        return BIFROST

    def _admin(self) -> Admin:
        if self.given is not None:
            return self.given
        runtime = current()
        gateway = runtime.agent.harness.gateway if runtime is not None else None
        if gateway is None:
            raise ConfigurationError("skills.bifrost(): set BIFROST_URL, or pass admin=")
        return gateway.admin

    async def list(self, scope: Scope) -> list[SkillInfo]:
        return [
            SkillInfo(s.name, s.description, s.version, s.updated_at)
            for s in await self._admin().skills.list()
            if allowed(s.extra_frontmatter, scope)
        ]

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc:
        found = await self._admin().skills.find(name, version)
        if found is None:
            raise NotFound(f"the gateway has no skill {name!r}")
        return SkillDoc(found.body, tuple(f.path for f in found.files))

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str:
        admin = self._admin()
        served = [s.version for s in await admin.skills.list(search=name) if s.name == name]
        if served != [version]:
            now = served[0] if served else "none"
            raise NotFound(f"the gateway serves files of version {now} only")
        return (await admin.skills.read_file(name, path)).decode("utf-8", errors="replace")


def bifrost(admin: Admin | None = None) -> SkillSet:
    """The gateway's skills repository: through the harness's gateway (``BIFROST_URL``) in a
    run, else ``admin`` (``bifrost_sdk.admin.Admin``)."""
    return SkillSet([BifrostSource(admin)])


class LangfuseSource:
    """Langfuse prompts tagged ``tag``, as skills: the version labelled ``label`` (read again
    every ``ttl`` seconds; the last read stands while Langfuse is down), its text the body,
    its ``config.description`` (else its first line) the description. Platform-wide, no
    files."""

    def __init__(
        self,
        client: telemetry.Langfuse | None = None,
        *,
        tag: str = "skill",
        label: str = "production",
        ttl: float = TTL_SECONDS,
    ) -> None:
        self.client = client
        self.tag = tag
        self.served = label
        self._kept: Kept[str, dict[str, Any]] = Kept(
            self._labelled,
            what=lambda name: f"the Langfuse prompt {name!r}",
            ttl=ttl,
            retry=RETRY_SECONDS,
        )

    @property
    def label(self) -> str:
        return "Langfuse"

    def _client(self) -> telemetry.Langfuse:
        if self.client is None:
            self.client = telemetry.Langfuse.of()
        if self.client is None:
            raise ConfigurationError(
                "skills.langfuse(): set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or pass "
                "client="
            )
        return self.client

    async def _labelled(self, name: str) -> dict[str, Any] | None:
        return await self._client().prompt(name, label=self.served)

    async def list(self, scope: Scope) -> list[SkillInfo]:
        found: list[SkillInfo] = []
        for listed in await self._client().prompts(tag=self.tag):
            prompt = await self._kept.get(str(listed["name"]))
            if prompt is not None:
                found.append(
                    SkillInfo(prompt["name"], _description(prompt), str(prompt["version"]))
                )
        return found

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc:
        prompt = await self._client().prompt(name, version=int(version))
        if prompt is None:
            raise NotFound(f"Langfuse has no prompt {name!r} version {version}")
        return SkillDoc(_prompt_text(prompt))

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str:
        raise NotFound("a Langfuse skill has no files")


def _prompt_text(prompt: Mapping[str, Any]) -> str:
    body = prompt.get("prompt")
    if isinstance(body, list):  # a chat prompt: its messages' contents
        return "\n\n".join(str(m.get("content", "")) for m in body if isinstance(m, dict))
    return str(body or "")


def _description(prompt: Mapping[str, Any]) -> str:
    config = prompt.get("config")
    given = config.get("description") if isinstance(config, dict) else None
    if isinstance(given, str) and given:
        return given
    return next(
        (line for line in _prompt_text(prompt).splitlines() if line.strip()), prompt["name"]
    )


def langfuse(
    client: telemetry.Langfuse | None = None, *, tag: str = "skill", label: str = "production"
) -> SkillSet:
    """Langfuse prompts tagged ``tag`` as skills, the version labelled ``label`` (Langfuse's
    keys from its own variables, unless ``client`` is given)."""
    return SkillSet([LangfuseSource(client, tag=tag, label=label)])


__all__ = [
    "LEVELS",
    "SKILLS_ROOT",
    "Chosen",
    "Scope",
    "SkillDoc",
    "SkillInfo",
    "SkillSet",
    "SkillSource",
    "bifrost",
    "combine",
    "deep_agent",
    "directory",
    "langfuse",
]
