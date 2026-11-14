"""A customer's skill store, in memory: every version kept, per tenant / workspace / user /
agent, editable between and during runs, and able to go down."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from trellis.harness.repository import NotFound
from trellis.harness.skills import Level, Scope, SkillDoc, SkillInfo


@dataclass
class Kept:
    description: str
    versions: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)
    current: str = ""
    enabled: bool = True


class SkillStore:
    """Skills by (name, level, owner): the owner of a ``tenant`` skill is the tenant, of a
    ``user`` skill the user, and so on."""

    def __init__(self, label: str = "store", *, cheap_checks: bool = False) -> None:
        self._label = label
        self.kept: dict[tuple[str, Level, str | None], Kept] = {}
        self.down = False
        self.lists = 0
        self.loads: list[tuple[str, str]] = []
        self.changed_at = datetime.now(UTC)
        if cheap_checks:
            self.changed_since = self._changed_since  # type: ignore[method-assign]

    @property
    def label(self) -> str:
        return self._label

    def put(
        self,
        name: str,
        body: str,
        *,
        level: Level = "tenant",
        owner: str | None = "default",
        description: str | None = None,
        files: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> str:
        """A new version of ``name`` (its level's initial and a number from 1: a version names
        one entry's text), now the current one."""
        entry = self.kept.setdefault((name, level, owner), Kept(description or f"{name} skill"))
        if description is not None:
            entry.description = description
        version = f"{level[0]}{len(entry.versions) + 1}"
        entry.versions[version] = (body, dict(files or {}))
        entry.current = version
        entry.enabled = enabled
        self.changed_at = datetime.now(UTC)
        return version

    def delete(self, name: str, *, level: Level = "tenant", owner: str | None = "default") -> None:
        self.kept.pop((name, level, owner))
        self.changed_at = datetime.now(UTC)

    def _owns(self, level: Level, owner: str | None, scope: Scope) -> bool:
        mine = {
            "platform": None,
            "tenant": scope.tenant,
            "workspace": scope.workspace,
            "user": scope.user,
            "agent": scope.agent,
        }[level]
        return level == "platform" or (mine is not None and owner == mine)

    async def _changed_since(self, scope: Scope, cursor: datetime) -> bool:
        if self.down:
            raise ConnectionError(f"{self._label} is down")
        return self.changed_at > cursor

    async def list(self, scope: Scope) -> list[SkillInfo]:
        self.lists += 1
        if self.down:
            raise ConnectionError(f"{self._label} is down")
        return [
            SkillInfo(name, entry.description, entry.current, level=level, enabled=entry.enabled)
            for (name, level, owner), entry in self.kept.items()
            if self._owns(level, owner, scope)
        ]

    def _version(self, name: str, version: str, scope: Scope) -> tuple[str, dict[str, str]]:
        for (kept, level, owner), entry in self.kept.items():
            if kept == name and self._owns(level, owner, scope) and version in entry.versions:
                return entry.versions[version]
        raise NotFound(f"no {name} version {version}")

    async def load(self, name: str, version: str, scope: Scope) -> SkillDoc:
        if self.down:
            raise ConnectionError(f"{self._label} is down")
        self.loads.append((name, version))
        body, files = self._version(name, version, scope)
        return SkillDoc(body, tuple(sorted(files)))

    async def read_file(self, name: str, version: str, path: str, scope: Scope) -> str:
        _, files = self._version(name, version, scope)
        if path not in files:
            raise NotFound(f"no file {path}")
        return files[path]


#: The runs of :func:`kept_apart`: two tenants' users of one name, and one with skills of her own.
APART: Final = (("acme", "bob"), ("globex", "bob"), ("acme", "ada"))


async def kept_apart(seen: Callable[[SkillStore, str, str], Awaitable[str]]) -> None:
    """SEC: each of :data:`APART`'s runs, at once, is given its own tenant's ``sql-review`` (a
    user's own over the tenant's), never another tenant's: ``seen`` runs one (the store, its
    tenant, its user) and says what its framework's skills slot was given."""
    store = SkillStore()
    store.put("sql-review", "Acme's rules.", owner="acme")
    store.put("sql-review", "Globex's rules.", owner="globex")
    store.put("sql-review", "Ada's own rules.", level="user", owner="ada")
    acme_bob, globex_bob, acme_ada = await asyncio.gather(*(seen(store, *run) for run in APART))
    assert "Acme's rules." in acme_bob and "Globex" not in acme_bob
    assert "Globex's rules." in globex_bob and "Acme" not in globex_bob
    assert "Ada's own rules." in acme_ada  # her override wins over the tenant's
