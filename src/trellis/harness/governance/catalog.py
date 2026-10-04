"""The tool catalog, as governance reads it and publishes to it.

The memory service keeps one catalog per tenant: each tool's ``risk`` (what the service decided
from what the harness published and what an administrator set) and its ``approve_when`` (an
administrator's rule, or an accepted approval suggestion). :class:`Rules` keeps what it says
about the tools governance has been asked about fresh: read again every
:data:`GOVERNANCE_TTL_SECONDS` — conditionally, with the ``ETag`` of the last answer — so an
administrator's rule reaches running agents within seconds; a tool asked about for the first
time is read at once. One read at a time: concurrent calls that find it stale share one.

A catalog that cannot be read fails closed: rules read in the last
:data:`GOVERNANCE_STALE_SECONDS` still stand, and past that (or with none read) every tool that
does more than read asks (:data:`CATALOG_UNREAD`) until the catalog answers again; a warning is
logged once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from trellis.contracts import Feedback, ToolSpec
from trellis.memory import MemoryContext
from trellis.memory.models import SideEffects

log = logging.getLogger("trellis.governance")

#: The approval rule of a tool that does more than read when the catalog could not be read:
#: whether an administrator wants its calls approved is unknown, so every call asks.
CATALOG_UNREAD: Final = "<the tool catalog could not be read>"
#: How long the catalog's rules are kept before they are read again; a catalog that cannot be
#: read is asked again after the same interval.
GOVERNANCE_TTL_SECONDS: Final = 30.0
#: How long rules read earlier still stand while the catalog cannot be read; past it (or with
#: none read) every tool that does more than read asks.
GOVERNANCE_STALE_SECONDS: Final = 300.0


@dataclass(frozen=True, slots=True)
class Rule:
    """The catalog's word on one tool: the risk it decided, and the approval rule an
    administrator (or an accepted suggestion) set."""

    risk: SideEffects
    approve_when: str | None = None


class Catalog(Protocol):
    """Where the rules are read, the tools published, and a person's decisions learned from."""

    async def read(
        self, names: Sequence[str], *, etag: str | None = None
    ) -> tuple[dict[str, Rule] | None, str | None]:
        """The rule of each named tool it knows, and the answer's ``ETag``; ``None`` when
        nothing changed since the answer ``etag`` came with."""
        ...

    async def publish(self, entries: Sequence[Mapping[str, object]]) -> None: ...

    async def feedback(self, record: Feedback) -> None: ...


class MemoryCatalog:
    """The memory service's tool catalog, in ``ctx``'s scope (its tenant)."""

    __slots__ = ("ctx",)

    def __init__(self, ctx: MemoryContext) -> None:
        self.ctx = ctx

    async def read(
        self, names: Sequence[str], *, etag: str | None = None
    ) -> tuple[dict[str, Rule] | None, str | None]:
        """What the catalog says about each named tool — tools it does not know are absent —
        and the answer's ``ETag``. With the ``etag`` of an earlier answer the request is
        conditional (``If-None-Match``): ``None`` means nothing changed since. A service that
        sends no ``ETag`` is simply read in full each time."""
        entries, tag = await self.ctx.advanced.tools.catalog_if_changed(names, etag=etag)
        if entries is None:
            return None, tag
        rules = {e.name: Rule(risk=e.risk, approve_when=e.approve_when or None) for e in entries}
        return rules, tag

    async def publish(self, entries: Sequence[Mapping[str, object]]) -> None:
        await self.ctx.advanced.tools.put_catalog([dict(e) for e in entries])

    async def feedback(self, record: Feedback) -> None:
        """A person's decision on a call, as the ``TOOL_CALL`` feedback approval suggestions
        are learned from. Its id is the idempotency key: a retried send is stored once."""
        await self.ctx.feedback(record, idempotency_key=record.feedback_id)


def entry(spec: ToolSpec, annotations: Mapping[str, Any] | None) -> dict[str, object]:
    """A tool as the catalog stores it. ``side_effects`` only where they are known (a local
    tool declares them, an OpenAPI method implies them); an MCP tool sends its server's
    annotations instead and the service derives the risk, so an administrator's stays."""
    stored: dict[str, object] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema or {"type": "object"},
        "source": spec.source,
        "server": spec.server,
    }
    if spec.source != "mcp" and spec.side_effects in ("read", "write", "irreversible"):
        stored["side_effects"] = spec.side_effects
    if annotations:
        stored["annotations"] = dict(annotations)
    return stored


class Rules:
    """The catalog's rules for every tool asked about so far, kept fresh (see the module)."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._lock = asyncio.Lock()
        #: the tools asked about, in the order they were first asked about
        self._names: dict[str, None] = {}
        #: the catalog's word on them (``None``: unread), and when and under which ETag
        self._rules: dict[str, Rule] | None = None
        self._etag: str | None = None
        self._read_at = 0.0
        self._asked_at = 0.0
        self._unread = False

    async def get(self, names: Sequence[str]) -> dict[str, Rule] | None:
        """The rules of ``names`` (and every other tool asked about) as the catalog says now;
        ``None`` while it cannot be read."""
        async with self._lock:
            now = _now()
            new = [n for n in names if n not in self._names]
            if new:
                self._names.update(dict.fromkeys(new))
                self._etag = None  # other names: the next read is a full one
                self._asked_at = 0.0
            if self._names and now - self._asked_at > GOVERNANCE_TTL_SECONDS:
                await self._read(now)
            return self._rules

    async def _read(self, now: float) -> None:
        """Read the catalog again (conditionally); keep what was read while it cannot be."""
        self._asked_at = now
        try:
            rules, etag = await self._catalog.read(list(self._names), etag=self._etag)
        except Exception as exc:
            if not self._unread:
                log.warning(
                    "the tool catalog could not be read (%s: %s): until it can, every tool "
                    "that does more than read asks for approval",
                    type(exc).__name__,
                    exc,
                )
                self._unread = True
            if self._rules is not None and now - self._read_at > GOVERNANCE_STALE_SECONDS:
                self._rules, self._etag = None, None
            return
        if self._unread:
            log.info("the tool catalog can be read again")
            self._unread = False
        self._read_at, self._etag = now, etag
        if rules is not None:  # changed (a 304 keeps what was read under that ETag)
            self._rules = rules


@dataclass(slots=True)
class Published:
    """What this process has stored in one tenant's catalog (by content digest), and what is
    on its way there."""

    done: set[str] = field(default_factory=set)
    pending: set[str] = field(default_factory=set)

    def fresh(self, entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        """The entries neither stored nor on their way."""
        sent = self.done | self.pending
        return [e for e in entries if digest(e) not in sent]


def digest(entry: Mapping[str, object]) -> str:
    text = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def _now() -> float:
    return time.monotonic()
