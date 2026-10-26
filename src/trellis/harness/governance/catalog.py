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

Each refresh also checks the approval suggestions the service learned from people's decisions
and accepts each one not accepted yet (its rule becomes the tool's ``approve_when``) — logged,
and listed by ``trellis doctor`` — except for a tool at the highest risk (:data:`TOP_RISK`):
relaxing one of those waits for a person. The service learns them per agent (from that agent's
decisions) and lists and accepts them in the agent's scope only, so they are read and accepted
for each agent governance was asked about (:meth:`Rules.get`'s ``agent_id``), at most
:data:`AGENTS_KEPT` of them, the most recently asked.
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
from trellis.harness.bounded import LRUSet
from trellis.memory import MemoryContext
from trellis.memory.models import ApprovalSuggestion, SideEffects

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
#: The tools whose rules are kept fresh (the most recently asked about), and the published
#: entries remembered as stored (the most recent): bounds, not limits on the toolbox.
NAMES_KEPT: Final = 4096
PUBLISHED_KEPT: Final = 4096
#: The agents whose learned approval suggestions are checked at each refresh (the most
#: recently asked about): a bound, not a limit on the agents of a tenant.
AGENTS_KEPT: Final = 256
#: The risks a tool may declare, lowest first.
RISKS: Final[tuple[SideEffects, ...]] = ("read", "write", "irreversible")
#: The highest risk: a learned approval suggestion for such a tool is never accepted
#: automatically — a person accepts it.
TOP_RISK: Final[SideEffects] = RISKS[-1]


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

    async def indexed(self, names: Sequence[str]) -> bool:
        """Whether tool search finds the named tools as they were published: none of them
        waits for its index."""
        ...

    async def feedback(self, record: Feedback) -> None: ...

    async def suggestions(self, agent_id: str) -> Sequence[ApprovalSuggestion]:
        """The approval suggestions learned from people's decisions on ``agent_id``'s calls,
        accepted or not."""
        ...

    async def accept(self, suggestion_id: str, *, agent_id: str) -> Rule:
        """Accept a suggestion learned for ``agent_id``: its rule becomes its tool's
        ``approve_when``; the tool's rule as it is now."""
        ...


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

    async def indexed(self, names: Sequence[str]) -> bool:
        """No listed entry says it is not indexed yet (``indexed: false``); a service that
        does not say is taken as ready."""
        entries = await self.ctx.advanced.tools.catalog(names=list(names))
        return all((e.model_extra or {}).get("indexed") is not False for e in entries)

    async def feedback(self, record: Feedback) -> None:
        """A person's decision on a call, as the ``TOOL_CALL`` feedback approval suggestions
        are learned from. Its id is the idempotency key: a retried send is stored once."""
        await self.ctx.feedback(record, idempotency_key=record.feedback_id)

    async def suggestions(self, agent_id: str) -> Sequence[ApprovalSuggestion]:
        """The suggestions learned for ``agent_id``: the service lists those of the agent its
        scope names (none for a tenant-wide scope)."""
        return await self._agent(agent_id).advanced.tools.approval_suggestions()

    async def accept(self, suggestion_id: str, *, agent_id: str) -> Rule:
        """Accepted in ``agent_id``'s scope: the service accepts a suggestion only for the
        agent it was learned for."""
        tools = self._agent(agent_id).advanced.tools
        accepted = await tools.accept_suggestion(suggestion_id)
        return Rule(risk=accepted.risk, approve_when=accepted.approve_when or None)

    def _agent(self, agent_id: str) -> MemoryContext:
        """This catalog's scope, for one agent."""
        if self.ctx.scope.agent_id == agent_id:
            return self.ctx
        return MemoryContext(
            self.ctx.client, self.ctx.scope.model_copy(update={"agent_id": agent_id})
        )


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
        #: the tools asked about, the least recently asked forgotten past the bound
        self._names: LRUSet[str] = LRUSet(maxsize=NAMES_KEPT)
        #: the agents asked for, whose learned suggestions each refresh checks
        self._agents: LRUSet[str] = LRUSet(maxsize=AGENTS_KEPT)
        #: the catalog's word on them (``None``: unread), and when and under which ETag
        self._rules: dict[str, Rule] | None = None
        self._etag: str | None = None
        self._read_at = 0.0
        self._asked_at = 0.0
        self._unread = False
        #: the agents whose suggestions could not be read at the last try (said once each)
        self._suggestions_unread: set[str] = set()

    async def get(
        self, names: Sequence[str], *, agent_id: str | None = None
    ) -> dict[str, Rule] | None:
        """The rules of ``names`` (and every other tool asked about) as the catalog says now;
        ``None`` while it cannot be read. ``agent_id`` is the agent asking: the suggestions
        learned from its decisions are checked from now on (at once for a new one)."""
        async with self._lock:
            now = _now()
            new = [n for n in names if n not in self._names]
            for name in names:
                self._names.add(name)
            if new:
                self._etag = None  # other names: the next read is a full one
                self._asked_at = 0.0
            if agent_id and agent_id not in self._agents:
                self._agents.add(agent_id)
                self._asked_at = 0.0  # its suggestions are checked now
            elif agent_id:
                self._agents.add(agent_id)  # the most recently asked
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
        await self._accept_learned()

    async def _accept_learned(self) -> None:
        """For each agent asked for: accept each approval suggestion learned from its
        decisions for a tool asked about, not accepted yet, unless the tool is at
        :data:`TOP_RISK` (a person accepts those). A suggestion that cannot be read or accepted
        now is checked again at the next refresh. The agents are asked at once: a refresh takes
        one round trip however many agents the tenant runs here."""
        agents = list(self._agents)
        self._suggestions_unread &= set(agents)  # an agent forgotten past the bound
        await asyncio.gather(*(self._accepted_for(agent_id) for agent_id in agents))

    async def _accepted_for(self, agent_id: str) -> None:
        try:
            found = await self._catalog.suggestions(agent_id)
        except Exception as exc:
            if agent_id not in self._suggestions_unread:
                log.warning(
                    "the approval suggestions could not be read: none is accepted until they "
                    "can be",
                    extra={"reason": type(exc).__name__, "agent_id": agent_id},
                )
                self._suggestions_unread.add(agent_id)
            return
        self._suggestions_unread.discard(agent_id)
        for suggestion in found:
            rule = (self._rules or {}).get(suggestion.tool)
            if suggestion.accepted or rule is None or rule.risk == TOP_RISK:
                continue
            try:
                accepted = await self._catalog.accept(suggestion.id, agent_id=agent_id)
            except Exception as exc:
                log.warning(
                    "an approval suggestion could not be accepted",
                    extra={
                        "agent_id": agent_id,
                        "tool": suggestion.tool,
                        "suggestion_id": suggestion.id,
                        "reason": type(exc).__name__,
                    },
                )
                continue
            self._rules = {**(self._rules or {}), suggestion.tool: accepted}
            log.info(
                "an approval suggestion was accepted automatically",
                extra={
                    "agent_id": agent_id,
                    "tool": suggestion.tool,
                    "suggestion_id": suggestion.id,
                    "suggestion": suggestion.suggestion,
                    "approve_when": accepted.approve_when,
                },
            )


@dataclass(slots=True)
class Published:
    """What this process has stored in one tenant's catalog (by content digest), and what is
    on its way there."""

    #: stored, the least recently stored forgotten past the bound (then sent again: the
    #: catalog's upsert is idempotent)
    done: LRUSet[str] = field(default_factory=lambda: LRUSet(maxsize=PUBLISHED_KEPT))
    pending: set[str] = field(default_factory=set)

    def fresh(self, entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        """The entries neither stored nor on their way."""
        return [e for e in entries if (d := digest(e)) not in self.done and d not in self.pending]


def digest(entry: Mapping[str, object]) -> str:
    text = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def _now() -> float:
    return time.monotonic()
