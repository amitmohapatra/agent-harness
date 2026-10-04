"""An agent's toolbox: its own tools (``tools=[...]`` / ``h.tools``), every MCP tool its Bifrost
virtual key allows, each with its risk tier and approval rule — and the catalog kept current.

1. **list** — the local sources, and the gateway's MCP listing for the virtual key (kept
   :data:`TOOLS_TTL_SECONDS`);
2. **tier** — an MCP tool's side effects come from its server's annotations, a local tool's
   from its declaration; the catalog's ``risk`` (what the memory service decided) overrides
   either, and its ``approve_when`` becomes the tool's approval rule (``tools.policy``). The
   catalog is read again every :data:`GOVERNANCE_TTL_SECONDS` — conditionally, with the
   ``ETag`` of the last answer — so an administrator's rule reaches running agents within
   seconds. A catalog that cannot be read leaves the tools their own tiers, except that every
   tool that does more than read asks for approval (``policy.CATALOG_UNREAD``) until it can;
3. **Code Mode** — the Code Mode servers whose tools all only read, when there are at least
   :data:`CODE_MODE_MIN_SERVERS` of them or :data:`CODE_MODE_MIN_TOOLS` tools between them,
   are offered as Bifrost's Code Mode meta-tools (one script instead of many calls); every
   other tool stays a normal tool, so no script ever reaches a tool that writes;
4. **publish** — every tool goes to the catalog (MCP tools with their annotations, local tools
   with their declared side effects), in the background, once per content — once it was
   stored: a publish that failed is sent again at the next listing.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.clients.bifrost import Gateway, code_mode_tools
from trellis.harness.clients.memory import Governance, RunMemory, catalog_entry
from trellis.harness.tools.base import Source, Tool
from trellis.harness.tools.policy import CATALOG_UNREAD, side_effects_of

if TYPE_CHECKING:
    from bifrost_sdk import ToolDef

    from trellis.harness.writes import Writes

log = logging.getLogger("trellis.tools")

#: Code Mode is chosen for the read-only Code Mode servers from this many servers, or tools.
CODE_MODE_MIN_SERVERS: Final = 3
CODE_MODE_MIN_TOOLS: Final = 20
#: How long the tools' definitions (the local sources, the gateway's MCP listing) are kept
#: before they are listed again.
TOOLS_TTL_SECONDS: Final = 300.0
#: How long the catalog's governance (tiers, ``approve_when``) is kept before it is read
#: again; a catalog that cannot be read is asked again after the same interval.
GOVERNANCE_TTL_SECONDS: Final = 30.0
#: How long governance read earlier still stands while the catalog cannot be read; past it
#: (or with none read) every tool that does more than read asks.
GOVERNANCE_STALE_SECONDS: Final = 300.0


@dataclass(frozen=True, slots=True)
class Listing:
    """What the toolbox holds before governance: the local tools and the MCP definitions."""

    local: list[Tool]
    defs: list[ToolDef]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.local] + [d.name for d in self.defs]


@dataclass(slots=True)
class Published:
    """What this process has stored in one tenant's catalog (by content digest), and what is
    on its way there."""

    done: set[str] = field(default_factory=set)
    pending: set[str] = field(default_factory=set)


async def listed(sources: Sequence[Source], *, gateway: Gateway | None) -> Listing:
    """The tools' definitions: the local sources and the MCP tools the virtual key allows."""
    local: list[Tool] = []
    for source in sources:
        local.extend(await source.resolve())
    defs = await gateway.tools() if gateway is not None else []
    listing = Listing(local, defs)
    twice = sorted(name for name, n in Counter(listing.names).items() if n > 1)
    if twice:
        raise ConfigurationError(f"two tools are named {', '.join(twice)}")
    return listing


def governed(
    listing: Listing, *, gateway: Gateway | None, entries: dict[str, Governance] | None
) -> list[Tool]:
    """The toolbox: the listing tiered by the catalog's ``entries`` (``None``: the catalog
    could not be read — own tiers, and every tool that does more than read asks)."""
    tools = [_governed(t, _entry(t.spec, entries)) for t in listing.local]
    if gateway is not None:
        tools.extend(_mcp(gateway, listing.defs, entries))
    return tools


async def resolve(
    sources: Sequence[Source],
    *,
    gateway: Gateway | None,
    catalog: RunMemory | None,
    writes: Writes,
    published: Published,
) -> list[Tool]:
    """The toolbox once (what ``h.tools`` builds an agent with). ``catalog`` is the memory
    service in the tenant's scope (``None`` with memory off: tiers are then the tools' own)."""
    box = Toolbox(sources, gateway=gateway, catalog=catalog, writes=writes, published=published)
    return await box.tools()


class Toolbox:
    """One agent's toolbox in one tenant, kept fresh: the definitions every
    :data:`TOOLS_TTL_SECONDS`, the governance every :data:`GOVERNANCE_TTL_SECONDS`. One
    refresh at a time: concurrent runs that find it stale share one read."""

    def __init__(
        self,
        sources: Sequence[Source],
        *,
        gateway: Gateway | None,
        catalog: RunMemory | None,
        writes: Writes,
        published: Published,
    ) -> None:
        self._sources = list(sources)
        self._gateway = gateway
        self._catalog = catalog
        self._writes = writes
        self._published = published
        self._lock = asyncio.Lock()
        self._listing: Listing | None = None
        self._listed_at = 0.0
        #: the catalog's word on the listing, and when and under which ETag it was read
        self._entries: dict[str, Governance] | None = {} if catalog is None else None
        self._etag: str | None = None
        self._read_at = 0.0
        self._asked_at = 0.0
        self._unread = False
        self._tools: list[Tool] | None = None

    async def tools(self) -> list[Tool]:
        async with self._lock:
            now = _now()
            if self._listing is None or now - self._listed_at > TOOLS_TTL_SECONDS:
                self._listing = await listed(self._sources, gateway=self._gateway)
                self._listed_at = now
                self._etag = None  # other names: the next read is a full one
                self._asked_at = 0.0
                self._tools = None
                if self._catalog is not None:
                    await _publish(self._catalog, self._listing, self._writes, self._published)
            if self._catalog is not None and now - self._asked_at > GOVERNANCE_TTL_SECONDS:
                await self._govern(self._catalog, self._listing, now)
            if self._tools is None:
                entries = self._entries if self._listing.names else {}
                self._tools = governed(self._listing, gateway=self._gateway, entries=entries)
            return list(self._tools)

    async def _govern(self, catalog: RunMemory, listing: Listing, now: float) -> None:
        """Read the catalog again (conditionally); keep what was read while it cannot be."""
        self._asked_at = now
        if not listing.names:
            return
        try:
            entries, etag = await catalog.catalog(listing.names, etag=self._etag)
        except Exception as exc:
            if not self._unread:
                log.warning(
                    "the tool catalog could not be read (%s: %s): until it can, every tool "
                    "that does more than read asks for approval",
                    type(exc).__name__,
                    exc,
                )
                self._unread = True
            if self._entries is not None and now - self._read_at > GOVERNANCE_STALE_SECONDS:
                self._entries, self._etag, self._tools = None, None, None
            return
        if self._unread:
            log.info("the tool catalog can be read again")
            self._unread = False
        self._read_at, self._etag = now, etag
        if entries is not None:  # changed (a 304 keeps what was read under that ETag)
            self._entries, self._tools = entries, None


def _governed(tool: Tool, entry: Governance | None) -> Tool:
    """The tool with the catalog's word on it: the tier it decided, and its approval rule."""
    if entry is None:
        return tool
    spec = tool.spec
    if entry.risk != spec.side_effects:
        spec = spec.model_copy(update={"side_effects": entry.risk})
    return Tool(spec, tool.run, code_mode=tool.code_mode, approve_when=entry.approve_when)


def _entry(spec: ToolSpec, entries: dict[str, Governance] | None) -> Governance | None:
    """The catalog's word on a tool; with the catalog unread, a tool that does more than read
    asks."""
    if entries is not None:
        return entries.get(spec.name)
    if spec.side_effects == "read":
        return None
    return Governance(risk=spec.side_effects, approve_when=CATALOG_UNREAD)  # type: ignore[arg-type]


def _mcp(
    gateway: Gateway, defs: list[ToolDef], entries: dict[str, Governance] | None
) -> list[Tool]:
    own = [
        Tool(
            ToolSpec(
                name=d.name,
                description=d.description,
                input_schema=d.parameters or {"type": "object"},
                source="mcp",
                server=d.client,
                side_effects=side_effects_of(d.annotations),
            ),
            functools.partial(gateway.execute, d.name, clients=(d.client,)),
        )
        for d in defs
    ]
    tools = [_governed(t, _entry(t.spec, entries)) for t in own]
    scriptable = _scriptable(defs, {t.name: t for t in tools})
    if not scriptable:
        return tools
    return [
        *(t for t in tools if t.spec.server not in scriptable),
        *code_mode_tools(gateway, sorted(scriptable)),
    ]


def _scriptable(defs: list[ToolDef], tools: dict[str, Tool]) -> set[str]:
    """The servers that go to Code Mode: Code Mode clients all of whose tools only read (and
    carry no approval rule), when they are many enough."""
    by_server: dict[str, list[Tool]] = defaultdict(list)
    code_mode: dict[str, bool] = {}
    for d in defs:
        by_server[d.client].append(tools[d.name])
        code_mode[d.client] = code_mode.get(d.client, True) and d.code_mode
    servers = {
        server
        for server, found in by_server.items()
        if code_mode[server]
        and all(t.spec.side_effects == "read" and t.approve_when is None for t in found)
    }
    count = sum(len(by_server[s]) for s in servers)
    large = len(servers) >= CODE_MODE_MIN_SERVERS or count >= CODE_MODE_MIN_TOOLS
    return servers if large else set()


async def _publish(
    catalog: RunMemory, listing: Listing, writes: Writes, published: Published
) -> None:
    entries = [catalog_entry(t.spec, None) for t in listing.local if t.spec.source != "memory"]
    for d in listing.defs:
        spec = ToolSpec(
            name=d.name,
            description=d.description,
            input_schema=d.parameters or {"type": "object"},
            source="mcp",
            server=d.client,
        )
        hints = (
            d.annotations.model_dump(by_alias=True, exclude_none=True) if d.annotations else None
        )
        entries.append(catalog_entry(spec, hints))
    fresh = [e for e in entries if _digest(e) not in published.done | published.pending]
    if not fresh:
        return
    digests = {_digest(e) for e in fresh}
    published.pending |= digests

    async def publish() -> None:
        try:
            await catalog.publish_catalog(fresh)
            published.done |= digests  # stored: not sent again
        finally:
            published.pending -= digests  # failed: sent again at the next listing

    await writes.submit(
        "memory.tool_catalog",
        publish,
        record=catalog.record("publish_catalog", entries=fresh),
    )


def _now() -> float:
    return time.monotonic()


def _digest(entry: dict[str, Any]) -> str:
    text = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()
