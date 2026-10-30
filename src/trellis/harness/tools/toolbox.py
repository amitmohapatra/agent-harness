"""An agent's toolbox: the tools its builder declares (``ReAct(tools=, subagents=)``:
functions, sub-agents, ``a2a``, ``openapi``, the MCP servers it reaches itself — ``tools.mcp``)
and every MCP tool its Bifrost virtual key allows, published to the tool catalog, with Code
Mode where it is safe.

1. **list** — the local sources, and the gateway's MCP listing for the virtual key; an MCP
   server reached both directly and through the gateway is an error (one path per server)
   (kept :data:`TOOLS_TTL_SECONDS`; while a listing fails the last one stands, listed again
   after :data:`TOOLS_RETRY_SECONDS`); an MCP tool's side effects come
   from its server's annotations (:func:`side_effects_of`, idempotent when it says so), a local
   tool's from its declaration; a tool the gateway would run itself (its client's
   ``tools_to_auto_execute``: a call of it never reaches the harness) is left out, with a
   warning;
2. **lint** — what makes the tools hard for a model to choose among (``tools.lint``: a
   missing or short description, an undescribed parameter, a name a model API refuses, two
   tools described alike, too many at once), logged as warnings when the names listed change;
3. **publish** — every tool goes to the catalog (MCP tools with their annotations, local tools
   with their declared side effects) through governance, in the background, once per content;
   a tenant's first hints wait, at most :data:`INDEX_WAIT_SECONDS`, until tool search finds
   them (:meth:`Toolbox.indexed`): asked before, every hint is a zero;
4. **Code Mode** — the Code Mode servers whose tools all only read (as governance says now:
   the catalog's risk over the tools' own, and no approval rule), when there are at least
   :data:`CODE_MODE_MIN_SERVERS` of them or :data:`CODE_MODE_MIN_TOOLS` tools between them,
   are offered as Bifrost's Code Mode meta-tools (one script instead of many calls); every
   other tool stays a normal tool, so no script ever reaches a tool that writes.

Whether a call runs, is announced or asks is not the toolbox's: governance decides it at each
call (``trellis.harness.governance``, asked by the bridge).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.clients.bifrost import Gateway, code_mode_tools
from trellis.harness.fresh import Fresh
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.tools.base import Source, Tool, data_of
from trellis.harness.tools.lint import lint
from trellis.harness.tools.mcp import MCPSource, side_effects_of

if TYPE_CHECKING:
    from bifrost_sdk import ToolDef

log = logging.getLogger("trellis.tools")

#: Code Mode is chosen for the read-only Code Mode servers from this many servers, or tools.
CODE_MODE_MIN_SERVERS: Final = 3
CODE_MODE_MIN_TOOLS: Final = 20
#: How long the tools' definitions (the local sources, the gateway's MCP listing) are kept
#: before they are listed again; while listing them fails (the gateway is down), the last
#: definitions stand and are listed again after :data:`TOOLS_RETRY_SECONDS`.
TOOLS_TTL_SECONDS: Final = 300.0
TOOLS_RETRY_SECONDS: Final = 30.0
#: How long a toolbox's first hints wait for tool search to find the tools it published, and
#: how often it looks.
INDEX_WAIT_SECONDS: Final = 5.0
INDEX_POLL_SECONDS: Final = 0.25


@dataclass(frozen=True, slots=True)
class Listing:
    """What the toolbox holds: the local tools and the MCP definitions the key allows."""

    local: list[Tool]
    defs: list[ToolDef]
    #: the annotations of the tools of the MCP servers reached directly, by tool name
    hints: dict[str, dict[str, Any]] | None = None

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.local] + [d.name for d in self.defs]


async def listed(sources: Sequence[Source], *, gateway: Gateway | None) -> Listing:
    """The tools' definitions: the local sources and the MCP tools the virtual key allows,
    less those the gateway would run itself."""
    local: list[Tool] = []
    hints: dict[str, dict[str, Any]] = {}
    servers: Counter[str] = Counter()
    for source in sources:
        local.extend(await source.resolve())
        if isinstance(source, MCPSource):
            hints.update(source.annotations)
            servers[source.server or ""] += 1
    defs: list[ToolDef] = []
    if gateway is not None:
        defs = await gateway.tools()
        defs = await _not_auto_executed(gateway, defs) if defs else defs
    listing = Listing(local, defs, hints)
    both = sorted({d.client for d in defs} & set(servers))
    if both:
        raise ConfigurationError(
            f"the MCP server {', '.join(both)} is reached both through the gateway and by "
            "mcp(...): keep one path (name= renames a server of your own)"
        )
    twice = sorted(name for name, n in servers.items() if n > 1)
    if twice:
        raise ConfigurationError(
            f"two mcp(...) entries are the server {', '.join(twice)}: give one a name="
        )
    twice = sorted(name for name, n in Counter(listing.names).items() if n > 1)
    if twice:
        raise ConfigurationError(f"two tools are named {', '.join(twice)}")
    return listing


async def _not_auto_executed(gateway: Gateway, defs: list[ToolDef]) -> list[ToolDef]:
    """``defs`` less the tools in their client's ``tools_to_auto_execute`` (Agent Mode): the
    gateway answers the model's call of one itself, inside the completion, so it would never
    reach the harness's governance, journal or records. Not checked (and every tool kept, the
    gateway refusing such a call under the completions' deny-all scope) when the gateway's
    management API cannot be read."""
    try:
        auto = await gateway.auto_executed()
    except Exception as exc:
        log.info("the MCP clients' Agent Mode lists were not checked: %s", exc)
        return defs
    kept: list[ToolDef] = []
    for d in defs:
        names = auto.get(d.client, frozenset())
        if "*" in names or d.name.removeprefix(f"{d.client}-") in names:
            log.warning(
                "%s is not offered: its MCP client %s lists it in tools_to_auto_execute, so "
                "the gateway would run it itself, out of governance and the run's records; "
                "empty that list to use it",
                d.name,
                d.client,
            )
            continue
        kept.append(d)
    return kept


class Toolbox:
    """One agent's toolbox (``agent_id``'s) in one tenant, kept fresh: the definitions every
    :data:`TOOLS_TTL_SECONDS` (published through ``governance`` when they change; the last
    ones kept while listing fails), the Code Mode choice as governance's rules stand. One
    refresh at a time: concurrent runs that find it stale share one listing."""

    def __init__(
        self,
        sources: Sequence[Source],
        *,
        gateway: Gateway | None,
        governance: Governance,
        agent_id: str | None = None,
    ) -> None:
        self._sources = list(sources)
        #: whose toolbox it is: governance reads the rules learned for that agent
        self._agent_id = agent_id
        self._gateway = gateway
        self._governance = governance
        self._lock = asyncio.Lock()
        self._listings = Fresh(
            functools.partial(listed, self._sources, gateway=gateway),
            what="the tools (the local sources and the gateway's MCP listing)",
            ttl=TOOLS_TTL_SECONDS,
            retry=TOOLS_RETRY_SECONDS,
            fatal=(ConfigurationError,),
        )
        #: the listing the tools were built from
        self._listing: Listing | None = None
        #: the tools as the rules last stood — with Code Mode and without — and those rules
        self._tools: dict[bool, list[Tool]] = {}
        self._rules: dict[str, Rule | None] | None = None
        #: the tools Code Mode reaches through its meta-tools, as the tools were last built
        self._hidden: dict[bool, list[str]] = {}
        #: whether the first hints waited for the index already
        self._index_waited = False

    async def tools(self, *, code_mode: bool = True) -> list[Tool]:
        """The tools: with Code Mode where it fits (``code_mode=False``: the Code Mode
        servers' tools as normal tools — ``without={"code_mode"}``)."""
        async with self._lock:
            listing = await self._listings.get()
            if listing is not self._listing:  # listed again
                if self._listing is None or listing.names != self._listing.names:
                    _linted(listing)
                self._listing, self._tools = listing, {}
                await _publish(self._governance, listing)
            rules = await self._governance.rules(listing.names, agent_id=self._agent_id)
            if rules != self._rules:
                self._tools, self._rules = {}, rules
            if code_mode not in self._tools:
                self._tools[code_mode] = self._built(listing, rules, code_mode=code_mode)
            return list(self._tools[code_mode])

    def hidden(self, *, code_mode: bool = True) -> list[str]:
        """The tools behind Code Mode's meta-tools, as :meth:`tools` last built them: the hints
        may name them, and a run reaches them through the meta-tools."""
        return list(self._hidden.get(code_mode, ()))

    async def indexed(self, *, within: float = INDEX_WAIT_SECONDS) -> None:
        """Wait - the first time only, at most ``within`` seconds - until tool search finds the
        tools this toolbox published (``Governance.indexed``): hints asked before it are all
        zeros, and a narrowing on them offers the toolbox's first names in alphabetical order.
        A catalog that cannot be asked is not waited for."""
        if self._index_waited or self._listing is None:
            return
        self._index_waited = True
        names = _published(self._listing)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + within
        while True:
            try:
                if await self._governance.indexed(names):
                    return
            except Exception as exc:
                log.info("whether the tools are indexed could not be read: %s", exc)
                return
            if loop.time() >= deadline:
                log.info("the tools were not indexed in %.1f s: hints may miss them", within)
                return
            await asyncio.sleep(INDEX_POLL_SECONDS)

    def _built(
        self, listing: Listing, rules: dict[str, Rule | None], *, code_mode: bool
    ) -> list[Tool]:
        tools = list(listing.local)
        if self._gateway is not None:
            tools.extend(_mcp(self._gateway, listing, rules, code_mode=code_mode))
        offered = {t.name for t in tools}
        self._hidden[code_mode] = [d.name for d in listing.defs if d.name not in offered]
        return tools


def _mcp(
    gateway: Gateway, listing: Listing, rules: dict[str, Rule | None], *, code_mode: bool
) -> list[Tool]:
    tools = [
        Tool(
            _spec(
                d,
                side_effects=side_effects_of(d.annotations),
                idempotent=bool(d.annotations and d.annotations.idempotent_hint),
                **data_of(d.annotations, strict=False),
            ),
            functools.partial(gateway.execute, d.name, clients=(d.client,)),
            feature="mcp",
        )
        for d in listing.defs
    ]
    if not code_mode:
        return tools
    scriptable = _scriptable(listing.defs, {t.name: t for t in tools}, rules)
    if not scriptable:
        return tools
    return [
        *(t for t in tools if t.spec.server not in scriptable),
        *code_mode_tools(gateway, sorted(scriptable)),
    ]


def _scriptable(
    defs: list[ToolDef], tools: dict[str, Tool], rules: dict[str, Rule | None]
) -> set[str]:
    """The servers that go to Code Mode: Code Mode clients all of whose tools only read (the
    catalog's risk over their own, and no approval rule), when they are many enough."""

    def reads(tool: Tool) -> bool:
        rule = rules.get(tool.name)
        if rule is None:
            return tool.spec.side_effects == "read"
        return rule.risk == "read" and rule.approve_when is None

    by_server: dict[str, list[Tool]] = defaultdict(list)
    code_mode: dict[str, bool] = {}
    for d in defs:
        by_server[d.client].append(tools[d.name])
        code_mode[d.client] = code_mode.get(d.client, True) and d.code_mode
    servers = {
        server
        for server, found in by_server.items()
        if code_mode[server] and all(reads(t) for t in found)
    }
    count = sum(len(by_server[s]) for s in servers)
    large = len(servers) >= CODE_MODE_MIN_SERVERS or count >= CODE_MODE_MIN_TOOLS
    return servers if large else set()


def _linted(listing: Listing) -> None:
    """What makes the listed tools hard to choose among (``tools.lint``), each a warning."""
    specs = [t.spec for t in listing.local if t.spec.source != "memory"]
    for finding in lint([*specs, *(_spec(d) for d in listing.defs)]):
        log.warning("tool choice: %s", finding)


def _published(listing: Listing) -> list[str]:
    """The names :func:`_publish` sends: every tool but the memory service's own."""
    return [t.name for t in listing.local if t.spec.source != "memory"] + [
        d.name for d in listing.defs
    ]


async def _publish(governance: Governance, listing: Listing) -> None:
    """Every tool to the catalog: MCP tools with their annotations instead of side effects."""
    specs = [t.spec for t in listing.local if t.spec.source != "memory"]
    specs.extend(_spec(d) for d in listing.defs)
    hints = {
        d.name: d.annotations.model_dump(by_alias=True, exclude_none=True)
        for d in listing.defs
        if d.annotations
    }
    hints.update(listing.hints or {})
    await governance.publish(specs, annotations=hints)


def _spec(d: ToolDef, **fields: Any) -> ToolSpec:
    return ToolSpec(
        name=d.name,
        description=d.description,
        input_schema=d.parameters or {"type": "object"},
        source="mcp",
        server=d.client,
        **fields,
    )
