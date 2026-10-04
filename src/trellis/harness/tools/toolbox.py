"""An agent's toolbox: its own tools (``tools=[...]`` / ``h.tools``) and every MCP tool its
Bifrost virtual key allows, published to the tool catalog, with Code Mode where it is safe.

1. **list** — the local sources, and the gateway's MCP listing for the virtual key (kept
   :data:`TOOLS_TTL_SECONDS`); an MCP tool's side effects come from its server's annotations
   (:func:`side_effects_of`), a local tool's from its declaration;
2. **publish** — every tool goes to the catalog (MCP tools with their annotations, local tools
   with their declared side effects) through governance, in the background, once per content;
3. **Code Mode** — the Code Mode servers whose tools all only read (as governance says now:
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
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.clients.bifrost import Gateway, code_mode_tools
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.tools.base import Source, Tool

if TYPE_CHECKING:
    from bifrost_sdk import ToolDef

#: Code Mode is chosen for the read-only Code Mode servers from this many servers, or tools.
CODE_MODE_MIN_SERVERS: Final = 3
CODE_MODE_MIN_TOOLS: Final = 20
#: How long the tools' definitions (the local sources, the gateway's MCP listing) are kept
#: before they are listed again.
TOOLS_TTL_SECONDS: Final = 300.0


@dataclass(frozen=True, slots=True)
class Listing:
    """What the toolbox holds: the local tools and the MCP definitions."""

    local: list[Tool]
    defs: list[ToolDef]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.local] + [d.name for d in self.defs]


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


def side_effects_of(annotations: Any) -> str:
    """An MCP tool's side effects from its server's annotations (bifrost-sdk
    ``ToolAnnotations``). Hints, not guarantees: the catalog has the last word."""
    if annotations is not None and annotations.read_only_hint:
        return "read"
    if annotations is not None and annotations.destructive_hint:
        return "irreversible"
    return "write"


class Toolbox:
    """One agent's toolbox in one tenant, kept fresh: the definitions every
    :data:`TOOLS_TTL_SECONDS` (published through ``governance`` when they change), the Code
    Mode choice as governance's rules stand. One refresh at a time: concurrent runs that find
    it stale share one listing."""

    def __init__(
        self, sources: Sequence[Source], *, gateway: Gateway | None, governance: Governance
    ) -> None:
        self._sources = list(sources)
        self._gateway = gateway
        self._governance = governance
        self._lock = asyncio.Lock()
        self._listing: Listing | None = None
        self._listed_at = 0.0
        #: the tools as the rules last stood, and those rules
        self._tools: list[Tool] | None = None
        self._rules: dict[str, Rule | None] | None = None

    async def tools(self) -> list[Tool]:
        async with self._lock:
            now = _now()
            if self._listing is None or now - self._listed_at > TOOLS_TTL_SECONDS:
                self._listing = await listed(self._sources, gateway=self._gateway)
                self._listed_at = now
                self._tools = None
                await _publish(self._governance, self._listing)
            rules = await self._governance.rules(self._listing.names)
            if self._tools is None or rules != self._rules:
                self._tools, self._rules = self._built(self._listing, rules), rules
            return list(self._tools)

    def _built(self, listing: Listing, rules: dict[str, Rule | None]) -> list[Tool]:
        tools = list(listing.local)
        if self._gateway is not None:
            tools.extend(_mcp(self._gateway, listing.defs, rules))
        return tools


def _mcp(gateway: Gateway, defs: list[ToolDef], rules: dict[str, Rule | None]) -> list[Tool]:
    tools = [
        Tool(
            _spec(d, side_effects=side_effects_of(d.annotations)),
            functools.partial(gateway.execute, d.name, clients=(d.client,)),
        )
        for d in defs
    ]
    scriptable = _scriptable(defs, {t.name: t for t in tools}, rules)
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


async def _publish(governance: Governance, listing: Listing) -> None:
    """Every tool to the catalog: MCP tools with their annotations instead of side effects."""
    specs = [t.spec for t in listing.local if t.spec.source != "memory"]
    specs.extend(_spec(d) for d in listing.defs)
    hints = {
        d.name: d.annotations.model_dump(by_alias=True, exclude_none=True)
        for d in listing.defs
        if d.annotations
    }
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


def _now() -> float:
    return time.monotonic()
