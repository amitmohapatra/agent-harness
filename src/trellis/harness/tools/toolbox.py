"""An agent's toolbox: its own tools (``tools=[...]`` / ``h.tools``), every MCP tool its Bifrost
virtual key allows, each with its risk tier and approval rule — and the catalog kept current.

1. **list** — the local sources, and the gateway's MCP listing for the virtual key;
2. **tier** — an MCP tool's side effects come from its server's annotations, a local tool's
   from its declaration; the catalog's ``side_effects`` overrides either, and its
   ``approve_when`` becomes the tool's approval rule (``tools.policy``);
3. **Code Mode** — the Code Mode servers whose tools all only read, when there are at least
   :data:`CODE_MODE_MIN_SERVERS` of them or :data:`CODE_MODE_MIN_TOOLS` tools between them,
   are offered as Bifrost's Code Mode meta-tools (one script instead of many calls); every
   other tool stays a normal tool, so no script ever reaches a tool that writes;
4. **publish** — every tool goes to the catalog (MCP tools with their annotations, local tools
   with their declared side effects), in the background, once per content.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.clients.bifrost import Gateway, code_mode_tools
from trellis.harness.clients.memory import Governance, RunMemory, catalog_entry
from trellis.harness.tools.base import Source, Tool
from trellis.harness.tools.policy import side_effects_of

if TYPE_CHECKING:
    from bifrost_sdk import ToolDef

    from trellis.harness.writes import Writes

#: Code Mode is chosen for the read-only Code Mode servers from this many servers, or tools.
CODE_MODE_MIN_SERVERS: Final = 3
CODE_MODE_MIN_TOOLS: Final = 20


async def resolve(
    sources: Sequence[Source],
    *,
    gateway: Gateway | None,
    catalog: RunMemory | None,
    writes: Writes,
    published: set[str],
) -> list[Tool]:
    """The toolbox. ``catalog`` is the memory service in the tenant's scope (``None`` with
    memory off: tiers are then the tools' own); ``published`` remembers what this process
    already sent to the catalog."""
    local: list[Tool] = []
    for source in sources:
        local.extend(await source.resolve())
    defs = await gateway.tools() if gateway is not None else []
    names = [t.name for t in local] + [d.name for d in defs]
    twice = sorted(name for name, n in Counter(names).items() if n > 1)
    if twice:
        raise ConfigurationError(f"two tools are named {', '.join(twice)}")
    entries = await catalog.catalog(names) if catalog is not None and names else {}
    if catalog is not None:
        _publish(catalog, local, defs, writes, published)
    tools = [_governed(t, entries.get(t.name)) for t in local]
    if gateway is not None:
        tools.extend(_mcp(gateway, defs, entries))
    return tools


def _governed(tool: Tool, entry: Governance | None) -> Tool:
    """The tool with the catalog's word on it: the tier it decided, and its approval rule."""
    if entry is None:
        return tool
    spec = tool.spec
    if entry.risk != spec.side_effects:
        spec = spec.model_copy(update={"side_effects": entry.risk})
    return Tool(spec, tool.run, code_mode=tool.code_mode, approve_when=entry.approve_when)


def _mcp(gateway: Gateway, defs: list[ToolDef], entries: dict[str, Governance]) -> list[Tool]:
    tools = [
        _governed(
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
            ),
            entries.get(d.name),
        )
        for d in defs
    ]
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


def _publish(
    catalog: RunMemory,
    local: list[Tool],
    defs: list[ToolDef],
    writes: Writes,
    published: set[str],
) -> None:
    entries = [catalog_entry(t.spec, None) for t in local if t.spec.source != "memory"]
    for d in defs:
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
    fresh = [e for e in entries if _digest(e) not in published]
    if not fresh:
        return
    published.update(_digest(e) for e in fresh)
    writes.submit("memory.tool_catalog", lambda: catalog.publish_catalog(fresh))


def _digest(entry: dict[str, Any]) -> str:
    text = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()
