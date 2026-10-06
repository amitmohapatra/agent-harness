"""The SELECTION dimension: what a user turns on (``SWITCHES``), the switches that do not exist
yet (``PENDING``, each a probe of the API the plan proposes), and the selections the matrix
runs — all on, nothing on, each switch alone on, each alone off, and all-pairs rows over the
switches (``allpairs``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from tests.matrix.allpairs import allpairs
from tests.matrix.model import PendingSwitch, Selection, Switch

if TYPE_CHECKING:
    from tests.matrix.world import World

SWITCHES: Final[tuple[Switch, ...]] = (
    Switch(
        "memory",
        "memory (push, pull, records)",
        "MEMORY_URL, or Harness(memory=client); off: memory=False",
    ),
    Switch(
        "gateway",
        "the Bifrost gateway (MCP tools, skills, prompts)",
        "BIFROST_URL, or Harness(gateway=); off: gateway=False",
    ),
    Switch("judges", "online judges", "Harness(judges=[...]) and TRELLIS_JUDGE_SAMPLE"),
    Switch(
        "grounding",
        "grounding checks",
        "TRELLIS_GROUNDING_SAMPLE (0 is off) with memory",
        requires=frozenset({"memory"}),
    ),
    Switch("tracing", "tracing (OTel spans)", "OTEL_EXPORTER_OTLP_ENDPOINT, or the app's provider"),
    Switch("agent_timeout", "the agent's time limit", "h.wrap(..., timeout=)"),
    Switch("version", "the agent's version", "h.wrap(..., version=) or TRELLIS_AGENT_VERSION"),
    Switch("hooks", "hooks around runs and tool calls", "h.wrap(hooks=[...]) / Harness(hooks=)"),
    # parts of a service, each off with without= while the service is on
    *(
        Switch(
            name, title, f"on with {'+'.join(needs)}; off: without={name!r}", frozenset(needs), name
        )
        for name, title, needs in (
            ("memory_push", "memory push (the context)", ("memory",)),
            ("memory_pull", "memory pull (the memory tools)", ("memory",)),
            ("records", "memory records", ("memory",)),
            # the hints come with the pushed context; Code Mode is scripts over the MCP tools
            ("hints", "tool hints", ("memory", "memory_push")),
            ("mcp", "the key's MCP tools (and Code Mode)", ("gateway",)),
            ("code_mode", "Code Mode", ("gateway", "mcp")),
            ("skills", "skills", ("gateway",)),
        )
    ),
)
BY_ID: Final = {s.id: s for s in SWITCHES}
ALL: Final = frozenset(BY_ID)


# --------------------------------------------------------------------------- pending switches
async def _without(world: World, name: str) -> None:
    """``h.wrap(..., without={name})``: one part off for an agent, the rest as deployed."""
    from tests.matrix.kit import Desk

    h = world.harness()
    proposed: dict[str, Any] = {"without": {name}}  # the argument the plan proposes
    agent = h.wrap(_echo, id=f"without.{name}", tools=[Desk().lookup()], **proposed)
    result = await agent.run("x", user="ada")
    assert result.status.value == "SUCCESS"


async def _echo(input: object, agent: object) -> object:
    return input


def _pending(name: str, title: str, gap: str = "G2") -> PendingSwitch:
    async def probe(world: World) -> None:
        await _without(world, name)

    return PendingSwitch(name, title, gap, probe)


#: Switches the plan adds and ``without=`` does not name yet (G2/G24): until they exist, a
#: selection naming one is one strict-xfail cell; when one lands, move it to ``SWITCHES``.
PENDING: Final[tuple[PendingSwitch, ...]] = (
    _pending("governance", "governance (a bare agent)"),
    _pending("redaction", "redaction (extra keys or off)", gap="G24"),
)


# --------------------------------------------------------------------------- selections
def closure(on: frozenset[str]) -> frozenset[str]:
    """``on`` with what each switch in it requires, and what those require."""
    found = set(on)
    while True:
        more = {r for name in found for r in BY_ID[name].requires} - found
        if not more:
            return frozenset(found)
        found |= more


def valid(row: Mapping[str, object]) -> bool:
    """A (partial) row of switch values breaks no ``requires``."""
    for name, value in row.items():
        if value:
            for needed in BY_ID[name].requires:
                if row.get(needed) is False:
                    return False
    return True


def selections() -> list[Selection]:
    """All on, nothing on, each switch alone on and alone off, then the all-pairs rows that
    are not one of those; then each pending switch alone on and alone off."""
    found: list[Selection] = [
        Selection("all", ALL, kind="all"),
        Selection("none", frozenset(), kind="none"),
    ]
    for switch in SWITCHES:
        found.append(Selection(f"only.{switch.id}", closure(frozenset({switch.id})), kind="only"))
    for switch in SWITCHES:
        off = {switch.id} | {s.id for s in SWITCHES if switch.id in s.requires}
        found.append(Selection(f"without.{switch.id}", ALL - off, kind="without"))
    seen = {s.on for s in found}
    rows = allpairs({s.id: (True, False) for s in SWITCHES}, valid)
    for n, row in enumerate(rows):
        on = frozenset(name for name, value in row.items() if value)
        if on not in seen:
            seen.add(on)
            found.append(Selection(f"pairs.{n}", on, kind="pairs"))
    for pending in PENDING:
        for kind in ("only", "without"):
            found.append(Selection(f"{kind}.{pending.id}", ALL, pending=pending, kind="pending"))
    return found


def pair_rows() -> list[frozenset[str]]:
    """The all-pairs rows over the switches (what the ``pairs.*`` selections come from)."""
    rows = allpairs({s.id: (True, False) for s in SWITCHES}, valid)
    return [frozenset(name for name, value in row.items() if value) for row in rows]


SELECTIONS: Final = selections()
SELECTION_BY_ID: Final = {s.id: s for s in SELECTIONS}
