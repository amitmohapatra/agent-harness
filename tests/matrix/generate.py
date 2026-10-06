"""Every cell of the matrix, generated from the tables: FEATURES (and the selection row and the
extension points) x ADAPTERS x WAYS x MODES x SELECTIONS, each a test that runs, a skip that
says why it does not apply, or a strict xfail naming the gap or bug it waits on.

Which selections a feature runs under: everything on and nothing on; when it needs switches,
those alone on and each of them alone off; the selection row (``SEL``) runs under every
selection. A cell that does not apply is listed once (under ``all``), a Way 2 cell too (Way 2
selects by import).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final

from tests.matrix.dimensions import SELECTION_BY_ID, SELECTIONS
from tests.matrix.extensions import EXTENSIONS
from tests.matrix.features import FEATURES, SELECTION
from tests.matrix.model import (
    ADAPTERS,
    MODES,
    NA,
    WAYS,
    Bug,
    Feature,
    Gap,
    Note,
    Selection,
    cell_id,
)
from tests.matrix.world import (
    MemoryContract,
    NoEnding,
    NotTimedOut,
    OffButCalled,
    OffButOffered,
    UnclosedToolCall,
)

#: Why a way does not apply to an adapter (whatever the feature).
NOT_BLOCKS: Final = NA(
    "ReAct with blocks is ReAct's way; another target with the team's blocks is Way 1 "
    "(Harness(runs=, memory=, gateway=, governance=))"
)
WAY2_ADAPTER: Final = NA(
    "a block is framework-neutral (your code calls it whatever the framework): run once as "
    "plain code; the framework recipes are the W2R row"
)
WAY2_MODE: Final = NA("a block runs inline in your code: no surface or queue of its own here")
NO_WAY2: Final = NA("no Way 2 block")


# --------------------------------------------------------------------------- known bugs
def _on(*switches: str) -> Callable[[str, str, str, str, Selection], bool]:
    return lambda feature, adapter, way, mode, selection: set(switches) <= selection.on


#: The real failures this matrix found (BUG-1, Claude running a call the CLI made after a pause,
#: was fixed by the Claude session and governance work merged since; it is not listed): where each holds (feature ids, adapters, ways, modes,
#: and a test on the selection), the bug, and how it fails (``raises``: any other failure of
#: the cell is still a failure). The reproduction of each is its cell's id
#: (``pytest tests/matrix -k <cell>``) and the script in the bug's ``why``. Remove an entry
#: when its bug is fixed: its cells then XPASS and fail the suite until it is.
KNOWN: Final[list[tuple[dict[str, Any], Bug]]] = [
    (
        {"features": {"F09", "F09r"}, "modes": {"worker", "elsewhere", "schedule"}},
        Bug(
            "BUG-7",
            "a queued run past its time limit: the runs SDK Worker's own timeout (G34) cancels "
            "the attempt first: it ends CANCELLED (no error), or TIMEOUT with no RUN_FINISHED "
            "on its event stream",
            raises=(NotTimedOut, NoEnding, UnclosedToolCall),
        ),
    ),
    (
        {"features": {"F05"}, "modes": {"a2a"}},
        Bug(
            "BUG-5",
            "A2A tasks/cancel of a working task: 'Task not found', or answered while the run "
            "goes on to its end",
        ),
    ),
    (
        # Claude: whether the time limit falls inside a tool call or between two is a race
        {"features": {"F09", "F09r"}, "adapters": {"claude"}},
        Bug(
            "BUG-2",
            "a tool call cut short by the run's time limit never ends on the event stream "
            "(Claude: when the limit falls inside a call)",
            raises=UnclosedToolCall,
            strict=False,
        ),
    ),
    (
        {"features": {"F36", "F05", "F09", "F09r"}},
        Bug(
            "BUG-2",
            "a tool call cut short (an ask inside it, a cancel, the run's time limit) never "
            "ends on the event stream: TOOL_CALL_START without TOOL_CALL_END/RESULT",
            raises=UnclosedToolCall,
        ),
    ),
    (
        {"features": {"F26"}, "adapters": {"react"}, "when": _on("grounding", "memory")},
        Bug(
            "BUG-3",
            "grounding sends answers over 8000 characters to /v1/verify (its maxLength): the "
            "memory service refuses them",
            raises=MemoryContract,
        ),
    ),
    (
        {"features": {"F61"}, "adapters": {"react"}, "when": _on("tracing")},
        Bug(
            "BUG-4",
            "ReAct's chat spans carry the conversation unredacted (tool-call arguments are "
            "JSON text the redactor does not parse; tool results as they are)",
            raises=AssertionError,
        ),
    ),
    (
        {
            "features": {"F42"},
            "adapters": {"langgraph", "deepagents"},
            "when": lambda f, a, w, m, s: "memory" in s.on and "memory_pull" not in s.on,
        },
        Bug(
            "BUG-10",
            "LangGraph/Deep Agents: without={'memory_pull'} still offers the memory tools "
            "h.tools() bound into the graph (a call is refused: 'off in this run')",
            raises=OffButOffered,
        ),
    ),
    (
        {"when": lambda f, a, w, m, s: "gateway" in s.on and "mcp" not in s.on},
        Bug(
            "BUG-9",
            "without={'mcp'}: the toolbox still lists the key's MCP tools from the gateway "
            "(and publishes them to the catalog) though none is offered",
            raises=OffButCalled,
        ),
    ),
    (
        {"features": {"F12"}, "adapters": {"claude"}},
        Bug(
            "BUG-6",
            "Claude: a tool result over 1 MiB fails the run (CLIJSONDecodeError: the SDK's "
            "1 MiB buffer); nothing cuts it first (G8)",
            raises=AssertionError,
        ),
    ),
    (
        {"features": {"F17", "F18"}, "adapters": {"claude"}, "when": _on("gateway")},
        Bug(
            "BUG-8",
            "Claude: a tool whose input schema has no 'properties' ({'type': 'object'}, as MCP "
            "servers declare an argument-less tool) is offered with a required 'type' argument "
            "(the SDK reads it as a name -> type map): every call fails validation",
            raises=AssertionError,
        ),
    ),
]


def _known(feature: str, adapter: str, way: str, mode: str, selection: Selection) -> Bug | None:
    for where, bug in KNOWN:
        if (
            feature in where.get("features", {feature})
            and adapter in where.get("adapters", {adapter})
            and way in where.get("ways", {way})
            and mode in where.get("modes", {mode})
            and where.get("when", lambda *_: True)(feature, adapter, way, mode, selection)
        ):
            return bug
    return None


@dataclass(frozen=True)
class Cell:
    id: str
    feature: Feature
    adapter: str
    way: str
    mode: str
    selection: Selection
    note: Note | None = None


def _note(feature: Feature, adapter: str, way: str, mode: str) -> Note | None:
    """Why the cell does not run as a plain test, whatever the selection (None: it does)."""
    if way == "react_with_blocks" and adapter != "react":
        return NOT_BLOCKS
    if way == "way2":
        return _way2_note(feature, adapter, mode)
    for key in (
        (adapter, way, mode),
        (adapter, None, mode),
        (adapter, way, None),
        (None, way, mode),
        (None, None, mode),
        (adapter, None, None),
    ):
        if key in feature.cells:
            return feature.cells[key]
    named = (feature.adapters.get(adapter), feature.modes.get(mode), feature.ways.get(way))
    found = [n for n in named if n]
    not_here = [n for n in found if isinstance(n, NA)]
    return (not_here or found or [None])[0]


def _way2_note(feature: Feature, adapter: str, mode: str) -> Note | None:
    if adapter != "function":
        return WAY2_ADAPTER
    if feature.way2 is None:
        return NO_WAY2
    if mode not in feature.way2_modes:
        return WAY2_MODE
    if isinstance(feature.way2, NA | Gap | Bug):
        return feature.way2
    return feature.way2_gap


def _selections(feature: Feature) -> list[Selection]:
    if feature is SELECTION:
        return [s for s in SELECTIONS if s.pending is None]
    chosen = ["all", "none"]
    if feature.needs:
        from tests.matrix.dimensions import closure

        alone = closure(feature.needs)
        chosen += [s.id for s in SELECTIONS if s.kind == "only" and s.on == alone][:1]
        chosen += [f"without.{n}" for n in sorted(feature.needs)]
    return [SELECTION_BY_ID[s] for s in dict.fromkeys(chosen)]


def on_in(feature: Feature, way: str, selection: Selection) -> bool:
    from tests.matrix.world import ON_WITH_BLOCKS

    return feature.needs <= selection.on or (
        way == "react_with_blocks" and feature.id in ON_WITH_BLOCKS
    )


def cells() -> Iterator[Cell]:
    everything = SELECTION_BY_ID["all"]
    for feature in [*FEATURES, SELECTION, *EXTENSIONS]:
        for adapter in ADAPTERS:
            for way in WAYS:
                for mode in MODES:
                    note = _note(feature, adapter, way, mode)
                    if isinstance(note, NA) or way == "way2":
                        yield Cell(
                            cell_id(feature.id, adapter, way, mode, "all"),
                            feature,
                            adapter,
                            way,
                            mode,
                            everything,
                            note,
                        )
                        continue
                    for selection in _selections(feature):
                        # a gap or bug of the feature holds where the feature is on
                        held = note if on_in(feature, way, selection) else None
                        if held is None and way != "way2":
                            held = _known(feature.id, adapter, way, mode, selection)
                        yield Cell(
                            cell_id(feature.id, adapter, way, mode, selection.id),
                            feature,
                            adapter,
                            way,
                            mode,
                            selection,
                            held,
                        )
    for selection in SELECTIONS:
        if selection.pending is not None:
            pending = selection.pending
            yield Cell(
                cell_id("SEL", "function", "way1", "run", selection.id),
                SELECTION,
                "function",
                "way1",
                "run",
                selection,
                Gap(pending.gap, f"no switch for {pending.title} (without=)"),
            )


CELLS: Final = list(cells())
