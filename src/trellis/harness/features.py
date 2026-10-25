"""What a wrapped agent does around every run, and the one switch that turns parts of it off:
``without=`` — on ``h.wrap`` for every run of the agent, on ``agent.run``/``stream``/``start``
for one run (added to the agent's). Everything configured is on unless named here; nothing
else is a setting. Six names, each one part (:data:`NAMES`):

* ``memory`` — the memory context pushed, the memory tools, the records, the tool narrowing;
* ``tool_search`` — the tool narrowing and the memory service's ``tool_search`` tool;
* ``mcp`` — the gateway's MCP tools, and Code Mode, whose scripts reach them;
* ``code_mode`` — Code Mode alone;
* ``grounding`` — the check of an answer against the context it was given;
* ``judges`` — the online judges.

A run's own ``without=`` is kept with its record (``RunStart.metadata``, as the parts it covers),
so every attempt — a resume, a worker's after a crash — and every child run it starts works
without the same.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, cast, get_args

from trellis.contracts import ConfigurationError, RunRecord

#: What ``without=`` names.
Name = Literal["memory", "tool_search", "mcp", "code_mode", "grounding", "judges"]
#: The parts of the harness's work a run asks about (``Runtime.uses``): what the names cover.
Feature = Literal[
    "memory_push",
    "memory_pull",
    "records",
    "hints",
    "tool_search",
    "mcp",
    "code_mode",
    "grounding",
    "judges",
]
#: The names, in the order they are said.
NAMES: Final[tuple[str, ...]] = get_args(Name)
#: The parts each name turns off.
COVERS: Final[dict[str, tuple[Feature, ...]]] = {
    "memory": ("memory_push", "memory_pull", "records", "hints", "tool_search"),
    "tool_search": ("hints", "tool_search"),
    "mcp": ("mcp", "code_mode"),
    "code_mode": ("code_mode",),
    "grounding": ("grounding",),
    "judges": ("judges",),
}
#: The parts of the memory service's agent tools (``Harness.pull_tools``): its pull tools and
#: ``tool_search``.
MEMORY_TOOLS: Final[frozenset[Feature]] = frozenset({"memory_pull", "tool_search"})
#: The ``RunStart.metadata`` key a run's own ``without=`` is kept under (as the parts).
WITHOUT: Final = "without"


def features(names: Iterable[str]) -> frozenset[Feature]:
    """``without=`` checked, each name as the parts it turns off; a name that is not one of
    :data:`NAMES` is refused (``ConfigurationError``, naming them)."""
    named = list(names)
    unknown = [name for name in named if name not in COVERS]
    if unknown:
        raise ConfigurationError(
            f"without= names no feature {', '.join(map(repr, unknown))}: the features are "
            f"{', '.join(NAMES)}"
        )
    return frozenset(f for name in named for f in COVERS[name])


def run_without(record: RunRecord) -> frozenset[Feature]:
    """The parts a run turned off itself (its ``without=``, kept with its record)."""
    known: tuple[str, ...] = get_args(Feature)
    kept = [f for f in record.metadata.get(WITHOUT, ()) if f in known]
    return frozenset(cast("list[Feature]", kept))
