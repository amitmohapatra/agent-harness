"""What a wrapped agent does around every run, and the one switch that turns parts of it off:
``without=`` — on ``h.wrap`` for every run of the agent, on ``agent.run``/``stream``/``start``
for one run (added to the agent's). Everything configured is on unless named here; nothing
else is a setting.

A run's own ``without=`` is kept with its record (``RunStart.metadata``), so every attempt —
a resume, a worker's after a crash — and every sub-agent run it starts works without the same.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, cast, get_args

from trellis.contracts import ConfigurationError, RunRecord

Feature = Literal[
    "memory",
    "memory_push",
    "memory_pull",
    "records",
    "judges",
    "grounding",
    "hints",
    "code_mode",
    "skills",
    "mcp",
]
#: The features a name turns off together: ``memory`` is its push, its pull and the records;
#: ``mcp`` the MCP tools and Code Mode, whose scripts reach them.
COVERS: Final[dict[str, tuple[Feature, ...]]] = {
    "memory": ("memory_push", "memory_pull", "records"),
    "mcp": ("mcp", "code_mode"),
}
#: The ``RunStart.metadata`` key a run's own ``without=`` is kept under.
WITHOUT: Final = "without"


def features(names: Iterable[str]) -> frozenset[Feature]:
    """``without=`` checked, each name as the features it turns off; a name that is not a
    :data:`Feature` is refused (``ConfigurationError``, naming the features)."""
    named = list(names)
    known: tuple[str, ...] = get_args(Feature)
    unknown = [name for name in named if name not in known]
    if unknown:
        raise ConfigurationError(
            f"without= names no feature {', '.join(map(repr, unknown))}: the features are "
            f"{', '.join(known)}"
        )
    chosen = cast("list[Feature]", named)
    return frozenset(f for name in chosen for f in COVERS.get(name, (name,)))


def run_without(record: RunRecord) -> frozenset[Feature]:
    """The features a run turned off itself (its ``without=``, kept with its record)."""
    return features(record.metadata.get(WITHOUT, ()))
