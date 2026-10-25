"""What a wrapped agent does around every run, and the one switch that turns parts of it off:
``without=`` — on ``h.wrap`` for every run of the agent, on ``agent.run``/``stream``/``start``
for one run (added to the agent's). Everything configured is on unless named here; nothing
else is a setting.

A run's own ``without=`` is kept with its record (``RunStart.metadata``), so every attempt —
a resume, a worker's after a crash — and every sub-agent run it starts works without the same.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable
from typing import Final, Literal, cast, get_args

from trellis.contracts import ConfigurationError, RunRecord

log = logging.getLogger("trellis.harness")

Feature = Literal[
    # the names a team turns off
    "memory",
    "tool_search",
    "mcp",
    "code_mode",
    "skills",
    "judges",
    "grounding",
    "planning",
    "files",
    "react_auto",
    # the parts they cover (``memory_push``, ``memory_pull``, ``records`` and ``hints`` are also
    # still accepted as names, deprecated: :data:`DEPRECATED`)
    "memory_push",
    "memory_pull",
    "records",
    "hints",
]
#: The features a name turns off together: ``memory`` is its push, its pull, the records and
#: the tool narrowing; ``tool_search`` the tool hints and the memory service's ``tool_search``
#: tool (the one owner of narrowing); ``mcp`` the MCP tools and Code Mode, whose scripts reach
#: them; ``react_auto`` what ``ReAct`` turns on for a long task.
COVERS: Final[dict[str, tuple[Feature, ...]]] = {
    "memory": ("memory_push", "memory_pull", "records", "hints", "tool_search"),
    "tool_search": ("hints", "tool_search"),
    "mcp": ("mcp", "code_mode"),
    "react_auto": ("planning", "files"),
    # a deprecated name: the memory tools, ``tool_search`` among them
    "memory_pull": ("memory_pull", "tool_search"),
}
#: The features of the memory service's agent tools (``Harness.pull_tools``): its pull
#: tools and ``tool_search``.
MEMORY_TOOLS: Final[frozenset[Feature]] = frozenset({"memory_pull", "tool_search"})
#: The names a release retired, still accepted (with a warning naming what replaced them).
DEPRECATED: Final[dict[str, str]] = {
    "memory_push": "memory (or tool_search for the narrowing alone)",
    "memory_pull": "memory",
    "records": "memory",
    "hints": "tool_search",
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
            f"{', '.join(k for k in known if k not in DEPRECATED)}"
        )
    for name in named:
        if name in DEPRECATED:
            _deprecated(name)
    chosen = cast("list[Feature]", named)
    return frozenset(f for name in chosen for f in COVERS.get(name, (name,)))


@functools.cache
def _deprecated(name: str) -> None:
    log.warning(
        "without={%r} is deprecated and goes in the next release: use %s", name, DEPRECATED[name]
    )


def run_without(record: RunRecord) -> frozenset[Feature]:
    """The features a run turned off itself (its ``without=``, kept with its record)."""
    return features(record.metadata.get(WITHOUT, ()))
