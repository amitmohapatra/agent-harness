"""The matrix's vocabulary: the dimensions, what a cell may be instead of a test that passes
(not applicable, a gap, a bug), and a feature's row.

A **cell** is one feature, on one adapter, one way, one mode, under one selection: its id is
``<feature>-<adapter>-<way>-<mode>-<selection>`` (no part holds a ``-``), which is the pytest
id, the junit name and the report's key.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from tests.matrix.world import World

#: The targets, as the adapters name themselves (``claude`` is the Claude Agent SDK).
ADAPTERS: Final = ("react", "function", "langgraph", "deepagents", "openai_agents", "claude")
#: Way 1 (``h.wrap`` on a deployment), Way 2 (the blocks, no Harness), with blocks: the team's
#: own blocks (``Harness(runs=, memory=, gateway=, governance=).wrap(ReAct(...))`` and its own
#: worker around ``agent.execute``; ReAct only, so far).
WAYS: Final = ("way1", "way2", "with_blocks")
#: How a run is driven: ``run``; ``stream``; ``worker`` (``start`` + a worker); ``elsewhere``
#: (queued, paused, resumed and continued by another process's worker on the same run store);
#: ``schedule`` (a schedule fires it, a worker runs it); ``agui`` (``serve_chat``); ``a2a``
#: (``serve_a2a``).
MODES: Final = ("run", "stream", "worker", "elsewhere", "schedule", "agui", "a2a")

Way = Literal["way1", "way2", "with_blocks"]


@dataclass(frozen=True, slots=True)
class NA:
    """Not applicable here, and why (the report lists the reason)."""

    reason: str


@dataclass(frozen=True, slots=True)
class Gap:
    """Not there yet: an audit gap (``G..``) or a plan item. The cell runs and must fail
    (``xfail(strict=True)``): the day the gap closes it passes, fails the suite as XPASS, and
    the marker comes off."""

    id: str
    why: str


@dataclass(frozen=True, slots=True)
class Bug:
    """A combination that should work and does not, found by this matrix (``BUG-..``: the
    ``KNOWN`` table of ``generate.py`` says where and how to reproduce it): runs and must fail
    until it is fixed."""

    id: str
    why: str
    #: the failure it shows as (the cell fails on any other: a regression is not hidden)
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None
    #: ``False`` for a race: the cell sometimes passes (a pass is then no news)
    strict: bool = True


Note = NA | Gap | Bug


@dataclass(frozen=True, slots=True)
class Switch:
    """Something a user turns on or off: how (``how``), and what it needs on (``requires``)."""

    id: str
    title: str
    how: str
    requires: frozenset[str] = frozenset()
    #: the ``without=`` name that turns it off for an agent whose deployment has what it
    #: requires (``None``: it is off when its service or argument is not given)
    without: str | None = None


@dataclass(frozen=True, slots=True)
class Selection:
    """What a user turned on: the switches in ``on``."""

    id: str
    on: frozenset[str]
    kind: Literal["all", "none", "only", "without", "pairs"] = "all"


Scenario = Callable[["World"], Awaitable[None]]


@dataclass(frozen=True)
class Feature:
    """One row of the matrix.

    ``needs``: the switches that must be on for the feature to be on (none: always on). The
    scenario reads ``world.on`` and checks the behaviour when the feature is on, and that it
    left no trace when it is off. ``adapters``/``ways``/``modes`` name only the exceptions
    (``NA``, ``Gap``, ``Bug``); ``cells`` names an exception for one (adapter, way, mode),
    ``None`` standing for any. ``way2`` is the Way 2 scenario (the block used without a
    Harness), or why there is none."""

    id: str
    title: str
    audit: str
    how: str
    scenario: Scenario
    needs: frozenset[str] = frozenset()
    adapters: Mapping[str, Note] = field(default_factory=dict)
    modes: Mapping[str, Note] = field(default_factory=dict)
    ways: Mapping[str, Note] = field(default_factory=dict)
    way2: Scenario | Note | None = None
    #: the gap a Way 2 scenario (a probe of the block the plan proposes) waits on
    way2_gap: Gap | None = None
    way2_modes: tuple[str, ...] = ("run",)
    cells: Mapping[tuple[str | None, str | None, str | None], Note] = field(default_factory=dict)


def cell_id(feature: str, adapter: str, way: str, mode: str, selection: str) -> str:
    parts = (feature, adapter, way, mode, selection)
    assert not any("-" in p for p in parts), parts
    return "-".join(parts)


def parse_cell(test_id: str) -> tuple[str, str, str, str, str] | None:
    """A cell's dimensions from its id (or a pytest node id ending ``[<id>]``)."""
    if "[" in test_id:
        test_id = test_id.rsplit("[", 1)[1].rstrip("]")
    parts = test_id.split("-")
    if len(parts) != 5:
        return None
    feature, adapter, way, mode, selection = parts
    return feature, adapter, way, mode, selection
