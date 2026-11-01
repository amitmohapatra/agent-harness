"""The generated matrix: one test per cell that applies (``tests/matrix/generate.py``)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Final

import pytest

from tests.matrix.generate import RUNNABLE, Cell
from tests.matrix.model import Bug, Gap
from tests.matrix.world import MemoryContract, World
from trellis.harness import pipeline


def _param(cell: Cell) -> object:
    marks: list[pytest.MarkDecorator] = [pytest.mark.matrix]
    if isinstance(cell.note, Gap):
        marks.append(pytest.mark.xfail(strict=True, reason=f"{cell.note.id}: {cell.note.why}"))
    elif isinstance(cell.note, Bug):
        reason = f"{cell.note.id}: {cell.note.why}"
        marks.append(
            pytest.mark.xfail(strict=cell.note.strict, reason=reason, raises=cell.note.raises)
        )
    return pytest.param(cell, id=cell.id, marks=marks)


@pytest.mark.parametrize("cell", [_param(c) for c in RUNNABLE])
async def test_cell(cell: Cell, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        feature=cell.feature,
        adapter=cell.adapter,
        way=cell.way,
        mode=cell.mode,
        selection=cell.selection,
        tmp=tmp_path,
        monkeypatch=monkeypatch,
    )
    try:
        await _run(cell, world)
    finally:
        await world.aclose()
        violations = world.contract_violations()
    if violations:
        raise MemoryContract("\n".join(violations))


async def _run(cell: Cell, world: World) -> None:
    if cell.way == "way2":
        scenario = cell.feature.way2
        assert callable(scenario)
        await scenario(world)
    else:
        await cell.feature.scenario(world)
        world.verify()


#: How long each attempt of a run waits before it starts working, in :func:`test_slow_runner`:
#: longer than a run took to reach its tool on a loaded CI runner, past the time limits of
#: seconds the time-limit cells had then.
SLOW_SECONDS: Final = 3.0
#: The time-limit cells of every adapter, run in this process.
TIME_LIMITS: Final = [
    pytest.param(c, id=c.id, marks=pytest.mark.matrix)
    for c in RUNNABLE
    if c.feature.id in ("F09", "F09r")
    and (c.way, c.mode) == ("way1", "run")
    and c.id.endswith("-none")
]


@pytest.mark.parametrize("cell", TIME_LIMITS)
async def test_slow_runner(cell: Cell, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A time-limit cell's verdict does not depend on the runner's speed: each attempt is slow to
    reach its tool here (``SLOW_SECONDS`` within the run's time first), and the run still runs
    out of time in the tool."""
    worked = pipeline._worked

    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(SLOW_SECONDS)
        return await worked(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_worked", slow)
    await test_cell(cell, tmp_path, monkeypatch)
