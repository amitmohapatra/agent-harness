"""The generated matrix: one test per cell (``tests/matrix/generate.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.matrix.generate import CELLS, Cell
from tests.matrix.model import NA, Bug, Gap
from tests.matrix.world import MemoryContract, World


def _param(cell: Cell) -> object:
    marks: list[pytest.MarkDecorator] = [pytest.mark.matrix]
    if isinstance(cell.note, NA):
        marks.append(pytest.mark.skip(reason=f"n.a.: {cell.note.reason}"))
    elif isinstance(cell.note, Gap):
        marks.append(pytest.mark.xfail(strict=True, reason=f"{cell.note.id}: {cell.note.why}"))
    elif isinstance(cell.note, Bug):
        reason = f"{cell.note.id}: {cell.note.why}"
        marks.append(
            pytest.mark.xfail(strict=cell.note.strict, reason=reason, raises=cell.note.raises)
        )
    return pytest.param(cell, id=cell.id, marks=marks)


@pytest.mark.parametrize("cell", [_param(c) for c in CELLS])
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
    if cell.selection.pending is not None:
        await cell.selection.pending.probe(world)
    elif cell.way == "way2":
        scenario = cell.feature.way2
        assert callable(scenario)
        await scenario(world)
    else:
        await cell.feature.scenario(world)
        world.verify()
