"""The matrix's own machinery (run by ``make test``, not marked ``matrix``): the all-pairs
rows cover every coverable pair, the cells are unique and parse back, the report reads junit."""

from __future__ import annotations

import itertools
from pathlib import Path

from tests.matrix.allpairs import allpairs, covered, required
from tests.matrix.dimensions import SELECTIONS, SWITCHES, valid
from tests.matrix.generate import CELLS, RUNNABLE
from tests.matrix.model import NA, parse_cell
from tests.matrix.report import read, render


def test_all_pairs_cover_every_pair_a_valid_row_can_hold() -> None:
    factors = {s.id: (True, False) for s in SWITCHES}
    rows = allpairs(factors, valid)
    assert required(factors, valid) <= covered(rows)
    assert all(valid(row) for row in rows)
    assert len(rows) < 2 ** len(factors) // 8  # far fewer than every combination

    mixed = {"a": (1, 2, 3), "b": ("x", "y"), "c": (True, False), "d": (0, 1, 2)}
    no_3x = lambda row: not (row.get("a") == 3 and row.get("b") == "x")  # noqa: E731
    rows = allpairs(mixed, no_3x)
    assert required(mixed, no_3x) <= covered(rows) and all(no_3x(r) for r in rows)
    assert (("a", 3), ("b", "x")) not in covered(rows)
    assert len(rows) < len(list(itertools.product(*mixed.values())))
    assert allpairs({"only": (1, 2)}) == [{"only": 1}, {"only": 2}] and allpairs({}) == []


def test_every_cell_is_unique_and_names_its_dimensions() -> None:
    ids = [c.id for c in CELLS]
    assert len(ids) == len(set(ids))
    for cell in CELLS:
        assert parse_cell(f"test_cell[{cell.id}]") == (
            cell.feature.id,
            cell.adapter,
            cell.way,
            cell.mode,
            cell.selection.id,
        )
    assert {s.id for s in SELECTIONS} >= {"all", "none"}
    assert parse_cell("not a cell") is None


def test_the_report_says_what_ran(tmp_path: Path) -> None:
    ran, skipped, waiting, failed = (c.id for c in RUNNABLE[:4])
    junit = tmp_path / "matrix-1.xml"
    junit.write_text(
        '<testsuites><testsuite time="3.5">'
        f'<testcase name="test_cell[{ran}]" time="1"/>'
        f'<testcase name="test_cell[{skipped}]"><skipped message="no model"/></testcase>'
        f'<testcase name="test_cell[{waiting}]"><skipped type="pytest.xfail" message="G8: cut"/>'
        "</testcase>"
        f'<testcase name="test_cell[{failed}]"><failure message="boom\nmore"/></testcase>'
        '<testcase name="test_other"/>'
        "</testsuite></testsuites>"
    )
    found = read([junit])
    assert [found[c].status for c in (ran, skipped, waiting, failed)] == [
        "pass",
        "FAIL",  # every cell that applies must run: a skip at run time fails it
        "xfail",
        "FAIL",
    ]
    text = render(found, 3.5)
    assert (
        f"- `{failed}`: boom" in text and f"- `{skipped}`: skipped while it ran: no model" in text
    )
    assert "- G8: cut: 1" in text
    assert f"{len(RUNNABLE) - 4} not run" in text  # a cell that does not apply is never "not run"
    na = [c.note for c in CELLS if isinstance(c.note, NA)]
    assert na and len(RUNNABLE) + len(na) == len(CELLS)
    assert f"- {na[0].reason}: " in text  # its reason, from the table
