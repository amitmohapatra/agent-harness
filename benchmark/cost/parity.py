"""Whether two runs of the ``complex`` suite give the same answers: the same cases run two ways
(``--variant main`` and ``--variant sandbox``; two windows; two adapters), compared turn by turn.

    python -m benchmark.cost.parity build/cx-main.jsonl build/cx-sandbox.jsonl

Every scored turn both runs hold is compared on the suite's own strict check — each figure
against the truth, within the case's tolerance — so two right answers state the same figures,
and a turn right one way and wrong the other is a difference, listed with what the wrong one
lacks. A harness check that failed on one side only (a file not kept, a request over the
window) is listed too. The exit code is 1 when anything differs.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def turns(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """A run's turns by case and number — each case's latest row, cases that did not apply
    left out."""
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["case"]] = row
    return {
        (case, t["turn"]): t
        for case, row in rows.items()
        if not row.get("not_applicable")
        for t in row["turns"]
    }


def differences(one: Path, other: Path) -> tuple[int, list[str]]:
    """How many turns the two runs share, and each difference between them, as a line."""
    a, b = turns(one), turns(other)
    shared = sorted(set(a) & set(b))
    out: list[str] = []
    for key in shared:
        left, right = a[key], b[key]
        marks = [(t.get("score") or {}).get("correct") for t in (left, right)]
        if marks[0] != marks[1]:
            wrong = left if marks[1] else right
            lacks = (wrong.get("score") or {}).get("missing") or wrong.get("error") or ""
            where = one.name if marks[1] else other.name
            out.append(f"{key[0]} turn {key[1]}: right in one only — {where} lacks {lacks}")
        failed = [{k["check"] for k in t.get("checks", ()) if not k["ok"]} for t in (left, right)]
        if failed[0] != failed[1]:
            out.append(
                f"{key[0]} turn {key[1]}: checks failed — {one.name}: {sorted(failed[0])}; "
                f"{other.name}: {sorted(failed[1])}"
            )
    return len(shared), out


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        sys.stdout.write(f"{__doc__}\n")
        return 2
    one, other = (Path(a) for a in args)
    shared, found = differences(one, other)
    lines = [f"{shared} turns in both; {len(found)} differences", *(f"- {x}" for x in found)]
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
