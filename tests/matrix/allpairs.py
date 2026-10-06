"""All-pairs (pairwise) rows over factors with constraints, written here (no dependency).

A row assigns every factor one of its values. The rows returned cover every pair of values of
two different factors that some valid row can hold: ``(a=x, b=y)`` appears in at least one row
for every such pair. A pair no valid row can hold (``grounding`` on with ``memory`` off, when
grounding needs memory) is not required. The result is deterministic: the same factors and
constraint give the same rows, in the same order, so test ids are stable.

The rows are built greedily (the classic AETG-style construction): each new row starts from the
first pair not yet covered and gives every other factor the value that covers the most pairs
still missing, among the values that keep the row completable.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Final

Value = Hashable
Row = dict[str, Value]
#: Whether a partial row (some factors assigned) breaks no rule among the factors it assigns.
Valid = Callable[[Mapping[str, Value]], bool]
Pair = tuple[tuple[str, Value], tuple[str, Value]]

#: A row search gives up past this many partial rows (the factors here are a handful).
MAX_SEARCH: Final = 1 << 16


def allpairs(factors: Mapping[str, Sequence[Value]], valid: Valid = lambda _: True) -> list[Row]:
    """Rows covering every coverable pair of ``factors`` values, each row ``valid``."""
    names = list(factors)
    if len(names) < 2:
        return [{names[0]: v} for v in factors[names[0]] if valid({names[0]: v})] if names else []
    missing = [pair for pair in _pairs(factors) if _completable(factors, dict(pair), valid)]
    uncovered = set(missing)
    rows: list[Row] = []
    while uncovered:
        start = next(p for p in missing if p in uncovered)
        row: Row = dict(start)
        for name in names:
            if name in row:
                continue
            best: tuple[int, int] | None = None
            chosen: Value = None
            for index, value in enumerate(factors[name]):
                trial = {**row, name: value}
                if not _completable(factors, trial, valid):
                    continue
                gain = sum(
                    1 for other, seen in row.items() if _pair(name, value, other, seen) in uncovered
                )
                if best is None or (gain, -index) > best:
                    best, chosen = (gain, -index), value
            assert best is not None, f"no value of {name} completes {row}"
            row[name] = chosen
        rows.append(row)
        uncovered -= _covered(row)
    return rows


def covered(rows: Sequence[Row]) -> set[Pair]:
    """Every pair the rows hold."""
    found: set[Pair] = set()
    for row in rows:
        found |= _covered(row)
    return found


def required(factors: Mapping[str, Sequence[Value]], valid: Valid = lambda _: True) -> set[Pair]:
    """Every pair some valid row can hold: what :func:`allpairs` must cover."""
    return {p for p in _pairs(factors) if _completable(factors, dict(p), valid)}


def _pairs(factors: Mapping[str, Sequence[Value]]) -> list[Pair]:
    names = list(factors)
    return [
        _pair(a, x, b, y)
        for a, b in itertools.combinations(names, 2)
        for x in factors[a]
        for y in factors[b]
    ]


def _pair(a: str, x: Value, b: str, y: Value) -> Pair:
    """The pair in a canonical order (by factor name)."""
    return ((a, x), (b, y)) if a < b else ((b, y), (a, x))


def _covered(row: Mapping[str, Value]) -> set[Pair]:
    return {_pair(a, row[a], b, row[b]) for a, b in itertools.combinations(sorted(row), 2)}


def _completable(
    factors: Mapping[str, Sequence[Value]], partial: Mapping[str, Value], valid: Valid
) -> bool:
    """Whether the factors ``partial`` leaves open can be given values making a valid row."""
    if not valid(partial):
        return False
    open_ = [n for n in factors if n not in partial]
    searched = 0

    def search(at: int, row: dict[str, Value]) -> bool:
        nonlocal searched
        searched += 1
        if searched > MAX_SEARCH:
            raise RuntimeError("the constraint is too loose to search: fewer factors, please")
        if at == len(open_):
            return True
        name = open_[at]
        for value in factors[name]:
            row[name] = value
            if valid(row) and search(at + 1, row):
                return True
            del row[name]
        return False

    return search(0, dict(partial))
