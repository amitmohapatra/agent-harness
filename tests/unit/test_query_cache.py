"""The typed tables ``query_result`` keeps as Parquet (``tools.query.Typed``): a text loaded
again — a later query, turn or past result — reads its tables from there, and every answer is
the one its text gives parsed afresh: the same values, types, order and parts."""

from __future__ import annotations

import json

import pytest

from tests.unit.test_query import ask, journal  # noqa: F401 - the fixture
from tests.unit.test_query_shapes import ORDERS
from trellis.harness.tools import query

pytestmark = pytest.mark.usefixtures("journal")

TEXTS = {
    "json": json.dumps(
        [
            {"id": 12345678901234567890, "p": "12.50", "c": "007", "x": 0.1 + 0.2},
            {"id": 7, "p": None, "c": "8", "x": 1e-3},
        ]
    ),
    "held": json.dumps({"orders": ORDERS, "refunds": [{"id": 1, "amt": "2.25"}]}),
    "json-lines": "\n".join(json.dumps(r) for r in ORDERS),
    "csv": '﻿नाम;amount;when\n"a; b";1.50;2026-03-08T01:30:00-05:00\nc;N/A;\nd;-2.25;2026-03-08T12:00:00Z\n',
    "markdown": "| sku | units |\n|---|---|\n| A | 3 |\n| B | |\n",
    "lines": "first\nsecond, with a comma\n",
}
SQL = (
    "SELECT * FROM result",
    "SELECT * FROM (DESCRIBE result)",
)


@pytest.mark.parametrize("kind", list(TEXTS))
def test_a_cached_text_answers_as_it_does_parsed_afresh(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = TEXTS[kind]
    fresh = []
    for sql in SQL:  # each parsed afresh
        monkeypatch.setattr(query, "TYPED", query.Typed())
        fresh.append(ask(text, sql))
    assert query.TYPED.get(query.Typed.key(text)) is not None  # kept on the first load
    read: list[object] = []
    monkeypatch.setattr(query.DuckDB, "_read", lambda *a: read.append(a) or iter(()))
    assert [ask(text, sql) for sql in SQL] == fresh  # no text read again: the Parquet
    assert read == []
    if kind == "held":
        assert ask(text, "SELECT count(*) n FROM result_refunds") == "n\n1"


def test_the_least_recently_used_text_is_forgotten_past_the_bound() -> None:
    typed = query.Typed(most=10)
    typed.put(b"a", [(None, b"123456")])
    typed.put(b"b", [(None, b"1234")])
    assert typed.get(b"a") is not None  # now the most recently used
    typed.put(b"b", [(None, b"12"), ("x", b"34")])  # replaced, not added: now the newest
    assert typed.size == 10
    typed.put(b"c", [(None, b"1")])  # past the bound: "a" goes, used before "b" was replaced
    assert (typed.get(b"a"), typed.size) == (None, 5)
    assert typed.get(b"b") is not None and typed.get(b"c") is not None
