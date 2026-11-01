"""``query_result`` (``tools.query``): read-only SQL over every row of a kept result, in
Python's own SQLite — JSON, CSV or plain lines as a table, exact answers, nothing but reading."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import statistics
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from trellis.harness import runtime as runtime_module
from trellis.harness.journal import Journal
from trellis.harness.tools import query, results


class Run:
    """A run in progress, as ``tools.query`` reads it."""

    def __init__(self, journal: Journal) -> None:
        self.replay = SimpleNamespace(journal=journal)
        self.toolbox: dict[str, object] = {}
        self.native: dict[str, object] = {}


@pytest.fixture
def journal() -> Iterator[Journal]:
    journal = Journal()
    run = Run(journal)
    token = runtime_module._current.set(run)  # type: ignore[arg-type]
    yield journal
    runtime_module._current.reset(token)


def ask(text: str, sql: str) -> str:
    path = results.keep(text)
    assert path is not None
    return asyncio.run(query.query({"file_path": path, "sql": sql}))


ROWS = [
    {"sku": f"S{n}", "stock": n % 7, "price": n / 4, "tags": ["a"], "ok": True} for n in range(30)
]


@pytest.mark.parametrize(
    "text",
    [json.dumps(ROWS), json.dumps({"total": 30, "items": ROWS})],
    ids=["array", "object-holding-one"],
)
def test_json_rows_are_a_table_with_typed_columns(journal: Journal, text: str) -> None:
    assert ask(text, "SELECT count(*) FROM result WHERE stock < 2") == "count(*)\n10"
    assert ask(text, "SELECT sum(price) FROM result;") == "sum(price)\n108.75"
    assert (
        ask(text, "SELECT json_extract(tags, '$[0]') t, ok FROM result LIMIT 1") == "t,ok\na,true"
    )
    assert query.columns_of(text) == ["sku", "stock", "price", "tags", "ok"]


def test_csv_with_any_delimiter_ragged_rows_and_odd_names(journal: Journal) -> None:
    text = "SKU id\tunits\tunits\t\nA\t3\t1.5\nB\t\t2\t9\n\nC\t1"
    assert query.columns_of(text) == ["SKU_id", "units", "units_2", "c4"]
    said = ask(text, "SELECT SKU_id, units, units_2 FROM result ORDER BY SKU_id")
    assert said == "SKU_id,units,units_2\nA,3,1.5\nB,,2.0\nC,1,"


def test_plain_text_is_one_row_per_line(journal: Journal) -> None:
    text = "first line\nsecond line\n" + "x" * 10
    assert query.columns_of(text) == ["line"]
    assert ask(text, "SELECT count(*) FROM result WHERE line LIKE '%line'") == "count(*)\n2"
    assert query.columns_of('"just a string"') == ["line"]  # JSON, but no rows
    assert query.columns_of("") == ["line"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM result",
        "PRAGMA table_info(result)",
        "ATTACH ':memory:' AS other",
        "CREATE TABLE t(x)",
        "SELECT load_extension('x')",
        "SELECT * FROM nowhere",
    ],
)
def test_only_reading_runs(journal: Journal, sql: str) -> None:
    said = ask("a,b\n1,2", sql)
    assert said.startswith("Error: the query failed:") and "Only a SELECT" in said
    assert ask("a,b\n1,2", "SELECT a + b FROM result") == "a + b\n3"  # the table is whole


def test_a_query_is_bounded_in_time_rows_and_length(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "n,sq\n" + "\n".join(f"{n},{n * n}" for n in range(1_000))
    many = ask(text, "SELECT n FROM result")
    assert many.count("\n") == query.ROWS + 1 and "only the first 200 rows" in many
    assert ask(text, "SELECT 1" + " + 1" * query.SQL_CHARS).startswith("Error: the query is longer")
    monkeypatch.setattr(query, "SECONDS", 0.0)
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"
    assert "took too long" in ask(text, endless)


def test_a_wrong_call_is_said(journal: Journal) -> None:
    assert "sql" in asyncio.run(query.query({"file_path": "/large_tool_results/x"}))
    missing = asyncio.run(query.query({"file_path": "/x", "sql": "SELECT 1"}))
    assert missing == "Error: File '/x' not found"
    given = asyncio.run(
        query.query({"file_path": "x", "sql": "SELECT count(*) FROM result"}, "a\nb")
    )
    assert given == "count(*)\n2"  # a text given (a file Deep Agents saved in the state)


def test_outside_a_run_nothing_is_read() -> None:
    said = asyncio.run(query.query({"file_path": "/x", "sql": "SELECT 1"}, "a,b\n1,2"))
    assert said == "Error: File '/x' not found"


@pytest.mark.parametrize(
    ("cells", "kind"),
    [
        (["1", "-2", " 3 "], "INTEGER"),
        (["1", "2.5"], "REAL"),
        (["007", "8"], "TEXT"),  # a code with a leading zero stays as written
        (["9" * 40], "TEXT"),  # a run of digits too long for a number
        (["12345678901234567890"], "TEXT"),  # a whole number past 64 bits: never rounded
        (["12345678901234567890", "1.5"], "TEXT"),
        ([2**64], "TEXT"),
        ([float("inf")], "TEXT"),
        (["1e999"], "TEXT"),
        (["", None], "TEXT"),
        (["0", "0.5"], "REAL"),
    ],
)
def test_a_column_is_a_number_only_when_sqlite_keeps_it_exactly(cells: list, kind: str) -> None:
    assert query._kind(cells) == kind


@pytest.mark.parametrize(
    "text",
    [
        "id,v\n12345678901234567890,1\n12345678901234567891,2\n7,3",
        json.dumps(
            [
                {"id": 12345678901234567890, "v": 1},
                {"id": 12345678901234567891, "v": 2},
                {"id": 7, "v": 3},
            ]
        ),
    ],
    ids=["csv", "json"],
)
def test_an_id_past_64_bits_keeps_every_digit(journal: Journal, text: str) -> None:
    """SQLite's integers stop at 64 bits: a longer id is its exact text, never a rounded REAL
    (two ids made one) nor a failed load."""
    said = ask(text, "SELECT id, typeof(id) FROM result WHERE v < 3 ORDER BY v")
    assert said == "id,typeof(id)\n12345678901234567890,text\n12345678901234567891,text"
    assert ask(text, "SELECT count(DISTINCT id) FROM result") == "count(DISTINCT id)\n3"


def aggregate(name: str, *rows: tuple[Any, ...]) -> Any:
    """``name`` (of ``query.AGGREGATES``) over ``rows`` on a database of its own."""
    db = sqlite3.connect(":memory:")
    try:
        for fn, arguments, made in query.AGGREGATES:
            db.create_aggregate(fn, arguments, made)
        width = len(rows[0]) if rows else 1
        db.execute(f"CREATE TABLE t ({', '.join(f'c{i}' for i in range(width))})")
        db.executemany(f"INSERT INTO t VALUES ({', '.join('?' * width)})", rows)
        cells = ", ".join(f"c{i}" for i in range(width))
        return db.execute(f"SELECT {name}({cells}) FROM t").fetchone()[0]
    finally:
        db.close()


def test_the_statistics_sqlite_lacks_are_pythons_own() -> None:
    xs, ys = [1.0, 2.0, 4.0, 7.0], [2.0, 4.1, 7.9, 14.2]
    pairs = [(y, x) for x, y in zip(xs, ys, strict=True)]
    assert aggregate("stddev", *[(x,) for x in xs]) == statistics.stdev(xs)
    assert aggregate("var_pop", *[(x,) for x in xs], (None,), ("text",)) == statistics.pvariance(xs)
    assert aggregate("covar_pop", *pairs, (None, 1)) == statistics.covariance(xs, ys) * 3 / 4
    assert aggregate("regr_r2", *pairs) == statistics.correlation(xs, ys) ** 2
    assert aggregate("percentile_cont", *[(x, 0.5) for x in xs]) == 3.0
    assert aggregate("stddev", (1.0,)) is None  # too few
    assert aggregate("corr", (1.0, 1.0), (2.0, 1.0)) is None  # no spread
    assert aggregate("corr", (1.0, 1.0)) is None
    assert aggregate("percentile_cont", (1.0, 2)) is None  # p outside 0..1
    assert aggregate("percentile_cont", (None, 0.5)) is None


def test_two_results_of_one_tool_are_two_tables(journal: Journal) -> None:
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    for text in ("a\n1", "a\n2", "b\n3"):
        results.keep(text, tool="rows")
    assert list(query.tables(run).values()) == ["rows", "rows_2", "rows_3"]
