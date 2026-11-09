"""``query_result`` over complex data (``tools.query``): nested and line-delimited JSON, CSV
variants, markdown tables, values (time zones, null markers, money as text), analysis SQL
(pivots, gaps, dedupe) — each answered exactly or said plainly — and data that is no table
(HTML, PDF, an image) refused with where to compute instead, never answered from its lines."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.unit.test_query import ask, journal  # noqa: F401 - the fixture
from trellis.harness import runtime as runtime_module
from trellis.harness.tools import query, results

pytestmark = pytest.mark.usefixtures("journal")


# --------------------------------------------------------------------------- nested JSON
ORDERS = [
    {"id": 1, "items": [{"sku": "a", "q": 2}, {"sku": "b", "q": 3}], "tags": ["x", "y"]},
    {"id": 2, "items": [{"sku": "a", "q": 5}], "note": "late"},
    {"id": "3A", "items": []},
]


def test_arrays_inside_rows_missing_keys_and_mixed_types() -> None:
    text = json.dumps(ORDERS)
    items = "SELECT sum(CAST(i->>'q' AS INT)) q FROM result, unnest(CAST(items AS JSON[])) t(i)"
    assert ask(text, items) == "q\n10"
    assert ask(text, "SELECT sum(len(CAST(tags AS JSON[]))) n FROM result") == "n\n2"
    assert ask(text, "SELECT count(*) n FROM result WHERE note IS NULL") == "n\n2"
    assert ask(text, "SELECT string_agg(id, '|') ids, any_value(typeof(id)) k FROM result") == (
        "ids,k\n1|2|3A,VARCHAR"  # a column of mixed values: each as written, as text
    )


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"data": {"page": 1, "rows": ORDERS}}),
        json.dumps({"meta": {"n": 3}, "result": {"data": {"items": ORDERS}}}),
        "\n".join(json.dumps(r) for r in ORDERS) + "\n",  # JSON Lines
    ],
    ids=["nested-holder", "deeper-holder", "json-lines"],
)
def test_rows_held_deeper_or_one_per_line_are_a_table(text: str) -> None:
    assert query.columns_of(text) == ["id", "items", "tags", "note"]
    assert len(query.rows_of(text)[1]) == 3
    assert ask(text, "SELECT count(*) n, count(note) late FROM result") == "n,late\n3,1"


def test_each_array_an_object_holds_is_a_table() -> None:
    """``{"orders": [...], "refunds": [...]}``: ``result`` is the first array, and each other
    array is a table of its own, never left out."""
    text = json.dumps(
        {"orders": [{"id": 1, "amt": "10.50"}], "refunds": [{"id": 1, "amt": "2.25"}]}
    )
    sql = "SELECT (SELECT sum(amt) FROM result) o, (SELECT sum(amt) FROM result_refunds) r"
    assert ask(text, sql) == "o,r\n10.5,2.25"
    big = text[:-1] + ', "pad": "' + "x" * results.MAX_CHARS + '"}'
    stub = results.readable(big)
    path = stub.split("at this path: ")[1].split("\n")[0]
    name = query.tables(runtime_module._current.get())[path]
    assert (
        f'Its other rows are tables too: "refunds" is the table {name}_refunds (columns: id, amt).'
        in stub
    ), stub
    assert ask(big, f"SELECT count(*) n FROM {name}_refunds") == "n\n1"


def test_a_part_whose_name_is_taken_takes_another() -> None:
    """A part's table (``<table>_<key>``) or view (``result_<key>``) is never made over a
    table of the same name: the other result keeps it, the part takes the next name free (its
    view still names it)."""
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    results.keep("v\n1\n2", tool="a_b")
    results.keep("v\n7\n8", tool="result_1")
    held = json.dumps({"rows": [{"v": 3}], "b": [{"v": 4}], "1": [{"v": 5}]})
    path = results.keep(held, tool="a")
    sql = "SELECT (SELECT sum(v) FROM a_b) ab, (SELECT sum(v) FROM result_1) r1, (SELECT v FROM result_b) b"
    said = asyncio.run(query.query({"file_path": path, "sql": sql}))
    assert said == "ab,r1,b\n3,15,4", said


def test_json_lines_with_a_broken_line_are_lines() -> None:
    assert query.columns_of('{"a": 1}\n{"a": ') == ["line"]


# --------------------------------------------------------------------------- CSV variants
@pytest.mark.parametrize(
    ("text", "sql", "said"),
    [
        (
            'name,amount\n"Smith, J",10.50\n"Doe, A",2.25\n',
            "SELECT sum(amount) s FROM result",
            "s\n12.75",
        ),
        ("a;b\n1;2\n3;4\n", "SELECT sum(b) s FROM result", "s\n6"),
        ("a|b\n1|2\n3|4\n", "SELECT sum(b) s FROM result", "s\n6"),
        ("a\tb\n1\t2\n3\t4\n", "SELECT sum(b) s FROM result", "s\n6"),
        ("﻿id,v\n1,2\n3,4\n", "SELECT sum(id) s FROM result", "s\n4"),  # a BOM
        (
            "a,b,c\n1,,3\n,2,\n",
            "SELECT count(a) a, count(b) b, sum(c) c FROM result",
            "a,b,c\n1,1,3",
        ),
        ('"x, y";z\n1;2\n3;4\n', "SELECT sum(z) s FROM result", "s\n6"),  # a delimiter quoted
        ("नाम,राशि\nराम,10\nश्याम,20\n", "SELECT sum(राशि) s FROM result", "s\n30"),
        ("名前,金額\n太郎,1\n", "SELECT 名前 FROM result", "名前\n太郎"),
    ],
    ids=[
        "quoted-commas",
        "semicolon",
        "pipe",
        "tab",
        "bom",
        "empty-cells",
        "quoted-head",
        "hindi",
        "japanese",
    ],
)
def test_csv_variants(text: str, sql: str, said: str) -> None:
    assert ask(text, sql) == said


def test_a_markdown_table_is_its_rows() -> None:
    """A table as markdown (``| a | b |`` over ``|---|---|``): its cells trimmed, its rule line
    no row — never counted as one."""
    text = "| sku | units |\n|:----|-----:|\n| A1 | 3 |\n| B2 | 4 |\n"
    assert query.columns_of(text) == ["sku", "units"]
    assert ask(text, "SELECT count(*) n, sum(units) u FROM result") == "n,u\n2,7"


# --------------------------------------------------------------------------- values
def test_times_with_zones_are_compared_in_utc_on_any_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    text = (
        "ts,v\n2026-03-08T01:30:00-05:00,1\n2026-03-08T12:00:00Z,2\n2026-03-08T10:00:00+05:30,3\n"
    )
    order = "SELECT v, CAST(ts AS TIMESTAMPTZ) t FROM result ORDER BY t"
    assert ask(text, order) == (
        "v,t\n3,2026-03-08 04:30:00+00\n1,2026-03-08 06:30:00+00\n2,2026-03-08 12:00:00+00"
    )


def test_null_markers_leave_a_column_a_number() -> None:
    """``NULL``, ``N/A``, ``-``… among numbers are missing values: the column stays a number
    (``count`` skips them); in a text column they are the text they are."""
    text = "a,b,c\n1.50,NULL,NA\n2.25,n/a,US\n,-,null\n3.00,7,None\n"
    said = ask(
        text,
        "SELECT sum(a) a, sum(b) b, count(b) n, any_value(typeof(b)) k, string_agg(c, ' ') c FROM result",
    )
    assert said == "a,b,n,k,c\n6.75,7,1,BIGINT,NA US null None"


@pytest.mark.parametrize(
    "cells", [["$1,234.50", "$2.00"], ["1,234.50", "2.00"], ["12.5%", "3%"], ["(12.50)", "1"]]
)
def test_a_figure_written_with_symbols_stays_text_and_the_failure_says_how(
    cells: list[str],
) -> None:
    """Money, separators and percentages stay text, exactly as written (never summed across
    currencies); summing one says to convert it."""
    text = "k,v\n" + "\n".join(f'{n},"{c}"' for n, c in enumerate(cells))
    said = ask(text, "SELECT sum(v) FROM result")
    assert said.startswith("Error: the query failed:") and "is text" in said, said
    assert "replace(" in said and "TRY_CAST(" in said, said


# --------------------------------------------------------------------------- analysis
def test_pivots_gaps_and_dedupe() -> None:
    sales = "r,p,v\nN,a,1\nN,b,2\nS,a,3\nS,a,3\n"
    pivot = "PIVOT (SELECT DISTINCT * FROM result) ON p IN ('a', 'b') USING sum(v) GROUP BY r ORDER BY r"
    assert ask(sales, pivot) == "r,a,b\nN,1,2\nS,3,"
    days = "d\n2026-01-01\n2026-01-02\n2026-01-05\n"
    gaps = (
        "SELECT CAST(g AS DATE) missing FROM generate_series(DATE '2026-01-01', "
        "DATE '2026-01-05', INTERVAL 1 DAY) t(g) WHERE CAST(g AS DATE) NOT IN "
        "(SELECT CAST(d AS DATE) FROM result)"
    )
    assert ask(days, gaps) == "missing\n2026-01-03\n2026-01-04"
    latest = "SELECT r, v FROM result QUALIFY row_number() OVER (PARTITION BY r ORDER BY v DESC) = 1 ORDER BY r"
    assert ask(sales, latest) == "r,v\nN,2\nS,3"
    within = "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v) m FROM result"
    assert ask(sales, within) == "m\n2.5"


def test_a_pivot_without_its_values_says_how_to_list_them() -> None:
    said = ask("r,p,v\nN,a,1\n", "PIVOT result ON p USING sum(v) GROUP BY r")
    assert said.startswith("Error: the query failed:") and "IN ('a', 'b')" in said, said


# --------------------------------------------------------------------------- no table
HTML = "<!DOCTYPE html><html><body><table><tr><th>a</th></tr>" + "<tr><td>7</td></tr>" * 3
DOCUMENTS = {
    "pdf": ("%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n" * 20, "a PDF"),
    "pdf-base64": ("JVBERi0xLjcKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyA+Pg==", "a PDF"),
    "png": (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "an image",
    ),
    "data-url": ("data:image/jpeg;base64,/9j/4AAQSkZJRg" + "A" * 200, "an image"),
    "binary": ("PK\x03\x04\x14\x00\x00\x00" + "\x00" * 50, "binary data"),
}


@pytest.mark.parametrize("kind", list(DOCUMENTS))
def test_what_has_no_rows_is_refused_with_where_to_compute(kind: str) -> None:
    text, what = DOCUMENTS[kind]
    said = ask(text, "SELECT count(*) FROM result")
    assert said.startswith(f"Error: {results.PREFIX}/") and f"is {what}, not rows" in said, said
    assert "read_file" in said and "sandbox" in said, said


def test_a_run_with_a_sandbox_is_told_to_extract_with_code() -> None:
    run = runtime_module._current.get()
    run.toolbox["sandbox_exec"] = object()  # type: ignore[union-attr]
    pdf = DOCUMENTS["pdf"][0]
    assert "with sandbox_exec" in ask(pdf, "SELECT count(*) FROM result")
    stub = results.readable(pdf * 2000)
    assert "is a PDF, not rows" in stub and "query_result(" not in stub, stub


@pytest.mark.parametrize("text", [HTML, '<?xml version="1.0"?>\n<rows>\n<r a="1"/>\n</rows>'])
def test_markup_is_queried_as_its_lines_and_the_answer_says_so(text: str) -> None:
    """HTML or XML is no table but its lines are: a query runs (its cells can be taken out),
    and its answer first says a count of lines is no count of rows."""
    said = ask(text, "SELECT count(*) n FROM result")
    assert said.startswith(f"Note: {results.PREFIX}/") and "no count of rows" in said, said
    cells = (
        "SELECT sum(CAST(c AS INT)) s FROM result, "
        "unnest(regexp_extract_all(line, '<td[^>]*>([^<]*)', 1)) t(c)"
    )
    if text == HTML:
        assert ask(text, cells).endswith("\ns\n21")
    stub = results.readable(HTML * 2000)
    assert "is an HTML or XML document: the table holds its lines" in stub, stub
    assert "query_result(" in stub, stub
