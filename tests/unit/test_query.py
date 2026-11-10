"""``query_result`` (``tools.query``): read-only SQL over every row of a kept result, on DuckDB —
JSON, CSV or plain lines as a table typed alike, exact answers, nothing but reading."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterator
from decimal import Decimal
from types import SimpleNamespace

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
    assert ask(text, "SELECT count(*) AS n FROM result WHERE stock < 2") == "n\n10"
    assert ask(text, "SELECT sum(price) FROM result;") == "sum(price)\n108.75"
    said = ask(text, "SELECT tags ->> '$[0]' AS t, tags, ok FROM result LIMIT 1")
    assert said == 't,tags,ok\na,"[""a""]",true'  # a nested value: its JSON text
    assert query.columns_of(text) == ["sku", "stock", "price", "tags", "ok"]


def test_csv_with_any_delimiter_ragged_rows_and_odd_names(journal: Journal) -> None:
    text = "SKU id\tunits\tunits\t\nA\t3\t1.5\nB\t\t2\t9\n\nC\t1"
    assert query.columns_of(text) == ["SKU_id", "units", "units_2", "c4"]
    said = ask(text, "SELECT SKU_id, units, units_2 FROM result ORDER BY SKU_id")
    assert said == "SKU_id,units,units_2\nA,3,1.5\nB,,2.0\nC,1,"


def test_plain_text_is_one_row_per_line(journal: Journal) -> None:
    text = "first line\nsecond line\n" + "x" * 10 + '\n"quoted", too'
    assert query.columns_of(text) == ["line"]
    assert ask(text, "SELECT count(*) AS n FROM result WHERE line LIKE '%line'") == "n\n2"
    assert ask(text, "SELECT line FROM result WHERE line LIKE '%too'") == 'line\n"""quoted"", too"'
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
    assert ask("a,b\n1,2", "SELECT a + b AS c FROM result") == "c\n3"  # the table is whole


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
        query.query({"file_path": "x", "sql": "SELECT count(*) AS n FROM result"}, "a\nb")
    )
    assert given == "n\n2"  # a text given (a file Deep Agents saved in the state)


def test_outside_a_run_nothing_is_read() -> None:
    said = asyncio.run(query.query({"file_path": "/x", "sql": "SELECT 1"}, "a,b\n1,2"))
    assert said == "Error: File '/x' not found"


#: The kind a column's ``typeof`` names.
KINDS = {"BIGINT": "INTEGER", "DOUBLE": "REAL", "DECIMAL": "DECIMAL", "VARCHAR": "TEXT"}


@pytest.mark.parametrize(
    ("cells", "kind"),
    [
        (["1", "-2", " 3 "], "INTEGER"),
        (["1", "2.5"], "DECIMAL"),
        (["-0.05", "12.50", ".5", "7."], "DECIMAL"),
        (["123456789012345", "0.000000000000001"], "DECIMAL"),
        (["1234567890123456.5"], "REAL"),  # 17 digits: more than a decimal keeps
        (["0.30000000000000004"], "REAL"),
        (["1e-3", "2"], "REAL"),
        (["007", "8"], "TEXT"),
        (["9" * 40], "TEXT"),
        (["12345678901234567890", "1.5"], "TEXT"),
        ([2**63 - 1, -(2**63)], "INTEGER"),
        ([2**64], "TEXT"),
        ([1, 2.5], "DECIMAL"),
        ([1.0], "DECIMAL"),
        ([0.1 + 0.2], "REAL"),
        ([float("inf")], "TEXT"),
        (["1e999"], "TEXT"),
        (["", None], "TEXT"),
        ([True, False], "TEXT"),
        ([{"a": 1}], "TEXT"),
        (["1.", ".5", "1e5", "+1.5E-3"], "REAL"),
        (["1_000"], "TEXT"),
        (["0x10"], "TEXT"),
        (["\u0663"], "TEXT"),  # a digit of another script
        (["nan"], "TEXT"),
        (["12345678901234567890"], "TEXT"),  # a whole number past 64 bits: never rounded
        (["0", "0.5"], "DECIMAL"),
        ([0.25, 3], "DECIMAL"),  # JSON numbers that are short decimals
        (["+5", "-0"], "INTEGER"),
    ],
)
def test_a_column_is_a_number_only_when_sql_keeps_it_exactly(
    journal: Journal, cells: list, kind: str
) -> None:
    """The typing rules, for JSON values and CSV text alike."""
    texts = [json.dumps([{"c": c, "n": n} for n, c in enumerate(cells)])]
    if all(isinstance(c, str) for c in cells):
        texts.append("n,c\n" + "\n".join(f'{n},"{c}"' for n, c in enumerate(cells)))
    for text in texts:
        said = ask(text, "SELECT typeof(c) AS k FROM result WHERE c IS NOT NULL LIMIT 1")
        found = said.splitlines()[1:] or ["VARCHAR"]  # only NULLs: a VARCHAR column
        assert KINDS[found[0].strip('"').split("(")[0]] == kind, (text, said)


def test_odd_json_keys_and_values_load_alike(journal: Journal) -> None:
    rows = [{"a b": 1, 'q"uote': "x", "sl/ash~": [1, "é"], "it's": None}, {"a b": 2, "new": 3}]
    for text in (json.dumps(rows), json.dumps({"n": 2, "rows": rows})):
        said = ask(text, "SELECT * FROM result")
        assert said == 'a_b,q_uote,sl_ash,it_s,new\n1,x,"[1,""é""]",,\n2,,,,3', said


def test_csv_as_pythons_csv_reads_it(journal: Journal) -> None:
    text = 'a,b\r\n"x,y","he said ""hi"""\r\n"two\nlines",2\r\n\r\n3,4,5\r\n6'
    said = ask(text, "SELECT a, b FROM result")
    assert said == 'a,b\n"x,y","he said ""hi"""\n"two\nlines",2\n3,4\n6,', said


def test_a_boolean_answer_reads_as_sql_writes_it(journal: Journal) -> None:
    said = ask("a,b\n1,2", "SELECT a < b AS yes, a > b AS no FROM result")
    assert said == "yes,no\ntrue,false"


def test_the_description_names_the_engine() -> None:
    """DuckDB, a required dependency: the description names it, so the model writes its SQL."""
    assert isinstance(query.ENGINE, query.DuckDB)
    assert "(DuckDB)" in query.QUERY_RESULT_DESCRIPTION


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
    """Integers stop at 64 bits (``BIGINT``): a longer id is its exact text, never a rounded
    ``DOUBLE`` (two ids made one) nor a failed load."""
    kind = "VARCHAR"
    said = ask(text, "SELECT id, typeof(id) AS k FROM result WHERE v < 3 ORDER BY v")
    assert said == f"id,k\n12345678901234567890,{kind}\n12345678901234567891,{kind}"
    assert ask(text, "SELECT count(DISTINCT id) AS n FROM result") == "n\n3"


def test_two_results_of_one_tool_are_two_tables(journal: Journal) -> None:
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    for text in ("a\n1", "a\n2", "b\n3"):
        results.keep(text, tool="rows")
    assert list(query.tables(run).values()) == ["rows", "rows_2", "rows_3"]


def test_a_result_is_queried_under_the_table_its_note_named(journal: Journal) -> None:
    """A kept table's note names its table (``results.tabled``) before the query runs; the
    results a Deep Agents graph saved in its state join the query only while it runs
    (``query.graph_results``). They never take the name the note gave: they did, and the
    footwear note's ``sku_catalog_2`` read another turn's denim rows (ReAct on Deep Agents,
    advconv-05). The conversation's earlier turns keep their names too."""
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    denim = "\n".join(["sku,cat", *(f"D{n},denim" for n in range(30))])
    run.earlier_results = {results.PREFIX + "/earlier": denim}  # type: ignore[union-attr]
    run.kept_from[results.PREFIX + "/earlier"] = "catalog"  # type: ignore[union-attr]
    footwear = "\n".join(["sku,cat", *(f"F{n},footwear" for n in range(30))])
    noted = results.tabled(footwear, tool="catalog")
    path = noted.split("also saved at ")[1].split(". ")[0]
    assert "FROM catalog_2" in noted  # the earlier turn's is catalog
    shirts = "\n".join(["sku,cat", *(f"S{n},shirts" for n in range(30))])
    evicted = {results.PREFIX + "/call_1": shirts}  # another call's, evicted by Deep Agents
    with query.graph_results(evicted, dict.fromkeys(evicted, "catalog")):
        sql = "SELECT DISTINCT cat FROM catalog_2"
        said = asyncio.run(query.query({"file_path": path, "sql": sql}))
        assert said == "cat\nfootwear", said
        earlier = asyncio.run(query.query({"file_path": path, "sql": "SELECT cat FROM catalog"}))
        assert earlier.splitlines()[1] == "denim", earlier


def amounts(seed: int, n: int = 6000) -> tuple[str, Decimal]:
    """``n`` two-decimal amounts as CSV, and their exact sum."""
    rng = random.Random(seed)
    values = [Decimal(rng.randint(-2000, 2000)).scaleb(-2) for _ in range(n)]
    lines = [f"{n},{v}" for n, v in enumerate(values)]
    return "line,amount\n" + "\n".join(lines), sum(values, Decimal(0))


@pytest.mark.parametrize("seed", [0, 71])
def test_money_adds_up_exactly(journal: Journal, seed: int) -> None:
    """Two-decimal amounts sum to the decimal figure, never a binary float's
    (``32266.149999999998`` for ``32266.15``; seed 71 is a sum near zero, where a compensated
    float sum read at 15 digits says ``-92.9599999999999``): the column is ``DECIMAL``
    (decimal text), added up exactly (ADV-2)."""
    text, exact = amounts(seed)
    assert ask(text, "SELECT sum(amount) s FROM result") == f"s\n{_text(exact)}"
    mean = float(exact / 6000)
    assert ask(text, "SELECT avg(amount) a FROM result") == f"a\n{mean:.15g}"
    whole = "SELECT sum(amount) OVER (ORDER BY line) s FROM result ORDER BY line DESC LIMIT 1"
    assert ask(text, whole) == f"s\n{_text(exact)}"  # a window sums exactly too
    moving = "SELECT sum(amount) OVER (ORDER BY line ROWS 2 PRECEDING) s FROM result"
    moved = ask(text, f"{moving} ORDER BY line LIMIT 3 OFFSET 10")
    values = [Decimal(v.split(",")[1]) for v in text.splitlines()[1:]]
    assert moved.splitlines()[1:] == [
        _text(sum(values[k - 2 : k + 1], Decimal(0))) for k in (10, 11, 12)
    ]


def _text(value: Decimal) -> str:
    """A decimal as a query writes it (a whole one with ``.0``)."""
    shown = format(float(value), ".15g")
    return shown if "." in shown or "e" in shown else f"{shown}.0"


def test_sums_keep_their_sql_meaning(journal: Journal) -> None:
    """Exact sums mean what SQL's do: all integers an integer, no value ``NULL``,
    ``DISTINCT``, a group's, a window's frame."""
    text = "k,n,x\na,1,0.1\na,2,0.2\nb,3,"
    assert ask(text, "SELECT sum(n) AS s FROM result") == "s\n6"
    assert ask(text, "SELECT sum(x) AS s FROM result") == "s\n0.3"
    assert ask(text, "SELECT sum(x) AS s, avg(x) AS a FROM result WHERE k = 'b'") == "s,a\n,"
    assert ask(text, "SELECT sum(DISTINCT n) AS s FROM result") == "s\n6"
    assert ask(text, "SELECT k, sum(x) AS s FROM result GROUP BY k ORDER BY k") == "k,s\na,0.3\nb,"
    frames = "n,x\n1,0.1\n2,\n3,0.2\n4,0.4\n5,1"
    moving = "SELECT sum(x) OVER w s, avg(x) OVER w a FROM result WINDOW w AS (ORDER BY n ROWS 1 PRECEDING)"
    said = ask(frames, moving + " ORDER BY n")
    assert said == "s,a\n0.1,0.1\n0.1,0.1\n0.2,0.2\n0.6,0.3\n1.4,0.7"


def test_a_figure_reads_as_its_digits_not_a_floats_tail(journal: Journal) -> None:
    """A number the query computes is written as 15 significant digits: a product or a ratio
    reads ``59.97``, not ``59.970000000000006``; a decimal ``6.0``."""
    text = "price,qty\n19.99,3\n0.1,3"
    assert ask(text, "SELECT price * qty v FROM result") == "v\n59.97\n0.3"
    assert ask(text, "SELECT 1.0 / 3 v") == "v\n0.333333333333333"
    assert ask(text, "SELECT 1e300 * 10 v, 2.0 * 3 w") == "v,w\n1e+301,6.0"


def test_a_kept_table_cut_to_fit_the_window_keeps_its_note(journal: Journal) -> None:
    """``FitWindow`` cuts the longest results to their head and tail; a kept table's note — its
    path and the table to query — is at its end, and was cut away: the model, told no table,
    queried ``FROM sku_catalog`` and read the earlier turn's knitwear rows for outerwear (the
    ``deepagents`` adapter at 6,000 tokens). The cut keeps the note whole and points at the
    same kept copy."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    rows = "\n".join(["sku,cat,price", *(f"O{n},outerwear,{100 + n}.00" for n in range(300))])
    noted = results.tabled(rows, tool="catalog")
    path = noted.split("also saved at ")[1].split(". ")[0]
    call = AIMessage(content="", tool_calls=[{"id": "c1", "name": "catalog", "args": {}}])
    messages = [HumanMessage("how many?"), call, ToolMessage(noted, tool_call_id="c1")]
    given = ModelRequest(model=ScriptedChatModel(turns=[]), messages=messages, tools=[])
    cut = middleware._cut(given, 1).messages[2].text  # as far as it goes
    assert "characters cut to fit the window" in cut
    assert cut.endswith(noted[len(rows) :])  # the note whole: the path and the table
    assert f'file_path="{path}"' in cut  # the rest is read from the same kept copy
    assert len(run.replay.journal.results) == 1  # type: ignore[union-attr]
