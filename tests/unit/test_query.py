"""``query_result`` (``tools.query``): read-only SQL over every row of a kept result, on DuckDB —
JSON, CSV or plain lines as a table typed alike, exact answers, nothing but reading."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterator
from decimal import Decimal
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


def test_a_failed_query_is_told_the_closest_tables_first_and_stays_short(
    journal: Journal,
) -> None:
    """A failed query's error named every table the run keeps with its columns: with twenty
    of them it was long enough that a small window cut its middle (``FitWindow``), and the
    table the query meant with it. It names those closest to the tables the query reads first,
    :data:`query.TOLD_TABLES` of them, then how many more there are and the query that lists
    them all: as many with forty tables as with twenty."""
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    kinds = ("orders", "order_items", "customers", "refunds", "stock")

    def kept(n: int) -> str:
        path = results.keep(f"id,col_{n}\n1,{n}", tool=f"{kinds[n % 5]}_{n}")
        assert path is not None
        return path

    def sent(sql: str) -> str:
        return asyncio.run(query.query({"file_path": paths[0], "sql": sql}))

    paths = [kept(n) for n in range(20)]
    said = sent("SELECT * FROM order_item_6")  # one letter off
    there = said.split("saved results: ")[1]
    assert there.startswith("order_items_6 (id, col_6); "), said
    assert there.count("(id, col_") == query.TOLD_TABLES
    assert "customers_" not in there and "refunds_" not in there and "stock_" not in there
    assert there.endswith(f"and {20 - query.TOLD_TABLES} more ({query.ALL_TABLES} lists them all).")
    # a query that names a table there is, and fails on a column: that table's columns first
    assert sent("SELECT nope FROM stock_19").split("saved results: ")[1].startswith("stock_19 (")
    own = asyncio.run(query.query({"file_path": paths[7], "sql": "SELECT nope FROM result"}))
    assert own.split("saved results: ")[1].startswith("customers_7 (id, col_7); ")
    # the rest is one query away
    listed = sent(f"{query.ALL_TABLES} ORDER BY 1").splitlines()
    assert {f"{kinds[n % 5]}_{n}" for n in range(20)} <= set(listed)
    paths += [kept(n) for n in range(20, 40)]
    again = sent("SELECT * FROM order_item_6").split("saved results: ")[1]
    assert again.count("(id, col_") == query.TOLD_TABLES and "; and 35 more (" in again


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


def test_a_graphs_copy_of_a_kept_table_is_that_table(journal: Journal) -> None:
    """A table read whole is kept and sent with its note (``results.tabled``); on a small
    window Deep Agents then saves that whole message — the rows and the note after them — as a
    file of its own. The copy is the kept result, not another table: read as it stood it was
    no JSON any more but "CSV" whose header was the whole JSON line, 3,902 columns, and
    ``query_result`` typed them until the run's time ran out (ReAct with a backend, a
    16,000-token window, long-data-02: 420 s)."""
    from langchain_core.messages import AIMessage

    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    rows = [{"key": f"SKU-{n:05d}", "value": round(n * 1.37, 2)} for n in range(1500)]
    noted = results.tabled(json.dumps({"group_by": "sku", "rows": rows}), tool="sales_summary")
    kept = noted.split("also saved at ")[1].split(". ")[0]
    copy = results.PREFIX + "/call_1"  # Deep Agents' file of the message: its call's id
    state = {
        "files": {copy: {"content": noted.split("\n")}},
        "messages": [
            AIMessage("", tool_calls=[{"id": "call_1", "name": "sales_summary", "args": {}}])
        ],
    }
    files, tools = middleware._graph_results(state, run)
    assert files == {} and tools == {}  # the run keeps it already: one table, not two
    # asked by the copy's path, the query is over the kept rows
    text = middleware._state_file(state, copy, run)
    assert text is not None and query._shape(text)[0] == "json"
    with query.graph_results(files, tools):
        said = asyncio.run(
            query.query({"file_path": copy, "sql": "SELECT count(*) FROM result"}, text)
        )
        assert said.splitlines()[-1] == "1500", said
        own = asyncio.run(query.query({"file_path": kept, "sql": "SELECT count(*) FROM result"}))
        assert own.splitlines()[-1] == "1500", own
    # Deep Agents' stub of that message names the kept table, with its columns
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import ToolMessage

    stub = ToolMessage(f"{results.EVICTED}: saved at {copy}", tool_call_id="call_1")
    asked = ModelRequest(
        model=None, messages=[stub], system_message=None, tools=[], tool_choice=None,  # type: ignore[arg-type]
        response_format=None, state=state, runtime=None, model_settings={},  # type: ignore[arg-type]
    )  # fmt: skip
    run.run_id = "run_copy"  # type: ignore[union-attr]
    told = middleware._stubs_tabled(run, asked).messages[0].text  # type: ignore[arg-type]
    assert "the table sales_summary (columns: key, value)" in told, told
    # a file of the graph's that is no copy stays a result of its own
    other = {"files": {copy: {"content": ["sku,cat", "S1,shirts"]}}, "messages": []}
    assert middleware._graph_results(other, run)[0] == {copy: "sku,cat\nS1,shirts"}


#: The states a run holds a result in: shown whole in the conversation (too small to keep),
#: kept by the run (a table read whole, or too large to show), cleared from the request with a
#: mark naming its call's path, and saved by Deep Agents as a file of the graph's state.
HELD = ("shown", "kept", "cleared", "deep agents")


def rows_of(shape: str, count: int, rng: random.Random) -> tuple[str, int]:
    """``count`` rows as a tool returns them (``shape``), and the sum of their ``units``."""
    units = [rng.randrange(1, 500) for _ in range(count)]
    if shape == "csv":
        return "sku,units\n" + "\n".join(f"S{n},{u}" for n, u in enumerate(units)), sum(units)
    listed = [{"sku": f"S{n}", "units": u} for n, u in enumerate(units)]
    return json.dumps(listed if shape == "array" else {"rows": listed}), sum(units)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("count", [1, 4, 29, 30, 400, 9_000])
@pytest.mark.parametrize("shape", ["csv", "array", "object"])
@pytest.mark.parametrize("held", HELD)
def test_a_result_is_queried_exactly_in_whatever_state_the_run_holds_it(
    journal: Journal, held: str, shape: str, count: int, seed: int
) -> None:
    """Any result the run holds is queried by the path the run names for it, over every row,
    whatever its size, its shape and where it is held: the model is never left to add up rows
    because a result was small, cleared or saved elsewhere."""
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    text, total = rows_of(shape, count, random.Random(f"{held}-{shape}-{count}-{seed}"))
    call = AIMessage("", tool_calls=[{"id": "call_1", "name": "stock", "args": {}}])
    by_call = f"{results.PREFIX}/call_1"  # what a cleared result's mark, and Deep Agents, name
    state: dict[str, object] = {"messages": [call]}
    if held == "kept":
        said = results.readable(text, "stock")
        if "saved at " not in said and "saved in " not in said:
            return  # too small to keep: the run shows it whole (the "shown" state)
        path = next(p for p in journal.results if p in said)
    elif held == "deep agents":  # its file in the graph's state, its stub in the conversation
        path, state["files"] = by_call, {by_call: {"content": text.split("\n")}}
        stub = results.STUB.split("{preview_note}")[0].format(tool_call_id="call_1", file_path=path)
        state["messages"] = [call, ToolMessage(stub, tool_call_id="call_1")]
    else:  # shown whole, or cleared from the request: the graph's state has the message
        path = by_call
        shown = results.readable(text, "stock")
        state["messages"] = [call, ToolMessage(shown, tool_call_id="call_1")]
    asked = {"file_path": path, "sql": "SELECT count(*) AS n, sum(units) AS units FROM result"}
    request = ToolCallRequest(
        tool_call={"name": query.QUERY_RESULT, "args": asked, "id": "q1", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,  # type: ignore[arg-type]
    )
    answer = asyncio.run(middleware._kept_read(run, request))  # type: ignore[arg-type]
    assert answer is not None and answer.text.splitlines()[-1] == f"{count},{total}", answer


def test_a_very_wide_table_is_typed_in_bounded_statements() -> None:
    """Typing counts six aggregates a column: over thousands of columns in one statement
    DuckDB ran for minutes and no interrupt stopped it, so the load's time limit did not hold.
    The columns are counted a few at a time, the load's time looked at between two statements:
    a wide table is typed, and one too wide for the time stops there."""
    import time

    def wide(columns: int) -> str:
        names, values = (f"c{n}" for n in range(columns)), map(str, range(columns))
        return ",".join(names) + "\n" + ",".join(values)

    engine = query.ENGINE
    columns = query.TYPED_COLUMNS + 3  # two statements a pass
    with engine.opened() as db:
        engine.load(db, {"wide": wide(columns)}, "wide", until=time.monotonic() + 60)
        last = db.execute(f"SELECT typeof(c{columns - 1}), c{columns - 1} FROM result")
        assert last.fetchone() == ("BIGINT", columns - 1)
    with engine.opened() as db, pytest.raises(TimeoutError, match="took too long"):
        engine.load(db, {"wide": wide(columns + 1)}, "wide", until=time.monotonic() - 1)


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


def test_a_querys_rows_past_what_a_query_loads_are_shown_and_not_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a sandbox a query's rows are kept as a file for code (``_whole``): one row is a
    figure, not a file; more are kept whole, read in batches; and a result larger than a query
    could load back (``LOAD_CHARS``) is shown as any other and not kept."""

    class Cursor:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.rows = rows

        def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
            taken, self.rows = self.rows[:size], self.rows[size:]
            return taken

    names = ["sku", "units"]
    assert query._whole(Cursor([("s1", 3)]), names) == ("sku,units\ns1,3", None)
    many = [(f"s{n}", n) for n in range(query.ROWS + 5)]
    shown, kept = query._whole(Cursor(list(many)), names)
    assert shown.endswith(
        f"only the first {query.ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    )
    assert kept is not None and kept.splitlines()[-1] == f"s{query.ROWS + 4},{query.ROWS + 4}"
    monkeypatch.setattr(query, "LOAD_CHARS", 100)
    again, dropped = query._whole(Cursor(list(many)), names)
    assert again == shown and dropped is None
