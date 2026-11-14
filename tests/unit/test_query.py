"""``query_result`` (``tools.query``): read-only SQL over every row of a kept result, on DuckDB —
JSON, CSV or plain lines as a table typed alike, exact answers, nothing but reading."""

from __future__ import annotations

import asyncio
import json
import random
import re
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


#: A kept result's path, as a message names it.
PATH = re.compile(rf"{results.PREFIX}/\w+")
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
    assert many.count("\n") == 200 + 1 and "only the first 200 rows" in many
    assert ask(text, "SELECT n FROM result LIMIT 200").count("\n") == 200  # all of them
    longest = "SELECT '" + "x" * 4_986 + "' AS v"
    assert len(longest) == 5_000 and ask(text, longest) == "v\n" + "x" * 4_986  # the longest
    longer = "SELECT '" + "x" * 4_987 + "' AS v"
    assert ask(text, longer) == "Error: the query is longer than 5000 characters"
    wide = ask(text, "SELECT repeat('x', 19998) AS t FROM result LIMIT 1")
    assert wide == "t\n" + "x" * 19_998  # 20,000 characters with its header: whole
    cut = ask(text, "SELECT repeat('x', 19999) AS t FROM result LIMIT 1")
    assert cut == "t\n" + "x" * 19_998 + "\n…[cut at 20000 characters: aggregate more]"
    assert query.SECONDS == 5.0  # docs/tools.md: "a query runs at most 5 s"
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


def test_a_query_names_at_most_8_past_results(journal: Journal) -> None:
    """Past 8 past results the call is refused, before any is read; 8 go on to be read."""

    def asked(n: int) -> str:
        refs = [f"msg_{k}" for k in range(n)]
        return asyncio.run(query.query({"sql": "SELECT 1", "results": refs}))

    assert "results names at most 8 past results" in asked(9)
    assert asked(8) == "Error: past results are read from memory, which this run does not have."


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
        (["123456789012345.6"], "REAL"),  # 16 digits: past the 15 a decimal keeps
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


#: The states a result is in when the model reads of it: shown whole in the conversation (too
#: small to keep), kept by the run (a table read whole with its note, or too large to show: a
#: stub), cleared from the request (a mark), cut to fit the window, saved by Deep Agents as a
#: file of the graph's state (its stub) — and that stub cleared in turn.
HELD = ("shown", "kept", "cleared", "cut", "deep agents", "deep agents cleared")
#: How a mark starts: a cleared result's, and a stub's.
MARKS = ("[result cleared", results.EVICTED)
#: What a held result's message says of it: its table, its columns, the path to query.
NAMED = re.compile(r"Its rows are the table (\w+)")
SAID = re.compile(
    rf"{NAMED.pattern} \(columns: ([^)]*)\)\. .*?"
    rf'{query.QUERY_RESULT}\(file_path="({results.PREFIX}/\w+)"'
)


def rows_of(shape: str, count: int, rng: random.Random) -> tuple[str, int]:
    """``count`` rows as a tool returns them (``shape``), and the sum of their ``units``."""
    units = [rng.randrange(1, 500) for _ in range(count)]
    if shape == "csv":
        return "sku,units\n" + "\n".join(f"S{n},{u}" for n, u in enumerate(units)), sum(units)
    listed = [{"sku": f"S{n}", "units": u} for n, u in enumerate(units)]
    found = {"store": "S20", "rows": listed, "units": sum(units)}
    return json.dumps(listed if shape == "array" else found), sum(units)


def as_read(run: Any, state: dict[str, Any], *, cleared: bool, cut: bool) -> str:
    """The call's tool message as the model reads it: Deep Agents' stubs named their table,
    then (``cleared``) LangChain's clearing with the harness's mark, then (``cut``) the
    request cut to fit a window with no room at all — the order ``middleware()`` runs them."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware
    from trellis.harness.react import CLEARED

    messages = [HumanMessage("how many?"), *state["messages"]]
    request = ModelRequest(
        model=ScriptedChatModel(turns=[]), messages=messages, tools=[], state=state  # type: ignore[arg-type]
    )  # fmt: skip
    request = middleware._stubs_tabled(run, request)
    if cleared:
        edited = [m.model_copy() for m in request.messages]
        clearing = middleware.ThisTurnsResults(
            trigger=0, clear_at_least=0, keep=0, placeholder=CLEARED
        )
        clearing.apply(edited, count_tokens=count_tokens_approximately)
        request = request.override(messages=edited)
    if cut:
        request = middleware._cut(request, 1)
    return str(request.messages[-1].text)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("count", [1, 4, 5, 29, 30, 400, 9_000])
@pytest.mark.parametrize("shape", ["csv", "array", "object"])
@pytest.mark.parametrize("held", HELD)
def test_what_the_model_reads_of_a_held_result_is_that_result(
    journal: Journal, held: str, shape: str, count: int, seed: int
) -> None:
    """Any result, whatever its size, its shape and the state it is in: the words the model
    reads of it name its own columns, and the path and the table they name give every row to
    a query — so a result is never added up by hand because it was small,
    cleared, cut or saved elsewhere. And a mark stays a mark: cut to a window with no room, it
    still names the path and the table, is not marked again, and is never kept as a result
    (told from the message's words, a kept JSON table's mark named columns made of its data,
    and the cut of that mark sent the model to read the mark: 12 runs of 12)."""
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    run.run_id = f"run-{held}-{shape}-{count}-{seed}"  # type: ignore[union-attr]
    text, total = rows_of(shape, count, random.Random(f"{held}-{shape}-{count}-{seed}"))
    call = AIMessage("", tool_calls=[{"id": "call_1", "name": "stock", "args": {}}])
    by_call = f"{results.PREFIX}/call_1"  # what a cleared result's mark, and Deep Agents, name
    state: dict[str, Any] = {"messages": [call]}
    if held.startswith("deep agents"):  # its file in the graph's state, its stub in the messages
        state["files"] = {by_call: {"content": text.split("\n")}}
        stub = results.STUB.split("{preview_note}")[0].format(
            tool_call_id="call_1", file_path=by_call
        )
        state["messages"].append(ToolMessage(stub, tool_call_id="call_1", name="stock"))
    else:  # as the harness's tool returned it: whole, whole with its note, or its stub
        shown = results.readable(text, "stock")
        state["messages"].append(ToolMessage(shown, tool_call_id="call_1", name="stock"))

    def answer(path: str, table: str) -> str:
        asked = {"file_path": path, "sql": f"SELECT count(*) AS n, sum(units) AS u FROM {table}"}
        request = ToolCallRequest(
            tool_call={"name": query.QUERY_RESULT, "args": asked, "id": "q1", "type": "tool_call"},
            tool=None,
            state=state,
            runtime=None,  # type: ignore[arg-type]
        )
        found = asyncio.run(middleware._kept_read(run, request))  # type: ignore[arg-type]
        assert found is not None
        return found.text.splitlines()[-1]

    assert answer(by_call, query.TABLE) == f"{count},{total}"  # by its call's path, always
    for cut in (held == "cut", True):  # as it is read, then with no room left in the window
        read = as_read(run, state, cleared=held.endswith("cleared"), cut=cut)
        marks = [mark for mark in MARKS if mark in read]
        named = NAMED.findall(read)
        if not marks and not named:  # too small to keep, too short to cut: the rows themselves
            assert read == text and count < results.TABLE_ROWS
            continue
        (path,) = set(PATH.findall(read))  # one result, at one path
        if held == "deep agents cleared":
            # its file is the graph's, which the clearing does not read: nothing is said of
            # its table (never a wrong one), and the path the mark names gives its rows
            assert marks and not named and answer(path, query.TABLE) == f"{count},{total}"
            continue
        # said once: its table, under which the path it names gives every row
        assert len(named) == 1 and len(marks) <= 1, read[:900]
        assert answer(path, named[0]) == f"{count},{total}", read[:900]
        said = SAID.findall(read)  # whole unless the window has no room for it
        assert said or (cut and marks), read[:900]
        assert not said or said[0] == (named[0], "sku, units", path), read[:900]
        if marks:  # a mark is whole or shortened, never cut as rows are, never marked twice
            assert read.startswith(marks[0]) and "cut to fit the window" not in read, read[:900]
            assert read.count(marks[0]) == 1
    # and nothing the run keeps is a mark: a mark is never a result
    assert not [p for p, kept in journal.results.items() if kept.startswith(MARKS)]


@pytest.mark.parametrize("count", [1, 4, 5, 30, 400])
@pytest.mark.parametrize("shape", ["csv", "array", "object"])
def test_a_mark_the_graph_wrote_back_or_saved_is_never_read_as_the_result(
    journal: Journal, shape: str, count: int
) -> None:
    """Deep Agents' summarization writes the cleared request back as the graph's state, and
    Deep Agents saves a request's messages as files: the mark then stands where the result
    was, at its call's path. A result the run keeps is still read whole by the path its mark
    names (the kept result's own); one it does not keep is said to be gone. Neither is ever
    answered from the mark's own words (a one-line table: 15 failed queries in 3 live runs)."""
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    run.run_id = f"run-back-{shape}-{count}"  # type: ignore[union-attr]
    text, total = rows_of(shape, count, random.Random(f"back-{shape}-{count}"))
    call = AIMessage("", tool_calls=[{"id": "call_1", "name": "stock", "args": {}}])
    by_call = f"{results.PREFIX}/call_1"
    shown = ToolMessage(results.readable(text, "stock"), tool_call_id="call_1", name="stock")
    state: dict[str, Any] = {"messages": [call, shown]}
    mark = as_read(run, state, cleared=True, cut=False)
    assert mark.startswith(MARKS[0])
    # the graph now holds the mark where the result was, and a file of it at the call's path
    state["messages"][-1] = ToolMessage(mark, tool_call_id="call_1", name="stock")
    state["files"] = {by_call: {"content": mark.split("\n")}}
    assert by_call not in middleware._graph_results(state, run)[0]  # type: ignore[arg-type]
    named = PATH.search(mark)[0]  # type: ignore[index]
    assert (named in journal.results) == (count >= results.TABLE_ROWS)  # kept: its own path
    for path in {by_call, named}:
        asked = {"file_path": path, "sql": "SELECT count(*) AS n, sum(units) AS u FROM result"}
        request = ToolCallRequest(
            tool_call={"name": query.QUERY_RESULT, "args": asked, "id": "q1", "type": "tool_call"},
            tool=None,
            state=state,
            runtime=None,  # type: ignore[arg-type]
        )
        found = asyncio.run(middleware._kept_read(run, request))  # type: ignore[arg-type]
        assert found is not None
        kept = path in journal.results
        wanted = f"{count},{total}" if kept else query.GONE.format(path=path)
        assert found.text.splitlines()[-1] == wanted.splitlines()[-1], found.text


@pytest.mark.parametrize("seed", range(24))
def test_a_page_read_of_a_result_is_cut_as_a_page_and_never_kept_as_a_table(
    journal: Journal, seed: int
) -> None:
    """A page ``read_file`` returns is a view of a result — its lines, numbered — not a
    result: cut to fit a small window, it says where to read on in the file it read, from the
    line after the last one shown, and nothing of it is kept. Kept as a result of its own,
    its rows were the numbered lines ("2  SKU-00501"), named as a table: a join against it
    matched nothing, and the turn ran to its step limit (one live run in three at 8,000)."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    rng = random.Random(seed)
    count, offset, limit = rng.randrange(60, 900), rng.randrange(0, 40), rng.randrange(30, 400)
    text = "sku,units\n" + "\n".join(f"S{n},{rng.randrange(1, 500)}" for n in range(count))
    path = results.keep(text, tool="stock")
    assert path is not None
    asked = {"file_path": path, "offset": offset, "limit": limit}
    page = results.read(dict(asked))
    call = AIMessage("", tool_calls=[{"id": "r1", "name": results.READ_FILE, "args": asked}])
    messages = [
        HumanMessage("which sku?"),
        call,
        ToolMessage(page, tool_call_id="r1", name=results.READ_FILE),
    ]
    given = ModelRequest(
        model=ScriptedChatModel(turns=[]), messages=messages, tools=[], state={"messages": messages}  # type: ignore[arg-type]
    )  # fmt: skip
    kept = dict(journal.results)
    cut = str(middleware._cut(given, rng.choice([1, 40, 150])).messages[-1].text)
    assert dict(journal.results) == kept  # nothing of a page is kept
    if cut == page:
        return  # the page fits the window: shown as it was read
    assert "the table" not in cut and "cut to fit the window" in cut
    assert set(PATH.findall(cut)) == {path}  # the one file: the result it read
    on = int(re.search(rf'{results.READ_FILE}\(file_path="{path}", offset=(\d+)\)', cut)[1])  # type: ignore[index]
    shown = [line for line in cut.split("…[", maxsplit=1)[0].splitlines() if line.strip()]
    lines = text.splitlines()
    assert on == offset + len(shown)  # read on from the line after the last one shown
    assert shown[-1].endswith(lines[on - 1]) and on < len(lines)
    assert lines[on] in results.read({"file_path": path, "offset": on, "limit": 1})


@pytest.mark.parametrize("seed", range(20))
def test_after_a_summarization_every_result_the_run_keeps_is_still_named(
    journal: Journal, seed: int
) -> None:
    """A summarization replaces the older messages — the ones that named the kept results —
    with a summary in the summarizer's own words: whatever it wrote, the request the model
    gets next still names each result the run keeps that no remaining message names, by its
    path and its table, so each is one query away. A summary that named 0 of 56 audit paths
    left a model saying it had "lost track of part of the audit results". Named once: a
    result a remaining message names is not listed again, and a request with no summary is
    as it was."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    rng = random.Random(seed)
    kept = {}
    for _ in range(rng.randrange(1, 60)):
        text, _ = rows_of(rng.choice(["csv", "array", "object"]), rng.randrange(5, 40), rng)
        path = results.keep(text, tool=rng.choice(["stock", "sales", "audit"]))
        assert path is not None
        kept[path] = query.tables(run)[path]
    paths = list(kept)
    shown = rng.sample(paths, rng.randrange(0, len(paths) + 1))  # still named after it
    said = rng.sample(shown, rng.randrange(0, len(shown) + 1))  # those the summary names
    summary = HumanMessage(
        "A condensed summary follows: " + " ".join(said),
        additional_kwargs={"lc_source": "summarization"},
    )
    tail = [ToolMessage(f"rows at {p}", tool_call_id=f"c{i}") for i, p in enumerate(shown)]
    calls = [{"id": f"c{i}", "name": "stock", "args": {}} for i in range(len(shown))]
    messages = [HumanMessage("how many?"), summary, AIMessage("", tool_calls=calls), *tail]
    given = ModelRequest(model=ScriptedChatModel(turns=[]), messages=messages, tools=[])  # type: ignore[arg-type]
    told = middleware._held_named(run, given)  # type: ignore[arg-type]
    read = " ".join(str(m.text) for m in told.messages)
    for path, table in kept.items():
        assert read.count(path) >= 1, path  # every kept result is named
        if path not in shown:
            assert f"{table} at {path}" in told.messages[1].text  # with its table, once
    listed = told.messages[1].text.removeprefix(summary.text)
    assert not [p for p in shown if p in listed]  # one a message still names: not again
    plain = given.override(messages=[messages[0], *messages[2:]])
    assert middleware._held_named(run, plain) is plain  # no summary: as it was


def test_past_the_most_held_results_named_after_a_summary_the_rest_are_counted(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Up to :data:`HELD_MOST` results no message names are named after the summary, with no
    count; past it, those and how many more."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import HumanMessage

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware

    monkeypatch.setattr(middleware, "HELD_MOST", 2)
    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    summary = HumanMessage("A summary.", additional_kwargs={"lc_source": "summarization"})
    given = ModelRequest(model=ScriptedChatModel(turns=[]), messages=[summary], tools=[])  # type: ignore[arg-type]

    def said() -> str:
        return middleware._held_named(run, given).messages[0].text  # type: ignore[arg-type]

    for n in range(2):
        assert results.keep(f"a,b\n{n},2\n" * 40, tool="stock") is not None
    assert "more" not in said()
    assert results.keep("a,b\n9,2\n" * 40, tool="stock") is not None
    assert said().endswith("; and 1 more.")


def test_a_page_whose_call_the_request_no_longer_holds_is_shortened_as_it_stands(
    journal: Journal,
) -> None:
    """A page with no call beside it (its turn left out), or with no whole line in the room
    there is: shortened where it stands — still nothing of it kept, no table named."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from tests.support.chat_model import ScriptedChatModel
    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    text = "sku,units\n" + "\n".join(f"S{n},{n}" for n in range(400))
    path = results.keep(text, tool="stock")
    assert path is not None
    page = results.read({"file_path": path, "limit": 300})
    line = "x" * 9_000  # a page of one long line: no whole line fits
    asked = {"file_path": path, "offset": 0, "limit": 300}
    call = AIMessage("", tool_calls=[{"id": "r2", "name": results.READ_FILE, "args": asked}])
    for messages in (
        [HumanMessage("which?"), ToolMessage(page, tool_call_id="r1", name=results.READ_FILE)],
        [
            HumanMessage("which?"),
            call,
            ToolMessage(line, tool_call_id="r2", name=results.READ_FILE),
        ],
    ):
        given = ModelRequest(
            model=ScriptedChatModel(turns=[]), messages=messages, tools=[], state={"messages": messages}  # type: ignore[arg-type]
        )  # fmt: skip
        kept = dict(journal.results)
        cut = str(middleware._cut(given, 1).messages[-1].text)
        assert dict(journal.results) == kept and "the table" not in cut
        assert "left out to fit the window" in cut and len(cut) < len(messages[-1].text)


def test_outside_a_run_a_cleared_results_mark_is_told_from_what_the_graph_shows() -> None:
    """A graph cleared outside a harness run keeps no result: a cleared result's mark names
    the rows its message showed, and of a stub nothing is said that is not known."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from trellis.harness import middleware
    from trellis.harness.react import CLEARED

    stub = results.STUB.split("{preview_note}")[0].format(
        tool_call_id="c2", file_path=f"{results.PREFIX}/c2"
    )
    calls = [{"id": f"c{n}", "name": "stock", "args": {"n": n}} for n in (1, 2)]
    messages = [
        HumanMessage("how many?"),
        AIMessage("", tool_calls=calls),
        ToolMessage(rows_of("object", 9, random.Random(1))[0], tool_call_id="c1"),
        ToolMessage(stub, tool_call_id="c2"),
    ]
    clearing = middleware.ThisTurnsResults(trigger=0, clear_at_least=0, keep=0, placeholder=CLEARED)
    clearing.apply(messages, count_tokens=count_tokens_approximately)
    shown, saved = (m.text for m in messages[2:])
    assert "the table result (columns: sku, units)" in shown and f"{results.PREFIX}/c1" in shown
    assert saved == CLEARED.replace("{path}", f"{results.PREFIX}/c2")


@pytest.mark.parametrize("seed", range(6))
def test_a_result_that_begins_with_anothers_rows_is_read_as_itself(
    journal: Journal, seed: int
) -> None:
    """Two results of one tool, the later beginning with the earlier's rows (a longer page, a
    wider date range): each is its own table by its call's path. The later one was taken for
    a copy of the earlier (any text that began with a kept result's), and a count over 20
    rows answered 10."""
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from trellis.harness import middleware

    run = runtime_module._current.get()
    run.kept_from = {}  # type: ignore[union-attr]
    rng = random.Random(seed)
    shape = rng.choice(["csv", "json lines"])
    first, more = rng.randrange(5, 60), rng.randrange(1, 60)
    rows = [{"sku": f"S{n}", "units": rng.randrange(1, 500)} for n in range(first + more)]

    def text(count: int) -> str:
        if shape == "json lines":
            return "\n".join(json.dumps(r) for r in rows[:count])
        return "sku,units\n" + "\n".join(f"{r['sku']},{r['units']}" for r in rows[:count])

    calls = [{"id": f"c{n}", "name": "sales", "args": {"n": n}} for n in (1, 2)]
    shown = [results.tabled(text(first), "sales"), results.tabled(text(first + more), "sales")]
    state = {
        "messages": [
            AIMessage("", tool_calls=calls),
            *(ToolMessage(s, tool_call_id=f"c{n}") for n, s in enumerate(shown, start=1)),
        ]
    }
    for n, count in ((1, first), (2, first + more)):
        asked = {"file_path": f"{results.PREFIX}/c{n}", "sql": "SELECT count(*) FROM result"}
        request = ToolCallRequest(
            tool_call={"name": query.QUERY_RESULT, "args": asked, "id": "q", "type": "tool_call"},
            tool=None,
            state=state,
            runtime=None,  # type: ignore[arg-type]
        )
        found = asyncio.run(middleware._kept_read(run, request))  # type: ignore[arg-type]
        assert found is not None and found.text.splitlines()[-1] == str(count), found


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
    two = "sku,units\ns1,3\ns2,4"
    assert query._whole(Cursor([("s1", 3), ("s2", 4)]), names) == (two, two)
    many = [(f"s{n}", n) for n in range(query.ROWS + 5)]
    shown, kept = query._whole(Cursor(list(many)), names)
    assert shown.endswith(
        f"only the first {query.ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    )
    assert kept is not None and kept.splitlines()[-1] == f"s{query.ROWS + 4},{query.ROWS + 4}"
    monkeypatch.setattr(query, "LOAD_CHARS", len(kept) + 1)  # exactly, with its last newline
    assert query._whole(Cursor(list(many)), names) == (shown, kept)
    monkeypatch.setattr(query, "LOAD_CHARS", 100)
    again, dropped = query._whole(Cursor(list(many)), names)
    assert again == shown and dropped is None
