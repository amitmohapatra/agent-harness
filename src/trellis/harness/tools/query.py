"""``query_result``: a read-only SQL query over every row of a large result the run keeps
(``tools.results``) — a count, a sum, a filter, a top-N or a group computed in process, exactly,
instead of the model paging thousands of rows into its context and adding them up itself.

No sandbox and no Code Mode needed. The engine is DuckDB (:class:`DuckDB`, a required
dependency: `ADR 0007 <docs/adr/0007-query-engine.md>`_), in memory, made for one query and
closed after it; the tool's description names it, so the model writes its dialect. A text's
typed tables are kept as Parquet by its content (:class:`Typed`): the next query of it, in
this run, a later turn or a past-result load, reads them instead of parsing it again.

The kept text becomes the table ``result``: a JSON array of objects, JSON Lines (an object
a line), or the arrays of objects a JSON object holds, however deep (the shallowest: the
first is ``result``, each other ``result_<key>``); a markdown table; CSV with a header line
(comma, tab, semicolon or pipe; one column when its values are numbers or dates); else one
row per line (``line``). A workbook kept for the user is a table a sheet (``result``, then
``result_<sheet>``). What is no table — a PDF, an image, binary data — is refused
(:func:`document`), never counted by its lines; HTML or XML is queried as its lines, the
answer saying so first (:data:`MARKED`).

A column whose every value (a missing one left out: :data:`MISSING`, ``NULL``, ``N/A``,
``-``…) is a whole number is ``BIGINT``, a decimal as written of at most
:data:`DECIMAL_DIGITS` digits (``12.50``, money) ``DECIMAL(p, s)``, another number
``DOUBLE``, else ``VARCHAR``: a code with a leading zero, or an id past 64 bits, keeps its
exact text; a nested value is its JSON text (``json_extract`` reads it). A ``DECIMAL`` column
adds up exactly, as written. Money, percentages and figures with separators stay text, as
written (one column may hold two currencies): a failed sum says how to convert them. Times
with a zone are compared and written in UTC, whatever the host's zone. Every number with a
fraction an answer holds is written with 15 significant digits (:func:`_shown`):
``32266.15``, never ``32266.149999999998``. A number is written as SQL writes one
(:data:`WHOLE`, :data:`DECIMAL`, :data:`NUMBER`): never ``1_000``, ``0x10`` or ``٣``.

Only reading is allowed, and only the run's own results are ever loaded: one ``SELECT``
statement (``WITH`` included), and once its tables are loaded no file or network access
(``enable_external_access``), no extension loaded or installed, and its settings locked. A
query runs for at most :data:`SECONDS` (then it is interrupted, as a cancelled call is), its
text is at most :data:`SQL_CHARS`, what it loads at most :data:`LOAD_CHARS`, its memory at most
:data:`DB_BYTES` (``memory_limit``, never spilled to disk), and at most :data:`ROWS` rows and
:data:`OUTPUT_CHARS` characters come back (said when there were more).

Formulas: arithmetic, ``CASE``, math (``sqrt``, ``power``, ``ln``, ``exp``…), dates
(``julianday``, ``date_diff``, ``d + INTERVAL 7 DAY``) and window functions
(``avg(x) OVER (ORDER BY d ROWS 6 PRECEDING)`` a moving average, ``lag`` a growth rate,
``rank``); and the statistics, named as PostgreSQL and DuckDB name them: ``stddev``/
``stddev_samp``, ``stddev_pop``, ``variance``/``var_samp``, ``var_pop``, ``median``,
``percentile_cont(x, p)`` (``p`` from 0 to 1, interpolated), ``corr(y, x)``,
``covar_samp(y, x)``, ``covar_pop(y, x)``, ``regr_slope(y, x)``, ``regr_intercept(y, x)``,
``regr_r2(y, x)``, ``NULL`` where there are too few values. ``percentile_cont(x, p)`` and
``julianday(d)`` are macros (:data:`MACROS`), so these spellings work as written.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import difflib
import hashlib
import importlib
import io
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections import ChainMap, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from decimal import Decimal
from functools import cached_property
from typing import Any, Final

from trellis.contracts import ToolError
from trellis.harness.files import file_name
from trellis.harness.runtime import current
from trellis.harness.tools import export
from trellis.harness.tools.base import arguments_problem, not_run

#: The tool, and what it is told and given.
QUERY_RESULT: Final = "query_result"
TABLE: Final = "result"
QUERY_RESULT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "The path the tool message gives."},
        "sql": {
            "type": "string",
            "description": f"One SELECT over the table {TABLE}, e.g. "
            f"SELECT count(*) FROM {TABLE} WHERE stock < 10.",
        },
        "results": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Past tool results to load too, by the id an earlier conversation's "
            "steps name (result msg_...): the tables t1, t2... in this order (without "
            f"file_path, t1 is also {TABLE}). This run's own results are tables already.",
        },
    },
    "required": ["sql"],
    "additionalProperties": False,
}
#: The schema of a run that saves a query's result as a file (:func:`saving`): with
#: ``save_as`` and ``file_name`` (``tools.export``).
QUERY_RESULT_SAVING_SCHEMA: Final[dict[str, Any]] = {
    **QUERY_RESULT_SCHEMA,
    "properties": {
        **QUERY_RESULT_SCHEMA["properties"],
        "save_as": {
            "enum": list(export.formats()),
            "description": "Also save the query's whole result as a file the user downloads.",
        },
        "file_name": {"type": "string", "description": "With save_as: the file's name."},
    },
}
#: The most past results one query names (``results``), and how the memory context names one:
#: an earlier conversation's steps list, ``result msg_...`` (the memory service's format).
PAST_RESULTS: Final = 8
_PAST: Final = re.compile(r"\bresult (msg_\w+)")
#: What a failure on a text column used as a number adds (:data:`_TEXT_MISUSED`): money,
#: separators and percentages are kept as written (``tools.query``'s typing rules).
TEXT_NUMBERS: Final = (
    ". The column is text: a figure written with a symbol or a separator ($1,234.50, 12%, "
    "(5.00)) is kept as written; convert it in the query, e.g. TRY_CAST(replace(replace(v, "
    "'$', ''), ',', '') AS DECIMAL(18, 2)), for one currency at a time"
)
_TEXT_MISUSED: Final = re.compile(r"No function matches|Cannot compare|Could not convert")
#: The rows a saved file's answer shows (the file holds them all).
SAVED_ROWS: Final = 5
#: The seconds a query saving its whole result may run, writing included.
SAVE_SECONDS: Final = 60.0
#: The most rows a query answers with, its longest text, the seconds it may run, the memory
#: its database may take, and the delimiters a CSV result is read with.
ROWS: Final = 200
SQL_CHARS: Final = 5_000
SECONDS: Final = 5.0
#: What a query may load (the run's saved results, in characters), the memory its database may
#: take, the most characters it answers with, how many queries run at once in a process (each
#: in a worker thread) and how long one waits its turn.
LOAD_CHARS: Final = 64 * 1024 * 1024
DB_BYTES: Final = 512 * 1024 * 1024
OUTPUT_CHARS: Final = 20_000
CONCURRENT: Final = 4
WAIT_SECONDS: Final = 60.0
#: The most a query's tables may take to load (the run's saved results parsed), beside its
#: own :data:`SECONDS`.
LOAD_SECONDS: Final = 60.0
#: How often a query's time is looked at: one past it is interrupted, again until it stops.
INTERRUPT_SECONDS: Final = 0.05
_RUNNING: Final = threading.BoundedSemaphore(CONCURRENT)
DELIMITERS: Final = ",\t;|"
_QUOTED: Final = re.compile(r'"[^"]*"')
#: A markdown table's rule line (``|---|:--:|``) and a header line that names one column.
_RULE: Final = re.compile(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?")
_HEAD: Final = re.compile(r"[^\W\d]\w*")
#: The typing rules (see the module), one text for Python's ``re`` and DuckDB's RE2: a whole
#: number as SQL writes one, and a number — never with a leading zero before a digit
#: (``00123``, ``-007``: a code, not a number); spaces around one are left out.
WHOLE: Final = r"[-+]?(0|[1-9][0-9]*)"
#: A decimal as written (digits and a point, no exponent: money), and any number.
DECIMAL: Final = r"[-+]?((0|[1-9][0-9]*)(\.[0-9]*)?|\.[0-9]+)"
NUMBER: Final = DECIMAL + r"([eE][-+]?[0-9]+)?"
#: The most digits a ``DECIMAL`` value has: an answer writes 15 significant digits
#: (:func:`_shown`), which a float keeps exactly (its text read back is the same).
DECIMAL_DIGITS: Final = 15
#: A value one column of a CSV holds under its header: a number, or an ISO date or time.
_VALUE: Final = re.compile(NUMBER + r"|[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.]+(Z|[+-][0-9:]+)?)?")
#: What a cell holding no value says: among numbers, it is missing (``NULL``), not text.
MISSING: Final = ("NULL", "NONE", "NA", "N/A", "#N/A", "NAN", "-", "--", "\u2014")
#: The longest text read as a number.
NUMBER_CHARS: Final = 32
#: The Parquet of typed tables a process keeps (:class:`Typed`): as much as the results its
#: conversations keep (``tools.results.KEPT_CHARS``), whose Parquet is several times smaller.
TYPED_BYTES: Final = LOAD_CHARS
#: The table a source's table is loaded as, before it takes its name.
LOADED: Final = "loaded"
#: The rows DuckDB first types a table's columns by: a column with a value no number there
#: is text, not looked at again over every row (:meth:`DuckDB._typed`).
SAMPLE_ROWS: Final = 2_048


def rows_of(text: str) -> tuple[list[str], list[list[Any]]]:
    """A kept result as a table: its columns and rows (see the module) — a JSON object's
    first array of objects."""
    kind, found = _shape(text)
    if kind == "json":
        rows = found[0][1]
        columns = _keys(rows)
        return columns, [[row.get(c) for c in columns] for row in rows]
    if kind == "table":
        return found
    if kind == "csv":
        read = list(csv.reader(io.StringIO(text), delimiter=found))
        width = len(read[0])
        return list(read[0]), [_padded(r, width) for r in read[1:] if r]
    return ["line"], [[line] for line in found]


def _shape(text: str, *, values: bool = True) -> tuple[str, Any]:
    """What a kept result is, as a table (see the module): ``("json", parts)`` (each array
    of objects with the keys that reach it, ``()`` for an array or JSON Lines; without
    ``values``, each row only its keys: :func:`_json_rows`), ``("table", (columns, rows))``
    (markdown), ``("csv", delimiter)`` or ``("lines", lines)``."""
    found = _json_rows(text, values=values)
    if found is not None:
        return "json", found
    lines = text.splitlines()
    table = _markdown(lines)
    if table is not None:
        return "table", table
    head = _QUOTED.sub("", lines[0]) if lines else ""  # a delimiter quoted is a name's
    delimiter = max(DELIMITERS, key=head.count)
    if len(lines) > 1 and (head.count(delimiter) > 0 or _one_column(lines)):
        return "csv", delimiter
    return "lines", lines


def _markdown(lines: Sequence[str]) -> Table | None:
    """A markdown table (``| a | b |`` over its rule line): its columns and its rows, each
    cell trimmed, the rule line no row; ``None`` when the text is none."""
    if len(lines) < 2 or "|" not in lines[0] or not _RULE.fullmatch(lines[1].strip()):
        return None

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().removeprefix("|").removesuffix("|").split("|")]

    columns = cells(lines[0])
    return columns, [_padded(cells(line), len(columns)) for line in lines[2:] if "|" in line]


def _one_column(lines: Sequence[str]) -> bool:
    """Whether lines with no delimiter are a one-column CSV: a name over numbers or dates."""
    values = [v for v in map(str.strip, lines[1:]) if v]
    head = lines[0].strip().lstrip("\ufeff")
    return bool(values) and bool(_HEAD.fullmatch(head)) and all(map(_VALUE.fullmatch, values))


def _keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(k for row in rows for k in row))


#: A table given as rows (a kept Excel file's sheet: ``tools.stored``): its columns and rows.
Table = tuple[list[str], list[list[Any]]]
#: A kept workbook's sheets, by name: the first is the result's table, each other one of its
#: own (:func:`part`).
Sheets = Mapping[str, Table]
#: What a query loads as a result's table (or tables).
Source = str | Sheets


def columns_of(text: Source) -> list[str]:
    """The columns a kept result's table has (as :func:`query` names them)."""
    return parts_of(text)[0][1]


def parts_of(text: Source) -> list[tuple[str | None, list[str]]]:
    """The tables a kept result is, each with its columns (as :func:`query` names them): its
    own (``None``), then each other array of objects its JSON holds, or each other sheet, by
    its key or name (:func:`part` names its table)."""
    if isinstance(text, Mapping):
        return [
            (None if n == 0 else s, column_names(t[0])) for n, (s, t) in enumerate(text.items())
        ]
    kind, found = _shape(text)
    if kind == "json":
        return [
            (keys[-1] if n else None, column_names(_keys(rows)))
            for n, (keys, rows) in enumerate(found)
        ]
    return [(None, column_names(rows_of(text)[0]))]


def part(table: str, name: str) -> str:
    """The table a result's part (an array its JSON holds, a sheet) is: ``<table>_<name>``."""
    return column_names([f"{table}_{name}"])[0]


def also(parts: Sequence[tuple[str | None, list[str]]], table: str) -> str:
    """What a result's other parts are (:func:`parts_of`), as tables of ``table``'s query."""
    said = [
        f'"{name}" is the table {part(table, name)} (columns: {", ".join(columns)})'
        for name, columns in parts
        if name is not None
    ]
    return f" Its other rows are tables too: {'; '.join(said)}." if said else ""


def _size(text: str | Table | Sheets) -> int:
    """What a table takes to load, in characters (a table given as rows: its cells' text)."""
    if isinstance(text, str):
        return len(text)
    if isinstance(text, Mapping):
        return sum(map(_size, text.values()))
    return sum(len(str(v)) for row in text[1] for v in row)


#: What markup is (:func:`document`): its lines are a table still, which a query reads with
#: care — an answer over them says first what they are (:data:`MARKED`).
MARKUP: Final = "an HTML or XML document"
MARKED: Final = (
    "Note: {path} is " + MARKUP + ": the table holds its lines (column line), not its own "
    "table's rows, so a count of lines is no count of rows; take its cells out with "
    "regexp_extract_all(line, '<td[^>]*>([^<]*)', 1), or with code in a sandbox."
)
#: What a result that is no table is (:func:`document`), and what a query of it answers.
NOT_ROWS: Final = (
    "{path} is {what}, not rows: query_result reads JSON or CSV rows, and would only count its "
    "lines. Read it with read_file{how}"
)


def document(text: str) -> str | None:
    """What ``text`` is when it is no table — HTML or XML (:data:`MARKUP`: its lines are
    queried, said so), a PDF, an image (its bytes or their base64), binary data (refused) —
    else ``None``."""
    head = text[:2048].lstrip("\ufeff \t\r\n")
    if head.startswith(("%PDF-", "JVBERi0")):
        return "a PDF"
    if head.startswith(("data:image/", "iVBORw0KGgo", "/9j/", "R0lGOD", "\x89PNG", "GIF8")):
        return "an image"
    if "\x00" in head:
        return "binary data"
    if head.startswith("<") and ("</" in head or "/>" in head or head.startswith("<?xml")):
        return MARKUP
    return None


def _markup(runtime: Any, path: str, given: Source | None) -> bool:
    """Whether the result a query read at ``path`` is HTML or XML (:data:`MARKUP`)."""
    text = given if given is not None else saved(runtime).get(path)
    return isinstance(text, str) and document(text) == MARKUP


def not_rows(path: str, what: str) -> str:
    """What a query of ``path``, which is ``what`` (:func:`document`), answers: where its
    tables or text are computed instead — code in the run's sandbox, when it has one."""
    from trellis.harness.tools.results import coding_tool  # noqa: PLC0415 - it imports this

    tool = coding_tool()
    how = (
        f"; to extract its tables or figures, run code over the file {path} with {tool}."
        if tool is not None
        else "; extracting its tables or figures takes code, run in a sandbox, which this run "
        "has not."
    )
    return NOT_ROWS.format(path=path, what=what, how=how)


#: Where a large result is kept (``tools.results.PREFIX``), and what a path there that no run
#: keeps reads as: a result saved in a turn this process no longer keeps.
KEPT_PREFIX: Final = "/large_tool_results/"
GONE: Final = (
    "Error: File '{path}' not found: it is not a result this conversation keeps here (results "
    "are kept for a conversation's later turns in the process that saved them, while there is "
    "room). Call the tool that returned it again."
)


def not_found(path: str) -> str:
    """What reading or querying ``path``, which nothing keeps, answers."""
    return (
        GONE.format(path=path)
        if path.startswith(KEPT_PREFIX)
        else f"Error: File '{path}' not found"
    )


#: The results a Deep Agents graph saved in its state (each at its own path,
#: ``/large_tool_results/<call id>``) and the tool that returned each, while a call of
#: ``query_result`` reads them (:func:`graph_results`): tables of the query, as the run's own are.
_GRAPH: ContextVar[tuple[Mapping[str, str], Mapping[str, str]]] = ContextVar(
    "graph_results", default=({}, {})
)


@contextlib.contextmanager
def graph_results(files: Mapping[str, str], tools: Mapping[str, str]) -> Iterator[None]:
    """While it lasts, the graph's saved results ``files`` (by path; ``tools``: the tool that
    returned each) are results of the run's (:func:`saved`), each a table named by its tool."""
    token = _GRAPH.set((files, tools))
    try:
        yield
    finally:
        _GRAPH.reset(token)


def saved(runtime: Any) -> Mapping[str, str]:
    """Every result ``runtime``'s run can read, by path: those it saved (its journal), then
    those its conversation's earlier turns saved in this process (``Runtime.earlier_results``:
    ``tools.results.Conversations``) — a follow-up reads or queries the path its history
    names — then those its Deep Agents graph saved in its state (:func:`graph_results`)."""
    own: dict[str, str] = runtime.replay.journal.results
    earlier: Mapping[str, str] = getattr(runtime, "earlier_results", None) or {}
    more = [dict(m) for m in (earlier, _GRAPH.get()[0]) if m]
    return ChainMap(own, *more) if more else own


def tables(runtime: Any) -> dict[str, str]:
    """Every result the run keeps, by its path, and the table it is in a query: the name of the
    tool that returned it (``results.keep``'s ``tool``, in this attempt), else ``result_<n>``
    — each name once, never :data:`TABLE` (the queried file's own) — given in the order the
    results became the run's: its conversation's earlier turns', then its own, then those its
    Deep Agents graph saved, which join only while a query runs (:func:`graph_results`). So a
    name a kept table's note gave (``results.tabled``) stays that table's: a graph's file named
    first took it, and a query read another turn's rows under it."""
    graph, tools = _GRAPH.get()
    tools = {**tools, **(getattr(runtime, "kept_from", None) or {})}
    earlier = getattr(runtime, "earlier_results", None) or {}
    used = {TABLE}
    found: dict[str, str] = {}
    order = dict.fromkeys([*earlier, *runtime.replay.journal.results, *graph])
    for n, path in enumerate(order, start=1):
        base = column_names([tools.get(path) or f"{TABLE}_{n}"])[0]
        name, more = base, 1
        while name in used:
            more += 1
            name = f"{base}_{more}"
        used.add(name)
        found[path] = name
    return found


def saving(runtime: Any) -> bool:
    """Whether ``query_result`` saves a query's whole result as a file in ``runtime``'s run
    (``save_as``): a run with no tool that runs code (``tools.export.runs_code``) — a run with
    one writes its files with its code."""
    return runtime is not None and export.runs_code(runtime) is None


def past_refs(text: str) -> tuple[str, ...]:
    """The past results ``text`` (a memory context) names, each once, in order."""
    return tuple(dict.fromkeys(_PAST.findall(text)))


async def _past(runtime: Any, refs: Sequence[str]) -> tuple[dict[str, str], list[str]] | str:
    """The past results ``refs`` name, read through the memory service's message read (the
    audience rules of the thread each was said in: another user's is refused), as the tables
    ``t1``, ``t2``… and where each came from, dated; or why one cannot be loaded. A ref that
    names a result this run keeps — its path, the id the path ends with, or its table
    (:func:`tables`: what its tool message calls it) — is that result; one memory does not
    know is said with the run's own tables, so the model queries them rather than fetching
    them again."""
    from trellis.memory import AuthorizationError, NotFoundError  # noqa: PLC0415

    memory = getattr(runtime, "run_memory", None)
    kept = saved(runtime) if runtime is not None else {}
    own = tables(runtime) if kept else {}
    named = {**{p.rsplit("/", 1)[-1]: p for p in own}, **{t: p for p, t in own.items()}}
    texts: dict[str, str] = {}
    told: list[str] = []
    for n, ref in enumerate(refs, start=1):
        path = named.get(ref) or "/" + ref.lstrip("/")
        if path in kept:  # a result of this run's, named here too: that table
            texts[f"t{n}"] = kept[path]
            told.append(f"t{n} is this run's result {path}")
            continue
        if memory is None:
            return "Error: past results are read from memory, which this run does not have."
        try:
            found = await memory.past_result(ref)
        except AuthorizationError:
            return (
                f"Error: past result {ref} is refused: it is not this user's to read. "
                "Nothing was run."
            )
        except NotFoundError:
            if own:
                return (
                    f"Error: {ref} is no past result. This run's own results are tables "
                    f"already: {', '.join(own.values())}; name them in the SQL. Nothing was run."
                )
            return (
                f"Error: past result {ref} is no longer kept (deleted, or past its retention): "
                f"call the tool again. Nothing was run."
            )
        if found.role != "TOOL":
            return f"Error: {ref} is not a tool's result. Nothing was run."
        texts[f"t{n}"] = found.content
        when = f", said {found.occurred_at.date().isoformat()}" if found.occurred_at else ""
        told.append(f"t{n} is past result {ref}{when}")
    return texts, told


def schema(runtime: Any) -> dict[str, Any]:
    """``query_result``'s input schema in ``runtime``'s run (:func:`saving`)."""
    return QUERY_RESULT_SAVING_SCHEMA if saving(runtime) else QUERY_RESULT_SCHEMA


async def query(args: dict[str, Any], text: str | None = None) -> str:
    """``query_result``: the query's rows as CSV (a header line first), at most :data:`ROWS`
    and :data:`OUTPUT_CHARS`; or what is wrong — the arguments, the path, the SQL (the engine's
    own words, and the tables there are). The file asked about is the table :data:`TABLE`;
    every other result the run keeps is a table too (:func:`tables`), so one query joins and
    groups across them — only the run's own: nothing else is ever loaded. With ``save_as`` (a
    run that is :func:`saving`), the query's every row is also kept as a file for the user
    (``Runtime.save_file``), and the answer says so, with its first :data:`SAVED_ROWS` rows.

    The query runs in a worker thread (the event loop never waits on it), at most
    :data:`CONCURRENT` at once in the process, on a database made for it and closed after it
    whatever happens; a cancelled call (the run cancelled, or out of time) interrupts it."""
    runtime = current()
    problem = _problem(runtime, args)
    if problem is not None:
        return problem
    sql = str(args["sql"]).strip().rstrip(";")
    refs = [str(r) for r in args.get("results") or ()]
    past = await _past(runtime, refs)
    if isinstance(past, str):
        return past
    path = None if args.get("file_path") is None else "/" + str(args["file_path"]).lstrip("/")
    if path is not None and path.rsplit("/", 1)[-1] in refs and path not in saved(runtime):
        path = None  # a path guessed for a past result it also names: that result is the table
    record: dict[str, Any] | None = None
    given: Source | None = text
    if text is None and path is not None and _stored(path):
        found = await _kept_file(path)
        if isinstance(found, str):
            return found
        given, record = found
    if args.get("save_as") is not None:
        return await _save(runtime, args, sql, given, record, past=past[0])
    answer = await run(path, sql, (), given, past=past[0])
    if answer.startswith("Error"):
        return answer
    if path is not None and _markup(runtime, path, given):
        answer = f"{MARKED.format(path=path)}\n{answer}"
    told = [_told(runtime, path, record, sql, given)] if record is not None and path else []
    told += past[1]
    return "; ".join(told) + f".\n{answer}" if told else answer


def _problem(runtime: Any, args: dict[str, Any]) -> str | None:
    """What is wrong with a call's arguments, as the model reads it; ``None`` when nothing."""
    problem = arguments_problem(schema(runtime), args)
    if problem is None and len(args.get("results") or ()) > PAST_RESULTS:
        problem = f"results names at most {PAST_RESULTS} past results"
    if problem is not None:
        return not_run(QUERY_RESULT, problem)
    if len(str(args["sql"]).strip().rstrip(";")) > SQL_CHARS:
        return f"Error: the query is longer than {SQL_CHARS} characters"
    return None


def _stored(path: str) -> bool:
    from trellis.harness.tools import stored  # noqa: PLC0415 - it reads the results' pages

    return stored.is_stored(path)


async def _kept_file(path: str) -> tuple[Source, dict[str, Any]] | str:
    """A file kept for the user (``tools.stored``), loaded from the store: its table and its
    record, or what is wrong."""
    from trellis.harness.tools import stored  # noqa: PLC0415

    return await stored.table(path)


def _told(runtime: Any, path: str, record: dict[str, Any], sql: str, given: Source | None) -> str:
    """Where a query over a kept file read its rows: the file — when it was saved, by what,
    from what fetched when (its record), each other sheet's table — and each of this run's
    results it names, fetched when: what a comparison of the two cites."""
    from trellis.harness.tools import stored  # noqa: PLC0415

    said = [f"{TABLE} is {path} ({stored.made(record)})"]
    if isinstance(given, Mapping):
        said.extend(f"its sheet {s} is {part(TABLE, s)}" for s in list(given)[1:])
    said.extend(
        f"{s['table']} is this run's {s['tool']} result, fetched {stored.when(s['fetched_at'])}"
        for s in _sources(runtime, path, sql)
        if s["table"] != TABLE
    )
    return "; ".join(said)


async def _save(
    runtime: Any,
    args: dict[str, Any],
    sql: str,
    text: Source | None,
    record: dict[str, Any] | None = None,
    *,
    past: Mapping[str, str] | None = None,
) -> str:
    """``query_result`` with ``save_as``: the query's every row kept as a file for the user
    (``Runtime.save_file``), its name ``file_name`` with the format's extension; what was kept,
    and its first :data:`SAVED_ROWS` rows — or why nothing was."""
    kind = args["save_as"]
    if kind not in export.formats():
        return not_run(QUERY_RESULT, f"save_as is one of: {', '.join(export.formats())}")
    name = str(args.get("file_name") or QUERY_RESULT)
    name = name if name.lower().endswith(f".{kind}") else f"{name}.{kind}"
    try:
        file_name(name)
    except ToolError as exc:
        return not_run(QUERY_RESULT, exc.message)
    path = args.get("file_path")
    found = await run(path, sql, (), text, save=kind, past=past)
    if isinstance(found, str):
        return found
    data, count, shown = found
    made = {"sql": sql, "from": _sources(runtime, path, sql, record)}
    stored = await runtime.save_file(name, data, source=QUERY_RESULT, made=made)
    return (
        f"Saved {stored.name} for the user: {count} rows, {stored.size} bytes; the chat shows "
        f"its download link. Its first rows:\n{shown}"
    )


def _sources(
    runtime: Any, file_path: str | None, sql: str, record: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The results a query read — the file asked about, and each other result the run keeps
    whose table the query names — each as its table, the tool it came from and when it was
    fetched (``results.keep``), for how a file it made was made (``FileStore.record``); a file
    kept for the user (its ``record``) as its path and when it was saved."""
    path = None if file_path is None else "/" + file_path.lstrip("/")
    named = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
    tools, times = getattr(runtime, "kept_from", {}), getattr(runtime, "kept_at", {})
    found: list[dict[str, Any]] = []
    if record is not None:
        found.append({"table": TABLE, "path": path, "fetched_at": record.get("saved_at")})
    for kept, table in tables(runtime).items():
        if kept == path or table in named:
            fetched = times.get(kept)
            found.append(
                {
                    "table": TABLE if kept == path else table,
                    "tool": tools.get(kept),
                    "fetched_at": fetched.isoformat() if fetched is not None else None,
                }
            )
    return found


def loadable(
    file_path: str | None, text: Source | None = None, past: Mapping[str, str] | None = None
) -> tuple[dict[str, Source], str] | str:
    """What a query of ``file_path`` loads — the run's saved results by table name (the file
    asked about among them, given as ``text`` when it is no result the run keeps: a file Deep
    Agents saved, a file kept for the user, read from the store), and the ``past`` results
    (``t1``…) — and the table :data:`TABLE` is (the file's; with no file, ``t1``; with neither,
    none: the query computes with the figures it names, and over the run's tables if it names
    them); or why there is nothing to load."""
    runtime = current()
    kept = saved(runtime) if runtime is not None else {}
    named = tables(runtime) if runtime is not None else {}
    texts: dict[str, Source] = {named[p]: kept[p] for p in named} | dict(past or {})
    if file_path is None:  # past results, or none at all: figures only
        own = next(iter(past or {}), "t1")
    else:
        path = "/" + file_path.lstrip("/")
        if text is None:
            text = kept.get(path)
        if text is None or runtime is None:
            return not_found(path)
        what = document(text) if isinstance(text, str) else None
        if what is not None and what != MARKUP:
            return f"Error: {not_rows(path, what)}"
        if path not in named:
            texts[f"{TABLE}_file"] = text  # a file Deep Agents saved, a kept one: ``result``
        own = named.get(path, f"{TABLE}_file")
    if sum(map(_size, texts.values())) > LOAD_CHARS:
        return (
            f"Error: this run's saved results are more than {LOAD_CHARS} characters, more than a "
            "query here loads: filter on the tool's side, or compute in a sandbox"
        )
    return texts, own


async def run(
    file_path: str | None,
    sql: str,
    params: Sequence[Any] = (),
    text: Source | None = None,
    *,
    save: str | None = None,
    past: Mapping[str, str] | None = None,
) -> Any:
    """``sql`` (with ``params``) over the run's saved results (:func:`loadable`), in a worker
    thread (see :func:`query`): its rows shown, or — ``save``, a format of ``tools.export`` —
    every row as a file of that format, how many, and the first :data:`SAVED_ROWS` shown; or
    what is wrong, as text."""
    found = loadable(file_path, text, past)
    if isinstance(found, str):
        return found
    texts, own = found
    call = _Call()
    take, seconds = (_first, SECONDS) if save is None else (_saved(save), SAVE_SECONDS)
    work = asyncio.to_thread(
        _answer, texts, own, sql, params, call=call, take=take, seconds=seconds
    )
    try:
        return await asyncio.shield(asyncio.ensure_future(work))
    except asyncio.CancelledError:
        call.cancel()  # the thread stops at once, and closes its database
        raise


class _Call:
    """One query's database while it is open, the time it may run until, and whether its call
    was cancelled: a cancel, or its time running out, interrupts the database (never one
    closed, nor one being closed: ``lock``)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.db: Any = None
        self.cancelled = False
        self.ends = time.monotonic() + LOAD_SECONDS

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            if self.db is not None:
                self.db.interrupt()

    def overdue(self) -> None:
        """Interrupt the database when it is past :attr:`ends`."""
        with self.lock:
            if self.db is not None and time.monotonic() > self.ends:
                self.db.interrupt()

    @contextlib.contextmanager
    def watched(self) -> Iterator[None]:
        """While in it, a thread looks every :data:`INTERRUPT_SECONDS` whether the database is
        past its time, and interrupts it until it stops (an interrupt may come between two
        statements, before the next one starts)."""
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(INTERRUPT_SECONDS):
                self.overdue()

        watcher = threading.Thread(target=watch, name="query-time", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            stop.set()
            watcher.join()


def _answer(
    texts: Mapping[str, Source],
    own: str,
    sql: str,
    params: Sequence[Any],
    *,
    call: _Call,
    take: Callable[[Any, list[str]], Any] | None = None,
    seconds: float = SECONDS,
) -> Any:
    """The query over ``texts`` (:func:`query`) on :data:`ENGINE`, in this worker thread, once
    its turn comes (and its call was not cancelled meanwhile), within ``seconds``; what
    ``take`` makes of its rows (:func:`_first` by default) while its database is open, which
    ``call`` holds."""
    if not _RUNNING.acquire(timeout=WAIT_SECONDS):
        return f"Error: the query waited {WAIT_SECONDS:g} s for others to finish: ask it again"
    engine = ENGINE
    try:
        with engine.opened() as db:
            with call.lock:
                if call.cancelled:
                    return "Error: cancelled"
                call.db = db
                call.ends = time.monotonic() + LOAD_SECONDS
            try:
                with call.watched():
                    engine.load(db, texts, own)
                    if time.monotonic() > call.ends:
                        raise TimeoutError("loading the saved results took too long")
                    call.ends = time.monotonic() + seconds  # the query's own, from here
                    cursor = engine.execute(db, sql, params)
                    names = [d[0] for d in cursor.description or ()]
                    return (take or _first)(cursor, names)
            except export.TooManyRows as exc:
                return f"Error: the query has {exc}: aggregate, or narrow it with WHERE"
            except (*engine.errors, TimeoutError) as exc:
                said = "took too long" if time.monotonic() > call.ends else engine.said(exc)
                if "VARCHAR" in said and _TEXT_MISUSED.search(said):
                    said += TEXT_NUMBERS
                return (
                    f"Error: the query failed: {said}. Only a SELECT runs, over the table "
                    f"{TABLE} (this file) and the run's other saved results: "
                    f"{_there(texts, own, engine.asked(db, sql))}."
                )
            finally:
                with call.lock:
                    call.db = None  # no interrupt from here on: it closes
    finally:
        _RUNNING.release()


def _first(cursor: Any, names: list[str]) -> str:
    """A query's first :data:`ROWS` rows as CSV, at most :data:`OUTPUT_CHARS`, saying when
    there were more."""
    rows = cursor.fetchmany(ROWS + 1)
    shown = _csv(names, rows[:ROWS])
    if len(shown) > OUTPUT_CHARS:
        shown = shown[:OUTPUT_CHARS] + f"\n…[cut at {OUTPUT_CHARS} characters: aggregate more]"
    elif len(rows) > ROWS:
        shown += f"\n…[only the first {ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    return shown


#: How many of the run's tables a failed query is told, with their columns — the closest to
#: those it reads (:func:`_there`) — and the query that lists them all: its error stays short,
#: so no window cuts the table it meant out of the middle.
TOLD_TABLES: Final = 5
ALL_TABLES: Final = "SELECT table_name FROM information_schema.tables"


def _there(texts: Mapping[str, Source], own: str, asked: Sequence[str]) -> str:
    """The tables a failed query could read, each with its columns: the :data:`TOLD_TABLES`
    whose names are closest to those it ``asked`` for (``difflib``'s ratio, :data:`TABLE`
    being ``own``; equally close, as they were loaded), then how many more there are and the
    query that lists them."""
    names = [own.lower() if a.lower() == TABLE else a.lower() for a in asked]

    def near(table: str) -> float:
        ratios = (difflib.SequenceMatcher(None, table.lower(), a).ratio() for a in names)
        return max(ratios, default=0.0)

    closest = sorted(texts, key=near, reverse=True)[:TOLD_TABLES]
    said = [f"{t} ({', '.join(columns_of(texts[t]))})" for t in closest]
    if more := len(texts) - len(closest):
        said.append(f"and {more} more ({ALL_TABLES} lists them all)")
    return "; ".join(said)


#: How many rows a saved query's cursor gives at a time.
BATCH_ROWS: Final = 1_000


def _saved(kind: str) -> Callable[[Any, list[str]], tuple[bytes, int, str]]:
    """What a query saving its result as a ``kind`` file takes of its rows: every row written
    as they come (``tools.export.written``; read with ``fetchmany``), how
    many, and the first :data:`SAVED_ROWS` as CSV."""

    def take(cursor: Any, names: list[str]) -> tuple[bytes, int, str]:
        first: list[Any] = []

        def rows() -> Iterator[Any]:
            while batch := cursor.fetchmany(BATCH_ROWS):
                first.extend(batch[: SAVED_ROWS - len(first)])
                yield from batch

        data, count = export.written(kind, names, rows())
        return data, count, _csv(names, first)

    return take


def _csv(names: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Rows as an answer shows them: CSV under a header, each cell :func:`_shown`."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    writer.writerows([_shown(v) for v in row] for row in rows)
    return out.getvalue().rstrip("\n")


def _shown(value: Any) -> Any:
    """A cell of an answer as it is written: a boolean as SQL and JSON write it; a number with
    a fraction — a float, or a decimal — with 15 significant digits (``%!.15g``, a whole one
    with ``.0``), so the binary tail of a sum, a product or a ratio (``59.970000000000006``) is
    not read as a figure."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        value = float(value)
    if not isinstance(value, float):
        return value
    shown = format(value, ".15g")
    return f"{shown}.0" if shown.lstrip("-").isdigit() else shown


#: DuckDB's names for the documented functions it spells otherwise (see the module).
MACROS: Final = (
    "CREATE MACRO percentile_cont(x, p) AS quantile_cont(x, p)",
    # days since noon UTC on 24 November 4714 BC, from an ISO date or time (SQLite's meaning)
    "CREATE MACRO julianday(d) AS epoch(CAST(d AS TIMESTAMP)) / 86400 + 2440587.5",
)


class DuckDB:
    """The engine (:data:`ENGINE`): each saved result written to a private
    temporary file and read by DuckDB's own CSV and JSON readers, every column as text, then
    typed by the module's rules in one SQL pass (:meth:`_typed`); then locked (see the
    module). DuckDB itself is imported at the first query."""

    name = "DuckDB"

    @cached_property
    def duckdb(self) -> Any:
        return importlib.import_module("duckdb")

    @property
    def errors(self) -> tuple[type[Exception], ...]:
        return (self.duckdb.Error,)

    @contextlib.contextmanager
    def opened(self) -> Iterator[Any]:
        config = {
            "memory_limit": f"{DB_BYTES // 2**20}MiB",
            "temp_directory": "",  # never spilled to disk: the memory limit holds
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        }
        with contextlib.closing(self.duckdb.connect(":memory:", config=config)) as db:
            yield db

    def load(self, db: Any, texts: Mapping[str, Source], own: str) -> None:
        used = {TABLE, *texts}
        views: list[tuple[str, str]] = []  # the queried result's parts, as result_<part>
        with tempfile.TemporaryDirectory(prefix="trellis-query-") as folder:
            for n, (name, text) in enumerate(texts.items()):
                for key in self._tables(db, os.path.join(folder, str(n)), text):
                    table = base = name if key is None else part(name, key)
                    while key is not None and table in used:  # a name taken: the next one
                        table = f"{base}_{len(used)}"
                    used.add(table)
                    db.execute(f"ALTER TABLE {LOADED} RENAME TO {_name(table)}")
                    if key is not None and name == own:
                        views.append((part(TABLE, key), table))
        for macro in MACROS:
            db.execute(macro)
        if own in texts:  # else sql alone: figures only, no table is :data:`TABLE`
            db.execute(f'CREATE VIEW {TABLE} AS SELECT * FROM "{own}"')
        for view, table in views:
            if view not in used:
                db.execute(f"CREATE VIEW {_name(view)} AS SELECT * FROM {_name(table)}")
        db.execute("SET TimeZone = 'UTC'")  # a time with a zone is read the same on any host
        db.execute("SET enable_external_access = false")
        db.execute("SET lock_configuration = true")

    def _tables(self, db: Any, path: str, text: Source) -> Iterator[str | None]:
        """Each table ``text`` is (:meth:`_read`: its own, ``None``, then each part by its
        key), typed, as the table :data:`LOADED` — taken before the next: read from
        :data:`TYPED`'s Parquet of the same text when it has it, else read, typed
        (:meth:`_typed`) and kept there as Parquet (a text's; a workbook's are not)."""
        digest = TYPED.key(text) if isinstance(text, str) else None
        kept = TYPED.get(digest) if digest is not None else None
        if kept is not None:
            for n, (key, data) in enumerate(kept):
                file = f"{path}_{n}.parquet"
                with open(file, "wb") as out:
                    out.write(data)
                db.execute(f"CREATE TABLE {LOADED} AS SELECT * FROM read_parquet({_quoted(file)})")
                yield key
            return
        made: list[tuple[str | None, bytes]] = []
        for n, (key, columns) in enumerate(self._read(db, path, text)):
            self._typed(db, LOADED, columns)
            if digest is not None:
                file = f"{path}_{n}.parquet"
                db.execute(f"COPY {LOADED} TO {_quoted(file)} (FORMAT parquet, COMPRESSION zstd)")
                with open(file, "rb") as written:
                    made.append((key, written.read()))
            yield key
        if digest is not None:
            TYPED.put(digest, made)

    def _read(
        self, db: Any, path: str, text: str | Table | Sheets
    ) -> Iterator[tuple[str | None, list[str]]]:
        """``text`` as the table ``raw``, every column ``VARCHAR`` (an empty cell ``NULL``),
        read by DuckDB's own reader from ``path`` (written here: the text, or its lines as
        CSV); its columns (:func:`column_names`) — once for each table it is (:func:`parts_of`:
        its own, ``None``, then each part by its key), ``raw`` taken before the next. Its shape
        is :func:`_shape`'s; a table given as rows (a sheet, a markdown table) is written as
        CSV under its columns and read as one."""
        found: Any
        if isinstance(text, Mapping):  # a workbook: a table a sheet
            for n, (sheet, table) in enumerate(text.items()):
                for _, names in self._read(db, f"{path}_{n}", table):
                    yield (sheet if n else None), names
            return
        kind, found = _shape(text, values=False) if isinstance(text, str) else ("table", text)
        body = text if isinstance(text, str) else ""
        if kind == "table":
            written = io.StringIO()
            csv.writer(written, lineterminator="\n").writerows([found[0], *found[1]])
            body, kind, found = written.getvalue(), "csv", ","
        file = _quoted(path)
        with open(path, "w", encoding="utf-8", errors="replace", newline="") as out:
            if kind == "lines":
                csv.writer(out).writerows([["line"], *([line] for line in found)])
            else:
                out.write(body)
        if kind == "json":  # a string as its text, a number as written, a nested value as JSON
            for n, (keys, rows) in enumerate(found):
                fields = _keys(rows)
                names = column_names(fields)
                source, cells = _json_source(file, body, keys, fields)
                listed = ", ".join(
                    f"NULLIF({c}, '') AS {_name(n)}" for c, n in zip(cells, names, strict=True)
                )
                db.execute(f"CREATE TEMP TABLE raw AS SELECT {listed} FROM {source}")
                yield (keys[-1] if n else None), names
            return
        if kind == "csv":
            names = column_names(next(csv.reader(io.StringIO(body), delimiter=found)))
            spec = ", ".join(f"{_quoted(n)}: 'VARCHAR'" for n in names)
            # as Python's csv reads it: short rows padded, long ones cut, blank lines skipped
            source = (
                f"read_csv({file}, delim = {_quoted(found)}, header = true, quote = '\"', "
                f"escape = '\"', columns = {{{spec}}}, auto_detect = false, "
                f"null_padding = true, strict_mode = false, parallel = false, "
                f"max_line_size = {max(len(body), 2**21)})"
            )
        else:  # one row per line
            names = ["line"]
            source = (
                f"read_csv({file}, header = true, columns = {{'line': 'VARCHAR'}}, "
                f"auto_detect = false, max_line_size = {max(len(body), 2**21)})"
            )
        listed = ", ".join(f"NULLIF({_name(n)}, '') AS {_name(n)}" for n in names)
        db.execute(f"CREATE TEMP TABLE raw AS SELECT {listed} FROM {source}")
        yield None, names

    def _typed(self, db: Any, name: str, columns: Sequence[str]) -> None:
        """The table ``name`` from ``raw`` (dropped after), each column typed by the module's
        rules (:func:`_counted`), counted in one pass over every row — only the columns still
        numbers in its first :data:`SAMPLE_ROWS` — then cast."""
        kinds: dict[str, str | None] = dict.fromkeys(columns, "")
        for rows in (f"(SELECT * FROM raw LIMIT {SAMPLE_ROWS})", "raw"):
            asked = [c for c in columns if kinds[c] is not None]
            if not asked:
                break
            counts = db.execute(f"SELECT {', '.join(map(_counted, asked))} FROM {rows}").fetchone()
            for n, column in enumerate(asked):
                present, whole, decimal, number, digits, scale = counts[6 * n : 6 * n + 6]
                if number < present or (rows == "raw" and not present):
                    kinds[column] = None
                elif whole == present:
                    kinds[column] = "BIGINT"
                elif decimal == present:  # exact, as written: money adds up
                    kinds[column] = f"DECIMAL({max(digits + scale, 1)}, {scale})"
                else:
                    kinds[column] = "DOUBLE"
        cast = [
            f"CAST(CASE WHEN {_missing(c)} THEN NULL ELSE {_name(c)} END AS {kinds[c]}) "
            f"AS {_name(c)}"
            if kinds[c]
            else _name(c)
            for c in columns
        ]
        db.execute(f"CREATE TABLE {_name(name)} AS SELECT {', '.join(cast)} FROM raw")
        db.execute("DROP TABLE raw")

    def execute(self, db: Any, sql: str, params: Sequence[Any]) -> Any:
        duckdb = self.duckdb
        statements = duckdb.extract_statements(sql)
        if len(statements) == 1 and statements[0].type == duckdb.StatementType.MULTI:
            raise duckdb.PermissionException(PIVOT_VALUES)  # a PIVOT that creates a type first
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise duckdb.PermissionException("not authorized: only one SELECT runs")
        cursor = db.execute(sql, list(params))
        described = cursor.description or ()
        zoned = [str(d[1]) in ZONED for d in described]
        if not any(zoned):
            return cursor
        # a time with a zone as its text in UTC: Python reads it only with pytz
        shown = ", ".join(
            f"CAST(#{n} AS VARCHAR) AS {_name(d[0])}" if z else f"#{n} AS {_name(d[0])}"
            for n, (d, z) in enumerate(zip(described, zoned, strict=True), start=1)
        )
        return db.execute(f"SELECT {shown} FROM ({sql}\n)", list(params))

    def asked(self, db: Any, sql: str) -> list[str]:
        """The tables ``sql`` reads, as DuckDB parses it; none of a text it does not parse."""
        try:
            return sorted(db.get_table_names(sql))
        except self.duckdb.Error:
            return []

    def said(self, exc: Exception) -> str:
        # its pointer under the SQL, and the functions it lists, left out
        return re.split(r"\n\nLINE |\n\tCandidate functions", str(exc))[0]


#: What a ``PIVOT`` that lists no values is told: DuckDB creates a type for them first.
PIVOT_VALUES: Final = (
    "a PIVOT lists the values it makes columns of: PIVOT result ON p IN ('a', 'b') USING "
    "sum(v) GROUP BY r (or sum(v) FILTER (WHERE p = 'a') per column)"
)
#: The types a time with a zone has, written as text in an answer.
ZONED: Final = frozenset({"TIMESTAMP WITH TIME ZONE", "TIME WITH TIME ZONE"})


def _json_source(
    file: str, text: str, keys: Sequence[str], fields: Sequence[str]
) -> tuple[str, list[str]]:
    """Where DuckDB reads the rows of a JSON result's array ``keys`` deep (:func:`_json_rows`),
    each field as text, and each field's cell there."""
    largest = f"maximum_object_size = {len(text) + 1}"
    if not keys:  # an array of objects, or JSON Lines: one record each
        spec = ", ".join(f"{_quoted(k)}: 'VARCHAR'" for k in fields)
        form = "array" if text.lstrip().startswith("[") else "newline_delimited"
        source = (
            f"read_json({file}, format = '{form}', records = true, columns = {{{spec}}}, {largest})"
        )
        return source, [_name(k) for k in fields]
    held = f"STRUCT({', '.join(f'{_name(k)} VARCHAR' for k in fields)})[]"
    reach = _name(keys[0])
    for key in keys[1:]:
        reach = f"struct_extract({reach}, {_quoted(key)})"
    for key in reversed(keys[1:]):
        held = f"STRUCT({_name(key)} {held})"
    source = (
        f"(SELECT unnest({reach}) AS r FROM read_json({file}, records = true, "
        f"columns = {{{_quoted(keys[0])}: {_quoted(held)}}}, {largest}))"
    )
    return source, [f"struct_extract(r, {_quoted(k)})" for k in fields]


def _missing(column: str) -> str:
    """In SQL: whether a text cell says it holds no value (:data:`MISSING`)."""
    said = ", ".join(map(_quoted, MISSING))
    return f"upper(trim({_name(column)})) IN ({said})"


def _counted(column: str) -> str:
    """In SQL, over a text column: how many values it has (a missing one not counted:
    :func:`_missing`), how many are whole numbers, how
    many decimals (:data:`DECIMAL`, at most :data:`DECIMAL_DIGITS` digits), how many numbers
    (:data:`WHOLE` within 64 bits, or :data:`NUMBER` and finite; DuckDB's casts allow the
    spaces around one), and the decimals' most digits before the point and after it."""
    text = f"trim({_name(column)})"
    whole = f"regexp_full_match({text}, {_quoted(WHOLE)})"
    fits = f"TRY_CAST({text} AS BIGINT) IS NOT NULL"
    number = (
        f"regexp_full_match({text}, {_quoted(NUMBER)}) AND isfinite(TRY_CAST({text} AS DOUBLE))"
    )
    short = f"length({text}) <= {NUMBER_CHARS}"
    digits = f"length(ltrim(regexp_replace({text}, '[^0-9]', '', 'g'), '0'))"
    decimal = (
        f"{short} AND regexp_full_match({text}, {_quoted(DECIMAL)}) "
        f"AND {digits} <= {DECIMAL_DIGITS}"
    )
    before = f"length(ltrim(split_part(ltrim({text}, '+-'), '.', 1), '0'))"
    after = f"length(split_part({text}, '.', 2))"
    return (
        f"count(*) FILTER (WHERE NOT {_missing(column)}), "
        f"count(*) FILTER ({short} AND {whole} AND {fits}), "
        f"count(*) FILTER ({decimal}), "
        f"count(*) FILTER ({short} AND (CASE WHEN {whole} THEN {fits} ELSE {number} END)), "
        f"coalesce(max({before}) FILTER ({decimal}), 0), "
        f"coalesce(max({after}) FILTER ({decimal}), 0)"
    )


def _quoted(text: str) -> str:
    """``text`` as a SQL string."""
    return "'" + text.replace("'", "''") + "'"


def _name(text: str) -> str:
    """``text`` as a SQL name."""
    return '"' + text.replace('"', '""') + '"'


class Typed:
    """The typed tables of the texts this process loaded, as Parquet (zstd), by each text's
    content: a later query, turn or past-result load of the same text reads its tables from
    here instead of parsing and typing it again — the same tables, as typed then. The least
    recently used text's are forgotten past :data:`TYPED_BYTES`. Queries load in worker
    threads: one lock."""

    def __init__(self, most: int = TYPED_BYTES) -> None:
        self.most = most
        self.size = 0
        self._kept: OrderedDict[bytes, list[tuple[str | None, bytes]]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str) -> bytes:
        """A text's key: its content's hash."""
        return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()

    def get(self, key: bytes) -> list[tuple[str | None, bytes]] | None:
        """The tables kept for the text of ``key`` (each by its part's key), or ``None``."""
        with self._lock:
            found = self._kept.get(key)
            if found is not None:
                self._kept.move_to_end(key)
            return found

    def put(self, key: bytes, tables: list[tuple[str | None, bytes]]) -> None:
        """The tables of the text of ``key``, kept."""
        with self._lock:
            self.size -= sum(len(d) for _, d in self._kept.pop(key, ()))
            self._kept[key] = tables
            self.size += sum(len(d) for _, d in tables)
            while self.size > self.most:
                _, gone = self._kept.popitem(last=False)
                self.size -= sum(len(d) for _, d in gone)


#: What every query runs on, and the typed tables it keeps.
ENGINE = DuckDB()
TYPED = Typed()

#: ``query_result``'s description: it names the engine, whose SQL the model writes.
QUERY_RESULT_DESCRIPTION: Final = (
    f"Runs a read-only SQL query ({ENGINE.name}) over every row of a large tool result that "
    f"was saved to a file: the rows are the table {TABLE} (the tool message names its path; "
    f"SELECT * FROM {TABLE} LIMIT 3 shows its columns). "
    "Use it to count, sum, average, filter, rank (ORDER BY ... LIMIT n) or group the rows "
    "instead of reading them page by page: the answer is computed over all of them, exactly. "
    "With sql alone it computes with figures you already have, exactly (SELECT 1123571.70 / "
    "11978): divide, multiply, take a percentage or round with it, never by hand."
)


def _json_rows(
    text: str, *, values: bool = True
) -> list[tuple[tuple[str, ...], Sequence[Any]]] | None:
    """The rows of a JSON result, each array of objects with the keys that reach it: an array
    (``()``), JSON Lines (an object a line: ``()``), or the arrays of objects an object holds
    at the shallowest depth any is (breadth first, in order); ``None`` when it is no such
    JSON. Without ``values``, each object is only its keys (:class:`_Keys`, one for objects
    alike), for a reader that reads the values itself: a fraction of the memory of dicts."""
    alike: dict[tuple[str, ...], _Keys] = {}

    def keys(pairs: list[tuple[str, Any]]) -> _Keys:
        found = _Keys(k for k, _ in pairs)
        held = {k: v for k, v in pairs if isinstance(v, _Keys | list)}
        if held:  # where rows may be: never one shared
            found.held = held
            return found
        return alike.setdefault(found, found)

    hook = None if values else keys
    kind = dict if values else _Keys
    try:
        found = json.loads(text, object_pairs_hook=hook)
    except ValueError as exc:
        if not (str(exc).startswith("Extra data") and text.lstrip().startswith("{")):
            return None
        try:  # JSON Lines
            found = [
                json.loads(line, object_pairs_hook=hook)
                for line in text.splitlines()
                if line.strip()
            ]
        except ValueError:
            return None
    if _objects(found, kind):
        return [((), found)]
    level = [((), found)] if isinstance(found, kind) else []
    while level:
        inside = [((*at, k), v) for at, o in level for k, v in _items(o)]
        arrays = [(at, v) for at, v in inside if _objects(v, kind)]
        if arrays:
            return arrays
        level = [(at, v) for at, v in inside if isinstance(v, kind)]
    return None


class _Keys(tuple[str, ...]):
    """A JSON object read for its keys only (:func:`_json_rows` without values), and the
    objects and arrays it holds (``held``), where rows may be."""

    held: Mapping[str, Any]


def _items(value: Any) -> Iterable[tuple[str, Any]]:
    """What a JSON object holds: a dict's items, or a :class:`_Keys`' objects and arrays."""
    return value.items() if isinstance(value, dict) else getattr(value, "held", {}).items()


def _objects(value: Any, kind: type = dict) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, kind) for v in value)


def _letter(char: str) -> str:
    """``char`` as a name keeps it (a letter, a mark, a digit, ``_``), else NUL."""
    kind = unicodedata.category(char)
    return char if char == "_" or kind[0] in "LM" or kind == "Nd" else "\0"


_NAME: Final = re.compile("\0+")


def _padded(row: Sequence[str], width: int) -> list[str | None]:
    return [*row[:width], *([None] * (width - len(row)))]


def column_names(columns: Sequence[str]) -> list[str]:
    """Column names SQL takes as they are: letters (of any script, their marks too), digits and
    ``_``, each once; a run of anything else is one ``_``."""
    seen: dict[str, int] = {}
    out = []
    for n, column in enumerate(columns):
        name = _NAME.sub("_", "".join(map(_letter, str(column)))).strip("_") or f"c{n + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out
