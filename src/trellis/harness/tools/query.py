"""``query_result``: a read-only SQL query over every row of a large result the run keeps
(``tools.results``) — a count, a sum, a filter, a top-N or a group computed in process, exactly,
instead of the model paging thousands of rows into its context and adding them up itself.

No sandbox and no Code Mode needed. The engine is chosen by what is installed
(:data:`ENGINE`, `ADR 0007 <docs/adr/0007-query-engine.md>`_): DuckDB (:class:`DuckDB`, the
``query`` extra) when it can be imported, else Python's own SQLite (:class:`SQLite`, no
dependency), each in memory, made for one query and closed after it. Both give the model one
contract: the tool's description names the engine (the model writes its dialect), the tables,
their typing and the limits are the same.

The kept text becomes the table ``result``: a JSON array of objects (or the first such array a
JSON object holds), CSV with a header line (comma, tab, semicolon or pipe), else one row per
line (``line``). A column whose every value is a whole number is ``INTEGER`` (DuckDB:
``BIGINT``), a number ``REAL`` (``DOUBLE``), else ``TEXT`` (``VARCHAR``): a code with a
leading zero, or an id past 64 bits, keeps its exact text; a nested value is its JSON text
(``json_extract`` reads it). A number is written as SQL writes one (:data:`WHOLE`,
:data:`NUMBER`): never ``1_000``, ``0x10`` or ``٣``.

Only reading is allowed, and only the run's own results are ever loaded. SQLite: an authorizer
admits ``SELECT`` (``WITH`` included), reading the tables and calling functions — never a write,
``ATTACH`` (no other file, no network), a ``PRAGMA`` or loading an extension (off, as Python
leaves it). DuckDB: one ``SELECT`` statement only, and once its tables are loaded no file or
network access (``enable_external_access``), no extension loaded or installed, and its settings
locked. A query runs for at most :data:`SECONDS` (then it is interrupted, as a cancelled call
is), its text is at most :data:`SQL_CHARS`, what it loads at most :data:`LOAD_CHARS`, its
memory at most :data:`DB_BYTES` (SQLite: its tables' pages, a value :data:`VALUE_CHARS`, its
sorts and joins :data:`TEMP_CACHE_BYTES` in memory, more in SQLite's temporary files, gone
when it closes; DuckDB: ``memory_limit``, never spilled to disk), and at most :data:`ROWS` rows
and :data:`OUTPUT_CHARS` characters come back (said when there were more).

Formulas: arithmetic, ``CASE``, math (``sqrt``, ``power``, ``ln``, ``exp``…), dates
(``julianday``; SQLite's ``strftime``, ``date(…, '+7 days')``) and window functions
(``avg(x) OVER (ORDER BY d ROWS 6 PRECEDING)`` a moving average, ``lag`` a growth rate,
``rank``); and the statistics, named as PostgreSQL and DuckDB name them: ``stddev``/
``stddev_samp``, ``stddev_pop``, ``variance``/``var_samp``, ``var_pop``, ``median``,
``percentile_cont(x, p)`` (``p`` from 0 to 1, interpolated as DuckDB's ``quantile_cont``),
``corr(y, x)``, ``covar_samp(y, x)``, ``covar_pop(y, x)``, ``regr_slope(y, x)``,
``regr_intercept(y, x)``, ``regr_r2(y, x)`` — DuckDB's own, and on SQLite aggregates of this
module (:data:`AGGREGATES`, Python's ``statistics``), ``NULL`` where there are too few values.
On DuckDB, ``percentile_cont(x, p)`` and ``julianday(d)`` are macros (:data:`MACROS`), so the
spelling above works on both engines.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import importlib
import importlib.util
import io
import json
import math
import os
import re
import sqlite3
import statistics
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any, Final, Protocol

from trellis.harness.runtime import current
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
    },
    "required": ["file_path", "sql"],
    "additionalProperties": False,
}
#: The most rows a query answers with, its longest text, the seconds it may run, the memory
#: its database may take, and the delimiters a CSV result is read with.
ROWS: Final = 200
SQL_CHARS: Final = 5_000
SECONDS: Final = 5.0
#: What a query may load (the run's saved results, in characters), the memory its database may
#: take (SQLite: its tables' pages; DuckDB: all of it), the longest value a SQLite query may
#: build, the memory SQLite's sorts and joins may hold, the most characters it answers with,
#: how many queries run at once in a process (each in a worker thread) and how long one waits
#: its turn.
LOAD_CHARS: Final = 64 * 1024 * 1024
DB_BYTES: Final = 512 * 1024 * 1024
VALUE_CHARS: Final = LOAD_CHARS
TEMP_CACHE_BYTES: Final = 8 * 1024 * 1024
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
#: What the authorizer admits: a SELECT, reading, a function, a recursive WITH.
_ALLOWED: Final = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
_NAME: Final = re.compile(r"[^0-9A-Za-z_]+")
#: The typing rules (see the module), one text for Python's ``re`` and DuckDB's RE2: a whole
#: number as SQL writes one, a number, a leading zero before a digit (``00123``, ``-007``: a
#: code, not a number), and the spaces a number may have around it.
WHOLE: Final = r"[-+]?[0-9]+"
NUMBER: Final = r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?"
LEADING_ZERO: Final = r"^[-+]?0[0-9]"
SPACES: Final = " \t\n\r\x0b\x0c"
_WHOLE: Final = re.compile(WHOLE)
_NUMBER: Final = re.compile(NUMBER)
_LEADING_ZERO: Final = re.compile(LEADING_ZERO)
#: The longest text read as a number.
NUMBER_CHARS: Final = 32


def rows_of(text: str) -> tuple[list[str], list[list[Any]]]:
    """A kept result as a table: its columns and rows (see the module)."""
    kind, found = _shape(text)
    if kind == "json":
        rows = found[0]
        columns = _keys(rows)
        return columns, [[_cell(row.get(c)) for c in columns] for row in rows]
    if kind == "csv":
        read = list(csv.reader(io.StringIO(text), delimiter=found))
        width = len(read[0])
        return list(read[0]), [_padded(r, width) for r in read[1:] if r]
    return ["line"], [[line] for line in found]


def _shape(text: str) -> tuple[str, Any]:
    """What a kept result is, as a table (see the module): ``("json", (rows, key))`` (the key
    of the object that holds them, ``None`` for an array), ``("csv", delimiter)`` or
    ``("lines", lines)``."""
    found = _json_rows(text)
    if found is not None:
        return "json", found
    lines = text.splitlines()
    head = lines[0] if lines else ""
    delimiter = max(DELIMITERS, key=head.count)
    if head.count(delimiter) > 0 and len(lines) > 1:
        return "csv", delimiter
    return "lines", lines


def _keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(k for row in rows for k in row))


def columns_of(text: str) -> list[str]:
    """The columns a kept result's table has (as :func:`query` names them)."""
    return column_names(rows_of(text)[0])


def tables(runtime: Any) -> dict[str, str]:
    """Every result the run keeps, by its path, and the table it is in a query: the name of the
    tool that returned it (``results.keep``'s ``tool``, in this attempt), else ``result_<n>``
    in the order kept — each name once, never :data:`TABLE` (the queried file's own)."""
    kept = runtime.replay.journal.results
    tools = getattr(runtime, "kept_from", None) or {}
    used = {TABLE}
    found: dict[str, str] = {}
    for n, path in enumerate(kept, start=1):
        base = column_names([tools.get(path) or f"{TABLE}_{n}"])[0]
        name, more = base, 1
        while name in used:
            more += 1
            name = f"{base}_{more}"
        used.add(name)
        found[path] = name
    return found


async def query(args: dict[str, Any], text: str | None = None) -> str:
    """``query_result``: the query's rows as CSV (a header line first), at most :data:`ROWS`
    and :data:`OUTPUT_CHARS`; or what is wrong — the arguments, the path, the SQL (the engine's
    own words, and the tables there are). The file asked about is the table :data:`TABLE`;
    every other result the run keeps is a table too (:func:`tables`), so one query joins and
    groups across them — only the run's own: nothing else is ever loaded.

    The query runs in a worker thread (the event loop never waits on it), at most
    :data:`CONCURRENT` at once in the process, on a database made for it and closed after it
    whatever happens; a cancelled call (the run cancelled, or out of time) interrupts it."""
    problem = arguments_problem(QUERY_RESULT_SCHEMA, args)
    if problem is not None:
        return not_run(QUERY_RESULT, problem)
    sql = str(args["sql"]).strip().rstrip(";")
    if len(sql) > SQL_CHARS:
        return f"Error: the query is longer than {SQL_CHARS} characters"
    return await run(str(args["file_path"]), sql, (), text)


def loadable(file_path: str, text: str | None = None) -> tuple[dict[str, str], str] | str:
    """What a query of ``file_path`` loads — the run's saved results by table name (the file
    asked about among them, given as ``text`` when it is no result the run keeps: a file Deep
    Agents saved) — and the file's own table; or why there is nothing to load."""
    path = "/" + file_path.lstrip("/")
    runtime = current()
    kept = runtime.replay.journal.results if runtime is not None else {}
    named = tables(runtime) if runtime is not None else {}
    if text is None:
        text = kept.get(path)
    if text is None or runtime is None:
        return f"Error: File '{path}' not found"
    texts = {named[p]: kept[p] for p in named}
    if path not in named:
        texts[f"{TABLE}_file"] = text  # a file Deep Agents saved: queried as ``result``
    if sum(map(len, texts.values())) > LOAD_CHARS:
        return (
            f"Error: this run's saved results are more than {LOAD_CHARS} characters, more than a "
            "query here loads: filter on the tool's side, or compute in a sandbox"
        )
    return texts, named.get(path, f"{TABLE}_file")


async def run(file_path: str, sql: str, params: Sequence[Any] = (), text: str | None = None) -> str:
    """``sql`` (with ``params``) over the run's saved results (:func:`loadable`), in a worker
    thread (see :func:`query`)."""
    found = loadable(file_path, text)
    if isinstance(found, str):
        return found
    texts, own = found
    call = _Call()
    work = asyncio.to_thread(_answer, texts, own, sql, params, call=call)
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
    texts: Mapping[str, str],
    own: str,
    sql: str,
    params: Sequence[Any],
    *,
    call: _Call,
) -> str:
    """The query over ``texts`` (:func:`query`) on :data:`ENGINE`, in this worker thread, once
    its turn comes (and its call was not cancelled meanwhile); ``call`` holds its database
    while it is open."""
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
                    call.ends = time.monotonic() + SECONDS  # the query's own, from here
                    cursor = engine.execute(db, sql, params)
                    rows = cursor.fetchmany(ROWS + 1)
                    names = [d[0] for d in cursor.description or ()]
            except (*engine.errors, TimeoutError) as exc:
                said = "took too long" if time.monotonic() > call.ends else engine.said(exc)
                there = "; ".join(f"{t} ({', '.join(columns_of(x))})" for t, x in texts.items())
                return (
                    f"Error: the query failed: {said}. Only a SELECT runs, over the table "
                    f"{TABLE} (this file) and the run's other saved results: {there}."
                )
            finally:
                with call.lock:
                    call.db = None  # no interrupt from here on: it closes
    finally:
        _RUNNING.release()
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    writer.writerows([_shown(v) for v in row] for row in rows[:ROWS])
    shown = out.getvalue().rstrip("\n")
    if len(shown) > OUTPUT_CHARS:
        shown = shown[:OUTPUT_CHARS] + f"\n…[cut at {OUTPUT_CHARS} characters: aggregate more]"
    elif len(rows) > ROWS:
        shown += f"\n…[only the first {ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    return shown


def _shown(value: Any) -> Any:
    """A cell of an answer as it is written: a boolean as SQL and JSON write it (DuckDB has
    them; SQLite's are 0 and 1)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


class Engine(Protocol):
    """What runs a query (:data:`ENGINE`): a database made for one call, its tables loaded from
    the run's saved results and read-only from then on, one ``SELECT`` over them."""

    #: As the tool's description names it, so the model writes its dialect.
    name: str

    @property
    def errors(self) -> tuple[type[Exception], ...]:
        """What a refused or failed query raises (an interrupted one too)."""
        ...

    def opened(self) -> contextlib.AbstractContextManager[Any]:
        """A database for one call, closed at the end; its ``interrupt()`` stops what it runs."""
        ...

    def load(self, db: Any, texts: Mapping[str, str], own: str) -> None:
        """Each of ``texts`` as a table (by its name), the view :data:`TABLE` on ``own``, then
        nothing but reading."""
        ...

    def execute(self, db: Any, sql: str, params: Sequence[Any]) -> Any:
        """``sql`` run: a cursor (``fetchmany``, ``description``); only one ``SELECT``."""
        ...

    def said(self, exc: Exception) -> str:
        """What a failed query says to the model."""
        ...


class SQLite:
    """The fallback, with no dependency: Python's own ``sqlite3``, the cells typed and loaded
    in Python, the statistics it lacks as Python aggregates (:data:`AGGREGATES`), an
    authorizer admitting only reading."""

    name = "SQLite"
    errors: tuple[type[Exception], ...] = (sqlite3.Error,)

    @contextlib.contextmanager
    def opened(self) -> Iterator[sqlite3.Connection]:
        with contextlib.closing(sqlite3.connect(":memory:", check_same_thread=False)) as db:
            yield db

    def load(self, db: sqlite3.Connection, texts: Mapping[str, str], own: str) -> None:
        """Its tables' pages at most :data:`DB_BYTES`, a value :data:`VALUE_CHARS`, its sorts'
        and joins' memory :data:`TEMP_CACHE_BYTES`."""
        for name, arguments, aggregate in AGGREGATES:
            db.create_aggregate(name, arguments, aggregate)
        size = int(db.execute("PRAGMA page_size").fetchone()[0])
        db.execute(f"PRAGMA main.max_page_count = {DB_BYTES // size}")
        db.execute(f"PRAGMA temp.cache_size = -{TEMP_CACHE_BYTES // 1024}")
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, VALUE_CHARS)
        for name, text in texts.items():
            _table(db, name, text)
        db.execute(f'CREATE VIEW {TABLE} AS SELECT * FROM "{own}"')
        db.commit()
        db.set_authorizer(lambda action, *_: 0 if action in _ALLOWED else 1)  # 1: SQLITE_DENY

    def execute(self, db: sqlite3.Connection, sql: str, params: Sequence[Any]) -> sqlite3.Cursor:
        return db.execute(sql, tuple(params))

    def said(self, exc: Exception) -> str:
        return str(exc)


#: DuckDB's names for the documented functions it spells otherwise (see the module).
MACROS: Final = (
    "CREATE MACRO percentile_cont(x, p) AS quantile_cont(x, p)",
    # SQLite's: days since noon UTC on 24 November 4714 BC, from an ISO date or time
    "CREATE MACRO julianday(d) AS epoch(CAST(d AS TIMESTAMP)) / 86400 + 2440587.5",
)


class DuckDB:
    """The engine when the ``query`` extra is installed: each saved result written to a private
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

    def load(self, db: Any, texts: Mapping[str, str], own: str) -> None:
        with tempfile.TemporaryDirectory(prefix="trellis-query-") as folder:
            for n, (name, text) in enumerate(texts.items()):
                self._typed(db, name, self._read(db, os.path.join(folder, str(n)), text))
        for macro in MACROS:
            db.execute(macro)
        db.execute(f'CREATE VIEW {TABLE} AS SELECT * FROM "{own}"')
        db.execute("SET enable_external_access = false")
        db.execute("SET lock_configuration = true")

    def _read(self, db: Any, path: str, text: str) -> list[str]:
        """``text`` as the table ``raw``, every column ``VARCHAR`` (an empty cell ``NULL``),
        read by DuckDB's own reader from ``path`` (written here: the text, or its lines as
        CSV); its columns (:func:`column_names`). Its shape is :func:`_shape`'s."""
        kind, found = _shape(text)
        file = _quoted(path)
        with open(path, "w", encoding="utf-8", errors="replace", newline="") as out:
            if kind == "lines":
                csv.writer(out).writerows([["line"], *([line] for line in found)])
            else:
                out.write(text)
        if kind == "json":
            rows, key = found
            keys = _keys(rows)
            names = column_names(keys)
            if key is None:  # an array of objects: one record each
                spec = ", ".join(f"{_quoted(k)}: 'JSON'" for k in keys)
                source = (
                    f"read_json({file}, format = 'array', records = true, columns = {{{spec}}}, "
                    f"maximum_object_size = {len(text) + 1})"
                )
                cells = [_name(k) for k in keys]
            else:  # an object holding the array: its rows unnested
                source = (
                    f"(SELECT unnest(CAST({_name(key)} AS JSON[])) AS r FROM read_json({file}, "
                    f"records = true, columns = {{{_quoted(key)}: 'JSON'}}, "
                    f"maximum_object_size = {len(text) + 1}))"
                )
                cells = [f"json_extract(r, {_quoted('/' + _pointer(k))})" for k in keys]
            # a string as its text, a number as written, a nested value as its JSON text
            listed = ", ".join(
                f"NULLIF(json_extract_string({c}, '$'), '') AS {_name(n)}"
                for c, n in zip(cells, names, strict=True)
            )
            db.execute(f"CREATE TEMP TABLE raw AS SELECT {listed} FROM {source}")
            return names
        if kind == "csv":
            names = column_names(next(csv.reader(io.StringIO(text), delimiter=found)))
            spec = ", ".join(f"{_quoted(n)}: 'VARCHAR'" for n in names)
            # as Python's csv reads it: short rows padded, long ones cut, blank lines skipped
            source = (
                f"read_csv({file}, delim = {_quoted(found)}, header = true, quote = '\"', "
                f"escape = '\"', columns = {{{spec}}}, auto_detect = false, "
                f"null_padding = true, strict_mode = false, parallel = false, "
                f"max_line_size = {max(len(text), 2**21)})"
            )
        else:  # one row per line
            names = ["line"]
            source = (
                f"read_csv({file}, header = true, columns = {{'line': 'VARCHAR'}}, "
                f"auto_detect = false, max_line_size = {max(len(text), 2**21)})"
            )
        listed = ", ".join(f"NULLIF({_name(n)}, '') AS {_name(n)}" for n in names)
        db.execute(f"CREATE TEMP TABLE raw AS SELECT {listed} FROM {source}")
        return names

    def _typed(self, db: Any, name: str, columns: Sequence[str]) -> None:
        """The table ``name`` from ``raw`` (dropped after), each column typed by the module's
        rules (see :func:`_kind`): counted in one pass, then cast."""
        tests = []
        for column in columns:
            text = f"trim({_name(column)}, {_quoted(SPACES)})"
            plain = (
                f"length({text}) <= {NUMBER_CHARS} "
                f"AND NOT regexp_matches({text}, {_quoted(LEADING_ZERO)})"
            )
            whole = f"regexp_full_match({text}, {_quoted(WHOLE)})"
            fits = f"TRY_CAST({text} AS BIGINT) IS NOT NULL"
            number = (
                f"regexp_full_match({text}, {_quoted(NUMBER)}) "
                f"AND isfinite(TRY_CAST({text} AS DOUBLE))"
            )
            tests += [
                f"count({_name(column)})",
                f"count(*) FILTER ({plain} AND {whole} AND {fits})",
                f"count(*) FILTER ({plain} AND (CASE WHEN {whole} THEN {fits} ELSE {number} END))",
            ]
        counts = db.execute(f"SELECT {', '.join(tests)} FROM raw").fetchone()
        cast = []
        for n, column in enumerate(columns):
            present, whole, number = counts[3 * n : 3 * n + 3]
            kind = "BIGINT" if present and whole == present else None
            kind = kind or ("DOUBLE" if present and number == present else None)
            cell = _name(column)
            if kind is not None:
                cell = f"CAST(trim({cell}, {_quoted(SPACES)}) AS {kind})"
            cast.append(f"{cell} AS {_name(column)}")
        db.execute(f"CREATE TABLE {_name(name)} AS SELECT {', '.join(cast)} FROM raw")
        db.execute("DROP TABLE raw")

    def execute(self, db: Any, sql: str, params: Sequence[Any]) -> Any:
        duckdb = self.duckdb
        statements = duckdb.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise duckdb.PermissionException("not authorized: only one SELECT runs")
        return db.execute(sql, list(params))

    def said(self, exc: Exception) -> str:
        return str(exc).split("\n\nLINE ")[0]  # its pointer under the SQL left out


def _quoted(text: str) -> str:
    """``text`` as a SQL string."""
    return "'" + text.replace("'", "''") + "'"


def _name(text: str) -> str:
    """``text`` as a SQL name."""
    return '"' + text.replace('"', '""') + '"'


def _pointer(key: str) -> str:
    """A JSON object's key as a step of a JSON Pointer (RFC 6901)."""
    return key.replace("~", "~0").replace("/", "~1")


def chosen() -> Engine:
    """The engine this process has: DuckDB when it can be imported (the ``query`` extra),
    else SQLite."""
    return DuckDB() if importlib.util.find_spec("duckdb") is not None else SQLite()


#: What every query runs on (:func:`chosen`).
ENGINE: Engine = chosen()


def description(engine: Engine) -> str:
    """``query_result``'s description on ``engine``: it names the engine, whose SQL the model
    writes."""
    return (
        f"Runs a read-only SQL query ({engine.name}) over every row of a large tool result that "
        f"was saved to a file: the rows are the table {TABLE} (the tool message names its path; "
        f"SELECT * FROM {TABLE} LIMIT 3 shows its columns). "
        "Use it to count, sum, average, filter, rank (ORDER BY ... LIMIT n) or group the rows "
        "instead of reading them page by page: the answer is computed over all of them, exactly."
    )


QUERY_RESULT_DESCRIPTION: Final = description(ENGINE)


class _Values:
    """An aggregate over one column's numbers (``NULL`` and text left out)."""

    def __init__(self) -> None:
        self.xs: list[float] = []

    def step(self, x: Any) -> None:
        if isinstance(x, int | float):
            self.xs.append(float(x))


class _Pairs:
    """An aggregate over pairs ``(y, x)`` of numbers (a pair with a ``NULL`` left out)."""

    def __init__(self) -> None:
        self.ys: list[float] = []
        self.xs: list[float] = []

    def step(self, y: Any, x: Any) -> None:
        if isinstance(y, int | float) and isinstance(x, int | float):
            self.ys.append(float(y))
            self.xs.append(float(x))


def _of(fn: Callable[[list[float]], float], least: int) -> type:
    """An aggregate of one column: ``fn`` of its numbers, ``NULL`` with fewer than ``least``."""

    class Aggregate(_Values):
        def finalize(self) -> float | None:
            return fn(self.xs) if len(self.xs) >= least else None

    return Aggregate


def _paired(fn: Callable[[list[float], list[float]], float]) -> type:
    """An aggregate of ``(y, x)`` pairs: ``fn(ys, xs)``, ``NULL`` with fewer than two or when
    it is not defined (no spread)."""

    class Aggregate(_Pairs):
        def finalize(self) -> float | None:
            if len(self.xs) < 2:
                return None
            try:
                return fn(self.ys, self.xs)
            except (statistics.StatisticsError, ZeroDivisionError):
                return None

    return Aggregate


class _Percentile:
    """``percentile_cont(x, p)``: the ``p`` quantile of ``x``, linearly interpolated."""

    def __init__(self) -> None:
        self.xs: list[float] = []
        self.p: float | None = None

    def step(self, x: Any, p: Any) -> None:
        if isinstance(p, int | float) and 0 <= p <= 1:
            self.p = float(p)
        if isinstance(x, int | float):
            self.xs.append(float(x))

    def finalize(self) -> float | None:
        if not self.xs or self.p is None:
            return None
        xs = sorted(self.xs)
        at = self.p * (len(xs) - 1)
        low = math.floor(at)
        high = min(low + 1, len(xs) - 1)
        return xs[low] + (xs[high] - xs[low]) * (at - low)


def _regression(ys: list[float], xs: list[float]) -> statistics.LinearRegression:
    return statistics.linear_regression(xs, ys)


def _r2(ys: list[float], xs: list[float]) -> float:
    return statistics.correlation(xs, ys) ** 2


def _covar_pop(ys: list[float], xs: list[float]) -> float:
    return statistics.covariance(xs, ys) * (len(xs) - 1) / len(xs)


#: The statistics SQLite lacks, as aggregates (name, arguments, class).
AGGREGATES: Final[tuple[tuple[str, int, type], ...]] = (
    ("stddev", 1, _of(statistics.stdev, 2)),
    ("stddev_samp", 1, _of(statistics.stdev, 2)),
    ("stddev_pop", 1, _of(statistics.pstdev, 1)),
    ("variance", 1, _of(statistics.variance, 2)),
    ("var_samp", 1, _of(statistics.variance, 2)),
    ("var_pop", 1, _of(statistics.pvariance, 1)),
    ("median", 1, _of(statistics.median, 1)),
    ("percentile_cont", 2, _Percentile),
    ("corr", 2, _paired(lambda ys, xs: statistics.correlation(xs, ys))),
    ("covar_samp", 2, _paired(lambda ys, xs: statistics.covariance(xs, ys))),
    ("covar_pop", 2, _paired(_covar_pop)),
    ("regr_slope", 2, _paired(lambda ys, xs: _regression(ys, xs).slope)),
    ("regr_intercept", 2, _paired(lambda ys, xs: _regression(ys, xs).intercept)),
    ("regr_r2", 2, _paired(_r2)),
)


def _table(db: sqlite3.Connection, name: str, text: str) -> None:
    """``text`` as the table ``name``: typed columns (see the module), its rows."""
    columns, rows = rows_of(text)
    names = column_names(columns)
    kinds = [_kind([r[n] for r in rows]) for n in range(len(names))]
    typed = ", ".join(f'"{n}" {k}' for n, k in zip(names, kinds, strict=True))
    db.execute(f'CREATE TABLE "{name}" ({typed})')
    marks = ", ".join("?" for _ in names)
    db.executemany(f'INSERT INTO "{name}" VALUES ({marks})', [_typed(r, kinds) for r in rows])


def _json_rows(text: str) -> tuple[list[Mapping[str, Any]], str | None] | None:
    """The rows of a JSON result: an array of objects, or the first such array an object
    holds (``{"items": [...]}``), with that key (``None`` for an array); ``None`` when it is
    no such JSON."""
    try:
        found = json.loads(text)
    except ValueError:
        return None
    if isinstance(found, dict):
        key = next((k for k, v in found.items() if _objects(v)), None)
        return None if key is None else (found[key], key)
    return (found, None) if _objects(found) else None


def _objects(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _cell(value: Any) -> Any:
    """A JSON value as a cell: a number or text as it is (a whole number past SQLite's 64 bits
    as its digits), anything else its JSON text, compact (as DuckDB writes it)."""
    if isinstance(value, int) and not isinstance(value, bool) and not _number(value, whole=True):
        return str(value)
    if value is None or (isinstance(value, str | int | float) and not isinstance(value, bool)):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _padded(row: Sequence[str], width: int) -> list[str | None]:
    return [*row[:width], *([None] * (width - len(row)))]


def column_names(columns: Sequence[str]) -> list[str]:
    """Column names SQL takes as they are: letters, digits and ``_``, each once."""
    seen: dict[str, int] = {}
    out = []
    for n, column in enumerate(columns):
        name = _NAME.sub("_", str(column)).strip("_") or f"c{n + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _kind(values: Sequence[Any]) -> str:
    """``INTEGER`` when every value is a whole number, ``REAL`` a number, else ``TEXT``."""
    present = [v for v in values if v not in (None, "")]
    if present and all(_number(v, whole=True) for v in present):
        return "INTEGER"
    if present and all(_number(v, whole=False) for v in present):
        return "REAL"
    return "TEXT"


def _number(value: Any, *, whole: bool) -> bool:
    """Whether a cell (text, or a JSON number: a JSON true is its text) is a number the
    engine keeps exactly: a whole one within 64 bits, a finite one — never text a number would
    change (a code with a leading zero, ``00123``; a run of digits too long for one; an id
    past 64 bits, which a ``REAL`` would round), nor one SQL would not read as a number
    (:data:`WHOLE`, :data:`NUMBER`). DuckDB applies the same rules in SQL
    (``DuckDB._typed``)."""
    if isinstance(value, int):
        return -(2**63) <= value < 2**63
    if isinstance(value, float):
        return not whole and math.isfinite(value)
    text = str(value).strip(SPACES)
    if _LEADING_ZERO.match(text) or len(text) > NUMBER_CHARS:
        return False
    # a whole number first: one past 64 bits stays its text, never a rounded REAL
    if _WHOLE.fullmatch(text):
        return _number(int(text), whole=whole)
    return not whole and bool(_NUMBER.fullmatch(text)) and math.isfinite(float(text))


def _typed(row: Sequence[Any], kinds: Sequence[str]) -> list[Any]:
    """A row's cells as their columns' kinds (an empty cell: ``NULL``)."""
    out: list[Any] = []
    for value, kind in zip(row, kinds, strict=True):
        if value in (None, ""):
            out.append(None)
        elif kind == "INTEGER":
            out.append(int(str(value).strip(SPACES)))
        elif kind == "REAL":
            out.append(float(str(value).strip(SPACES)))
        else:
            out.append(value)
    return out
