"""``query_result``: a read-only SQL query over every row of a large result the run keeps
(``tools.results``) — a count, a sum, a filter, a top-N or a group computed in process, exactly,
instead of the model paging thousands of rows into its context and adding them up itself.

No sandbox and no Code Mode needed, and no dependency: Python's own SQLite (``sqlite3``), in
memory. The kept text becomes the table ``result``: a JSON array of objects (or the first such
array a JSON object holds), CSV with a header line (comma, tab, semicolon or pipe), else one
row per line (``line``). A column whose every value is a whole number is ``INTEGER``, a number
``REAL``, else ``TEXT``; a nested value is its JSON text (``json_extract`` reads it).

Only reading is allowed: an authorizer admits ``SELECT`` (``WITH`` included), reading the table
and calling functions — never a write, ``ATTACH`` (no other file, no network), a ``PRAGMA`` or
loading an extension (off, as Python leaves it). A query runs for at most :data:`SECONDS`, its
text is at most :data:`SQL_CHARS`, what it loads at most :data:`LOAD_CHARS` (its tables'
pages at most :data:`DB_BYTES`), its sorts and joins hold at most :data:`TEMP_CACHE_BYTES` in
memory (more goes to SQLite's temporary files, gone when it closes), and at most :data:`ROWS`
rows and :data:`OUTPUT_CHARS` characters come back (said when there were more).

Formulas: SQLite's arithmetic, ``CASE``, math (``sqrt``, ``power``, ``ln``, ``exp``…), dates
(``julianday``, ``strftime``, ``date(…, '+7 days')``) and window functions (``avg(x) OVER
(ORDER BY d ROWS 6 PRECEDING)`` a moving average, ``lag`` a growth rate, ``rank``); and the
statistics it lacks, as aggregates of this module (:data:`AGGREGATES`, Python's
``statistics``): ``stddev``/``stddev_samp``, ``stddev_pop``, ``variance``/``var_samp``,
``var_pop``, ``median``, ``percentile_cont(x, p)`` (``p`` from 0 to 1, interpolated as
DuckDB's ``quantile_cont``), ``corr(y, x)``, ``covar_samp(y, x)``, ``covar_pop(y, x)``,
``regr_slope(y, x)``, ``regr_intercept(y, x)``, ``regr_r2(y, x)`` — named as PostgreSQL and
DuckDB name them, ``NULL`` where there are too few values.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import math
import re
import sqlite3
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from trellis.contracts import ToolError
from trellis.harness.files import file_name
from trellis.harness.runtime import current
from trellis.harness.tools import export
from trellis.harness.tools.base import arguments_problem, not_run

#: The tool, and what it is told and given.
QUERY_RESULT: Final = "query_result"
TABLE: Final = "result"
QUERY_RESULT_DESCRIPTION: Final = (
    "Runs a read-only SQL query (SQLite) over every row of a large tool result that was saved to "
    f"a file: the rows are the table {TABLE} (the tool message names its path; SELECT * FROM "
    f"{TABLE} LIMIT 3 shows its columns). "
    "Use it to count, sum, average, filter, rank (ORDER BY ... LIMIT n) or group the rows "
    "instead of reading them page by page: the answer is computed over all of them, exactly."
)
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
#: The rows a saved file's answer shows (the file holds them all).
SAVED_ROWS: Final = 5
#: The seconds a query saving its whole result may run, writing included.
SAVE_SECONDS: Final = 60.0
#: The most rows a query answers with, its longest text, the seconds it may run, the memory
#: its database may take, and the delimiters a CSV result is read with.
ROWS: Final = 200
SQL_CHARS: Final = 5_000
SECONDS: Final = 5.0
#: What a query may load (the run's saved results, in characters), the pages its tables may
#: take, the memory its sorts and joins may hold, the most characters it answers with, how many
#: queries run at once in a process (each in a worker thread) and how long one waits its turn.
LOAD_CHARS: Final = 64 * 1024 * 1024
DB_BYTES: Final = 512 * 1024 * 1024
TEMP_CACHE_BYTES: Final = 8 * 1024 * 1024
OUTPUT_CHARS: Final = 20_000
CONCURRENT: Final = 4
WAIT_SECONDS: Final = 60.0
#: The most a query's tables may take to load (the run's saved results parsed), beside its
#: own :data:`SECONDS`.
LOAD_SECONDS: Final = 60.0
_RUNNING: Final = threading.BoundedSemaphore(CONCURRENT)
DELIMITERS: Final = ",\t;|"
#: What the authorizer admits: a SELECT, reading, a function, a recursive WITH.
_ALLOWED: Final = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
_NAME: Final = re.compile(r"[^0-9A-Za-z_]+")
#: Text that is a code, not a number: a leading zero before a digit (``00123``, ``-007``).
_LEADING_ZERO: Final = re.compile(r"^[-+]?0\d")
#: The longest text read as a number.
NUMBER_CHARS: Final = 32


def rows_of(text: str) -> tuple[list[str], list[list[Any]]]:
    """A kept result as a table: its columns and rows (see the module)."""
    found = _json_rows(text)
    if found is not None:
        columns = list(dict.fromkeys(k for row in found for k in row))
        return columns, [[_cell(row.get(c)) for c in columns] for row in found]
    lines = text.splitlines()
    head = lines[0] if lines else ""
    delimiter = max(DELIMITERS, key=head.count)
    if head.count(delimiter) > 0 and len(lines) > 1:
        read = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        width = len(read[0])
        return list(read[0]), [_padded(r, width) for r in read[1:] if r]
    return ["line"], [[line] for line in lines]


#: A table given as rows (a kept Excel file's: ``tools.stored``): its columns and its rows.
Table = tuple[list[str], list[list[Any]]]


def columns_of(text: str | Table) -> list[str]:
    """The columns a kept result's table has (as :func:`query` names them)."""
    return column_names(_rows(text)[0])


def _rows(text: str | Table) -> Table:
    return rows_of(text) if isinstance(text, str) else text


def _size(text: str | Table) -> int:
    """What a table takes to load, in characters (a table given as rows: its cells' text)."""
    if isinstance(text, str):
        return len(text)
    return sum(len(str(v)) for row in text[1] for v in row)


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


def saving(runtime: Any) -> bool:
    """Whether ``query_result`` saves a query's whole result as a file in ``runtime``'s run
    (``save_as``): a run with no tool that runs code (``tools.export.runs_code``) — a run with
    one writes its files with its code."""
    return runtime is not None and export.runs_code(runtime) is None


def schema(runtime: Any) -> dict[str, Any]:
    """``query_result``'s input schema in ``runtime``'s run (:func:`saving`)."""
    return QUERY_RESULT_SAVING_SCHEMA if saving(runtime) else QUERY_RESULT_SCHEMA


async def query(args: dict[str, Any], text: str | None = None) -> str:
    """``query_result``: the query's rows as CSV (a header line first), at most :data:`ROWS`
    and :data:`OUTPUT_CHARS`; or what is wrong — the arguments, the path, the SQL (SQLite's own
    words, and the tables there are). The file asked about is the table :data:`TABLE`; every
    other result the run keeps is a table too (:func:`tables`), so one query joins and groups
    across them — only the run's own: nothing else is ever loaded. With ``save_as`` (a run that
    is :func:`saving`), the query's every row is also kept as a file for the user
    (``Runtime.save_file``), and the answer says so, with its first :data:`SAVED_ROWS` rows.

    The query runs in a worker thread (the event loop never waits on it), at most
    :data:`CONCURRENT` at once in the process, on a database made for it and closed after it
    whatever happens; a cancelled call (the run cancelled, or out of time) interrupts it."""
    runtime = current()
    problem = arguments_problem(schema(runtime), args)
    if problem is not None:
        return not_run(QUERY_RESULT, problem)
    sql = str(args["sql"]).strip().rstrip(";")
    if len(sql) > SQL_CHARS:
        return f"Error: the query is longer than {SQL_CHARS} characters"
    path = "/" + str(args["file_path"]).lstrip("/")
    record: dict[str, Any] | None = None
    given: str | Table | None = text
    if text is None and _stored(path):
        found = await _kept_file(path)
        if isinstance(found, str):
            return found
        given, record = found
    if args.get("save_as") is not None:
        return await _save(runtime, args, sql, given, record)
    answer = await run(path, sql, (), given)
    if record is None or answer.startswith("Error"):
        return answer
    return f"{_told(runtime, path, record, sql)}\n{answer}"


def _stored(path: str) -> bool:
    from trellis.harness.tools import stored  # noqa: PLC0415 - it reads the results' pages

    return stored.is_stored(path)


async def _kept_file(path: str) -> tuple[str | Table, dict[str, Any]] | str:
    """A file kept for the user (``tools.stored``), loaded from the store: its table and its
    record, or what is wrong."""
    from trellis.harness.tools import stored  # noqa: PLC0415

    return await stored.table(path)


def _told(runtime: Any, path: str, record: dict[str, Any], sql: str) -> str:
    """Where a query over a kept file read its rows: the file — when it was saved, by what,
    from what fetched when (its record) — and each of this run's results it names, fetched
    when: what a comparison of the two cites."""
    from trellis.harness.tools import stored  # noqa: PLC0415

    said = [f"{TABLE} is {path} ({stored.made(record)})"]
    said.extend(
        f"{s['table']} is this run's {s['tool']} result, fetched {stored.when(s['fetched_at'])}"
        for s in _sources(runtime, path, sql)
        if s["table"] != TABLE
    )
    return "; ".join(said) + "."


async def _save(
    runtime: Any,
    args: dict[str, Any],
    sql: str,
    text: str | Table | None,
    record: dict[str, Any] | None = None,
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
    found = await run(str(args["file_path"]), sql, (), text, save=kind)
    if isinstance(found, str):
        return found
    data, count, shown = found
    made = {"sql": sql, "from": _sources(runtime, str(args["file_path"]), sql, record)}
    stored = await runtime.save_file(name, data, source=QUERY_RESULT, made=made)
    return (
        f"Saved {stored.name} for the user: {count} rows, {stored.size} bytes; the chat shows "
        f"its download link. Its first rows:\n{shown}"
    )


def _sources(
    runtime: Any, file_path: str, sql: str, record: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The results a query read — the file asked about, and each other result the run keeps
    whose table the query names — each as its table, the tool it came from and when it was
    fetched (``results.keep``), for how a file it made was made (``FileStore.record``); a file
    kept for the user (its ``record``) as its path and when it was saved."""
    path = "/" + file_path.lstrip("/")
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
    file_path: str, text: str | Table | None = None
) -> tuple[dict[str, str | Table], str] | str:
    """What a query of ``file_path`` loads — the run's saved results by table name (the file
    asked about among them, given as ``text`` when it is no result the run keeps: a file Deep
    Agents saved, a file kept for the user, read from the store) — and the file's own table;
    or why there is nothing to load."""
    path = "/" + file_path.lstrip("/")
    runtime = current()
    kept = runtime.replay.journal.results if runtime is not None else {}
    named = tables(runtime) if runtime is not None else {}
    if text is None:
        text = kept.get(path)
    if text is None or runtime is None:
        return f"Error: File '{path}' not found"
    texts: dict[str, str | Table] = {named[p]: kept[p] for p in named}
    if path not in named:
        texts[f"{TABLE}_file"] = text  # a file Deep Agents saved, a kept one: ``result``
    if sum(map(_size, texts.values())) > LOAD_CHARS:
        return (
            f"Error: this run's saved results are more than {LOAD_CHARS} characters, more than a "
            "query here loads: filter on the tool's side, or compute in a sandbox"
        )
    return texts, named.get(path, f"{TABLE}_file")


async def run(
    file_path: str,
    sql: str,
    params: Sequence[Any] = (),
    text: str | Table | None = None,
    *,
    save: str | None = None,
) -> Any:
    """``sql`` (with ``params``) over the run's saved results (:func:`loadable`), in a worker
    thread (see :func:`query`): its rows shown, or — ``save``, a format of ``tools.export`` —
    every row as a file of that format, how many, and the first :data:`SAVED_ROWS` shown; or
    what is wrong, as text."""
    found = loadable(file_path, text)
    if isinstance(found, str):
        return found
    texts, own = found
    call = _Call()
    take, seconds = (_shown, SECONDS) if save is None else (_saved(save), SAVE_SECONDS)
    work = asyncio.to_thread(
        _answer, texts, own, sql, params, call=call, take=take, seconds=seconds
    )
    try:
        return await asyncio.shield(asyncio.ensure_future(work))
    except asyncio.CancelledError:
        call.cancel()  # the thread stops at once, and closes its database
        raise


class _Call:
    """One query's database while it is open, and whether its call was cancelled: a cancel
    interrupts the database (never one closed, nor one being closed: ``lock``)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.db: sqlite3.Connection | None = None
        self.cancelled = False

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            if self.db is not None:
                self.db.interrupt()


def _answer(
    texts: Mapping[str, str | Table],
    own: str,
    sql: str,
    params: Sequence[Any],
    *,
    call: _Call,
    take: Callable[[sqlite3.Cursor, list[str]], Any] | None = None,
    seconds: float = SECONDS,
) -> Any:
    """The query over ``texts`` (:func:`query`), in this worker thread, once its turn comes
    (and its call was not cancelled meanwhile), within ``seconds``; what ``take`` makes of its
    rows (:func:`_shown` by default); ``call`` holds its database while it is open."""
    if not _RUNNING.acquire(timeout=WAIT_SECONDS):
        return f"Error: the query waited {WAIT_SECONDS:g} s for others to finish: ask it again"
    deadline = time.monotonic() + LOAD_SECONDS
    ends = [deadline]  # what the progress handler stops at: loading's, then the query's
    try:
        with contextlib.closing(sqlite3.connect(":memory:", check_same_thread=False)) as db:
            with call.lock:
                if call.cancelled:
                    return "Error: cancelled"
                call.db = db
            try:
                db.set_progress_handler(lambda: int(time.monotonic() > ends[0]), 10_000)
                _loaded(db, texts, own, deadline)
                deadline = ends[0] = time.monotonic() + seconds  # the query's own, from here
                cursor = db.execute(sql, tuple(params))
                names = [d[0] for d in cursor.description or ()]
                return (take or _shown)(cursor, names)
            except export.TooManyRows as exc:
                return f"Error: the query has {exc}: aggregate, or narrow it with WHERE"
            except sqlite3.Error as exc:
                said = "took too long" if time.monotonic() > deadline else str(exc)
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


def _shown(cursor: sqlite3.Cursor, names: list[str]) -> str:
    """A query's first :data:`ROWS` rows as CSV, at most :data:`OUTPUT_CHARS`, saying when
    there were more."""
    rows = cursor.fetchmany(ROWS + 1)
    shown = _csv(names, rows[:ROWS])
    if len(shown) > OUTPUT_CHARS:
        shown = shown[:OUTPUT_CHARS] + f"\n…[cut at {OUTPUT_CHARS} characters: aggregate more]"
    elif len(rows) > ROWS:
        shown += f"\n…[only the first {ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    return shown


def _saved(kind: str) -> Callable[[sqlite3.Cursor, list[str]], tuple[bytes, int, str]]:
    """What a query saving its result as a ``kind`` file takes of its rows: every row written
    as they come (``tools.export.written``), how many, and the first :data:`SAVED_ROWS` as
    CSV."""

    def take(cursor: sqlite3.Cursor, names: list[str]) -> tuple[bytes, int, str]:
        first: list[Any] = []

        def rows() -> Any:
            for row in cursor:
                if len(first) < SAVED_ROWS:
                    first.append(row)
                yield row

        data, count = export.written(kind, names, rows())
        return data, count, _csv(names, first)

    return take


def _csv(names: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    writer.writerows(rows)
    return out.getvalue().rstrip("\n")


def _loaded(
    db: sqlite3.Connection, texts: Mapping[str, str | Table], own: str, deadline: float
) -> None:
    """``db`` (in memory) holding each of ``texts`` as a table (by its name), the view
    :data:`TABLE` on ``own``, read-only from then on (the authorizer); its tables' pages at most
    :data:`DB_BYTES`, its sorts' and joins' memory :data:`TEMP_CACHE_BYTES`; its
    loading within ``deadline`` (``time.monotonic``)."""
    for name, arguments, aggregate in AGGREGATES:
        db.create_aggregate(name, arguments, aggregate)
    size = int(db.execute("PRAGMA page_size").fetchone()[0])
    db.execute(f"PRAGMA main.max_page_count = {DB_BYTES // size}")
    db.execute(f"PRAGMA temp.cache_size = -{TEMP_CACHE_BYTES // 1024}")
    for name, text in texts.items():
        _table(db, name, text)
    db.execute(f'CREATE VIEW {TABLE} AS SELECT * FROM "{own}"')
    db.commit()
    if time.monotonic() > deadline:
        raise sqlite3.OperationalError("loading the saved results took too long")
    db.set_authorizer(lambda action, *_: 0 if action in _ALLOWED else 1)  # 1: SQLITE_DENY


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


def _table(db: sqlite3.Connection, name: str, text: str | Table) -> None:
    """``text`` (or a table's rows) as the table ``name``: typed columns (see the module), its
    rows."""
    columns, rows = _rows(text)
    names = column_names(columns)
    kinds = [_kind([r[n] for r in rows]) for n in range(len(names))]
    typed = ", ".join(f'"{n}" {k}' for n, k in zip(names, kinds, strict=True))
    db.execute(f'CREATE TABLE "{name}" ({typed})')
    marks = ", ".join("?" for _ in names)
    db.executemany(f'INSERT INTO "{name}" VALUES ({marks})', [_typed(r, kinds) for r in rows])


def _json_rows(text: str) -> list[Mapping[str, Any]] | None:
    """The rows of a JSON result: an array of objects, or the first such array an object
    holds (``{"items": [...]}``); ``None`` when it is no such JSON."""
    try:
        found = json.loads(text)
    except ValueError:
        return None
    if isinstance(found, dict):
        found = next((v for v in found.values() if _objects(v)), None)
    return found if _objects(found) else None


def _objects(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _cell(value: Any) -> Any:
    """A JSON value as a cell: a number or text as it is, anything else its JSON text."""
    if value is None or (isinstance(value, str | int | float) and not isinstance(value, bool)):
        return value
    return json.dumps(value)


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
    """Whether a cell (text, or a JSON number: a JSON true is its text) is a number SQLite
    keeps exactly: a whole one within 64 bits, a finite one — never text a number would change
    (a code with a leading zero, ``00123``; a run of digits too long for one)."""
    if isinstance(value, int):
        return -(2**63) <= value < 2**63
    if isinstance(value, float):
        return not whole and math.isfinite(value)
    text = str(value).strip()
    if _LEADING_ZERO.match(text) or len(text) > NUMBER_CHARS:
        return False
    try:
        found = (int if whole else float)(text)
    except ValueError:
        return False
    return _number(found, whole=whole)


def _typed(row: Sequence[Any], kinds: Sequence[str]) -> list[Any]:
    """A row's cells as their columns' kinds (an empty cell: ``NULL``)."""
    out: list[Any] = []
    for value, kind in zip(row, kinds, strict=True):
        if value in (None, ""):
            out.append(None)
        elif kind == "INTEGER":
            out.append(int(str(value).strip()))
        elif kind == "REAL":
            out.append(float(str(value).strip()))
        else:
            out.append(value)
    return out
