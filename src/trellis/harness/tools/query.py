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
text is at most :data:`SQL_CHARS`, the database's memory is bounded (:data:`HEAP_BYTES`), and at
most :data:`ROWS` rows come back (said when there were more).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from trellis.harness.runtime import current
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
#: The most rows a query answers with, its longest text, the seconds it may run, the memory
#: its database may take, and the delimiters a CSV result is read with.
ROWS: Final = 200
SQL_CHARS: Final = 5_000
SECONDS: Final = 5.0
HEAP_BYTES: Final = 256 * 1024 * 1024
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


def columns_of(text: str) -> list[str]:
    """The columns a kept result's table has (as :func:`query` names them)."""
    return _names(rows_of(text)[0])


def query(args: dict[str, Any], text: str | None = None) -> str:
    """``query_result``: the query's rows as CSV (a header line first), at most :data:`ROWS`;
    or what is wrong — the arguments, the path, the SQL (SQLite's own words)."""
    problem = arguments_problem(QUERY_RESULT_SCHEMA, args)
    if problem is not None:
        return not_run(QUERY_RESULT, problem)
    path = "/" + str(args["file_path"]).lstrip("/")
    sql = str(args["sql"]).strip().rstrip(";")
    runtime = current()
    if text is None and runtime is not None:
        text = runtime.replay.journal.results.get(path)
    if text is None or runtime is None:
        return f"Error: File '{path}' not found"
    if len(sql) > SQL_CHARS:
        return f"Error: the query is longer than {SQL_CHARS} characters"
    deadline = time.monotonic() + SECONDS
    with contextlib.closing(_loaded(text)) as db:  # loaded per query: 10,000 rows in ~50 ms
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = db.execute(sql)
            rows = cursor.fetchmany(ROWS + 1)
        except sqlite3.Error as exc:
            said = "took too long" if time.monotonic() > deadline else str(exc)
            return f"Error: the query failed: {said}. Only a SELECT over the table {TABLE} runs."
        names = [d[0] for d in cursor.description or ()]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(names)
    writer.writerows(rows[:ROWS])
    shown = out.getvalue().rstrip("\n")
    if len(rows) > ROWS:
        shown += f"\n…[only the first {ROWS} rows: aggregate, or narrow it with WHERE / LIMIT]"
    return shown


def _loaded(text: str) -> sqlite3.Connection:
    """An in-memory database holding ``text`` as the table :data:`TABLE`, read-only from then
    on (the authorizer), its memory bounded."""
    columns, rows = rows_of(text)
    names = _names(columns)
    kinds = [_kind([r[n] for r in rows]) for n in range(len(names))]
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute(f"PRAGMA hard_heap_limit = {HEAP_BYTES}")
    typed = ", ".join(f'"{n}" {k}' for n, k in zip(names, kinds, strict=True))
    db.execute(f"CREATE TABLE {TABLE} ({typed})")
    marks = ", ".join("?" for _ in names)
    db.executemany(f"INSERT INTO {TABLE} VALUES ({marks})", [_typed(r, kinds) for r in rows])
    db.commit()
    db.set_authorizer(lambda action, *_: 0 if action in _ALLOWED else 1)  # 1: SQLITE_DENY
    return db


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


def _names(columns: Sequence[str]) -> list[str]:
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
