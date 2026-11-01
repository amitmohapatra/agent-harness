"""The engines compared, each loading the same texts (the run's saved results) and answering the
same SQL, as the harness's ``query_result`` would use it: a database made for the call from the
texts, read-only from then on, closed after.

* ``sqlite`` — the harness's own code (``tools.query._loaded``: Python's parse, ``executemany``,
  the statistics aggregates, the authorizer);
* ``duckdb`` — each text written to a private temporary file and read by DuckDB's own CSV and
  JSON readers, then ``enable_external_access = false`` and ``lock_configuration = true``, a
  ``memory_limit``, and only a single ``SELECT`` admitted (``duckdb.extract_statements``);
* ``polars`` — ``pl.read_csv`` / ``pl.read_json`` (whole-column type inference) and
  ``pl.SQLContext``;
* ``pandasql`` — pandas' readers and ``pandasql.sqldf`` (SQLite underneath, a copy per query).

DuckDB, Polars and pandasql are not dependencies of the harness: an engine that does not import
is reported as not installed. ``QUERY_ENGINES_SITE`` names a folder they were installed in
(``pip install --target``), searched after the harness's own packages."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, Protocol

if os.environ.get("QUERY_ENGINES_SITE"):
    sys.path.append(os.environ["QUERY_ENGINES_SITE"])

#: The most rows an answer holds (as the harness's ``query.ROWS``, and one more to say so).
ROWS: Final = 201
#: DuckDB's memory limit per query database, as the harness bounds SQLite's pages
#: (``QUERY_ENGINES_DUCKDB_MEMORY`` to measure past it).
DUCKDB_MEMORY: Final = os.environ.get("QUERY_ENGINES_DUCKDB_MEMORY", "512MB")


class Engine(Protocol):
    name: str

    def load(self, texts: Mapping[str, str], own: str) -> Any: ...

    def run(self, db: Any, sql: str) -> tuple[list[str], list[list[Any]]]: ...

    def close(self, db: Any) -> None: ...


def _json(text: str) -> bool:
    return text.lstrip()[:1] in ("[", "{")


# ------------------------------------------------------------------------------------- SQLite


class SQLite:
    """The harness's engine, through its own code."""

    name = "sqlite"

    def __init__(self) -> None:
        from trellis.harness.tools import query

        self.query = query

    def load(self, texts: Mapping[str, str], own: str) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        self.query._loaded(db, texts, own, math.inf)
        return db

    def run(self, db: sqlite3.Connection, sql: str) -> tuple[list[str], list[list[Any]]]:
        cursor = db.execute(sql)
        rows = cursor.fetchmany(ROWS)
        return [d[0] for d in cursor.description or ()], [list(r) for r in rows]

    def close(self, db: sqlite3.Connection) -> None:
        db.close()


# ------------------------------------------------------------------------------------- DuckDB


class DuckDB:
    """DuckDB, locked down after its tables are loaded."""

    name = "duckdb"

    def __init__(self, threads: int | None = None) -> None:
        import duckdb  # pyright: ignore[reportMissingImports]

        self.duckdb = duckdb
        self.threads = threads

    def load(self, texts: Mapping[str, str], own: str) -> Any:
        config: dict[str, Any] = {
            "memory_limit": DUCKDB_MEMORY,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        }
        if self.threads:
            config["threads"] = self.threads
        con = self.duckdb.connect(":memory:", config=config)
        with tempfile.TemporaryDirectory(prefix="duckq-") as folder:
            for name, text in texts.items():
                path = os.path.join(folder, name)
                with open(path, "w") as out:
                    out.write(text)
                con.execute(f'CREATE TABLE "{name}" AS {self._reader(con, path, text)}')
        con.execute(f'CREATE VIEW result AS SELECT * FROM "{own}"')
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        return con

    def _reader(self, con: Any, path: str, text: str) -> str:
        quoted = path.replace("'", "''")
        if not _json(text):
            return f"SELECT * FROM read_csv('{quoted}')"
        if text.lstrip().startswith("["):
            return f"SELECT * FROM read_json('{quoted}', format = 'array')"
        # an object holding the array (a paged tool's rows): its first list of objects
        whole = f"read_json('{quoted}', maximum_object_size = {len(text) + 1})"
        rel = con.sql(f"SELECT * FROM {whole}")
        key = next(c for c, t in zip(rel.columns, rel.types, strict=True) if "STRUCT" in str(t))
        return f'SELECT u.* FROM (SELECT unnest("{key}") AS u FROM {whole})'

    def run(self, db: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
        statements = self.duckdb.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != self.duckdb.StatementType.SELECT:
            raise PermissionError("only one SELECT runs")
        cursor = db.execute(sql)
        rows = cursor.fetchmany(ROWS)
        return [d[0] for d in cursor.description or ()], [list(r) for r in rows]

    def close(self, db: Any) -> None:
        db.close()


# ------------------------------------------------------------------------------------- Polars


class Polars:
    """Polars' readers and its SQL context."""

    name = "polars"

    def __init__(self) -> None:
        import polars  # pyright: ignore[reportMissingImports]

        self.pl = polars

    def load(self, texts: Mapping[str, str], own: str) -> Any:
        frames = {name: self._frame(text) for name, text in texts.items()}
        return self.pl.SQLContext(frames={**frames, "result": frames[own]})

    def _frame(self, text: str) -> Any:
        pl = self.pl
        if not _json(text):
            head = text.split("\n", 1)[0]
            sep = max(",\t;|", key=head.count)
            return pl.read_csv(io.StringIO(text), separator=sep, infer_schema_length=None)
        frame = pl.read_json(io.StringIO(text), infer_schema_length=None)
        if text.lstrip().startswith("["):
            return frame
        key = next(c for c, t in frame.schema.items() if isinstance(t, pl.List))
        return frame.select(pl.col(key).explode()).unnest(key)

    def run(self, db: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
        frame = db.execute(sql).limit(ROWS).collect()
        return list(frame.columns), [list(r) for r in frame.rows()]

    def close(self, db: Any) -> None:
        del db


# ----------------------------------------------------------------------------------- pandasql


class PandaSQL:
    """pandas' readers and ``pandasql.sqldf`` (each query copies the frames into SQLite)."""

    name = "pandasql"

    def __init__(self) -> None:
        import pandas  # pyright: ignore[reportMissingImports]
        import pandasql  # pyright: ignore[reportMissingImports]

        self.pd = pandas
        self.sqldf = pandasql.sqldf

    def load(self, texts: Mapping[str, str], own: str) -> dict[str, Any]:
        frames = {}
        for name, text in texts.items():
            if not _json(text):
                frames[name] = self.pd.read_csv(io.StringIO(text))
            elif text.lstrip().startswith("["):
                frames[name] = self.pd.DataFrame(json.loads(text))
            else:
                found = json.loads(text)
                frames[name] = self.pd.DataFrame(
                    next(v for v in found.values() if isinstance(v, list))
                )
        frames["result"] = frames[own]
        return frames

    def run(self, db: dict[str, Any], sql: str) -> tuple[list[str], list[list[Any]]]:
        frame = self.sqldf(sql, db)
        return list(frame.columns), [list(r) for r in frame.head(ROWS).itertuples(index=False)]

    def close(self, db: dict[str, Any]) -> None:
        db.clear()


ENGINES: Final[dict[str, Callable[[], Engine]]] = {
    "sqlite": SQLite,
    "duckdb": DuckDB,
    "polars": Polars,
    "pandasql": PandaSQL,
}


def value(v: Any) -> Any:
    """An engine's cell as Python's answer has it: a decimal as a float, a date as ISO text, a
    duration in days, a NumPy scalar as its Python value."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, timedelta):
        return v.days
    item = getattr(v, "item", None)
    return item() if callable(item) else v


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


@contextlib.contextmanager
def opened(engine: Engine, texts: Mapping[str, str], own: str) -> Any:
    db = engine.load(texts, own)
    try:
        yield db
    finally:
        engine.close(db)
