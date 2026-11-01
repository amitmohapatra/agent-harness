"""The engines compared, each loading the same texts (the run's saved results) and answering the
same SQL, as the harness's ``query_result`` would use it: a database made for the call from the
texts, read-only from then on, closed after.

* ``sqlite`` and ``duckdb`` — the harness's own engines, through their ``load`` and ``execute``:
  ``tools.query.SQLite``, the fallback (Python's parse, ``executemany``, the statistics
  aggregates, the authorizer), and ``tools.query.DuckDB``, the engine when the ``query`` extra
  is installed (DuckDB's own CSV and JSON readers from a private temporary file, the harness's
  typing rules in one SQL pass, then ``enable_external_access = false``,
  ``lock_configuration = true``, a ``memory_limit`` and only a single ``SELECT``);
* ``polars`` — ``pl.read_csv`` / ``pl.read_json`` (whole-column type inference) and
  ``pl.SQLContext``;
* ``pandasql`` — pandas' readers and ``pandasql.sqldf`` (SQLite underneath, a copy per query).

Polars and pandasql are not dependencies of the harness (DuckDB is its ``query`` extra): an
engine that does not import is reported as not installed. ``QUERY_ENGINES_SITE`` names a
folder they were installed in (``pip install --target``), searched after the harness's own
packages."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, Protocol

from trellis.harness.tools import query

if os.environ.get("QUERY_ENGINES_SITE"):
    sys.path.append(os.environ["QUERY_ENGINES_SITE"])

#: The most rows an answer holds (as the harness's ``query.ROWS``, and one more to say so).
ROWS: Final = query.ROWS + 1


class Engine(Protocol):
    name: str

    def load(self, texts: Mapping[str, str], own: str) -> Any: ...

    def run(self, db: Any, sql: str) -> tuple[list[str], list[list[Any]]]: ...

    def close(self, db: Any) -> None: ...


def _json(text: str) -> bool:
    return text.lstrip()[:1] in ("[", "{")


# --------------------------------------------------------------------------- the harness's own


class Harness:
    """One of the harness's engines (``tools.query.Engine``), as ``query_result`` runs it:
    ``opened``, ``load``, ``execute``, ``fetchmany`` (its limits on time and concurrency are
    ``query._answer``'s, which ``worker.concurrent`` runs)."""

    def __init__(self, engine: query.Engine) -> None:
        self.engine = engine
        self.name = engine.name.lower()
        self._open: dict[int, contextlib.AbstractContextManager[Any]] = {}

    def load(self, texts: Mapping[str, str], own: str) -> Any:
        opened = self.engine.opened()
        db = opened.__enter__()
        self._open[id(db)] = opened
        try:
            self.engine.load(db, texts, own)
        except BaseException:
            self.close(db)
            raise
        return db

    def run(self, db: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
        cursor = self.engine.execute(db, sql, ())
        rows = cursor.fetchmany(ROWS)
        return [d[0] for d in cursor.description or ()], [list(r) for r in rows]

    def close(self, db: Any) -> None:
        self._open.pop(id(db)).__exit__(None, None, None)


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
    "sqlite": lambda: Harness(query.SQLite()),
    "duckdb": lambda: Harness(query.DuckDB()),
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
