# 0007. `query_result` runs on DuckDB when it is installed, with SQLite as the zero-dependency fallback

**Status:** Accepted (2026-10-31). Amends [ADR 0006](0006-what-the-model-gets.md), whose list of
libraries proposed "DuckDB for compute, or stdlib `sqlite3` if it measures as sufficient".
Implemented in `tools/query.py` (the `query` extra): its numbers are under
[Implemented](#implemented).

## Context

#66 shipped `query_result` on stdlib SQLite, so it needs no dependency. The statistics SQLite
lacks (`stddev`, `median`, `percentile_cont`, `corr`, `regr_*`) are Python aggregates. Each call
parses the run's saved results in Python, types every cell, loads them, and queries.

The [query engine benchmark](../research/QUERY_ENGINE.md) (`benchmark/query_engines/`) ran the
same 18 questions over the #66 data shapes (10k to 2M rows, CSV and JSON, 1 to 3 tables) through
SQLite, DuckDB, Polars SQL and pandasql. It checked every answer against Python's own, and
measured load, query time, memory, 8 calls at once, and what a hostile query can do.

## Decision

The order is the keep rule's: accuracy and coverage, then safety, then latency and memory, then install cost.

1. **Accuracy.** SQLite and DuckDB tie at 18/18. Polars SQL is silently wrong on one formula,
   turns codes such as `007` into numbers, and lacks `regr_*`. pandasql lacks the statistics.
2. **Safety.** DuckDB, locked after loading, refuses everything SQLite's authorizer refuses. It
   also bounds a query's own memory (`memory_limit`), which SQLite's page cap does not: one query
   built +370 MB in 0.7 s. Polars SQL reads local files and opens network connections with no
   setting to stop it, which disqualifies it.
3. **Latency.** DuckDB wins from 50k rows:
   - a call is 2x faster at 50k and 4.5x at 500k;
   - the 18 questions run 7 to 14x faster;
   - 8 calls at once are 6x faster at 50k and 9.5x at 500k. At 500k, SQLite timed out 4 of the 8;
     DuckDB timed out none.

   At 10k rows they tie, and SQLite uses 8x less memory.
4. **Install.** SQLite costs 0 MB. DuckDB adds 67 MB (a 24 MB wheel, no dependencies, MIT): one
   `abi3` wheel covers Python 3.12 and 3.13 on x86_64 and arm64, but there is no musllinux
   (Alpine) wheel.

So:

- **DuckDB is the engine when it is installed.** It comes as a `query` extra, included in `all`.
  It loads each saved result with its own CSV and JSON readers, from a private temporary file.
  The harness's typing rules are applied in one SQL pass, so a column's type is the same on both
  engines: a whole number within 64 bits, a number, or text (a leading zero, or past 64 bits).
  Then the database is locked:
  - `enable_external_access = false`, `lock_configuration = true`;
  - `memory_limit` of `DB_BYTES`, no extension autoload;
  - only one `SELECT` admitted;
  - `interrupt()` for the time limit and for a cancelled run.
- **SQLite stays only as the fallback.** It is justified by the install numbers (0 MB against
  67 MB in an otherwise small core, and every platform) and by the 10k tie with less memory. It
  is not a second engine to choose between: the harness uses DuckDB whenever it can import it.
- **One tool contract on both engines:**
  - the same table names, typing rules and limits;
  - the documented function names: on DuckDB, `percentile_cont(x, p)` and `julianday(d)` are
    macros;
  - the description names the engine, so the model writes its dialect.

## Consequences

- The Python statistics aggregates and the Python parse stay only on the fallback path.
- The tests run the query suites (`test_query`, `test_formulas`, `test_query_safety`,
  `test_large_data`) on both engines. The fallback stays at 100% coverage.
- `benchmark/query_engines` is re-run when either engine is upgraded or the loader changes, and
  its results go to `benchmark/query_engines/results/`.
- Still to do: caching a run's loaded database between its queries, instead of loading it
  again on every call. (SQLite's per-value length limit, `setlimit`, came with the switch: a
  query building a 200 MB value now stops at 64 MB.)

## Implemented

`tools/query.py` has one engine strategy (`Engine`): `DuckDB` when `duckdb` can be imported,
else `SQLite`, chosen by `chosen()`, never by a setting. Both share the call's limits, its
worker thread and a watcher that interrupts a query past its time, the answer's CSV, and the
typing rules (one set of regular expressions, for Python's `re` and DuckDB's RE2). DuckDB
reads JSON cells as text (a number as written, a nested value as its JSON), types the
columns in one pass (columns already text in their first 2,048 rows are not checked again),
and keeps no spill file (`temp_directory` empty). The harness's own engines, measured on the
same machine (`benchmark/query_engines/results/harness-main.md` is main's SQLite before,
[harness.md](../../benchmark/query_engines/results/harness.md) after):

| | SQLite (before) | DuckDB (after) |
|---|---|---|
| right, 18 questions, 10k to 500k rows, CSV and JSON | 18 | 17 as documented, 1 in DuckDB's dialect (`strftime(d, fmt)`), 0 wrong |
| load of the 4 tables, 50k rows, CSV | 1.01 s | 0.37 s |
| load of the 4 tables, 500k rows, CSV / JSON | 9.5 s / 9.3 s | 2.5 s / 3.0 s |
| the 18 questions, 500k rows, CSV | 5.9 s | 0.89 s |
| 8 calls at once, 50k / 500k rows | 7.2 s / 89 s | 1.8 s / 12 s |
| peak memory, 10k / 500k rows, CSV | +9 MB / +398 MB | +89 MB / +348 MB |
