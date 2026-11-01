# 0007. `query_result` runs on DuckDB when it is installed, with SQLite as the zero-dependency fallback

**Status:** Accepted (2026-10-31). Amends [ADR 0006](0006-what-the-model-gets.md), whose list of
libraries proposed "DuckDB for compute, or stdlib `sqlite3` if it measures as sufficient". The
switch itself is the next item in the [improvement backlog](../research/IMPROVEMENT_BACKLOG.md).
Until it lands, `query_result` runs on SQLite as #66 shipped it.

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
- Still to do:
  - SQLite's unbounded query memory on the fallback path (a per-value length limit, `setlimit`);
  - caching a run's loaded database between its queries, instead of loading it again on every call.
