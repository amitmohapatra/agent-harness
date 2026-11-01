# The query engine behind `query_result`: SQLite, DuckDB, Polars, pandasql

`query_result` (#66) runs read-only SQL over a run's saved results on stdlib SQLite, with the
statistics SQLite lacks written as Python aggregates (`tools/query.py`). This measures that
choice against DuckDB and Polars SQL (and pandasql, which is SQLite underneath). Every engine
gets the same questions over the same data, and each answer is checked against Python's own.
The decision is in [ADR 0007](../adr/0007-query-engine.md).

**Run it:** `QUERY_ENGINES_SITE=<folder> python -m benchmark.query_engines` (`benchmark/query_engines/`).
Its `sqlite` and `duckdb` engines are now the harness's own (`tools.query.SQLite`,
`tools.query.DuckDB`), so it measures the code `query_result` runs: the switch's before and
after are in [ADR 0007](../adr/0007-query-engine.md#implemented) and
[results/harness.md](../../benchmark/query_engines/results/harness.md). The tables below are the
measurement that chose the engine, with DuckDB as a prototype loader.
DuckDB 2.0.0, Polars 2.1.0 and pandasql 0.7.3 were installed with `uv pip install --target` into
a scratch folder, not into the harness's environment. Each measurement runs in its own process.
Raw numbers are in [results/local.json](../../benchmark/query_engines/results/local.json), the
tables in [results/local.md](../../benchmark/query_engines/results/local.md).

**Machine:** Linux x86_64, 4 CPUs shared with other jobs (timings are noisy, medians shown), Python 3.13.18, SQLite 3.53.1.

## What was measured

- **Data:** the #66 test shapes at 10k, 50k, 500k and 2M rows.
  - The ledger of `test_formulas.py` (7 columns), as CSV and as JSON.
  - The three tools of `test_large_data.py`: inventory as CSV, sales as a JSON array, lead times as a JSON object holding the array.
- **Questions (18):**
  - filter + sort + top-N;
  - group-by sums;
  - two 3-table joins;
  - formulas: margin, growth %, weighted score, what-if;
  - windows: a moving average and `rank`;
  - statistics: `stddev`, `median`, a percentile, `corr`, regression slope and intercept, `regr_r2`;
  - date math: age in days, and counts by month.
- **Correctness:** text and integers must match exactly. Floating values must match to 12 significant digits (as `test_formulas`), against `math.fsum` and `statistics`.
  - Each engine first runs the SQL the tool documents (SQLite's).
  - Where the engine's dialect differs, it then runs its own SQL for the same question.
- **Load:** parse plus ingest of the 4 tables, from the text the run holds. This is what every `query_result` call does today.

## Results

**Correctness: identical at every size and in both formats** (10k, 50k, 500k and 2M; CSV and JSON):

| engine | right as documented | right in its own dialect | wrong | cannot |
|---|---|---|---|---|
| SQLite (harness) | 18 | 0 | 0 | 0 |
| DuckDB | 16 | 2: `percentile_cont(x, p)` is `quantile_cont`; `julianday` is date subtraction | 0 | 0 |
| Polars SQL | 12 | 5: percentile, `regr_slope` / `regr_intercept` / `regr_r2` rewritten from `corr` and `stddev`, both date questions | **1, silently**: `avg((revenue - cost) * 1.0 / revenue)` is rounded to 11 decimals (the `1.0` literal is a decimal) | 0 |
| pandasql | 14 | 0 | 0 | 4: `stddev`, `corr`, `regr_*` |

**Value fidelity:**

- SQLite (#66 code) lost ids past 64 bits:
  - CSV: rounded to a `REAL`, so two ids became one;
  - JSON: the whole load failed with `OverflowError`.
  - Fixed in #83, found by this probe.
- DuckDB keeps them (`HUGEINT`).
- Polars turns the code `007` into the number `7`.
- pandasql cannot load them at all.

**Load and one call** (a call loads the 4 tables, then runs `join_groups`):

| rows (ledger CSV text) | SQLite load + query | DuckDB load + query | Polars load + query |
|---|---|---|---|
| 10k (1.4 MB) | 0.11 s + 12 ms | 0.12 s + 5 ms | 0.06 s + 3 ms |
| 50k (6.8 MB) | 0.65 s + 64 ms | 0.33 s + 13 ms | 0.20 s + 7 ms |
| 500k (70 MB) | 9.4 s + 960 ms | 2.2 s + 126 ms | 2.8 s + 116 ms |
| 2M (285 MB) | 33 s + 3.3 s | 6.6 s + 219 ms (with a 3 GB memory limit; at 512 MB it refused to load) | 8.0 s + 199 ms |

**All 18 questions on a loaded database** (sum of p50 times):

| rows | SQLite | DuckDB | Polars |
|---|---|---|---|
| 10k | 90 ms | 30 ms | 19 ms |
| 50k | 547 ms | 75 ms | 50 ms |
| 500k | 5.6 s | 0.56 s | 0.64 s |
| 2M | 19.3 s | 1.4 s | 1.6 s |

The gap is widest where SQLite runs Python. At 500k: `stddev` 220 ms against 4 ms, `regression` 495 against 8, a 3-table join 1.2 s against 0.12, `rank` 690 against 140. The per-question p50/p95 tables are in results/local.md.

**Peak memory above the process before loading:**

| rows | SQLite | DuckDB | Polars |
|---|---|---|---|
| 10k | +9 MB | +75 MB | +63 MB |
| 50k | +44 MB | +86 MB | +139 MB |
| 500k | +398 MB | +399 MB | +585 MB |
| 2M CSV | +1,661 MB | +1,314 MB | +1,622 MB |

**8 calls at once,** each loading the 4 tables and joining (SQLite through the harness's own `_answer`, 4 at a time, 5 s per query):

| rows | SQLite: wall s, errors | DuckDB | Polars |
|---|---|---|---|
| 50k | 6.2 s, 0 | 1.0 s, 0 | 1.0 s, 0 |
| 500k | 75 s, **4 timed out** | 7.9 s, 0 | 13 s, 0 |

The SQLite path holds the GIL: Python parses and types every cell, and the statistics run in Python.

**DuckDB with the harness's typing rules:**

- DuckDB's own type sniffing differs from the harness's contract. JSON numeric strings stay `VARCHAR`, it samples rows, and it reads dates as `DATE`.
- A prototype loader reads everything as text, then types each column in one SQL pass by the harness's rules (whole number within 64 bits, number, leading zero or past 64 bits: text). Ledger alone:
  - 50k CSV: 0.25 s; JSON: 0.28 s;
  - 500k CSV: 0.65 s; JSON: 1.85 s.
- For comparison, SQLite loads the same ledger with the other 3 tables in 0.65 s (50k) and 7 to 9 s (500k).

## Safety: a query trying more than reading its own tables

DuckDB was locked after loading:

- `enable_external_access = false`, `lock_configuration = true`;
- `memory_limit` 512 MB, extension autoload and autoinstall off;
- only one `SELECT` admitted (`duckdb.extract_statements`).

| probe | SQLite (harness authorizer) | DuckDB (locked) | Polars SQL | pandasql |
|---|---|---|---|---|
| read another file | blocked | blocked | **reads it** (`read_csv('…/secret.csv')` returned the secret) | **runs** (`ATTACH`) |
| network | none | blocked | **tries** (`read_csv('http://…')` connects, with 10 retries) | none |
| write a file | blocked | blocked | none | **writes** (`ATTACH` creates the file) |
| load or install an extension | blocked | blocked | none | blocked by Python's default |
| change a setting | blocked | blocked (and locked) | none | **runs** |
| write a table, delete rows | blocked | blocked | **runs** | **runs** |
| an endless query | stopped at 5 s (progress handler) | stopped at 2 s (`interrupt()`) | no way to interrupt a running `collect()` | none |
| a query building ~0.2 GB of text | **runs: +370 MB in 0.7 s** (only the tables' pages are capped) | stopped by `memory_limit` (+258 MB) | no limit | no limit |
| result size | `fetchmany` | `fetchmany` | `.limit()` | whole frame |

## Install

| | SQLite | DuckDB 2.0.0 | Polars 2.1.0 | pandasql 0.7.3 |
|---|---|---|---|---|
| wheel | stdlib | 24 MB (x86_64), 22 MB (aarch64), 18 MB (macOS arm64) | 1 MB plus a 54 MB runtime | needs pandas (12 MB), NumPy (17 MB), SQLAlchemy |
| installed | 0 | 67 MB | 180 MB | ~130 MB with its dependencies |
| Python 3.12 / 3.13, x86_64 and arm64 | yes | yes: one `cp312-abi3` wheel (manylinux_2_28, macOS, Windows); **no musllinux** (Alpine) | yes (`abi3`) | yes |
| dependencies | none | none | its runtime | pandas, NumPy, SQLAlchemy |
| licence | public domain | MIT | MIT | MIT (pandas BSD-3) |

## Decision by the keep rule

The order is accuracy and coverage, then safety, then latency and memory, then install cost.

1. **Accuracy and coverage.** SQLite and DuckDB tie: 18/18 right. DuckDB's two dialect names can
   keep the documented spelling with two macros (`percentile_cont`, `julianday`). Polars is out:
   it is silently wrong on a margin, it turns codes into numbers, and it lacks `regr_*`. pandasql
   is out: it lacks the statistics.
2. **Safety.** DuckDB is ahead. Both refuse files, network, extensions, settings and writes, and
   both stop an endless query. Only DuckDB bounds a query's own memory: SQLite's page cap does
   not cover the values a query builds. Polars SQL reads local files and opens network
   connections with no setting to stop it, which disqualifies it.
3. **Latency and memory.** DuckDB is clearly ahead from 50k rows: a call is 2x faster at 50k and
   4.5x at 500k, the questions run 7 to 10x faster, and 8 calls at once run 6 to 9.5x faster
   with no timeouts (SQLite: 4 of 8 at 500k). At 10k rows the two tie on time, and SQLite uses 8x
   less memory.
4. **Install.** SQLite wins: 0 MB, every platform. DuckDB adds 67 MB and has no Alpine wheel.

**DuckDB wins.** It becomes the engine when installed, as an optional `query` extra included in
`all`. SQLite stays as the zero-dependency fallback, justified by the numbers in step 4 (0 MB
against 67 MB in a core that is otherwise small, and no musl wheel) and by the 10k tie with 8x
less memory. One tool contract covers both engines: the same table names, the harness's typing
rules, the documented function names, and the same limits. Details are in
[ADR 0007](../adr/0007-query-engine.md).
