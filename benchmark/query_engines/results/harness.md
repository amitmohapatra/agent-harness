Machine: {'python': '3.13.18', 'sqlite': '3.53.1', 'cpus': 4, 'platform': 'Linux-6.18.44-fc-v185-x86_64-with-glibc2.39'}

## Correctness (18 questions, against Python's answers)

| engine | size/format | as-is | dialect | wrong | error | not as-is |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 10000/csv | 17 | 1 | 0 | 0 | by_month (dialect) |
| sqlite | 10000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 10000/json | 17 | 1 | 0 | 0 | by_month (dialect) |
| sqlite | 50000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 50000/csv | 17 | 1 | 0 | 0 | by_month (dialect) |
| sqlite | 50000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 50000/json | 17 | 1 | 0 | 0 | by_month (dialect) |
| sqlite | 500000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 500000/csv | 17 | 1 | 0 | 0 | by_month (dialect) |
| sqlite | 500000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 500000/json | 17 | 1 | 0 | 0 | by_month (dialect) |

## Load (parse + ingest of the 4 tables), peak memory, whole suite

| engine | size/format | text MB | load p50 s | suite query p50 sum ms | slowest p95 ms | peak RSS +MB |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 1.4 | 0.13 | 87.3 | 25.1 | 9 |
| duckdb | 10000/csv | 1.4 | 0.10 | 28.1 | 6.4 | 89 |
| sqlite | 10000/json | 2.3 | 0.12 | 71.2 | 18.7 | 8 |
| duckdb | 10000/json | 2.3 | 0.16 | 27.2 | 8.2 | 85 |
| sqlite | 50000/csv | 6.8 | 0.60 | 402.2 | 66.6 | 44 |
| duckdb | 50000/csv | 6.8 | 0.37 | 75.8 | 25.6 | 102 |
| sqlite | 50000/json | 11.4 | 0.73 | 451.2 | 100.6 | 37 |
| duckdb | 50000/json | 11.4 | 0.49 | 77.5 | 24.5 | 105 |
| sqlite | 500000/csv | 69.9 | 7.55 | 5356.2 | 1363.7 | 376 |
| duckdb | 500000/csv | 69.9 | 2.54 | 886.3 | 362.9 | 348 |
| sqlite | 500000/json | 115.4 | 7.00 | 5014.0 | 873.6 | 364 |
| duckdb | 500000/json | 115.4 | 3.01 | 374.3 | 139.3 | 347 |

## Query time p50 / p95 (ms), 10,000 rows, CSV ledger

| question | sqlite | duckdb |
|---|---|---|
| top_n | 0.3 / 12.3 | 0.9 / 1.2 |
| group_sums | 1.6 / 9.7 | 1.4 / 1.7 |
| join_top | 11.4 / 15.3 | 4.3 / 5.4 |
| join_groups | 8.4 / 12.7 | 4.2 / 6.4 |
| margin | 3.0 / 5.1 | 1.3 / 1.4 |
| growth | 0.6 / 0.8 | 0.9 / 0.9 |
| weighted_top | 1.1 / 1.7 | 1.5 / 1.5 |
| what_if | 0.5 / 0.5 | 0.9 / 1.2 |
| moving_avg | 5.3 / 6.7 | 2.0 / 2.0 |
| rank | 16.9 / 24.7 | 3.9 / 4.1 |
| stddev | 4.0 / 4.7 | 0.5 / 0.6 |
| median | 4.0 / 4.3 | 1.1 / 1.2 |
| percentile | 3.9 / 4.2 | 0.6 / 0.7 |
| corr | 3.6 / 5.3 | 0.5 / 0.6 |
| regression | 14.7 / 25.1 | 0.7 / 0.9 |
| r2 | 4.9 / 13.0 | 0.6 / 0.7 |
| older_90d | 0.6 / 0.6 | 1.1 / 1.1 |
| by_month | 2.6 / 2.6 | 1.6 / 2.0 (dialect) |

## Query time p50 / p95 (ms), 50,000 rows, CSV ledger

| question | sqlite | duckdb |
|---|---|---|
| top_n | 1.3 / 9.3 | 1.5 / 1.7 |
| group_sums | 16.7 / 31.6 | 2.0 / 2.4 |
| join_top | 45.7 / 66.6 | 11.5 / 13.2 |
| join_groups | 52.8 / 65.9 | 12.3 / 25.6 |
| margin | 8.9 / 13.1 | 2.6 / 22.5 |
| growth | 2.8 / 2.8 | 1.6 / 5.7 |
| weighted_top | 5.3 / 5.5 | 2.9 / 5.2 |
| what_if | 2.6 / 2.6 | 1.7 / 1.7 |
| moving_avg | 26.9 / 34.3 | 6.0 / 6.8 |
| rank | 50.9 / 51.7 | 18.6 / 24.0 |
| stddev | 25.4 / 34.5 | 0.9 / 0.9 |
| median | 21.9 / 32.0 | 2.4 / 2.6 |
| percentile | 19.7 / 23.7 | 1.2 / 1.3 |
| corr | 18.1 / 18.5 | 1.0 / 1.1 |
| regression | 46.8 / 50.7 | 1.5 / 1.5 |
| r2 | 35.0 / 54.8 | 1.0 / 1.0 |
| older_90d | 3.0 / 7.0 | 2.4 / 2.6 |
| by_month | 18.2 / 19.2 | 4.7 / 8.0 (dialect) |

## Query time p50 / p95 (ms), 500,000 rows, CSV ledger

| question | sqlite | duckdb |
|---|---|---|
| top_n | 12.9 / 14.2 | 5.9 / 7.4 |
| group_sums | 107.3 / 120.6 | 10.2 / 98.0 |
| join_top | 751.0 / 933.4 | 248.1 / 282.5 |
| join_groups | 1044.8 / 1363.7 | 192.6 / 362.9 |
| margin | 165.9 / 207.3 | 13.9 / 20.9 |
| growth | 40.5 / 58.3 | 7.2 / 7.4 |
| weighted_top | 79.0 / 110.7 | 23.0 / 101.7 |
| what_if | 41.5 / 49.5 | 12.2 / 19.4 |
| moving_avg | 454.9 / 548.6 | 56.3 / 63.0 |
| rank | 691.2 / 881.8 | 240.4 / 361.0 |
| stddev | 210.5 / 359.7 | 4.6 / 4.9 |
| median | 285.8 / 390.7 | 15.4 / 16.8 |
| percentile | 220.0 / 255.5 | 7.3 / 8.4 |
| corr | 212.9 / 222.6 | 3.0 / 5.2 |
| regression | 496.2 / 536.7 | 4.8 / 6.4 |
| r2 | 311.8 / 341.0 | 2.8 / 3.0 |
| older_90d | 45.3 / 56.1 | 8.4 / 12.2 |
| by_month | 184.7 / 277.5 | 30.2 / 37.5 (dialect) |

## 8 calls at once (each loads the 4 tables and runs join_groups)

| engine/size | wall s | call p50 s | call p95 s | errors | peak RSS +MB |
|---|---|---|---|---|---|
| sqlite/50000 | 6.95 | 5.36 | 6.94 | 0 | 209 |
| duckdb/50000 | 1.83 | 1.34 | 1.82 | 0 | 228 |
| sqlite/500000 | 88.22 | 65.65 | 88.21 | 0 | 1497 |
| duckdb/500000 | 12.10 | 9.51 | 12.08 | 0 | 898 |

## Safety probes (a query trying more than reading its own tables)

| probe | sqlite | duckdb |
|---|---|---|
| change a setting | blocked: DatabaseError: not authorized | blocked: PermissionException: not authorized: only one SELECT runs |
| delete rows | blocked: DatabaseError: not authorized | blocked: PermissionException: not authorized: only one SELECT runs |
| file written | no | no |
| install an extension | n/a | blocked: PermissionException: not authorized: only one SELECT runs |
| list files | n/a | blocked: PermissionException: Permission Error: Cannot access file "/etc/*" - file system operations are disabled by configuration |
| load an extension | blocked: OperationalError: not authorized | blocked: PermissionException: not authorized: only one SELECT runs |
| memory | stopped: DataError: string or blob too big; peak +64 MB, 0.3 s | stopped: OutOfMemoryException: Out of Memory Error: failed to allocate data of size 256.0 MiB (275.5 MiB/512.0 MiB used); peak +256 MB, 0.2 s |
| network | n/a | blocked: PermissionException: Permission Error: Cannot access file "http://127.0.0.1:9/x.csv" - file system operations are disabled by confi |
| read another file | blocked: DatabaseError: not authorized | blocked: PermissionException: Permission Error: Cannot access file "<data> |
| timeout | Error: the query failed: took too long. Only a SELECT runs,  after 5.2 s | Error: the query failed: took too long. Only a SELECT runs,  after 5.2 s |
| write a file | blocked: DatabaseError: not authorized | blocked: PermissionException: not authorized: only one SELECT runs |
| write a table | blocked: DatabaseError: not authorized | blocked: PermissionException: not authorized: only one SELECT runs |

## Value fidelity (codes, big integers, empty cells, nested values)

- sqlite: {'csv': [['007', '12345678901234567890', 1.5, None], ['010', '1', 2.0, 'x']], 'json': [['007', '12345678901234567890', '{"a":1}', 'true'], ['010', '1', '{"a":2}', 'false']]}
- duckdb: {'csv': [['007', '12345678901234567890', 1.5, None], ['010', '1', 2.0, 'x']], 'json': [['007', '12345678901234567890', '{"a":1}', 'true'], ['010', '1', '{"a":2}', 'false']]}
