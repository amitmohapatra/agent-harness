Machine: {'python': '3.13.18', 'sqlite': '3.53.1', 'cpus': 4, 'platform': 'Linux-6.18.44-fc-v185-x86_64-with-glibc2.39'}

## Correctness (18 questions, against Python's answers)

| engine | size/format | as-is | dialect | wrong | error | not as-is |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 10000/csv | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 10000/csv | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| pandasql | 10000/csv | 14 | 0 | 0 | 4 | stddev (error), corr (error), regression (error), r2 (error) |
| sqlite | 10000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 10000/json | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 10000/json | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| pandasql | 10000/json | 14 | 0 | 0 | 4 | stddev (error), corr (error), regression (error), r2 (error) |
| sqlite | 50000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 50000/csv | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 50000/csv | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| pandasql | 50000/csv | 14 | 0 | 0 | 4 | stddev (error), corr (error), regression (error), r2 (error) |
| sqlite | 50000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 50000/json | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 50000/json | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| pandasql | 50000/json | 14 | 0 | 0 | 4 | stddev (error), corr (error), regression (error), r2 (error) |
| sqlite | 500000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 500000/csv | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 500000/csv | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| sqlite | 500000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 500000/json | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 500000/json | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| sqlite | 2000000/csv | 18 | 0 | 0 | 0 |  |
| duckdb | 2000000/csv | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 2000000/csv | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |
| sqlite | 2000000/json | 18 | 0 | 0 | 0 |  |
| duckdb | 2000000/json | 16 | 2 | 0 | 0 | percentile (dialect), older_90d (dialect) |
| polars | 2000000/json | 12 | 5 | 1 | 0 | margin (wrong), percentile (dialect), regression (dialect), r2 (dialect), older_90d (dialect), by_month (dialect) |

## Load (parse + ingest of the 4 tables), peak memory, whole suite

| engine | size/format | text MB | load p50 s | suite query p50 sum ms | slowest p95 ms | peak RSS +MB |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 1.4 | 0.11 | 89.8 | 43.0 | 9 |
| duckdb | 10000/csv | 1.4 | 0.12 | 29.6 | 5.7 | 75 |
| polars | 10000/csv | 1.4 | 0.06 | 19.0 | 3.1 | 63 |
| pandasql | 10000/csv | 1.4 | 0.02 | 686.4 | 243.1 | 19 |
| sqlite | 10000/json | 2.3 | 0.11 | 70.1 | 23.5 | 8 |
| duckdb | 10000/json | 2.3 | 0.10 | 28.1 | 4.8 | 74 |
| polars | 10000/json | 2.3 | 0.07 | 20.7 | 3.9 | 77 |
| pandasql | 10000/json | 2.3 | 0.03 | 695.1 | 197.7 | 19 |
| sqlite | 50000/csv | 6.8 | 0.65 | 546.6 | 304.9 | 44 |
| duckdb | 50000/csv | 6.8 | 0.33 | 75.1 | 21.6 | 86 |
| polars | 50000/csv | 6.8 | 0.20 | 50.1 | 11.8 | 139 |
| pandasql | 50000/csv | 6.8 | 0.14 | 4278.9 | 612.4 | 65 |
| sqlite | 50000/json | 11.4 | 0.69 | 465.6 | 362.0 | 37 |
| duckdb | 50000/json | 11.4 | 0.27 | 69.8 | 23.5 | 133 |
| polars | 50000/json | 11.4 | 0.29 | 55.8 | 25.7 | 208 |
| pandasql | 50000/json | 11.4 | 0.19 | 3580.9 | 779.7 | 78 |
| sqlite | 500000/csv | 69.9 | 9.37 | 5642.6 | 1625.8 | 398 |
| duckdb | 500000/csv | 69.9 | 2.15 | 563.2 | 179.3 | 399 |
| polars | 500000/csv | 69.9 | 2.80 | 644.8 | 196.3 | 585 |
| sqlite | 500000/json | 115.4 | 6.99 | 5075.8 | 1374.0 | 365 |
| duckdb | 500000/json | 115.4 | 2.08 | 576.6 | 165.5 | 391 |
| polars | 500000/json | 115.4 | 2.80 | 348.7 | 82.7 | 888 |
| sqlite | 2000000/csv | 284.6 | 32.99 | 19282.6 | 3344.5 | 1661 |
| duckdb | 2000000/csv | 284.6 | 6.60 | 1380.3 | 568.9 | 1314 |
| polars | 2000000/csv | 284.6 | 8.01 | 1589.9 | 500.9 | 1622 |
| sqlite | 2000000/json | 466.6 | 40.06 | 28427.8 | 4939.7 | 1479 |
| duckdb | 2000000/json | 466.6 | 8.01 | 1185.4 | 301.1 | 1707 |
| polars | 2000000/json | 466.6 | 9.64 | 1628.7 | 536.0 | 2867 |

## Query time p50 / p95 (ms), 10,000 rows, CSV ledger

| question | sqlite | duckdb | polars | pandasql |
|---|---|---|---|---|
| top_n | 0.3 / 0.3 | 1.1 / 1.3 | 0.7 / 0.9 | 34.4 / 58.7 |
| group_sums | 1.6 / 5.7 | 1.6 / 1.8 | 0.9 / 1.0 | 46.1 / 88.4 |
| join_top | 7.7 / 12.6 | 4.3 / 4.7 | 2.3 / 2.9 | 74.3 / 114.9 |
| join_groups | 12.2 / 43.0 | 5.0 / 5.7 | 2.5 / 2.7 | 75.6 / 111.7 |
| margin | 1.7 / 5.7 | 1.5 / 1.5 | – / – (wrong) | 37.8 / 79.5 |
| growth | 0.6 / 3.6 | 0.9 / 1.0 | 0.7 / 0.8 | 38.3 / 59.8 |
| weighted_top | 1.3 / 5.3 | 1.6 / 1.9 | 1.3 / 1.6 | 37.0 / 43.4 |
| what_if | 0.5 / 4.6 | 1.0 / 1.0 | 0.5 / 0.7 | 37.3 / 52.8 |
| moving_avg | 9.4 / 13.1 | 2.0 / 2.2 | 0.8 / 0.9 | 64.6 / 243.1 |
| rank | 16.9 / 20.7 | 4.1 / 4.3 | 3.0 / 3.1 | 42.6 / 100.2 |
| stddev | 6.5 / 8.0 | 0.5 / 0.7 | 0.1 / 0.2 | – / – (error) |
| median | 4.2 / 8.0 | 1.2 / 1.2 | 0.8 / 0.9 | 51.3 / 78.8 |
| percentile | 3.8 / 10.7 | 0.6 / 0.7 (dialect) | 0.5 / 0.8 (dialect) | 63.8 / 136.6 |
| corr | 5.4 / 7.8 | 0.6 / 0.7 | 0.3 / 0.5 | – / – (error) |
| regression | 7.7 / 11.6 | 0.7 / 1.1 | 0.5 / 0.7 (dialect) | – / – (error) |
| r2 | 6.7 / 9.2 | 0.6 / 0.7 | 0.3 / 0.5 (dialect) | – / – (error) |
| older_90d | 0.6 / 3.9 | 0.7 / 0.8 (dialect) | 1.5 / 1.6 (dialect) | 31.9 / 57.4 |
| by_month | 2.8 / 6.5 | 1.4 / 1.4 | 2.4 / 2.5 (dialect) | 51.2 / 73.2 |

## Query time p50 / p95 (ms), 50,000 rows, CSV ledger

| question | sqlite | duckdb | polars | pandasql |
|---|---|---|---|---|
| top_n | 1.3 / 5.5 | 1.5 / 1.7 | 1.0 / 1.5 | 339.9 / 411.2 |
| group_sums | 12.6 / 19.0 | 2.0 / 2.3 | 1.4 / 1.6 | 342.8 / 400.6 |
| join_top | 65.6 / 154.5 | 12.2 / 13.0 | 6.0 / 7.7 | 476.9 / 612.4 |
| join_groups | 63.6 / 108.4 | 13.1 / 21.6 | 6.9 / 7.7 | 521.1 / 571.1 |
| margin | 13.0 / 18.0 | 2.4 / 2.7 | – / – (wrong) | 234.0 / 283.9 |
| growth | 2.7 / 6.9 | 1.5 / 2.3 | 1.5 / 1.6 | 207.7 / 306.0 |
| weighted_top | 5.9 / 10.7 | 3.0 / 3.1 | 3.4 / 3.6 | 268.0 / 319.1 |
| what_if | 2.7 / 4.1 | 1.6 / 1.7 | 1.0 / 1.1 | 286.2 / 326.9 |
| moving_avg | 43.8 / 95.0 | 6.1 / 7.2 | 1.4 / 1.7 | 293.8 / 443.9 |
| rank | 84.6 / 304.9 | 19.3 / 21.3 | 8.8 / 10.2 | 291.0 / 577.1 |
| stddev | 22.8 / 25.7 | 0.9 / 1.0 | 0.2 / 0.3 | – / – (error) |
| median | 22.8 / 66.2 | 2.7 / 3.6 | 1.5 / 1.6 | 354.9 / 492.4 |
| percentile | 37.4 / 79.2 | 1.1 / 1.2 (dialect) | 1.5 / 1.5 (dialect) | 198.3 / 255.6 |
| corr | 39.1 / 43.2 | 1.1 / 1.2 | 0.5 / 1.5 | – / – (error) |
| regression | 77.5 / 151.0 | 1.5 / 1.7 | 0.7 / 0.8 (dialect) | – / – (error) |
| r2 | 27.0 / 155.6 | 1.0 / 1.7 | 0.5 / 0.6 (dialect) | – / – (error) |
| older_90d | 3.1 / 37.2 | 0.8 / 1.0 (dialect) | 5.5 / 6.3 (dialect) | 253.9 / 276.3 |
| by_month | 21.2 / 66.0 | 3.3 / 3.4 | 8.4 / 11.8 (dialect) | 210.3 / 353.7 |

## Query time p50 / p95 (ms), 500,000 rows, CSV ledger

| question | sqlite | duckdb | polars |
|---|---|---|---|
| top_n | 13.4 / 14.8 | 10.2 / 29.3 | 4.4 / 5.3 |
| group_sums | 204.2 / 271.0 | 14.4 / 61.2 | 4.4 / 5.9 |
| join_top | 1231.9 / 1625.8 | 120.4 / 155.0 | 128.3 / 174.0 |
| join_groups | 960.3 / 1121.9 | 125.9 / 163.3 | 116.1 / 172.1 |
| margin | 127.9 / 177.1 | 13.7 / 15.3 | – / – (wrong) |
| growth | 32.1 / 39.0 | 7.1 / 7.3 | 8.5 / 11.9 |
| weighted_top | 54.7 / 67.2 | 10.8 / 13.3 | 27.9 / 39.2 |
| what_if | 26.6 / 50.3 | 5.8 / 6.1 | 5.1 / 7.4 |
| moving_avg | 371.2 / 462.2 | 50.2 / 55.7 | 18.3 / 20.2 |
| rank | 689.0 / 691.6 | 140.4 / 179.3 | 162.3 / 196.3 |
| stddev | 219.7 / 265.5 | 3.9 / 4.5 | 0.8 / 1.1 |
| median | 265.1 / 365.2 | 11.0 / 12.9 | 9.6 / 10.9 |
| percentile | 219.6 / 238.1 | 7.8 / 10.0 (dialect) | 12.8 / 22.9 (dialect) |
| corr | 251.2 / 262.1 | 4.8 / 5.4 | 1.8 / 2.7 |
| regression | 494.5 / 607.9 | 8.4 / 8.5 | 2.8 / 3.0 (dialect) |
| r2 | 273.2 / 337.0 | 5.0 / 5.1 | 1.8 / 2.3 (dialect) |
| older_90d | 33.3 / 105.5 | 1.6 / 1.8 (dialect) | 55.6 / 72.7 (dialect) |
| by_month | 174.6 / 183.6 | 21.8 / 23.5 | 84.6 / 135.9 (dialect) |

## Query time p50 / p95 (ms), 2,000,000 rows, CSV ledger

| question | sqlite | duckdb | polars |
|---|---|---|---|
| top_n | 56.5 / 57.0 | 13.5 / 14.1 | 10.2 / 13.1 |
| group_sums | 526.3 / 537.4 | 24.1 / 31.5 | 18.7 / 22.4 |
| join_top | 3116.8 / 3189.6 | 216.5 / 234.2 | 207.7 / 246.0 |
| join_groups | 3266.9 / 3344.5 | 219.3 / 228.8 | 198.6 / 211.7 |
| margin | 505.2 / 506.5 | 36.3 / 40.3 | – / – (wrong) |
| growth | 112.3 / 112.4 | 18.0 / 18.1 | 20.4 / 20.7 |
| weighted_top | 208.8 / 209.2 | 38.2 / 48.5 | 55.1 / 65.8 |
| what_if | 102.4 / 102.6 | 21.6 / 30.5 | 12.3 / 12.8 |
| moving_avg | 1236.3 / 1240.9 | 146.0 / 148.4 | 106.8 / 118.4 |
| rank | 2375.5 / 2404.7 | 441.6 / 568.9 | 460.7 / 500.9 |
| stddev | 830.4 / 895.7 | 9.3 / 10.4 | 2.0 / 3.0 |
| median | 1128.3 / 1129.9 | 49.0 / 50.3 | 36.7 / 38.3 |
| percentile | 940.7 / 1009.2 | 28.6 / 28.6 (dialect) | 24.2 / 26.9 (dialect) |
| corr | 958.0 / 958.5 | 14.2 / 16.7 | 6.8 / 7.0 |
| regression | 2035.0 / 2088.5 | 28.0 / 28.0 | 5.5 / 6.7 (dialect) |
| r2 | 945.2 / 947.5 | 13.2 / 13.9 | 3.0 / 4.9 (dialect) |
| older_90d | 141.0 / 153.3 | 4.0 / 4.6 (dialect) | 179.5 / 224.3 (dialect) |
| by_month | 797.0 / 803.4 | 59.0 / 68.4 | 241.7 / 434.3 (dialect) |

## 8 calls at once (each loads the 4 tables and runs join_groups)

| engine/size | wall s | call p50 s | call p95 s | errors | peak RSS +MB |
|---|---|---|---|---|---|
| sqlite/50000 | 6.21 | 4.92 | 6.21 | 0 | 238 |
| duckdb/50000 | 1.04 | 1.03 | 1.04 | 0 | 390 |
| polars/50000 | 0.98 | 0.92 | 0.93 | 0 | 381 |
| sqlite/500000 | 74.98 | 60.19 | 74.97 | 4 | 1548 |
| duckdb/500000 | 7.89 | 7.61 | 7.89 | 0 | 1958 |
| polars/500000 | 12.97 | 12.00 | 12.82 | 0 | 1940 |

## Safety probes (a query trying more than reading its own tables)

| probe | sqlite | duckdb | polars | pandasql |
|---|---|---|---|---|
| change a setting | blocked: DatabaseError: not authorized | blocked: PermissionError: only one SELECT runs | n/a | ALLOWED (it ran; no rows to read) |
| delete rows | blocked: DatabaseError: not authorized | blocked: PermissionError: only one SELECT runs | ALLOWED: [] | ALLOWED (it ran; no rows to read) |
| file written | no | no | no | ALLOWED |
| install an extension | n/a | blocked: PermissionError: only one SELECT runs | n/a | n/a |
| list files | n/a | blocked: PermissionException: Permission Error: Cannot access file "/etc/*" - file system operations are disabled by configuration | n/a | n/a |
| load an extension | blocked: OperationalError: not authorized | blocked: PermissionError: only one SELECT runs | n/a | blocked: DatabaseError: Execution failed on sql 'SELECT load_extension('/nonexistent')': (sqlite3.OperationalError) not authorized |
| memory | ran: [[193999999]]; peak +370 MB, 0.7 s | stopped: OutOfMemoryException: Out of Memory Error: failed to allocate data of size 256.0 MiB (274.0 MiB/488.2 MiB used); peak +258 MB, 0.2 s | no memory limit setting (not run) | no memory limit setting (not run) |
| network | n/a | blocked: PermissionException: Permission Error: Cannot access file "http://127.0.0.1:9/x.csv" - file system operations are disabled by confi | ALLOWED (the connection was tried: ConnectionRefusedError: object-store error: Generic HTTP err) | n/a |
| read another file | blocked: DatabaseError: not authorized | blocked: PermissionException: Permission Error: Cannot access file "<scratch>| ALLOWED: [['hunter2']] | ALLOWED (it ran; no rows to read) |
| timeout | Error: the query failed: took too long. Only a SELECT runs,  after 5.1 s | stopped (InterruptException) after 2.0 s | no way to interrupt a running query (not run) | no way to interrupt a running query (not run) |
| write a file | blocked: DatabaseError: not authorized | blocked: PermissionError: only one SELECT runs | n/a | ALLOWED (it ran; no rows to read) |
| write a table | blocked: DatabaseError: not authorized | blocked: PermissionError: only one SELECT runs | ALLOWED: [['CREATE TABLE z']] | ALLOWED (it ran; no rows to read) |

## Value fidelity (codes, big integers, empty cells, nested values)

- sqlite: {'csv': [['007', 1.2345678901234567e+19, 1.5, None], ['010', 1.0, 2.0, 'x']], 'json': 'OverflowError: Python int too large to convert to SQLite INTEGER'}
- duckdb: {'csv': [['007', 12345678901234567890, 1.5, None], ['010', 1, 2.0, 'x']], 'json': [['007', 12345678901234567890, {'a': 1}, True], ['010', 1, {'a': 2}, False]]}
- polars: {'csv': [[7, 12345678901234567890, 1.5, None], [10, 1, 2.0, 'x']], 'json': [['007', 12345678901234567890, {'a': 1}, True], ['010', 1, {'a': 2}, False]]}
- pandasql: {'csv': 'ValueError: Unsigned 64 bit integer datatype is not supported', 'json': 'ValueError: Unsigned 64 bit integer datatype is not supported'}
