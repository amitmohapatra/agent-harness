Machine: {'python': '3.13.18', 'sqlite': '3.53.1', 'cpus': 4, 'platform': 'Linux-6.18.44-fc-v185-x86_64-with-glibc2.39'}

## Correctness (18 questions, against Python's answers)

| engine | size/format | as-is | dialect | wrong | error | not as-is |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 18 | 0 | 0 | 0 |  |
| sqlite | 10000/json | 18 | 0 | 0 | 0 |  |
| sqlite | 50000/csv | 18 | 0 | 0 | 0 |  |
| sqlite | 50000/json | 18 | 0 | 0 | 0 |  |
| sqlite | 500000/csv | 18 | 0 | 0 | 0 |  |
| sqlite | 500000/json | 18 | 0 | 0 | 0 |  |

## Load (parse + ingest of the 4 tables), peak memory, whole suite

| engine | size/format | text MB | load p50 s | suite query p50 sum ms | slowest p95 ms | peak RSS +MB |
|---|---|---|---|---|---|---|
| sqlite | 10000/csv | 1.4 | 0.13 | 72.9 | 32.0 | 9 |
| sqlite | 10000/json | 2.3 | 0.15 | 120.7 | 35.9 | 8 |
| sqlite | 50000/csv | 6.8 | 1.01 | 478.7 | 125.8 | 44 |
| sqlite | 50000/json | 11.4 | 0.77 | 474.5 | 92.7 | 36 |
| sqlite | 500000/csv | 69.9 | 9.54 | 5897.0 | 1139.5 | 398 |
| sqlite | 500000/json | 115.4 | 9.34 | 5322.9 | 1119.2 | 365 |

## Query time p50 / p95 (ms), 10,000 rows, CSV ledger

| question | sqlite |
|---|---|
| top_n | 0.3 / 0.3 |
| group_sums | 1.7 / 2.5 |
| join_top | 8.6 / 32.0 |
| join_groups | 8.6 / 10.3 |
| margin | 1.8 / 1.9 |
| growth | 0.6 / 0.7 |
| weighted_top | 1.0 / 1.3 |
| what_if | 0.5 / 0.6 |
| moving_avg | 5.4 / 7.6 |
| rank | 8.9 / 11.0 |
| stddev | 5.0 / 11.7 |
| median | 4.0 / 4.2 |
| percentile | 3.7 / 4.6 |
| corr | 3.9 / 4.3 |
| regression | 8.2 / 8.6 |
| r2 | 4.0 / 4.4 |
| older_90d | 0.6 / 0.6 |
| by_month | 6.1 / 9.0 |

## Query time p50 / p95 (ms), 50,000 rows, CSV ledger

| question | sqlite |
|---|---|
| top_n | 1.3 / 9.3 |
| group_sums | 12.4 / 12.6 |
| join_top | 58.7 / 70.1 |
| join_groups | 60.8 / 82.0 |
| margin | 12.8 / 21.0 |
| growth | 6.3 / 10.8 |
| weighted_top | 12.5 / 21.3 |
| what_if | 3.5 / 7.3 |
| moving_avg | 35.4 / 64.7 |
| rank | 66.7 / 125.8 |
| stddev | 23.3 / 27.3 |
| median | 22.6 / 29.6 |
| percentile | 26.9 / 30.4 |
| corr | 20.9 / 23.1 |
| regression | 73.9 / 81.9 |
| r2 | 20.9 / 35.0 |
| older_90d | 3.4 / 9.7 |
| by_month | 16.4 / 26.4 |

## Query time p50 / p95 (ms), 500,000 rows, CSV ledger

| question | sqlite |
|---|---|
| top_n | 21.7 / 38.2 |
| group_sums | 180.5 / 193.6 |
| join_top | 1038.6 / 1124.8 |
| join_groups | 902.8 / 1139.5 |
| margin | 131.5 / 136.6 |
| growth | 45.7 / 53.9 |
| weighted_top | 55.3 / 92.4 |
| what_if | 26.0 / 30.2 |
| moving_avg | 352.6 / 434.9 |
| rank | 758.0 / 817.6 |
| stddev | 273.5 / 373.6 |
| median | 404.9 / 442.8 |
| percentile | 339.4 / 387.1 |
| corr | 280.2 / 309.6 |
| regression | 550.8 / 613.6 |
| r2 | 290.1 / 369.8 |
| older_90d | 35.5 / 71.4 |
| by_month | 209.9 / 241.4 |

## 8 calls at once (each loads the 4 tables and runs join_groups)

| engine/size | wall s | call p50 s | call p95 s | errors | peak RSS +MB |
|---|---|---|---|---|---|
| sqlite/50000 | 7.18 | 5.68 | 7.15 | 0 | 224 |
| sqlite/500000 | 88.75 | 66.56 | 88.66 | 0 | 1504 |

## Safety probes (a query trying more than reading its own tables)

| probe | sqlite |
|---|---|
| change a setting | blocked: DatabaseError: not authorized |
| delete rows | blocked: DatabaseError: not authorized |
| file written | no |
| load an extension | blocked: OperationalError: not authorized |
| memory | ran: [[193999999]]; peak +370 MB, 0.8 s |
| read another file | blocked: DatabaseError: not authorized |
| timeout | Error: the query failed: took too long. Only a SELECT runs,  after 5.2 s |
| write a file | blocked: DatabaseError: not authorized |
| write a table | blocked: DatabaseError: not authorized |

## Value fidelity (codes, big integers, empty cells, nested values)

- sqlite: {'csv': [['007', '12345678901234567890', 1.5, None], ['010', '1', 2.0, 'x']], 'json': [['007', '12345678901234567890', '{"a": 1}', 'true'], ['010', '1', '{"a": 2}', 'false']]}
