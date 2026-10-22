# Thread history: framework thread vs memory window (granite/granite-3.3-2b)

3 conversations of 5 turns, 2 repeat(s); recall: the later turns' answers holding the fact said in the first turns. Overhead: a turn's seconds less its model calls' (the harness's and the memory service's share).

| variant | recall | errors | input tokens, mean per turn | by turn (1-5) | output tokens | seconds per turn | model seconds | overhead p50 / p95 ms | context p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| framework | 1.00 (n=12) | 0 | 205 | 93 / 130 / 184 / 295 / 324 | 16 | 3.3 | 3.1 | 103 / 200 | 195 |
| memory | 1.00 (n=12) | 0 | 165 | 93 / 131 / 173 / 200 / 227 | 18 | 2.9 | 2.8 | 173 / 210 | 102 |
| framework-restart | 0.50 (n=12) | 0 | 136 | 93 / 136 / 102 / 158 / 190 | 18 | 2.7 | 2.6 | 92 / 128 | 122 |
| memory-restart | 0.92 (n=12) | 0 | 170 | 93 / 134 / 176 / 206 / 240 | 20 | 2.8 | 2.6 | 157 / 202 | 95 |

Worker restart (`-restart`): the memory service's worker is killed (`-9`) and started again
between turns 2 and 3, and the harness and agent are new from then on, as after a process
restart. The framework variant's thread is kept in the process, so the restart loses it.

## Before: main (memory window as text in the context), same stack, same day

| variant | recall | errors | input tokens, mean per turn | by turn (1-5) | output tokens | seconds per turn | model seconds | overhead p50 / p95 ms | context p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| memory | 1.00 (n=12) | 0 | 141 | 93 / 87 / 132 / 156 / 238 | 25 | 3.3 | 3.2 | 102 / 178 | 174 |
| memory-restart | 0.92 (n=12) | 0 | 159 | 93 / 87 / 173 / 154 / 288 | 36 | 4.3 | 4.2 | 87 / 119 | 116 |

On main the framework thread measured 1.00 recall / 202 input tokens and the memory window
0.92 / 214 (the committed run before this one).
