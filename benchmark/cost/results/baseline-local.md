# Cost benchmark: granite/granite-3.3-2b

Up to 2 repeats of 10 tasks per cell (`runs`); each figure is the mean per run ± the half-width of its 95% interval (Student's t over the runs). Measured 2026-10-21 on x86_64, 4 CPUs, Linux.

| tools | variant | runs | input tokens | cached | output tokens | model calls | refused prompt tokens | seconds | tool coverage | success | errors | $ per run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | baseline | 20 | 2,654 ± 336 | 2,303 ± 348 | 182 ± 52 | 1.90 ± 0.14 | 4,537 ± 6,536 | 38.3 ± 10.2 | 1.00 ± 0.00 | 0.90 ± 0.14 | 0.10 ± 0.14 | 0.00000 ± 0.00000 |
| 10 | main | 20 | 2,794 ± 335 | 2,343 ± 330 | 223 ± 85 | 1.85 ± 0.17 | 0 ± 0 | 51.2 ± 20.6 | 1.00 ± 0.00 | 0.95 ± 0.10 | 0.00 ± 0.00 | 0.00000 ± 0.00000 |
| 50 | baseline | 6 | 17,742 ± 4,517 | 16,294 ± 6,144 | 204 ± 151 | 2.33 ± 0.54 | 0 ± 0 | 159.1 ± 94.3 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00000 ± 0.00000 |
| 50 | main | 6 | 15,046 ± 141 | 14,893 ± 70 | 168 ± 99 | 2.00 ± 0.00 | 0 ± 0 | 146.3 ± 94.6 | 1.00 ± 0.00 | 0.67 ± 0.54 | 0.33 ± 0.54 | 0.00000 ± 0.00000 |
| 100 | baseline | 20 | 0 ± 0 | — | 0 ± 0 | 1.00 ± 0.00 | 14,486 ± 35 | 0.1 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00000 ± 0.00000 |
| 100 | main | 20 | 0 ± 0 | — | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00000 ± 0.00000 |
| 500 | baseline | 20 | 0 ± 0 | — | 0 ± 0 | 1.00 ± 0.00 | 71,051 ± 29 | 0.6 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00000 ± 0.00000 |
| 500 | main | 20 | 0 ± 0 | — | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0.3 ± 0.3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00000 ± 0.00000 |

By task length (short: two parts or fewer; long: four and five parts; large: the ~100k-character export):

| tools | variant | length | input tokens | model calls | seconds | tool coverage | success |
|---|---|---|---|---|---|---|---|
| 10 | baseline | short | 2,629 ± 413 | 1.83 ± 0.25 | 26.6 ± 9.2 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 10 | baseline | long | 3,148 ± 163 | 2.00 ± 0.00 | 63.7 ± 13.1 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 10 | baseline | large | 1,323 ± 0 | 2.00 ± 0.00 | 32.1 ± 142.4 | 1.00 ± 0.00 | 0.00 ± 0.00 |
| 10 | main | short | 2,629 ± 413 | 1.83 ± 0.25 | 27.7 ± 8.1 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 10 | main | long | 3,148 ± 163 | 2.00 ± 0.00 | 64.8 ± 10.2 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 10 | main | large | 2,717 ± 17,712 | 1.50 ± 6.35 | 151.8 ± 719.7 | 1.00 ± 0.00 | 0.50 ± 6.35 |
| 50 | baseline | short | 14,964 ± 96 | 2.00 ± 0.00 | 115.8 ± 122.9 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 50 | baseline | long | 23,298 ± 0 | 3.00 ± 0.00 | 245.7 ± 30.9 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 50 | main | short | 14,964 ± 96 | 2.00 ± 0.00 | 92.4 ± 30.7 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| 50 | main | long | 15,209 ± 0 | 2.00 ± 0.00 | 254.0 ± 615.4 | 1.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | baseline | short | 0 ± 0 | 1.00 ± 0.00 | 0.1 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | baseline | long | 0 ± 0 | 1.00 ± 0.00 | 0.1 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | baseline | large | 0 ± 0 | 1.00 ± 0.00 | 0.1 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | main | short | 0 ± 0 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | main | long | 0 ± 0 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 100 | main | large | 0 ± 0 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | baseline | short | 0 ± 0 | 1.00 ± 0.00 | 0.8 ± 0.4 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | baseline | long | 0 ± 0 | 1.00 ± 0.00 | 0.5 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | baseline | large | 0 ± 0 | 1.00 ± 0.00 | 0.4 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | main | short | 0 ± 0 | 0.00 ± 0.00 | 0.4 ± 0.5 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | main | long | 0 ± 0 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 500 | main | large | 0 ± 0 | 0.00 ± 0.00 | 0.0 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 |

Priced as other models (an estimate: the same token counts at their rates, every input token at the full input price — the measured model's own cache is not theirs — and their tokenizers count the same text differently):

| tools | variant | $ per run as claude-sonnet-5-5 | $ per run as gpt-6-sol |
|---|---|---|---|
| 10 | baseline | 0.00713 ± 0.00100 | 0.00713 ± 0.00100 |
| 10 | main | 0.00781 ± 0.00125 | 0.00781 ± 0.00125 |
| 50 | baseline | 0.03752 ± 0.01050 | 0.03752 ± 0.01050 |
| 50 | main | 0.03178 ± 0.00127 | 0.03178 ± 0.00127 |
| 100 | baseline | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| 100 | main | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| 500 | baseline | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| 500 | main | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |

Prices (per million tokens: input / cached input / cache write / output):

* `granite-3.3-2b`: 0.0 / 0.0 / — / 0.0 — self-hosted on this machine: no per-token price (hardware and power not priced)
* `claude-sonnet-5-5`: 2.0 / 0.1 / 2.5 / 10.0 — https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-21)
* `gpt-6-sol`: 2.0 / 0.2 / — / 10.0 — LiteLLM 1.106.1 model_prices_and_context_window_backup.json (2026-10-20; a secondary source: openai.com was not reachable), read 2026-10-21

Model: Granite 3.3 2B Q4_K_M on llama.cpp 8baabd5 (llama-server -c 8192 --cache-ram 0 -np 1 -n 512 --temp 0.1 -t 4, the Granite 4.0 tool-call template) behind Bifrost ad8ba2b, on 4 CPUs. Cached tokens are llama.cpp's slot prefix cache, not a vendor prompt cache.

At 50 tools only the travel set (3 tasks x 2 repeats) was run: the full set at 50 tools takes about 65 minutes per variant on this machine, past the time the run was allowed.

No vendor key (ANTHROPIC_API_KEY, OPENAI_API_KEY) and no vendor provider in the gateway: no vendor model was measured.
