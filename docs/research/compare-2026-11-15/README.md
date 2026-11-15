# Which framework Trellis wraps best, on Haiku, 2026-11-15

One like-for-like run of every adapter `benchmark/cost/complex.py` wraps, on the real Haiku
model, on the product path: the memory service on (`--history memory`), agent-runs as the run
store (`--runs`, new on this branch), `trellis.contracts` between them, all from main
(harness `0047ecf`, memory service `c835844`, agent-runs `7dfe8df`, contracts `27a818d`) plus
the two fixes below. Data: [benchmark/cost/results/compare-2026-11-15](../../../benchmark/cost/results/compare-2026-11-15/);
scripts: [scripts](scripts/). Spend: about **$5.25** of the $10 cap.

## The set and the run

- The 357-turn set of [haiku-2026-11-15](../haiku-2026-11-15/README.md): its five case lists
  and windows (`long57`, `ordinary`, `sw16` at 16,000, `sw8` at 8,000, `swown`). One run is
  99 cases, 139 turns, 121 scored turns of which 119 strict (2 are judge-only). One run per
  framework; scores are x/y on what each ran.
- Same model and price table (`benchmark.cost.prices`) for every framework, its own defaults
  otherwise (temperature, reply size); step limit 12 everywhere it is taken (`ReAct` and
  `middleware()` steps, LangGraph's `recursion_limit`, OpenAI's and Claude's `max_turns` 24,
  Kimi's steps per turn).
- `scripts/run.sh <arm> ...` runs one framework; `cx_compare.py` wraps `complex.py` to also save
  what the harness accounted, what each memory request was, which runs started, and for the
  CLIs the context their scorer read. Re-scored from the saved transcripts with today's scorer
  (`scripts/analyse.py`); agent-runs and memory checked back from the stack
  (`scripts/runs_check.py`); `scripts/table.py` makes the table.
- A private stack (`scripts/scenarios/stack.sh`, as the scenario and live suites start it):
  the memory service and its worker, agent-runs and its ticker, on databases of this job on
  the 55957 cluster. As there, the memory service has no outbound network: it refused the
  model endpoint the harness registers for its own LLM work ("cannot reach ... /v1/models"),
  so memory's extraction ran on its local models only, the same for every framework.

### How each framework reached the model

| arm | what it is | route to Haiku | metered |
|---|---|---|---|
| react-plain | `ReAct` (`--variant main`) | Anthropic's own client (prompt cache) | no¹ |
| react-deep | `ReAct` on a Deep Agents backend (`--variant deep`) | same | no¹ |
| langgraph | LangChain `create_agent` + `middleware()` | same | no¹ |
| graph | `create_agent` as a LangGraph team builds it, no harness middleware | same | no¹ |
| deepagents | `create_deep_agent` + `middleware()`, own checkpointer | same | no¹ |
| openai | OpenAI Agents SDK, `OpenAIChatCompletionsModel` on an `AsyncOpenAI` client | **direct, chat completions** on Anthropic's OpenAI-compatible API (no Bifrost) | yes |
| claude | Claude Agent SDK 0.2.182, its bundled CLI | Anthropic's API, through the CLI | yes |
| kimi | Kimi Code 2.6.2 over ACP | Anthropic's API (`KIMI_MODEL_PROVIDER_TYPE=anthropic`, as the suite's Kimi target sets it) | no¹ |

¹ The harness picks Anthropic's own client only for Anthropic's own URL (any other URL gets
the OpenAI-compatible client, which drops prompt caching), and Kimi picks its provider the
same way, so a local meter would have changed their route; their figures are the harness's.
`scripts/meter.py` only forwards and records each response's usage: it renames nothing and
answers nothing. **Bifrost was not tested.** langgraph-swarm is not an adapter of `complex.py`
(only its handoff tools are recognised by the middleware), so it was not run.

## The table (one run each)

Accuracy is strict, re-scored; turns that ran while the memory service was down are not
counted (see "Memory down"). Latency excludes nothing (the max is the restart case's 600 s
timeout, see below). Tokens are per turn (input includes cached). Cost is per run at
`benchmark.cost.prices`; where metered, the provider's usage priced the same way follows.

| framework | strict right / scored | wrong turns: ours · model · scorer · framework | p50 / p95 s per turn | tokens per turn in / cached / out | $ per run | model calls per turn (provider) | tool calls per turn | failures (not SUCCESS turns) | memory: turns read · written / turns; requests per turn; context reads or transcript writes failed | agent-runs: runs / turns; spend in RUN_FINISHED = rows |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct plain | **116/118** | 1 · 1 · 0 · 0 | **6.7** / 19.2 | 32.3k / 23.1k / 1.26k | **$0.248** | 3.02 | 5.55 | 1 timeout (restart case) | 139 · 139 / 139; 27.9; 0 | 139 / 139; yes |
| ReAct + Deep Agents | **106/107** (+12 not counted, right) | 1 · 0 · 0 · 0 | 7.0 / 43.9 | 46.2k / 35.0k / 1.48k | $0.307 | 3.11 | 5.56 | 1 timeout (restart case) | 138 · 139 / 139; 27.0; 1 | 139 / 139; all but the restart case |
| LangGraph `create_agent` | **116/119** | 1 · 2 · 0 · 0 | 8.4 / 26.4 | 32.5k / 22.5k / 1.11k | **$0.247** | 2.91 | 5.06 | 1 timeout (restart case) | 139 · 139 / 139; 26.2; 0 | 139 / 139; all but the restart case |
| LangGraph, no harness middleware | 92/105 (+12 not counted) | 1 · 4 · 0 · 8 | 9.4 / 52.1 | 38.1k / **0** / 4.03k | $0.798 | 2.48 | 4.63 | 1 timeout, 1 provider refusal (prompt 1.57M tokens) | 136 · 137 / 137; 31.0; 1 | 137 / 137; all but the restart case |
| Deep Agents `create_deep_agent` | **115/117** (+2 not counted) | 1 · 0 · 1 · 0 | 10.1 / 27.8 | 39.9k / 29.6k / 1.16k | $0.265 | 2.87 | 5.02 | 1 timeout (restart case) | 138 · 139 / 139; 26.4; 1 | 139 / 139; all but the restart case |
| OpenAI Agents (direct, chat completions) | 114/117 | 1 · 2 · 0 · 0 | 8.2 / 28.8 | 43.2k / **0** / 1.18k | $0.672; provider $0.709² | 3.01 (3.26³) | 4.37 | 1 timeout (restart case) | 137 · 136 / 137; 27.2; 0 | 137 / 137; yes |
| Claude Agent SDK (real CLI) | 108/116 | 1 · 1 · 0 · 6 | 7.1 / **18.4** | 81.8k / 73.7k / 1.21k | $0.295; provider $0.323⁴ | **1.07** (3.00) | 4.53 | 7 paused on its own `Bash` | 137 · 135 / 137; 27.0; 4 (422) | 137 / 137; yes |
| Kimi Code 2.6.2 (ACP) | 105/115 | 1 · 0 · 3 · 6 | 13.4 / 38.8 | 106.1k / 97.6k / 1.36k | $0.343 | 3.05 | 4.67 | 7 paused on its own tools, 1 timeout (restart case) | 138 · 138 / 137; 27.1; 0 | 138 / 137; all but the restart case |

² The meter's total covers the cases run again (their first rows set aside): with those rows
the harness's figure is the provider's exactly (6,231,146 in / 172,108 out, $0.7092).
³ 447 responses: 441 completions plus the memory service's 6 probes of `/v1/models` (400).
⁴ Tokens: harness 11,200,445 in / 10,091,744 cached / 166,331 out, provider 11,244,325 /
10,135,223 / 166,335 (0.4 % short: the 7 paused queries, a known gap). Dollars: the benchmark
prices the CLI's 1,108,280 cache-write tokens as plain input ($0.10, not $0.125 per million).

**On one common set** — the 100 strict turns every framework ran with memory up (left out:
`long-flow-02` ×2, which only `ReAct` and `middleware()` keep a window for; `long-flow-05` and
`conversation-10`, memory down for some arm; the 5 turns judge-only or "cannot" kinds any arm's
scorer left unverifiable):

| ReAct + DA | ReAct | Deep Agents | OpenAI Agents | LangGraph | Claude | Kimi | LangGraph, no middleware |
|---|---|---|---|---|---|---|---|
| 99 | 98 | 98 | 98 | 97 | 94 | 91 | 88 |

Every one of them fails `long-flow-04` (ours, below); without it the top five are 100, 99,
99, 99, 98.

### Every wrong turn, by cause

- **Ours, every framework — `long-flow-04`** (the suite's worker restart): on agent-runs the
  run the dying worker held is not claimable when the suite's second worker asks once, 30 s
  later (agent-runs requeues it on its own schedule); nothing finishes it and the turn times
  out at 600 s, the run left `QUEUED` (its RUN_FINISHED spend is the first attempt's only:
  the one spend mismatch per arm). The in-process store, which the acceptance used, lets a
  lapsed lease be claimed at once. A suite limitation on agent-runs, not fixed. Claude fails
  it differently: the suite's simulated worker death (a `BaseException` from a tool) stops the
  SDK's in-process MCP server while the live CLI keeps going, so every later tool call fails;
  the same exception then makes `Harness.aclose` raise at the end of the set (exit 1, all rows
  written).
- **Model** — `edge-07` (ReAct, LangGraph, no-middleware LangGraph: said a 50,000-line scan
  cannot be done instead of querying the rows); `freshness-03` turn 2 (LangGraph, no-middleware,
  OpenAI, Claude: kept the 30 October stock count, not the one after the clock moved; Kimi
  paused there, below); OpenAI `cantdo-04` ("I ran k-means" with no code run);
  no-middleware `freshness-01` turn 2 and `cantdo-01` (repeated the old figure; did not say
  why).
- **Scorer** — Deep Agents and Kimi `conversation-02` turn 2, Kimi `freshness-01` turn 2 and
  `conflict-02`: the right figure, with the superseded one also named ("£10,076.60 less than
  July's £356,561.76"), which `forbidden` refuses.
- **Framework** — Claude 6 (`aggregate-04`, `long-data-08`, `sw16 long-data-02`: the CLI saved a
  large tool result to a file and reached for its own `Bash` to read it; `cantdo-01/02/04`: it
  tried to run code). Its `Bash` asks first, the suite approves only plain reads in the run's
  folder, so the turn stays paused (governance did its job; the user saw no answer). Kimi 6:
  `freshness-03` ×2, `multihop-09`, `cantdo-02/04` paused on its own tools the same way, and
  `multihop-03`, a session that said it had no harness tools and searched the disk with its own
  `Bash`/`Grep` (cause not determined). No-middleware LangGraph 8: large results went into the
  prompt whole (`long-data-01/02/03`, `long-multi-02`, `planning-03`, their `sw16` copies), one
  request of 1,565,095 tokens refused by the provider (`advdata-02`): no result clearing,
  `query_result` keep or window fitting without the harness middleware.

## Verdict, per parameter

- **Accuracy: a tie at the top, within noise.** `ReAct` + Deep Agents (99/100 common), `ReAct`,
  `create_deep_agent`, OpenAI Agents (98) and `create_agent` (97) differ by 1–2 turns, each a
  single model slip or scorer strictness. The acceptance measured `ReAct` alone at 115–119 a
  run over six runs, so one run cannot order them. Claude (94) and Kimi (91) are behind for a
  framework reason, their own shells pausing under governance on this suite, not their models.
  LangGraph without the harness middleware is last (88): the middleware is what keeps large
  results and windows in hand.
- **Latency: ReAct and Claude fastest, Kimi slowest; the middle is noise.** p50 6.7 s (ReAct),
  7.0 (ReAct + DA), 7.1 (Claude), 8.2–10.1 (OpenAI, LangGraph, Deep Agents), 13.4 (Kimi). The
  arms ran three or four at a time on a shared 4-core machine whose load went from 5 to 10
  during the day (ReAct, ReAct + DA and Claude in the first, quietest batch), so differences
  under about 30 % are not evidence. Kimi's are larger than that; Claude's p95 (18.4 s) is the
  lowest. For the two metered arms the model itself took 63 % (OpenAI) and 76 % (Claude) of the
  turn time.
- **Cost: `create_agent` and `ReAct`, $0.25 a run.** Then `create_deep_agent` $0.26, `ReAct` +
  DA $0.31, Claude $0.32, Kimi $0.34; OpenAI Agents $0.71 and LangGraph without middleware $0.80.
  The two expensive ones get **no prompt cache**: Anthropic's OpenAI-compatible layer drops it
  (OpenAI), and without the harness middleware nothing marks a cache point (graph). That is the
  route and the middleware, not the framework's loop.
- **Tokens: `create_agent` and `ReAct` the fewest (32k in, 1.1–1.3k out a turn).** The CLIs
  carry their own system prompts and tools: Claude's smallest request was 20,949 tokens (an
  OpenAI Agents request about 6,700 at the smallest), Kimi's about 35,000 a call; both are cached, so they
  cost little but fill the window. No-middleware LangGraph writes 4k output tokens a turn
  (results echoed back).
- **Overall: `create_agent` with the harness middleware, or `ReAct`.** Same accuracy as the
  best within noise, the lowest cost and tokens, middle-to-fast latency, and every governance,
  memory and agent-runs check held. `create_deep_agent` is as good at a little more latency;
  `ReAct` on Deep Agents costs 25 % more for the same score. Prefer Anthropic's own client to
  an OpenAI-compatible route on Anthropic (2.7× the cost). Claude and Kimi are the choice when
  their own tools (shell, files) are wanted, and then a team must approve those tools.

## The product path: memory and agent-runs

- **agent-runs**: every turn's run landed (runs found = turns made for every framework, Kimi
  one more for the restart case's re-claim), each with `RUN_STARTED` and `RUN_FINISHED`, 2,859
  to 3,616 events a run, and the spend in `RUN_FINISHED` equal to the row's tokens for every
  run except the restart case's (above).
- **Memory**: every framework read the memory context and wrote its transcript on at least
  135 of its 137–139 turns (counted per case against its turns). About 27 memory requests a
  turn for every framework, of which ~17 are `GET /v1/tools/approval-suggestions`: the
  governance refresh asks once for each agent the harness knows, and the suite wraps an agent
  per case. Failures: context reads that missed their 5 s read deadline while the machine was
  loaded (the turn goes on without the context: those turns are not counted); `POST /v1/verify`
  timeouts (0–5 a run); Claude's transcript writes refused with 422 in two `long-flow` cases
  (4 writes, the reason not captured).
- **Approvals and pauses**: the `long-flow` approval cases paused once and ran the approved
  write once for every framework (every harness check of those turns held, apart from the
  restart case); Claude's and Kimi's own shells were held
  for approval as configured.

### Memory down, and what was run again

The machine's memory service was briefly unhealthy several times (another job stopped
PostgreSQL at 09:22:58; it was OOM-killed at 09:36 when eight arms ran at once under a memory
limit shared by every job; 10–40 s health gaps under load later). The two runs during the
first two events were stopped and are not counted (`NOT_COUNTED` in their folders, not
committed). Afterwards every case with a request refused by an open circuit, or a context read
or transcript write that got no answer, was set aside (`*.replaced-memory-down.jsonl`) and run
again (`*md` sets) with at most two arms at once; where that happened again (ReAct + DA and
no-middleware LangGraph `long-flow-05`, Deep Agents `conversation-10`) those turns are listed
in the data and not counted — they were all right.

## Harness findings

- **Fixed, test first — `middleware(context_window=)` refused every request below the model's
  own window.** `FitWindow._budget` reserved the reply (64,000 tokens) out of the window given
  (16,000), not the model's own, leaving 0 for the input: every `sw16`, `sw8`, `swown` and
  `long-flow-02` turn of `create_agent` ended `ERROR` (`ContextTooLarge`, "more than the 0 this
  model's window holds"), 10 scored turns. `ReAct` was right because it keeps the model's own
  window beside the given one. Now the profile's own window is used
  (`src/trellis/harness/middleware.py`, test
  `test_a_window_given_below_the_models_own_reserves_the_reply_out_of_the_models_own`). The
  LangGraph rows from before the fix are kept in `langgraph-1/oldcode-window-bug/`; the cases
  were run again on the fix.
- **Fixed (benchmark) — `--runs`**: `complex.py` could not use agent-runs (the benchmark harness
  turned the run store off); `runner.harness(runs=True)` and `--runs` keep `RUNS_URL`.
- **Fixed (benchmark) — two variants shared agent ids.** `ReAct` plain and on Deep Agents were
  both `complex-react-<case>`; on one agent-runs a worker of one claimed the other's restart
  run (its tools then saw none of the first run's state). Each variant has its own id now;
  both arms' `long-flow-04` were run again.
- **Not fixed — Claude's model calls**: the adapter counts one per query result (1.07 a turn),
  the provider served 3.00 a turn.
- **Not fixed — the benchmark prices cache writes as input** (`Spend.add` takes no cache-write
  count): Claude's run is $0.028 short of the provider's price.
- **Not fixed — Claude's kept CLIs**: one process per conversation, kept 300 s idle, up to 16:
  about 3.3 GB at once in this suite (16 × ~200 MB).
- **Not fixed — the write spool is shared per machine user** (`~/.local/state/trellis`): a
  process replays another process's undelivered memory writes; 33 transcript writes of a
  stopped run were replayed by another arm and refused (403). The comparison gave each arm its
  own `XDG_STATE_HOME`.

## Fairness notes (what could not be made equal)

- The CLIs bring their own system prompts and built-in tools (Claude: `Bash`, `Write`...;
  Kimi: its shell and file tools); the suite's instructions reach Kimi as an `AGENTS.md`.
- Windows: `graph`, OpenAI, Claude and Kimi take no window, so `sw16`, `sw8` and `advconv-05`
  ran with their own context (scored all the same), and `long-flow-02` (×2) was not applicable.
- Routes: OpenAI Agents on Anthropic's OpenAI-compatible API has no prompt cache.
- Latency was measured under different machine load for different arms (above).
- One run per framework: no second or third run, because the top five are within 1–2 turns,
  less than one framework's own run-to-run spread, and a second run of the five (about $1.80)
  would not order them.

## Not measured

- Bifrost (not on the machine); langgraph-swarm (not a `complex.py` adapter).
- The provider's usage for the LangChain arms and Kimi (a meter changes their route); the
  harness's figures were exact against the provider on the OpenAI-compatible route in the
  acceptance and for OpenAI Agents here.
- The memory service's own LLM work (it has no outbound network on this stack).
- Runs 2 and 3, and a variance for latency or cost.

## Spend

| what | $ |
|---|---|
| pilots: memory off (8 arms, 3 cases), then on Anthropic's own client (5 arms) | 0.23 |
| memory-off runs stopped by the scope change, and LangChain arms stopped when the meter turned out to change their route | 0.70 |
| memory pilots (one with memory down, not counted) | 0.21 |
| first product-path run, stopped when the memory service was OOM-killed (not counted) | 0.71 |
| the counted run of all eight, with every case run again | 3.40 |
| **total** | **about 5.25** |
