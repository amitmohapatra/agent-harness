# Deep Agents' planning beside `middleware()`, on Haiku, 2026-11-15

The owner's decision: with the harness's `middleware()` on, a `create_deep_agent` graph plans
with Deep Agents' own `write_todos`, and the harness's `Planning` stays out of it; planning
is not held back for a long task; and `write_todos` works at every window size whenever it is
enabled. This page is the evidence for the change and the window sweep that checked it, on
`claude-haiku-5-5` only (prices: `benchmark.cost.prices`). Data:
[benchmark/cost/results/deepagents-planning-2026-11-15](../../../benchmark/cost/results/deepagents-planning-2026-11-15/)
(the raw files are on the `data/research-archive` branch, each listed with its sha256 in the folder's `MANIFEST.md`);
scripts: [scripts](scripts/).

## What was there (read from the code, Deep Agents 0.7.38)

- `create_deep_agent` has **no** `write_todos` of its own: its base stack is the file tools,
  the sub-agents, summarization and tool-call patching. Planning is opt-in there, LangChain's
  `TodoListMiddleware` added by the team (`examples/02_way1_deepagents/agent.py` already says
  so). Only its OpenAI Codex profile adds one.
- `middleware()` brought `automatic.Planning`, named `TodoListMiddleware`. In a Deep Agents
  graph it was therefore the only planning, offered only on a long task at a window of 32,000
  tokens or more: at 8,000, 16,000 or an unknown window, never. And a team that added Deep
  Agents' planning beside `middleware()` failed the build with LangChain's
  `AssertionError: Please remove duplicate middleware instances`.

## What changed (fix commits on `claude/cool-maxwell-w5avo8-decide2`)

1. `Planning(yields=True)` (`middleware()`'s): `create_deep_agent` binds its backend to the
   middleware it is given (`BackendConsumer`), and there `Planning` steps aside: no tool, not
   named as LangChain's, the team's `write_todos` let through, no after-model check of its own.
   `ReAct`'s planning and `create_agent`'s are unchanged.
2. `FitWindow._budget`: a window given below the model's own took the reply out of the given
   window when the model's own was said by its profile (not `ReAct`'s copy): beside
   `middleware(context_window=8_000)` (or 16,000, 32,000) a graph's Haiku had **0 tokens** for
   a request and every call failed (the first pilot: 7 of 7 turns `ERROR`, $0 spent).
3. `todos_shown` (`FitWindow`): the model reads the todo list only in the `write_todos` call
   that wrote it. When no such call is in a request — left out to fit a small window, a summary
   in its place, or the earlier turns replayed as text — the list as last written opens the
   turn's question.
4. `adapters.langgraph._carrying`: a graph with no thread of its own lost its todos between
   turns (LangChain omits `todos` from a graph's input); a carried channel the input leaves out
   is now written to the state as the turn starts.
5. `_narrowed` (`FitWindow`'s tools step): the tools it must keep took their whole room first,
   so at 8,000 tokens `write_todos` (1,116 tokens whole, 130 compacted) and Deep Agents' file
   tools held back the sales tools two long questions needed — 11 held on the first call,
   against 2 to 4 without `write_todos` — and the descriptions step then cut them anyway. They
   now take their compacted room first and are whole only where the others still fit.

Each has an offline test that failed before (named in the commits).

## The sweep

`--adapter deepagents`, Deep Agents given `TodoListMiddleware` beside `middleware()`:
`--variant todos` (a checkpointer of its own) and `todos-unthreaded` (none: the harness carries
its files and todos, and replays earlier turns as text). Cases: the `carry` set's todo cases
(`carry-03`: a list planned, then worked over two turns; `carry-04`: a list and a file over
three turns) and five long multi-step cases (`planning-02`, `planning-03`, `long-multi-01`,
`long-multi-05`, `long-chain-01`); 7 cases, 10 turns, 8 strictly scored. One run per window and
variant. "Later requests" are those of a turn after the one that first wrote the list: each
sees the list either in a `write_todos` call in its messages or in the harness's note.

| run | strict | write_todos offered | turns calling it | later requests seeing the list (call / note / not) | input tok | output tok | model calls | seconds | $ |
|---|---|---|---|---|---|---|---|---|---|
| fix-todos-w8000 | 8/8 | 36/36 | 5 | 10 / 0 / 0 | 274,015 | 14,156 | 36 | 77.1 | 0.0303 |
| fix-todos-unthreaded-w8000 | 8/8 | 37/37 | 3 | 1 / 11 / 0 | 283,538 | 16,458 | 37 | 87.9 | 0.0269 |
| fix-todos-w16000 | 8/8 | 34/34 | 5 | 12 / 0 / 0 | 395,249 | 13,231 | 34 | 72.0 | 0.0151 |
| fix-todos-unthreaded-w16000 | 7/8 | 33/33 | 3 | 1 / 9 / 0 | 375,893 | 13,896 | 33 | 73.4 | 0.0163 |
| fix-todos-w32000 | 8/8 | 34/34 | 5 | 13 / 0 / 0 | 395,172 | 13,150 | 34 | 72.3 | 0.0133 |
| fix-todos-unthreaded-w32000 | 8/8 | 32/32 | 3 | 1 / 9 / 0 | 383,786 | 14,108 | 32 | 75.0 | 0.0135 |
| fix-todos-w128000 | 8/8 | 33/33 | 5 | 10 / 0 / 0 | 396,797 | 12,707 | 33 | 67.5 | 0.0133 |
| fix-todos-unthreaded-w128000 | 8/8 | 32/32 | 3 | 1 / 9 / 0 | 360,962 | 13,233 | 32 | 72.6 | 0.0137 |
| fix-todos-wnone | 8/8 | 34/34 | 5 | 10 / 0 / 0 | 422,923 | 12,388 | 34 | 67.5 | 0.0134 |
| fix-todos-unthreaded-wnone | 8/8 | 33/33 | 3 | 1 / 9 / 0 | 373,192 | 13,229 | 33 | 75.3 | 0.0141 |

Final code: 79 of 80 strict turns, every todo turn right at every window. The one wrong turn,
`planning-03` unthreaded at 16,000, gave footwear's revenue per unit as £93.81 where it is
£93.80 (1,123,571.70 / 11,978 = 93.803; tolerance 0.006) — **model** arithmetic, not planning;
the same case was right in the nine other final runs.

`write_todos` offered: on every request of the main agent; the requests without it are the
general-purpose sub-agent's (Deep Agents gives a sub-agent no `TodoListMiddleware` unless its
spec does). The model planned with `write_todos` in every todo turn that asked for a plan, and
updated it in the turns that worked it (threaded); unthreaded, the later turns read the list
from the note and answered right without rewriting it.

### Before the narrowing fix (5), same code otherwise

| run | strict | write_todos offered | turns calling it | later requests seeing the list (call / note / not) | input tok | output tok | model calls | seconds | $ |
|---|---|---|---|---|---|---|---|---|---|
| control-main-w8000 | 8/8 | 0/35 | 0 | 0 / 0 / 0 | 225,710 | 12,749 | 35 | 69.0 | 0.0232 |
| todos-w8000 | 7/8 | 46/49 | 5 | 9 / 0 / 0 | 303,600 | 17,282 | 49 | 103.9 | 0.0265 |
| todos-unthreaded-w8000 | 6/8 | 37/37 | 3 | 1 / 9 / 0 | 203,680 | 13,271 | 37 | 76.1 | 0.0205 |
| todos-w16000 | 8/8 | 33/33 | 5 | 10 / 0 / 0 | 380,596 | 12,596 | 33 | 71.3 | 0.0150 |
| todos-unthreaded-w16000 | 8/8 | 34/34 | 3 | 1 / 9 / 0 | 386,205 | 13,956 | 34 | 74.7 | 0.0149 |
| todos-w32000 | 8/8 | 32/32 | 5 | 10 / 0 / 0 | 383,143 | 12,036 | 32 | 67.3 | 0.0126 |
| todos-unthreaded-w32000 | 8/8 | 32/32 | 3 | 1 / 8 / 0 | 373,239 | 12,790 | 32 | 72.0 | 0.0128 |
| todos-w128000 | 8/8 | 31/31 | 5 | 10 / 0 / 0 | 361,187 | 12,760 | 31 | 69.0 | 0.0125 |
| todos-unthreaded-w128000 | 8/8 | 34/34 | 3 | 1 / 9 / 0 | 407,187 | 13,657 | 34 | 75.6 | 0.0138 |
| todos-wnone | 8/8 | 31/31 | 5 | 10 / 0 / 0 | 361,352 | 12,702 | 31 | 66.5 | 0.0125 |
| todos-unthreaded-wnone | 8/8 | 32/32 | 3 | 1 / 9 / 0 | 361,395 | 13,442 | 32 | 75.0 | 0.0130 |

At 8,000 the two wrong answers (`long-chain-01` both variants, `long-multi-01` unthreaded) said
the sales data was not reachable: the tools step had held back `sales_rows` and
`sales_summary` to keep `write_todos` whole — **window** cost of the tool, fixed in (5). The
same cases with no planning at all (`control-main-w8000`: `--variant main`) were 8/8. After
the fix both 8,000 runs are 8/8 (above). The 3 requests of 49 without `write_todos` in
`todos-w8000` are `long-chain-01`'s general-purpose sub-agent.

### What was not measured live

- A todo list carried through `FitWindow` leaving out the older turns' calls (a conversation
  long enough at 8,000): the cases here fit by cutting results. The path is covered offline
  (`test_the_todo_list_stays_in_view_when_the_turns_that_wrote_it_do_not`), and the same
  note is what every unthreaded later turn read live.
- Deep Agents' own summarization (it triggers near the model's whole window, 1,000,000
  tokens for this Haiku): the note covers a summary in the calls' place by the same rule.
- Arm A (the harness's `Planning` in Deep Agents) was not rerun: the owner decided before the
  A/B. Its behaviour is read from the code above, and the earlier `carry` runs
  ([haiku-2026-11-15](../haiku-2026-11-15/README.md#2-deep-agents-multi-turn-files-and-todos-carried))
  record it: `write_todos` never offered, the todo turns answered from the conversation.

## Spend

| runs | $ |
|---|---|
| first pilot (3 items at 8,000; every call refused by the 0-token budget, fixed in 2) | 0.0000 |
| priced pilot (`carry-03`, `carry-04`, `planning-02` at 8,000) | 0.0127 |
| first sweep, 10 runs | 0.1541 |
| control without planning at 8,000 | 0.0232 |
| final code, 10 runs | 0.1699 |
| **total** (cap $2.50) | **0.360** |

At the pilot's $0.0042 a case, a run of 7 cases was priced at about $0.03; runs cost $0.0125
to $0.0303 (8,000 the most: more model calls).
