# Holistic Haiku evaluation: harness and memory (2026-10-31)

**Model.** `claude-haiku-5-5` and nothing else. It answered as the agent on every adapter, did memory's own model work (summaries, extraction, the grounding judge), and served as the Claude CLI's model with every alias pinned to it. The endpoint was Anthropic's OpenAI-compatible `/v1` route (`LLM_GATEWAY=openai-compatible`); the Claude Agent SDK used the CLI's native API. The key was passed through the environment only.

**Judge.** The session's own big model judged per `docs/research/EVAL_PROTOCOL.md`. Packets were blind. Each turn got two verdicts: answer correct, and context sufficient. The judge also wrote its own reference answer from the same captured input (computing allowed, no tools) and scored it with the same rule. Every failed and every free-text turn was judged by hand. A turn that passed the strict scorer is taken as correct and sufficient (spot-checked).

**Deployment.** No sandbox and no Code Mode (the default deployment). The only compute available was `query_result` and `read_file` (#66). Across every run, the tools offered were: the 14 retail tools, the memory tools, `tool_search`, `query_result`, `read_file`, and Deep Agents' scratch-file tools plus `task`. `execute` was never offered, and neither was any Code Mode tool.

## Stack

Everything ran privately under `/var/tmp/hstack`:

- **PostgreSQL 16** on port 55432, with its own data directory.
- **Redis** on port 56379.
- **Memory service** from `origin/main` on port 58191, with its frozen ONNX models.
- **Arm D memory instance**: memory main plus `explain2`, on port 58192, with its own database.
- **Harness**: `origin/main` plus #66 (later merged), plus #67 (the complex suite).

What I could not run as specified:

- **No Qdrant server and no OpenFGA.** GitHub release downloads are refused by the proxy (403), and there is no Docker daemon. I used the memory repo's own offline stand-ins (`Overrides`): qdrant-client local mode on a private path, and in-process authorization. Local mode allows one process, so memory jobs ran inline in the API process; there was no separate worker.
- **No local tooling proxy was needed.**
- **An operator slip on my side (private stack only).** While restarting the memory API for the after-run, I wrongly took the private Postgres for stopped and deleted its `postmaster.pid` while it was still running. I then stopped it with a fast shutdown and started it again. The shutdown was clean with no recovery needed and no data was lost. Nothing outside `/var/tmp/hstack` was touched.

## 1. Complex suite (81 conversations, 106 scored turns): before and after the fixes

Each cell is **strict / judged / big-model reference**. "Before" is main + #66 + #67. "After" is the same with PRs #71, #73, #76, #77 and #79 applied, with only the failed cases re-run (failures-only, as the brief says) and the freshness category re-run.

| category | ReAct before → after | create_agent before → after | Deep Agents before → after | OpenAI Agents before → after | Claude Agent SDK before → after (judged only) |
|---|---|---|---|---|---|
| multihop | 100/100/100 → 100/100/100 | 100 → 100 | 90/90/90 → 100/100/100 | 100 → 100 | 90 → 90 |
| aggregate | 88/88/100 → 100/100/100 | 88/88/100 → 100 | 62/62/100 → 100 | 75/75/100 → 100 | 88 → 100 |
| formula | 100 → 100 | 100 → 100 | 83/83/100 → 100 | 83/83/100 → 100 | 83 → 100 |
| stats | 88/100/100 → 100 | 75/75/100 → 100 | 100 → 100 | 88/100/100 → 100 | 75 → 100 |
| whatif | 100 → 100 | 100 → 100 | 100 → 100 | 100 → 100 | 100 → 100 |
| conversation | 91/96/100 → 100 | 100 → 100 | 91/96/100 → 100 | 91/100/100 → 100 | 91 → 96 |
| explain | 70/92/92 → 80/92/92 | 90/92/92 → 90/92/92 | 60/77/85 → 90/92/92 | 70/85/92 → 90/92/92 | 85 → 92 |
| freshness | 67/78/78 → 78/89/100 | 67/78/78 → 89/100/100 | 67/78/78 → 89/100/100 | 67/67/78 → 78/89/100 | 78 → 89 |
| ambiguity | 83/83/100 → 100 | 83/83/100 → 100 | 100 → 100 | 83/83/100 → 100 | 100 → 100 |
| nodata | 50/100/100 → 50/100/100 | 25/100/100 → 50/100/100 | 25/75/100 → 25/100/100 | 25/100/100 → 50/100/100 | 100 → 100 |
| conflict, edge, planning | 100, 86, 100 → 100 | 100, 100, 80 → 100 | 100 → 100 | 100, 86/100, 80 → 100 | 100, 100, 80 → 100, 100, 80 |
| **all** | **86 / 93 / 97 → 94 / 98 / 99** | **88 / 92 / 97 → 96 / 99 / 99** | **83 / 89 / 95 → 95 / 99 / 99** | **83 / 91 / 97 → 95 / 98 / 99** | **79 strict, 89 judged, 93 ref → 94 judged, 96 ref** |

**Gap to the big-model reference (reference minus judged, all turns).**

| adapter | before (points) | after (points) |
|---|---|---|
| ReAct | 4 | 1 |
| create_agent | 5 | 0 |
| Deep Agents | 6 | 0 |
| OpenAI Agents | 6 | 1 |
| Claude Agent SDK | 4 | 2 |

Before the fixes, the gap was concentrated in aggregate, formula and stats (up to 38 points on Deep Agents). After them, every harness-controlled category has a gap of 0 except explain: the reference also fails the cross-chat case, so judged and reference are equal there.

**Turns where the reference was right and Haiku was wrong, by reason (before, all adapters).**

| reason | turns | harness-side help |
|---|---|---|
| Arithmetic in the head: counting or summing 200 to 1,900 rows, joins, a Pearson r, quartiles, a weighted mean | 14 | #73, exact compute through `query_result` on tables read whole. All 14 were right on re-run. |
| Wrong tool: a needed tool narrowed away (`find_sku`, `sales_summary`), or the model disowns a tool it had used | 9 | #77 |
| Stale value used | 6 | #79, plus #66's latest change |
| Stall ended with no answer | 2 | #76 |
| Missing decomposition: gave up before fetching | 1 | Fixed by the re-run. The remaining help is a data map saying what can be queried. |
| Format or copy slip ("SKU-0034", 93.79 for 93.80) | 2 | None. These are model gaps. |

The strict scorer was wrong on 3–9 turns per adapter. Those were all correct answers it could not read: a date read as a made-up figure, "six" written in words, U+2212 used as the minus sign, or an earlier value mentioned in passing. U+2212 is fixed in #74; the rest are listed in that PR.

**Made-up rate (judged turns with an unsupported claim).**

| adapter | before | after |
|---|---|---|
| ReAct | 1.9% | 0.9% |
| create_agent | 0.9% | 0% |
| Deep Agents | 5.7% | 0.9% |
| OpenAI Agents | 2.8% | 0.9% |

The made-up claims were mostly "that tool is not in my tool list" (caused by narrowing) and "no tool can do X".

**Harness gaps versus model gaps (protocol diagnosis, all four LangChain/OpenAI adapters).**

| | before | after |
|---|---|---|
| harness gaps | 18 | 4 |
| model gaps | 17 | 2 |
| memory gaps | 2 | 1 |

The 4 harness gaps left after the fixes are all `explain-08`, the cross-chat question "which tool did you use in our other chat". The other chat's steps cannot be reached: memory search returns its question and answer only, and it found another conversation of the same user. Every arm fails it, D and D′ included.

**Claude Agent SDK.** Strict accuracy was 79% before. Its remaining gaps are specific to the adapter:

- Tools from an earlier turn's MCP server show as "disconnected" on the resumed session: conversation-02, explain-06, freshness-04.
- The CLI defers the harness's MCP tools behind ToolSearch, and Haiku sometimes stops after loading their schemas (multihop-02).
- Built-in Write and Bash are on: in cantdo-01 it wrote an HTML chart to the server's directory, and the Bash or Write requests in cantdo-02 to 04 paused the run for approval.
- The CLI logs `unrecognized_model` for `claude-haiku-5-5`, but serves it.

**Controls.** I re-ran 18 cases that had passed before, on the fixed tree, with ReAct and OpenAI Agents. ReAct was right on 22/23 turns judged: one model copy slip ("SKU-0034"). OpenAI Agents was right on 22/23 judged, with one new failure, planning-01: a memory note that the Claude run had written for the same user ("compute using a script (python3 or awk)… ToolSearch") was pushed as a standing rule, and OpenAI Agents then called a non-existent `bash`. The SDK fails the whole run on an unknown tool. That is a memory gap (agent-written notes stored as user rules) plus an OpenAI-adapter gap. It is not a regression from the fixes.

**Data and compute group** (aggregate, formula, stats, whatif, planning, huge/edge results; no sandbox, no Code Mode).

| | ReAct | create_agent | Deep Agents | OpenAI Agents | Claude |
|---|---|---|---|---|---|
| before, judged | 94% | 88% | 85% | 88% | 85% |
| after, judged | 100% | 100% | 100% | 100% | 97% |

After the fixes, every one of these cases went through `query_result` or `read_file`.

**Cannot-do group** (new, `cantdo-01` to `cantdo-04`: a chart, a custom Python script, an Excel file, a k-means clustering). Correct only when the answer says it cannot do this here, says why, offers what it can do instead, and makes nothing up.

| ReAct | create_agent | Deep Agents | OpenAI Agents | Claude |
|---|---|---|---|---|
| 4/4 | 4/4 | 3/4 | 4/4 | 0/4 |

- Deep Agents saved a CSV to its own scratch path and offered it.
- Claude wrote a real HTML file to the server, and three of its runs paused on approval.
- On the computable ones (the Python standard deviation), the models computed the equivalent with `query_result` and said the script itself was not run.
- The first scorer version was too narrow ("couldn't", "no Python runner"). It was revised, with every verdict checked by hand, in #74.

### 1b. After on main: one full ReAct run (85 conversations, 110 scored turns, cannot-do included)

The tree was `origin/main` after #71, #73 and #77 merged, plus the complex branch (#67, with #74 merged into it), plus the green heads of #76 and #79. Memory was on `origin/main`. Every case was run, not only the failures. Every non-passing turn was judged by hand (`judge/react-after2.json`).

| | before (main + #66 + #67) | after on main (fresh bench user) |
|---|---|---|
| strict / judged / reference | 86 / 93 / 97 | **92 / 99 / 100** |
| wrong turns (judged) | 7 of 106 | **1 of 110** (ambiguity-03: assumed a parka from a saved note instead of asking) |
| made-up rate | 1.9% | **0%** |
| aggregate, formula, stats, whatif, edge, planning | 88–100 | 100 on every one |
| conversation / freshness / nodata | 96 / 78 / 100 | 100 / 100 / 100 |
| cannot-do | – | 4/4 |
| spend | $0.333 | $0.302 |
| input per turn (mean / median) | 18.0k / 12.8k; 2.62 model calls | 22.0k / 14.9k; 2.62 model calls |
| latency p50 / p95 | 6.1 s / 16.5 s | 6.2 s / 15.2 s |

Tokens per layer, ReAct, all requests (before → after): tool schemas 1.05M → 1.39M (#77 offers all ~20 tools below 30), memory context 516k → 348k, tool results 363k → 690k (tables read whole, #73), instructions 42k → 42k, replayed history 36k → 40k.

**A first after-run was confounded, and I re-ran it.** The suite uses one fixed user (`trellis-bench`) for every run, so memory carried about ten earlier runs. Two effects showed up:

- **A standing rule from an earlier session's question.** The profile had picked up "Compute one exact number with a script (python3)… use bash", from my earlier cannot-do and probe runs. Because of it, `edge-07` refused ("I have no way to run a script"), and `explain-04` told the user "your instructions call for a python3 script".
- **Self-doubt on explain turns.** The model's account of its earlier answer turned into "each result carried an out-of-date notice, so I may have used stale data" (explain-03, 04, 05). The cause is that main's live-by-default freshness (`6320c3d`, after my baseline) replays even a seconds-old lookup as `out of date`.

That run scored 90 strict ($0.40). With a fresh user (`run.USER` patched in the process, no code change), both effects went away. The explain turns still mention the marks, but they answer correctly. **Both are memory/harness findings, listed in §7.** The suite should use a fresh user per run.

### 1c. Coverage against the user's six asks (clean after-run on main, ReAct, judged)

| ask | cases in the suite (multi-turn unless noted) | score | cost | input/turn | gap |
|---|---|---|---|---|---|
| 1. Past chats | explain-07 (recall an earlier chat's reorder qty and how it was worked out), explain-08 (which tool the other chat used); in-chat recall 7 turns back (conversation-08); in-chat corrections where the latest wins (conversation-02, 06, 10) | 2/2 cross-chat, 23/23 conversation | $0.049 + part of explain | 15k | **Not covered:** 2+ earlier chats, decisions or approvals, a correction made in another chat, other-user isolation |
| 2. Analytics | 48 cases: multihop 10, aggregate 8, formula 6, stats 8, what-if 4, planning 5, edge 7 (filters, group sums, joins over 2–3 tools, ranking, % change, scenarios; truth computed in Python by the suite) | 48/48 | $0.164 | 17k–65k | none found |
| 3. Explanation | explain-01..06 (source, tools, steps and call count, why, how sure, what changed) on the turn just done; explain-07/08 on an earlier chat | 13/13 | $0.031 | 14k | not covered: explaining a turn 2+ back in the same chat |
| 4. Mixed | only partial: conversation-08 (recall turn 1, then compute a new % against it), conversation-10 (redo a calculation with a changed parameter), conversation-09 (compare, then switch metric), freshness-03 (re-fetch, then compute), explain-07 (recall plus explain) | 100% of these | – | – | **Mostly missing:** "approved last week vs today, explain why", "redo my earlier chat's calculation on new data plus how sure", a category switch mid-chat |
| 5. Can't-do | nodata 4 (single-turn) + cantdo 4 | 8/8 | $0.026 | 20k–35k | none found |
| 6. Unclear → ask | ambiguity 4 (2 asks-first, 2 answered follow-ups) | 5/6 | $0.011 | 15k | ambiguity-03 assumed from a memory note (model) |

Asks 1 and 4 need new cases: multi-chat setups, recorded approvals, and data that changes between chats. That is suite work (truth generation plus the scorer). It was not done in this run; see the summary for the cost estimate.

## 2. Explain arms A / D / D′ (explain cases, ReAct, memory on, two repeats each)

| arm | strict | judged | made-up turns | input tokens per conversation | seconds per conversation |
|---|---|---|---|---|---|
| A (main) | 17/20 | 23/26 | 3/26 | 26,220 | 12.8 |
| D (steps list) | 18/20 | 24/26 | 0/26 | 23,658 (−10%) | 10.0 (−22%) |
| D′ (steps list + data cards) | 16/20 | 24/26 | 0/26 | 20,983 (−20%) | 9.7 (−24%) |

- D and D′ removed A's "sales_summary is not in my tool list" claims. D′'s strict misses were scorer misses ("four" written in words).
- The 2/26 left on D and D′ is the cross-chat `explain-08` (harness gap: no cross-chat steps).
- D′ (cards in place of raw replayed rows) matches D on accuracy at 11% fewer tokens.
- D′ is an eval-only arm (a history switch in a local tree), not a PR.

## 3. Memory

**Capabilities benchmark in harness mode: blocked.**

- **Infrastructure.** With inline jobs (forced by local-mode Qdrant), the first attempt exhausted the database pool (`QueuePool limit 8+8`), and the outbox relay stopped dispatching. After a restart, every later attempt hit a deterministic 500 on `POST /v1/messages`: a `threads_pkey` UniqueViolation when a scenario's first batch creates its thread. Both are memory-service bugs in this stand-in mode; they need a real worker process, which needs a Qdrant server.
- **Code fixed locally first.** Harness mode required Bifrost's gateway client and sent `temperature=0`, which Haiku refuses. It now answers through the harness's own route (branch `agent-memory-service/claude/cool-maxwell-w5avo8-capharness`, not opened as a PR).
- **What the complex suite shows instead.**
  - Cross-chat recall: `explain-07` was right once tools were not narrowed away.
  - Corrections: `conversation-02/06/10` were right on every adapter.
  - Preferences: a stored constraint was applied, sometimes to the wrong case (below).
  - Forget was not exercised.

**Learning over time.** One agent and one user, 10 repeats of "who supplies <product>, and the lead time":

- Accuracy 10/10.
- 14 tools offered from the first repeat; `find_sku` was always offered and there were no misses.
- 4 tool calls every time; the 4th is a redundant `supplier_portal`, which the learned plan never dropped.
- Seconds went from 6.1 to 5.7, and output tokens from 674 to 590.
- No measurable learning effect.

**Memory gaps found.**

- Earlier questions are stored as user "standing rules" and pushed into unrelated cases, for example "Take our best-selling SKU…". That is what drove the ambiguity-02 miss.
- Agent-written notes become user rules for other agents (the "use bash" note).
- Tool-hint bindings put a product name into the supplier argument, and literals into `sku` and `group_by`.
- Agent endpoint registration is refused (`GET /v1/models` returns 400 with a Bearer key on Anthropic's OpenAI-compatible route), so memory falls back to its own endpoint. That fallback worked.
- The harness profile claims `temperature=true` for a model LangChain does not know, and does not know Haiku's 1M window, so clearing and long-task tools were off.

## 4. Tokens per layer and latency (before, full runs)

Shares of input tokens by layer:

| adapter | instructions | memory context | tool schemas | history | question | tool results |
|---|---|---|---|---|---|---|
| ReAct | 2% | 25% | 51% | 2% | 1% | 18% |
| create_agent | 2% | 33% | 44% | 3% | 1% | 16% |
| Deep Agents | 1% | 21% | 60% | 2% | 0% | 16% |
| OpenAI Agents (chars/4 estimate) | 2% | 30% | 46% | 3% | 1% | 18% |

- **Tokens per turn.** Mean input per turn: ReAct 17.9k, create_agent 21.1k, Deep Agents 32.3k, OpenAI 23.5k, Claude 86.8k (its CLI prompt). Output per turn is about 2k, thinking included.
- **Fixed cases, before → after.** On the cases the fixes changed: input +19% to +62% (all tools offered, the query note added), output −57% to −85%, seconds −31% to −69%. Cost per case was flat or lower.
- **Latency per turn.**

| adapter | median, before → after | p90, before → after |
|---|---|---|
| ReAct | 6.1 s → 6.3 s | 16.5 s → 14.8 s |
| create_agent | 6.4 s → 6.6 s | 17.6 s → 17.1 s |
| Deep Agents | 6.0 s → 6.3 s | 23.9 s → 18.1 s |
| OpenAI Agents | 6.4 s → 6.6 s | 23.0 s → 15.3 s |
| Claude | 13.4 s → 12.8 s | 28.6 s → 26.1 s |

## 5. Spend (from response usage; cap $5)

| item | US$ |
|---|---|
| Five full runs (ReAct 0.333, create_agent 0.391, Deep Agents 0.511, OpenAI 0.403, Claude 0.327) | 1.965 |
| Re-runs of failures (0.38 + Claude 0.088) | 0.468 |
| Controls, cannot-do and freshness re-runs | 0.317 |
| Explain arms | 0.146 |
| Smoke and probes | 0.060 |
| Learning run | 0.021 |
| After on main, ReAct: first run (shared user, confounded) 0.400 + clean run 0.302 | 0.702 |
| **Agent total** | **3.68** |
| Memory's own model work (≈ 445k tokens plus two more runs, priced conservatively at the output rate) | ≤ 0.27 |
| Reserve for Claude turns that may have passed the 100k-prompt price tier | ≤ 0.19 |
| **Total, worst case** | **≤ 4.14** |

## 6. PRs (agent-harness)

| PR | what | state |
|---|---|---|
| #71 | A recorded tool output over 256 KiB was refused by memory (422), so the call was lost. It is now recorded cut to fit. | **Merged** (CI green) |
| #73 | Tables of 30 or more rows that are read whole are also queryable (`query_result`) | **Merged** (CI green) |
| #77 | Hints narrow the tools only from 30 tools in the run | **Merged** (CI green, after a fix: the feature-matrix cells F28/F28c lower `NARROW_FROM` as the unit tests do, and the docs that said "from 5 tools" now say 30) |
| #74 | Complex suite: `--adapter`, the cannot-do cases, scorer fixes | **Merged into the complex branch (#67)** (CI green; the one 3.13 failure was a timing flake in `test_bench_load`, re-run green) |
| #76 | A stall ends with one answer and no tools, instead of an error with no answer | STATUS76 |
| #79 | Live earlier values are marked out of date in framework-kept threads and in the OpenAI memory session | STATUS79 |

Every PR has its tests written first and the challenge checklist. Two CI problems were real and fixed: #77's matrix cells, and #79's new narrowing test, which needed the narrowing fixture once #77 landed (found by the after-run's own sanity tests before #79 could break main). #76's one red job was a timing flake in the matrix (`F09r`, the Kimi CLI's run timeout), re-run. The merge commit `fb604a9` on #79 (main merged in) went out without the session trailer; I did not force-push to fix it. #67 itself (the complex suite) is still a draft and was not merged by me.

## 7. Remaining gaps

**Model gaps:**

- Copy slips: a SKU missing a zero, a division a penny off.
- OpenAI Agents answered from values marked out of date (freshness-04).
- Answering for both candidates instead of asking (ambiguity-03). It is the only wrong turn left in the clean after-run on main, where it assumed the pair named in a saved memory note.

**Harness gaps, not fixed:**

- Cross-chat steps cannot be reached (`explain-08`).
- Claude Agent SDK: tools disconnect on a resumed session, ToolSearch defers the harness's tools, and built-in Write and Bash are on.
- OpenAI Agents: a call to an unknown tool fails the whole run.
- No model-profile setting for a model LangChain does not know: the window and the temperature are wrong for Haiku. This is in the backlog.

**Harness/memory findings from the after-run on main (not fixed here):**

- Live-by-default freshness (`6320c3d`, on main) replays a seconds-old earlier result as `out of date`. On "how did you get that / how sure are you" turns, Haiku then doubts or disowns its own earlier answer. On a clean user it still answers correctly, but the wording ("out of date" for a value that was current when used) should say it was current then and may have changed since.
- A question from an earlier session became a profile standing rule ("compute with python3 via bash"). On later runs it made the agent refuse a computable question and invent an instruction. This is the same memory gap as "questions stored as standing rules", now shown to change answers across sessions.
- The complex suite uses one fixed bench user across runs. It should use a fresh user per run (or wipe it), or earlier runs leak into later ones.

**Memory gaps:** listed in section 3.

## 8. Break-and-fix loop (rounds 1–3, all five adapters)

The suite is `benchmark/cost/breakfix.py` (#111, based on #98). It has 60 new multi-turn cases (20 per round), plus 5 held-out cases that are never fixed against. They mix past chats (2+ earlier chats, approvals, corrections, a colleague's conflicting claim, user-defined terms and defaults), analytics over 2–3 tools, explanations, what-ifs, topic switches, corrections, ambiguity and can't-do requests. Every truth is computed from the world, and the scripted run passes 100%. Each case's user is scoped to the run and the case, and memory writes are drained between chats.

The stack was harness main plus the open PRs that touch this area (#67, #98, #76, #79, #72, #78, #83, #84/#105, #85, #89, #104, #108, #115) and memory main plus #64, #71, #72, #73 and mem#74. Left out: #58 and #62 (conflicts), and the sandbox, file-store, AG-UI, MCP and CI PRs.

**Judged accuracy** (strict in brackets). Every non-passing turn was read by hand.

| adapter | rounds | judged | past | analytics | explain | mixed | can't-do | clarify | held-out |
|---|---|---|---|---|---|---|---|---|---|
| ReAct | r1 (after fixes) / r2 / r3 | 93 / 100 / 98 (87/91/81) | 6/6, 6/6, 6/7 | 7/8, 11/11, 7/7 | all | 16/17, 14/14, 16/16 | 3/4, 4/4, 4/4 | all | 8/9 → 9/9 |
| create_agent | r1+r2 / r3 | 100 / 98 | 12/12, 6/7 | all | all | all | all | all | 9/9 |
| OpenAI Agents | r1+r2 / r3 | 98 / 100 | all | all | all | 30/31, 16/16 | all | 3/4, 2/2 | 9/9 |
| Deep Agents | r1+r2 / r3 | 98 / 93 | 12/12, 6/7 | 19/19, 6/7 | all | 29/31, 15/16 | all | all | 7/9 → 8/9 |
| Claude SDK | r1+r2 | 95 | 11/12 | 19/19 | 16/17 | 30/31 | 7/8 | 3/4 | 8/9 |

Analytics and explanation are at 100% on every adapter except two known harness items (Claude's ToolSearch deferral; Deep Agents' 12-call limit, #117). The made-up rate is 0 on all adapters after #108 and #115.

**Fixed (PRs, tests first):**

- #104 (merged): a memory outage mid-chat lost the conversation; it now falls back to the run store.
- #105 (folded into #84): the context and the question had no boundary.
- #108: a kept-result path was offered to the user as a CSV.
- #115: Deep Agents' scratch SVG was offered as a file. After the fix it says "saved to scratch space you can't access" and gives the content.
- mem#74: "Approve it" now keeps the figure from the agent's answer. The OpenAI case that failed now passes.

**Remaining:**

- **Model:** hedging on a revoked approval ("check your procurement system") on ReAct, Deep Agents and LangGraph, although memory marks it superseded; OpenAI Agents concludes correctly. Picking one of 14 "black slim jeans" instead of asking (OpenAI, Claude). Over-clarifying "that list". Dropping the period on a follow-up (once). Claude implying it could schedule.
- **Memory:** a figure the agent computed in an earlier chat is reachable only by message search. The user's memories exclude agent text, by design. Proposal: keep the agent answer's headline beside the user's question, as mem#74 does for referring turns. That is a design call, so it was left for review.
- **Harness, known:** Claude adapter (tools deferred by ToolSearch; earlier chats not consulted); the step-limit end with no answer (#117).

**Environment:** my memory was OOM-killed once and that round was re-run. The read deadline was raised to 20 s in my private launcher only, because jobs run inline there.

**Spend:** agent $2.24 in the loop, $6.2 total; with memory's own model work at most about $6.7 worst case of the $15.
