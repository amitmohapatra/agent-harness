# Haiku acceptance of the harness, 2026-11-15

Real-model runs of main (`873e630`) with `claude-haiku-5-5` only, the key from the
environment. Prices are `benchmark.cost.prices` (Haiku: $0.10 / $0.01 cached / $0.125
cache write / $0.50 per million tokens). Data: [benchmark/cost/results/haiku-2026-11-15](../../../benchmark/cost/results/haiku-2026-11-15/)
(the case lists; the raw files are on the `data/research-archive` branch, each listed with its sha256 in the folder's `MANIFEST.md`);
scripts: [scripts](scripts/). Total spend about $0.70 of the $40 allowed.

| run | arm | score | record | spend |
|---|---|---|---|---|
| ReAct accuracy, 1 run | plain (`--variant main`) | 117/118 strict (+1 unverifiable) | 352/357 over 3 runs (117.3 a run) | $0.249 |
| ReAct accuracy, 1 run | backend (`--variant deep`) | 118/118 strict (+1 unverifiable) | 356/357 over 3 runs (118.7 a run) | $0.262 |
| Deep Agents multi-turn (`carry`), 3 runs | ReAct with a backend | 24/24, checks 72/72 | none | $0.036 |
| Deep Agents multi-turn, 3 runs | `create_deep_agent`, own checkpointer | 24/24, checks 71/72 (probe false positive) | none | $0.029 |
| Deep Agents multi-turn, 3 runs | `create_deep_agent`, no checkpointer | 24/24, checks 72/72 | none | $0.033 |
| Claude Code, real CLI 2.1.325 | 5 probes, 4 full rounds | all pass every round | stand-in CLI only | about $0.02 |
| OpenAI Agents, Responses fallback | gateway stand-in | 2/2 runs, hosted tool refused | none | < $0.003 |
| Kimi Code 2.6.2 | 5 cases, 7 turns | 7/7, checks pass | none | $0.022 |

## 1. ReAct accuracy (the 357-turn set)

The set is the closing run's (`pilot/close`, 2026-11-11): five case lists
([sets](../../../benchmark/cost/results/haiku-2026-11-15/sets/)) run with
`python -m benchmark.cost.complex --adapter react --cases <list> [--window 16000|8000]`:
`long57`, `ordinary`, `sw16` at a 16,000-token window, `sw8` at 8,000, `swown`. A run is 99
cases, 139 turns, 119 strictly scored; **357 is three runs** (3 × 119), not one.

Re-derived from the saved transcripts (`scripts/rescore.py`, today's scorer):

| round | plain | backend |
|---|---|---|
| close (2026-11-11), scored when run | 119 + 116 + 117 = **352**/357 | 119 + 118 + 117 = 354/357 |
| final (2026-11-12), scored when run | 118 + 119 + 115 = 352/355 | 119 + 118 + 119 = **356**/356 |
| close, today's scorer | 349/354 | 352/355 |
| final, today's scorer | 352/355 | 356/356 |

So the record pairs plain from the `close` round with backend from the `final` round; in one
round the gap is 2 turns (close) or 4 (final). Today's scorer counts one refusal of
`cantdo-04` as unverifiable where it was right when run.

This run, re-scored from its transcripts (121 of 121 turns agree with the run's own score):

- plain: one wrong turn, `long-multi-02` at 8,000 tokens: Q3 denim gross profit left out —
  **model** (it had `query_result` offered and said it could not join units with unit costs;
  12 of 12 earlier runs answered it).
- both arms: `cantdo-04` (k-means of the stores) is unverifiable — **scorer**: the case expects
  "cannot do here", and the model now computed it with `query_result` (the deployment can do
  what the case says it cannot; `cando-04` already proves a k-means without a sandbox).

Both arms are within 2 turns of the record a run, so runs 2 and 3 were not made.

## 2. Deep Agents multi-turn: files and todos carried

New set `carry` ([carry.py](../../../benchmark/cost/carry.py)): six conversations, 17 turns, 8
scored, whose later turns need a file or todo list an earlier turn made (read back, edited,
overwritten, kept through a turn that does not touch it, a todo list over three turns). Every
turn that uses an earlier file checks that no file tool said "not found". New
`--variant unthreaded` builds `create_deep_agent` without a checkpointer, so the harness carries
the state (the fix in 5608343). Offline, the same checks catch the file loss on the tree
before that fix (7 of 24 checks fail).

Re-scored: every arm 8/8 in all three runs. Findings:

- One failed check (own checkpointer, run 2, carry-06 turn 2): the turn-1 `read_file` probe of a
  file not yet written stays in that graph's own thread — **probe** false positive (the check
  reads every request of the turn, history included).
- Two turns were first scored wrong by over-strict `forbidden` ids in the new cases (an answer
  naming the completed todo, or the replaced line, as such) — **scorer**, fixed in the cases
  and re-scored.
- `create_deep_agent` with `middleware()` is never offered `write_todos` here: the harness's
  `Planning` takes the place of Deep Agents' planning and holds it until a long task, which an
  unknown window never is (documented in deepagents.md). The todo cases were answered from the
  conversation. **Design question**, not a bug.
- With no checkpointer and no known window, earlier turns are replayed as text only (no tool
  calls). In carry-06, three runs of three, Haiku then said its earlier figures were never
  fetched ("I described tool results I never got") though they were. **Inference**: the
  text-only replay for a graph whose window the harness cannot read; `ReAct` (window known)
  gets the whole transcript and did not.

## 3. Claude Code against the real CLI

`scripts/claude_probe.py` (the bundled CLI 2.1.325 of claude-agent-sdk 0.2.182, a home of its
own, no settings), four full rounds (the last one's data kept, `claude/run6`):

- A governed write (`irreversible`) pauses once, runs once on approval (`ran == ["T-1"]`).
- A rejected call never runs; the model says it was refused and does not retry.
- A background subagent: its child run is `RUNNING` after turn 1, ends `SUCCESS` with the
  note; turn 2 (served by the same live session) quotes its code.
- A live session's second turn is the same process and remembers turn 1.
- The team's hooks between turns: the subagent's `PreToolUse` and `SubagentStop` (its
  `agent_id`), then the session's own turn's `UserPromptSubmit` and `Stop` with no run under
  way (`trellis.current()` is `None`).
- At `max_turns=3` the run ends `PARTIAL` (`BUDGET_EXCEEDED`, the turn limit) after one
  prompt (one `UserPromptSubmit`): no extra query. The next turn continued the task.

## 4. OpenAI Agents: the Responses fallback

No Bifrost binary or Go sources on the machine, so a gateway stand-in served: `meter_proxy.py`
in front of Anthropic's OpenAI-compatible API, which has no `/v1/responses` (404), mapping
Bifrost's `anthropic/<model>` name and answering `/mcp` with no tools. Through the harness's
Bifrost route (`LLM_GATEWAY=bifrost`): run 1 asked `/v1/responses` (404), then chat completions
(200) and answered; run 2 went straight to chat completions; an agent with `WebSearchTool` was
refused before any model call, naming the tool. **Bifrost itself was not tested.**

## 5. Cost and usage against the provider

The provider's usage is each response's, recorded by `meter_proxy.py`; the harness's is every
`budget.account` call.

| path | provider | harness | verdict |
|---|---|---|---|
| ReAct plain, 4 cases (OpenAI-compatible client) | 88,295 in / 4,319 out, $0.010989 | same; benchmark rows same | exact |
| ReAct backend, 4 cases | 188,663 / 4,633, $0.021183 | same | exact |
| OpenAI Agents, 2 runs | 7,794 / 144 | same | exact (no gateway cost: tokens only) |
| Claude, live session 2 turns | 6,202 / 12, $0.000471 | same, cost $0.000471 | exact |
| Claude, turn limit | 21,414 / 516, $0.000930 | same | exact |
| Claude, background subagent | 35,413 / 365, $0.001461 | 29,025 / 229, $0.001461 | dollars exact; tokens short — **fixed** (below) |
| Claude, approved call | 12,925 / 194, $0.000659 | 9,686 / 107, $0.000163 | paused attempt's dollars not charged — **fixed** (below) |
| Claude, rejected call | 9,512 / 141, $0.000580 | 6,297 / 69, $0.000099 | same — **fixed** |

The Claude CLI's `total_cost_usd` equals the provider's usage at the list price wherever a
result is charged. Findings:

- **Fixed** (`bridges/claude_live.py`): the turn a live session runs by itself once a task
  ended was charged its cost but not its tokens or model call (6,388 input and 70 output
  tokens on no run). Now its run is charged in full, or, with no run to charge, the next turn.
  After the fix the background probe's input tokens match the provider's exactly
  (35,507 = 35,507; 35,454 = 35,454).
- **Fixed** (`adapters/claude.py`, `_settled`; decided: charge the paused attempt in full): a
  query that paused in a process of its own was charged its replies' tokens with no dollars
  (the CLI gives a cost only in a result), and the model's reply to the pause placeholder
  ("Waiting for a person's approval.") was billed but never seen; the paused attempt was 75%
  and 83% of these runs' dollars (51% and 60% in the round before, with the prompt cache
  warm). Now the process is read to its turn's result (30 s at most; a call made in the
  reply is refused, ending the turn) and the run is charged it. Re-run on Haiku
  (`benchmark/cost/results/haiku-2026-11-15/claude/paused-charge/` on `data/research-archive`),
  provider against harness: approved 13,233 / 185, $0.000626 both (4 replies, 4.2 s);
  rejected 9,663 / 216, $0.000588 both (3 replies, 3.6 s); spend $0.0012.
- **Not fixed (CLI behaviour)**: a subagent's or a paused reply's output tokens are the
  stream's snapshot (24 of 57, 8 of 43, 32 of 61): the SDK's `AssistantMessage.usage` is
  taken before the reply ends, and only a result carries the final count.
- Not measured: Anthropic's native client path of `ReAct` (a local meter makes the route
  OpenAI-compatible) and Kimi's spend against the provider.

## 6. Kimi

Kimi Code 2.6.2 takes Anthropic's API natively (`KIMI_MODEL_PROVIDER_TYPE=anthropic`, as the
suite's Kimi target sets it): `multihop-01`, `stats-01`, `conversation-01` (3 turns),
`long-chain-01`, `long-flow-03` (a transfer paused, approved, made once) — 7 of 7 right, every
check held.
