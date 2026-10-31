# Improvement backlog

Every open gap the research found, with where it stands. A living list: update a row when its
work lands, is measured, or is dropped.

**Keep rule.** Ship only measured improvements. A harness change is kept only if it passes the
[evaluation protocol](EVAL_PROTOCOL.md) against main, on the same cases and the same model: a
big-model judge says the answer is correct and that the captured context was sufficient to
answer. The context-sufficiency rate goes up (or stays at 100%), strict accuracy does not drop,
the made-up rate does not rise, controls do not drop, and tokens, model calls and seconds are not
materially up (accuracy wins ties against cost). A change that does not pass is not kept, however
reasonable it looks.

**Design rules.** Every change follows [DESIGN_RULES.md](DESIGN_RULES.md): the framework's own mechanism first, the harness adds only what is missing, library-first and research-backed, and the code is checked before every change. Every PR answers the [challenge checklist](DESIGN_RULES.md#challenge-every-change).

**Status.** *exists*: on main. *in flight*: being built, on the branch named. *next*: the
next to build, once measured. *later*: deliberately deferred.

The direction, the intelligent harness and what the model gets (context map, data cards, built-in tools by run state), is [ADR 0006](../adr/0006-what-the-model-gets.md).

Sources: [SCENARIO_COVERAGE_AUDIT.md](SCENARIO_COVERAGE_AUDIT.md) (rows A1–M7, a snapshot of
2026-10-30) and [SELF_EXPLANATION.md](SELF_EXPLANATION.md) (the explain shortlist).

## Conversation

| Gap | Status | Evidence |
|---|---|---|
| Follow-ups with pronouns: the memory query is the raw latest message | in review: memory #66 — a short question whose own search is weak is searched again with the thread's previous question (follow-ups top-3 10 → 14/15, topic changes unchanged, `benchmark/follow_ups.py`) | audit A1, M1 |
| Honest "I don't know": the abstain note names the missing detail and offers to look it up or ask; it reaches the model in every adapter (matrix F41a) | done (#89) | audit I6, M2 |
| The abstain note is skipped whenever a tool hint reaches 0.5, and the memory service's hints claim 0.63–0.74 for questions no tool answers ("What's my sister's name?" → `find_sku` 0.74): with any tools, the note almost never reaches the model (0/12 requests on the complex suite's nodata cases) | next: calibrate the hint confidence (memory) or gate the note on the hinted tool's fit, then measure | audit I6, B13 |
| Messages sent mid-run (steer, interrupt, queue), memory-off history, stable message ids, session continuity | exists (#64) | audit A6, G6 |
| `mode` (queue / interrupt / steer) on AG-UI (`forwardedProps.mode`) and A2A (`metadata.mode`) | exists | `tests/integration/test_agui_frontend.py`, `test_a2a_server.py` |
| `mode` on `start()`: a queued run waits its turn by `concurrency_key` | next | #64 |
| A steered message reaches OpenAI Agents and Claude models but is recorded after the question in their transcripts: record messages in the order they came | later | #64 |
| Claude and ACP sessions are kept per process: a turn on another worker starts a new session (it falls back to the run store's history or the memory window) | later | #64 |
| The memory service keys threads by tenant and thread only: the harness passes the scoped conversation key, but a direct memory API user can still collide; consider scoping in memory | later | #63, #64 |
| Conversation scoped by tenant, user and thread | exists (#63) | audit J1 (memory side) |
| agent-runs reads are tenant-wide for every key | in flight: user-scoped threads | audit J1 |
| Window fixed at 2,000 tokens; a large turn drops out | in flight: memory `claude/cool-maxwell-w5avo8-winbudget` (#62), harness/memory `-explain2` (#62/#64, arm C) | audit A1, A4, K3, M4 |
| Corrections across sessions (`update_contradiction` 0.0–0.1) | next: slot-keyed latest-wins selection at read time (memory) | audit A2, H3 |
| Topic switch mid-thread has no test | later | audit A3 |

## Explaining earlier answers

| Gap | Status | Evidence |
|---|---|---|
| Sources, steps, tools, seconds, approvals of an earlier answer | in flight: `claude/cool-maxwell-w5avo8-explain` (#59), `-explain2` (#62, arm D) | audit B1, B2, B5, B6; [SELF_EXPLANATION.md](SELF_EXPLANATION.md) |
| Explain test fixed first (template, memory-off floor, prompt capture, strict scorer) | in flight: `-explain2` | [SELF_EXPLANATION.md](SELF_EXPLANATION.md#fix-four-flaws-in-the-test-before-re-measuring) |
| Pilot arm E (shortlist 2–5: tool names, worked example, repeated-call wording, confidence/why from the record) | next, one change at a time | [shortlist](SELF_EXPLANATION.md#shortlist-to-test-against-main-one-change-at-a-time) |
| Why a step was taken; how sure the answer is | later (after arm D) | audit B3, B4 |
| What changed since last time (tool values) | later (needs fetch time) | audit B7 |

## Live data and large results

| Gap | Status | Evidence |
|---|---|---|
| Values that change fetched fresh; replayed results dated; a stale tool kept offered | in flight: `claude/cool-maxwell-w5avo8-data` (freshness: `tool(freshness="live", max_age=)`, dated snapshots) | audit C1, C2, M5 |
| Plain `ReAct` cuts a large result with no way back | in flight: `-data` (results kept and paged by every adapter, `FitWindow` cut says where the rest is) | audit D1, M4 |
| Summation/aggregation without a sandbox: tool-side aggregation parameters, then a read-only query over saved results (a library, e.g. DuckDB with external access off), then a sandbox or Code Mode | in flight: `-data` (kept results written into the run's sandbox; `--classes aggregate`) | audit D2, E4 |
| Agent-written memories copying a tool value | later | audit C3 |

## Tools

| Gap | Status | Evidence |
|---|---|---|
| Tool chains: output→input links from the tools' published schemas, said in descriptions, a hinted tool's chain kept offered | exists once `claude/cool-maxwell-w5avo8-chains` lands (`tools/links.py`; matrix F28c; `--classes chain`) | [tools.md](../tools.md#tool-chains); on main the F28c cells fail: hints narrowed to `find_sku` drop `get_stock`. Proxy on a 2B model (#65): first step right 8/10 → 10/10, invented ids 4 → 2, +8.5% input tokens; an end-to-end re-run with a chat template that parses tool calls is next |
| Learned plans with "next step" and argument bindings (earlier output → later argument) | exists (memory ADR 0018: `modules/tools/procedures.py`, `hints.py` `_from_earlier_step`) | memory `tests/unit/test_tool_learning.py` |
| Learned plans matched by meaning, not only token Jaccard | exists (memory #53: `modules/tools/matching.py`, embeddings with a calibrated `procedure_cosine` floor) | memory `tests/eval/test_procedure_match_gate.py` |
| B11: a learned literal (`ST-101`) prefilled when the task names another value | exists, fixed (memory `procedures._constant`, `hints._names_another`) | memory `test_tool_learning.py` B11 tests |
| Long chains routed through Code Mode or a sandbox | next: measure `--classes chain` with Code Mode on first; gateway MCP listings carry no output schema, so their chains are not linked yet | `benchmark/cost` `chain` class |
| Gateway MCP tools' output schemas (Bifrost listing has none) | later: needs the gateway SDK to carry `outputSchema` | `bifrost_sdk.ToolDef` |
| Cold catalogs of 100–500 tools (0 successes recorded on an 8k window) | next: re-run `benchmark/cost` at 100/500 with hints on | audit M3 |
| Clarifying questions (ask vs guess) | in flight: `claude/cool-maxwell-w5avo8-reactnative` (#58, `ask_user` arms) | audit F |
| AG-UI frontend tools, `context` and shared `state` (`frontend.py`): the client's tools offered to the model, a call streamed and paused (interrupt reason `frontend_tool`), answered by a `tool` message or a `resume` entry; context and state in the pushed context within half its budget; state changes streamed as `STATE_DELTA`/`STATE_SNAPSHOT` | exists | `tests/integration/test_agui_frontend.py` (a fake CopilotKit-style client); matrix F62t (every adapter that takes a run's tools) |
| AG-UI activity summaries ("used N tools · ran K commands") from the tool events | next | `Role.ACTIVITY` in `agui/events.py` |
| A structured `ask_user` (on #58) with optional `options[]`, `date`, `date_range` and `number`, so a UI shows dropdowns and pickers and plain chat shows text | next | #58 |

## ReAct

| Gap | Status | Evidence |
|---|---|---|
| A published ReAct-vs-`create_agent` scorecard (accuracy, tokens, calls, seconds per class) | next: `benchmark/cost` variants `baseline` / `main` / `deep` exist; publish the scorecard per class | [evaluation.md](../evaluation.md) |
| A "test my tools" scenario command: run a team's own tools through the scenario suite | next | `examples/06_scenarios/tool_choice.py`, `tests/live/heldout.py` |
| LangChain limits, errors and retries measured by a ReAct suite | in flight: `-reactnative` (#58) | audit I, E4 |
| `delegate` removed; helpers are Deep Agents' `task` | done (#57) | audit E2; duplication audit A14 |
| In-run verification, deterministic: every number traced to a tool or query result, every tool and source named present in the run record (no extra model call) | later (re-test with arm D's record) | audit E3; ADR 0006 |
| Decomposing multi-part questions (Plan-and-Solve, least-to-most), and selective self-consistency only on conflicting evidence or an unverified number, capped in cost | later | ADR 0006 |
| Grammar-constrained tool calls for local models (llama.cpp grammars, XGrammar) | next: the chain measurement showed a 2B model writing its calls as text the server did not parse | #65; ADR 0006 |
| The complex suite (retrieve, compute, chain, converse, explain, with edge cases) and a big-model reference answer per case | next | ADR 0006; [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) |
| Prompt cache: memory context before the history | later | audit K1 |

## Models and CI

| Gap | Status | Evidence |
|---|---|---|
| A model profile for a model LangChain's route does not know (an Anthropic model on the OpenAI-compatible route refuses `temperature`): look the model up in the models.dev data that LangChain's installed provider packages ship (data, not rules on names), else use one documented profile setting | later | [ADR 0004](../adr/0004-provider-neutral-models.md) |
| Test-order flake: `tests/integration/test_fit_window.py` fails when it runs after the unit suite, on main too. Fix it: nothing may fail | next | reported against main; reproduce first |
| Memory draft #64: CI was red before its last push, and one merge commit lacks the commit trailer. Fix both before the merge | later | memory #64 |

## Safety and privacy

| Gap | Status | Evidence |
|---|---|---|
| Prompt injection in tool results and the memory context | **later** (by the user's decision) | audit J2 |
| PII detection in memory and run records | **later** (by the user's decision) | audit J3 |
| "Forget that" said in chat | later | audit H4 |

## Resume plan

1. Merge the in-flight branches when their CI is green: the data PR (`claude/cool-maxwell-w5avo8-data`), the provider fixes (`-provfix`), and arm D, the steps list as the run's record (`-explain2`).
2. Check for duplication: for each adapter, does the framework's own compaction (LangChain context editing, Deep Agents summarization, OpenAI Agents and Claude session trimming) overlap with `FitWindow` and clearing? Keep one owner per adapter.
3. A holistic run with a small hosted model: the complex suite (#67) across `ReAct`, `create_agent`, Deep Agents, OpenAI Agents and the Claude Agent SDK, plus the memory capabilities and learning over time.
   - Cap the cost at about $5.
   - Judge per [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).
   - Fix the harness gaps it finds, then re-run the failures.
4. Add the model provider's key again as an environment secret (`ANTHROPIC_API_KEY`, in the environment's settings). A copy in a session's scratch directory is lost on restart.
5. B11 (a learned literal prefilled when the task names another value) needs no work: it is fixed and tested on memory main (see Tools above).
