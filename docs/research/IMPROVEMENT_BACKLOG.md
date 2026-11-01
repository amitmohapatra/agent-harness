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
| `mode` (queue / interrupt / steer) on AG-UI, A2A and `start()`: today only `run()` and `stream()` take it | next | #64 |
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
| Why a step was taken; how sure the answer is | in flight: `-explain2` (memory's steps list: `from step N` on a call whose argument an earlier result returned; an evidence line per turn — calls that succeeded, sources and how many calls name each, values out of date — graded high/medium/low from the record alone) | audit B3, B4; explain pilot on a small hosted model: strict 27/27 as arm D, controls 6/6, context sufficient 27/27, turn-2 input 13,199 → 11,898 tokens |
| Earlier turns' results as data cards (the result kept for `read_file` and `query_result`) | dropped: failed the keep rule | explain pilot: turn-2 input −35% (13,199 → 8,628), but strict 27/27 → 26/27 and more model calls and seconds: the model read the card in its own earlier turn as all it had seen and disowned that answer |
| What changed since last time (tool values) | later (the fetch time is now in the run record: `-explain2`) | audit B7 |

## Live data and large results

| Gap | Status | Evidence |
|---|---|---|
| Values that change fetched fresh; replayed results dated; a stale tool kept offered | in flight: `claude/cool-maxwell-w5avo8-data` (freshness: `tool(freshness="live", max_age=)`, dated snapshots) | audit C1, C2, M5 |
| Plain `ReAct` cuts a large result with no way back | in flight: `-data` (results kept and paged by every adapter, `FitWindow` cut says where the rest is) | audit D1, M4 |
| Summation/aggregation without a sandbox: tool-side aggregation parameters, then a read-only query over saved results, then a sandbox or Code Mode | exists (#66: `query_result` on SQLite) | audit D2, E4 |
| `query_result` on DuckDB when installed (a `query` extra), SQLite as the fallback: the same tables, typing rules, function names and limits on both | in review: `claude/cool-maxwell-w5avo8-duckdb` (4 tables of 50k rows load in 0.37 s, was 1.01 s; 8 calls at once over 500k rows 12 s, was 89 s) | [QUERY_ENGINE.md](QUERY_ENGINE.md), [ADR 0007](../adr/0007-query-engine.md): ties SQLite on accuracy (18/18), bounds a query's memory, a call 2x faster at 50k rows and 4.5x at 500k, 8 calls at once 6 to 9.5x faster |
| An id past 64 bits rounded (CSV) or failing the load (JSON) in `query_result` | in flight: #83 | [QUERY_ENGINE.md](QUERY_ENGINE.md) value fidelity |
| SQLite fallback: a query's own values are not memory-bounded (+370 MB in 0.7 s) | in review with the DuckDB switch: a value is at most 64 MB (`setlimit`) | [QUERY_ENGINE.md](QUERY_ENGINE.md) safety; `tests/integration/test_query_safety.py` |
| A run's results are loaded again on every query (either engine) | later | [QUERY_ENGINE.md](QUERY_ENGINE.md) safety and load |
| Agent-written memories copying a tool value | later | audit C3 |

## Tools

| Gap | Status | Evidence |
|---|---|---|
| Tool chains: output→input links from the tools' published schemas, said in descriptions, a hinted tool's chain kept offered | exists once `claude/cool-maxwell-w5avo8-chains` lands (`tools/links.py`; matrix F28c; `--classes chain`) | [tools.md](../tools.md#tool-chains); on main the F28c cells fail: hints narrowed to `find_sku` drop `get_stock`. Proxy on a 2B model (#65): first step right 8/10 → 10/10, invented ids 4 → 2, +8.5% input tokens; an end-to-end re-run with a chat template that parses tool calls is next |
| Learned plans with "next step" and argument bindings (earlier output → later argument) | exists (memory ADR 0018: `modules/tools/procedures.py`, `hints.py` `_from_earlier_step`) | memory `tests/unit/test_tool_learning.py` |
| Learned plans matched by meaning, not only token Jaccard | exists (memory #53: `modules/tools/matching.py`, embeddings with a calibrated `procedure_cosine` floor) | memory `tests/eval/test_procedure_match_gate.py` |
| B11: a learned literal (`ST-101`) prefilled when the task names another value | exists, fixed (memory `procedures._constant`, `hints._names_another`) | memory `test_tool_learning.py` B11 tests |
| Long chains routed through Code Mode or a sandbox | next, where the Bifrost gateway runs (not measured: none on the benchmark machine): `benchmark/tool_chains` with Code Mode; gateway MCP listings carry no output schema, so their chains are not linked yet | `benchmark/cost` `chain` class |
| Dependent tool chains measured end to end: depth 2, 3 and 5 (a date passed on), fan-out, fan-in, a step that fails once or stays down, an id in a ~110k-character kept result, a follow-up reusing the ids | exists (`benchmark/tool_chains`, 42 cases, a held-out split) | Haiku, 50 tools: 42/42 on `ReAct`, `create_agent` and Deep Agents, memory off, cold and warm; 4.4 model calls a case against a minimum of 4.1 (the parallel calls are already made at once). [results](../../benchmark/tool_chains/results/2026-11-01-haiku.md) |
| A planner arm (LLMCompiler's DAG, ReWOO's plan with placeholders) | not built: it could not beat plain `ReAct` on accuracy (42/42) and has 0.3 model calls a case to save | `benchmark/tool_chains` |
| Narrowed to one hint, a fan-in sibling or a renamed argument's source is dropped: `find_store` beside `find_product`, `next_truck_date` → `date`, `store_id` → `from_store_id` | later: no live run lost a needed tool (0 of 42 cases on every memory arm: the hints name several tools, a learned skill names its steps) | model-free check in `benchmark/tool_chains`: 81 of 162 (case, single hint) pairs drop one |
| Learned plans of calls made at once: they are recorded in the order they arrived, so two runs that ordered them differently split the prefix tree and the procedure stops at their common prefix | later (memory): seen once in a pilot (a reorder chain learned as `find_product` alone), not in the full warm run; record a model step's calls as one step, or order them by their data dependencies | memory `procedures._best_sequence`; `benchmark/tool_chains` warm arm |
| A person's or a judge's confirm citing ten or more memories taught nothing: its reindex key overran the outbox column, so the verdict was never projected and the run never labelled | in flight (memory #72) | warm arm: skills offered 0 → 38 of 42 cases, input tokens 65,244 → 44,000 a case, 42/42 both |
| Memory off, a follow-up re-runs the earlier chain to get its ids back (ADV-7: earlier tool results are not kept) | ADV-7 | `followup-ship`: 7 model calls memory off, 5 with memory |
| Gateway MCP tools' output schemas (Bifrost listing has none) | later: needs the gateway SDK to carry `outputSchema` | `bifrost_sdk.ToolDef` |
| Cold catalogs of 100–500 tools (0 successes recorded on an 8k window) | next: re-run `benchmark/cost` at 100/500 with hints on | audit M3 |
| Clarifying questions (ask vs guess) | in flight: `claude/cool-maxwell-w5avo8-reactnative` (#58, `ask_user` arms) | audit F |
| AG-UI frontend tools: `RunAgentInput.tools` is parsed but never used. Offer the client's tools to the model. On a call, stream `TOOL_CALL` events and pause through the existing interrupt and resume path, then continue when the client returns the tool's result. Test it with a fake AG-UI client | next | `src/trellis/harness/agui/events.py` (`tools` is declared, and nothing reads it) |
| MCP elicitation: an `mcp(...)` server's question in a call pauses the run as `ask` does (a choice, or a form with the server's schema) and the answer goes back to the server | exists (#80) | `tools/elicit.py`; matrix F18e; `tests/integration/test_mcp_elicitation.py` (both protocol eras) |
| MCP resources: a server's resources readable by the model | exists (#82): one read tool whose description maps them; a large one kept and queried as any result | `tools/resources.py`; matrix F18r; `tests/integration/test_mcp_resources.py` |
| MCP resources in the data map (ADR 0006 §1): the map is not built yet; today the read tool's description is the map | later: with the data map | [ADR 0006](../adr/0006-what-the-model-gets.md) |
| MCP resource updates: subscribe (2026-07-28 `Client.listen(resource_subscriptions=)`) so a changed resource marks its earlier reads out of date; today a read is live data (a dated snapshot when replayed) and the map says `lastModified` | later | the MCP SDK's `listen` |
| MCP tasks (a long call answered later): the official SDK has no client for them (2.2 and 2.3 leave `tasks/*` out; 2026-07-28 makes them an extension the SDK does not ship), so nothing is built on our own copy of the protocol. The design when it ships: a call that returns a task pauses the run (the existing interrupt and resume; no worker waits), agent-runs' scheduler polls `tasks/get` (or the server's notification resumes it), the task's result is the call's result, and cancelling the run sends `tasks/cancel`; no new queue or store | later: when the SDK ships it | `mcp_types.methods` ("tasks/* deliberately absent") |
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
| Memory draft #64: CI was red before its last push (the complexity ratchet: `replay_window`), and one merge commit lacks the commit trailer | CI fixed on `-explain2` (split in three); the trailer cannot be added without rewriting shared history | memory #64 |

## Safety and privacy

| Gap | Status | Evidence |
|---|---|---|
| Prompt injection in tool results and the memory context | **later** (by the user's decision) | audit J2 |
| PII detection in memory and run records | **later** (by the user's decision) | audit J3 |
| "Forget that" said in chat | done: `memory_forget` of the user's own statement asks a person, then forgets it as the user (`harness._forgotten`); the service names the memory it refuses (memory `-forgetid`) | audit H4 |

## Resume plan

1. Merge the in-flight branches when their CI is green: the data PR (`claude/cool-maxwell-w5avo8-data`), the provider fixes (`-provfix`), and arm D, the steps list as the run's record (`-explain2`).
2. Check for duplication: for each adapter, does the framework's own compaction (LangChain context editing, Deep Agents summarization, OpenAI Agents and Claude session trimming) overlap with `FitWindow` and clearing? Keep one owner per adapter.
3. A holistic run with a small hosted model: the complex suite (#67) across `ReAct`, `create_agent`, Deep Agents, OpenAI Agents and the Claude Agent SDK, plus the memory capabilities and learning over time.
   - Cap the cost at about $5.
   - Judge per [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).
   - Fix the harness gaps it finds, then re-run the failures.
4. Add the model provider's key again as an environment secret (`ANTHROPIC_API_KEY`, in the environment's settings). A copy in a session's scratch directory is lost on restart.
5. B11 (a learned literal prefilled when the task names another value) needs no work: it is fixed and tested on memory main (see Tools above).
