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
| Follow-ups with pronouns: the memory query is the raw latest message | exists (memory #66): a short question whose own search is weak is searched again with the thread's previous question (follow-ups top-3 10 → 14/15, topic changes unchanged, `benchmark/follow_ups.py`) | audit A1, M1 |
| Honest "I don't know": the abstain note names the missing detail and offers to look it up or ask; it reaches the model in every adapter (matrix F41a) | done (#89) | audit I6, M2 |
| A new chat that refers back to an earlier one ("that same colour, but in cotton"): memory's context for it holds nothing, and the note sent the model to ask the user again | exists (#131): with `memory_search` offered, the note says to search the earlier conversations first (`agent.EARLIER_CHATS`) | break-and-fix `bf3-mixed-05`, Haiku ReAct: 0/3 → 3/3 |
| The abstain note is skipped whenever a tool hint reaches 0.5, and the memory service's hints claim 0.63–0.74 for questions no tool answers ("What's my sister's name?" → `find_sku` 0.74): with any tools, the note almost never reaches the model (0/12 requests on the complex suite's nodata cases) | next: calibrate the hint confidence (memory) or gate the note on the hinted tool's fit, then measure | audit I6, B13 |
| Messages sent mid-run (steer, interrupt, queue), memory-off history, stable message ids, session continuity | exists (#64) | audit A6, G6 |
| `mode` (queue / interrupt / steer) on AG-UI (`forwardedProps.mode`) and A2A (`metadata.mode`) | exists | `tests/integration/test_agui_frontend.py`, `test_a2a_server.py` |
| `mode` on `start()`: a queued run waits its turn by `concurrency_key` | next | #64 |
| A steered message reaches OpenAI Agents and Claude models but is recorded after the question in their transcripts: record messages in the order they came | later | #64 |
| Claude and ACP sessions are kept per process: a turn on another worker starts a new session (it falls back to the run store's history or the memory window) | later | #64 |
| The memory service keys threads by tenant and thread only: the harness passes the scoped conversation key, but a direct memory API user can still collide; consider scoping in memory | later | #63, #64 |
| Conversation scoped by tenant, user and thread | exists (#63) | audit J1 (memory side) |
| agent-runs reads are tenant-wide for every key | next: user-scoped threads (no branch yet) | audit J1 |
| Window fixed at 2,000 tokens; a large turn drops out | exists (#62, memory #64; memory off: #113): the context's budget is 5% of a known window, 2,000 to 8,000 tokens (`clients/memory.py`), with whole turns replayed | audit A1, A4, K3, M4 |
| Corrections across sessions (`update_contradiction` 0.0–0.1) | next: slot-keyed latest-wins selection at read time (memory) | audit A2, H3 |
| Topic switch mid-thread has no test | later | audit A3 |

## Explaining earlier answers

| Gap | Status | Evidence |
|---|---|---|
| Sources, steps, tools, seconds, approvals of an earlier answer | exists (#62, memory #64: arm D; #59 was closed for it) | audit B1, B2, B5, B6; [SELF_EXPLANATION.md](SELF_EXPLANATION.md) |
| Explain test fixed first (template, memory-off floor, prompt capture, strict scorer) | exists (#62) | [SELF_EXPLANATION.md](SELF_EXPLANATION.md#fix-four-flaws-in-the-test-before-re-measuring) |
| Pilot arm E (shortlist 2–5: tool names, worked example, repeated-call wording, confidence/why from the record) | next, one change at a time | [shortlist](SELF_EXPLANATION.md#shortlist-to-test-against-main-one-change-at-a-time) |
| Why a step was taken; how sure the answer is | exists (#62, memory #64: memory's steps list: `from step N` on a call whose argument an earlier result returned; an evidence line per turn — calls that succeeded, sources and how many calls name each, values out of date — graded high/medium/low from the record alone) | audit B3, B4; explain pilot on a small hosted model: strict 27/27 as arm D, controls 6/6, context sufficient 27/27, turn-2 input 13,199 → 11,898 tokens |
| Earlier turns' results as data cards (the result kept for `read_file` and `query_result`) | dropped: failed the keep rule | explain pilot: turn-2 input −35% (13,199 → 8,628), but strict 27/27 → 26/27 and more model calls and seconds: the model read the card in its own earlier turn as all it had seen and disowned that answer |
| What changed since last time (tool values) | later (the fetch time is now in the run record: `-explain2`) | audit B7 |

## Live data and large results

| Gap | Status | Evidence |
|---|---|---|
| Values that change fetched fresh; replayed results dated; a stale tool kept offered | exists (#66; earlier values marked: #79, #134. Freshness: `tool(freshness="live", max_age=)`, dated snapshots) | audit C1, C2, M5 |
| Plain `ReAct` cuts a large result with no way back | exists (#66, #78: results kept and paged by every adapter, `FitWindow` cut says where the rest is) | audit D1, M4 |
| Summation/aggregation without a sandbox: tool-side aggregation parameters, then a read-only query over saved results, then a sandbox or Code Mode | exists (#66: `query_result` on SQLite) | audit D2, E4 |
| `query_result` on DuckDB when installed (a `query` extra), SQLite as the fallback: the same tables, typing rules, function names and limits on both | exists (#116: 4 tables of 50k rows load in 0.37 s, was 1.01 s; 8 calls at once over 500k rows 12 s, was 89 s) | [QUERY_ENGINE.md](QUERY_ENGINE.md), [ADR 0007](../adr/0007-query-engine.md): ties SQLite on accuracy (18/18), bounds a query's memory, a call 2x faster at 50k rows and 4.5x at 500k, 8 calls at once 6 to 9.5x faster |
| An id past 64 bits rounded (CSV) or failing the load (JSON) in `query_result` | exists (#83) | [QUERY_ENGINE.md](QUERY_ENGINE.md) value fidelity |
| SQLite fallback: a query's own values are not memory-bounded (+370 MB in 0.7 s) | exists (#116): a value is at most 64 MB (`setlimit`) | [QUERY_ENGINE.md](QUERY_ENGINE.md) safety; `tests/integration/test_query_safety.py` |
| A run's results are loaded again on every query (either engine) | later | [QUERY_ENGINE.md](QUERY_ENGINE.md) safety and load |
| Agent-written memories copying a tool value | later | audit C3 |

## Tools

| Gap | Status | Evidence |
|---|---|---|
| Tool chains: output→input links from the tools' published schemas, said in descriptions, a hinted tool's chain kept offered | exists (#65: `tools/links.py`; matrix F28c; `--classes chain`) | [tools.md](../tools.md#tool-chains); on main the F28c cells fail: hints narrowed to `find_sku` drop `get_stock`. Proxy on a 2B model (#65): first step right 8/10 → 10/10, invented ids 4 → 2, +8.5% input tokens; an end-to-end re-run with a chat template that parses tool calls is next |
| Learned plans with "next step" and argument bindings (earlier output → later argument) | exists (memory ADR 0018: `modules/tools/procedures.py`, `hints.py` `_from_earlier_step`) | memory `tests/unit/test_tool_learning.py` |
| Learned plans matched by meaning, not only token Jaccard | exists (memory #53: `modules/tools/matching.py`, embeddings with a calibrated `procedure_cosine` floor) | memory `tests/eval/test_procedure_match_gate.py` |
| B11: a learned literal (`ST-101`) prefilled when the task names another value | exists, fixed (memory `procedures._constant`, `hints._names_another`) | memory `test_tool_learning.py` B11 tests |
| Long chains routed through Code Mode or a sandbox | next, where the Bifrost gateway runs (not measured: none on the benchmark machine): `benchmark/tool_chains` with Code Mode; gateway MCP listings carry no output schema, so their chains are not linked yet | `benchmark/cost` `chain` class |
| Dependent tool chains measured end to end: depth 2, 3 and 5 (a date passed on), fan-out, fan-in, a step that fails once or stays down, an id in a ~110k-character kept result, a follow-up reusing the ids | exists (`benchmark/tool_chains`, 42 cases, a held-out split) | Haiku, 50 tools: 42/42 on `ReAct`, `create_agent` and Deep Agents, memory off, cold and warm; 4.4 model calls a case against a minimum of 4.1 (the parallel calls are already made at once). [results](../../benchmark/tool_chains/results/2026-11-01-haiku.md) |
| A planner arm (LLMCompiler's DAG, ReWOO's plan with placeholders) | not built: it could not beat plain `ReAct` on accuracy (42/42) and has 0.3 model calls a case to save | `benchmark/tool_chains` |
| Narrowed to one hint, a fan-in sibling or a renamed argument's source is dropped: `find_store` beside `find_product`, `next_truck_date` → `date`, `store_id` → `from_store_id` | later: no live run lost a needed tool (0 of 42 cases on every memory arm: the hints name several tools, a learned skill names its steps) | model-free check in `benchmark/tool_chains`: 81 of 162 (case, single hint) pairs drop one |
| Learned plans of calls made at once: they are recorded in the order they arrived, so two runs that ordered them differently split the prefix tree and the procedure stops at their common prefix | later (memory): seen once in a pilot (a reorder chain learned as `find_product` alone), not in the full warm run; record a model step's calls as one step, or order them by their data dependencies | memory `procedures._best_sequence`; `benchmark/tool_chains` warm arm |
| A person's or a judge's confirm citing ten or more memories taught nothing: its reindex key overran the outbox column, so the verdict was never projected and the run never labelled | exists (memory #72) | warm arm: skills offered 0 → 38 of 42 cases, input tokens 65,244 → 44,000 a case, 42/42 both |
| Memory off, a follow-up re-runs the earlier chain to get its ids back (ADV-7: earlier tool results are not kept) | ADV-7 | `followup-ship`: 7 model calls memory off, 5 with memory |
| Gateway MCP tools' output schemas (Bifrost listing has none) | later: needs the gateway SDK to carry `outputSchema` | `bifrost_sdk.ToolDef` |
| Cold catalogs of 100–500 tools (0 successes recorded on an 8k window) | next: re-run `benchmark/cost` at 100/500 with hints on | audit M3 |
| Clarifying questions (ask vs guess) | in flight: `claude/cool-maxwell-w5avo8-reactnative` (#58, `ask_user` arms) | audit F |
| AG-UI frontend tools, `context` and shared `state` (`frontend.py`): the client's tools offered to the model, a call streamed and paused (interrupt reason `frontend_tool`), answered by a `tool` message or a `resume` entry; context and state in the pushed context within half its budget; state changes streamed as `STATE_DELTA`/`STATE_SNAPSHOT` | exists | `tests/integration/test_agui_frontend.py` (a fake CopilotKit-style client); matrix F62t (every adapter that takes a run's tools) |
| A structured `ask_user` with optional `options[]`, `date`, `date_range` and `number`, so a UI shows dropdowns and pickers and plain chat shows text | exists, opt-in (`tools/ask.py`, `trellis.ask_user()`); offering it by default is measured on #58 | `tests/integration/test_ask_user.py`; matrix F36u |
| AG-UI activity summaries ("used N tools · ran K commands") from the tool events | exists (`agui/activity.py`) | `tests/integration/test_ask_user.py` |
| MCP elicitation: an `mcp(...)` server's question in a call pauses the run as `ask` does (a choice, or a form with the server's schema) and the answer goes back to the server | exists (#80) | `tools/elicit.py`; matrix F18e; `tests/integration/test_mcp_elicitation.py` (both protocol eras) |
| MCP resources: a server's resources readable by the model | exists (#82): one read tool whose description maps them; a large one kept and queried as any result | `tools/resources.py`; matrix F18r; `tests/integration/test_mcp_resources.py` |
| MCP resources in the data map (ADR 0006 §1): the map is not built yet; today the read tool's description is the map | later: with the data map | [ADR 0006](../adr/0006-what-the-model-gets.md) |
| MCP resource updates: subscribe (2026-07-28 `Client.listen(resource_subscriptions=)`) so a changed resource marks its earlier reads out of date; today a read is live data (a dated snapshot when replayed) and the map says `lastModified` | later | the MCP SDK's `listen` |
| MCP tasks (a long call answered later): the official SDK has no client for them (2.2 and 2.3 leave `tasks/*` out; 2026-07-28 makes them an extension the SDK does not ship), so nothing is built on our own copy of the protocol. The design when it ships: a call that returns a task pauses the run (the existing interrupt and resume; no worker waits), agent-runs' scheduler polls `tasks/get` (or the server's notification resumes it), the task's result is the call's result, and cancelling the run sends `tasks/cancel`; no new queue or store | later: when the SDK ships it | `mcp_types.methods` ("tasks/* deliberately absent") |

## ReAct

| Gap | Status | Evidence |
|---|---|---|
| A published ReAct-vs-`create_agent` scorecard (accuracy, tokens, calls, seconds per class) | next: `benchmark/cost` variants `baseline` / `main` / `deep` exist; publish the scorecard per class | [evaluation.md](../evaluation.md) |
| A "test my tools" scenario command: run a team's own tools through the scenario suite | next | `examples/06_scenarios/tool_choice.py`, `tests/live/heldout.py` |
| LangChain limits, errors and retries measured by a ReAct suite | in flight: `-reactnative` (#58) | audit I, E4 |
| `delegate` removed; helpers are Deep Agents' `task` | done (#57) | audit E2; duplication audit A14 |
| In-run verification, deterministic: every number traced to a tool or query result, every tool and source named present in the run record (no extra model call) | later (re-test with arm D's record) | audit E3; ADR 0006 |
| A cheap claim check: a tool named in the final answer that the conversation never called gets one harness-written correction step | measured, not built: on the small hosted model, 0 made-up tool claims in 248 answers (the complex suite's explain, edge and conversation cases); both names a naive check flags are tools it suggests next ("a `warehouse_scan` would show…"), so a correction would remove good advice. The scripted model: 0 → 0. Re-test with a small local model | `-reactcache` |
| A read-only call repeated in a later step of one run answered from the earlier result, with a harness-written note | measured, not built: on the small hosted model, 1 repeat of a call that had succeeded in 248 turns (a retry after a malformed result) and no stalls; a tool's data is live by default (`max_age` 0), so an earlier result is not current. `StallGuard` and `max_steps` stay the backstop. Re-test with a small local model | `-reactcache`; research: harness-written feedback cut repeats ~76% |
| Decomposing multi-part questions (Plan-and-Solve, least-to-most), and selective self-consistency only on conflicting evidence or an unverified number, capped in cost | later | ADR 0006 |
| Grammar-constrained tool calls for local models (llama.cpp grammars, XGrammar) | next: the chain measurement showed a 2B model writing its calls as text the server did not parse | #65; ADR 0006 |
| The complex suite (retrieve, compute, chain, converse, explain, with edge cases) and a big-model reference answer per case | next | ADR 0006; [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) |
| Prompt cache: memory context before the history | done (`-reactcache`): sent with the question, after the history; the answer ending the history marked for a cache that keeps only what is marked. On the small hosted model, 15 multi-turn cases: cached share of a later turn's first call 0.70 → 0.80, of all input 0.858 → 0.877, uncached input −10%, accuracy unchanged | audit K1 |
| Prompt cache for a model given by name: the harness's routed chat model is not a `ChatAnthropic`, so Deep Agents' `AnthropicPromptCachingMiddleware` does nothing for it, and `ReAct` on `create_agent` has no caching middleware at all | done (`-promptcache`): `middleware.PromptCache` runs LangChain's middleware on the chat model the call is made with, in Deep Agents' own one's place; the history marker moved with it (Deep Agents' ran after the context moved, so it was never set). The small hosted model by name, #84's 18 multi-turn cases, 3 runs per arm, plain and Deep Agents: cached share of input 0 → 0.87, uncached input 610–620k → 78–79k (−87%), cost −62%, seconds per turn unchanged (4.0 → 4.0–4.1), strict accuracy 0.929 → 0.909 / 0.919 (within the runs' own spread; the turns that differ are correct answers the strict scorer missed). Through Bifrost the gateway caches. Anthropic's API given as an OpenAI-compatible endpoint, whose compatibility layer drops prompt caching (Anthropic's docs), is called with Anthropic's own client (`-anthropicnative`, `LLM_GATEWAY=openai-compatible` keeps the OpenAI client): the same cases through that endpoint, 4 runs per arm, cached share 0 → 0.88, uncached input 600–618k → 66–94k, cost $0.079 → $0.028, seconds per turn 4.8 → 4.0, strict accuracy 0.917 → 0.909 (the same turns miss on both). Any other OpenAI-compatible endpoint carries no breakpoints | `-reactcache` (found while measuring K1) |
| Replayed results re-marked as dated snapshots differ from what the earlier turn sent: the next turn's cached prefix stops at the latest exchange | measured, nothing to gain: in all 27 later turns of #84's captured requests the prefix ends earlier, at the previous question, which carried that turn's memory context; the marks come after it. Marks stay (tool data is live) | `-promptcache` |

## Models and CI

| Gap | Status | Evidence |
|---|---|---|
| A model profile for a model LangChain's route does not know (an Anthropic model on the OpenAI-compatible route refuses `temperature`): look the model up in the models.dev data that LangChain's installed provider packages ship (data, not rules on names), else use one documented profile setting | later | [ADR 0004](../adr/0004-provider-neutral-models.md) |
| Test-order flake: `tests/integration/test_fit_window.py` fails when it runs after the unit suite, on main too. Fix it: nothing may fail | next | reported against main; reproduce first |
| Memory draft #64: CI was red before its last push (the complexity ratchet: `replay_window`), and one merge commit lacks the commit trailer | CI fixed on `-explain2` (split in three); the trailer cannot be added without rewriting shared history | memory #64 |

## Adversarial suite

What the adversarial suite found (`benchmark/cost/adversarial.py`, 58 cases and 154 turns, `--cases adversarial`;
`tests/matrix/adversarial.py`, rows X01–X17 on every adapter). Each entry names the failing case
ids; the suite holds each as known (`adversarial.KNOWN`, `generate.KNOWN`), so the day it is fixed
its cases pass and the suite fails until the entry is removed. Reproduce a complex case with
`python -m benchmark.cost.complex --scripted --cases <id> --adapter <adapter> --out build/x`, a
matrix cell with `pytest tests/matrix -k <cell>`.

| Id | Issue | Failing cases | Status |
|---|---|---|---|
| ADV-1 | **A blank or whitespace-only message is sent to the model and kept in the thread.** `run("")` and `run("  \n\t")` reach the model as an empty user message on every adapter, and every later request of that thread carries it. A provider that refuses an empty text block (Anthropic does) would fail that turn and every turn after it: the thread can never be continued. Expected: refuse it before the model call with a clear error (or answer without a model call), and never keep it in the history. To verify against a real provider in the final gate. | `advconv-07` turns 0–3 (`nonblank`), langgraph, react, deepagents | fixed on main (#107): a blank message opens no run; `advconv-07` expects the refusal |
| ADV-2 | **`query_result` sums money in binary floating point.** The September refunds (6,000 lines of two-decimal amounts) sum to `32266.149999999998`, not `32266.15`: a small model may repeat the long figure. Expected: exact decimal sums for decimal text (e.g. SQLite's `decimal` extension or Python `Decimal` aggregates), or the result rounded to the inputs' places. | `advdata-11` turn 0 | fixed on main (#109): exact decimal sums on both engines |
| ADV-3 | **`query_result` fails the whole run on an integer past 64 bits** (`OverflowError`, run ERROR, no answer). | `advdata-12` turn 0 | fixed on main: `advdata-12` passes |
| ADV-4 | **A stall ends the run ERROR with no answer** (StallGuard's `ModelError`): the user is never told what went wrong. | `advres-02` turn 0; matrix X06 react, middleware | fixed on main (#76): the stall gets a tool-free last answer (matrix X06 on ReAct, whose test model honours `tool_choice="none"`); X06 `middleware` is held as ADV-4m: its test model is bound with no tools and calls one regardless |
| ADV-5 | **Three different malformed calls in a row stop the run** (`ERROR_STREAK`): a model correcting itself (wrong type, then a missing argument, then a made-up tool name, then the right call) is cut before its fourth call, though no call repeated. Expected: count a streak only of the same error, or allow more steps when each error differs; at least, answer with what is known (after #76). | `advres-01` turn 0 | fixed on main (#110) |
| ADV-6 | **A result saved in one turn is gone in the next, while the history still names its path.** A follow-up's `query_result` or `read_file` on `/large_tool_results/...` answers "File not found", so every what-if on a 50k-line result needs the whole fetch again. Deep Agents keeps it (its state backend); `create_agent` with the middleware and `ReAct` do not. Expected: keep a conversation's saved results for its later turns (or say in the history that they expired and how to fetch them again). | `advscen-01` turn 7, `advscen-02` turn 6, `advscen-03` turn 6 (langgraph, react) | fixed on main (#112): langgraph and Deep Agents pass; on react these turns fail on ADV-7 |
| ADV-7 | **ReAct with memory off replays only questions and answers, of the last 10 runs.** A follow-up has no earlier tool result (a vendor name read in turn 1, the 503 behind an explanation, the figures behind a recommendation: the `grounded` checks flag the reasons' figures as not in the context), and a figure from 13 turns back is gone with no summary. Partly by design (ADR 0005); the explain part is the steps list in flight (#62, arm D). | react: `advconv-05` t13, `advdata-06` t1, `advexp-02` t1, `advexp-05` t1, `advscen-01` t6, `advscen-02` t5, `advscen-03` t5, `advfile-02` t2 | fixed on main for a known window (#113: earlier turns replayed whole); still reproduces with no known window (the scripted runs), its turns held as known on react |
| ADV-8 | **A tool name the model made up ends an OpenAI Agents run ERROR** (the SDK's `ModelBehaviorError: Tool lookup_all not found`) instead of an error the model reads, as on every other adapter. Expected: the harness answers the unknown call with what tools exist. | matrix X05 openai_agents (run, stream, worker) | fixed on main (#103) |
| ADV-9 | **Nothing stops a looping model without the trellis middleware**: a plain `create_agent`/`create_deep_agent` graph and the Claude CLI ran all 30 identical calls; OpenAI Agents stops at its own `max_turns` but ends ERROR (`MaxTurnsExceeded`) with no answer. Expected: the harness's tool bridge (which every adapter's calls go through) stops the same call repeated, and a framework's own limit ends with a last answer, as `StepLimit` does. | matrix X06 langgraph, deepagents, claude, openai_agents | fixed on main for langgraph, Deep Agents and Claude (#114); OpenAI Agents still ends ERROR (`MaxTurnsExceeded`): held as known |
| ADV-10 | **A helper's own time limit (`h.wrap(run_timeout=)`) was dropped when its parent had a limit**: the child ran its full 2 s inside a parent with 60 s, though wrapped with 0.2 s. | matrix X09; `test_subagents.py::test_a_childs_own_time_limit_holds_inside_a_parent_that_has_one` | fixed in this PR (`subagents.child_run` takes the agent's own limit) |
| ADV-11 | **What a finished run spent is not on its end**: its tokens, model calls and cost are kept in the journal while it works (`budget.account`) and dropped at the end — neither `RUN_FINISHED` nor the record says them (only per-call spans, with tracing on). A scenario job's total cost cannot be read back. | matrix X17 (a strict gap on every adapter with a model) | fixed on main (#102): `RUN_FINISHED` carries `spent`; X17 reads it |
| ADV-12 | **No file store**: a run cannot hand its user a file, so "the report you gave me yesterday" works only through the suite's own file tools (`advfile-02..04`), not the harness. | matrix X13 (a strict gap) | fixed on main (#91): X13 passes |

Covered and passing on main (no finding): two people on one thread id (X01, `advconv-09`); queue,
interrupt and steer while busy (X02, `advconv-11..13`, `advexp-01`); a burst of 20 runs (X03); a
slow run store (X04); 24 helpers under the admission limit (X07), nested (X08), cancelled with
their parent (X10, X16), asking for an approval (X11), steered around (X12); a scenario job as a
durable run with progress over AG-UI, an approval before applying and an explanation after (X14),
its worker dying mid-way without redoing a scenario (X15); 20 Deep Agents helpers at once
(`advflow-02`); the allocation pipeline's waterfall, its missing and plain-text stage logs
(`advpipe-01..02`); the allocation domain's open questions, each checked for the source it must
use (tables for history, a run's logs for its why) and the one it must not (`advdom-01..10`; with
a scripted model these prove the pipeline, the judge gate measures the model); unicode, nulls, wrong types, duplicate keys, malformed CSV, a 503 or a timeout
mid-chain, pages without end, a 60,000-line log (`advdata-*`).

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
