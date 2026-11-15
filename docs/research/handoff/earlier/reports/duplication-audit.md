# Duplication audit (harness, memory service, bifrost-sdk)

Date: 2026-10-31. Read-only: no code changed, nothing pushed. Note: the gstack check from the repos' CLAUDE.md returned `GSTACK_MISSING` in this container. The audit went ahead because it only reads; install gstack before any change.

**Snapshots read:**

- `agent-harness` origin/main `7ac33a4` (#68)
- `agent-memory-service` origin/main `2f8bc6f` (#65)
- `bifrost-sdk` origin/main `ccaaef8` (#9)

Paths are relative to each repo. A line number belongs to that repo's main unless a branch is named.

**Rules applied:** `docs/research/DESIGN_RULES.md` (the framework's own mechanism first; one owner per concern) and ADR 0003 §6 (one owner, no fallback chains). Also ADR 0005 §1 and §4 (the gap gets a reader, not a store; a session continues the history) and ADR 0006 (not doing: a second history store, rules keyed on how a question is worded).

**PR state (GitHub, today):**

| Repo | PR | Branch | State | Behind main |
|---|---|---|---|---|
| harness | #66 | `-data` | open, ready | 0 |
| harness | #67 | `-complex` | draft | 14 |
| harness | #62 | `-explain2` | draft | 11 |
| harness | #59 | `-explain` | draft | 21 |
| harness | #58 | `-reactnative` | draft | 21 |
| harness | #57 | `-rmdelegate` | open, ready | 21 |
| memory | #64 | `-explain2` | draft | 0 |
| memory | #62 | `-winbudget` | draft | 2 |

Legend: **yes** = two mechanisms do the same job and one should go. **partial** = they overlap, but part of the difference is justified. **no** = it looks like a duplicate but is justified.

## A. Harness vs adapter-native

| Concern | Where (each copy) | Duplicate? | Keep (native first) | Remove | Risk | Tests that cover it |
|---|---|---|---|---|---|---|
| A1. History when a Claude / ACP session continues with **memory on** | Native: the session is continued (`pipeline.py:380` `_continued_session`, `Agent.sessions` `agent.py:237`). Harness: the memory window is still rendered into the context, because `Agent.window()` (`agent.py:957-968`) never checks `runtime.continued`. `recalled()` checks it (`agent.py:857`), but only for memory off. | **yes**. The resumed session replays the earlier turns, and the memory window repeats them in the prompt text. | The framework's session (ADR 0005 §4) | Make `window()` return `Window.NONE` when `runtime.continued` | medium: tokens up every turn, duplicated facts, prompt cache broken | `tests/integration/test_conversation.py::test_the_conversations_session_is_continued_only_where_it_is_kept` (`:218-245`) covers **memory off only**. Add a memory-on case. |
| A2. Clearing old tool results (ReAct) | LangChain `ContextEditingMiddleware(ClearToolUsesEdit)` at 50% of the window (`react.py:190-210`, placeholder `CLEARED` `react.py:82`). FitWindow step `results` → `_cleared` (`middleware.py:1446-1455`, placeholder `FIT_CLEARED` `middleware.py:1166`). | **yes**. Two clearing passes with two placeholders. `_cleared` also rewrites `CLEARED` into `FIT_CLEARED`. | LangChain `ClearToolUsesEdit` | FitWindow's `_cleared` pass and `FIT_CLEARED`. FitWindow should cut (`_cut`), not clear again. | medium. To verify: `FIT_CLEARED` names `read_result`, but `ReadTools` (`middleware.py:1121-1150`, which runs before FitWindow) offers `read_result` only when it sees the `CLEARED` placeholder. | `tests/integration/test_fit_window.py:111`, `test_react_context.py`, `test_react.py`, `contract/test_langchain_internals.py`. Nothing checks that `read_result` is offered after `FIT_CLEARED`. |
| A3. Context overflow and summarization (ReAct with Deep Agents) | Deep Agents summarization at 85%, which also summarizes and retries on an overflow (`react.py:305,397,405`; `docs/frameworks/react.md:198`). FitWindow at 95% plus its own retry on the server's refusal (`middleware.py:1203-1283`; `turns=False` in this mode, `react.py:233`). | **partial**. Overflow recovery is done twice. Tool narrowing, description compaction, memory trimming and `ContextTooLarge` are not native, so they are justified. | Deep Agents' summarization and its overflow retry | FitWindow's retry (`refit`) when a summarization follows it; keep the retry for plain `create_agent` | medium | `test_react_deep_agents.py`, `test_fit_window.py` |
| A4. Large-result offload | Native: Deep Agents `FilesystemMiddleware` eviction (Deep Agents mode). Harness: `tools/results.py:79-132`, kept in `Journal.results` (`journal.py:150`), used by `convert/claude.py:42`, `convert/mcp.py:81,99`, `convert/openai_agents.py:75,88,171,200`. | **no** for OpenAI Agents and MCP (they have no cut of their own). **partial** for Claude, whose MCP output limit (about 25k tokens) never fires because the harness cuts at 20k tokens (`results.py:26`). Justified: the journal survives a resume on another worker. #66 extends it to plain ReAct and skips graphs that have their own `read_file` (`-data` `middleware.py` `_paged`). | both, as now | — | low | `test_openai_agents.py`, `test_claude.py`, `test_acp.py`, `unit/test_results.py`; #66 `test_large_data.py` |
| A5. Model retries | Bifrost gateway retries; client retries set to 0 (`gateways.py:127`, `openai_agents.py:736`). ProviderKey route: the provider client's own retries. #58 adds `ModelRetryMiddleware` and sets client retries to 0 (`-reactnative` `gateways.py`). A worker attempt is run again on a retryable model error (`pipeline.py:569-573`). | **no** at the call level (one layer each). **partial** at the run level: gateway retries, then the whole attempt is queued again. | gateway or client retries | nothing. Cap how many attempts agent-runs makes for 429 and 5xx. | low | `test_gateways.py`, `test_restarts.py`, `contract/test_runs_wire.py` |
| A6. Tool retries | `tools/base.py:64,123-126,181` (tenacity: reads and idempotent calls only). LangChain `ToolRetryMiddleware` is not used. OpenAI Agents, Claude and ACP have no tool retry. | **no** | harness (it is the only layer for every adapter) | — | low | `test_reliability.py`, `test_middleware.py` |
| A7. Step and call limits | Harness `StepLimit` (`middleware.py:946-999`, `LAST_STEP` `:156`) and `StallGuard` (`:1001`). Native: LangChain `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` (ADR 0003 lists them). #58 swaps them in (`-reactnative` `react.py` `limits()`). | **yes** (StepLimit). **no** (StallGuard: LangChain has no "same call repeated" check). | LangChain's limits (#58) | `StepLimit`, `STEPS`, `LAST_STEP`, `_failing` (as #58 does) | medium: behaviour changes, the run ends with LangChain's message | `test_react.py`, `test_middleware.py`, `test_react_deep_agents.py`, matrix F09r |
| A8. Run budget vs native caps | `budget.py:44-84` (`max_tool_calls`, tokens, cost), `Budgeted` (`automatic.py:169-191`). Claude's own `max_turns` / `max_budget_usd` are not used (cost is only counted after a query, `claude.py:470-480`). | **partial**. It overlaps `ToolCallLimitMiddleware` in ReAct, but it is the only cap that works across adapters and sub-agent runs. Claude: a gap, not a duplicate. | harness budget | — (option: pass `max_cost` to Claude's `max_budget_usd`) | low | `test_budget.py`, `test_react_spending.py` |
| A9. Stall and timeout | Stall watchdog (`runtime.py:411-420`, fed by `accounting.py:43-66` and `hooks/openai_agents.py:49`). `ModelHooks` `call_timeout` (`middleware.py:854-865`). | **partial**, justified: one times silence, the other the whole call | both | — | low | `test_stalls.py`, `test_run_limits.py` |
| A10. Approvals and permissions | One governance path for all adapters (`governance/`), entered through each framework's own hook: Claude `PreToolUse`, with the team's `can_use_tool` asked by the hook (`claude.py:540-544`); OpenAI `needs_approval`; LangChain HITL; ACP permission. `Journal.approved` (`journal.py:119-123`) stops a second ask. | **no** | as now | — | — | `test_governance*`, `test_hitl_feedback.py`, `test_acp.py` |
| A11. Sessions and history ownership (other adapters) | One owner each: memory window vs run store (`agent.py:843-846`); OpenAI Agents uses `MemorySession` as the SDK session, and `window()` returns NONE (`openai_agents.py:631-640`, `agent.py:964`); a LangGraph checkpointer owns its thread (`langgraph.py:151-155`) | **no** (except A1) | — | — | — | `test_conversation.py`, `test_memory_window.py` |
| A12. Prompt caching | No harness cache (removed per ADR 0003 §7). Deep Agents' Anthropic caching runs only on a `ChatAnthropic` model. Bifrost has `auto_inject`; Claude Code caches on its own. | **no** in code. To check in deployment config: Bifrost `auto_inject` on the Anthropic route alongside Claude Code's own `cache_control`. | gateway / CLI | — | low | `test_react_golden.py` |
| A13. Structured output | native only (`langgraph.py:230`, `claude.py:259`, OpenAI `output_type`) | **no** | — | — | — | — |
| A14. Sub-agents / task | Deep Agents `task` (general-purpose helper, offered by `LongTaskTools` when there are no native sub-agents: `automatic.py:76,272`, `react.py:388-390`). Harness `delegate` (`delegate.py`, `automatic.Delegation` `automatic.py:354-383`, built for both modes at `react.py:340,422-433`). | **yes**. On a long, wide task on a large model in Deep Agents mode, both helpers are offered. ADR 0003 §1 and §7 say no automatic helper. | Deep Agents `task` | `delegate.py`, `Delegation`, `wide()`/`_WIDE_CUE`, `model_tier` (`automatic.py:296-351`); this is what #57 does | low. #57 deletes 1,507 lines, including `test_delegate.py` and `test_delegation_routing.py`. | `test_react_deep_agents.py`, `test_subagents.py` |
| A15. Wrapped agents as sub-agents | `subagents.py` (child runs from any framework), vs Deep Agents `CompiledSubAgent`, Claude `agents=`, OpenAI handoffs and agent tools | **no**, justified: governed, journaled child runs across frameworks; native sub-agents stay native (`react.py:488-560`) | both | — | — | `test_subagents.py`, `test_a2a_remote.py` |
| A16. Planning | `automatic.Planning` is `TodoListMiddleware` under that name, so it replaces Deep Agents' own (`automatic.py:194`) | **no**. (Separate finding: `asks_for_steps` keys on wording, `automatic.py:69,100`, which ADR 0006 lists under "not doing".) | — | — | — | `test_react_golden.py` |
| A17. Tool narrowing | Memory hints (`agent.py:114,925-940`). FitWindow word-overlap narrowing, which ranks hinted tools first (`middleware.py:1383-1424`). `automatic.ToolSelection` wraps `LLMToolSelectorMiddleware` as a **no-hints fallback** (`automatic.py:136-166`, `react.py:224`, used only by the benchmark arm `benchmark/cost/variants.py:200`). | **partial**. ToolSelection is a fallback chain, which ADR 0003 §6 forbids (the LLM selector is meant as the team's own opt-in). FitWindow's narrowing is justified as the last resort when the request does not fit. | memory hints; FitWindow as a last resort | `ToolSelection` (a team can still add `LLMToolSelectorMiddleware` through `middleware=`) | low | `test_tool_choice.py:40,118` |

## B. Inside the harness and memory

| Concern | Where (each copy) | Duplicate? | Keep | Remove | Risk | Tests that cover it |
|---|---|---|---|---|---|---|
| B1. Two history paths | memory window (`agent.py:871-910`) vs run-store turns `conversation.earlier` (`conversation.py:125-170`) | **no**: they never run together (`agent.py:843-846`, ADR 0005 §1) | both | — | — | `test_conversation.py`, `test_conversation_scope.py` |
| B2. Read-back tools and their paging limits | `read_result`: graph state, 20,000 characters (`middleware.py:1086-1118`, `READ_LIMIT` `:154`). `read_file`: journal, 100 lines and 20,000 characters (`tools/results.py:106-132`, `:38-39`; `MAX_CHARS` 80k `:26`). Deep Agents' own `read_file`. More size limits: `MAX_FILE_CHARS` / `MAX_RUN_CHARS` (`automatic.py:80-81`); #66 adds `query_result` and `pages.MAX_PAGES` 200. | **yes**: two read-back tools for one concern (part of a result the model no longer sees), with separate limits | `read_file` (Deep Agents' contract); keep the limits in `tools/results.py` | `read_result`, `READ_LIMIT`, the two placeholders. #66's `_cut` already keeps the whole result and points to `read_file`; extend that to cleared results. | medium (model-facing text) | `test_react_context.py`, `test_hot_paths.py`, `test_middleware.py`, `unit/test_results.py`; #66 `test_large_data.py`, `unit/test_pages.py` |
| B3. Tool-chain links vs learned plans | Harness `tools/links.py:185-228` (from schemas), used at `agent.py:930-935`. Memory `modules/tools/procedures.py` (learned argument bindings) and `hints.py:129,320` | **partial**, justified: links cover a cold start, plans what was learned; both feed one offered set | both | — | low | `unit/test_tool_links.py`; memory `tests/unit/test_procedures*` |
| B4. Explaining an earlier answer | #59: a pull tool `earlier_answers` (`-explain` `explain.py`, 556 lines) with its own `Journal.approvals` store. #62: push, with run-record facts on stored messages (`-explain2` `pipeline.py` `_with_facts`, which reads the existing `journal.answers` and `approved`) and memory #64's steps lists (`-explain2` memory `window.py`). Memory #62 adds a window budget. | **yes**: three mechanisms, two of them pulling in opposite directions (#59 commit `6a1be7b` "drop the sized window" vs #62 `f1f9aba` "Size the replayed conversation") | push: harness #62 + memory #64 (ADR 0006 §1 History row, §4) | #59 (and its `Journal.approvals`, which duplicates `answers` + `approved`) | low (drafts) | #62 `test_run_record_facts.py`, `test_conversation_budget.py`; memory #64 `test_window_budget.py`, `test_conversation_window.py` |
| B5. Message ids | stable `msg_` id (`conversation.py:72-95`, in `custom_metadata`) vs `source_message_id` `{run}:{attempt}:msg:{i}` (`clients/memory.py:236-250`) | **partial, likely a real duplicate**. The source id depends on the attempt, so a resumed graph run sends attempt 1's assistant and tool messages again under new ids, and the service would store them twice. Only the USER id is stable. | the stable `msg_` id (ADR 0005 §3) | the attempt-based `source_message_id`: use the `msg_` id for de-duplication | medium (duplicated history) | `test_memory_pipeline.py:481-509` covers a function target (a user message only). Add a resumed LangGraph case. |
| B6. Steps lists vs `Journal.steps` | `journal.py:144` (the checkpoint step of each call) vs the explain steps lists (#62/#64) | **no**: same name, different concept | both (consider renaming one) | — | — | — |
| B7. Token budgets from the model window | `context_budget` 5% (`clients/memory.py:388-393`, `agent.py:218`). `FitWindow._budget` 95% minus the reply (`middleware.py:1285-1296`, `FIT_SHARE` `:1159`). #62 `conversation_budget` (`-explain2` `agent.py`, which moves `FIT_SHARE` to `adapters/base.py`). Memory's default `conversation_token_budget` 2000 (`config/constants.py:925`) and #64's layered spend. | **partial**: three formulas over one window; #62 already shares `FIT_SHARE` | the harness computes them (it knows the window); memory's value only as the default when none is given | — (after #62, nothing; keep one module for the shares) | low | `test_memory_pipeline.py`, `unit/test_adapters.py`, #62 `test_conversation_budget.py` |
| B8. Model profile and window, computed in several places | `adapters/base.py:262-274` `context_window`; `react.py:698-722` `window_of` / `_profiled`; `FitWindow._budget` reads the profile again (`middleware.py:1285-1290`); `automatic.model_tier` guesses size from the model's name (`automatic.py:312-351`); `gateways.py:210-214,602-619` `profile` / `_profile_of` | **partial** | `gateways._profile_of` + `context_window` as the one reader | `model_tier` (goes with #57); FitWindow should take the window only from its caller | low | `test_gateways.py`, `test_adapters.py` |
| B9. Freshness and time labels | #66: `snapshots.py` STAMP / STALE on replayed tool results, plus a CHANGES line in the tool description (`-data` `tools/base.py`). #62: run-record facts (status, seconds, sources with `Citation.date`), which memory #64 renders in steps lists. Memory #64 never replays older results whole, so #66's stamps on old results mostly never reach the model. | **yes** across PRs | one record: add `fetched_at` to #62's facts and have memory's steps list render it; keep #66's `max_age` and stale note, reading the time from the facts | #66's STAMP in result text, once #64 lands | medium (merge order) | #66 `test_freshness.py`, `unit/test_snapshots.py`; #62 `test_run_record_facts.py` |
| B10. Two "answer now" final calls | `StepLimit` + `LAST_STEP` (`middleware.py:156,946-999`) and `Budgeted` + `SPENT` (`automatic.py:169-191`, `budget.py:37`) | **partial**; #58 removes the StepLimit one | `Budgeted` | goes with A7 | low | `test_budget.py` |
| B11. Benchmark judges and suites | `evals.py:111,699-724` (LLM judge); #67 `benchmark/cost/judge.py` (offline packets); two different `benchmark/cost/explain.py` (#59 and #62); #58 `benchmark/cost/classes.py` beside main's class suite (`benchmark/cost/catalog.py:574`) | **partial** (benchmark code only) | #67 complex suite + its judge (EVAL_PROTOCOL); #62's explain.py | #59's explain.py; fold #58's classes into the catalog or complex suite | low | `unit/test_bench_*` |

## C. Across repos

| Concern | Where (each copy) | Duplicate? | Keep | Remove | Risk | Tests that cover it |
|---|---|---|---|---|---|---|
| C1. Retryable HTTP statuses | harness `middleware.py:168` `RETRYABLE_STATUS` {408, 409, 429} + any status ≥500 (`:886`) vs bifrost-sdk `_retry.py:21` `RETRYABLE` {408, 409, 425, 429, 500, 502, 503, 504}. Memory imports the SDK's set (`adapters/models/llm.py:38`). | **yes**, and they already differ (425; 501 and 505) | `bifrost_sdk.RETRYABLE` | the harness constant | low | harness `test_gateways.py`, `test_middleware.py`; bifrost `tests/` |
| C2. OpenAI-compatible clients | harness `_chat` → `ChatOpenAI` (`gateways.py:256-270`); `BifrostProvider` → `AsyncOpenAI` (`openai_agents.py:719-743`); the judge → `AsyncOpenAI` (`evals.py:699-724`); `clients/bifrost.py` → `bifrost_sdk.Bifrost` (MCP and admin); memory → `bifrost_sdk.Bifrost` (`llm.py:487-505`) | **partial**. The framework clients are justified (LangChain and OpenAI Agents need their own model objects). The judge needs no framework, so it duplicates bifrost-sdk's `complete()`. `BifrostProvider` repeats retries and key instead of using `Bifrost.client_options()` (`gateways.py:397-400`), and skips `openai_auth` (Google and AWS strategies), which `_chat` and the judge apply. | framework clients; bifrost-sdk for everything else | the judge's own client (use bifrost-sdk); have `BifrostProvider` reuse `client_options` and `openai_auth` | low-medium | `test_gateways.py`, `test_evals*` |
| C3. Auth strategies | `bifrost_sdk/auth.py:41-227`, used by the harness (`gateways.py:296-330`) and by memory (`llm.py:109-115`) | **no** (shared already) | — | — | — | bifrost `tests/` |
| C4. ModelProfile | `trellis.contracts.ModelProfile` vs `bifrost_sdk/_profile.py:9`, mapped by memory's `wire_profile` (`llm.py:118-120`) | **partial**, justified (bifrost-sdk takes no contracts dependency; the keys match) | both | — | low | memory `test_model_catalog.py` |
| C5. Profile discovery | harness `gateways._profile_of` (LangChain classes) → endpoint profile sent to memory (`gateways.py:356-366`) → memory overlays its model listing (`adapters/models/catalog.py:123-125`) | **no**: one chain, each step adds | — | — | — | — |
| C6. Layered memory-write retries | trellis-memory SDK `max_retries=3` (`sdk/python/src/trellis/memory/client.py:104`) × harness `writes.py` `WRITE_ATTEMPTS=3` (`writes.py:66,212-218`) | **partial**: up to 12 sends per write; the spool is justified, the second backoff loop is not | SDK request retries + harness spool | `writes.py` retries for failures the SDK already retried (or give the write client `max_retries=0`) | low | `unit/test_writes.py` |
| C7. Memory LLM retries on a Bifrost route | memory `max_retries=2` on every route (`config/constants.py:508`, `llm.py:497`); Bifrost retries too. The harness set its own client to 0 for this reason. | **partial** | gateway retries | 0 client retries when `route == bifrost` | low | memory `test_llm*` |
| C8. Circuit breakers | `bifrost_sdk/_breaker.py` vs `trellis.memory/breaker.py` ("Modelled on bifrost-sdk's breaker") vs harness `fresh.py` + `repository.RETRY_SECONDS` | **partial**: copied code; separate packages | — | (optional) share one implementation | low | — |

## D. Draft PRs that overlap each other or main

| PRs | Overlap | Recommendation |
|---|---|---|
| harness #59 `-explain` vs #62 `-explain2` | The same Citation commit: `tools/sources.py` diff identical, `tools/base.py` diff identical except one docstring line. Both rewrite `benchmark/cost/explain.py`. Opposite window decisions (B4). #59 adds `Journal.approvals`, which duplicates `answers` + `approved`. | **Close #59.** Keep #62 (rebase: 11 behind). |
| memory #62 `-winbudget` vs #64 `-explain2` | #64 contains #62's commits `ea106c0` and `ff6a857`. #62's only extra commit is the SDK 0.6.1 bump `42f6f9f`. | **Close #62** and carry the version bump into #64 (or merge #62 first, then #64). |
| harness #66 `-data` vs #62 + memory #64 | Freshness and time labels (B9); both add to `tools/base.py`, `tools/sources.py` and `agent.remembered`. #64 stops replaying older results, which #66 stamps. | Merge #66 (ready, 0 behind) first. Then put `fetched_at` into #62's run-record facts and drop the text STAMP. |
| harness #57 `-rmdelegate` vs main and #58 | Removes the duplicate helper (A14). 21 behind. #58 still threads `delegation` and `named=` through `_subagents`. | **Merge #57** after a rebase; then rebase #58 onto it. |
| harness #58 `-reactnative` vs main | Replaces `StepLimit` with LangChain's limits (A7, B10). Its benchmark `classes.py` overlaps main's class suite and #67. | Keep the source change (native first); move its suite into the catalog or #67; rebase (21 behind). |
| harness #67 `-complex` | Benchmark only; its `judge.py` overlaps the `evals.py` judge (B11); no source overlap | Keep (it is the protocol's suite); rebase (14 behind). |

## Summary

**Real duplicates (yes): 11.**

- Seven on main:
  - A1: session + memory window
  - A2: two clearing passes
  - A7: StepLimit
  - A14: delegate vs task
  - B2: `read_result` vs `read_file`
  - C1: two retryable-status sets
  - A17's ToolSelection fallback (counted though A17 is partial overall: only the fallback goes)
- Four between PRs: B4 explain, B9 freshness labels, the #59/#62 shared commit, memory #62 ⊂ #64.

**Partial:** A3, A5, A8, A9, B5, B7, B8, B10, B11, C2, C4, C6, C7, C8. Of these, B5 (attempt-based message ids) may store history twice and needs a test.

**Looked duplicate, but justified:**

- the large-result journal for OpenAI, Claude and MCP (it survives a resume on another worker; the frameworks have no cut of their own);
- harness tool retries (the only layer for every adapter);
- one governance path entered through each framework's own approval hook;
- the memory window vs run-store history (they never run together);
- wrapped agents as sub-agents (governed child runs across frameworks);
- `Planning` (it replaces Deep Agents' own by name);
- links vs learned plans (cold start vs learned);
- auth strategies (already shared from bifrost-sdk).
