# What remains to be done (Trellis stack) — written 2026-11-14 21:25 UTC

The owner asked for everything to go to main now, with these notes. Detailed history,
decisions and per-job reports: `FOLD_LEDGER.md`, `DECISIONS.md`, `ARCHITECTURE_DECISIONS.md`,
`ARCHITECT_BRIEF.md` (coding standard), `ROUND3_NOTE.txt` (job rules), `TEST_PLAN.md`,
the two reviews and the library inventories, all in this folder.

## 1. State of main

| Repo | On main | Verified by CI |
|---|---|---|
| agent-contracts | #18, #19 (`RunRecord.placed_at`) | yes, green |
| agent-runs | #35, #36 (`placed_at`, `concurrency_key` list filter, date answers), #37 (fixed set of answer formats) | yes, green |
| agent-memory-service | #115, #116 (504 in OpenAPI, redaction on decisions, two fixes) | yes, green |
| agent-harness | #257 = combined branch part 1 (OpenAI fixes, chat order, scenario fixes, LangChain 1.5 / Deep Agents 0.7.38, ReAct background, feature suite, harness side of `placed_at`) | **NO. Merged before CI finished.** Each folded step passed ruff, pyright (whole repo incl. tests), docs check and its touched test files only |

**First thing to do:** look at the CI run on harness main after #257 and fix what is red
(full suite, the 100% line+branch coverage gate of `trellis.harness`, examples, matrix).
Expect coverage gaps and cross-branch test failures: no job ran the whole suite locally.

## 2. Coded and pushed, NOT yet on main (harness)

All on branches `claude/cool-maxwell-w5avo8-<name>` of agent-harness. A fold job was merging
them into `-to-main-2` (worktree `/var/tmp/cons/stage2`, may hold unpushed merges) in this
order; whatever it had pushed after #257 needs a new PR to main.

| Branch | Head when written | What it holds | Notes for the merge |
|---|---|---|---|
| `-bg-claude-da` | 826a245 (moving) | Claude: SDK 0.2.182, verbatim prompts, per tenant+user config folder, fail-closed governance hooks, shared "workflows" run, turn limit → PARTIAL, PERF-3; round 2 in progress (one session-cost rule, replay rule, hooks between live turns, thinking blocks, fair share of live slots) | wins over react-background's commit 115348a on Claude `Workflow`; needs SDK 0.2.182 in the venv; `routed_env(env, runtime, headers)` and `LiveSessions.open(..., owner=, among=)` changed |
| `-react-parity` | da31ac9 | ReAct: provider tools, plan mode, checkpoint history kept, `langgraph-swarm` handoffs, compaction tool read; `Agent.checkpoints` / `Agent.rewind` for PAUSED runs | `CARRIED` = async_tasks, files, todos, ACTIVE_AGENT; golden test must stay byte-identical |
| `-scenarios` | eec031f (moving) | `tests/scenarios` (398+ tests, live stack), SCN-7/9/10 fixes; missing cases being written | `tests/support/invariants.py`: scenarios version + the features branch's four false-positive fixes |
| `-bg-kimi` | 41524e4 (moving) | Kimi 2.6.2 (2.1.x refused), asking mode confirmed, plans reviewed, allow-for-conversation, A3; round 2 in progress | CI must install kimi-code 2.6.2; `bridge.permitted/asked(kept=)`, `Runtime.asking_within` |
| `-fix-round` | c11b1ac (moving) | harness bug list: tool call id reuse after a resume (H-ID), identical writes in one step, a user's RetryPolicy on a failed read (SCN-11), AutoMode recording rule, carried-state bounds, AG-UI event ids across replicas, feedback `source`, approvals default assignee, remote chain depth | branched from `-to-main-2` |
| `-langchain-fixes` | 42e3233 (moving) | leftovers: no double model retry, summarization with FitWindow, defer_loading kept, cache_control for routed Anthropic names, sync-only middleware, tracing, MCP elicitation | first 444f1e3 is already on main |
| `-native-skills` | 23e82ba (WIP, a job just started on it) | each library's own skills under governance; must first merge `-to-main-2` | `governed_capability` wins over `SANDBOX_TOOLS` |

Memory, not yet on main: `-memory-fixes` (PR agent-memory-service#117: admin key forgets in
its tenant, extracted user facts ask first, inventory lists chat statements, redaction at
intake; CI was re-running after a test-setup fix) and `-memory-fixes-2` (thread delete takes
its documents and messages out of every read, derived profile lines follow their source,
migration for old feedback rows; in progress).

After memory#117 is on main remove these strict xfails in harness
`tests/features/test_mem_lifecycle.py`: `test_forgetting_a_fact_read_from_the_users_words_asks_first`,
`test_the_users_inventory_lists_what_they_said_in_a_chat`,
`test_erasing_a_user_leaves_nothing_any_read_returns` (the last also needs `-memory-fixes-2`).

## 3. Decisions waiting for the owner

- **Rewind of ENDED runs** needs an additive agent-runs change: `finish` keeps the run's
  checkpoint when asked (today it is cleared when the run ends). Until then `Agent.rewind` /
  `Agent.checkpoints` work for paused runs only.
- Rewind: only the run's own user may rewind (`reviewer` must be the run's user); a lead
  rewinding someone else's run is refused. `Agent.checkpoints` is tenant- and agent-checked,
  not user-checked (no caller identity in its signature). Confirm or change.
- Kimi: a team config with allow rules stops the run today; the per tenant+user config
  overlay (so Kimi asks and governance decides) is in progress and depends on Kimi 2.6.2
  offering a way to point at a config home.

## 4. Not started

- **Performance job** (excluded for now by the owner): PERF-2 one heartbeat per process
  (may need an additive agent-runs endpoint), PERF-7 event batching, PERF-11 hold sweep,
  PERF-13, PERF-14 (checkpoint history makes the stored run state 8–13× larger and it is
  rewritten whole every 16th save; make bytes written linear in steps).
- **Cleanup refactor** (last, by the owner's order): ARCH-3, 4, 6 (remove the harness retry
  loop around a graph's own tools, `Deferring`, the `_arun_one` wrap), 8, 9, 11, 12, 17, 18,
  21–25; DUP-1..10; test clones T1–T11; one `helping()` context manager for the six adapter
  copies of `helpers.closed(error, paused=helpers.suspended(error))`; `Adapter` protocol
  capabilities; boolean-flag parameters; the two OpenAI code paths that build an SDK approval
  pause become one function (no behaviour change; see DECISIONS.md).
- **One full test pass** on main: suite, coverage, matrix, contract tests, `tests/scenarios`
  and `tests/features` against a live stack (`scripts/scenarios/stack.sh`), the live stack.
- **Haiku phase** (needs separate credit, about $40): Deep Agents profiles for routed model
  names, Deep Agents backend multi-turn accuracy, ReAct re-measure, memory accuracy items
  (task list item 56), Claude real-CLI checks, the gateway Responses fallback against a real
  Bifrost.

## 5. Open product issues not owned by a running job

- Memory: after erasing a user, a profile line learned from them can outlive the memory
  (in `-memory-fixes-2`); decisions stored before redaction-at-intake keep raw arguments
  until that branch's migration runs.
- Sub-agent runs have no checkpoint history of their own; a rewound run re-records the
  question as a new turn in memory.
- OpenAI: sandbox memory consolidation writes are outside governance (the SDK has no seam);
  Redis / Dapr session backends are untested (packages not installed).
- Library limits, documented only: MCP elicitation on Claude SDK 0.2.182; Kimi `AgentSwarm`
  and goal mode over ACP; `middleware()` beside a user's same-named middleware and swarm
  per-message attribution (each needs a new public argument); computer use.
- Scenario tests still unwritten or unrun are listed per plan ID in
  `docs/testing/TEST_PLAN.md` on `-scenarios`.

## 6. Where things are on the build machine (ephemeral)

Worktrees under `/var/tmp/cons/` (`stage2` = the fold, `fixround`, `nskills`, `bgclaude`,
`bgkimi`, `memfix`, …). Private test stack `/var/tmp/cons/live57` (Postgres 127.0.0.1:55957).
The security review (`/var/tmp/cons/review/security.md`, SEC-R-1..11) is NOT in this
repository because the repository is public; SEC-R-1, 2, 5, 7, 9, 10, 11 are fixed in the
branches above, the rest are in the Claude and fix-round work.
