# What remains to be done (Trellis stack) — updated 2026-11-14 22:00 UTC

The owner asked for everything that is coded to go to main at once, without waiting for
tests, with these notes. History, decisions and each job's report: `FOLD_LEDGER.md`,
`DECISIONS.md`, `ARCHITECTURE_DECISIONS.md`, `ARCHITECT_BRIEF.md` (coding standard),
`ROUND3_NOTE.txt`, `TEST_PLAN.md`, the two reviews and the library inventories, all in this
folder.

## 0. Read this first

**Harness main is unverified.** #257 and #259 were merged before any full CI run passed,
and the later merges were made with only: conflicts resolved, ruff clean, the package
imports, pyright over the whole repo. No job ever ran the whole suite on a combined head.
Expect red CI on harness main: test failures across branches, the 100% line+branch coverage
gate of `trellis.harness`, tests that were written but never run (listed in section 3).

**First job next time:** open the CI run on harness main, fix what is red, and only then
build further. Then run the one full test pass (section 6).

## 1. State of main

| Repo | On main | Verified |
|---|---|---|
| agent-contracts | #18, #19 (`RunRecord.placed_at`), #20 (`FeedbackSource.SYSTEM`) | CI green |
| agent-runs | #35, #36 (`placed_at`, `concurrency_key` list filter, date answers), #37 (fixed set of answer formats) | CI green |
| agent-memory-service | #115, #116 (504 in OpenAPI, redaction on decisions), #117 (admin key forgets in its tenant, extracted user facts ask first, inventory lists chat statements, redaction at intake) | #115/#116 green; #117 green on 3.12, 3.13 still running at merge |
| agent-memory-service | #118 (a deleted thread leaves every read, profile lines follow their memories, migration 0039 for old decisions) | **not verified by CI at merge**; item tests green, whole suite not run on the final head |
| agent-harness | #257 (OpenAI fixes, chat order, scenario fixes, LangChain 1.5 / Deep Agents 0.7.38, ReAct background, feature suite, `placed_at`) and #259 (Claude, SDK 0.2.182) | **NO** (see section 0) |

## 2. Coded and pushed, being folded into `claude/cool-maxwell-w5avo8-to-main-2`

A fold job merges these final heads in this order and each push is merged to main. If the
session ended first, the unpushed merges are in the worktree `/var/tmp/cons/stage2`
(ephemeral); anything below whose head is not an ancestor of harness main still has to be
merged (`git merge-base --is-ancestor origin/<branch> origin/main`).

| Branch `claude/cool-maxwell-w5avo8-…` | Final head | What it holds |
|---|---|---|
| `-bg-claude-da` | 9b286c9 | Claude round 2: one session-cost rule (`SessionCost`; a lower total is a reset, charged whole), replay rule (a write drops the reads journaled before it), a team's own hooks between live turns run once, thinking and server-tool blocks recorded |
| `-react-parity` | da31ac9 | ReAct: provider tools, plan mode, checkpoint history, `langgraph-swarm` handoffs; `Agent.checkpoints` / `Agent.rewind` for PAUSED runs |
| `-fix-round` | d21c1ca | identical writes in one step asked once each, AutoMode recording rule, carried-state bounds, AG-UI event ids the same on every replica, docs fix for approvals' assignee |
| `-langchain-fixes` | 9e89fb1 | no double model retry, the graph's own summary kept when the window guard drops turns, provider tool fields kept. Merge 9e89fb1 or later only (e70d8e0 up to it was a broken span) |
| `-bg-kimi` | 0802086 | Kimi 2.6.2 (2.1.x refused), asking mode confirmed before any prompt, plans reviewed by a person, allow-for-conversation, telemetry and auto-update off, mode confirmation waits on the event (up to 30 s), a Kimi home per tenant and user that turns team allow rules into ask rules |
| `-scenarios` | 4df022b | `tests/scenarios` (live stack), fixes SCN-7/9/10/14, the missing background and reliability cases, INV-4 / INV-10 corrections |
| `-native-skills` | 9a24f76 | each library's own skills governed and recorded; already merged with `-to-main-2` |

Fold rules that matter if it is redone by hand: `CARRIED` = async_tasks, files, todos,
ACTIVE_AGENT; journal fields `called_by`, `rejected` (fix-round) and `reread` (Claude) all
stay; `tests/support/invariants.py` = scenarios' latest on top of the features branch's four
false-positive fixes; native-skills' `LINKED` must point into bg-kimi's per-user Kimi home;
`uv.lock` is regenerated with `uv lock`, never hand-edited; the ReAct golden test must stay
byte-identical.

## 3. Written but never run, or known to be wrong, on the pushed heads

- `-fix-round`: the tests for the AutoMode rule, the carried-state bounds (three tests) and
  `test_agui_event_ids_are_the_same_on_every_replica` were never run. Three older tests in
  `tests/integration/test_agui.py` (about lines 88, 291, 314) expect ids `0..n` and the old
  `publish(derived=)` signature and will fail until updated. Pyright was not re-run after
  items 2–6.
- `-bg-kimi`: never run after their last edit, may be red: in `test_acp.py`
  `test_each_user_has_a_home_of_the_agents_own_an_overlay_of_the_teams`,
  `test_a_config_whose_allow_rules_take_no_rule_after_them_stops_the_run_before_its_prompt`,
  `test_a_session_the_agent_no_longer_holds_is_run_again_from_its_prompt`,
  `test_the_runs_skills_are_in_kimis_skills_folder_and_its_login_is_linked`; the `kimi_allows`
  unit test in `test_acp_adapter.py`; `test_acp_background.py` and
  `test_background_by_adapter.py` not re-run after the home change.
  `test_a_config_that_approves_on_its_own...[auto]` failed twice under load before the wait
  was changed.
- `-bg-claude-da`: other test files not re-run after the thinking-blocks commit; pyright not
  re-run; tested against SDK 0.2.182 from an overlay (the shared venv had 0.2.160).
- `-native-skills`: only `test_skills`, `test_acp`, `test_claude`, `test_kimi` ran green; one
  pytest timeout in an unknown test in the wider run.
- `-scenarios`: never run on this head: CONV-06, CONV-07, CONV-14, WF-01..06, the
  stack-dependent `test_s06`, CONV-04 across servers; LR-04/05 `function` variants not re-run.
- `-react-parity`: Claude and Kimi take the "rewind not supported" path untested.
- Strict xfails that must be removed when their fix is in the same tree (they fail as XPASS
  otherwise): tests marked "bg-claude-da", "react-background", "call identity"; harness
  `tests/features/test_mem_lifecycle.py` ×3 (memory fixes are on memory main);
  `tests/features/test_surf.py::test_agui_event_ids_are_the_same_on_every_replica`;
  `tests/features/test_har_adapters.py::test_a_date_question_refuses_an_answer_that_is_not_a_date`.
- Rolling upgrades: the journal model forbids unknown fields, so a server on older code
  cannot read a journal written with the new fields (`called_by`, `rejected`, `reread`).
- Kimi sessions now live per user under the XDG data folder: conversations begun before the
  change do not find their old session (they take the "session not resumed" path).

## 4. Open bugs and items NOT coded

**P0**
- **SCN-13**: a run that agent-runs ended `holder_lost` during a network cut keeps working
  and makes later writes. Root cause `src/trellis/harness/harness.py` about line 490:
  `Harness._lost` stops only CANCELLED runs. Fix: stop the attempt on any ended status,
  writing nothing more (as `pipeline.py` about line 876). Failing test (strict xfail):
  `tests/integration/test_scenario_findings.py::test_scn13_a_run_the_store_ended_stops_where_it_works`.
  The fold job was asked to make this fix as its last step; check whether it did.

**Harness**
- SCN-11: a LangGraph user's own `RetryPolicy` never retries a failed harness read (the
  bridge returns the failure as text). Route: an adapter hook that reads the node's policy
  and a raise in `convert/langchain.run` for failed reads only, with the harness's own read
  retries off there so one layer retries. Strict xfail in `tests/scenarios/test_s16`.
- BG-11: a background task's outcome is not told to the next turn when that turn lands on
  another server (Claude live session). Strict xfail in `tests/scenarios/test_s17_background_more.py`.
- Remote background-task chain depth (Deep Agents' own remote tasks are always depth 1).
- Claude: a fair share of the 16 live slots per tenant and user (one user with 16 working
  conversations can hold every slot for up to 6 h); sub-agents' thinking and server tools
  are not recorded; `compact_boundary` not recorded; `RemoteTrigger` hiding not shown on the
  real CLI (needs a claude.ai login).
- Kimi: PARTIAL at Kimi's step limit (D1); session and file cleanup when a hold lapses;
  probes on 2.6.2 (session lock, MCP reconnect, duplicate notifications, Read cap); the
  `Skill` call cannot be made to ask (needs Kimi approval config in the run home); a config
  with inline allow rules is still refused.
- LangChain / Deep Agents: prompt `cache_control` for routed Anthropic names (needs the
  resolved model handed to the model node; routing and attribution impact unverified); the
  memory context and the run's question after LangChain's own summarization; a client's own
  `max_retries` on a graph built without `ModelHooks`; `mcp_elicitation` mapping (adapter
  package not installed).
- Skills: a `SKILL.md` read through the `/skills/` route is not a recorded call (Deep Agents,
  ReAct); OpenAI `load_skill` / `exec_command` ask on every call unless the catalog declares
  their risk.
- Rewind: ended and cancelled runs (section 5); sub-agent runs have no history of their own;
  a rewound run re-records the question as a new turn in memory.

**Memory**
- A line added to an edited profile block from a memory later forgotten stays.
- Graph facts built from a deleted thread's documents are not removed.
- A decision written by the old version during a rolling upgrade, after migration 0039 has
  run, is not redacted. 0039 has only run on seeded test databases.

**Library limits, documented only (not to be coded without a new decision)**
MCP elicitation on Claude SDK 0.2.182; Kimi `AgentSwarm` and goal mode over ACP; Kimi
thinking / usage updates / session titles (no contract event); filtering Kimi's compaction
notices (would need wording rules); `middleware()` beside a user's same-named middleware and
swarm per-message attribution (each needs a new public argument); OpenAI sandbox memory
writes under governance (no SDK seam); computer use; Redis / Dapr session backends untested.

## 5. Decisions waiting for the owner

- **Rewind of ended runs** needs an additive agent-runs change: `finish` keeps the run's
  checkpoint when asked (today it is cleared when the run ends).
- Rewind: only the run's own user may rewind; `Agent.checkpoints` is tenant- and
  agent-checked, not user-checked. Confirm or change.
- A call approved in a framework's own gate (LangChain HITL) counts as the person's approval
  for a repeat after a rewind. Confirm or change.

## 6. Not started

- **One full test pass** on main: suite, coverage, matrix, contract tests, `tests/scenarios`
  and `tests/features` against a live stack (`scripts/scenarios/stack.sh`).
- **Performance job** (excluded for now by the owner): PERF-2 one heartbeat per process (may
  need an additive agent-runs endpoint), PERF-7 event batching, PERF-11 hold sweep, PERF-13,
  PERF-14 (checkpoint history makes the stored run state 8–13× larger and it is rewritten
  whole every 16th save).
- **Cleanup refactor** (last, by the owner's order): ARCH-3, 4, 6, 8, 9, 11, 12, 17, 18,
  21–25; DUP-1..10; test clones T1–T11; one `helping()` context manager for the six adapter
  copies of `helpers.closed(error, paused=helpers.suspended(error))`; `Adapter` protocol
  capabilities; boolean-flag parameters; Claude `Background` / `LiveSession` explicit state
  transitions; the two OpenAI code paths that build an SDK approval pause become one
  (no behaviour change; DECISIONS.md).
- **Haiku phase** (separate credit, about $40): Deep Agents profiles for routed model names,
  Deep Agents backend multi-turn accuracy, ReAct re-measure, memory accuracy items, Claude
  real-CLI checks, the gateway Responses fallback against a real Bifrost.

## 7. Build machine (ephemeral)

Worktrees under `/var/tmp/cons/` (`stage2` = the fold, `fixround`, `nskills`, `bgclaude`,
`bgkimi`, `memfix`, …); private test stack `/var/tmp/cons/live57` (Postgres
127.0.0.1:55957). The security review (SEC-R-1..11) is not in this repository because the
repository is public; SEC-R-1, 2, 5, 7, 9, 10, 11 are fixed in the work above.
