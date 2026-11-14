# Fold ledger (architect) — what is on main, what is in flight, what is owed

Updated 2026-11-14 19:45 UTC.

## On main
- agent-contracts#18 (57139dc), agent-runs#35 (15c419b).

## In PR
- agent-memory-service#115 (final, 0f20438): CI 3.12 green, 3.13 running. Merge when green.
- agent-harness#256, branch `-to-main` (worktree /var/tmp/cons/tomain, head 0879e13):
  openai-fixes 31f2250 + chat-order 90451bd + scn-fixes f019315 + type/test fixes.
  First CI run failed at pyright (20 errors, 7 real test failures in test_bridge_helpers
  from scn-fixes' `Helpers.closed(paused=)`); fixed in 0879e13; CI re-running.

## Stage 2 (fold job running, branch `-to-main-2`, worktree /var/tmp/cons/stage2)
Merging onto `-to-main`: langchain-fixes 444f1e3, react-background 8ce1034, features cb80b3c.
Needs memory `-features` f1f166a on memory main first (504 in OpenAPI; harness tests fail
against the old document) → open memory PR #2 after #115 merges.

## Still coding (fold after they report; each told to pass /var/tmp/cons/review/GATES_NOTE.txt)
bg-kimi, bg-claude-da, react-parity, scenarios. Held: native-skills (23e82ba WIP).
Fold rules: bg-claude-da wins on Claude `Workflow` over react-background 115348a;
openai-fixes' `governed_capability` wins over native-skills' `SANDBOX_TOOLS`;
scenarios: delete its test_scn4_*/test_scn6_*, unmark SCN-1/2/3 xfails;
`tests/support/invariants.py`: features version + scenarios' newer (c7d6f17) merged by hand.

## Fix round after the fold (found by jobs, not yet fixed)
- H-ID (harness tools/bridge.py:359): on Claude and Kimi a resumed run reuses tool call id
  `run_x:1` for different calls (step restarts each attempt). Fix: use the call's
  idempotency key when the framework gives no id. Two feature tests error on INV-10.
- LangChain identical writes in one step: strict xfail "call identity lands at the fold";
  scn-fixes' `journal.call_key` is the mechanism → fix and unmark.
- LangChain AutoMode block recording is switched off after any HITL answer (coarse).
- Carried state over 256 KiB is dropped whole (task list with it); deletion markers accumulate.
- SEC-R-10 residual (memory): raw args stay from intake to projection; redact at intake.
- Memory strict xfails: BUG-M1 (agent forgets extracted facts without asking), BUG-M3
  (inventory misses thread-anchored statements), BUG-M4 (tenant admin key cannot forget).
- H-AGUI (event ids differ across replicas on reconnect), F-DATE (runs SDK answers.py:57
  date question accepts any string).
- React-background: chain depth covers harness tasks only (remote tasks always depth 1).
- OpenAI `needs_approval`: DECIDED (DECISIONS.md): behaviour stays; cleanup job merges the two internal code paths only.
- `gateways.py::_routed_class` complexity 11 (pre-existing).
- Then: ONE cleanup job (ARCH-3,4,6 full,8,9,11,12,17,18,21–25; DUP-1..10; T1–T11;
  `helping()`; Adapter protocol), performance job (PERF-2,7,11,13), native-skills.
- Then: one full test pass (suite, matrix, contract, scenarios, features, live57), then Haiku.

## Lessons applied
- Jobs must run the repo's gates (ruff, pyright incl. tests, docs check, all callers of a
  changed signature) before reporting. CI (GitHub) runs the full suite + 100% coverage +
  matrix at no credit cost: use it as the full-suite runner; the local machine is too loaded.
- /var/tmp/rx/models restored (moved from feat-tmp). reactbg-tmp/venv now links to
  rparity-tmp/venv (langchain 1.5.0, deepagents 0.7.38, openai-agents 0.24.0).

## Update 20:05 UTC
- User (20:00): "Fix any small gap or any issue asap … keep pushing those finished … refactoring
  can be done last … make sure what we planned, not a single penny is missed."
  → the two additive runs-API changes for chat order are approved (placed_at; list filter
  by concurrency_key).
- agent-memory-service#115 MERGED (e43fc3b). agent-memory-service#116 opened (features
  f1f166a: 504 in OpenAPI, SEC-R-10, BUG-M2, BUG-M5); subscribed.
- New jobs (Opus): chat-order-2 (ARCH-15, SEC-R-11, F-DATE; branches `-chat-order-2` in
  runs/contracts/harness; worktrees /var/tmp/cons/co2-*); memory-fixes (BUG-M4, M1, M3,
  SEC-R-10 at intake; branch `-memory-fixes`, worktree /var/tmp/cons/memfix, on top of #116).
- Next harness fix job, to start on top of stage 2 when the fold pushes: H-ID, identical
  writes via call identity (unmark xfail), AutoMode recording rule, carried state > 256 KiB,
  H-AGUI, remote-task chain depth.
- Order after that: fold bg-claude-da, bg-kimi, react-parity, scenarios as they report →
  native-skills → cleanup (refactor last, per user) → performance job → one full test pass → Haiku.

## Update 20:15 UTC — order confirmed by user
"Cleanup and refactor we will do in last but other finish first and push to main."
Order: PRs in flight → stage 2 fold → chat-order-2, memory-fixes → harness fix round →
bg-claude-da, bg-kimi, react-parity, scenarios → native-skills → performance job
(PERF-2, 7, 11; PERF-13 parts that do not need ARCH-4) → LAST: cleanup/refactor
(incl. PERF-13 remainder, OpenAI approval plumbing) → one full test pass → Haiku.
- 20:03 UTC: agent-memory-service#116 MERGED (e3d68cb). Memory main now has the 504 document. agent-harness#256 CI still running (past lint and type check).

## 20:20 UTC — bg-claude-da reported done (b211b75), queued as stage-2 merge 4
Gates passed by the job: ruff, pyright (0), docs. Not run: full suite, coverage, matrix.
Needs claude-agent-sdk 0.2.182 (overlay /var/tmp/cons/bgclaude-tmp/sdk182).
Claude round 2 (owed, after the fold; goes into the harness fix round):
- ARCH-13 `SessionCost`: DECIDED — one object; a total lower than the last seen is a reset
  (SDK zeroes totals on a conversation reset): charge the whole new total, never
  `max(0, Δ)` (that under-counts spend against the budget). Test both sides of a reset.
- Team's non-PreToolUse hooks between live turns (probe the real CLI with the scripted model first).
- Replay rule: never replay reads across a write.
- Thinking and server-tool blocks recorded (check what the transcript can carry).
- Fairness residue: one user with 16 conversations with background work can hold every
  live place for up to 6 h → per-user share of the cap.
- Cleanup phase: `Background`/`LiveSession` transition functions; boolean flags
  `_options(kept=)`, `_live(unsent=)`, `charged(own=, taken=)`.
- Not reachable (library limit, document only): MCP elicitation on SDK 0.2.182.
- RemoteTrigger: hiding not directly shown by the real CLI; test checks the flag only.
- Left on disk: /var/tmp/cons/bgclaude-tmp (~480 MB; the job's cleanup was denied by a
  safety check; keep sdk182 until the fold's venv has 0.2.182).

## 20:30 UTC — scenarios reported done (272c1d7), queued as stage-2 merge 5
398 scenario tests collect; never run whole on one head (needs a live stack) → the one
full test pass runs it. Product fixes included: SCN-7 (long message to memory), SCN-9
(AG-UI same runId twice), SCN-10 (AG-UI empty message).
Owed (fix round / test round, nothing dropped):
- SCN-11: a LangGraph `RetryPolicy` never retries a failed harness read (the bridge returns
  the failure as result text; the node never raises). Per ARCH decision 1 a user's own
  retry policy must work → fix; strict xfail in test_s16 (WF-07).
- SCN-12: always-allow on Kimi passes but breaks INV-4 ("ran though it was rejected"):
  diagnose product vs invariant (with bg-kimi's fold).
- SCN-5: docs/api.md says the run's user is the approvals' default assignee; code has none
  (runtime.py:737) → make docs and code agree (decide which).
- C-2: harness files run outcomes with `source: "system"`, not in agent-contracts'
  `FeedbackSource` (agent.py:1332) → use an existing value if one fits, else additive enum.
- Tests not written: BG-08 (scenario level), BG-11 (live Claude session moved across
  workers), REL: disk full, clock skew, network partition, duplicate webhooks, model
  429/overflow. LR-04/05 queued-long-task test never ran. Remaining sleeps in CONV-06,
  LR hold watching, s12 cancel wait → gates.
- Not re-run after edits: CONV-06, CONV-07, CONV-14 (failed under memory load), WF-01..06,
  test_s06 stack-dependent tests; CONV-04 across servers failed once in a combined run.

## 20:40 UTC — react-parity reported done (2265673), queued as stage-2 merge 6
Gates passed by the job: ruff, pyright (0), docs; 1421 passed on touched/caller files.
Not run: full suite, coverage, matrix. Golden test byte-identical.
Architect decisions on its open points:
- Provider (hosted) tools: CONFIRMED — a provider tool whose governance decision would ask
  a person is withheld from that model call with one `PROVIDER_TOOL_HELD` warning per run
  (no approval can stand before a call the provider makes inside its step: fail closed).
  Default risks confirmed: search / fetch / file search = read; code execution = write,
  announced; the catalog overrides; a dict a client would have to run is refused.
- Rewind: history is kept and readable (done). Rewinding a run needs new public surface:
  `Agent.rewind(run_id, checkpoint_id, *, values=None, reviewer, tenant=None)`,
  `Agent.checkpoints(run_id, tenant=None)`, and a journal ledger so a non-read call the run
  already made asks a person again. Additive, and the user asked for the feature
  ("we should have this feature") → APPROVED for the fix round, after the fold.
- PERF-14 (new, performance job): checkpoint history makes the journal graph 8–13× larger
  (11 tool calls: 217 KB vs 17 KB) and it is rewritten whole every 16th save → bytes
  written grow faster than linearly with steps. Target: bytes written per run linear in
  steps (append-only segments), measured at 200 steps; assert counts/bytes, not time.
- Handoffs limits (document only, no new argument): a swarm's inner ReAct function tools
  are not the run's (members need LangChain tools); no per-message attribution by name.
- General-purpose helper profile settings + routed-name profiles → Haiku phase.
- Computer use: not now (needs image tool results, a display in the sandbox, a beta route).
- vulture / pylint duplicate-code not run by this job (not installed; uvx denied).

## 20:50 UTC — chat-order-2 reported done
Heads: contracts 6bc8396 (PR agent-contracts#19 opened, subscribed), runs 8779f37
(cd353f8 ARCH-15+SEC-R-11, 8779f37 F-DATE; PR after contracts#19 merges), harness 901eb60
(queued as stage-2 merge 7; needs runs main first for CI).
Wire additions: `RunRecord.placed_at` / `RunSummary.placed_at` (AwareDatetime|None,
read-only); `GET /v1/runs?concurrency_key=`; SDK `RunsClient.list/iterate(concurrency_key=)`.
Found and fixed on the way: the line listed RUNNING then QUEUED, so a run claimed between
the two listings was missed → IN_LINE is now (QUEUED, RUNNING).
Notes: agent-runs `created_at` is set once at insert to the place and never rewritten (left).
F-DATE side effect: every format jsonschema knows is now enforced (email, uuid, …); formats
whose checker needs an optional library stay unchecked → environments could disagree
(inference) → pin the checker's optional deps or enforce a fixed list (fix round, small).
Commits on these three branches carry a "Claude Opus 5.5" co-author line (the job's own
attribution reminder); history not rewritten.
No version/pin bump made (runs pins trellis-contracts>=0.7.1; CI uses the sibling checkout).
- 20:22 UTC: agent-contracts#19 MERGED (6d3db4b). agent-runs#36 opened (chat-order-2: placed_at, concurrency_key filter, F-DATE), subscribed.
- 20:27 UTC: agent-runs#36 MERGED (8923113). Runs main now has placed_at, the concurrency_key filter and F-DATE. Harness chat-order-2 (901eb60) rides stage 2.

## 21:00 UTC — memory-fixes reported done (432a168); PR agent-memory-service#117 opened, subscribed
Fixed with tests: BUG-M4 (admin key forgets in its tenant), BUG-M1 (extracted user facts ask
first; migration 0036), BUG-M3 (inventory lists chat statements; migration 0037 GIN index),
SEC-R-10 at intake (one owner). Gates by the job: ruff, pyright 0 (repo config), OpenAPI drift
none, 523 + 3533 passed; 1 unit failure is a benchmark test that refuses a `memfix_` DB name.
After it is on main, remove strict xfails in harness tests/features/test_mem_lifecycle.py:
`test_forgetting_a_fact_read_from_the_users_words_asks_first`,
`test_the_users_inventory_lists_what_they_said_in_a_chat`,
`test_erasing_a_user_leaves_nothing_any_read_returns` (may still fail: it asserts message rows
have deleted_at set; deleting a thread is a soft delete that leaves its messages → decide:
fix the test's expectation or delete messages with the thread).
Owed (memory, next round):
- Erase leaves a document attached in the deleted thread (still in context and recall) and a
  profile line learned from the user; there is no route to delete a document.
- Decisions stored before this version keep raw arguments; old rows with a client-sent
  `args_hash` are now trusted as the call's hash (low).
Notes saved to the public repo branch `claude/cool-maxwell-w5avo8-handoff`
(docs/research/handoff/); the security review is NOT there (public repo), only in
/var/tmp/cons/review/security.md.
