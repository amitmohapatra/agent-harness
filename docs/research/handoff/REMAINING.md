# What remains to be done (Trellis stack) — updated 2026-11-15

The owner asked for everything that is coded to go to main at once, without waiting for
tests, and for this file to list everything that is done, left, found, and still to be
tested. Nothing known is left out. Where a status is not confirmed it says so.

Companion files in this folder: `FOLD_LEDGER.md` (the running record: what merged when,
each job's report), `DECISIONS.md` and `ARCHITECTURE_DECISIONS.md` (decisions taken),
`ARCHITECT_BRIEF.md`, `ROUND3_NOTE.txt`, `GATES_NOTE.txt`, `WRAP_NOTE.txt` (the standard and
rules the coding jobs worked to), `REVIEW_architecture.md`, `REVIEW_performance.md`, `REVIEW_security.md` (the
full findings with file and line), `TEST_PLAN.md` (the test catalogue and the limitations
table), `inventory/` (what each library can do, and the collision and ReAct-parity design).

Status words: **done** = coded with its test and on main or in a branch below;
**reported** = the coding job said so, not re-checked by the architect; **partly**;
**open** = not coded; **unconfirmed** = was assigned, no report seen, must be checked.

---

## Session of 2026-11-15: what was done (read this before section 0)

Done on a fresh local checkout (macOS, 8 GB) plus a Linux container that mirrors CI
(Python 3.13, all extras, Kimi Code 2.6.2). GitHub Actions never ran: every run on main stays
queued (no runner), so local runs are the only check. Everything below is pushed to main.

**Verified state at the start:** `uv.lock` consistent (`uv lock --check`), ruff clean, pyright
0 errors over the whole repo. agent-contracts 526 passed; bifrost-sdk 291 passed; agent-runs 622
passed (one test made independent of machine speed: agent-runs `e1a2492`), its SDK 235 at 100%.
Harness on the Mac: 6,619 passed, 16 failed, 168 skipped, 99% coverage — the failures traced to
the bugs below, out-of-date tests, and Linux-only tests (`/proc`); most skips are missing extras
and the real Kimi CLI (`make install` is `uv sync` without extras; CI uses `--all-extras`).

**Bugs fixed (harness 2c4f495c … 5f3ee697; each with a test, most checked to fail without it):**
1. The hanging test (section 4) was a deadlock: `Turns.cancel` waited for a run that held its
   turn but had not begun its attempt (background runs got a conversation line at the fold).
2. SCN-11: a failed harness read inside a node whose `RetryPolicy` retries the error raises it
   once recorded; the node retries, the harness does not on top (`tools.base.NodeRetries`).
   The scenario's strict xfail is removed (WF-07 still to run on a live stack).
3. New: the "memory context reached nothing" warning had stopped firing for graph inputs with
   no messages (the SEC-R-1 copy of the input broke `prepared is input`).
4. Claude/Kimi live sessions: at most `KEPT_PER_OWNER` (4) of the 16 places per user.
5. LangChain's own `SummarizationMiddleware`: the run's question and memory context are sent
   ahead of the summary although it removed them from the state (`middleware._asked_again`).
6. Prompt `cache_control` for routed Anthropic names: `PromptCache` hands the resolved model on,
   so Deep Agents' memory block is cached (attribution headers are set before it).
7. Kimi at its step limit ends `PARTIAL`, no paid last answer (D1).
8. OpenAI `SandboxAgent` capability tools carry a risk (`CAPABILITY_SIDE_EFFECTS`): `load_skill`
   and `view_image` read, `exec_command`, `write_stdin`, `apply_patch` write.
9. A rewound run records its question once in memory (under the original run's question id).
10. A client's own `max_retries` counts on a graph without `ModelHooks` (`accounting.retries_itself`):
    the run store does not run the whole run again on top of the client's tries.
11. New (Linux CI, real Kimi CLI): after a resume Kimi gives the repeated call a new id, so the
    run's `before_tool` hooks were asked again; a hook's verdict is now journaled by the call's
    content and order (`Replay.by_content`).

**Owner decisions built (`DECISIONS.md`, last section):**
- Rewind of ended runs, opt-in: `h.wrap(rewindable=True)`. agent-contracts `27a818d` (a finished
  `RunRecord` may carry a kept checkpoint), agent-runs `7dfe8df` (`finish`/`cancel` take
  `keep_checkpoint`, `finish` a final `checkpoint`; SDK and `docs/openapi.json`), harness
  `913ddcc7` (the ending sends the journal as it ends; `Agent.checkpoints` / `Agent.rewind` take
  such an ended run; `LocalRuns` keeps it too). A sub-agent's run keeps none.
- `Agent.checkpoints(reader=)`: the run's own user only (admin path waits for a harness role model).
- The journal keeps unknown fields (`extra="allow"`) for rolling upgrades.
- `require_sandbox` (`TRELLIS_REQUIRE_SANDBOX`, off by default): `h.wrap` refuses Claude Code
  without its sandbox enabled and an `ACPAgent` without `sandbox=`.
- A framework gate's approval after a rewind counts (no code change: a rewound journal starts
  without earlier approvals).

**Tests brought up to date:** AG-UI event numbers (H-AGUI), the journal's per-write `effects`
record (rewind), queued memory writes spooled at close (SCN-6), macOS sandbox path grants, a
race in `test_i_an_outcome_read_into_a_turn...` (the task ends well within the turn).
agent-runs `e1a2492`: the rate-limit test independent of machine speed.

**Checked, already fixed on main:** SEC-R-3, SEC-R-4, SEC-R-6, SEC-R-8 (each has its test), so
`REVIEW_security.md` no longer exposes an open finding.

**Linux run (CI mirror: Python 3.13, all extras, Kimi Code 2.6.2):** 6,779 passed, 22 failed,
2 skipped. Of the 22: 19 pass alone (load: 3 workers in a ~7 GB VM), 2 were bug 11 (fixed), 1
the race above (fixed). The 89% coverage that run printed is an artifact: source files were
edited while it ran, so line numbers shifted (untouched files match the Mac's 99%). The 2 skips
need `aiosqlite`, not in the lock. **Not yet done: one clean full Linux run with no edits during
it** — it is the confirmation that everything holds together, coverage gate included.

**Why the Mac and Linux runs differ:** Linux is what CI runs. On the Mac `make install` is
`uv sync` without extras (CI: `--all-extras`) and the real Kimi CLI is absent, so ~166 tests skip
there; Linux-only tests read `/proc`; OpenAI's local sandbox uses `sandbox-exec` on macOS (the
tests grant their temp folder). Locally put `.venv/bin` first on `PATH` (the stand-in Claude CLI's
`#!/usr/bin/env python3` takes a system Python 3.9 otherwise).

**Out of scope by the owner:** the Kimi swarm (terminal-only `/swarm`, not reachable over ACP)
and Claude `RemoteTrigger` (stays disallowed): neither is used in the enterprise deployment.

**Still open after this session (in order):**
1. One clean full Linux run (no edits during it) and the coverage gate; fix what it shows.
2. Offline harness bugs: Kimi session/file cleanup when a hold lapses; Kimi `Skill` call cannot
   ask; Kimi config with inline allow rules refused (these two may be Kimi limits: document if
   so); Claude sub-agent thinking, server tools and `compact_boundary` not recorded; a
   `SKILL.md` read through `/skills/` not a recorded call; remote background-task chain depth.
3. Left by the owner for later: BG-11 and the live-stack suites (features, scenarios, real
   memory recall); the memory-service bugs (section 5); security tests SEC-01..07; performance
   (section 7); cleanup (section 8); the Haiku phase.
- Note for the cleanup: ARCH-6 says delete `_node_retries` (now `middleware.graph_retries`); the
  SCN-11 fix reads a node's `RetryPolicy` through it, so it cannot simply go.
- CI: every GitHub Actions run on main stays queued (no runner); restoring it is the fastest way
  to run the whole suite on each change.

## 0. Read this first

**Harness main is unverified.** agent-harness #257 and #259 were merged before any full CI
run passed, and later merges were made with only: conflicts resolved, ruff clean, the
package imports, pyright over the whole repo. No job ever ran the whole suite on a combined
head. Expect red CI on harness main: cross-branch test failures, the 100% line+branch
coverage gate of `trellis.harness`, and the tests in section 4 that were never run.

**Order of work next time**
1. Look at CI on harness main and fix what is red (start with `uv.lock`, which was merged
   as text and never regenerated, and the test that hangs, section 4). Every branch of the
   round is on main; there is nothing left to merge.
2. The one full test pass (section 9).
3. The open bugs and uncoded items (section 5), security first (section 6).
4. The performance job (section 7).
5. The cleanup refactor, last (section 8).
6. The Haiku phase (section 10), when credit is added.

**Owner decisions taken 2026-11-15** (rewind, who rewinds, framework approvals after a
rewind, rolling upgrades, sandbox default): `DECISIONS.md`, last section.

**Two things only the owner can settle**

1. **The security review is now in this repository** (`REVIEW_security.md`, added at the
   owner's instruction). This repository is public, so the findings that are not confirmed
   fixed (SEC-R-3, SEC-R-4, SEC-R-6, SEC-R-8; section 6) are publicly readable with their
   scenarios. Either confirm and fix those four first, or move the file to a private place
   and remove it here (it stays in the git history unless the history is rewritten).
2. **Three rewind decisions** (section 11):
   - Rewinding an ENDED run needs a small additive change in agent-runs: `finish` keeps the
     run's checkpoint when asked (today it is cleared when the run ends; cancelled runs
     need the same final save). Until then `Agent.rewind` and `Agent.checkpoints` work for
     paused runs only. Approve the addition or leave rewind for paused runs only.
   - Who may rewind: today only the run's own user (`reviewer` must be the run's user; a
     lead rewinding someone else's run is refused), and `Agent.checkpoints` is tenant- and
     agent-checked, not user-checked. Confirm or change.
   - Whether an approval given at a library's own gate (LangChain's human-in-the-loop
     middleware) counts as the person's approval when the same write repeats after a
     rewind. Today it counts. Confirm or change.

---

## 1. On main

| Repo | PRs merged this round | Verified |
|---|---|---|
| agent-contracts | #18 (round 1 contract changes), #19 (`RunRecord.placed_at`), #20 (`FeedbackSource.SYSTEM`) | #18, #19 CI green; #20 merged while CI was queued, 526 tests green locally |
| agent-runs | #35 (chat order on the server: place under the key's lock, holds, sweeps), #36 (`placed_at` on record and summary, `concurrency_key` list filter, date answers), #37 (one fixed set of enforced answer formats: date, email, ipv4, ipv6, regex, uuid) | CI green |
| agent-memory-service | #115 (round 1), #116 (504 documented on 69 operations, a decision keeps arguments as the tool redacts them, two feature-suite fixes), #117 (admin key forgets in its tenant, extracted user facts ask first, inventory lists chat statements, redaction at intake) | #115, #116 green; #117 green on 3.12, 3.13 still running at merge |
| agent-memory-service | #118 (a deleted thread leaves every read, profile lines follow their memories, migrations 0038 and 0039) | **no CI at merge**; item tests green, whole suite not run on the final head |
| agent-harness | #257 (OpenAI Agents fixes and SDK 0.24, a conversation's turns in order, scenario fixes, LangChain 1.5 / Deep Agents 0.7.38, ReAct background tasks, the feature suite, the harness side of `placed_at`), #259 (Claude, SDK 0.2.182), #262 (Claude round 2, ReAct parity and rewind, the fix round, LangChain leftovers, the scenario suite), #263 (CHANGELOG conflict markers removed), #264 (native skills, the SCN-13 fix, the fold's final head), #266 and #267 (Kimi 2.6.2 and one kept-process mechanism for Claude and Kimi); #258, #260, #261, #265 and later (these notes) | **not verified** (section 0) |

What those harness merges contain, by area (all **reported** unless noted):
- OpenAI Agents: library features pass through as the SDK does them; tool order kept;
  `raise_error` respected; Responses route learned from the gateway with a chat-completions
  fallback on 404/405/501 (not checked against a real Bifrost); "allow for this chat" sticks
  at the SDK's own gate; governed MCP server and route-learning model split into
  `adapters/openai_sdk.py`.
- Chat order: one line per conversation (`line.Line`, `concurrency_key`), `Holds(lost=)`,
  the place read from the run store, the line listed by its key.
- Scenario fixes: journal call identity `<run>:<digest>:<occurrence>` and record envelope;
  a run's open children end with it whatever its final status; SCN-1, 2, 3, 4, 6, 8.
- LangChain / Deep Agents: files and todos carried across turns; a sub-agent's
  `interrupt_on` passed through; `_node_retries` refuses loudly if LangGraph's internals
  moved; decision-model spend counted; `step_limit` default restored (12).
- ReAct background: one ownership gate for every task tool; admission cap, chain limit 3,
  1,024 attempts at work; bounded carried list (last 50 ended); one outbound URL check.
- Feature suite `tests/features` (live stack), `trellis doctor`, A2A card and CSV fixes,
  memory 504 handling.
- Claude (first list): `verbatim_prompts`; config folder per tenant and user, auto-memory
  off; a governance hook that fails blocks the call; approvals as Claude Code asks
  (ExitPlanMode, WebFetch, WebSearch, Skill); turn limit ends PARTIAL; the 16-process cap
  enforced across the server; concurrent Workflows share one run; tool risk table from CLI
  2.1.325; `RemoteTrigger` disallowed; tool-use id attribution.

---

## 2. The fold: complete

Every branch of the round is on harness main: the fold's head 76afbb3 (PR #264) and the
Kimi merge 9bd7380 (PRs #266, #267). Nothing is left to merge.

The Kimi merge (reported by its job): one kept-process ("live session") mechanism in
`bridges/live.py`, shared by Claude and Kimi. Generic parts of the Claude work moved there:
the 16-process cap counted over every agent of the same adapter kind in the server process
(`live.among`), the place reserved before the await (`LiveSessions.place`), fair
least-recently-used eviction (`_yielding`, `owner` / `used` on the shared `Keeper`); Kimi
now gets the same cap and eviction. Claude-only parts stayed in `claude_live.py`: workflows
and the shared run, `SessionCost`, tool-use-id attribution, a team's own hooks between
turns, removal of the unattended message buffer. `LiveSessions.open` became
`claude_live.opened()`. `adapters/claude.py` `_answered` goes through
`Runtime.asking_within`; `tools/bridge.py` keeps `kept=` beside the rewind guard and
`called_by`. Kimi's per-user home and the run home's links (AGENTS.md, config, sessions,
skills) combine: the links point into the per-user home.
Its targeted test run (the Claude and Kimi kept-process files, `test_one_question`,
`test_background_by_adapter`): 991 passed, 54 skipped, 1 deselected (the hanging test,
section 4), 1 failed and fixed (a Kimi allow-rule test that contradicted the documented
"ask rules come first"; the test was corrected, not the product). The four other Kimi tests
that had never been run passed. Not run for it: vulture and pylint duplicate-code.

What the Kimi work holds: Kimi 2.6.2 (2.1.x refused); asking mode set and confirmed before
any prompt; plans reviewed by a person with every option; allow-for-conversation mapped to
Kimi's allow-always; telemetry and auto-update off; a failed turn is ERROR with Kimi's
reason; mode confirmation waits on the event up to 30 s; a Kimi home per tenant and user
that turns team allow rules into ask rules; background work through a kept process.

On main from the fold (reported by the fold job): Claude round 2 (one `SessionCost`; the
replay rule; a team's own hooks between live turns; thinking and server-tool blocks); ReAct
parity (provider tools, plan mode, checkpoint history, `langgraph-swarm` handoffs,
`Agent.checkpoints` / `Agent.rewind` for paused runs); the fix round (identical writes,
AutoMode recording, carried-state bounds, AG-UI event ids, the approvals doc fix); LangChain
leftovers (no double model retry, the graph's own summary kept, provider tool fields kept);
the scenario suite and its fixes; native skills; `Tasks._watch` deleted (`Holds(lost=)` is
the one detector; `AsyncTasks._watch` keeps its remote read); a background task's runs get
their conversation's `concurrency_key`; the three AG-UI hub tests moved to the new
`publish(event, number)` signature; strict xfails removed for fixes now in the tree.

Fold rules if it is redone by hand: `CARRIED` = async_tasks, files, todos, ACTIVE_AGENT;
journal fields `called_by`, `rejected` (fix-round) and `reread` (Claude) all stay;
`tests/support/invariants.py` = scenarios' latest on top of the features branch's four
false-positive fixes; native-skills' `LINKED` must point into bg-kimi's per-user Kimi home;
bg-claude-da wins on Claude `Workflow`; `uv.lock` is regenerated with `uv lock`, never
hand-edited; the ReAct golden test must stay byte-identical; CI must install kimi-code 2.6.2
and claude-agent-sdk 0.2.182.

Other branches that exist and are NOT to be merged: `-chat-order-cancel` (parked, harness
ae30b77 / runs 89fe70f), every `-adapter-features-*`, `background-agents` and PR
agent-harness#169 (background work was rebuilt fresh; do not revive), and older experiment
branches under the same prefix.

---

## 3. Findings of this round and where each stands

### 3.1 Scenario and feature findings

| ID | What | Status |
|---|---|---|
| SCN-1 | a burst to one chat not answered in send order | done (chat order) |
| SCN-2 | call ids reused after a resume on a checkpointed graph | done |
| SCN-3 | replayed calls announced again | done |
| SCN-4 | next turn misses the previous one when memory writes take over 5 s | done |
| SCN-5 | docs said the run's user is an approval's default assignee | done (docs fixed, `-fix-round`) |
| SCN-6 | queued memory writes lost when the memory circuit opens | done |
| SCN-7 | a message over 4,000 characters lost its memory context | done (`-scenarios`) |
| SCN-8 | a message cancelled while waiting its turn still ran | done |
| SCN-9, SCN-10 | AG-UI: same run id twice started two runs; an empty message started a failing run | done (`-scenarios`) |
| SCN-11 | a LangGraph user's `RetryPolicy` never retries a failed harness read | **open** (section 5) |
| SCN-12 | always-allow on Kimi tripped INV-4 | done: the invariant was wrong, fixed (`-scenarios`) |
| SCN-13 | a run the store ended during a network cut keeps writing | done (#264): `Harness._lost` stops the attempt on any ended status; its test passes |
| SCN-14 | a failed upload left a half-written file when the store was full | done (`-scenarios`) |
| C-1 | memory OpenAPI listed no 504 | done (memory #116) |
| C-2 | `source: "system"` not in the contract's enum | done (contracts #20) |
| H-DOC, H-A2A, H-CSV | doctor false alarm, A2A card description, CSV header counted | done |
| H-ID | tool call id reused after a resume | done on main; test added in `-fix-round` |
| H-AGUI | event ids differ across replicas | reported (`-fix-round`), its test never run |
| F-DATE | a date answer accepted any string | done (runs #36, #37) |
| BUG-M1..M5 | memory service defects | done (memory #116, #117) |

### 3.2 Architecture review (ARCH-1..25; full text in `REVIEW_architecture.md`)

| ID | Finding (short) | Status |
|---|---|---|
| ARCH-1 | children left RUNNING when a run ends ERROR after a stall | done |
| ARCH-2 | a cancel from another process not honoured by the holder | **unconfirmed** (a parked branch `-chat-order-cancel` exists; check what main does) |
| ARCH-3 | two cascades end a run's children with different rules | cleanup (section 8) |
| ARCH-4 | four detectors of "ended elsewhere", four copies of "stop the attempt" | partly: `Tasks._watch` deleted at the fold; the four copies of "stop the attempt" are cleanup |
| ARCH-5 | a turn recorded RUNNING but never held blocks its line (birth grace) | **unconfirmed** |
| ARCH-6 | monkeypatch and private LangGraph imports to retry a graph's own tools | partly: fails loudly now; full removal is cleanup |
| ARCH-7 | `Tasks._done` forgets the wrong attempt | reported done |
| ARCH-8 | carried-state persistence lives in `pipeline.py` | cleanup |
| ARCH-9 | layering not as stated (bridges importing adapters, private members used across modules) | partly (Kimi moved names to `bridges/acp_live.py`); rest is cleanup |
| ARCH-10 | two liveness mechanisms in agent-runs (hold and lease) | cleanup, small step only |
| ARCH-11 | "a turn the chat started itself" implemented twice | cleanup |
| ARCH-12 | a detached helper's run ends by four paths | cleanup |
| ARCH-13 | Claude session cost tracked in four places | reported done (`-bg-claude-da`) |
| ARCH-14 | five async-task tools described in four tables | reported done (`task_tools.py`) |
| ARCH-15 | the line's place computed on both sides of the API | done (contracts #19, runs #36, harness) |
| ARCH-16 | Claude harness tool calls attributed by equal name and arguments | reported done (by tool-use id) |
| ARCH-17 | journal facts as parallel arrays; middleware writes journal internals | cleanup |
| ARCH-18 | process-global registries | partly (`_BESIDE` moved to `Runtime.beside`); rest is cleanup |
| ARCH-19 | small correctness gaps | first bullet reported done; rest cleanup |
| ARCH-20 | code kept for tests only, or with no reader | reported done for Claude; rest cleanup |
| ARCH-21 | library internals read by reflection with silent fallbacks | cleanup |
| ARCH-22 | `Kind` branches in the shared helper | cleanup |
| ARCH-23 | names that mislead | cleanup |
| ARCH-24 | behaviour gated on module constants with real timers | cleanup |
| ARCH-25 | module size (`claude_live.py` holds four responsibilities) | cleanup |

---

## 4. Written but never run, or known to be wrong, on the pushed heads

- `-fix-round`: the tests for the AutoMode rule, the carried-state bounds (three tests) and
  `test_agui_event_ids_are_the_same_on_every_replica` were never run. Three older tests in
  `tests/integration/test_agui.py` (about lines 88, 291, 314) expect ids `0..n` and the old
  `publish(derived=)` signature and fail until updated. Pyright not re-run after items 2–6.
  One unit test timed out once (`test_with_memory_a_tenants_catalog_publishes_through_the_background_writes`).
- Kimi: the five tests that had never been run after their last edit were run at the
  merge (four passed, one test corrected). `test_a_config_that_approves_on_its_own...[auto]`
  failed twice under load before the confirmation wait was changed; not seen since.
  Coverage of `kimi_home` and `_kimi_rules` not checked.
- `-bg-claude-da`: other test files not re-run after the thinking-blocks commit; pyright not
  re-run; tested against SDK 0.2.182 from an overlay (the shared venv had 0.2.160).
- `-native-skills`: only `test_skills`, `test_acp`, `test_claude`, `test_kimi` ran green; one
  pytest timeout in an unknown test in the wider run.
- `-scenarios`: never run on the final head: CONV-06, CONV-07, CONV-14 (failed earlier under
  memory-service load), WF-01..06, the stack-dependent `test_s06`, CONV-04 across servers
  (failed once in a combined run); LR-04/05 `function` variants not re-run. Whole-repo
  pyright not re-run after the last commit.
- On main after the fold: `tests/integration/test_claude_background_session.py::test_a_background_run_cancelled_before_its_first_attempt_began_stops_its_task`
  hangs (it also hung on the earlier base 0879e13); not fixed. `uv.lock` was merged as
  text and never regenerated (`uv lock` needs the sibling checkouts): CI's install step is
  its only check. After the fourth merge of the fold no pytest was run except
  `test_agui.py` and the SCN-13 test. Left in place: an xfail in `tests/features/test_har_more.py`
  (a mix that includes Kimi) and an `invariants.allow` line for SCN-13 in
  `tests/scenarios/test_s18_faults.py` (remove it once the scenario passes on a live stack).
- `-react-parity`: Claude and Kimi take the "rewind not supported" path untested.
- `-langchain-fixes`: no integration suite and no coverage run; three pyright errors in
  `tests/integration/test_react_async_subagent.py` on that branch (fixed on the fold's line).
- Memory #118: migration 0039 has run only on seeded test databases.
- Strict xfails that must be removed when their fix is in the same tree (they fail as XPASS
  otherwise): tests marked "bg-claude-da", "react-background", "call identity"; harness
  `tests/features/test_mem_lifecycle.py` ×3; `tests/features/test_surf.py::test_agui_event_ids_are_the_same_on_every_replica`;
  `tests/features/test_har_adapters.py::test_a_date_question_refuses_an_answer_that_is_not_a_date`.
  Strict xfails that stay until their bug is fixed: BG-11, SCN-11, SCN-13, the Kimi swarm test.
- Rolling upgrades: the journal model forbids unknown fields, so a server on older code
  cannot read a journal written with the new fields (`called_by`, `rejected`, `reread`).
  Decide: tolerate unknown fields on read, or require a full stop to upgrade.
- Kimi sessions now live per user under the XDG data folder: conversations begun before the
  change do not find their old session (they take the "session not resumed" path).
- Commits on some branches carry a "Claude Opus 5.5" co-author line instead of the Fable one
  (the coding jobs followed their own attribution setting); history was not rewritten.

---

## 5. Open bugs and items not coded

### P0
- None open that is known. SCN-13 was fixed in #264 (one in-process test run green; the
  network-partition scenario has not been re-run on a live stack).

### Harness, all adapters
- SCN-11: a LangGraph user's own `RetryPolicy` never retries a failed harness read. Route:
  an adapter hook that reads the node's policy, a raise in `convert/langchain.run` for
  failed reads only, and the harness's own read retries off there so one layer retries.
- BG-11: a background task's outcome is not told to the next turn when that turn lands on
  another server (Claude live session).
- Remote background-task chain depth: Deep Agents' own remote tasks are always depth 1.
- `RAN_BEFORE` repeat refusal in the shared bridge and what Kimi needs from it: note never
  written.
- The OpenAI SDK's own `needs_approval` pause: behaviour stays (DECISIONS.md); only the two
  internal code paths are to be merged, in the cleanup.

### Claude
- A fair share of the 16 live slots per tenant and user (one user with 16 working
  conversations can hold every slot for up to 6 h).
- Sub-agents' thinking and server tools not recorded; other server tools recorded with no
  output (the SDK parses only the advisor's result); `compact_boundary` not recorded.
- `RemoteTrigger` hiding not shown on the real CLI (needs a claude.ai login).
- `Background` / `LiveSession` explicit state transitions and boolean flags: cleanup.

### Kimi
- PARTIAL at Kimi's step limit with no extra model call (decision D1).
- Session and file cleanup when a conversation's hold lapses.
- Probes on 2.6.2 not done: session lock (two servers, one conversation), MCP reconnect,
  duplicate notifications, the Read cap.
- The `Skill` call cannot be made to ask (needs Kimi approval config in the run home).
- A config with inline allow rules is still refused (cannot be overlaid).

### LangChain / Deep Agents / ReAct
- Prompt `cache_control` for routed Anthropic names (needs the resolved model handed to the
  model node; routing and attribution impact unverified).
- The memory context and the run's question are lost after LangChain's own summarization.
- A client's own `max_retries` on a graph built without `ModelHooks` is not counted.
- `mcp_elicitation` mapping once the adapter package is a dependency.
- A `SKILL.md` read through the `/skills/` route is not a recorded call.
- OpenAI `load_skill` / `exec_command` ask on every call unless the catalog declares them.
- Rewind: ended and cancelled runs (section 11); sub-agent runs have no history of their
  own; a rewound run re-records the question as a new turn in memory.
- ReAct handoffs: a swarm's inner ReAct function tools are not the run's (members need
  LangChain tools).

### Memory
- A line added to an edited profile block from a memory later forgotten stays.
- Graph facts built from a deleted thread's documents are not removed.
- A decision written by the old version during a rolling upgrade, after migration 0039 has
  run, is not redacted.
- There is no single "erase a user" operation; it is forget each memory plus delete the thread.

### Library limits, documented only (need a new decision before any code)
MCP elicitation on Claude SDK 0.2.182; Kimi `AgentSwarm` and goal mode over ACP; Kimi
thinking, usage updates and session titles (no contract event); filtering Kimi's compaction
notices (would need wording rules, which the owner's rules exclude); `middleware()` beside a
user's same-named middleware and swarm per-message attribution (each needs a new public
argument); OpenAI sandbox memory consolidation writes under governance (no SDK seam);
OpenAI background agents (not wired, by the owner's decision); computer use (needs image
tool results and a display in the sandbox); Redis and Dapr session backends untested
(packages not installed).

---

## 6. Security

The full security review is `REVIEW_security.md` in this folder (added at the owner's
instruction; the repository is public, see "Two things only the owner can settle" above).
It was written against the branches as they stood before the fixes; the status of each
finding today is in this table:

| ID | Severity | Finding (short) | Status |
|---|---|---|---|
| SEC-R-1 | High | background-task tools acted on any run a task record named | done (one ownership gate) |
| SEC-R-2 | Medium | background tasks had no cap and could keep themselves going | done (admission cap, chain limit) |
| SEC-R-3 | Medium | a cancel from another process is not prompt and the holder ignores it | **unconfirmed** (same as ARCH-2) |
| SEC-R-4 | Low | any key of the tenant can take, extend or drop a hold | **unconfirmed, treat as open** |
| SEC-R-5 | Low | the journal's outcome marker was in-band | done (record envelope) |
| SEC-R-6 | Low | the journal read uses a permissive deserializer | **unconfirmed, treat as open** |
| SEC-R-7 | Low | Claude's carried approval fell back to the tool's name | done |
| SEC-R-8 | Low | an edited computer action may name any attribute | **unconfirmed, treat as open** |
| SEC-R-9 | Low | the outbound URL check was lexical | done (resolve and refuse, checked at task start) |
| SEC-R-10 | Low | tool arguments stored raw around the tool's redact paths | done (memory #116, #117; old rows #118) |
| SEC-R-11 | Low | a conversation's line listed by thread name | done (listed by its own key) |

Also fixed this round: end-user text no longer dispatched by Claude Code (`@path`,
`/command`); Claude config shared across tenants; governance hooks failing open on Claude;
a Kimi config that auto-approves silently disabling governance; an OpenAI session set on
`h.wrap` shared by every user (reported by the OpenAI job).

Security tests never written (TEST_PLAN section 5.12, all P0): SEC-01 tenant isolation fuzz
on every endpoint; SEC-02 prompt injection through tool output, memory and file content
(self-approval, secret exfiltration, disallowed tools); SEC-03 secret redaction in logs,
traces, events and memory; SEC-04 SSRF through webhooks, A2A push, MCP URLs, Deep Agents
`webhook` / `url`; SEC-05 sandbox escape (path traversal, symlinks, reads outside the run
folder); SEC-06 authentication and authorization on every route, key rotation, expired
keys; SEC-07 shell classifier bypass.

---

## 7. Performance job (not started; excluded for now by the owner)

Scale target used to judge: 1,000 concurrent runs per server, several servers, 200-step
runs, conversations of hundreds of turns. Regression tests assert counts (requests, calls,
open streams, bytes, plan nodes), never elapsed time. Full text: `REVIEW_performance.md`.

| ID | Finding (short) | Decision | Status |
|---|---|---|---|
| PERF-1 (High) | a waiting run holds one of the client's 100 connections | event streams get their own HTTP client; one follower per run-ahead per process; a cap | **unconfirmed** (assigned to chat order) |
| PERF-2 (High) | one heartbeat request and one row write per held run (50 req/s per server at 1,000 runs; a missed beat ends a live run) | liveness is a property of the process: one heartbeat per process, runs reference their holder; may need an additive agent-runs endpoint (report its diff first); meanwhile a rate-limited heartbeat is retried inside the hold window | open |
| PERF-3 (High) | the cap of 16 live Claude processes checked before an await | reserve the slot first, per server, evict least recently used | reported done |
| PERF-4 | one store read per background task per model call | at most one request per model call; back-off with jitter | reported done |
| PERF-5 | the carried task list never shrinks | keep unfinished plus the last 50 ended | reported done |
| PERF-6 | a turn's start does one `get` per earlier run | stop at the first older run; page by 50 | **unconfirmed** |
| PERF-7 | the run log sends one request per event | linger 100–250 ms or 100 events; flush at once on interrupt, finish, close | open |
| PERF-8 | every framework-run call's whole output journaled twice | record the output only where a replay needs it | **unconfirmed** (assigned to scenario fixes) |
| PERF-9, PERF-10 | claim and place scans | fixed earlier; keep the plan-based tests | done |
| PERF-11 | the hold sweep ends 100 runs per 5 s tick | repeat within a tick until a short batch | open |
| PERF-12 | an unattributed sub-agent call waits 5 s every time | remember the miss | reported done |
| PERF-13 | a paused sub-agent's run read every 10 s for 12 h; `_allowed` fetches every run of the thread (53 requests at 50 turns, 512 at 500); bind-parameter status in sweeps | with ARCH-4 and this job | open |
| PERF-14 (new) | checkpoint history makes the stored run state 8–13× larger (217 KB against 17 KB at 11 tool calls) and it is rewritten whole every 16th save | bytes written per run linear in steps (append-only segments), measured at 200 steps | open |
| new | carried state now costs a file-store list and reads at each turn's start and end for any graph with files or todos (default ReAct included) | measure; cache or skip when unchanged | open |

Performance tests never written (TEST_PLAN section 5.13): SSE p95 latency; 10,000-user
memory load; 1,000 concurrent chats; a 24 h soak (memory, file descriptors, processes,
journal growth); backpressure when memory or runs are slow; DuckDB memory under load.
Measured this round: `/v1/context` p95 used 7–14% of its 5 s deadline at 1 concurrent,
25–65% at 8, 48–68% at 16, no 504; the memory service holds about 3.5 GB resident.

---

## 8. Cleanup refactor (not started; last, by the owner's order)

A pure refactor: test ids, matrix cells and the ReAct golden requests unchanged; no public
argument or signature change. Target: one mechanism per job (table in
`ARCHITECTURE_DECISIONS.md`).

- **One owner each**: keep a run alive and learn it was ended elsewhere (`holding.Holds`
  with `lost=`; delete `Tasks._watch`, `AsyncTasks._watch` keeps only the remote read);
  stop an attempt (`Agent.stop_attempt`, DUP-2); end a run's children (`end_children` for
  every final status and for cancel, ARCH-3, DUP-3); open and close an attempt's helpers
  (`async with helping(run, name)`, replacing the six adapter copies of
  `helpers.closed(error, paused=helpers.suspended(error))`); end a helper's run
  (`Helpers.finished`, ARCH-12, DUP-8); ask a paused child's question through a turn
  (DUP-1); a turn the chat started itself (`Runtime.own_turn`, ARCH-11); store behaviour
  (one contract suite against `LocalRuns`, `FileRuns` and the agent-runs client);
  adapter capabilities on the `Adapter` protocol, no `getattr` probes; explicit state
  transition functions for run, helper, task and session states (Claude `Background` /
  `LiveSession` included).
- **ARCH items**: 3, 4, 6 (remove the harness retry loop around a graph's own tools,
  `Deferring`, `_Calling.owed/entries/tried`, `_again`, `_node_retries`, the `_arun_one`
  wrap; retries of read tools then come from LangChain's own `ToolRetryMiddleware` where
  the harness builds the stack), 8 (carried-state store out of `pipeline.py` into a
  repository class), 9 (layering: adapters import bridges, shared code never imports an
  adapter, no cross-module private members), 10 (small step: one predicate, one expiry
  function), 11, 12, 17, 18, 21, 22, 23, 24, 25.
- **Duplicates**: DUP-1..10 (DUP-4 ref-counted keyed lock, DUP-5 task record update, DUP-6
  Claude task message parser, DUP-7 token sums, DUP-9 hold against lease expiry, DUP-10
  `holder_lost` sweep text shared through contracts); the duplicated-knowledge table (the
  10 s poll period in three files, the `mcp__<server>__` prefix in 11 places, the "cut off"
  wording and codes, the conversation key built by hand once).
- **Test clones**: T1–T9 and T11 (T10 is left: the pair only looks alike).
- **Others**: boolean-flag parameters (`_options(kept=)`, `_live(unsent=)`,
  `charged(own=, taken=)`); the two OpenAI code paths that build an SDK approval pause
  become one; `gateways.py::_routed_class` complexity 11; a `uv.lock` regenerated cleanly.
- **Checks to run for it**: `ruff --select F,ARG,ERA,PIE,SIM,RET,C90`, `vulture
  --min-confidence 80`, `pylint --disable=all --enable=duplicate-code
  --min-similarity-lines=6 src tests` (21 duplicate hits before this round's branches;
  several jobs could not run vulture or pylint).

---

## 9. The one full test pass (not started)

Nothing below has run on a combined head.

| Suite | How | Notes |
|---|---|---|
| Harness lint, format, types | `make lint`, `make typings typecheck` | must be 0 errors over src, tests, examples, benchmark |
| Harness suite with coverage | `make test` | 100% line and branch of `trellis.harness`; last full local result is from before this round (6,119 passed) |
| Examples, docs, diagrams | `make examples`, `make docs-check` | |
| Feature matrix | `make matrix`, `make matrix-models` | last result 12,688 cells, from before this round |
| Contract tests | harness `tests/contract`, the two OpenAPI drift checks | against agent-runs and memory main |
| Feature suite | `make test-features` | live stack; 195 passed, 2 errors, 11 strict xfails before the fixes |
| Scenario suite | `make test-scenarios`, `scripts/scenarios/stack.sh reset / test / drop` | live stack; 398+ tests, never run whole on one head |
| Live stack | `tests/live`, the private stack | last result 87 passed, 0 failed, 48 skipped, before this round |
| agent-runs | its suite and SDK tests | 11 storage-emulator tests error in this environment, as on main |
| agent-memory-service | unit, integration, agent, e2e, contract | run migrations 0036–0039 up, down, up on a scratch database |
| agent-contracts | its suite | 526 passed |

Test plan areas and where each stands (`TEST_PLAN.md`, section 5; the per-ID status table is
in `docs/testing/TEST_PLAN.md` on `-scenarios` and in `docs/research/FEATURE_COVERAGE.md`):

- CONV (one chat, many messages, many chats): written; CONV-06, 07, 14 and CONV-04 across
  servers to be re-run.
- BG (background work): written for Claude, Deep Agents, ReAct, Kimi; BG-11 open.
- WF (workflows): WF-01..06 written and not re-run; WF-07 waits on SCN-11.
- LR (long-running): written; LR-04/05 partly re-run.
- APR (approvals), MEM, RUN, SCH, FS, SURF, CON: written (feature and scenario suites).
- REL (reliability): kill while running, paused, resuming, mid-workflow; memory down; runs
  slow or down; model error; limits; file store full; run store full; clock skew; duplicate
  webhooks; model 429 and overflow: written. Network partition: written, fails on SCN-13.
- SEC, PERF, COMPAT: not written (sections 6 and 7; COMPAT below).
- ACC: the Haiku phase (section 10).

Compatibility tests never written (TEST_PLAN section 5.14): Python 3.12 and 3.13; the
minimum and maximum of each library range (langgraph, langchain 1.5, deepagents 0.7.38,
openai-agents 0.24, claude-agent-sdk 0.2.182, kimi-code 2.6.2, ACP 0.12.1, langgraph-swarm
0.1); runs paused on an older library version resume after the upgrade; contracts version
mixes; old and new harness servers side by side on one run store (the journal fields in
section 4).

Not runnable in the build environment: tests that need Docker, Qdrant, OpenFGA, the GCS
emulator or model weights beyond the frozen set.

Exit criteria (TEST_PLAN section 8): every P0 and P1 case passes on every applicable
adapter with skips explained; INV-1..10 hold on every test; no test weakened; matrix, suite,
scenarios, features and live show no unexplained difference; the limitations table accepted
by the owner.

---

## 10. Haiku phase (real model; needs separate credit, about $40 for the harness, $20–25 for memory)

Rules: Haiku only; the key is used from the environment and never printed, logged or
committed; private test stacks only.

Harness
- ReAct accuracy re-measured on the 357-turn set, plain and with the Deep Agents backend
  (the 352/357 and 356/357 figures on record were not re-derived from data this round).
- Deep Agents backend multi-turn accuracy now that files and todos are carried across turns.
- Deep Agents model profiles for routed model names, and the general-purpose helper's
  profile settings.
- `ReAct(skills=...)`: the helper's instructions gained Deep Agents' skills section; measure
  only if a benchmark variant uses skills and delegates.
- The model actually uses the background tools for long work, and understands steering.
- Claude against the real CLI: built-in tools, background sub-agents, Workflows, the live
  session, sandbox, compaction, skills, hooks between turns (all coverage so far is the
  stand-in CLI plus scripted probes).
- Kimi against the real endpoint beyond the scripted one.
- The OpenAI Responses fallback against a real Bifrost gateway.
- Cost and usage against the provider's bills (the new `SessionCost`, decision-model spend).

Memory (parked by the owner until this phase)
- Live check of draft PR agent-memory-service#114 (about $0.40).
- LoCoMo with everything on, with and without restating (decides draft PR #113).
- Finish LongMemEval facts-only (67 of 100 ingested, unanswered) and the 22 missed questions
  with everything on.
- Then: closest-sentence scoring, profile-repeat removal, the breakdown of the 58 LoCoMo misses.
- Scores on record: LongMemEval no model 82.2%; LoCoMo no model 87.5% (86.0% categories
  1–4), facts-only 87.1% (85.6%), beliefs-only 87.4% (86.4%).

---

## 11. Decisions waiting for the owner

- **Rewind of ended runs** needs an additive agent-runs change: `finish` keeps the run's
  checkpoint when asked (today it is cleared when the run ends). Cancelled runs take a
  different ending path and need the same final save.
- Rewind: only the run's own user may rewind; `Agent.checkpoints` is tenant- and
  agent-checked, not user-checked. Confirm or change.
- A call approved in a framework's own gate (LangChain HITL) counts as the person's approval
  for a repeat after a rewind. Confirm or change.
- Rolling upgrade and the journal's new fields (section 4).
- OpenAI 0.23+ hides tool error details from the model by default: kept as the library's
  default; never decided.
- Sandbox as the default for the Claude and Kimi adapters.

---

## 12. Older backlog, untouched this round

- Files and sandbox: one file store, capability-driven export, auto-install from an index,
  lifecycle and reaper.
- A break-it round on the complex areas not yet attacked.
- Earlier ordered backlog: docs update, Deep Agents token overhead, rate-limit default, SSE
  p95, a doctor warning for an MCP bypass, unverified references, schedule misses.
- Parked list: write-asks-by-default with allow-for-conversation, OpenAI in-place resume
  (1 in 7 still re-pauses), Deep Agents small-window error and hist-long-01 recheck, docs
  (skills example for OpenAI, Word and Confluence), memory catalog default "ask", temp
  databases.
- Memory: beliefs for named people who are not users; a scale proof (10,000-user load test,
  a LongMemEval slice, backpressure); the LongMemEval and LoCoMo benchmarks parked by the owner.
- The limitations table L1–L47 in `TEST_PLAN.md` section 6 has not been re-read against
  this round's fixes: L1, L5, L23–L35, L42, L46 are reported fixed; L36, L38–L41, L43–L45,
  L47 are partly fixed and must be re-checked row by row; L2, L3, L7, L9–L16 are by design
  or library limits; L17, L18, L37 wait for the Haiku phase; L19–L22 are environment limits.

---

## 13. Build machine (ephemeral) and where everything was saved

Nothing that is a note, a decision, a review or a plan exists only on the build machine any
more: all of it is in this folder (`earlier/README.md` lists the older material: two
reports, the Confluence documentation set as an archive, and earlier memory-experiment
notes). Code that existed only there is on backup branches named in `earlier/README.md`.

Still only on the build machine, by choice (data, not notes): raw experiment outputs, test
result files, benchmark databases, the private test stack `/var/tmp/cons/live57` (Postgres
127.0.0.1:55957), model weights, virtual environments, the Claude SDK 0.2.182 overlay. All
can be rebuilt; none is needed to continue.

Worktrees of this round, for reference if the machine is still up: under `/var/tmp/cons/`
(`stage2` = the fold, `kimimerge`, `fixround`, `nskills`, `bgclaude`, `bgkimi`, `memfix`,
`co2-*`, `handoff`). Every one is clean and pushed.
