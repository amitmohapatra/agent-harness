# Research: index

Every report in this folder, what it found, its date, and whether it is current. A report is
**current** while its finding still holds on main; **superseded** reports are in
[archive/](archive/) with a note on what replaced them. Open work from all of them is in one
place: [handoff/REMAINING.md](handoff/REMAINING.md).

Raw outputs of the runs (per-case results, request logs, transcripts, recordings) are not on
main: they are on this repository's `data/research-archive` branch, at the paths they had, and
each run's data folder keeps a `MANIFEST.md` listing every file with its sha256 and how to
restore it. Each run keeps its README (the summary and tables) and the scripts that produced it.

## Standing references

| Report | What it is | Date | Status |
|---|---|---|---|
| [DESIGN_RULES.md](DESIGN_RULES.md) | the rules for every harness and `ReAct` change, and the questions each PR answers | 2026-10 | current |
| [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) | the big-model judge protocol: a change is kept only when context sufficiency and accuracy do not drop | 2026-10 | current |
| [FEATURE_COVERAGE.md](FEATURE_COVERAGE.md) | every documented feature, its purpose, and the test that proves it at each level (the `plan(...)` ids of `tests/features`) | 2026-11 | current |
| [QUERY_ENGINE.md](QUERY_ENGINE.md) | `query_result` on SQLite, DuckDB, Polars and pandasql: identical answers at 10k–2M rows; DuckDB chosen for bounded memory and speed ([ADR 0007](../adr/0007-query-engine.md)) | 2026-11-08 | current |

## Runs and checks of 2026-11-15

| Report | What it found | Status |
|---|---|---|
| [compare-2026-11-15](compare-2026-11-15/README.md) | every adapter on Haiku on the product path (memory on, agent-runs as the run store): accuracy ties at the top (`ReAct`, `ReAct` + Deep Agents, `create_deep_agent`, OpenAI Agents, `create_agent` within 1–2 turns); Claude and Kimi behind for their shells pausing under governance; LangGraph without `middleware()` last; `ReAct` cheapest and fastest | current |
| [haiku-2026-11-15](haiku-2026-11-15/README.md) | Haiku acceptance of main: `ReAct` 117–118/118 strict, Deep Agents multi-turn 24/24 on every arm, the real Claude CLI and Kimi Code pass; a paused Claude attempt's spend now charged to the run | current (its `ReAct` accuracy is re-measured in compare-2026-11-15) |
| [deepagents-planning-2026-11-15](deepagents-planning-2026-11-15/README.md) | Deep Agents plans with its own `write_todos` beside `middleware()` (the harness's `Planning` stays out); the window sweep at 8,000–128,000 tokens keeps the list in view on every later turn | current |
| [sdk-full-stack-2026-11-15](sdk-full-stack-2026-11-15/README.md) | each SDK's own features alone, wrapped, on the real stack and with `middleware()`, on Haiku: the bugs fixed on `-sdkfull`, and the ones that need a decision (its "Bugs found, not fixed": B1 and on) | current |
| [live-direct-2026-11-15](live-direct-2026-11-15/README.md) | the live suite with Bifrost optional: 155/156 through Bifrost, 12/12 directly; the one failure is Haiku not honouring a forced named `tool_choice` | current |
| [onerule-2026-11-15](onerule-2026-11-15/README.md) | one approval rule, live: an MCP tool with no annotations asks; "Always allow" lets the user's next conversation run it unasked, on `ReAct` and the Claude Agent SDK | current |
| [refactor-plan-2026-11-15.md](refactor-plan-2026-11-15.md) | the targeted refactor plan (one owner for stopping, one cascade, one `helping()`, duplication budgets in CI) and its re-check script ([recheck.py](refactor-plan-2026-11-15/recheck.py)) | current: not started (REMAINING section 8) |

## The handoff

[handoff/](handoff/) is the state of the work for whoever continues it.

| File | What it is |
|---|---|
| [REMAINING.md](handoff/REMAINING.md) | everything done, left, found and still to test; the one list of open items |
| [DECISIONS.md](handoff/DECISIONS.md), [ARCHITECTURE_DECISIONS.md](handoff/ARCHITECTURE_DECISIONS.md) | the decisions taken, and the round's working orders |
| [ARCHITECT_BRIEF.md](handoff/ARCHITECT_BRIEF.md), `GATES_NOTE.txt`, `WRAP_NOTE.txt`, `CLEAN_CODE_NOTE.txt` | the standard and the rules the coding jobs work to (jobs still read these) |
| [REVIEW_architecture.md](handoff/REVIEW_architecture.md), [REVIEW_performance.md](handoff/REVIEW_performance.md), [REVIEW_security.md](handoff/REVIEW_security.md) | the reviews' full findings, with file and line |
| [TEST_PLAN.md](handoff/TEST_PLAN.md) | the test catalogue and the limitations table |
| [FOLD_LEDGER.md](handoff/FOLD_LEDGER.md) | the running record: what merged when, each job's report |
| [inventory/](handoff/inventory/) | what each library can do; the collision and `ReAct`-parity design |

Retired (listed with their sha256 in [handoff/MANIFEST.md](handoff/MANIFEST.md)): the notes `STOP_NOTE.txt`, `CONTINUE_NOTE.txt`, `ROUND3_NOTE.txt` and the older
material of `handoff/earlier/` (two reports of 31 Oct and 1 Nov, the Confluence documentation
set, memory-experiment notes) are on `data/research-archive`; what still applied is in
REMAINING.md (sections 12 and 13) and DECISIONS.md.

## Archive (superseded)

| Report | Date | Replaced by |
|---|---|---|
| [archive/IMPROVEMENT_BACKLOG.md](archive/IMPROVEMENT_BACKLOG.md) | 2026-10-30 to 2026-11 | its open rows are in REMAINING.md section 12; kept as the record of what landed, with the evidence |
| [archive/SCENARIO_COVERAGE_AUDIT.md](archive/SCENARIO_COVERAGE_AUDIT.md) | 2026-10-30 | a snapshot of main then; its gaps went to the backlog, the open ones to REMAINING.md |
| [archive/SELF_EXPLANATION.md](archive/SELF_EXPLANATION.md) | 2026-10-30 | shipped as arm D (the steps list as the run's record); pilot arm E is open in REMAINING.md |
