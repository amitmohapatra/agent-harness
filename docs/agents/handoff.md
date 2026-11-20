# Handoff: where the platform stands and the phase-wise plan

Written 2026-11-18, at the end of the regression and hardening round on branch
`claude/amazing-meitner-22ul02` (agent-harness, agent-contracts, agent-runs,
agent-memory-service, bifrost-sdk, ai-registry, mcp-sdk). Read this first when you pick the
work up.

## The principle

**Like Claude for the end user, enterprise governance for everything else.**

* The end user gets a conversation: natural questions with options, approvals only for actions
  in their own name on their business data (an order, a refund, an email, a price, their
  schedules), plan mode, skills.
* The framework (this harness) decides *how* and enforces; the company using it decides *how
  much* (numbers in settings), *where* (delivery connectors) and *what it looks like* (its UI).

## What this round delivered

| Area | Change |
|---|---|
| Governance | One approval rule on every adapter; read-only shell commands run unasked everywhere (the OpenAI `SandboxAgent` shell made consistent); an administrator's explicit catalog risk wins over an announced write; Claude and Kimi plan mode hold until the plan is approved |
| Asking | An answer that is an option's value or label is that option; any other answer reaches the model as the user's words (never refused, never guessed); the "when to ask" guidance reaches every adapter that has a question tool, memory on or off |
| Graphs | `h.harness_tools()`: a graph that binds its tools at build time gets the same harness tools as every other adapter (`ask_user`, schedules, memory, the gateway's MCP tools) |
| Runs | A dead worker's model spend is saved as it accrues (a small spending segment after each reply); a run's end lists its children from an index, not a scan |
| Memory service | Tool search ranks among the run's own tools (indexed `tool_name`); verify answers at its deadline; database connections are not held while waiting on locks; request priority over background jobs, with a bound so jobs are not starved |
| Kimi | A session that never reached the harness's tools is started again, then fails, rather than answering without them |
| Examples | All 55 live examples pass against real services (the setup is described below) |
| Benchmark | All 139 turns of the comparison set strictly scored (was 119) |

Decisions taken with the owner during the round:

* `TRELLIS_HOST_EXECUTION` stays `allow` by default; the Docker sandbox is unchanged.
* No infrastructure-specific approval rule: shells and the network follow the same catalog and hooks as any tool.
* The spend-per-reply save is kept.
* The memory lock queue and request priority are kept.
* A graph binds `ask_user` in every run; build an unattended graph without it.

## Phase-wise plan

Each phase: design (gstack `/plan-eng-review` for anything that changes behaviour or public
API), tests first, build, docs and CHANGELOG, the full gate, then merge. One phase at a time.

### Phase 0 — performance (next)

On a fixed machine (staging, or a 16-core VM), never this 4-core container:

* the harness's overhead (`make bench`, `make bench-load`) with a new baseline recorded there;
* the memory service: latency budgets, context lookup, load at rising rates, the network
  gates, model concurrency; pool sizing for each process role (`combined` exceeds the
  connection budget by design: production runs API and worker apart);
* agent-runs: bursts of starts, resumes and schedule fires; claims at 10k and 100k queued runs;
  event-log paging;
* end to end: many users at once with tools, memory and schedules; a soak of several hours.

The stress scenario (`test_s14::test_stress_many_chats_on_three_servers`, 100 chats on three
servers; `SCEN_STRESS_CHATS`) fails on this 4-core container on main and on this branch: with
three servers, the memory service, agent-runs and PostgreSQL on four cores, `/v1/context` and
`/v1/verify` pass their deadlines and agent-runs answers late, so a turn runs without its
history (said as `memory_unavailable`). The bugs it uncovered are fixed (agent-runs' key cache,
hold heartbeats, turn order, memory's hot thread cache, the answer check's own circuit, the
write queue); run it here on a sized machine.

Known items to settle here: the memory pipeline's model call inside a database transaction
(conflict adjudication in LLM mode); a rare "bundle not found" when the cache write times out;
verify capacity (about 2 s per real verify on 4 cores); the spending-segment write cost per
model reply.

### Phase 1 — security: user isolation

* Kimi's (and Claude's) own shell and file tools can read other users' files on a shared
  host: isolate per user (the user's home is already per user; the file store and the host
  shell are not).
* Review every path where one user's data could reach another's run.

### Phase 2 — conversational tools (Claude-like)

Turn hard-coded rules in the native tools into guidance the model follows, and keep code for
safety only:

1. `schedule_task`: no fixed "9:00"; a condition becomes a recurring check; ask one question
   (with options) only when frequency or time matters, else choose and say the schedule, zone
   and next run; tell the model the "reported once until it clears" rule.
2. Timezone: never silently UTC — ask, or say so.
3. A missing time: tell the model to ask, not "write 5 cron fields".
4. Confirmation buttons in plain words ("every Monday at 9:00"), never raw cron.
5. Duplicate schedules judged by the model (it has `list_schedules`); code keeps only the
   exact-duplicate check.
6. Questions the code asks (the 90-day "still need this?") in the model's words, translatable.
7. One question format every app can draw: options, a recommended one, free text, a plain-text
   fallback for surfaces without buttons.

### Phase 3 — schedule settings (the PM's S1–S10)

One settings object with defaults (to confirm with the PM): shortest repeat 1 h (15 min by an
admin), 5 schedules per user, 50 per organisation, 500 runs a day, 10 minutes and 30 tool calls
per run, pause after 3 failures, pause if unread 14 days, expire after 90 days (renewable),
writes off unless allowed, raw cron for admins only, 9:00 when no time is said (and say so).
Plus: organisation and per-day limits, expiry and renewal, "nobody reads this" tracking, the
owner-left pause, a scheduled run waiting on an approval with nobody there.

### Phase 4 — skills: format, visibility, creation

* Format: the open Agent Skills `SKILL.md` (name, description, steps), enterprise fields under
  `metadata` (`level`, `owner`, `status`, `version`, `inputs`, `tools`, `writes`, `based_on`).
* Visibility: skills labelled by level (company, team, mine) for pickers and questions; show
  which skill answered.
* Creation: "save this as a skill" makes a draft, checks for duplicates (fingerprint, then
  the model judges), offers "improve the existing one", and submits it to the company's
  approval flow; a draft is not active until approved. The harness defines a small
  `save_draft` / `submit` / `status` interface; ai-registry is the optional reference store with
  approvals; a company may plug its own.
* Checks: an approver sees the tools and writes a skill needs; a run says plainly when a skill
  needs a tool this user is not allowed.

### Phase 5 — rules (memory) files from the store

Company, team and personal rules files by level (company "must" rules not overridable), on
every adapter, pinned per run like skills; "remember this rule" becomes a draft for approval.

### Phase 6 — channels

A channel interface and a native "send" tool: email, Slack, Teams. Connected by an admin or by
the user signing in, outside the chat (never a key in chat); the model offers only connected
channels; preview on a schedule's first send; rate limits; retries; delivery of schedule
results.

### Phase 7 — enterprise operations

System schedules (admin-managed, versioned, outside user quotas), cost and usage per schedule,
an admin view and an emergency stop, "wait, then continue" in skills, an admin screen for the
tool catalog (today: API and SDK only).

### Outside clients over MCP

Tools exposed to an outside client (Claude Desktop) bypass the harness: govern them at the tool
server (mcp-sdk with ai-registry: scopes, audiences, argument checks, server-side approval for
business writes) and the gateway (which tools per key, audit); prefer exposing the governed
agent (the harness's MCP/A2A endpoint) over raw tools.

## Open items carried over

* The tool catalog is per tenant: a tenant's `read` on a shell or network tool still lets it
  run (platform ownership is not enforced at the catalog).
* Examples share tool names in one tenant's catalog and one user (`ada`); `examples-env.sh
  fresh` gives each full run a clean tenant.
* The overhead baseline (`benchmark-results.json`) holds a reference workload's time beside
  the cases': each run scales its numbers by that workload timed on its own machine, so the
  baseline holds on any machine (re-record it only when the harness's overhead changes on
  purpose).

## How to run things

* Harness gates: `make lint`, `.venv/bin/pyright`, `scripts/check_docs.py`, `make test` (100%
  line and branch coverage, about 55 minutes), `make matrix`, the feature suite, the scenario
  suite (`scripts/scenarios/stack.sh reset && scripts/scenarios/stack.sh test`).
* Run one heavy suite at a time on a small machine: parallel suites exhaust memory and the
  shared pytest temp directory.
* Worktrees share the main checkout's `.venv`: run them with `PYTHONPATH=<worktree>/src` and
  never `uv sync` inside a worktree whose `.venv` is a link (it repoints the shared install).
* agent-runs keeps a checkpoint as segments only when its keys are exactly
  `{trellis_segments}`; the in-process `LocalRuns` hides this, so test checkpoint shapes against
  agent-runs.
* The memory service vendors agent-contracts: run `make vendor` there after a contracts change.
* Live examples need: a private memory service and agent-runs onboarded with tenant `default`;
  the examples' own tools served as MCP servers and declared on the gateway; a virtual key
  scoped to the model and those servers; the judge alias; the `triage` stored prompt; the
  tenant tool catalog the Code Mode scenario describes; `EXAMPLES_MODEL` set. The round's setup
  script did this outside the repository; adding it under `scripts/` is a follow-up.
