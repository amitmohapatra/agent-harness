# Architecture decisions (architect, 2026-11-14)

Source: `/var/tmp/cons/review/architecture.md` (ARCH-1..25, sections A–D, 10 open questions),
`/var/tmp/cons/review/security.md`. These are binding for every coding job. IDs refer to
those reviews.

## Answers to the review's open questions

| # | Question | Decision |
|---|---|---|
| 1 | Harness-side retry of a graph's **own** tools? | **No harness retry loop around a library's own tools.** Delete the `ToolNode._arun_one` retry, `Deferring`, `_Calling.owed/entries/tried`, `_again`, `_node_retries` and the two private LangGraph imports (ARCH-6). Where the harness builds the stack (ReAct, `middleware()`), retries of the graph's own *read* tools come from LangChain's own `ToolRetryMiddleware`, configured from the harness's read/write knowledge; a user's own retry middleware or `RetryPolicy` wins. A plain graph the user built gets exactly the retries the user configured. The pre-existing governance wrap of `ToolNode` stays (it governs, it does not change behaviour) but must fail loudly if LangGraph's internals move. |
| 2 | Is a hold a lease? | Not unified now (schema rewrite, no user value). Do the small step: one predicate for "kept in its caller's process", one function for expiry under a cancel request, used by both. Revisit after release. |
| 3 | Should agent-runs end a run's children itself? | No (contract change). One client-side `end_children` in the harness, called for **every** final status (ARCH-1, ARCH-3). |
| 4 | `start_async_task` write when remote, read when local? | Same in both: one table of the five tools, one side effect each, as documented for the remote case (start/update/cancel are announced writes, not asked unless the catalog says so; check/list are reads). ARCH-14. |
| 5 | Tool-use id in a Claude MCP handler? | Claude job checks after the SDK upgrade. If available, key attribution by id. If not, queue per (name, args) in arrival order (no "first match"). ARCH-16. |
| 6 | Task list in the file store vs derived from the run store? | Keep the carried-state store: Deep Agents' files and todos must now be carried too, and they do not belong in run metadata. Move it out of `pipeline.py` into one repository class beside the file store; the adapter says what it carries through the `Adapter` protocol, not `carries(target)` (ARCH-8). |
| 7 | May `created_at` be rewritten? | **No.** `created_at` is immutable. The line's place uses a field that already means "started/resumed running" (`running_since`) if the record exposes it; if nothing existing can carry it, report the exact minimal addition before making it (ARCH-15). The place is computed once, on the server side of the API; the harness does not recompute it. |
| 8 | What is `bridges/`? | "Code that connects a library's mechanism to a harness mechanism; may serve one library." Rules that hold: adapters import bridges; bridges and other shared code (`pipeline`, `background`, `agent`, `conversation`, `tools/bridge`, `journal`, `line`) never import an adapter; an adapter never imports another adapter. Names shared by adapters move to `bridges/` or `adapters/base.py`. Cross-module use of `_private` members is replaced by public methods (ARCH-9). |
| 9 | Which wins at the fold: react-background's watchers or chat-order's `Holds(lost=)`? | **`Holds(lost=)`** is the only detector of "a run I keep was ended elsewhere". `Tasks._watch` is deleted; `AsyncTasks._watch` keeps only the remote read (ARCH-4). |
| 10 | End a never-held `RUNNING` keyed run after a grace? | Yes: a non-queued run started with a `concurrency_key` is born with the same grace `resume` gives, in agent-runs and in `LocalRuns` (ARCH-5). |

## One mechanism per job (target state)

| Job | The one owner |
|---|---|
| Keep a run alive / learn it was ended elsewhere | `holding.Holds` (with `lost=`) |
| Stop an attempt | `Agent.stop_attempt(run_id, reason)` (one guard, one reason rule) |
| End a run's children | `end_children(...)`, called from `_concluded` for every final status and from cancel |
| Open/close an attempt's helpers | `async with helping(run, name) as helpers:` in `bridges/helpers.py` (replaces six adapter copies; passes "paused" explicitly) |
| End a helper's run (events, memory, outcome) | `Helpers.finished` (usable without a `Runtime`) |
| Ask a paused child's question through a turn | one function in `subagents.py` |
| A turn the chat started itself (schedule fire, background notification) | one `Runtime.own_turn` with `stale()` / `quiet()` |
| The five background-task tools | one table (names, side effects, result shape) |
| Claude session cost | one `SessionCost` object, persisted by the journal |
| Store behaviour | one contract test suite run against `LocalRuns`, `FileRuns` and the agent-runs client |
| Adapter capabilities (`aclose`, `child_cancelled`, `tasks`, what it carries) | declared on the `Adapter` protocol; no `getattr` probes |
| Run/helper/task/session states | explicit transition function per class that rejects invalid moves |

## Sequencing

1. **Now, on the branch that owns the code** (small, local): ARCH-1, 2, 5, 6, 7, 13, 14, 15, 16,
   19 (first bullet), 20; SEC-R-1..11; the tool findings each job's own checks report.
2. **Fold** all branches into one integration branch per repo (chat-order's `Holds(lost=)`
   wins over react-background's watchers; bg-claude-da wins on Claude `Workflow`).
3. **One cleanup job on the integrated branch** (cross-cutting, would conflict if done
   earlier): ARCH-3, 4, 8, 9, 10 (small step), 11, 12, 17, 18, 21–25; DUP-1..10; the
   duplicated-knowledge table; test clones T1–T11; boolean-flag parameters; the `helping()`
   context manager; `Adapter` protocol capabilities. Pure refactor: test ids, matrix cells and
   the ReAct golden requests must be unchanged; no public argument or signature changes.
4. Full verification, then the Haiku phase.

## Performance decisions (`/var/tmp/cons/review/performance.md`, PERF-1..13)

Scale target used to judge: 1,000 concurrent runs per server, several servers, 200-step runs,
conversations of hundreds of turns.

| Finding | Decision | Owner / when |
|---|---|---|
| PERF-1 (High) a waiting run holds one of the client's 100 connections | Event streams get their own HTTP client, never the control client's pool; one follower per run-ahead per process shared by its waiters; a cap, with the timed re-read beyond it | chat-order, now |
| PERF-2 (High) one heartbeat request and one index-rewriting row write per held run (50 req/s per server at 1,000 runs = the whole default tenant budget; a missed beat now ends a live run) | **Liveness is a property of the process, not the run.** Target: one heartbeat per process per interval; runs reference their holder; the sweep ends runs whose holder lapsed. First check whether an existing endpoint can carry it; if not this is a *necessary* additive contract change and is reported with its exact diff before it is made. Until then: a heartbeat refused by the rate limit is retried inside the hold window and is never counted as a miss | post-fold performance job |
| PERF-3 (High) the cap of 16 live Claude processes is checked before an await (100 first turns at once keep 100 processes, 69 MB private each) | Reserve the slot before the await; cap is per server process, not per agent; evict least recently used | Claude job, now |
| PERF-4 one store read per background task per model call and per 10 s | Tasks at work in this process are not read from the store; the rest in one request; remote watch backs off with jitter; `list` reads unfinished tasks only. Target: at most 1 request per model call | react-background, now |
| PERF-5 carried task list never shrinks; stops being kept at 705 tasks | Ended tasks whose outcome was told are dropped, keeping the last K (named constant, tested at the bound) | react-background, now |
| PERF-6 turn start does one `get` per earlier run | `_served_elsewhere` stops at the first run older than its own (list is newest first); `earlier` filters by status on the server if an existing filter allows and pages by 50; `Turns.ended` bounded by bytes. No `RunSummary` field additions unless that proves insufficient (then report first) | scn-fixes (`_served_elsewhere`), chat-order (`earlier`) |
| PERF-7 the run log sends one request per event | Linger 100–250 ms or 100 events; flush at once on interrupt, finish and close | post-fold performance job |
| PERF-8 every framework-run call's whole output is journaled a second time | Record the output only where a replay needs it (an open foreground sub-agent); otherwise a reference or preview | scn-fixes, now (same code as SEC-R-5) |
| PERF-9, PERF-10 | Fixed on chat-order `07e8197`; keep the EXPLAIN-based regression tests | done |
| PERF-11 the hold sweep ends 100 runs per 5 s tick | Repeat within a tick until a short batch, under a time budget | post-fold performance job |
| PERF-12 an unattributed sub-agent call waits 5 s, twice, every call | Remember the miss after the first timeout | Claude job, now |
| PERF-13 smaller polls; `_allowed` fetches every run of the thread; bind-parameter status in sweeps | With ARCH-4 and the post-fold job | post-fold |

Regression tests for performance assert **counts** (requests, calls, open streams, bytes,
EXPLAIN plan nodes), never elapsed time.

## Do not change (found well designed)

`holding.Holds`; `line.Line`; agent-runs `_move` and the advisory-locked `_placed`; stable
child-run ids; `Journal.handed_on`; `background.delivered` and once-per-outcome `notify`;
`outbound.validate_url`; `handed_to`'s loud failure; `Harness.aclose`; refusing unattended
calls in a live Claude session; the explanatory comments.
