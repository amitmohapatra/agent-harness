# Performance review: this round of the Trellis stack

Read-only review. No product code changed, no commits, no model calls, no test suites run.
Measurement scripts are in `/var/tmp/cons/review/perf-scratch`. EXPLAINs ran on a database of my
own on 127.0.0.1:55957 (`perfreview_runs`, created, migrated, filled with 200,620 runs and 27,824
resolutions, then dropped).

## Refs reviewed

Two of the branches were pushed to while I worked, so every finding names a commit.

| Repo | Branch | Commit reviewed |
|---|---|---|
| agent-harness | `-final` | `d2bf754` |
| agent-harness | `-react-background` | `983923b` |
| agent-harness | `-chat-order` | `1968640` (a WIP commit pushed mid-review; `aac6a95` before it) |
| agent-runs | `-final` | `422ef68` |
| agent-runs | `-chat-order` | `087259c` (WIP), then `07e8197` (pushed later; it fixes two findings below) |

Symbols: N tool calls in a run, T turns in a conversation, R concurrent runs per server,
B background tasks, S servers, W messages waiting in one conversation's line.

## Findings, by impact

### PERF-1 (High): a waiting run holds one connection of a 100-connection pool; at 100 waiters the process cannot reach agent-runs

- **Where:** agent-harness `1968640`, `src/trellis/harness/line.py:157-182` (`Line._left`, the
  `stream_events` at line 171), called from `Line.turn` (line 132). The client is the process's
  one `RunsClient` (`harness.py:172`), built with httpx defaults: 100 connections, HTTP/1.1,
  10 s pool timeout.
- **Complexity:** one open SSE connection per run waiting behind a run of another process (or a
  worker's), for as long as it waits. Connections are O(waiters), from a pool of 100 shared with
  every heartbeat, event append, start and finish. On the server each stream polls the database
  every 0.5 s with two queries (`agent-runs routers/runs.py:506-526`): 4 queries/s per waiter.
- **Scale at which it hurts:** 100 runs on one process waiting behind turns held elsewhere. That
  is 10% of 1,000 concurrent runs. Past it, control requests only get a connection when a stream
  times out (every 30 s per stream, about 3 a second for 100 streams), while 1,000 held runs need
  50 heartbeats a second. Holds then lapse and agent-runs ends live runs `holder_lost`.
- **Evidence:** `mb_pool.py`, the real `RunsClient` against a local fake:

  | Runs waiting behind a run elsewhere | A hold heartbeat from the same client |
  |---|---|
  | 0 | answered in 21.9 ms |
  | 50 | answered in 1.7 ms |
  | 99 | answered in 2.0 ms |
  | 100 | not sent within 3 s (no free connection) |
  | 150 | not sent within 3 s (no free connection) |

- **Before the WIP commit (`aac6a95`, `line.py:93-106`):** every waiter re-listed the line every
  0.5 s when any run ahead was elsewhere. For W=100 that is 2 list pages per read (`mb_requests.py
  line`), so about 400 requests/s, eight times agent-runs' default tenant budget of 50 requests/s.
  One shared event also woke all W waiters on each departure. `1968640` replaced the polling with
  the stream; the request rate is fixed, the connection count is the new problem.
- **Refutation tried:** a run waits only on the run just ahead of it, so one 100-message burst
  spread over S servers opens about W(S-1)/S² streams per server (25 for S=2), not 100. The limit
  is reached by many conversations at once, or wherever the run ahead is a worker's.
- **Fix:** give streams their own HTTP client, never the control client's pool. Share one follower
  per run-ahead per process among all its waiters, and cap followers (say 32), falling back to the
  30 s timed re-read beyond the cap. Target: connections O(min(distinct runs ahead, cap)), control
  requests unaffected by W.
- **Regression test:** with a transport that counts open streams, put 200 runs in line behind runs
  elsewhere; assert open streams ≤ cap and that a `heartbeat` call is sent (a call count on the
  fake, no timing).

### PERF-2 (High): hold heartbeats are one request and one non-HOT row write per run, unbatched

- **Where:** agent-harness `d2bf754`, `src/trellis/harness/holding.py:134-147` (`Holds._beating`:
  one task per held run, a heartbeat every `HOLD_SECONDS / 3` = 20 s). agent-runs `422ef68`,
  `src/agent_runs/store/runs.py:713-755` (`heartbeat`, `_hold`) and `api/ratelimit.py:65`.
- **Complexity:** requests = (held runs)/20 per second, where held runs are every in-process run,
  waiting turn, child run, helper and background task. Not batched: O(R) requests and O(R) row
  writes per server per interval, where O(1) per server would do.
- **Answer to the question asked:** at R=1,000 per server the rate is **50 heartbeat requests/s
  per server**, each its own asyncio task, HTTP request and two PostgreSQL transactions (the
  rate-limit upsert, then `SELECT … FOR UPDATE` + `UPDATE`). With S=10 that is 500 requests/s and
  1,000 transactions/s for holds alone.
- **Scale at which it hurts:**
  - 50 requests/s is exactly agent-runs' default tenant budget (`settings.py:198`, 3,000 a
    minute). Heartbeats for 1,000 runs of one tenant spend all of it. Every request of a tenant
    also serialises on that tenant's one bucket row.
  - A refused or delayed heartbeat now costs a live run. Three missed beats (60 s) and the ticker
    ends the run `ERROR`. Before this round, throttling only dropped events.
  - `held_until` is the column of `ix_runs_held`, so the update is never HOT: a new heap tuple and
    a new entry in each of the row's 10 or so indexes on every beat.
- **Evidence:** `heartbeat_wal4.sql`, one committed transaction per update, 24,000 updates on the
  200k-run table:

  | Update | WAL bytes/update | HOT updates | Time for 24,000 |
  |---|---|---|---|
  | An unindexed column | 125 | 23,664 of 24,000 | 294 ms |
  | `holder`, `held_until` (the hold) | 702 | 0 of 24,000 | 881 ms |

  That is 5.6 times the WAL and 3 times the CPU of a HOT update; at 500 beats/s about 30 GB of
  WAL a day. The row lock itself is cheap: a primary-key lookup, 0.05-0.08 ms, 6 buffers.
- **Refutation tried:** I checked whether dead `ix_runs_held` entries make the sweep slow between
  vacuums. They do not: after 107,100 unvacuumed heartbeats the first sweep read 6,077 buffers
  (8 ms), the next 109 (0.17 ms), because the index hints dead entries.
- **Fix:** hold by process, not by run. One row per holder (`holder`, `held_until`), runs
  reference it, the sweep ends runs whose holder lapsed. One request and one HOT row update per
  process per interval. A smaller step is a batch endpoint, one request per process (still O(R)
  rows). Target: S/20 requests/s instead of R·S/20.
- **Regression test:** hold 1,000 runs on a fake store for two beat intervals under a fake clock;
  assert heartbeat requests ≤ 2, not 2,000.

### PERF-3 (High): the cap on live Claude processes is not enforced when turns start together

- **Where:** agent-harness `d2bf754`, `src/trellis/harness/bridges/claude_live.py:1189-1206`
  (`LiveSessions.open`): the check `len(self._open) >= KEPT` (line 1198) comes before
  `await session.start(...)` (line 1204), and the session is registered only after it (line 1205).
- **Complexity:** intended bound 16 processes (`KEPT`, line 96) per agent per server process. The
  actual bound is the number of `open` calls in flight together: unbounded.
- **Answers to the questions asked:**
  - **Upper bound per server:** nominally 16 × (agents with a Claude target), since `LiveSessions`
    is per adapter and each agent has its own adapter. Under concurrent first turns: none.
  - **Idle eviction:** 300 s after the last activity when nothing works in the session, 6 h after
    its last turn when something does (lines 92-94, `_rest` at 1109). When 16 are open, a newcomer
    closes the first idle one in dict order (oldest opened, not least recently used); if none is
    idle the turn runs without a live session.
  - **Memory per process:** one idle bundled CLI in streaming-input mode, no prompt sent, no model
    call: RSS 197 MB, of which 69 MB is private anonymous and 126 MB is the binary's file pages
    (shared between processes). That is the floor before any session is loaded.
- **Scale at which it hurts:** before this round a turn's process ended with the turn; now each is
  kept at least 300 s. 100 conversations starting within the CLI's connect time keep 100 processes
  for five minutes: at least 6.9 GB private. 1,000 keep at least 69 GB.
- **Evidence:** `mb_live.py`, `LiveSession.start` stubbed to an await (no process started):

  | First turns | Processes started | Kept open |
  |---|---|---|
  | 100, one after another | 100 | 16 |
  | 100, at once | 100 | 100 |
  | 1,000, at once | 1,000 | 1,000 |

- **Fix:** reserve the slot before awaiting the start (count sessions being opened, or a
  semaphore of `KEPT` around check and start). Make the cap process-wide, not per agent. Evict by
  least recently rested. Target: live processes ≤ `KEPT` always.
- **Regression test:** `asyncio.gather` 100 `open()` calls with `start` stubbed; assert
  `len(live) <= KEPT` and that `start` was called at most `KEPT` times.

### PERF-4 (Medium): background tasks are read one request per task, per model call and per 10 s

- **Where:** agent-harness `983923b`.
  - `src/trellis/harness/background.py:989-1012` (`pulled`), called from
    `bridges/background_tasks.py:127, 145, 152` (turn start, before every model call, after a
    final answer).
  - `background.py:362-370` (`Tasks._watch`) and `bridges/async_tasks.py:173-192`
    (`AsyncTasks._watch`), both every `WATCH_SECONDS` = 10 s.
  - `background.py:951-971` (`Offered._list`) and `async_tasks.py:110-137` (`said`).
- **Complexity:**
  - Pull: U `runs.get` per model call, U = unfinished tasks, each a full run record. A task
    waiting for a person counts as unfinished and is read on every model call until answered.
  - A task at work in this process: one `get` per 10 s on top of its heartbeat. The heartbeat's
    refusal already says the run ended.
  - A held Deep Agents remote task: one `get` to agent-runs and one check on the task's server
    per 10 s, plus the heartbeat: 0.25 requests/s per task, with no back-off as the task ages
    (8,640 reads a day on each side). `KEPT` is 1,024 per agent, so 256 requests/s at the cap.
  - `list_async_tasks`: one `get` per task the conversation ever started, all at once. For Deep
    Agents' own tasks, `said` then makes up to two sequential `get`s per finished task.
- **Scale at which it hurts:** 1,000 runs at one model call every 5 s with U=3 is 600 `get`s/s.
  A 200-step run with 5 unfinished tasks makes 1,000.
- **Evidence:** `mb_requests.py pulled`:

  | Tasks carried | Pull, per model call | `list_async_tasks` |
  |---|---|---|
  | 0 unfinished, 10 finished | 0 | 10 |
  | 5 unfinished, 20 finished | 5 | 25 |
  | 20 unfinished, 200 finished | 20 | 220 |

  220 concurrent `get`s exceed the client's 100 connections.
- **Fix:** skip the store for tasks at work in this process (their state is known here), read the
  rest in one request, and pull at most every few seconds. Replace `Tasks._watch` with the hold
  heartbeat's refusal (chat-order already adds `Holds(lost=…)`). Back the remote watch off from
  10 s to 60-120 s with jitter. List only unfinished tasks. Target: at most 1 request per model
  call whatever U is.
- **Regression test:** with a counting store and 20 unfinished tasks, assert ≤ 1 store request
  per `abefore_model`; assert `list_async_tasks` requests equal the unfinished count.

### PERF-5 (Medium): the carried task list never shrinks and stops being kept at 705 tasks

- **Where:** agent-harness `983923b`, `src/trellis/harness/tools/results.py:385` (`merged` keeps
  the entries of both sides); `src/trellis/harness/pipeline.py:545` (`CARRIED_BYTES` = 256 KiB)
  and `_keep_carried` (`d2bf754` line 609).
- **Complexity:** the list grows by one record per task started and nothing removes one: O(all
  tasks ever). The files themselves are fine: each write merges what it read into one file and
  removes the rest, so a turn costs 2 listings, 2 reads, 1 put and 1 remove, independent of T.
- **Scale at which it hurts:** one task record is 370 bytes of JSON; the 705th task makes the
  file pass the cap. From then on the write is refused with a log warning every turn and other
  servers no longer see new tasks. Every task ever started is also in the graph state and in each
  `list_async_tasks` (PERF-4). The `.carried/...` file of a conversation is never deleted.
- **Evidence:** `mb_carried.py`: "the 705th task is the first not kept"; at 704 tasks the
  per-turn merge, dump, load and size accounting total under 4 ms.
- **Fix:** drop ended tasks once their outcome was told, keeping the last K (say 50) or a few
  days. Target: O(unfinished + K).
- **Regression test:** after 1,000 start-and-finish cycles in one conversation, assert the
  carried list holds ≤ unfinished + K entries and the kept file is under `CARRIED_BYTES`.

### PERF-6 (Medium): a turn's start reads the conversation with one `get` per run

- **Where:**
  - agent-harness `d2bf754`, `src/trellis/harness/pipeline.py:467-489` (`_served_elsewhere`, the
    `get` at line 481), run at each first attempt of an adapter that keeps sessions.
  - `src/trellis/harness/conversation.py:193-221` (`earlier`, the `get` at 219).
  - agent-harness `1968640`, `conversation.py:195-223` (`earlier` without `max_pages`) and line
    357 (`Turns.ended`).
- **Complexity:** `RunSummary` has no `user_id`, `parent_run_id` or `created_at`, so each listed
  run is fetched whole. `_served_elsewhere` is O(min(T, 50)) sequential `get`s. `earlier` at
  `d2bf754` is 1 list + up to 10 `get`s on every turn, uncached. At `1968640` the `get`s are
  cached, but the listing pages 10 at a time past every newer waiting message: O(W/10) lists per
  turn start, O(W²/20) over a burst.
- **Scale at which it hurts:** a conversation of 50 turns or more pays 50 requests at each turn
  start (about 250 ms at 5 ms each, and one second of the default tenant budget).
- **Evidence:** `mb_requests.py served` and `earlier`:

  | Case | Requests |
  |---|---|
  | `_served_elsewhere`, 10 earlier turns (`d2bf754`) | 11 |
  | `_served_elsewhere`, 50 or 500 earlier turns | 50 |
  | `earlier`, 10+ answered turns (`d2bf754`) | 10, every turn |
  | `earlier`, 100 messages waiting behind (`1968640`) | 22 first, 12 lists each later turn |
  | `earlier`, 1,000 messages waiting behind (`1968640`) | 112 first, 102 lists each later turn |

- **Also:** `Turns.ended` caches up to 1,024 whole `RunRecord`s per agent, bounded by count, not
  bytes (`Conversations` next to it is bounded by characters).
- **Functional note, not performance:** at `d2bf754` a turn with 10 or more newer messages
  waiting reads no history at all (`max_pages=1`; "0 messages read" in the run above).
  `1968640` fixes that by paging.
- **Fix:** add `user_id`, `parent_run_id` and `created_at` to `RunSummary` (the list query
  already reads the rows). `_served_elsewhere` then needs one list and can stop at the first run
  older than `mine`. For `earlier`, filter by status on the server and page by 50. Bound
  `Turns.ended` by bytes or keep only the fields `earlier` uses. Target: O(1) requests per turn
  start, O(newly ended turns) `get`s.
- **Regression test:** counting store: a 500-turn conversation gives `_served_elsewhere` ≤ 2
  requests; `earlier` with 1,000 waiting messages makes ≤ 3 list requests.

### PERF-7 (Medium, older code on a listed hot path): the run log sends one request per event

- **Where:** agent-harness `1968640` (and `d2bf754`, `origin/main`),
  `src/trellis/harness/runlog.py:86-120` (`RunLog.__call__`, `_send`). This round only added
  `ended`.
- **Complexity:** an append starts the moment an event arrives and the next starts when it
  returns; there is no linger or minimum batch. Requests per run per second = min(event rate,
  1/append latency). Batches form only once agent-runs is slower than the event stream.
- **Scale at which it hurts:** one run streaming 50 deltas a second makes 50 requests/s, a
  tenant's whole default budget. 1,000 streaming runs keep the client's 100 connections
  permanently busy, in front of the heartbeats of PERF-2. Each append also takes the run's row
  lock (`store/runs.py:1043`).
- **Evidence:** `mb_runlog.py`:

  | Events/s | Append latency | Requests/s | Events per request |
  |---|---|---|---|
  | 50 | 5 ms | 50.0 | 1.0 |
  | 200 | 5 ms | 200.0 | 1.0 |
  | 200 | 50 ms | 21.3 | 9.4 |

- **Fix:** linger 100-250 ms or until 100 events, flushing at once on `INTERRUPT`,
  `RUN_FINISHED` and close. Target: at most 4-10 requests/s per run at any event rate.
- **Regression test:** emit 600 events through a `RunLog` on a zero-latency fake store under a
  fake clock; assert appends ≤ duration/linger + 1.

### PERF-8 (Medium): every framework-run call's whole output is now journaled, beside the transcript

- **Where:** agent-harness `d2bf754`, `src/trellis/harness/tools/bridge.py:333-335` (`observed`:
  `record_call(key, _journaled(outcome), scope=key)`).
- **Complexity:** journal bytes grow by the full output of every successful built-in call
  (Claude Code's `Read`, `Bash`…, OpenAI hosted tools, ACP calls). The Claude transcript mirrored
  into the same journal already holds it. These records are never dropped.
- **Scale at which it hurts:** `mb_journal.py`, a Claude run of 200 built-in calls:

  | Output per call | Journal before | Journal now |
  |---|---|---|
  | 4 KB | 1.34 MB | 2.17 MB (+62%) |
  | 20 KB | 4.62 MB | 8.73 MB (+89%) |

  At 1,000 concurrent runs that is 0.8 to 4 GB more memory, and the same again in every pause
  checkpoint.
- **Refutation tried:** I expected `Journal.save` (a full dump and compare per save) to be the
  cost. It is not: dump plus delta take under 2.5 ms at 8.7 MB, because strings are shared, and
  only a worker's run saves.
- **Fix:** record the output only while a foreground subagent is open (the one case that replays
  it), or keep a reference to the transcript entry or a preview. Target: O(1) journal bytes per
  observed call outside a subagent.
- **Regression test:** after 200 observed 10 KB calls with no helper open, assert the dumped
  journal's `calls` is under a small bound (bytes, not time).

### PERF-9 (High at `087259c`, fixed at `07e8197`): a claim scanned `run_resolutions` once per waiting pair

- **Where:** agent-runs `087259c`, `src/agent_runs/store/runs.py:307-316` (`_place`: the subquery
  correlated on `run_id` only, while `ix_run_resolutions_run` leads with `tenant_id`), used in
  `_next_queued` (line 695) and `_key_has_room` (line 743).
- **Complexity:** O(queued runs blocked behind a running run × running runs of that key ×
  |run_resolutions|) per claim.
- **Evidence:** EXPLAIN (ANALYZE, BUFFERS), 2,020 queued, 5,600 running, 27,824 resolutions:

  | Statement | `422ef68` | `087259c` | `07e8197` |
  |---|---|---|---|
  | Claim, every tenant | 15.0 ms | 7,478 ms, 1,589,083 buffers | 22.9 ms |
  | Claim, one tenant | 3.2 ms | 2,180 ms | 11.2 ms |
  | `_key_has_room`, 100 running in the key | n/a | 118 ms | 0.35 ms |

  At `087259c` the plan is `Seq Scan on run_resolutions`, 2,020 loops; the statement timeout is
  15 s. At `07e8197` it is `Index Only Scan Backward using ix_run_resolutions_run`.
- **Status:** fixed at `07e8197` (line 312 adds `ResolutionRow.tenant_id == table.tenant_id`).
- **Still true at `07e8197`, older design:** a claim evaluates the key subquery for every queued
  run before ordering, so it is O(queued) per claim: 11-23 ms at 2,000 queued.
- **Regression test:** on a seeded table, EXPLAIN (FORMAT JSON) the claim statement and assert no
  `Seq Scan` node on `run_resolutions` or `agent_runs`.

### PERF-10 (Medium at `087259c`, fixed at `07e8197`): placing a run read all of the tenant's live runs

- **Where:** agent-runs `087259c`, `src/agent_runs/store/runs.py:769-787` (`_placed`: `status IN
  ('RUNNING','QUEUED')`, which the partial index `ix_runs_concurrency`, `status = 'RUNNING'`,
  cannot serve).
- **Complexity:** O(the tenant's running and queued runs) per start and per resume of a keyed
  run, under the key's advisory lock. It should be O(log n).
- **Evidence:**

  | Commit | Plan | Cost |
  |---|---|---|
  | `5298410` (`status = 'RUNNING'`) | Index Scan `ix_runs_concurrency` | 2-6 buffers, 0.03-0.08 ms |
  | `087259c` | Bitmap scan `ix_runs_inbox`, 3,870 rows filtered | 282-333 buffers, 0.7-2.9 ms |
  | `07e8197` (index widened, migration `c9e4f5a6b7d8`) | Index Scan `ix_runs_concurrency` | 0.01-0.14 ms |

  The `087259c` cost is linear in the tenant's live runs: about 13 times that at 50,000.
- **Status:** fixed at `07e8197`.
- **New at `07e8197`, Low:** `_placed` (line 772) now updates the row just inserted (`created_at`,
  `updated_at`, `running_since`, `queued_at`, all indexed), so every keyed start writes the row
  and its indexes twice.
- **Regression test:** EXPLAIN the placing statement on a tenant with 5,000 live runs; assert an
  index scan on `ix_runs_concurrency` and no bitmap or sequential scan.

### PERF-11 (Low): the hold sweep ends 100 runs per 5 s tick

- **Where:** agent-runs `422ef68`, `src/agent_runs/ticker.py:135, 174-184` (one `end_abandoned`
  batch of `SWEEP_BATCH` = 100 per `TICK_SECONDS` = 5 s; `config/constants.py:124-126`).
- **Complexity:** drain rate is 20 runs/s whatever the backlog.
- **Scale:** a server that dies with 1,000 held runs is fully ended 50 s after the holds lapse,
  about 115 s after the death. On chat-order each of those runs blocks its conversation's line
  until then.
- **Evidence:** the query is served by the migration's index: `Index Scan using ix_runs_held`,
  100 rows locked in 0.2-1.1 ms, 302 buffers.
- **Fix:** repeat the sweep within a tick until a short batch, under a time budget.
- **Regression test:** 1,000 lapsed holds, one tick; assert all 1,000 ended (a count).

### PERF-12 (Low, conditional): an unattributed subagent call waits 5 s, every time

- **Where:** agent-harness `d2bf754`, `src/trellis/harness/bridges/helpers.py:295-307`
  (`Helpers.of`, `ATTRIBUTED_SECONDS` = 5 at line 83), called in both the `PreToolUse` and
  `PostToolUse` hooks.
- **Complexity:** when the framework never says whose a worker is, each call waits the full 5 s
  twice; the miss is not remembered. O(calls) waits where one would do.
- **Scale:** `mb_helpers.py`: 50 tool calls of such a subagent make 100 waits, 500 s of added
  latency. An attributed worker makes none. I did not see a CLI that omits `task_started`, so
  this needs that condition.
- **Fix:** after the first timeout, record the worker as the helper named last.
- **Regression test:** count `Event.wait` calls for 50 `of()` calls of one unknown worker;
  assert ≤ 1.

### PERF-13 (Low): smaller polls and one older loop

- agent-harness `d2bf754`, `bridges/claude_live.py:346-362`: a background subagent's run waiting
  for a person reads its record every 10 s for up to 12 h (4,320 `get`s). Fix as PERF-4.
- agent-harness `d2bf754`, `conversation.py:158-176` (`_allowed`, on `origin/main` since
  `5d58e00`, not this round): the first call that asks in each attempt fetches every run of the
  thread. `mb_allowed.py`: 53 requests at 50 turns, 512 at 500.
- agent-runs, every sweep: `status` is a bind parameter. With a forced generic plan the hold
  sweep becomes a `Seq Scan` of all 200k rows (82 ms). Under the default `plan_cache_mode` the
  seventh execution still used `ix_runs_held`, so this bites only if generic plans are forced.
  Writing the status as a literal removes the risk.

## What the migration's indexes serve

| Query | Commit | Plan on 200k runs | Served? |
|---|---|---|---|
| `end_abandoned` sweep, and the resume-grace runs it also ends | `422ef68` | Index Scan `ix_runs_held`, 0.2-1.1 ms per 100 rows | Yes (`b8d3e4f5a6c7`) |
| Heartbeat row lock | `422ef68` | Index Scan `agent_runs_pkey`, 0.05-0.08 ms | Yes |
| `_placed`, latest run of a key | `5298410` / `087259c` / `07e8197` | index / bitmap over the tenant's live runs / index | Yes / No / Yes (`c9e4f5a6b7d8`) |
| Claim with places | `087259c` / `07e8197` | Seq Scan `run_resolutions` × 2,020 / index-only | No / Yes |
| The line's listing (`Line.ahead`) | `1968640` | `ix_runs_thread`, 0.13-0.25 ms | Yes |
| Children still running (`_children_ended`) | `d2bf754` | `ix_runs_parent`, 0.02 ms | Yes |

The advisory lock in `_placed` is per (tenant, key) and transaction-scoped; at `07e8197` it is
taken as the transaction's last step. Claims try-lock the same id and skip, never wait.

## Hot paths judged fine

| Hot path | Complexity | Basis |
|---|---|---|
| Tool call: `bridge.call`, `permitted`, `_repeated`, `_deciding`, `_answered` | O(size of args) for about 6 content keys, else O(1) | `mb_bridge.py`: 0.02 ms a call at 300 B args, 1.9 ms at 100 KB, 25 ms at 1 MB |
| `after_tool` hooks' change check (two `model_dump`s) | O(size of output), only with hooks | 6 ms for a 10,000-row table; a string output costs nothing |
| Journal replay and cursors (`Replay._index`, `occurrence`, `calling`, `paused`, `cursor`) | O(1) per read; O(distinct keys) per pause or subagent start | Read; `passed` only walks the current step's keys |
| `Journal.save` (dump and delta) | O(journal size), a worker's run only | Under 2.5 ms at 8.7 MB |
| Helpers attribution, once attributed | O(1) | 0 waits in 50 calls |
| Model-call middleware (`typed`, `_earlier`, FitWindow, clearing, steering) | O(messages) per call, unchanged in order this round | Read of the diff; `_calling` searches from the end |
| Event emit | O(1), plus O(size of output) for a non-string preview | Read |
| Run start and end: `attempt`, `_turned`, the hold context | O(1); two extra list requests per turn on chat-order | Read; `mb_requests.py line` |
| `_children_ended` | One indexed listing plus O(children), only on timeout or cancel | Plan above |
| Carried state files | Two listings, two reads, one put, one remove per turn; one file after each write | Read of `_keep_carried`; the list inside is PERF-5 |
| Live-session relays (`_of`, `_call`, `convert.handler`, `_shape`) | O(background tasks of the session), O(tools) | Read |
| Hold sweep between vacuums | O(batch) plus hinted dead entries | 109 buffers after 107,100 heartbeats |

## Limits of this review

- The claim, `_placed` and sweep plans come from one synthetic distribution (10 tenants, one with
  half the runs; one 100-message burst; one 500-turn conversation). The cluster runs with fsync
  off, so I report buffers, plans and WAL bytes, not commit latency.
- The CLI memory figure is one idle process with no session, tools or MCP servers loaded.
- PERF-1's stall is shown at the connection pool; I did not run 1,000 real runs to watch holds
  lapse. PERF-2's rate-limit arithmetic is from the default settings, not a load test.
- I did not profile the model-call middleware under a real graph; that row of the table rests on
  reading the diff.
