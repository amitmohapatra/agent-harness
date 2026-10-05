# Runs, workers, schedules

This page is Way 1, wrapped: the harness records every run of an agent it wraps, and gives
you the inbox, schedules and workers. A team that keeps its own framework uses agent-runs
directly, with `trellis.runs.RunsClient` and `trellis.runs.Worker`:
[blocks/runs.md](blocks/runs.md) (Way 2).

Every run has a record (contracts `RunRecord`): in agent-runs when `RUNS_URL` is set
(`h.runs` is a `trellis.runs.RunsClient`), in process otherwise (`h.runs` is a `LocalRuns`:
same behaviour, nothing survives a restart). Both are the harness's `RunStore`
(`trellis/harness/runs.py`): the part of `RunsClient` the harness calls, with the same
signatures.

| Call | Record |
|---|---|
| `agent.run` / `agent.stream` | `RUNNING` (recorded in process), then `PAUSED` or an ending |
| `agent.start` | `QUEUED`; `RunHandle.result()` waits for a pause or an ending |
| `agent.resume` | the next attempt: `RUNNING` for an in-process run, `QUEUED` again for one that came from the queue; `CANCELLED` on cancel |
| `agent.cancel(run_id, reason=)` / `handle.cancel(reason=)` | `CANCELLED`: at once when queued or paused; a run running here stops now; one on a worker elsewhere is stopped by that worker at its next heartbeat |
| `agent.schedule(cron, input, on_behalf_of=, tz=, tenant=)` | a `Schedule`; each fire queues a run acting for `on_behalf_of` |

`run`, `stream` and `start` take `timeout=` (the most working time, in seconds, across every
attempt — pauses and the queue not counted) and `deadline=` (when the run must have ended):
past either the run ends `TIMEOUT`, never retried. They are the record's `timeout_seconds` and
`deadline`, and agent-runs keeps the time worked (`worked_seconds`) across attempts and crashes;
`h.wrap(..., version=)` (or `TRELLIS_AGENT_VERSION`) is its `agent_version`
([reliability.md](reliability.md#run-time-limit-and-deadline)).

`start` returns a `RunHandle`: `run_id`, `await handle.status()` (the `RunRecord`; a run the
store does not have raises `ConfigurationError`), `await handle.cancel(reason=None)` and
`await handle.result(timeout=None)`, which
reads the run every 0.5 s until it pauses or ends and returns a `Result` (`asyncio.timeout`
raises `TimeoutError` past `timeout`). A queued run's input must be JSON (`start` refuses
anything else); a run started in process records its input as JSON where it can and as text
where it cannot, and so does its answer.

## Workers

```python
await h.worker([agent_a, agent_b]).run()                  # until stopped; CPU count runs at a time
python -m trellis.harness.worker app.agents:h             # every agent the Harness `h` wraps
python -m trellis.harness.worker app.agents:h --concurrency 8
```

The claim loop is agent-runs' SDK's, `trellis.runs.Worker`; the harness's worker
(`trellis.harness.worker`) runs it with your wrapped agents: each claimed run is its agent's
next attempt, and the harness's background writes start before the first claim and drain when
the loop ends. The same loop with a handler of your own: [blocks/runs.md](blocks/runs.md#workers).

`h.worker(agents, *, concurrency=None)` needs at least one agent. `concurrency` — runs executed
at once — defaults to `TRELLIS_WORKER_CONCURRENCY`, else the machine's CPU count between 1 and
8. `await worker.run()` claims and executes until stopped: an idle worker asks again after
0.5 s, doubling the pause while the queue stays empty up to 10 s (half of it random, so a fleet
of idle workers does not ask in step), and asks at once again after it got work.
`await worker.run_once()` claims one run and executes it to its end or pause, and returns
`False` when nothing was queued. A claim that fails is logged and counts as no work; a run that
breaks the harness itself is logged and the worker goes on.

**Stopping.** `worker.stop()` — what `await worker.serve()` and so
`python -m trellis.harness.worker` call on `SIGTERM` or `SIGINT` — stops claiming and lets the
runs the worker holds finish, for up to 25 s (`trellis.runs`' `GRACE_SECONDS`). A run still
going then is *released* (cancelled with `trellis.runs.RELEASED`): stopped without writing
anything, and handed back to agent-runs (`POST /v1/runs/{id}/release`), which queues it at once
as its next attempt for another worker — no waiting for the lease to lapse, and no lapse counted
(the journal replays what its last checkpoint holds — its progress, below: every tool
call with side effects it completed). A second signal releases the runs at once. Then the
memory write queue drains (at most 10 s; what is left is spooled or counted lost —
[memory.md](memory.md)) and the process exits `0`. Give the container at least 40 s to stop
(e.g. a termination grace period of 45 s). Cancelling `worker.run()` instead cancels the runs
it holds: they end `CANCELLED`.

A worker runs any target. Build it the same way in every worker process (at import, in the
module the worker loads); a LangGraph graph's own `interrupt()` (or HITL middleware) pause needs a
checkpointer every worker can reach — a harness approval or `ask` resumed by another worker is
answered from the journal even with an `InMemorySaver`
([langgraph.md](frameworks/langgraph.md#approvals-and-pauses)).

A worker claims a queued run under a 60 s lease and heartbeats it every 20 s (a failed
heartbeat is logged and retried at the next beat). It names itself
on the pause, the finish and an artifact upload, so a worker whose lease lapsed cannot write
over a run another worker has since claimed; a heartbeat refused (`409`) stops the run without
writing, and one that answers `cancel_requested` (someone cancelled the run) stops it and ends
it `CANCELLED`. Each lease says the working time the run has left (`remaining_seconds`: its
`timeout` or agent-runs' maximum, the lesser, less what it worked), and the attempt stops then,
`TIMEOUT`. A lapsed lease sends the run back to the queue as its next attempt (after 5 s,
doubling per lapse; the fifth ends it `ERROR`); a run its worker ends `ERROR` with an error that
may pass is queued again too, up to 3 times, after 10 s, 20 s, 40 s. A paused run
carries its journal as the run's checkpoint, so the worker that claims it after a resume — any
worker — repeats no question and no tool call made before the pause.

**Progress checkpoints: a worker that dies repeats no side effect.** While a worker runs it,
the run also saves its journal — the tool calls completed and their outputs, the answers it
was given, and `ReAct`'s model steps — as the run's checkpoint on a heartbeat
(`POST /v1/runs/{id}/heartbeat {worker_id, lease_seconds, checkpoint}`, which only the lease
holder may send): at once after every completed call that does more than read (a `write` or
`irreversible` tool, or one under a catalog rule), and after reads and model steps at most every
20 s (`PROGRESS_SECONDS`). When the worker dies — killed, out of memory, its machine gone — the
lease lapses, agent-runs queues the run again with that checkpoint, and the next attempt
replays the recorded calls instead of running them again: a payment made before the crash is
not made twice. A call that does more than read is also saved as *started* before it runs, so
what the attempt was doing *during* the crash is known: such a call is not run again blind —
the model reads that it was interrupted and may or may not have taken effect, and checks; a
tool that is idempotent runs again with the same idempotency key
([reliability.md](reliability.md#unknown-outcomes)). A read that was in flight simply runs
again.
A save that fails is a `warning` event and a log line — the run goes on, and the next save
tries again; a save refused with `409` (the lease is gone) stops the run without writing
anything more. Runs started in process (`run`, `stream`) save no progress: nobody resumes them
after their process died.

**A journal larger than a checkpoint.** agent-runs keeps at most 1 MiB of compact JSON as a
run's checkpoint (a larger one is refused with `413`); a run whose tools returned more than
that still saves its progress and its pause. The journal is then uploaded as a run artifact
(`POST /v1/runs/{id}/artifacts`, as a large `ask` table is) and the checkpoint holds only its
reference, `{"journal_ref": <ArtifactRef>}`; the attempt that continues the run — after a
crash, or after a resume in any process — reads the journal back from it. Nothing to
configure. Each such save is one more artifact of the run, kept with its others (deleted 7 days
after the run ends). A resume reads the journal before it answers the run, so a journal that
cannot be read leaves the run waiting. With no `RUNS_URL` the store is this process's memory:
it bounds nothing, and the same reference is kept there with the run.

## Schedules

`agent.schedule` is one `runs.schedules.create(spec)` (`POST /v1/schedules`): agent-runs upserts
on `(tenant, agent, on_behalf_of, cadence, sha256 of the canonical input)`, so scheduling the
same thing again (a redeploy) answers the schedule that exists, unchanged. The cadence is a cron
expression or one of `hourly`, `daily`, `weekly`, `weekdays`, `manual`. Pause and resume a
schedule with `RunsClient`: `await runs.schedules.update(schedule_id,
ScheduleUpdate(enabled=False))` (or `True`). In process, a due schedule fires when a worker asks
for work; in agent-runs its ticker queues the run.

## The inbox

`await h.inbox("role:procurement")` — the paused runs waiting on that assignee (or, with no
argument, on anyone in the tenant), newest first, as `trellis.runs.RunSummary` (`run_id`,
`agent_id`, `status`, `awaiting` — the interrupt —, `assignee`, `deadline`, `updated_at`). An
`ask` with no `assignee` waits on the run's user (`user:<user>`). Answer one with
`agent.resume`. It is `runs.iterate(status=PAUSED, assignee=..., tenant=..., max_pages=10)`
with pages of 500 (`INBOX_LIMIT`, `INBOX_MAX_PAGES`): past 5000 runs it returns the newest and
logs a warning.

The assignee is a filter, not a lock: every key of the tenant reads every inbox. Answering is
checked in agent-runs: the application's key (it may act for anyone, the default) or an admin
key answers any run, and its `reviewer` is taken as given; a key restricted to listed people
answers only as one of them, a run assigned to that person or to nobody, never a group's run
(`AuthorizationError` otherwise). See [interrupts.md](interrupts.md#answering) and
[agent-runs' rule](https://github.com/amitmohapatra/agent-runs/blob/main/README.md#who-may-answer-a-paused-run).

Notifications (a run paused, escalated or finished) are agent-runs' tenant webhook
subscriptions (`RunsClient.webhooks.create`, `POST /v1/webhooks`), not a harness setting; a
receiver checks each delivery with `trellis.runs.webhooks.verify_signature`
([blocks/runs.md](blocks/runs.md#webhooks)).

## agent-runs wire

`RunsClient` sends `X-API-Key: TRELLIS_API_KEY` on every call. The harness calls, by operation
id: `runs.start` (`POST /v1/runs`, `RunStart` + `queue`), `runs.claim`
(`POST /v1/runs/claim` → `Claimed{run, lease}` or `204`), `runs.heartbeat`
(`POST /v1/runs/{id}/heartbeat`, with the progress `checkpoint` when there is one → `Lease`),
`runs.pause` (`POST /v1/runs/{id}/pause?worker_id=`, an `Interrupt` and the checkpoint),
`runs.resume` (`POST /v1/runs/{id}/resume`, an `InterruptResolution`), `runs.finish`
(`POST /v1/runs/{id}/finish?worker_id=`), `runs.get` (`GET /v1/runs/{id}`), `runs.list`
page by page (`GET /v1/runs?status=PAUSED&assignee=&limit=500&cursor=`, the inbox),
`artifacts.upload` (`POST /v1/runs/{id}/artifacts?worker_id=&checksum=`, an `ask` payload or
a journal larger than a checkpoint → `ArtifactRef`), `artifacts.download` (`GET /v1/artifacts/{id}`),
`runs.cancel` (`POST /v1/runs/{id}/cancel {reason}`), `runs.release`
(`POST /v1/runs/{id}/release {worker_id, checkpoint?}`, a stopping worker's runs) and
`schedules.create` (`POST /v1/schedules`, `ScheduleSpec`).

**The tenant is explicit.** A call whose body names the tenant (a start, a pause, a schedule)
sends it as `X-Trellis-Tenant`; every other call the harness makes passes `tenant=` — the run's
own, from its record or its runtime — so a platform key (which has no tenant of its own) works
on every call, and nothing is remembered between calls. `h.inbox`, `h.feedback` and
`agent.resume` take `tenant=` for a platform key, like `agent.run`.

Every call is retried when it fails on the way — a transport error (a refused connection, a
timeout), `429`, `502`, `503` or `504` — up to 3 times, after the `Retry-After` agent-runs sent
(at most 30 s) or else with exponential backoff and full jitter (a random wait under a ceiling
of 0.25 s, doubled for each retry, at most 5 s). Every write is safe to repeat: a run start is
idempotent on its id, a pause or finish repeated by the same worker with the same status
answers the stored record, an artifact is stored once per checksum, a schedule is upserted. A
claim whose answer was lost leaves that run leased to this worker unworked until the lease
lapses (60 s), when agent-runs queues it again — late, never lost or run twice at once. A
pause or finish the store refuses as a conflict (`ConflictError`) is read back once: when the
run already is what was written (the first attempt landed, its answer did not), the run goes on
as recorded — it is not failed, queued again or executed again.

A refusal raises the SDK's errors (`trellis.runs`), read from agent-runs' problem document
(RFC 9457) by its `code`: every one is a `RunsError` with the problem's `code`, the `status` and
`retryable` (so a run that fails on it keeps whether it may be retried). `LEASE_LOST` is
`LeaseLostError` — the worker no longer holds the run: it stops and writes nothing more — and is
**not** a `ConflictError`, so the conflict read-back above never swallows a lost lease;
`CONFLICT` is `ConflictError` (a run id taken, an answer to another interrupt, an illegal
transition); `NOT_FOUND` is `NotFoundError` (a read by id answers `None` instead). A heartbeat
refused with `409`, whatever its code, is `LeaseLostError`. A call that still fails after its
retries raises `DependencyUnavailableError`, `retryable` true. `LocalRuns` raises the same
classes.
