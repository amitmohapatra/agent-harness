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
| `agent.schedule(cron, input, on_behalf_of=, tz=, tenant=, ...)` | a `Schedule`; each fire queues a run acting for `on_behalf_of`, with the run options `start` takes (below) |

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

**Your own worker or scheduler.** `await agent.execute(job)` is the claimed run's next attempt
— fresh, continued after an answer, or after a worker died — with its journal, governance,
memory and time limit: the one thing a worker does with a wrapped agent's run. Any loop can do
it: `trellis.runs.Worker(runs, agent.execute, [agent.id])`, or your own around
`runs.claim(worker_id, [agent.id])` (`job` is a `trellis.runs.Job`: the claimed record, the
worker's id, the lease's length). A job of another agent is refused (`ConfigurationError`).
[examples/react_with_blocks.py](../examples/react_with_blocks.py) runs one with a loop of its own.

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
(e.g. a termination grace period of 45 s). Cancelling `worker.run()` instead stops the runs
it holds without writing anything: their leases lapse and agent-runs queues them again. A
worker's run ends `CANCELLED` only when someone asked (`agent.cancel`: its heartbeat says
`cancel_requested`), and `TIMEOUT` when its working time is used up. The attempt writes that,
as any run of it out of time does: its error (`run_timeout`, "the run worked past its time
limit of …"), its tool call cut short and its `RUN_FINISHED` (outcome `timeout`) on its events
— also when the worker's own clock for the working time, armed at the claim, stops it first.

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
was given, and a `ReAct`'s graph checkpoint — as the run's checkpoint on a heartbeat
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

## Queue order and busy conversations

**What.** `agent.start(..., priority=0, concurrency_key=None)`: among the tenant's queued runs a
higher `priority` (-1000 to 1000) is claimed first, then the oldest; runs sharing a
`concurrency_key` run one at a time (agent-runs' limit, one unless its operator says
otherwise), the others wait `QUEUED`.

**When.** `priority` for an urgent run; `concurrency_key` for anything that must not run twice
at once (one customer's account, one repository).

**Automatic: a second message to a busy conversation waits.** A queued run of a conversation
(`thread=`) gets `concurrency_key="thread:<thread>"` unless you name one: a second message sent
while the first run works is answered after it, never beside it (both would write the same
thread). The same holds in process: a run of a conversation started with `run`, `stream`,
`serve_chat` or `serve_a2a` (or resumed) waits while another run of the same agent and
conversation works in this process, in order. A run with no `thread` is its own conversation.
`LocalRuns` claims the same way agent-runs does. A sub-agent's run and a worker's are not held
in process (the queue holds a worker's).

**On failure.** A priority outside -1000..1000 or a key over 200 characters is refused by the
contract (`ValidationError`). A conversation's runs waiting in process across replicas are not
held across them: queue them (`start`) for that.

## A run's events from anywhere

**What.** With `RUNS_URL`, every attempt appends its events (the `RunEvent`s `stream` yields)
to the run's event log in agent-runs as they happen, so any replica streams any run:
`agent.events(run_id, *, after=0)` yields them — those past position `after`, then each as it
is appended, until the run ends (a paused run's stream stays open for its next attempt) —
wherever the run executes (a worker, another replica). `serve_chat`'s reconnect route reads a
run this process did not serve from there too ([surfaces.md](surfaces.md)).

**Automatic.** Batched (at most 500 an append), in order, one append at a time, in the
background: the run never waits for it except at its end, when its last events (`INTERRUPT`,
`RUN_ERROR`, `RUN_FINISHED`) are appended before the pause or the ending is recorded (the log
takes nothing after) — and whoever watches in process hears `RUN_FINISHED` only once it is
recorded. A worker's appends name it (fenced by its lease). Without `RUNS_URL` nothing is kept:
`agent.events` follows the run's attempts in this process from now on.

**On failure.** Best-effort: events agent-runs refuses or cannot take (after its SDK's retries)
are dropped with one `warning` event (`events_undelivered`, logged, counted
`trellis.run_events.undelivered`); the run goes on. A lost lease stops the log.

## Admission: agent-runs' rate limit

A call agent-runs refuses with `429` (the tenant's request budget) is retried by its SDK after
the `Retry-After` it sends (at most 3 times, each wait at most 30 s). When it still refuses,
`run`, `stream`, `start`, `resume` and `schedule` raise `trellis.harness.agent.Throttled`: a
contracts error (`RUNS_RATE_LIMITED`, category `RATE_LIMIT`, `retryable=True`,
`details["retry_after"]` the seconds asked for), counted (`trellis.runs.rate_limited`). The
harness does not retry it again (one retry layer, the SDK's); `serve_chat` answers `429`
(`RATE_LIMIT`) with `Retry-After`. A tenant over agent-runs' cap of running runs is not refused:
its queued runs wait.

## Schedules

`agent.schedule` is one `runs.schedules.create(spec)` (`POST /v1/schedules`): agent-runs upserts
on `(tenant, agent, on_behalf_of, cadence, sha256 of the canonical input)`, so scheduling the
same thing again (a redeploy) answers the schedule that exists, unchanged. A scheduled run
carries everything a started one can (contracts 0.6.1, ADR 0007): `schedule` takes the same
run options as `start`, and every fired run gets them — agent-runs and `LocalRuns` alike:

| Run option | `agent.start(...)` keeps it in | `agent.schedule(...)` keeps it in | Each fired run gets |
|---|---|---|---|
| `timeout=` (else the agent's `h.wrap(timeout=)`) | `RunStart.timeout_seconds` | `ScheduleSpec.timeout_seconds` | `timeout_seconds` |
| the agent's `version` | `RunStart.agent_version` | `ScheduleSpec.agent_version` | `agent_version` |
| `without=` | `RunStart.metadata["without"]` | `ScheduleSpec.metadata["without"]` | `metadata["without"]` |
| `framework_options=` (JSON) | `RunStart.metadata["framework_options"]` | `ScheduleSpec.metadata["framework_options"]` | `metadata["framework_options"]` |
| `priority=` | `RunStart.priority` | `ScheduleSpec.priority` | `priority` |
| `concurrency_key=` (`start`: by default `thread:<thread>`) | `RunStart.concurrency_key` | `ScheduleSpec.concurrency_key` | `concurrency_key` |

agent-runs copies a schedule's `metadata` into each fired run's `RunStart.metadata`, its own
keys (`schedule_id`...) laid over it, so a worker reads a scheduled run's choices exactly as a
started run's. `priority=` and `concurrency_key=` need trellis-contracts 0.6.1 or later: with an
older one `schedule` refuses them (`ConfigurationError`) rather than drop them. In Way 2 the
same keys work without a harness: `runs.schedules.create(ScheduleSpec(..., priority=,
metadata={"without": [...], "framework_options": {...}}))`, and whatever executes the fired run
(a `trellis.runs.Worker` handler, `agent.execute`) reads them from `job.record`. The cadence is a cron
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
([blocks/runs.md](blocks/runs.md#webhooks), [interrupts.md](interrupts.md#telling-people)).
Runs kept in process (no `RUNS_URL`) have no webhooks: nothing is told unless a run hook of
yours tells it.

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
