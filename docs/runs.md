# Runs, workers, schedules

Every run has a record (contracts `RunRecord`): in agent-runs when `RUNS_URL` is set, in
process otherwise (`LocalRuns`, same behaviour, nothing survives a restart). Both are in
`trellis/harness/clients/runs.py`.

| Call | Record |
|---|---|
| `agent.run` / `agent.stream` | `RUNNING` (recorded in process), then `PAUSED` or an ending |
| `agent.start` | `QUEUED`; `RunHandle.result()` waits for a pause or an ending |
| `agent.resume` | the next attempt: `RUNNING` for an in-process run, `QUEUED` again for one that came from the queue; `CANCELLED` on cancel |
| `agent.schedule(cron, input, on_behalf_of=, tz=, tenant=)` | a `Schedule`; each fire queues a run acting for `on_behalf_of` |

`start` returns a `RunHandle`: `run_id`, `await handle.status()` (the `RunRecord`; a run the
store does not have raises `ConfigurationError`) and `await handle.result(timeout=None)`, which
reads the run every 0.5 s until it pauses or ends and returns a `Result` (`asyncio.timeout`
raises `TimeoutError` past `timeout`). A queued run's input must be JSON (`start` refuses
anything else); a run started in process records its input as JSON where it can and as text
where it cannot, and so does its answer.

## Workers

```python
await h.worker([agent_a, agent_b]).run()     # until cancelled; 4 runs at a time
python -m trellis.worker app.agents:h        # every agent the Harness `h` wraps
```

`h.worker(agents, *, concurrency=4)` needs at least one agent. `await worker.run()` claims
and executes until cancelled, `concurrency` runs at a time, asking again every second when
the queue is empty; cancelling it cancels the runs it holds (they end `CANCELLED`) and drains
the background writes. `await worker.run_once()` claims one run and executes it to its end or
pause, and returns `False` when nothing was queued. A claim that fails is logged and counts as
no work; a run that breaks the harness itself is logged and the worker goes on.

A worker claims a queued run under a 60 s lease and heartbeats it every 20 s (a failed
heartbeat is logged and retried at the next beat). It names itself
on the pause, the finish and an artifact upload, so a worker whose lease lapsed cannot write
over a run another worker has since claimed; a heartbeat refused (`409`) stops the run without
writing. A lapsed lease sends the run back to the queue as its next attempt. A paused run
carries its journal as the run's checkpoint, so the worker that claims it after a resume — any
worker — repeats no question and no tool call made before the pause.

## Schedules

One `POST /v1/schedules`: agent-runs upserts on `(tenant, agent, on_behalf_of, cadence,
sha256 of the canonical input)`, so scheduling the same thing again (a redeploy) answers the
schedule that exists, unchanged. The cadence is a cron expression or one of `hourly`, `daily`,
`weekly`, `weekdays`, `manual`. Pause and resume a schedule in agent-runs:
`PATCH /v1/schedules/{id} {"enabled": false | true}`. In process, a due schedule fires when a
worker asks for work; in agent-runs its ticker queues the run.

## The inbox

`await h.inbox("role:procurement")` — the paused runs waiting on that assignee (or, with no
argument, on anyone in the tenant), newest first, as `RunSummary` (`run_id`, `agent_id`,
`status`, `awaiting` — the interrupt —, `assignee`, `deadline`, `updated_at`). An `ask` with
no `assignee` waits on the run's user (`user:<user>`). Answer one with `agent.resume`.

Notifications (a run paused, escalated or finished) are agent-runs' tenant webhook
subscriptions (`POST /v1/webhooks`), not a harness setting.

## agent-runs wire (0.2)

`X-Api-Key: TRELLIS_API_KEY` on every call, `X-Trellis-Tenant` naming the run's tenant.
`POST /v1/runs` (`RunStart` + `queue`), `POST /v1/runs/claim` (`{run, lease}` or `204`),
`POST /v1/runs/{id}/heartbeat`, `POST /v1/runs/{id}/pause?worker_id=` (an `Interrupt` and the
checkpoint), `POST /v1/runs/{id}/resume` (an `InterruptResolution`),
`POST /v1/runs/{id}/finish?worker_id=`, `POST /v1/runs/{id}/artifacts?worker_id=&checksum=`
(an `ask` payload, → `ArtifactRef`), `GET /v1/artifacts/{id}`, `GET /v1/runs/{id}`,
`GET /v1/runs?status=PAUSED&assignee=&limit=500&cursor=` (summaries, page by page),
`POST /v1/schedules` (`ScheduleSpec`).

Every call is retried when it fails on the way — a transport error (a refused connection, a
timeout), `429`, `502`, `503` or `504` — up to 3 times, after the `Retry-After` agent-runs sent
(at most 30 s) or else with exponential backoff and full jitter (a random wait under a ceiling
of 0.25 s, doubled for each retry, at most 5 s). Every call is safe to repeat: a run start is
idempotent on its id, a pause or finish repeated by the same worker with the same status
answers the stored record, an artifact is stored once per checksum, a schedule is upserted. A
pause or finish the store refuses as a conflict is read back once: when the run already is
what was written (the first attempt landed, its answer did not), the run goes on as recorded —
it is not failed, queued again or executed again.

A refusal raises `RunStoreError` — with the problem's `code`, the `status` and `retryable`, so a
run that fails on it keeps whether it may be retried — read from agent-runs' problem document
(RFC 9457) by its `code`: `LEASE_LOST` is `LeaseLost` (the worker no longer holds the run: it
stops and writes nothing more), `CONFLICT` is `Conflict` (a run id taken, an answer to another
interrupt, an illegal transition), `NOT_FOUND` is `NotFound`; an answer without a code is read
by its status (`409` `Conflict`, `404` `NotFound`). A heartbeat refused with `409`, whatever its
code, is `LeaseLost`. A call that still fails after its retries raises `RunStoreError` with
`retryable` true.

The inbox follows agent-runs' `Link: <…>; rel="next"` cursor page by page (500 runs a page), up
to 10 pages; past that it returns the newest 5000 and logs a warning.
