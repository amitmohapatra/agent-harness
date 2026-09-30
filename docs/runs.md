# Runs, workers, schedules

Every run has a record (contracts `RunRecord`): in agent-runs when `RUNS_URL` is set, in
process otherwise (`LocalRuns`, same behaviour, nothing survives a restart). Both are in
`trellis/harness/clients/runs.py`.

| Call | Record |
|---|---|
| `agent.run` / `agent.stream` | `RUNNING` (recorded in process), then `PAUSED` or an ending |
| `agent.start` | `QUEUED`; `RunHandle.result()` waits for a pause or an ending |
| `agent.resume` | the next attempt: `RUNNING` for an in-process run, `QUEUED` again for one that came from the queue; `CANCELLED` on cancel |
| `agent.schedule(cron, input, on_behalf_of=, tz=)` | a `Schedule`; each fire queues a run acting for `on_behalf_of` |

## Workers

```python
await h.worker([agent_a, agent_b]).run()     # until cancelled; 4 runs at a time
python -m trellis.worker app.agents:h        # every agent the Harness `h` wraps
```

A worker claims a queued run under a 60 s lease and heartbeats it every 20 s. It names itself
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
`GET /v1/runs?status=PAUSED&assignee=` (summaries), `POST /v1/schedules` (`ScheduleSpec`). A
refused or unreachable write raises `RunStoreError` (`LeaseLost` for a lease that is no longer
the worker's).
