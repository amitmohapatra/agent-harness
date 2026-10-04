# Runs: `trellis.runs` (Way 2)

`trellis.runs` is agent-runs' Python SDK: durable run records, the worker queue, the inbox of
paused runs, schedules and webhooks, and a framework-neutral worker loop. agent-runs keeps the
state; it never executes an agent. Your framework executes it, and tells agent-runs what
happened: a run started, paused for a person, resumed, finished.

Use it on its own when your framework runs the agent and you want what agent-runs adds
without handing over execution: a run that outlives the process, a queue that a fleet of
workers drains, an inbox where people answer, schedules, and webhooks. Your framework keeps its
own state (a LangGraph checkpointer, an OpenAI Agents `RunState`, a Claude session); agent-runs
keeps the record and the checkpoint you hand it.

Way 1 does all of this with no code: a wrapped agent's runs are recorded, paused, resumed and
queued by the harness ([runs.md](../runs.md)).

## Install and set up

```bash
pip install -e ../agent-runs/sdk/python    # pip trellis-runs; imports as trellis.runs
```

It depends on `httpx`, `pydantic` and `trellis-contracts` only: no framework, no harness.

```python
from trellis.runs import RunsClient

runs = RunsClient()  # RUNS_URL (default http://localhost:8090) and TRELLIS_API_KEY
...
await runs.aclose()  # or: async with RunsClient() as runs
```

| Argument | Default | |
|---|---|---|
| `base_url` | `$RUNS_URL`, else `http://localhost:8090` | where agent-runs is |
| `api_key` | `$TRELLIS_API_KEY` | the Trellis key (issued by the memory service), sent as `X-API-Key` |
| `tenant` | none | the tenant a platform key acts for when a call names none |
| `timeout`, `max_retries` | `10.0`, `3` | seconds per attempt; how often a failed call is sent again |
| `http_client` | its own | an `httpx.AsyncClient` of yours (never closed by the client) |

## A run, a pause, an answer

The records are the contracts models ([contracts.md](contracts.md)): a run is a `RunRecord`
started from a `RunStart`, paused with an `Interrupt`, resumed with an `InterruptResolution`.

```python
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ToolCall,
)
from trellis.runs import RunsClient

async with RunsClient() as runs:
    run = await runs.start(
        RunStart(tenant_id="acme", agent_id="buyer", user_id="ada", input="Top up SKU-1")
    )
    # RUNNING: your process runs it. queue=True instead: QUEUED, for a worker (below)

    # your framework stopped for a person: the run waits in role:procurement's inbox
    asked = Interrupt(
        tenant_id="acme",
        run_id=run.run_id,
        reason=InterruptReason.APPROVAL,
        question="Approve create_po? create_po is irreversible.",
        tool_call=ToolCall(tool="create_po", args={"sku": "SKU-1", "qty": 20}),
        assignee="role:procurement",
    )
    # the checkpoint: what your framework continues from
    await runs.pause(asked, checkpoint={"thread_id": run.run_id})

    # later, any process: the answer
    answer = InterruptResolution(
        interrupt_id=asked.interrupt_id,
        run_id=run.run_id,
        decision=InterruptDecision.APPROVE,
        reviewer="user:lead",
    )
    await runs.resume(answer, tenant="acme")  # RUNNING again, attempt 2

    record = await runs.get(run.run_id, tenant="acme")  # checkpoint, last_resolution
    ...  # continue your framework from record.checkpoint with record.last_resolution
    await runs.finish(run.run_id, RunStatus.SUCCESS, output="PO-SKU-1-20", tenant="acme")
```

* `pause(interrupt, *, checkpoint=None, worker_id=None)`: the `checkpoint` is any JSON your
  framework needs to continue (a thread id, a serialized `RunState`, a session id), up to
  1 MiB. Every read returns it as `RunRecord.checkpoint` until the run ends.
* `resume(resolution)`: the decision is `ANSWER` (with `answer`), `APPROVE`, `REJECT`, `EDIT`
  (with the edited arguments as `payload`) or `CANCEL` (the run ends `CANCELLED`). A run that
  was never queued is `RUNNING` again, for the process that resumes it; a queued one goes back
  to the queue for any worker. The answer is `RunRecord.last_resolution`.
* `deadline` and `escalate_to` on the `Interrupt`: when the deadline passes, agent-runs' ticker
  hands the question to `escalate_to` once, or ends the run `TIMEOUT`.
* Something too large for the question (a table, a diff) is an artifact:
  `ref = await runs.artifacts.upload(run_id, data)`, then `Interrupt(payload_ref=ref, ...)`; a
  reviewer's UI reads it with `runs.artifacts.download(ref.artifact_id)`.
* `finish(run_id, status, *, output=None, error=None, worker_id=None)`: `error` is a contracts
  `AgentError` (`AgentError.of(exc, source="langgraph")`: `source` is an `ErrorSource`, the framework or service that failed).

## The inbox

A person's or a role's inbox is the paused runs assigned to them. `iterate` follows the pages
(`list` returns one `Page`):

```python
async for waiting in runs.iterate(
    status=RunStatus.PAUSED, assignee="role:procurement", tenant="acme"
):
    print(waiting.run_id, waiting.awaiting.question)  # a RunSummary; awaiting is the Interrupt
```

`runs.resolutions(run_id)` is the audit trail: every answer, who gave it and when.

## Workers

`Worker` claims queued runs of some agents and hands each to your handler, an
`async (Job) -> ...`. The job carries the claimed `record` (with its `checkpoint` and
`last_resolution`) and writes as this worker:

```python
from trellis.contracts import RunStart, RunStatus
from trellis.runs import Job, RunsClient, Worker


async def handle(job: Job) -> None:
    answer = await my_graph.ainvoke(job.record.input)  # your framework, untouched
    await job.finish(RunStatus.SUCCESS, output=answer)


async with RunsClient() as runs:
    await runs.start(RunStart(tenant_id="acme", agent_id="triage", input={"ticket": 7}), queue=True)
    await Worker(runs, handle, ["triage"], concurrency=4).serve()  # until SIGTERM
```

* The lease (60 s) is renewed every third of it while the handler runs. A heartbeat refused
  with `LEASE_LOST` cancels the handler: another worker has the run.
* `job.checkpoint(data)` saves progress (and extends the lease): the attempt after a crash
  claims the run with that checkpoint. `job.pause(interrupt, checkpoint=...)` and
  `job.finish(...)` are fenced: after the lease was lost they raise `LeaseLostError`.
* `serve()` stops on SIGTERM or SIGINT: no new claims, the runs held get 25 s, then are
  released (cancelled with `trellis.runs.RELEASED`, nothing written) for another worker. `run()`
  is the same loop without signals (`stop()` ends it), `run_once()` claims and runs one.
* A resumed queued run comes back to a worker like any other: the same handler reads
  `record.last_resolution` to know what the person said.

[`examples/blocks_worker.py`](../../examples/blocks_worker.py) runs a queue of refunds through
a `Worker`, pauses one for finance, answers it from the inbox, and finishes it on the next claim.

## Schedules

```python
from trellis.contracts import ScheduleSpec
from trellis.runs import ScheduleUpdate

briefing = await runs.schedules.create(
    ScheduleSpec(
        tenant_id="acme",
        agent_id="briefing",
        name="morning briefing",
        cadence="0 8 * * 1-5",
        timezone="Europe/Berlin",
        on_behalf_of="ada",
        input={"topic": "inbox"},
    )
)  # an upsert: the same agent, person, cadence and input answer the existing one
await runs.schedules.fire(briefing.schedule_id)  # a QUEUED run now
await runs.schedules.update(briefing.schedule_id, ScheduleUpdate(enabled=False))  # pause it
```

A fired run is queued: your `Worker` runs it.

## Webhooks

Instead of polling the inbox, subscribe once per tenant to pauses, escalations and endings.
The secret is in the create's answer only:

```python
from trellis.runs import WebhookEvent

hook = await runs.webhooks.create(
    "https://ui.example/hooks/trellis", [WebhookEvent.PAUSED, WebhookEvent.FINISHED]
)
keep_secret(hook.secret)
```

Every delivery is signed (`X-Trellis-Signature: t=<unix seconds>,v1=<HMAC-SHA256>` over
`"<t>.<raw body>"`). A receiver verifies the raw bytes before it parses them:

```python
from trellis.runs.webhooks import SIGNATURE_HEADER, parse_delivery, verify_signature


async def trellis_hook(request: Request) -> Response:  # Starlette or FastAPI
    body = await request.body()
    if not verify_signature(SECRET, request.headers.get(SIGNATURE_HEADER), body):
        return Response(status_code=401)
    delivery = parse_delivery(body)  # a WebhookDelivery: event_id, type, data.run (a RunSummary)
    if not seen(delivery.event_id):  # delivered at least once: drop repeats by event_id
        notify(delivery.type, delivery.data.run)
    return Response(status_code=204)
```

`verify_signature(secret, header, body, *, now=None, tolerance=300)` refuses a signature more
than five minutes old or ahead and never raises on a malformed header. `sign` is the one
implementation of the scheme: agent-runs signs with it, and so does the harness's A2A push
(Way 1), so one receiver checks both. A `2xx` accepts a delivery; `408`, `429`, `5xx` and an
unreachable receiver are retried.

## Behaviour

* **Tenancy.** A tenant key names its tenant. A platform key names it on every call: from the
  body for `start`, `pause` and `schedules.create`, else from `tenant=`, else from the client's
  `tenant`. Nothing is remembered between calls.
* **Reads and writes.** A read by id answers `None` for a record that does not exist; a write
  raises. Every write is safe to repeat: a start is idempotent on its `run_id` (or
  `idempotency_key`), a repeated pause or finish answers the stored run, an artifact is stored
  once per checksum, a schedule create is an upsert.
* **Retries.** A call that fails on the way (no response, `429`, `502`, `503`, `504`) is sent
  again up to `max_retries` times, after the `Retry-After` agent-runs asked for (at most 30 s)
  or a full-jitter backoff from 0.25 s doubling to 5 s.
* **Errors.** Every refusal is a `RunsError` with `code`, `status`, `retryable`, `request_id`:
  `NotFoundError`, `ConflictError` (an illegal transition, an answer to another interrupt),
  `LeaseLostError` (the worker no longer holds the run: stop, write nothing; **not** a
  `ConflictError`), `ValidationError`, `PayloadTooLargeError`, `RateLimitedError`,
  `AuthenticationError`, `AuthorizationError`, `DependencyUnavailableError`.

The full operation table, the configuration and the error classes are in the SDK's
[README](https://github.com/amitmohapatra/agent-runs/blob/main/sdk/python/README.md); the API
in agent-runs' [docs/api.md](https://github.com/amitmohapatra/agent-runs/blob/main/docs/api.md).

## With Way 1

The harness's run store *is* `RunsClient` when `RUNS_URL` is set (`h.runs`), and its worker
(`h.worker`, `python -m trellis.harness.worker`) is `trellis.runs.Worker` running wrapped
agents. So both ways share one agent-runs: a wrapped agent's paused runs and your own are in
the same inbox, and one webhook receiver serves both ([mixing.md](mixing.md)).

Recipes: [LangGraph](langgraph.md) (the checkpointer keeps the graph; agent-runs the run and
the inbox), [OpenAI Agents SDK](openai-agents.md) (the `RunState` as the checkpoint),
[Claude Agent SDK](claude-agent-sdk.md) (the session id as the checkpoint).
