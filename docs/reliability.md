# Reliability: time limits, retries, crashes, cancel

What keeps a run bounded and its side effects counted when a tool hangs, a service blips, a
worker dies or somebody changes their mind. You write only what only you know — how long a
thing may take (`timeout=`, `deadline=`) and which version of your code is running
(`version=`). Everything else on this page is automatic, with no setting: retries of calls that
only read, idempotency keys, the crash-safe record of every write, unknown outcomes, the last
good tool list.

It all sits where every framework passes: the tool bridge (every harness tool call, whichever
framework makes it), `trellis.current()` (the run a tool or node executes in) and the pipeline
(every attempt of every run). So it holds the same for a function, `ReAct`, LangGraph and
LangChain, Deep Agents, the OpenAI Agents SDK and the Claude Agent SDK, for `run`, `stream`,
`start` + workers, resumes in another process, `serve_chat` and `serve_a2a`. Way 2 has the same
rules where it has the same pieces: `governed` for your own tools, `trellis.runs` for the run.

| Feature | You write | Automatic |
|---|---|---|
| [Tool timeouts](#tool-timeouts) | `@tool(timeout=)`, `openapi(timeout=)`, `a2a(timeout=)`, `governed(timeout=)` | MCP calls and every call bounded by what is left of the run |
| [Model timeouts](#model-timeouts) | `ReAct(model_timeout=)` | the gateway's own retries stay inside it |
| [Run time limit and deadline](#run-time-limit-and-deadline) | `run`/`stream`/`start(timeout=, deadline=)` | the clock survives a crash; workers stop in time |
| [Retries](#retries) | — | reads retried, writes run once |
| [Idempotency keys](#idempotency-keys) | the tool hands `trellis.current().idempotency_key` to its service | OpenAPI, A2A and MCP calls carry it |
| [Unknown outcomes](#unknown-outcomes) | — | a write that timed out or was cut by a crash is told to the model, never re-run blind |
| [Calls made at once](#calls-made-at-once) | — | numbered in order, identical ones in turn, replayed call by call, saved one save at a time |
| [Cancel](#cancel) | `agent.cancel(run_id, reason=)` | wherever the run is: queued, paused, here, on a worker; its sub-agents' runs with it |
| [Releasing on shutdown](#releasing-on-shutdown) | — | a stopping worker hands its runs back at once |
| [The last good tool list](#the-last-good-tool-list) | — | a gateway that is down does not empty the toolbox |
| [Agent version](#agent-version) | `h.wrap(..., version=)` or `TRELLIS_AGENT_VERSION` | recorded with each run; a resume on another version says so |

## Tool timeouts

**What.** The most one tool call may take, in seconds, its retries included.

**When.** Any tool that calls something that can hang: an HTTP API, a database, another agent.

**Where.** Way 1: every harness tool, in every adapter (they all call the bridge). Way 2:
`governed(fn, gov, timeout=...)` around your own tools.

**How.**

```python
@tool(side_effects="write", timeout=20)
async def transfer(amount: int, to: str) -> str: ...


tools = [transfer, openapi(spec_url, timeout=10), a2a(planner_url, timeout=300)]
```

| Source | Its own limit |
|---|---|
| `@tool(timeout=)` / `tool(fn, timeout=)` | none unless you give one |
| `openapi(spec, timeout=)` | 120 s (`REMOTE_TIMEOUT_SECONDS`, the one default of a remote tool); also the document's fetch |
| `a2a(url, timeout=)` | 120 s (`REMOTE_TIMEOUT_SECONDS` too; `remote()` the same), the whole exchange |
| an MCP tool (Bifrost) | none of its own: the request waits what is left of the run's time (else the Bifrost SDK's 60 s) |
| `governed(fn, gov, timeout=)` (Way 2) | none unless you give one |

**Automatic.** The limit that applies is the tightest of the tool's own, what is left of the
run's `timeout` and what is left until its `deadline` — an inner limit only ever tightens an
outer one. `trellis.current().remaining()` tells code in a tool how long it may still take;
the gateway's requests for MCP tools and model calls are sent with it. A sync function runs in
a worker thread, so a slow one blocks neither the run's other work (its heartbeats, other runs
on the worker) nor its timeout.

**On failure.** Past the limit the call is stopped — an async tool and an HTTP request really
are cancelled — and the outcome is `TIMEOUT` (`ToolOutcome.status`; a call that fails with a
timeout of its own, an `httpx` read timeout, counts too):

* a call that only reads: the model reads `"<tool> timed out after 20s"`;
* a call that does more than read: its effect is unknown — see [unknown outcomes](#unknown-outcomes).

Way 2: `governed` raises `ToolTimeout` (a `ToolError`, category `TIMEOUT`, `unknown` set for a
call that does more than read) whose message is the same text, for your framework to show its
model.

**Limitation: a sync function cannot be stopped.** Python cannot kill a thread: past its
timeout a sync tool's result is dropped, but the thread runs on to its end, side effects and
all. That is why a write that timed out is *unknown*, never *failed*. Make a tool that writes
`async`, with a client that honours cancellation (`httpx`, `asyncpg`, …), when it must really
stop.

## Model timeouts

**What.** The most one model call of `ReAct` may take.

**When.** A model that sometimes hangs, or a run that must not spend its whole budget on one
call.

**Where.** `ReAct(..., model_timeout=seconds)`. A framework's own model clients (LangChain's
`ChatOpenAI(timeout=)`, the OpenAI Agents SDK's) are configured on the client, as before; the
run's own limit still bounds them.

**How.** `ReAct(system=..., model="provider/model", model_timeout=30)`.

**Automatic.** The Bifrost SDK retries a completion on `429`/`5xx`/a dropped connection (3
attempts in all, honouring `Retry-After`); those retries happen *inside* `model_timeout`, and
each request is sent with what is left of it. Without `model_timeout` each attempt has the
SDK's 60 s, and what is left of the run bounds them all.

**On failure.** The run fails `ERROR` with a `ModelError` that says
`"the model did not answer within 30s"`, `retryable` true: a queued run goes back on the
queue for a later attempt ([run retries](#retries)), and its model steps already journaled
are not asked again.

## Run time limit and deadline

**What.** `timeout=`: the most *working* time a run may take — time `RUNNING`, across every
attempt, not time queued or waiting for a person. `deadline=`: when it must have ended,
waiting included. Either, both, or neither.

**When.** `timeout` for "do not work on this for more than ten minutes", even if a review in
the middle takes a day; `deadline` for "the answer is useless after 9:00".

**Where.** The agent's default limit: `h.wrap(..., timeout=)` — every run it starts, on every
entry: `run`, `stream`, `start`, `serve_chat`, `serve_a2a`, `h.evaluate` and its scheduled
runs. A run's own: `agent.run`, `agent.stream` and `agent.start(timeout=, deadline=)`, which
override the agent's (Way 1); `RunStart.timeout_seconds` and `RunStart.deadline` with
`RunsClient.start` (Way 2). Every adapter.

**How.**

```python
agent = h.wrap(target, id="reconcile", timeout=900)  # every run of it, wherever it starts
result = await agent.run("Reconcile March", user="ada", timeout=600)  # this one: 600
handle = await agent.start(task, user="ada", timeout=600, deadline=tomorrow_9am)
```

**Automatic.**

* Sent as `RunStart.timeout_seconds` and `RunStart.deadline` (only when set), so the record
  says it; agent-runs adds up the run's working time (`RunRecord.worked_seconds`) every time it
  leaves `RUNNING`, so the clock continues after a pause, a crash and a retry.
* An attempt works at most what is left: in process, the limit less the time already worked,
  and the deadline; in a worker, what its lease says is left (`Lease.remaining_seconds`: the
  run's limit or the platform's — `RUNS__RUNS__MAX_RUN_SECONDS` in agent-runs, the lesser —
  less what it worked), so a worker stops on time instead of being cut off.
* agent-runs enforces both itself (its ticker), so a run is bounded even when its worker dies
  and never comes back.
* Every entry starts an attempt the one way (`pipeline.attempt`, from the run's record), so a
  chat run, an A2A task and an evaluation item are bounded as `agent.run` is. A scheduled run's
  record names no limit (agent-runs' schedules carry none yet): the worker that runs it holds
  it to the agent's `timeout`, but agent-runs' ticker does not, and the run records no agent
  version.

**On failure.** The run ends `TIMEOUT` with an error `run_timeout` (working time) or
`run_deadline`, category `TIMEOUT`, `retryable` false — a run out of time is not run again.
`RUN_ERROR` and `RUN_FINISHED` (outcome `timeout`) are on the stream; the record and the
memory service's transcript say what it did until then.

## Retries

**What.** A call that only reads (or whose tool says it is idempotent) is tried again after an
error that may pass; a call that does more runs once.

**When.** Always; there is nothing to set. The tool's side effects decide — as governance sees
them: the catalog's `risk` over the tool's own declaration.

**Where.** Every harness tool call, every adapter; `governed` in Way 2 (`side_effects="read"`).

**How it decides.** An error may pass when it says so (`AgentError.of(exc).retryable`: its own
`retryable`, or its category — a timeout, a rate limit, a dependency down). An OpenAPI
operation answering `408`, `425`, `429`, `500`, `502`, `503` or `504` raises such an error; a
`404` or a `422` does not.

**Automatic.** Up to 2 retries (`READ_RETRIES`), each after a random wait under 0.5 s, doubled
per retry (`RETRY_BACKOFF_SECONDS`), all within the call's timeout. `ToolOutcome.attempts`
says how many were made. An MCP tool whose server says `idempotentHint` is idempotent; a
function tool is when `@tool(idempotent=True)` says so (it hands its service the call's
`trellis.current().idempotency_key`), and a tool of your own source when its
`ToolSpec(idempotent=True)` does.

**On failure.** The last error is the outcome the model reads (`"<tool> failed: ..."`), as
before.

### Every retry layer

Each layer bounds something else; they nest, so their attempts multiply unless a limit above
them bounds them.

| Layer | What it retries | How often, how long | Bounded by |
|---|---|---|---|
| Bifrost SDK (models) | a completion on a dropped connection, `408/409/425/429/5xx` | 3 attempts, 0.5 s jittered backoff, `Retry-After` ≤ 30 s, 60 s per attempt; a circuit breaker opens for 30 s after 5 failed calls | `model_timeout`, the run's time |
| Bifrost SDK (MCP) | nothing: a tool may have side effects | 1 attempt | the call's limit (sent as the request timeout) |
| memory SDK | reads, and writes with an idempotency key, on `429/502/503/504`, timeouts, dropped connections | 4 attempts, full jitter, `Retry-After` ≤ 30 s, 10 s per attempt; breaker 5 / 30 s | the run's time (context push, pull tools) |
| runs SDK | every call (each is safe to repeat) on transport errors, `429/502/503/504` | 4 attempts, 0.25 s → 5 s full jitter, `Retry-After` ≤ 30 s, 10 s per attempt | nothing: a pause or an ending must land |
| harness tool retries | a read (or idempotent) tool after an error that may pass | 3 attempts, 0.5 s doubling, random | the tool's timeout, the run's time |
| background writes | memory writes, judges, the grounding check | 3 attempts, 0.5 s doubling, full jitter; then the spool | the drain bound (10 s) at shutdown |
| agent-runs: run retries | a queued run its worker ended `ERROR` with a retryable error | 3 more attempts, after 10 s, 20 s, 40 s (jittered, ≤ 10 min) | the run's `timeout` and `deadline` |
| agent-runs: lapsed leases | a queued run whose worker stopped heartbeating | up to 5 lapses, after 5 s doubling (≤ 1 min); then `ERROR` (`lease_expired`) | the run's `timeout` and `deadline` |

**What multiplies.** A model call without `model_timeout` can take 3 × 60 s plus the waits;
with it, at most `model_timeout`. A read tool is tried 3 times, but always within its own
timeout. A run that fails retryably is run up to 4 times by agent-runs — each attempt
replaying its journal, so no write repeats — but its working time is one clock across all of
them: `timeout=` bounds the sum, and `deadline=` the wall clock. A run kept in its caller's
process (`run`, `stream`) is never retried by agent-runs: the caller gets the error.

## Idempotency keys

**What.** A key for each tool call, the same for that call in every attempt of the run — after
a pause, after a crash — and different for every other call (the run, the tool, its arguments
and how many such calls came before).

**When.** A tool that calls a service which deduplicates (`Idempotency-Key` on an HTTP API, a
`client_request_id`, a unique constraint).

**Where.** `trellis.current().idempotency_key` inside any harness tool call (Way 1, every
adapter). Way 2 code has no harness run: derive one from your run (`job.record.run_id`) and
the call.

**How.**

```python
@tool(side_effects="write", timeout=20)
async def pay(invoice: str, amount: int) -> str:
    key = trellis.current().idempotency_key
    return await bank.pay(invoice, amount, idempotency_key=key)
```

**Automatic.** An OpenAPI operation that does more than read sends it as `Idempotency-Key`; an
A2A call opens its task with it as the message id; an MCP call through Bifrost carries it as
the tool call's id (in the gateway's log — the MCP protocol gives a server no field for it, so
an MCP server that must deduplicate needs a key argument of its own). It is not
`ToolCall.idempotency_key`: that one names the call itself — the same tool and arguments in any
run — so the memory service learns approvals across runs from it.

**On failure.** Nothing to fail: a service that ignores the key simply does not deduplicate,
and the journal still keeps the harness from repeating a call it saw complete.

## Unknown outcomes

**What.** A call that does more than read and whose effect is not known: it timed out, or it
was running when its worker died.

**When.** Automatically, for `write` and `irreversible` tools (as governance sees them).

**Where.** Every harness tool call, every adapter, in process and in workers.

**How it works.**

1. Before a call that does more than read runs, the journal marks it *started* and saves
   itself as the run's progress checkpoint (in a worker, on a heartbeat).
2. It times out: the outcome is `TIMEOUT`, `metadata["unknown"]` true, `error_class`
   `OutcomeUnknown`, and the model reads `"<tool> timed out after 20s; it may or may not have
   taken effect: check before calling it again"`.
3. Its worker died while it ran: the next attempt finds it started and never finished. A tool
   that is idempotent runs again, with the same idempotency key; any other is not run — the
   outcome is `CANCELLED`, `OutcomeUnknown`, and the model reads `"<tool> was interrupted by
   a crash; it may or may not have taken effect: check before calling it again"`.

**Automatic.** What the model was told is journaled like an output: a resumed run, a re-run
after another crash, a resume in another process all read the same text and never run the call
again. The memory service's tool record has the status (`timeout`, `cancelled`) and
`error_class` `OutcomeUnknown`. A call that failed outright, or that paused to ask a person, is
not unknown: it runs again on resume, as before. A call that only reads is simply run again.

**On failure.** By design the harness never guesses: the model (or the person reading the
transcript) checks — a "get transfer status" tool next to "transfer" is the usual answer.

### Crash behaviour, end to end

A worker holds a run under a 60 s lease, renewed every 20 s. When it dies the lease lapses and
agent-runs queues the run again (after 5 s, doubling per lapse; the fifth lapse ends it
`ERROR`). The next attempt claims it with the last checkpoint — the journal — and replays it:
questions answered are not asked again, tool calls completed return their recorded output,
`ReAct`'s model steps are not asked again, a write that was in flight is [unknown](#unknown-outcomes)
(or re-run with its key when idempotent), and the working time already spent still counts
against `timeout`. Runs kept in process (`run`, `stream`) save no progress: nobody resumes them
after their process died. A [sub-agent](subagents.md)'s run working inside a call when the
worker died is saved with its parent's progress, and the parent's next attempt continues it the
same way.

## Calls made at once

**What.** Tool calls that run at the same time in one run: `ReAct`'s reads (several calls in
one step: the reads at once, then the writes one at a time in the model's order —
[react.md](frameworks/react.md)), and the frameworks that run tools concurrently themselves
(LangGraph's tool node, the OpenAI Agents SDK, a function's `asyncio.gather`).

**When.** Automatically, whenever a framework (or the model) makes them.

**Where.** The bridge, every adapter; Way 2's `governed` calls are your framework's to order.

**Automatic.**

* Each call's step is numbered as it arrives — `ReAct` numbers a step's calls in the model's
  order before any runs — and is the call's `ToolCall.step` (an approval shows it).
* Identical calls (the same tool with the same arguments) made at once take their turn: one
  runs, then the next, in the order they were made, so each has its own occurrence in the
  journal and its own idempotency key; different calls run together.
* The journal records each call where it belongs, so a resumed run hands each call the output
  it got the first time, whatever order they finished in.
* Progress saves go one at a time, and each saves the journal as it is then: a later checkpoint
  never lands before an earlier one, and the last holds every call that finished.

**On failure.** A call that pauses (an approval, an `ask` inside a tool) lets the calls beside
it finish — they are journaled — and then the run pauses; on resume they replay. Cancelling the
run cancels every call still running.

## Cancel

**What.** Stop a run, whatever it is doing, and keep why.

**When.** A duplicate, a user who changed their mind, a run that is no longer needed.

**Where.** Way 1: `await agent.cancel(run_id, reason=...)` (a platform key adds `tenant=`) and
`await handle.cancel(reason=...)` on a `RunHandle`. Way 2: `await runs.cancel(run_id,
reason=...)` with `RunsClient`, and `trellis.runs.Worker` stops your handler. Every adapter.

**How it works.**

| The run is | It ends |
|---|---|
| queued, or paused for a person | `CANCELLED` at once; no worker will claim it |
| running in this process (`run`, `stream`, a worker of this process) | its task is cancelled now: `CANCELLED`; `run` returns a `CANCELLED` result, `stream` ends with `RUN_FINISHED` (outcome `cancelled`, `data.reason`) |
| running on a worker elsewhere | agent-runs asks that worker to stop: its next heartbeat (within 20 s) says `cancel_requested`, the worker cancels the attempt, and the run ends `CANCELLED`; a worker that never answers loses the run when its lease runs out |
| already ended | `ConflictError` |

**Automatic.** agent-runs keeps the reason and who asked; the pipeline ends the attempt as a
cancellation — the event, the transcript so far, no outcome feedback, no judges. A cancelled
run is never retried. Who may cancel: whoever may answer the run (agent-runs' rule). The run's
[sub-agents](subagents.md)' runs that have not ended are cancelled with it (one paused for a
person at once), and theirs.

**On failure.** A run kept in *another* process's `run`/`stream` (no worker, so no heartbeat)
is cancelled in agent-runs at once, but that process learns only at its next write, which is
refused (`ConflictError`): cancel such runs where they run, or `start` them for a worker.

## Releasing on shutdown

**What.** A worker that stops (`SIGTERM`, `worker.stop()`) lets its runs finish for 25 s, then
hands the rest back to agent-runs at once (`runs.release`): they are queued for another worker
as their next attempt without waiting for the lease to lapse, and without counting as a crash.

**Automatic.** Nothing to do; the journal saved so far goes with them, so the next worker
repeats nothing. A run whose cancel was asked for ends `CANCELLED` instead. If even the release
cannot be sent, the lease lapses as before.

## The last good tool list

**What.** When the gateway (or a local source: an OpenAPI document's URL) cannot be listed, the
toolbox keeps its last listing — per agent and tenant — and lists again after 30 s
(`TOOLS_RETRY_SECONDS`), like the key and the memory tools. Runs go on with the tools they
had; a call to a server that is really down fails as that call's error.

**On failure.** Only a toolbox never listed raises (the run fails saying why), and two tools
with one name still do (`ConfigurationError`).

## Agent version

**What.** Which version of the agent's code started a run.

**When.** Always worth setting in production: a run paused on Monday is resumed by Tuesday's
deploy.

**Where.** `h.wrap(target, id=..., version="2026.10.5")`, or `TRELLIS_AGENT_VERSION` for every
agent of the deployment (`wrap` wins); Way 2: `RunStart.agent_version`.

**Automatic.** Sent as `RunStart.agent_version` (only when set), on every attempt's
`invoke_agent` span (`gen_ai.agent.version`, `langfuse.version`). A run resumed — or claimed
after a crash — by another version goes on, with a `warning` event (`agent_version`) and a log
line naming both versions.

## Every timeout, and who sets it

| Timeout | Default | Set by |
|---|---|---|
| a tool call: `@tool(timeout=)`, `governed(timeout=)` | none | the tool's author |
| an OpenAPI operation, its document: `openapi(timeout=)` | 120 s (`REMOTE_TIMEOUT_SECONDS`) | the author |
| an A2A exchange: `a2a(timeout=)`; `remote(timeout=)` (per request) | 120 s (the same) | the author |
| an MCP call | the run's remaining time, else the Bifrost SDK's 60 s | — |
| a model call: `ReAct(model_timeout=)` | the Bifrost SDK's 60 s per attempt | the author |
| a sub-agent's run (`agent.as_tool()`) | what is left of its parent's time, and its parent's deadline | — |
| an agent's runs' working time: `h.wrap(timeout=)` | none | the agent's author (every entry, scheduled runs too) |
| a run's working time: `timeout=` | the agent's | the caller of `run`/`stream`/`start` |
| a run's end: `deadline=` | none | the caller |
| the platform's longest run | none | operations: `RUNS__RUNS__MAX_RUN_SECONDS` in agent-runs |
| an answer: `ask(deadline=, escalate_to=)` | none | the agent's code ([interrupts.md](interrupts.md)) |
| a lease / its heartbeat | 60 s / every 20 s | constants (`trellis.runs`) |
| a stopping worker's grace | 25 s | constant (`GRACE_SECONDS`) |
| the background writes' drain | 10 s | constant (`DRAIN_SECONDS`) |
| one memory / runs SDK request | 10 s | the SDKs |
| an A2A push delivery | 10 s | constant |
| waiting for a queued run: `RunHandle.result(timeout=)` | none | the caller (it only stops waiting) |

## Limitations

* **Sync functions** run in a thread that cannot be stopped ([above](#tool-timeouts)).
* **A framework's own tools** (`function_tool`, Deep Agents' file tools, Claude's `Bash`) do not
  pass the bridge: no timeouts, retries, keys or unknown outcomes from the harness — give them
  their framework's own.
* **MCP servers** do not receive the idempotency key (the protocol has no field for it).
* **Cancelling an in-process run from another process** ([cancel](#cancel)).
* **Without `RUNS_URL`**, `LocalRuns` keeps the working time, leases, cancels and releases as
  agent-runs does, but nothing survives a restart and nothing enforces a limit for a run no
  attempt is working on (the attempts themselves still stop on time).

## Run it

* [`examples/reliability.py`](../examples/reliability.py) — a read retried, a write past its
  timeout of unknown effect, a run past its time limit, a cancel.
* Tests: `tests/integration/test_reliability.py` (every adapter), `test_run_limits.py`,
  `test_progress.py` (crashes), and against the real services `tests/live/test_live_reliability.py`.
