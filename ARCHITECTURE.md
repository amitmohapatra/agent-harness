# Architecture

The harness is an attach layer. It owns no control flow: a framework runs the agent, and the
harness sits around one run of it — identity, the run record, memory in and out, the tools the
agent may call and who must approve them, the pause, the recording, the trace.

## System context

What a process that imports `trellis` talks to. Every arrow out of the harness is one client
module (`clients/bifrost.py`, `clients/memory.py`, `clients/runs.py`) or one surface
(`surfaces/agui`, `surfaces/a2a`); the OTLP exporter and the Langfuse scores API are
`telemetry.py`.

```mermaid
flowchart LR
  subgraph app["Your service (one process)"]
    target["Agent you built<br/>LangGraph · OpenAI Agents<br/>Claude Agent SDK · ReAct · function"]
    harness["trellis-harness<br/>Harness · Agent · pipeline"]
    target <--> harness
  end
  ui["Chat UI<br/>(AG-UI client)"] -- "AG-UI over SSE" --> harness
  harness <-- "A2A: serve_a2a · a2a(url)" --> peer["Remote A2A agents"]
  harness -- "MCP tools · ReAct chat" --> bifrost["Bifrost gateway"]
  target -. "the team's model client" .-> bifrost
  bifrost --> mcp["MCP servers"]
  bifrost --> models["Model providers"]
  harness -- "context · records · feedback" --> memory["Memory service"]
  harness -- "runs · queue · inbox" --> runs["agent-runs"]
  worker["Workers<br/>python -m trellis.worker"] -- "claim · heartbeat" --> runs
  harness -- "traces · scores" --> otel["Langfuse or an<br/>OTel collector"]
```

| Neighbour | What the harness uses it for | Endpoints (module) |
|---|---|---|
| Bifrost gateway (`BIFROST_URL`, `BIFROST_VIRTUAL_KEY`) | the MCP tools the virtual key allows, their execution, Code Mode, `ReAct`'s model calls, the MCP log of Code Mode scripts | `POST /mcp` (`tools/list`), `POST /v1/mcp/tool/execute`, `POST /v1/chat/completions`, `GET /api/mcp-logs` (`clients/bifrost.py`, through `bifrost-sdk`) |
| Memory service (`MEMORY_URL`, `TRELLIS_API_KEY`) | who the key is, the pushed context, the pull tools, transcripts and tool records, the tool catalog, outcomes and feedback, the grounding check, documents, the agent's model key | `/v1/keys/self`, `/v1/context`, `/v1/agent-tools`, `/v1/messages`, `/v1/tools/invocations`, `/v1/tools`, `/v1/tools/catalog`, `/v1/feedback`, `/v1/verify`, `/v1/documents`, `/v1/agents/model-key` (`clients/memory.py`, through `trellis-memory`) |
| agent-runs (`RUNS_URL`, `TRELLIS_API_KEY`) | run records, the worker queue and leases, pauses with their checkpoint, the inbox, schedules, `ask` artifacts | `/v1/runs`, `/v1/runs/claim`, `/v1/runs/{id}/heartbeat`, `/pause`, `/resume`, `/finish`, `/artifacts`, `/v1/artifacts/{id}`, `/v1/schedules` (`clients/runs.py`) |
| Chat UI | runs and their events, resumes, reconnects, large interrupt payloads | `serve_chat`: `POST {path}/run`, `GET {path}/runs/{id}/events`, `GET {path}/runs/{id}/artifacts/{artifact_id}` (`surfaces/agui`) |
| Remote A2A agents | callers of this agent, and agents this agent calls | `serve_a2a`: the card and JSON-RPC at `url`; `a2a(url)`: `SendStreamingMessage`, `CancelTask` (`surfaces/a2a`) |
| Langfuse / an OTel collector (`OTEL_EXPORTER_OTLP_*`) | traces; grounding and feedback scores | OTLP/HTTP `<endpoint>/v1/traces`, `POST /api/public/scores` (`telemetry.py`) |


Unset variables remove a box: no `BIFROST_URL` means no MCP tools and no `ReAct` model names,
no `MEMORY_URL` means no memory, no `RUNS_URL` keeps runs, the queue and schedules in this
process (`LocalRuns`), no `OTEL_EXPORTER_OTLP_ENDPOINT` means no export
([docs/configuration.md](docs/configuration.md)).

## Modules

```
src/trellis/
  __init__.py          the public API (lazy; extends __path__ for trellis.contracts / .memory)
  worker.py            python -m trellis.worker module:harness
  harness/
    harness.py         Harness: settings → clients, writes, scores; the key (tenant, kept fresh);
                       wrap / tools / worker / inbox / feedback / add_document
    agent.py           Agent: run, stream, start, resume, schedule, serve_*; RunHandle
    pipeline.py        one attempt of one run (the fixed pipeline below)
    runtime.py         Runtime (trellis.current()), ask, the pause exception, interrupt ids
    journal.py         what a re-run needs: answers and tool outputs, keyed by content
    events.py          a run's RunEvent stream (built only when someone listens)
    writes.py          background writes: retries, backpressure, the spool, auto-drain
    fresh.py           a value read from a service, kept for a TTL, the last one through outages
    identity.py        tenant / user / thread / agent / run → memory scope, contracts context
    result.py          Result
    settings.py        the environment
    telemetry.py       OTel GenAI spans, trace ids per run, counters; OTLP export; Langfuse scores
    redaction.py       what may leave the process
    worker.py          Worker: claim, lease, heartbeat
    adapters/          detect(target) and one adapter per framework (base, langgraph,
                       openai_agents, claude, react, function)
    tools/             base (Tool), sources (tool, a2a, openapi), toolbox (MCP tools, catalog
                       tiers and approve_when, Code Mode, publishing), policy (tiers,
                       conditions), bridge (every call), convert/ (one module per native format)
    clients/           bifrost, memory, runs — the only modules that call those services
    surfaces/          agui (serve_chat), a2a (serve_a2a, the a2a() client)
```

Each service has exactly one client module; nothing else in the harness calls it. The core
imports no framework: an adapter imports its framework the first time a target of its type is
wrapped, and `tests/contract` checks that `import trellis` and `Harness()` load none — and that
what the clients send, and what the test doubles of the memory service and agent-runs answer,
match those services' committed OpenAPI documents.

### Components

How the modules depend on each other (an arrow reads "uses"). The adapters and the tool
converters are the only modules that import a framework; the surfaces are the only ones that
import FastAPI or the A2A SDK.

```mermaid
flowchart TB
  api["trellis (public API, lazy)"] --> harness["harness.Harness"]
  cli["trellis.worker (CLI)"] --> harness
  harness --> agent["agent.Agent · RunHandle"]
  harness --> workerm["worker.Worker"]
  harness --> writes["writes.Writes"]
  harness --> telemetry["telemetry<br/>(spans, counters, Scores)"]
  workerm --> agent
  agent --> pipeline["pipeline.attempt"]
  agent --> surfaces
  subgraph surfaces["surfaces"]
    agui["agui: mount · Hub · translate · sse"]
    a2a["a2a: mount · RunExecutor · RunTaskStore<br/>PushNotifier · HeaderIdentity · client"]
  end
  agui --> pipeline
  a2a --> pipeline
  pipeline --> runtime["runtime.Runtime · ask"]
  pipeline --> journal["journal.Journal · Replay"]
  pipeline --> events["events.RunEvents"]
  pipeline --> adapters
  subgraph adapters["adapters (detect)"]
    langgraph["LangGraphAdapter"]
    openai["OpenAIAgentsAdapter"]
    claude["ClaudeAdapter"]
    react["ReActAdapter"]
    function["FunctionAdapter"]
  end
  adapters --> convert["tools.convert<br/>(langchain · openai_agents · claude · openai_chat)"]
  convert --> bridge["tools.bridge.call"]
  runtime --> bridge
  bridge --> policy["tools.policy.tier"]
  bridge --> journal
  agent --> toolbox["tools.toolbox.resolve"]
  toolbox --> sources["tools.sources<br/>(tool · a2a · openapi)"]
  sources --> a2a
  subgraph clients["clients (one per service)"]
    bifrost["bifrost.Gateway"]
    memory["memory.Memory · RunMemory"]
    runs["runs.HttpRuns · LocalRuns"]
  end
  toolbox --> bifrost
  toolbox --> memory
  agent --> memory
  agent --> runs
  pipeline --> runs
  harness --> clients
  telemetry --> redaction["redaction.Redactor"]
  pipeline --> telemetry
  bridge --> telemetry
```

## The pipeline

Every run of every framework goes through `pipeline.attempt`:

```mermaid
flowchart LR
  A[identity] --> B[run record<br/>agent-runs or in process]
  B --> C[toolbox<br/>local + MCP + memory pull]
  C --> D[memory push<br/>/v1/context + tools]
  D --> E[adapter<br/>prepare · invoke/stream · extract]
  E -->|paused| F[record PAUSED<br/>interrupt + journal]
  E -->|ended| G[record SUCCESS / ERROR]
  G --> H[background: transcript,<br/>system outcome, sampled grounding]
```

A memory service that is down degrades a run, it never fails it: a failed context read is a
`warning` event; the memory tools that cannot be listed are left out (a `warning` event); a
catalog that cannot be read makes every tool that does more than read ask; who
`TRELLIS_API_KEY` is stays what it was last read (`fresh.Fresh`). Only a key the service refuses,
or one it could never be asked about, is a `ConfigurationError`. Writes to agent-runs are
awaited (a pause that was not recorded cannot be resumed) and retried; writes to the memory
service are queued.

### One run, end to end

A wrapped agent answers with memory recalled, calls an MCP tool, writes a memory through the
`memory_remember` pull tool, and a person's feedback arrives later. Calls in the `Writes` lane
are background writes: the run does not wait for them.

```mermaid
sequenceDiagram
  autonumber
  actor User as Application / user
  participant Agent as Harness h · Agent (h.wrap)
  participant Runs as Runs (agent-runs or LocalRuns)
  participant P as pipeline.attempt
  participant Mem as Memory service
  participant FW as Adapter + framework
  participant Br as tools.bridge
  participant GW as Bifrost
  participant W as Writes (background)
  participant LF as Langfuse (scores API / OTLP)

  User->>Agent: await agent.run(input, user=, thread=)
  Agent->>Mem: GET /v1/keys/self (tenant, kept 10 min, the last answer while memory is down)
  Agent->>Runs: started(RunStart) → RUNNING
  Agent->>P: attempt(agent, identity, input)
  P->>GW: MCP tools/list with the virtual key (definitions, kept 300 s)
  P->>Mem: GET /v1/tools?names= + If-None-Match (tiers, approve_when, every 30 s)
  P->>Mem: GET /v1/agent-tools (pull tools, kept 10 min)
  P->>Mem: POST /v1/context (memory recall: retrieve memory span)
  Mem-->>P: rendered, bundle_id, tools [name, confidence]
  P->>FW: prepare_input(input, context), invoke(native tools)
  FW->>GW: chat completion (the team's model through Bifrost)
  FW->>Br: call erp-get_stock(sku)
  Br->>Br: replay? tier: read → runs (write → tool_notice, irreversible → ask)
  Br->>GW: POST /v1/mcp/tool/execute (execute_tool span)
  GW-->>Br: result
  Br-)W: memory.record_tool
  Br-->>FW: result text
  FW->>Br: call memory_remember(content)
  Br->>Mem: POST /v1/agent-tools/memory_remember (memory write, in the run's scope)
  Br-->>FW: stored
  FW-->>P: output → extract(answer, transcript)
  P->>Runs: finished(SUCCESS, output)
  P-->>User: Result(SUCCESS, answer)
  W-)Mem: POST /v1/tools/invocations (the MCP call)
  W-)Mem: POST /v1/messages (transcript, one batch per attempt)
  W-)Mem: POST /v1/feedback (system: confirm)
  W-)Mem: POST /v1/verify (sampled 10 %) → grounding score
  W-)LF: score grounding on the run's trace
  User->>Agent: await h.feedback(run_id, "correct", correction)
  Agent->>Runs: get(run_id)
  Agent->>Mem: POST /v1/feedback (human, review pending)
  Agent->>LF: score feedback (POST /api/public/scores and a score span)
```

The memory tools' own calls are not recorded again (the service logs them); everything else
the bridge runs is.

## The adapter contract

Four functions per framework, nothing else (`adapters/base.py`):

* `prepare_input(target, input, context)` — the framework's input, the memory context as a
  system message (or appended to the system prompt);
* `invoke(target, native_input, run)` / `stream(...)` — run it; the stream yields text deltas
  and finally `Output(value)`;
* `extract(target, output)` — the answer, the assistant transcript, and a pause the framework
  reported itself (LangGraph's `interrupt`, an OpenAI Agents `needs_approval`);
* `resume_input(target, native_input, pending, resolution)` — what continues a pause:
  `Command(resume=...)` for a checkpointed graph, the SDK's `RunState` for its approvals,
  otherwise the original input (a re-run).

Per-run harness tools reach the adapter already converted (`tools/convert/<format>.py`). An
adapter with fixed tools (a compiled graph) refuses `tools=` at wrap time; its tools come from
`h.tools(...)` when the graph is built.

## Tools

The toolbox (`tools/toolbox.py`, one `Toolbox` per agent and tenant) keeps two things fresh on
two clocks: the definitions — the local sources and every MCP tool the Bifrost virtual key
allows — listed again after `TOOLS_TTL_SECONDS` (300), and the governance — the catalog's word on
each tool (`risk`, `approve_when`) — read again after `GOVERNANCE_TTL_SECONDS` (30) with the last
answer's `ETag` (`If-None-Match`; a `304` keeps what was read), so an administrator's new rule
reaches running agents within half a minute. One refresh at a time: concurrent runs that find
the toolbox stale share one read. Code Mode is chosen for the read-only Code Mode servers when
there are enough of them, and every tool is published to the catalog in the background (and
published again at the next listing if that failed). A catalog that cannot be read leaves each
tool its own tier, except that every tool that does more than read asks for approval
(`policy.CATALOG_UNREAD`) until it can (governance read in the last 300 s still stands); the
warning is logged once. Every call, whoever makes it, goes through `tools/bridge.call`:

1. **replay** — the journal already has this call (same tool, same arguments, n-th time): its
   recorded output is returned and nothing runs;
2. **policy** — the tier from the tool's side effects (annotations → declaration → the
   catalog's `risk`): `read` runs, `write` runs and is announced (`tool_notice` event),
   `irreversible` asks for approval. The catalog's `approve_when` replaces the tier: it asks
   exactly when the expression holds, evaluated by `trellis.memory.approval` — the memory
   service's own implementation, which also writes and validates the rules (a rule that cannot
   be read or evaluated asks);
3. **execution** — in an `execute_tool` span, between `TOOL_CALL_*` events; a failure is an
   error result the model reads, a pause propagates;
4. **record** — journaled (the tool is then offered for the rest of the run), counted, and
   with memory writes on sent to the memory service's tool records in the background.

A tool called outside a harness run is refused. What the model is *offered* (the tools the
context names, the memory tools, the tools already used) is `Runtime.offers`; each adapter
narrows as far as its framework allows (`Adapter.narrows`: per turn, per run, or none).
Agent Mode is never used.

## Pauses and resumes

`Runtime.ask` is the one pause. Its interrupt (a contracts `Interrupt`) has the id
`<run_id>.<attempt>.<n>`: it names its run, so `resume` needs nothing else. How a run
continues:

* **LangGraph with a checkpointer**: `ask` *is* `langgraph.types.interrupt`; the resume is
  `Command(resume={<LangGraph interrupt id>: resolution})` and the graph continues where it stopped.
* **Everything else**: `ask` raises and the attempt ends (a framework that swallows the
  exception is still paused: the runtime records the pause first). The resume runs the agent
  again from its input, as the next attempt, with the **journal**: questions already answered
  return their answers where they are asked, and tool calls already made return their
  recorded outputs (keyed by content, consumed in order — a re-planned call nobody approved is
  asked about again, never matched to another approval).

The journal is the run's checkpoint: `runs.paused(interrupt, checkpoint=journal)` stores it
with the pause, agent-runs returns it as `RunRecord.checkpoint` on every read and claim (and
clears it when the run ends), and the attempt that resumes the run — in this process or in a
worker elsewhere — files `last_resolution` under the pending question and replays the rest.

A run started in process (`run`/`stream`) continues in the process that resumes it; a run that
came from the queue (`start`, a schedule) goes back to it and a worker continues it. Approve,
reject and edit decisions on tool calls are also feedback records.

### A pause through agent-runs

A queued run asks for approval of an `irreversible` tool; a person answers from the inbox; a
worker (any worker) continues it from the checkpoint.

```mermaid
sequenceDiagram
  autonumber
  actor App as Application
  participant Agent as Harness h · Agent
  participant AR as agent-runs
  participant Wk as Worker (h.worker / python -m trellis.worker)
  participant P as pipeline.attempt + bridge
  participant Mem as Memory service
  actor CFO as Approver

  App->>Agent: handle = await agent.start(input, user=)
  Agent->>AR: POST /v1/runs (queue: true) → QUEUED
  Wk->>AR: POST /v1/runs/claim (lease 60 s)
  AR-->>Wk: {run, lease} → RUNNING, attempt 1
  Wk->>P: agent._claimed(record, worker_id)
  loop every 20 s while it runs
    Wk->>AR: POST /v1/runs/{id}/heartbeat (409 → LeaseLost: stop, write nothing)
  end
  P->>P: refund is irreversible → Runtime.approve → Paused
  P->>AR: POST /v1/runs/{id}/pause?worker_id= (Interrupt + checkpoint = journal)
  AR-->>P: PAUSED
  CFO->>Agent: await h.inbox("role:finance")
  Agent->>AR: GET /v1/runs?status=PAUSED&assignee=role:finance
  CFO->>Agent: await agent.resume(interrupt_id, "approve", reviewer="cfo")
  Agent->>AR: GET /v1/runs/{id} (the interrupt it waits on)
  Agent->>AR: POST /v1/runs/{id}/resume (InterruptResolution)
  AR-->>Agent: QUEUED, attempt 2 (it came from the queue)
  Agent-)Mem: POST /v1/feedback (TOOL_CALL approve, background)
  Agent-->>CFO: Result(QUEUED)
  Wk->>AR: POST /v1/runs/claim
  AR-->>Wk: {run: checkpoint + last_resolution}
  Wk->>P: attempt 2: the journal answers the approval, earlier calls replay
  P->>P: refund runs (once)
  P->>AR: POST /v1/runs/{id}/finish?worker_id= (SUCCESS, output)
  App->>Agent: await handle.result(timeout=)
  Agent->>AR: GET /v1/runs/{id} (polled every 0.5 s) → SUCCESS
```

Without `RUNS_URL` the same sequence runs against `LocalRuns` in the process, and a run
started with `run`/`stream` resumes in the process that calls `resume` (no queue, no worker).

## Calling a remote agent over A2A

`a2a(url)` makes a remote agent one tool. The remote side is another harness's
`serve_a2a(app, url)` (or any A2A server); the task id there is the remote run id.

```mermaid
sequenceDiagram
  autonumber
  participant P as Calling run (bridge)
  participant C as a2a(url) tool (surfaces.a2a.client)
  participant S as Remote serve_a2a (DefaultRequestHandler)
  participant X as RunExecutor
  participant RP as Remote pipeline.attempt

  Note over C: resolve once per toolbox: GET {url}/.well-known/agent-card.json
  P->>C: call greeter(message)
  C->>S: SendStreamingMessage (A2A-Extensions: trusted-identity,<br/>x-trellis-identity: tenant + user, context id = the calling thread)
  S->>X: execute(context, event queue)
  X->>X: HeaderIdentity: the header's tenant must be the key's
  X->>RP: Agent._opened(run_id = task id) and attempt(...)
  RP-->>X: RunEvents
  X-->>C: task SUBMITTED, WORKING (text, progress), artifact "result", COMPLETED
  C-->>P: the result artifact (or the text)
  alt the remote run asks something
    X-->>C: INPUT_REQUIRED + the question
    C->>P: runtime.ask(question): the calling run pauses
    C->>S: CancelTask (the remote task is not left waiting)
    Note over P: on resume the call is made again and ask returns the answer,<br/>which is sent on the new remote task as the next message
  else the remote run fails
    X-->>C: FAILED
    C-->>P: ToolError: the calling model reads "greeter failed: ..."
  end
```

## Run states

The run record's status (contracts `RunStatus`) and who moves it. The harness writes `QUEUED`,
`RUNNING`, `PAUSED`, `SUCCESS`, `ERROR` and `CANCELLED`. agent-runs' ticker moves the rest: a
lapsed lease back to `QUEUED` (or to `ERROR` once every attempt lapsed), and a paused run past
its interrupt's deadline to `escalate_to` (once) or to `TIMEOUT`. `PARTIAL` and `REJECTED` are
valid endings of the contract that the harness never writes.

```mermaid
stateDiagram-v2
  [*] --> QUEUED: agent.start, a schedule fires
  [*] --> RUNNING: agent.run, stream, serve_chat, serve_a2a
  QUEUED --> RUNNING: a worker claims it (lease)
  QUEUED --> CANCELLED: finished CANCELLED in agent-runs before a claim
  RUNNING --> PAUSED: ask or an approval (interrupt + journal)
  RUNNING --> QUEUED: lease lapsed (next attempt)
  RUNNING --> SUCCESS: answered
  RUNNING --> ERROR: the agent failed, or the lease lapsed on every attempt
  RUNNING --> CANCELLED: stream closed, lease lost, a cancel
  PAUSED --> RUNNING: resume (started in process)
  PAUSED --> QUEUED: resume (came from the queue)
  PAUSED --> CANCELLED: resume with cancel, A2A cancel
  PAUSED --> PAUSED: deadline passed, escalated to escalate_to
  PAUSED --> TIMEOUT: deadline passed, nobody to escalate to
  SUCCESS --> [*]
  ERROR --> [*]
  CANCELLED --> [*]
  TIMEOUT --> [*]
```

Every attempt (each `RUNNING` stretch) is one `invoke_agent` span in the run's one trace.

## Background writes

`writes.Writes`: a bounded queue (`MAX_PENDING` 10 000) drained by `WRITERS` (4) tasks. A
write whose failure may pass is tried again (`WRITE_ATTEMPTS` 3, full-jitter backoff); a full
queue makes the writer wait (`SUBMIT_WAIT_SECONDS` 5) instead of dropping the write. A write
given up is logged, counted and emitted as a `warning` event to the run's listeners — or, with
`TRELLIS_SPOOL_DIR`, appended to a JSONL spool that the next process replays. When the event loop
shuts down it cancels the workers, and a cancelled worker finishes the queue first (the write it
was cut off in included; writes are idempotent), within `DRAIN_SECONDS` (10); what is left is
spooled or counted lost (`trellis.writes.undelivered`). `await h.aclose()` drains explicitly.
The guarantee is in [docs/memory.md](docs/memory.md#background-writes-what-is-guaranteed).

## Stopping a worker

`python -m trellis.worker` turns `SIGTERM`/`SIGINT` into `Worker.stop()`:

```mermaid
sequenceDiagram
  autonumber
  participant OS as Orchestrator
  participant CLI as python -m trellis.worker
  participant Wk as Worker
  participant R as Runs it holds
  participant AR as agent-runs
  participant W as Writes
  OS->>CLI: SIGTERM
  CLI->>Wk: stop()
  Wk--xAR: no more claims
  par within GRACE_SECONDS (25 s)
    R->>AR: finish / pause (as usual)
  end
  alt a run is still going
    Wk->>R: cancel(RELEASED): nothing written
    Note over AR: its lease lapses → QUEUED, next attempt
  end
  CLI->>W: aclose(): drain ≤ DRAIN_SECONDS, then spool or count the rest
  CLI-->>OS: exit 0
```

## Telemetry

The OTel API only: an `invoke_agent` span per attempt in a trace whose id derives from the run
id (every attempt, score and piece of feedback of a run in one trace), `execute_tool`,
`chat` (the `ReAct` model calls) and `retrieve memory` spans with GenAI attributes and
Langfuse's trace attributes, `score` spans; counters `trellis.runs`, `trellis.tool_calls`,
`trellis.writes.failed`. Attributes pass the redactor and are built only for a recording span.
`OTEL_EXPORTER_OTLP_ENDPOINT` installs an SDK provider with one OTLP exporter unless the
application installed one. See [docs/observability.md](docs/observability.md).
