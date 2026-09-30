# Architecture

The harness is an attach layer. It owns no control flow: a framework runs the agent, and the
harness sits around one run of it — identity, the run record, memory in and out, the tools the
agent may call and who must approve them, the pause, the recording, the trace.

## Modules

```
src/trellis/
  __init__.py          the public API (lazy; extends __path__ for trellis.contracts / .memory)
  worker.py            python -m trellis.worker module:harness
  harness/
    harness.py         Harness: settings → clients, writes, scores; the key (tenant, role);
                       wrap / tools / worker / inbox / feedback
    agent.py           Agent: run, stream, start, resume, schedule, serve_*; RunHandle
    pipeline.py        one attempt of one run (the fixed pipeline below)
    runtime.py         Runtime (trellis.current()), ask, the pause exception, interrupt ids
    journal.py         what a re-run needs: answers and tool outputs, keyed by content
    events.py          a run's RunEvent stream (built only when someone listens)
    writes.py          background writes with auto-drain
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
wrapped, and `tests/contract` checks that `import trellis` and `Harness()` load none.

## The pipeline

Every run of every framework goes through `pipeline.attempt`:

```mermaid
flowchart LR
  A[identity] --> B[run record<br/>agent-runs or in process]
  B --> C[toolbox<br/>local + MCP + memory pull]
  C --> D[memory push<br/>/v1/context + tool candidates]
  D --> E[adapter<br/>prepare · invoke/stream · extract]
  E -->|paused| F[record PAUSED<br/>interrupt + journal]
  E -->|ended| G[record SUCCESS / ERROR]
  G --> H[background: transcript,<br/>system outcome, sampled grounding]
```

A failed memory read is a `warning` event, not a failed run. Writes to agent-runs are awaited
(a pause that was not recorded cannot be resumed); writes to the memory service are queued.

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

The toolbox (`tools/toolbox.py`) is resolved once per agent and tenant and again after
`TOOLS_TTL_SECONDS` (300): the local sources, every MCP tool the Bifrost virtual key allows,
the catalog's word on each (`side_effects`, `approve_when`), Code Mode for the read-only Code
Mode servers when there are enough of them, and every tool published to the catalog in the
background. Every call, whoever makes it, goes through `tools/bridge.call`:

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

A tool called outside a harness run is refused. What the model is *offered* (the tool hints'
candidates, the memory tools, the tools already used) is `Runtime.offers`; each adapter
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

## Background writes

`writes.Writes`: a bounded queue (`MAX_PENDING` 10 000) drained by `WRITERS` (4) tasks. A
failed write is logged, counted (`trellis.writes.failed`) and emitted as a `warning` event to
the run's listeners. When the event loop shuts down it cancels the workers, and a cancelled
worker finishes the queue first (the write it was cut off in included; writes are idempotent),
within `DRAIN_SECONDS` (10). `await h.aclose()` drains explicitly.

## Telemetry

The OTel API only: an `invoke_agent` span per attempt in a trace whose id derives from the run
id (every attempt, score and piece of feedback of a run in one trace), `execute_tool`,
`chat` (the `ReAct` model calls) and `retrieve memory` spans with GenAI attributes and
Langfuse's trace attributes, `score` spans; counters `trellis.runs`, `trellis.tool_calls`,
`trellis.writes.failed`. Attributes pass the redactor and are built only for a recording span.
`OTEL_EXPORTER_OTLP_ENDPOINT` installs an SDK provider with one OTLP exporter unless the
application installed one. See [docs/observability.md](docs/observability.md).
