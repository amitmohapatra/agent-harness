# Memory

Memory is on when `MEMORY_URL` is set (or `Harness(memory=MemoryClient(...))`), for every
agent; `without={"memory"}` — or one part: `memory_push`, `memory_pull`, `records` — turns it
off for an agent (`h.wrap`) or a run (`agent.run`)
([what is on, and how to turn it off](configuration.md#what-is-on-and-how-to-turn-it-off)). Every
call the harness makes to the memory service is in `trellis/harness/clients/memory.py`, in the
run's scope (tenant, user, agent, run, thread). With memory on, every run reads (push, pull)
and records (transcript, tool calls, outcome); with it off there is no context, no memory
tools, no records and no catalog, and `trellis.current().memory`, `tools.hints(...)` and
`h.add_document(...)` raise `ConfigurationError`.

This page is what a wrapped agent gets (Way 1). Code on its own framework makes the same calls
itself with the memory SDK: [blocks/memory.md](blocks/memory.md) (Way 2).

## Push

Before the agent runs, `/v1/context` for the run's question — its token budget 5 % of the
model's context window when the target says it (a `context_window` or `max_input_tokens`
attribute on the target or its `model`, or a LangChain model's `profile`), between 2000 and
8000, else 2000 — answered in
the prompt format — `{bundle_id, rendered, token_estimate, evidence_status, tools?}`, items cited
by short per-bundle handles (`[m1]`, `[d2]`). The rendered text reaches the framework as a system
message (see the README's matrix, and each [framework page](README.md#which-target)) and is `trellis.current().context`; the `bundle_id` is what
the grounding check verifies the answer against.

* `window=false` when the framework keeps the thread's messages itself (a LangGraph graph with
  a checkpointer): the service leaves the recent conversation out. Such a graph's context
  message has a fixed id (`trellis-memory-context`), so its thread holds one that each turn
  replaces.
* With 5 or more tools of the run's own, the request carries their names: the context then
  includes the procedures learned for the task and the tools section (next step, argument
  values found in memory, what is missing), and its `tools` (`[{name, confidence}]`,
  confidence 0–1) narrow the tools the model is offered ([tools.md](tools.md)). One call: the
  harness asks for no hints of its own.

A failed context call is a `warning` event (`memory_unavailable`); the run continues without
it. A `CONTEXT_LOADED` event reports its size; the call is a `retrieve memory` span.

## Pull

The service's agent tools (listed again every 10 minutes; while the service cannot be reached
the last listing stands, and a run when they were never listed goes without them, with a
`warning` event) are added to the run's tools:

| Tool | |
|---|---|
| `memory_search(query, kinds?, threads?, time_from?, time_to?, k?)` | memories, documents and — `kinds: ["message"]` — this conversation's messages; with `threads: "all"`, the messages of every earlier conversation of the same user (never another user's), each with its `thread_id` ([past conversations](https://github.com/amitmohapatra/agent-memory-service/blob/main/docs/guide/04-retrieval.md#past-conversations)); `kinds: ["episode"]` finds which earlier conversation, one summary each |
| `memory_remember(...)` | store a memory verbatim |
| `memory_update(id, content)` | replace a memory (an `[m1]` handle from the context works) |
| `memory_forget(id)` | forget one |
| `profile_edit(block, old?, new)` | edit a pinned profile block |
| `tool_search(task)` | the next step, plan, prefill and missing arguments among the run's own tools — the harness passes the run's toolbox with the call, and offers the tools it returns (`{tools: [{name, confidence, next?, args?, missing?}], plan?}`) to the model |

`memory_search` and `tool_search` only read (they run unannounced); the others write. For a compiled graph they are built in
with `h.tools(...)`. Inside a tool or a node, `trellis.current().memory` is the memory SDK's
context bound to the run, with all its verbs.

## What is recorded

In the background, after each attempt — whether it succeeded, paused or failed:

* the transcript, in one batch (`history.add`) — each message named by the harness
  (`source_system: trellis-harness`, `source_message_id`: the run for the question, which
  every attempt asks, the attempt and position for what the agent said), so the service stores
  a message it has already seen once. A run started with no thread — a scheduled run — is its
  own thread (`thread = run_id`);
* every harness tool call (not the memory tools, which the service logs itself), and Code
  Mode's nested calls from Bifrost's log — their arguments and outputs redacted
  ([observability.md](observability.md#redaction));
* approve/reject/edit decisions as `TOOL_CALL` feedback.

## Background writes: what is guaranteed

Every write above is queued and the run moves on (`writes.py`); four writers drain the queue.

* A write that fails is tried again — 3 attempts in all, full-jitter backoff from 0.5 s — when
  the failure may pass (the service unavailable, a timeout, a dropped connection); a refusal the
  same request would get again (a `4xx` the SDK marks not retryable) is not repeated.
* A full queue (10 000 writes) holds the run that writes, for up to 5 s, rather than dropping
  the write after the run was told it would happen.
* A write still undelivered — its attempts spent, a queue that stayed full, the process
  stopping with it queued (shutdown drains for at most 10 s) — is logged and counted
  (`trellis.writes.undelivered`). With `TRELLIS_SPOOL_DIR` set, a write that is data (a
  transcript, a tool record, the run's outcome, a decision, catalog entries) is appended to
  `<dir>/trellis-writes.jsonl` and replayed — the file claimed by renaming it, then removed —
  the next time a harness with the same directory starts writing (a worker at its start, any
  other process at its first write). The service stores a replayed write that had in fact
  landed once (every write carries its idempotency key or message id). Without the directory,
  or for a write that is not data (the sampled grounding check, the model key, Code Mode
  calls read back from the gateway), it is lost: counted (`trellis.writes.failed`) and, during a
  run, a `warning` event.

So: delivered at least once while the process lives or, with a spool directory on a volume that
outlives the process, after its next start; never silently dropped.

## Documents

`h.add_document(file, user=..., thread=None, title=None, visibility=None, wait=60)` uploads a
file to the memory service (`POST /v1/documents`) in that user's scope (or one thread's) and
waits until it is parsed and indexed (`GET /v1/documents/{id}`). Nothing else is needed: the
push context of the user's next run retrieves and cites its passages like any other
document. `visibility` widens who may retrieve it (`WORKSPACE`, `TENANT`); inside a run the
SDK's `agent.memory.advanced.documents` does the same in the run's scope.

## Outcome and grounding

The run's outcome is a projection in the memory service of the feedback on the run, by
precedence **human > judge > system**:

* **system** — the harness, when the run ends: `SUCCESS` → `confirm`, `ERROR` → `reject` (a
  cancelled run says nothing about the agent). Never "no exception = success" on its own: the
  judge and people outrank it.
* **judge** — on a sampled share of successful runs with an answer (a structured one as its JSON)
  (`TRELLIS_GROUNDING_SAMPLE`, default 10 %, chosen by the run id), `/v1/verify {bundle_id, answer, run_id}` checks the answer against the
  context the run was given; the service records the verdict itself (`source=judge`), and the
  harness puts the same score — the share of the answer's claims the evidence supports — on
  the run's trace (`grounding`). An answer with no checkable claim is no verdict and no score.
* **human** — `h.feedback(run_id, verdict, correction=None)`. The memory service stores a
  person's verdict as a vote that waits for the tenant administrator (`review.state ==
  "pending"`, its ADR 0028): it changes the run's outcome, and the confidence of what the run
  cited, only once approved (`GET /v1/feedback/pending`, `POST /v1/feedback/{id}/approve`
  with the tenant's admin key). The stored record is returned so a surface can show that.
  The `system` outcome and the judge's verdict are applied as they arrive.

## The memory model key

`BIFROST_VIRTUAL_KEY` is registered for each agent (tenant, agent) once per process, in the
background, idempotently: the memory service's own LLM work for the agent (extraction,
summaries, procedures, the grounding judge) runs on the agent's own key and budget. A memory
service that takes no model keys (`PUT /v1/agents/model-key` answers that its credential
encryption is not configured) is logged once per process, at `INFO`, and not asked again — no
failed-write warning for it; the service then uses the tenant's or the operator's key.
