# Memory

Memory is on exactly when `MEMORY_URL` is set; there is nothing to configure per agent. Every
call the harness makes to the memory service is in `trellis/harness/clients/memory.py`, in the
run's scope (tenant, user, agent, run, thread). With memory on, every run reads (push, pull)
and records (transcript, tool calls, outcome); with it off there is no context, no memory
tools, no records and no catalog, and `trellis.current().memory`, `tools.hints(...)` and
`h.add_document(...)` raise `ConfigurationError`.

## Push

Before the agent runs, `/v1/context` for the run's question (budget 2000 tokens), answered in
the prompt format — `{bundle_id, rendered, token_estimate, evidence_status, tools?}`, items cited
by short per-bundle handles (`[m1]`, `[d2]`). The rendered text reaches the framework as a system
message (see the README's matrix) and is `trellis.current().context`; the `bundle_id` is what
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

The service's agent tools (listed once per process) are added to the run's tools: 

| Tool | |
|---|---|
| `memory_search(query, kinds?, time_from?, time_to?, k?)` | memories, documents and — `kinds: ["message"]` — the conversation history |
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
  Mode's nested calls from Bifrost's log;
* approve/reject/edit decisions as `TOOL_CALL` feedback.

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
* **judge** — on a sampled 10 % of successful runs with a text answer (`GROUNDING_SAMPLE`,
  chosen by the run id), `/v1/verify {bundle_id, answer, run_id}` checks the answer against the
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
summaries, procedures, the grounding judge) runs on the agent's own key and budget.
