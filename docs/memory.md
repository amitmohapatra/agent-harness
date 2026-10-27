# Memory

Memory is on when `MEMORY_URL` is set (or `Harness(memory=MemoryClient(...))`), for every
agent; `without={"memory"}` turns it off for an agent (`h.wrap`) or a run (`agent.run`)
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
the prompt format — `{bundle_id, rendered, token_estimate, evidence_status, memory_ids?, tools?}`, items cited
by short per-bundle handles (`[m1]`, `[d2]`). The rendered text reaches the framework as a system
message (see the README's matrix, and each [framework page](README.md#which-target)) and is `trellis.current().context`; the `bundle_id` is what
the grounding check verifies the answer against. Where the input has no place for it — a
function target given text, a graph whose input has no `messages` — and the run's code never
reads `trellis.current().context`, the agent answers without it: a warning says so once per
agent (`memory_context`), naming the fix (read `agent.context`, or give the input its
messages); the harness puts the context nowhere else.

```python
import trellis


# a function target given text: the runtime it is given, or trellis.current() in code it calls
async def answer(question: str, agent: trellis.Runtime) -> str:
    return await my_llm(system=agent.context or "", user=question)


# a graph whose state has no messages: a node reads the run's context
def triage(state: Ticket) -> dict[str, str]:
    context = trellis.current().context  # None when nothing was pushed
    return {"answer": classify(state["question"], context)}
```

* `document_ids` when the run names documents (`agent.run(..., document_ids=[...])`, the ids
  `h.add_document` returned): the context draws its passages from those alone. Kept with the
  run, so its resume, a worker's attempt and its sub-agents' runs use them too.
* `window=false` when the framework keeps the thread's messages itself in a durable store (a
  LangGraph graph with a durable checkpointer, an OpenAI Agents session on a file;
  `trellis.harness.history.owner`): the service leaves the recent conversation out. Such a
  graph's context message has a fixed id (`trellis-memory-context`), so its thread holds one
  that each turn replaces. A store kept in RAM only (`InMemorySaver`, `SQLiteSession(":memory:")`)
  replays the thread while the process lives and loses it on a restart: memory's window stays
  on there, in the context's text (`## Recent conversation`), never as messages next to the
  store's own.
* Otherwise memory keeps the conversation, and a framework that replays one as messages gets
  it as its own. LangGraph (`ReAct`, `create_agent`, Deep Agents): `window_format="messages"`,
  and the recent turns follow the context message as `HumanMessage`, `AIMessage` with its
  `tool_calls`, `ToolMessage` — exactly what a checkpointer would replay — while the context
  stays one system message without them. The OpenAI Agents SDK: its session is the memory
  service's thread (`bridges.session.MemorySession`), read as the SDK's input items
  (`function_call`, `function_call_output` included), so the window is not pushed again.
  A target that takes one prompt (the Claude Agent SDK, a function) gets them in the context's
  text as before (`## Recent conversation`).
* The request carries the run's own tool names — the harness's and the ones the framework gives
  the model itself, where they can be read: an OpenAI Agents agent's `tools`, Claude's built-in
  tools (its `tools`, less `disallowed_tools`) and its MCP tools named in `allowed_tools`, the
  tools a compiled graph's tool nodes run: the context then includes the skills the agent
  learned for the task ([skills.md](skills.md#learned-skills)) and, with 5 or more tools, the
  tools section (next step, argument values found in memory, what is missing) and its `tools`
  (`[{name, confidence}]`, confidence 0–1), which narrow the tools the model is offered
  ([tools.md](tools.md)). One call: the harness asks for no hints of its own;
  `without={"tool_search"}` keeps the learned skills and offers every tool.

A failed context call is a `warning` event (`memory_unavailable`); the run continues without
it. A `CONTEXT_LOADED` event reports its size; the call is a `retrieve memory` span.

## Pull

The service's agent tools (listed again every 10 minutes; while the service cannot be reached
the last listing stands, and a run when they were never listed goes without them, with a
`warning` event) are added to the run's tools:

| Tool | |
|---|---|
| `memory_search(query, kinds?, time_from?, time_to?, k?)` | memories, documents and — `kinds: ["message"]` — what was said: this conversation's messages first, then the same user's earlier conversations' (never another user's), each with its `thread_id` ([past conversations](https://github.com/amitmohapatra/agent-memory-service/blob/main/docs/guide/04-retrieval.md#past-conversations)); `kinds: ["episode"]` finds which earlier conversation, one summary each |
| `memory_remember(...)` | store a memory verbatim |
| `memory_update(id, content)` | replace a memory (an `[m1]` handle from the context works) |
| `memory_forget(id)` | forget one |
| `profile_edit(block, old?, new)` | edit a pinned profile block |
| `tool_search(task)` | the next step, plan, prefill and missing arguments among the run's own tools — the harness passes the run's toolbox with the call, and offers the tools it returns (`{tools: [{name, confidence, next?, args?, missing?}], plan?}`) to the model |

`memory_search` and `tool_search` only read (they run unannounced); the others write. A graph of
your own gets them with `HarnessTools` (or `await h.memory_tools()`). Inside a tool or a node, `trellis.current().memory` is the memory SDK's
context bound to the run, with all its verbs.

## What is recorded

After each attempt — whether it succeeded, paused or failed — in the background except where
said:

* the transcript, in one batch (`history.add`) — the question, then what the agent said: its
  tool calls (on the assistant message, `tool_calls`) and their results (`TOOL` messages naming
  their call) with its answers, where the framework reports them (LangGraph, the OpenAI Agents
  SDK) — each message named by the harness (`source_system: trellis-harness`,
  `source_message_id`: the run for the question, which every attempt asks, the attempt and
  position for what the agent said), so the service stores a message it has already seen once.
  A run started with no thread — a scheduled run — is its own thread (`thread = run_id`). Where
  memory keeps the conversation the transcript is written before the run returns (at most 5 s,
  then queued like the rest), so the next turn's window holds it wherever that turn runs;
* every harness tool call (not the memory tools, which the service logs itself), and Code
  Mode's nested calls from Bifrost's log — their arguments and outputs as they are
  ([observability.md](observability.md#redaction)), in the order the log has them, each at
  the step of the script that made it, so a learned procedure keeps the script's order;
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
  calls read back from the gateway), it is lost.
* Either way a write given up is never only a warning: it is counted (`trellis.writes.failed`,
  and `trellis.writes.undelivered` with `outcome` `spooled` or `lost`), a `write_failed <label>`
  span with an error status in the run's trace, a `write_failed` warning event while the run's
  stream is open, and an error in the log — the first time each kind of write fails in the
  process, a warning after that. `trellis doctor`'s "records failing" check fails while any
  write waits in the spool, counting them by kind.

So: delivered at least once while the process lives or, with a spool directory on a volume that
outlives the process, after its next start; never silently dropped.

## Documents

`h.add_document(file, user=..., thread=None, title=None, visibility=None, wait=60)` uploads a
file to the memory service (`POST /v1/documents`) in that user's scope (or one thread's) and
waits until it is parsed and indexed (`GET /v1/documents/{id}`). Nothing else is needed: the
push context of the user's next run retrieves and cites its passages like any other
document; to have a run's context draw on given documents alone, pass their ids
(`agent.run(question, user=..., document_ids=[doc.document_id])`). `visibility` widens who may retrieve it (`WORKSPACE`, `TENANT`); inside a run the
SDK's `agent.memory.advanced.documents` does the same in the run's scope.

## Outcome and grounding

The run's outcome is a projection in the memory service of the feedback on the run, by
precedence **human > judge > system**:

* **system** — the harness, when the run ends: `SUCCESS` → `confirm`, `ERROR` → `reject` (a
  cancelled run says nothing about the agent). Never "no exception = success" on its own: the
  judge and people outrank it, and the service keeps an unverified `confirm` as the run's
  outcome only — it moves no memory's confidence, learned skill or tool statistic. What the
  service learns from comes from the judge's verdict on a sampled run and from people's
  reviewed verdicts. The record keeps what the run's context showed, in its metadata under
  `pushed`: the context's `bundle_id` and the ids of the memories in it (`memory_ids`, at
  most 50).
* **judge** — on a sampled share of successful runs with an answer (a structured one as its JSON)
  (a fixed 10 %, chosen by the run id), `/v1/verify {bundle_id, answer, run_id}` checks the answer against the
  context the run was given; the service records the verdict itself (`source=judge`), and the
  harness puts the same score — the share of the answer's claims the evidence supports — on
  the run's trace (`grounding`). An answer with no checkable claim, or a context that held no
  evidence (every claim would be "unsupported" against nothing), is no verdict and no score.
* **human** — `h.feedback(run_id, verdict, correction=None)`, citing the memories the answer
  rests on (`evidence_refs`), whose confidence it moves: the ones the judge's verdict found on a
  run the sample verified, and on any other run the memories its context showed (the `system`
  record's `pushed`); none when no context was pushed. The memory service stores a
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
