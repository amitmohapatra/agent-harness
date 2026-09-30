# Memory

`memory=` on `wrap`: `"off"` (default), `"read"` or `"read_write"`. Anything but `"off"` needs
`MEMORY_URL`. Every call the harness makes to the memory service is in
`trellis/harness/clients/memory.py`, in the run's scope (tenant, user, agent, run, thread).

## Push

Before the agent runs, `/v1/context` for the run's question (budget 2000 tokens). The rendered
text reaches the framework as a system message (see the README's matrix) and is
`trellis.current().context`. With `tool_hints=True` the request carries the run's tool names
(`k` = 8) so the context includes the service's tool suggestions. A failed context call is a
`warning` event (`memory_unavailable`); the run continues without it. A `CONTEXT_LOADED` event
reports its size.

## Pull

The service's agent tools (listed once per process) are added to the run's tools — all of them
for `read_write`, only `memory_search`, `history_search`, `procedures_search` and `tool_search`
for `read`. For a compiled graph, build them in: `h.tools(..., memory=True)`. Inside a tool or
a node, `trellis.current().memory` is the memory SDK's context bound to the run, with all its
verbs.

## What is recorded (`read_write`)

In the background, after the run:

* the transcript — the question and the assistant messages, each once
  (`<run_id>:msg:<n>` idempotency keys);
* every harness tool call (not the memory service's own tools, which it logs itself), and Code
  Mode's nested calls from Bifrost's log;
* the outcome (success, or failure with the error message);
* approve/reject/edit decisions as `TOOL_CALL` feedback, and the judge's verdicts as `ANSWER`
  feedback; `h.feedback(run_id, verdict, correction)` records a person's `RUN` feedback.

`read` writes nothing: no transcript, tool calls, outcome or automatic feedback (the judge
still scores a sampled run; its verdict is a metric only). `h.feedback(...)` is explicit and
always recorded.

## The memory model key

With `TRELLIS_MEMORY_MODEL_KEY` set, the key is registered for each agent (tenant, agent) once
per process, in the background, idempotently — the LLM key the service uses for that agent's
memory work.
