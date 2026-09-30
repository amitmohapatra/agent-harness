# Memory

`memory=` on `wrap`: `"off"` (default), `"read"` or `"read_write"`. Anything but `"off"` needs
`MEMORY_URL`. Every call the harness makes to the memory service is in
`trellis/harness/clients/memory.py`, in the run's scope (tenant, user, agent, run, thread).

## Push

Before the agent runs, `/v1/context` for the run's question (budget 2000 tokens). The rendered
text reaches the framework as a system message (see the README's matrix) and is
`trellis.current().context`. With `tool_hints=True` the request carries the run's tool names
(`k` = 8; the run's own tools, never the pull tools) so the context includes the service's tool
suggestions. A failed context call is a `warning` event (`memory_unavailable`); the run
continues without it. A `CONTEXT_LOADED` event reports its size.

A LangGraph graph with a checkpointer keeps the thread's messages itself: its context message
has a fixed id (`trellis-memory-context`), so the checkpointed thread holds one that each turn
replaces, and it leaves out the service's recent-conversation section, which would repeat what
the checkpointer already holds.

## Pull

The service's agent tools (listed once per process) are added to the run's tools — all of them
for `read_write`, only `memory_search`, `history_search`, `procedures_search` and `tool_search`
for `read`. For a compiled graph, build them in: `h.tools(..., memory=True)`; a run of an
agent wrapped with `memory="read"` refuses the ones that write. `tool_search` answers among the
run's own tools. Inside a tool or a node, `trellis.current().memory` is the memory SDK's
context bound to the run, with all its verbs.

## What is recorded (`read_write`)

In the background, after each attempt — whether it succeeded, paused or failed:

* the transcript — the question once per run (`<run_id>:user:<n>`), and what the agent said in
  each attempt (`<run_id>:<attempt>:msg:<n>`), on the run's thread (a run started with no
  thread, a scheduled run for one, is its own thread: `thread = run_id`);
* every harness tool call (not the memory service's own tools, which it logs itself), and Code
  Mode's nested calls from Bifrost's log;
* the outcome when the run ends (success, or failure with the error message) — unless the agent
  recorded its own with the `record_outcome` pull tool, which the harness never overwrites
  (and a person's later `h.feedback` on the run is the last word);
* approve/reject/edit decisions as `TOOL_CALL` feedback, and the judge's verdicts as `ANSWER`
  feedback; `h.feedback(run_id, verdict, correction)` records a person's `RUN` feedback.

`read` writes nothing: no transcript, tool calls, outcome or automatic feedback (the judge
still scores a sampled run; its verdict is a metric only). `h.feedback(...)` is explicit and
always recorded.

## The memory model key

With `TRELLIS_MEMORY_MODEL_KEY` set, the key is registered for each agent (tenant, agent) once
per process, in the background, idempotently — the LLM key the service uses for that agent's
memory work.
