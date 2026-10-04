# Which to use when

The harness has one way to attach (`h.wrap(target, id=...)`) and a few choices around it: this
page is Way 1. (Not wrapping, and calling the blocks from your own framework, is Way 2:
[docs/README.md](README.md#way-2-pluggable-blocks-your-framework-our-pieces).) Each section
below starts from what you are trying to do and names the call. The API itself is in
the [README](../README.md); how it works is in [ARCHITECTURE.md](../ARCHITECTURE.md).

## What to wrap

| You have / want | Wrap | Why this one |
|---|---|---|
| A LangGraph graph, or a Deep Agent (`create_deep_agent`) | the compiled graph | Your graph keeps its control flow. Build it with `await h.tools(..., framework="langgraph")` (a compiled graph refuses `tools=`). Compile it with a checkpointer to resume in place. |
| An OpenAI Agents SDK `Agent` | the `Agent` | Harness tools are added to a copy per run (your agent is not changed), narrowed per turn; the SDK's own `needs_approval` tools pause and resume from its `RunState`. |
| A Claude Agent SDK setup | the `ClaudeAgentOptions` | Harness tools reach the CLI as one in-process MCP server (`mcp__trellis__*`, pre-allowed); the memory context is appended to your system prompt. |
| No framework, but a model that should call tools | `ReAct(system=..., model="provider/model")` | The smallest tool-calling loop: native tool messages, a `chat` span per model call, the tools narrowed per call, `output=` for a pydantic answer. A model name needs `BIFROST_URL`; any object with `async complete(messages, **body)` works instead. |
| Code that decides itself what happens (a workflow, a router, glue between agents) | an async function `(input, agent)` | No model of its own: call tools with `await agent.tools.call(...)`, ask with `await agent.ask(...)`, read `agent.context` and `agent.memory`. |

Teams bring their own model objects pointed at Bifrost's OpenAI-compatible endpoint
(`ChatOpenAI(base_url=BIFROST_URL)`, `OpenAIChatCompletionsModel(AsyncOpenAI(base_url=...))`);
the harness wraps agents, not models. Each target has a page with the lines to add to an
existing project and its limits: [LangGraph and LangChain](frameworks/langgraph.md),
[Deep Agents](frameworks/deepagents.md), [OpenAI Agents SDK](frameworks/openai-agents.md),
[Claude Agent SDK](frameworks/claude-agent-sdk.md), [ReAct](frameworks/react.md),
[plain functions](frameworks/functions.md); the short decision tables are in
[docs/README.md](README.md#what-to-use-when).

## Where a tool comes from

| The tool is | Use | Notes |
|---|---|---|
| A Python function in this process | `tools=[fn]` or `@tool(side_effects=...)` | Schema from the signature, description from the docstring. Declare `side_effects`: `read` runs, `write` (default) runs and is announced, `irreversible` asks a person. |
| An HTTP API with an OpenAPI 3 document | `openapi(spec, only=[...])` | One tool per `operationId`; the method decides the side effects (GET read, POST/PUT/PATCH write, DELETE irreversible). |
| Another agent | `a2a(url)` | One `write` tool; its questions become this run's questions. |
| Shared by many agents, owned by a platform team, budgeted | an MCP server registered in Bifrost, allowed on the agent's virtual key | Nothing in code: the toolbox is what the key allows, governed by the server's annotations and the catalog. Many read-only Code Mode servers become Code Mode meta-tools. |
| Needed when the agent is built (a compiled graph; any framework's agent built before wrapping) | `await h.tools(*sources, framework=...)` | The same toolbox in the framework's own type; every call still goes through the bridge. |

Approvals by tool: make the tool `irreversible` (or let its MCP server say `destructiveHint`).
Approvals by call: an administrator's `approve_when` rule in the memory service's tool catalog
(`amount > 10000`) asks exactly when it holds — no code change, and it replaces what the risk
decides ([governance.md](governance.md)).

## Memory: reading and writing

| You want | Use |
|---|---|
| The agent to know the user (profile, past conversations, documents, learned procedures) | nothing: with `MEMORY_URL` the context is pushed before every run |
| The model to search or change memory itself | nothing: the memory pull tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are in its toolbox |
| Your own code to read or write memory (a node, a tool, a function target) | `trellis.current().memory` — the memory SDK's verbs, already scoped to the run's tenant, user, agent, run and thread |
| A file the user's runs should cite | `await h.add_document(file, user=..., thread=None)` |
| A person's verdict on a run (thumbs up/down, a correction) | `await h.feedback(run_id, verdict, correction=None)` — a Langfuse score now, memory's `human` feedback once the tenant administrator approves it |
| The run's outcome to teach memory | nothing: `SUCCESS`/`ERROR` are the run's `system` feedback; a sampled share (`TRELLIS_GROUNDING_SAMPLE`, 10 % by default) is checked for grounding |
| No memory at all (tests, a stateless tool agent) | leave `MEMORY_URL` unset |

## Pausing for a person

| You want | Use |
|---|---|
| A question with a free-form answer | `await trellis.current().ask("...", expects={...})` (a form) |
| A choice | `ask("...", options=[...])` |
| A person to check rows, or a before/after | `ask("...", table=rows)` / `ask("...", diff=(before, after))`; add `expects=` to make it a review with a correction (large tables and diffs travel as run artifacts) |
| Someone other than the run's user to answer | `assignee="role:finance"` (or `user:…`) and find it with `h.inbox("role:finance")` |
| An answer by a deadline | `deadline=...`, and `escalate_to=` for who gets it after: agent-runs hands the question to `escalate_to` when the deadline passes, or ends the run `TIMEOUT` when nobody is named |
| Every call of a tool approved | `side_effects="irreversible"` |
| Some calls of a tool approved | the catalog's `approve_when` |

Where the answer comes from does not matter to the run: `await agent.resume(interrupt_id,
decision, answer=..., reviewer=...)` in code, a `resume` entry from the chat UI, or the next
A2A message on the task.

How the run continues: a LangGraph graph with a checkpointer resumes where it stopped (where
that checkpointer still holds the pause — an `InMemorySaver` only in the process that paused),
and an OpenAI Agents `needs_approval` pause continues the SDK's saved run; every other pause
runs the target again from its input, and the journal returns the answers and tool outputs it
already has. Keep side effects in harness tools: a call through the bridge runs once, but code
outside a tool (and the model calls) run again on the re-run.

## Running it

| You want | Use |
|---|---|
| An answer in the request that asked (an API handler, a script) | `await agent.run(input, user=...)` |
| Text and tool events as they happen (your own UI) | `agent.stream(input, user=...)`, or `serve_chat` for an AG-UI client |
| A run that outlives this process: long work, approvals that take days, many workers | `await agent.start(...)`, `h.worker([...]).run()` (or `python -m trellis.harness.worker module:h`), and `RUNS_URL` |
| A run on a cadence, acting for someone | `await agent.schedule(cron, input, on_behalf_of=...)` — idempotent, so it is safe in deployment code |
| Development and tests | leave `RUNS_URL` unset: runs, the queue and schedules live in the process (`LocalRuns`) and nothing survives a restart |
| Workers in production: deploys, scaling, stop signals | `python -m trellis.harness.worker module:h --concurrency N` (or `TRELLIS_WORKER_CONCURRENCY`; default the CPU count, 1–8); `SIGTERM` lets held runs finish for 25 s, then releases them for another worker — give the container ~45 s ([runs.md](runs.md#workers)) |
| Memory writes that survive an outage and a restart | `TRELLIS_SPOOL_DIR` on a volume that outlives the process: what could not be delivered is replayed at the next start ([memory.md](memory.md#background-writes-what-is-guaranteed)) |

A run started with `run`/`stream` resumes in the process that calls `resume`; one started with
`start` (or by a schedule) goes back to the queue on resume and any worker continues it.

## AG-UI or A2A

| Who is on the other side | Use |
|---|---|
| A person, in a chat UI that speaks AG-UI | `agent.serve_chat(app, identity=...)`: SSE with numbered events, reconnect after a dropped connection, interrupts as resume entries, large payloads as artifacts |
| Another agent, any framework or vendor, that should call yours | `agent.serve_a2a(app, url)`: the agent card, JSON-RPC, streaming, signed push notifications; pauses are `input-required` |
| Your agent needs another agent | `a2a(url)` in `tools=[...]` |

Both surfaces can be mounted on one FastAPI app (`examples/serve_chat.py`). Identity is always
the deployment's: pass `identity=` (or put an authenticating edge in front for A2A's trusted
header); without it every caller is `anonymous`.

Both show up in the app's OpenAPI document (tags `agui`, `a2a`). With several replicas, keep a
chat thread on one (sticky sessions): its events are buffered in the process that serves it
([surfaces.md](surfaces.md#ag-ui-agentserve_chatapp--pathagui-identitynone)).

## Seeing what happened

| You have | Set |
|---|---|
| Langfuse only | `OTEL_EXPORTER_OTLP_ENDPOINT=https://…/api/public/otel` and the Basic auth header: traces and scores go straight there |
| Langfuse and Datadog | the collector in `deploy/otel-collector.yaml`: every span to Datadog, the GenAI spans to Langfuse |
| Your own OpenTelemetry setup | nothing: the harness uses the API, and an installed provider is kept |
| A run to debug locally | `agent.stream(...)` events, and `trellis.current().log(...)` lines (also `log` events) |

## Evaluating it

| You want | Use |
|---|---|
| A score for every item of a test set, before shipping | `await h.evaluate(agent, "dataset-name" or [items], [exact_match(), llm_judge("...")])`: each item through the real pipeline, scores on the traces, a Langfuse dataset run, an `EvalReport` |
| Quality on live traffic | `Harness(judges=[llm_judge("...")])`: a sampled share of runs (`TRELLIS_JUDGE_SAMPLE`) judged in the background |
| A judge that does not grade itself, on its own budget | `TRELLIS_JUDGE_MODEL` (a stronger model than the agent's) and `TRELLIS_JUDGE_VIRTUAL_KEY` |
| A check of your own | any `async (EvalCase) -> EvalScore \| None` in the evaluators or judges |
| Evaluation of an agent you do not wrap | `evaluate(any_async_callable, dataset, [...])` and `judge(case, [...], services=...)` from `trellis.harness.evals` ([blocks/evaluation.md](blocks/evaluation.md)) |

Annotation queues and datasets built from traces are Langfuse's ([evaluation.md](evaluation.md));
whether the harness itself got slower is `make bench`.
