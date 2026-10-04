# Documentation

Start with the [README](../README.md) (the API, what is automatic, the framework matrix) and
[ARCHITECTURE.md](../ARCHITECTURE.md) (system context, components, the pipeline, sequence
diagrams of a run, a pause and an A2A call, run states). This page is the map: what each page
covers, and which feature to use when.

## Pages

| Page | |
|---|---|
| **Frameworks** — one page each: install, the lines to add to an existing project, what is automatic, approvals, streaming, durable runs, surfaces, evaluation, limits | [LangGraph and LangChain](frameworks/langgraph.md) (`create_agent`, a hand-built `StateGraph`, checkpointers, `HumanInTheLoopMiddleware`) · [Deep Agents](frameworks/deepagents.md) · [OpenAI Agents SDK](frameworks/openai-agents.md) · [Claude Agent SDK](frameworks/claude-agent-sdk.md) · [ReAct](frameworks/react.md) · [plain functions](frameworks/functions.md) |
| [scenarios.md](scenarios.md) | which to use when, in more detail: targets, tools, memory, pauses, runs, surfaces, observability, evaluation |
| [configuration.md](configuration.md) | the environment, `Settings`, and who the key says the deployment is |
| [tools.md](tools.md) | the toolbox and where tools come from, their side effects, tool hints, Code Mode, `h.tools` |
| [governance.md](governance.md) | which calls run, are announced or ask: risks, the catalog's `approve_when`, failing closed; automatic in `h.wrap`, or `Governance` and `governed` in your own code |
| [memory.md](memory.md) | push, pull, what is recorded, background writes, documents, outcomes and grounding, the model key |
| [interrupts.md](interrupts.md) | `ask`, approvals (the harness's and the frameworks' own), `resume`, the journal, artifacts |
| [runs.md](runs.md) | run records, `start` and the worker, progress checkpoints, schedules, the inbox, the agent-runs wire |
| [surfaces.md](surfaces.md) | `serve_chat` (AG-UI), `serve_a2a`, and `a2a(url)` tools |
| [observability.md](observability.md) | OTel GenAI spans, Langfuse, scores, the collector |
| [evaluation.md](evaluation.md) | offline (a dataset) and online (sampled runs) evaluation: automatic in `h.wrap` (`h.evaluate`, `judges=`), or `EvalServices`, `evaluate` and `judge` in your own code; the evaluators, the judge's model and budget |

`docs/agents/` holds notes for coding agents working on this repo.

## What to use when

### Which target

| You have | Wrap | Page |
|---|---|---|
| a LangChain v1 agent (`create_agent`) or any compiled LangGraph graph | the graph, built with `await h.tools(..., framework="langgraph")` | [langgraph.md](frameworks/langgraph.md) |
| a Deep Agent (`create_deep_agent`) | the graph it returns, built the same way | [deepagents.md](frameworks/deepagents.md) |
| an OpenAI Agents SDK `Agent` (handoffs included) | the `Agent`, with `tools=[...]` (a handoff's specialist: `h.tools(..., framework="openai-agents")`) | [openai-agents.md](frameworks/openai-agents.md) |
| a Claude Agent SDK setup | the `ClaudeAgentOptions`, with `tools=[...]` | [claude-agent-sdk.md](frameworks/claude-agent-sdk.md) |
| a model and tools, no framework | `ReAct(system=..., model=...)` | [react.md](frameworks/react.md) |
| code that decides itself (a workflow, a router, glue) | `async def fn(input, agent)` | [functions.md](frameworks/functions.md) |

Every target gets the same harness: memory push and pull, governance and approvals, records,
grounding, judges, traces, durable runs and both surfaces. What differs is how a pause resumes
(in place for a checkpointed graph and the OpenAI SDK's own approvals; a re-run against the
journal otherwise), how far the tool schemas are narrowed, and what the framework does on its
own (built-in tools, sub-agents, handoffs) — each page says.

### How a run is started

| You want | Use | Needs |
|---|---|---|
| the answer in the request that asked (an API handler, a script) | `await agent.run(input, user=...)` | — |
| text and tool events as they happen (your own UI) | `agent.stream(input, user=...)` | — |
| a chat UI that speaks AG-UI | `agent.serve_chat(app, identity=...)` | `[agui]` |
| a run that outlives the request: long work, approvals that take days, many workers | `await agent.start(...)` + `h.worker([...]).run()` or `python -m trellis.harness.worker module:h` | `RUNS_URL` (else in process) |
| a run on a cadence, acting for someone | `await agent.schedule(cron, input, on_behalf_of=...)` + a worker | `RUNS_URL` (its ticker fires it) |
| another agent (any vendor) calling yours | `agent.serve_a2a(app, url)` | `[a2a]` |

A run started with `run`/`stream`/`serve_chat` resumes in the process that calls `resume`; one
started with `start` or a schedule goes back to the queue and any worker continues it.

### Local or agent-runs

| | Without `RUNS_URL` | With `RUNS_URL` |
|---|---|---|
| run records, the inbox | this process (`LocalRuns`), lost on restart | agent-runs (Postgres) |
| `start` + workers | workers in this process | any worker process, leases, crash recovery |
| schedules | fire when a worker in this process asks for work | agent-runs' ticker |
| deadlines and escalation (`ask(deadline=, escalate_to=)`) | not enforced | the ticker escalates or times the run out |
| large `ask` payloads | in process | run artifacts (`payload_ref`) |

### AG-UI or A2A

| Who is on the other side | Use |
|---|---|
| a person in a chat UI | `serve_chat`: SSE, reconnect with `Last-Event-ID`, interrupts as resume entries |
| another agent that should call yours | `serve_a2a`: agent card, JSON-RPC, streaming, signed push notifications |
| your agent needs another agent | `a2a(url)` in `tools=[...]` or `h.tools(...)`: its questions become your run's |

### Which tools

| The tool is | Use | Governed by |
|---|---|---|
| a Python function in this process | `@tool(side_effects=...)`, or a bare function in `tools=[...]` | its declared side effects, overridden by the catalog |
| an HTTP API with an OpenAPI document | `openapi(spec, only=[...])` | the method (GET read … DELETE irreversible), and the catalog |
| another agent | `a2a(url)` | `write`, and the catalog |
| shared across agents, owned by a platform team | an MCP server in Bifrost, allowed on the agent's virtual key — nothing in code | the server's annotations, and the catalog |
| the agent's own memory | nothing: the memory tools are added when `MEMORY_URL` is set | `memory_search`/`tool_search` read, the rest write |
| a framework's own tool (`function_tool`, Deep Agents' file tools, Claude's `Bash`) | as the framework does | the framework's permissions, not the harness's |

### Approvals and pauses

| You want | Use |
|---|---|
| every call of a tool approved | `side_effects="irreversible"` (or the MCP server's `destructiveHint`) |
| some calls approved, decided by an administrator without a deploy | the catalog's `approve_when` (`amount > 10000`) |
| the framework's own gate (`HumanInTheLoopMiddleware`, `interrupt_on`, `needs_approval`) | keep it: it becomes the same approval — gate each tool in one place |
| the same decisions for tools of an agent you do not wrap | `Governance.from_env(...)` with `check` or `governed(fn, gov, on_ask=...)` ([governance.md](governance.md#way-2-from-your-own-code)) |
| a question, a choice, a table or diff to review | `trellis.current().ask(...)` (or a graph's own `interrupt()`) |
| someone else to answer, by a deadline | `ask(..., assignee="role:…", deadline=..., escalate_to=...)` and `h.inbox(...)` |
| to answer | `agent.resume(id, "approve" \| "reject" \| "edit" \| "answer" \| "cancel", answer=..., reviewer=...)` |

### Memory

| You want | Use |
|---|---|
| the agent to know the user | nothing: the context is pushed before every run |
| the model to search or change memory | nothing: the memory tools |
| your code to read or write memory | `trellis.current().memory` (the SDK's verbs, scoped to the run) |
| a file the user's runs should cite | `await h.add_document(file, user=...)` |
| a person's verdict on a run | `await h.feedback(run_id, verdict, correction=None)` — pending the tenant administrator's review in memory, a score on the trace now |

### Checking quality

| You want | Use |
|---|---|
| every run's answer checked against what memory gave it, cheaply | nothing: the sampled grounding check (`TRELLIS_GROUNDING_SAMPLE`, 10 %) |
| a score for every item of a test set before shipping | `await h.evaluate(agent, dataset, [...])` — a Langfuse dataset run |
| an answer graded by a model against criteria | `llm_judge("criteria")` — offline in `h.evaluate`, online in `judges=` |
| an exact or partial match against an expected answer | `exact_match()`, `contains()` (offline: they need `expected`) |
| grounding as an explicit evaluator in a report | `grounding()` |
| quality on live traffic | `Harness(judges=[...])`, sampled by `TRELLIS_JUDGE_SAMPLE` |
| the same for an agent you do not wrap | `evaluate(my_agent, dataset, [...])` and `judge(case, [...], services=EvalServices.from_env())` ([evaluation.md](evaluation.md#way-2-pluggable-from-your-own-code)) |

### What each environment variable turns on

| Variable | Turns on |
|---|---|
| `MEMORY_URL` (+ `TRELLIS_API_KEY`) | memory: push, pull tools, records, the tool catalog (risks, `approve_when`), grounding, documents, feedback in memory |
| `RUNS_URL` | agent-runs: durable runs, workers across processes, the ticker's schedules and deadlines, run artifacts |
| `BIFROST_URL` (+ `BIFROST_VIRTUAL_KEY`) | MCP tools (what the key allows), Code Mode, `ReAct` and `llm_judge` model names, the memory model key |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `_HEADERS` | trace export; with Langfuse's credentials, its scores API and datasets |
| `TRELLIS_SPOOL_DIR` | memory writes kept on disk across an outage and a restart |
| `TRELLIS_WORKER_CONCURRENCY` | runs a worker executes at once |
| `TRELLIS_GROUNDING_SAMPLE` | the share of runs checked for grounding |
| `TRELLIS_JUDGE_MODEL`, `TRELLIS_JUDGE_VIRTUAL_KEY`, `TRELLIS_JUDGE_SAMPLE` | the judge's model, its budget, and the share of runs online judges score |

Details: [configuration.md](configuration.md).
