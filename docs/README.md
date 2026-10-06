# Documentation

Trellis is used in one of two ways ([README](../README.md#two-ways-to-use-trellis), with the
decision table): **wrapped**, where the harness runs your agent and every piece is automatic,
or as **pluggable blocks**, where your framework runs the agent and your code calls the pieces
it wants. This page is the map: the pages of each way, and which feature to use when.
[ARCHITECTURE.md](../ARCHITECTURE.md) has how it is built (system context, components, the
pipeline, sequence diagrams of a run, a pause and an A2A call, run states).

## Way 1: wrapped (the harness runs your agent)

`h.wrap(agent)`, then `agent.run`. Each page says how a feature works and what the harness does
for you.

| Page | |
|---|---|
| **Frameworks** — one page each: install, the lines to add to an existing project, what is automatic, approvals, streaming, durable runs, surfaces, evaluation, limits | [LangGraph and LangChain](frameworks/langgraph.md) (`create_agent`, a hand-built `StateGraph`, checkpointers, `HumanInTheLoopMiddleware`) · [Deep Agents](frameworks/deepagents.md) · [OpenAI Agents SDK](frameworks/openai-agents.md) · [Claude Agent SDK](frameworks/claude-agent-sdk.md) · [ReAct](frameworks/react.md) · [plain functions](frameworks/functions.md) |
| [onboarding.md](onboarding.md) | getting started: onboard a tenant (the platform key, the admin key, the application's key), the environment, and a key per person for an approvals UI |
| [scenarios.md](scenarios.md) | which to use when, in more detail: targets, tools, memory, pauses, runs, surfaces, observability, evaluation |
| [configuration.md](configuration.md) | the environment, `Settings`, and who the key says the deployment is |
| [tools.md](tools.md) | the toolbox and where tools come from, their side effects, tool hints, Code Mode, `h.tools` |
| [gateway.md](gateway.md) | the Bifrost gateway: stored prompts (`prompt=`), skills (`skills=`), Virtual MCPs (`mcp=`), who an MCP call is for, what the gateway never does for a run (no injected tools, no Agent Mode, Code Mode through the bridge), frameworks' own MCP clients |
| [prompts.md](prompts.md) | prompts from code, `.md` files (`PROMPTS_DIR`), Langfuse and the gateway: one name, the order they are looked up in, `ReAct(prompt=)`, `h.prompt` for any framework, the judge; pinned per run |
| [skills.md](skills.md) | Agent Skills from code, `SKILL.md` folders (`SKILLS_DIR`) and the gateway, mixed in one run: progressive disclosure, pinned per run, `without={"skills"}` |
| [subagents.md](subagents.md) | `agent.as_tool()`: an agent as another agent's tool — child runs, their pauses answered through the parent, crashes, cancel, time |
| [sandbox.md](sandbox.md) | `sandbox()`: commands and files in a sandbox of the run's own (Docker; how E2B, Daytona, Modal plug in) — its life, pauses, crashes, timeouts, governance; the frameworks' own sandboxes or ours |
| [governance.md](governance.md) | which calls run, are announced or ask: risks, the catalog's `approve_when`, failing closed, and what the harness does with each decision |
| [hooks.md](hooks.md) | your code around runs, model calls and tool calls: guardrails (deny, ask, rewrite), redaction of your own, audit — where each hook fires on each adapter |
| [memory.md](memory.md) | push, pull, what is recorded, background writes, documents, outcomes and grounding, the model key |
| [interrupts.md](interrupts.md) | `ask`, approvals (the harness's and the frameworks' own), `resume`, the journal, artifacts |
| [runs.md](runs.md) | run records, `start` and the worker, progress checkpoints, schedules, the inbox, the agent-runs wire |
| [reliability.md](reliability.md) | time limits (tools, models, runs, deadlines), retries and every retry layer, idempotency keys, crashes and unknown outcomes, cancel, the agent's version |
| [surfaces.md](surfaces.md) | `serve_chat` (AG-UI), `serve_a2a`, and `a2a(url)` tools |
| [observability.md](observability.md) | OTel GenAI spans, Langfuse, scores, the collector |
| [evaluation.md](evaluation.md) | offline (`h.evaluate` over a dataset) and online (`judges=` on sampled runs) evaluation; the evaluators, the judge's model and budget, Langfuse experiments |

## Way 2: pluggable blocks (your framework, our pieces)

Your framework runs the agent, untouched; your code imports a block and calls it. Each page:
what the block is, install, setup from the environment, the API, its behaviour (errors,
retries, tenancy), and how it relates to Way 1.

| Page | Block |
|---|---|
| [blocks/memory.md](blocks/memory.md) | `trellis.memory`: the context into your prompt, the turn and each tool call recorded, feedback |
| [blocks/runs.md](blocks/runs.md) | `trellis.runs`: durable runs, a pause with your framework's checkpoint, the inbox, resume, `Worker`, schedules, artifacts, webhooks and `verify_signature` |
| [blocks/governance.md](blocks/governance.md) | `trellis.harness.governance`: `Governance.check` and `governed` on your own tools, `publish`, `decided` |
| [blocks/evaluation.md](blocks/evaluation.md) | `trellis.harness.evals`: `evaluate` on any async function, `judge` on one run, `EvalServices.from_env` |
| [blocks/a2a.md](blocks/a2a.md) | `trellis.harness.a2a.remote`: call any A2A agent (serving is Way 1) |
| [sandbox.md](sandbox.md#way-2-without-a-harness) | `trellis.harness.sandbox`: a provider (`DockerSandbox`) and its sandboxes, your commands governed with `governed` |
| [blocks/contracts.md](blocks/contracts.md) | `trellis.contracts`: which records each block takes and returns, and why they are shared |

**Recipes**, end to end with the framework's own pause and state: an unmodified agent with
memory context and recording, governed tools, a durable pause in agent-runs answered from the
inbox, and a judge.

| Page | |
|---|---|
| [blocks/langgraph.md](blocks/langgraph.md) | a LangGraph graph: `governed` tools asking through `interrupt`, the checkpointer, `Command(resume=)`; a `Worker` continuing the graph across processes |
| [blocks/openai-agents.md](blocks/openai-agents.md) | an OpenAI Agents `Agent`: `needs_approval` from governance, the `RunState` as the run's checkpoint |
| [blocks/claude-agent-sdk.md](blocks/claude-agent-sdk.md) | a Claude Agent SDK `query()`: `can_use_tool` from governance, the session as the checkpoint |

## Composition: a Harness is the blocks you give it

The two ways are one set of blocks — the run store, the memory client, the Bifrost gateway,
governance, the prompt and skill sources — composed differently.

| You want | Write | What you get |
|---|---|---|
| everything, from the deployment (Way 1) | `Harness()` | each block built from its environment variable (`RUNS_URL`, `MEMORY_URL`, `BIFROST_URL`; governance from the catalog per tenant), each off when its variable is unset |
| some blocks of your own, the rest from the deployment | `Harness(runs=RunsClient(...), governance=Governance(...))` | the blocks you pass, used as they are (and yours to close); the others built from the environment |
| a block off although the deployment names it | `Harness(memory=False)` (`gateway=False`, `runs=False`: runs kept in this process) | that block off for every agent of this harness |
| your own scheduler or worker | `await agent.execute(job)` for each run it claims (`trellis.runs.Worker(runs, agent.execute, [agent.id])`, or a loop of yours around `runs.claim`) | the run's next attempt with its journal, governance, memory and limits ([runs.md](runs.md#workers)) |
| `ReAct` (or any target) with your blocks | `Harness(<your blocks>).wrap(ReAct(...))` | one loop and one path: the same `ReAct` as Way 1, on your blocks |
| the blocks without the harness (Way 2) | import the block and call it ([the blocks](#way-2-pluggable-blocks-your-framework-our-pieces)) | your framework runs the agent; your code calls each block where it chooses |

```python
from trellis import Harness, ReAct
from trellis.harness.governance import Governance
from trellis.runs import RunsClient

runs = RunsClient()  # RUNS_URL, TRELLIS_API_KEY
h = Harness(runs=runs, memory=False, governance=Governance())
agent = h.wrap(ReAct(system="You handle refunds.", model="provider/model"), id="refunds")
```

`Harness(config=None, *, runs=None, memory=None, gateway=None, governance=None, prompts=None,
skills=None, judges=(), hooks=())`: `runs` a `RunStore` (`trellis.runs.RunsClient`,
`trellis.harness.runs.LocalRuns`, or your own with the same calls), `memory` a
`trellis.memory.MemoryClient`, `gateway` a `trellis.harness.clients.bifrost.Gateway(url,
virtual_key)`, `governance` a `Governance` (used for every tenant), `prompts` and `skills` the
prompt and skill sources, in the order they are asked (`[prompts_dir("prompts"),
Prompt(...)]`, `[skills_dir("skills"), Skill(...)]`: [prompts.md](prompts.md),
[skills.md](skills.md); `[]` is none); `hooks` the [hooks](hooks.md) of every agent it wraps. Runnable: [examples/react_with_blocks.py](../examples/react_with_blocks.py)
(its own run store, its own scheduler loop, governance, no memory).

## Mixing both ways

[blocks/mixing.md](blocks/mixing.md): wrapped agents and your own in one deployment, sharing one
inbox, one tool catalog, one memory and one place for scores; answering each run its own way;
calling across; moving an agent from one way to the other.

`docs/agents/` holds notes for coding agents working on this repo.

## What to use when

For a wrapped agent (Way 1); where code you do not wrap has its own answer, the row links its
block page.

### Which target

| You have | Wrap | Page |
|---|---|---|
| a LangChain v1 agent (`create_agent`) or any compiled LangGraph graph | the graph, built with `await h.tools(..., framework="langgraph")` | [langgraph.md](frameworks/langgraph.md) |
| a Deep Agent (`create_deep_agent`) | the graph it returns, built the same way | [deepagents.md](frameworks/deepagents.md) |
| an OpenAI Agents SDK `Agent` (handoffs included) | the `Agent`, with `tools=[...]` (a handoff's specialist: `h.tools(..., framework="openai_agents")`) | [openai-agents.md](frameworks/openai-agents.md) |
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

### Time, failures and cancelling

| You want | Use |
|---|---|
| a tool that may hang to give up | `@tool(timeout=20)`, `openapi(spec, timeout=)`, `a2a(url, timeout=)` — a read says it timed out, a write is reported as of unknown effect |
| a model call bounded | `ReAct(..., model_timeout=30)` |
| a `ReAct` model whose window is not 128k tokens (and whose model object does not say) | `ReAct(..., context_window=32_000)`: older results are cleared and older turns compacted from it |
| every run of an agent bounded, however it starts (chat, A2A, evaluation, schedules too) | `h.wrap(..., timeout=900)` → `TIMEOUT` |
| a run that may not work longer than N seconds (pauses not counted), or must end by a time | `agent.run/stream/start(..., timeout=600, deadline=...)` (over the agent's) → `TIMEOUT` |
| reads retried, writes never repeated, after a crash too | nothing: automatic ([reliability.md](reliability.md#retries)) |
| a tool's service to deduplicate | hand it `trellis.current().idempotency_key` (OpenAPI writes send it already) |
| to stop a run | `await agent.cancel(run_id, reason=...)` or `await handle.cancel()` — queued, paused, here or on a worker |
| to know which code ran a run | `h.wrap(..., version=)` or `TRELLIS_AGENT_VERSION` |

### Local or agent-runs

| | Without `RUNS_URL` | With `RUNS_URL` |
|---|---|---|
| run records, the inbox | this process (`LocalRuns`), lost on restart | agent-runs (Postgres) |
| `start` + workers | workers in this process | any worker process, leases, crash recovery |
| schedules | fire when a worker in this process asks for work | agent-runs' ticker |
| deadlines and escalation (`ask(deadline=, escalate_to=)`) | not enforced | the ticker escalates or times the run out |
| a run's `timeout=` and `deadline=` | each attempt stops on time | each attempt stops on time, and the ticker ends a run past either, its worker dead or not |
| a queued run that fails with an error that may pass | ends `ERROR` | queued again, up to 3 times, after a backoff |
| large `ask` payloads | in process | run artifacts (`payload_ref`) |

### AG-UI or A2A

| Who is on the other side | Use |
|---|---|
| a person in a chat UI | `serve_chat`: SSE, reconnect with `Last-Event-ID`, interrupts as resume entries |
| another agent that should call yours | `serve_a2a`: agent card, JSON-RPC, streaming, signed push notifications |
| your agent needs another agent | `a2a(url)` in `tools=[...]` or `h.tools(...)`: its questions become your run's |
| code that is not wrapped (any framework) needs another agent | `remote(url, tenant=, user=)` ([blocks/a2a.md](blocks/a2a.md)) |

### Which tools

| The tool is | Use | Governed by |
|---|---|---|
| a Python function in this process | `@tool(side_effects=...)`, or a bare function in `tools=[...]` | its declared side effects, overridden by the catalog |
| an HTTP API with an OpenAPI document | `openapi(spec, only=[...])` | the method (GET read … DELETE irreversible), and the catalog |
| another agent served elsewhere | `a2a(url)` | `write`, and the catalog |
| another agent this harness wraps (a sub-agent) | `agent.as_tool()` in `tools=[...]` or `h.tools(...)` ([subagents.md](subagents.md)) | `read` when every tool it declares reads, else `write`; and the catalog |
| code the model writes and runs, away from the host | `sandbox()` in `tools=[...]` or `h.tools(...)`, `SANDBOX=docker` ([sandbox.md](sandbox.md)) | `sandbox_exec` and `sandbox_write` write, `sandbox_read` reads; and the catalog |
| shared across agents, owned by a platform team | an MCP server in Bifrost, allowed on the agent's virtual key — nothing in code | the server's annotations, and the catalog |
| the agent's own memory | nothing: the memory tools are added when `MEMORY_URL` is set | `memory_search`/`tool_search` read, the rest write |
| a framework's own tool (`function_tool`, Deep Agents' file tools) | as the framework does | the framework's permissions, not the harness's |
| Claude Code's built-in tools (`Bash`, `Write`, `Read`...) | as the CLI does | governance by risk (`Bash` asks, writes announced, reads run) and your hooks, through the SDK's permission callback; then your own `can_use_tool` |

### Approvals and pauses

| You want | Use |
|---|---|
| every call of a tool approved | `side_effects="irreversible"` (or the MCP server's `destructiveHint`) |
| some calls approved, decided by an administrator without a deploy | the catalog's `approve_when` (`amount > 10000`) |
| the framework's own gate (`HumanInTheLoopMiddleware`, `interrupt_on`, `needs_approval`) | keep it: it becomes the same approval — gate each tool in one place |
| the same decisions for tools of an agent you do not wrap | `Governance.from_env(...)` with `check` or `governed(...)` ([blocks/governance.md](blocks/governance.md)) |
| a rule only your code knows: deny a call, rewrite its arguments, ask someone | a hook: `before_tool` returning `Deny`, `Rewrite` or `Ask` ([hooks.md](hooks.md)) |
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
| which tools a run called, in what order, with what arguments | `called("lookup", before="refund")`, `tool_sequence([...])`, or your own reading `case.trajectory` ([evaluation.md](evaluation.md#trajectories)) |
| quality on live traffic | `Harness(judges=[...])`, sampled by `TRELLIS_JUDGE_SAMPLE` |
| the same for an agent you do not wrap | `evaluate(my_agent, dataset, [...])` and `judge(case, [...], services=...)` ([blocks/evaluation.md](blocks/evaluation.md)) |

### What is on, and how to turn it off

Everything the deployment configures is on for every agent — nothing to set. One switch turns
parts of it off: `without={...}`.

```python
agent = h.wrap(graph, id="triage", without={"judges"})  # every run of this agent
await agent.run(question, user="ada", without={"memory"})  # this run: no memory at all
```

| Feature (`without=` name) | On when | What it is | Turned off |
|---|---|---|---|
| `memory` | `MEMORY_URL` | `memory_push`, `memory_pull` and `records` together | the run has no memory scope: no context, no memory tools, nothing recorded, `trellis.current().memory` refused |
| `memory_push` | `MEMORY_URL` | the memory context pushed into the framework's input (and the tool hints with it) | no context, no `/v1/context` call |
| `memory_pull` | `MEMORY_URL` | the memory tools (`memory_search`, `tool_search`, ...) | not offered (a graph's, bound at build: hidden by its `ModelHooks()` middleware, else offered and answering that they are off; `h.tools(without=)` leaves them out) |
| `records` | `MEMORY_URL` | the transcript, every tool call, the outcome, decisions as feedback | nothing written to memory about the run |
| `hints` | `MEMORY_URL`, from 5 tools | the tool hints narrow the tools the model is offered | every tool offered |
| `grounding` | `MEMORY_URL`, a sampled share (`TRELLIS_GROUNDING_SAMPLE`) | the answer checked against the context it was given | not checked |
| `judges` | `Harness(judges=[...])`, a sampled share (`TRELLIS_JUDGE_SAMPLE`) | the online judges | not judged |
| `mcp` | `BIFROST_URL` | the MCP tools the virtual key allows (or those of `mcp=`'s Virtual MCPs), Code Mode included | no MCP tools, and the gateway is not asked for them (`h.tools(without={"mcp"})` for a graph's; `mcp=[]` on `wrap` or `h.tools` says the same for every run of the agent) |
| `code_mode` | `BIFROST_URL`, enough read-only Code Mode servers | their tools behind Bifrost's Code Mode meta-tools (one script instead of many calls) | those servers' tools offered one by one |
| `skills` | `skills=` / `skills(...)` | the skills' section in the context and `load_skill`, `read_skill_file` | neither |

`without=` on `h.wrap` turns them off for every run of the agent; on `agent.run`, `stream` and
`start` for that run, on top of the agent's — kept with the run's record, so its resume, the
worker that continues it and its sub-agents' runs are without them too. A name not in the table
is refused (`ConfigurationError`, naming them). Not switchable, because they are automatic and
deterministic: governance and approvals, the journal and replay, retries and time limits,
tracing and redaction, the run record.

### What each environment variable turns on

| Variable | Turns on |
|---|---|
| `MEMORY_URL` (+ `TRELLIS_API_KEY`) | memory: push, pull tools, records, the tool catalog (risks, `approve_when`), grounding, documents, feedback in memory |
| `RUNS_URL` | agent-runs: durable runs, workers across processes, the ticker's schedules and deadlines, run artifacts |
| `BIFROST_URL` (+ `BIFROST_VIRTUAL_KEY`) | MCP tools (what the key allows), Code Mode, `ReAct` and `llm_judge` model names, the memory model key, the gateway's stored prompts and skills as sources |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `_HEADERS` | trace export; with Langfuse's credentials, its scores API and datasets |
| `TRELLIS_SPOOL_DIR` | memory writes kept on disk across an outage and a restart |
| `TRELLIS_WORKER_CONCURRENCY` | runs a worker executes at once |
| `TRELLIS_AGENT_VERSION` | the agents' version, recorded with every run they start |
| `TRELLIS_GROUNDING_SAMPLE` | the share of runs checked for grounding |
| `TRELLIS_JUDGE_MODEL`, `TRELLIS_JUDGE_VIRTUAL_KEY`, `TRELLIS_JUDGE_SAMPLE` | the judge's model, its budget, and the share of runs online judges score |
| `PROMPTS_DIR`, `SKILLS_DIR` | a folder of `.md` prompts, a folder of `SKILL.md` skills, as sources ([prompts.md](prompts.md), [skills.md](skills.md)) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`) | Langfuse's prompt management as a prompt source |
| `SANDBOX` (+ `SANDBOX_IMAGE`) | the sandboxes `sandbox()` makes when given no provider: `docker`, of that image |

Details: [configuration.md](configuration.md).
