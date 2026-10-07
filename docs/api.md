# API reference

Every public name of the wrapped API (Way 1), with its arguments, their defaults and the page
that explains them. The names import from `trellis`; the blocks (Way 2) import from their own
modules (last section). What each environment variable does, and what is automatic, is
[configuration.md](configuration.md); how it is built is [architecture.md](architecture.md).

```python
from trellis import (
    Agent,
    Ask,
    Deny,
    Harness,
    Hooks,
    ModelCall,
    ReAct,
    Result,
    Rewrite,
    RunHandle,
    Runtime,
    Settings,
    a2a,
    current,
    openapi,
    sandbox,
    skills,
    tool,
)
```

## Harness

```text
Harness(config=None, *, runs=None, memory=None, gateway=None, governance=None, prompts=None,
        skills=None, judges=(), hooks=())
```

The deployment's blocks: the ones given, the rest built from the environment. `async with
Harness() as h:` (or `await h.aclose()`) drains the background writes, exports the queued
spans and closes the clients it built; a block you give is yours to close.

| Argument | Default | What it is |
|---|---|---|
| `config` | `Settings.from_env()` | the deployment as fields, instead of the environment ([Settings](#settings)) |
| `runs` | agent-runs' `RunsClient` with `RUNS_URL`, else the in-process `LocalRuns` | the run store: `trellis.runs.RunsClient`, `trellis.harness.runs.LocalRuns`, or yours with the same calls; `False`: in process whatever `RUNS_URL` says ([runs.md](runs.md)) |
| `memory` | a `MemoryClient` with `MEMORY_URL`, else none | `trellis.memory.MemoryClient`; `False`: no memory whatever `MEMORY_URL` says ([memory.md](memory.md)) |
| `gateway` | a `Gateway` with `BIFROST_URL`, else none | `trellis.harness.clients.bifrost.Gateway(url, virtual_key)`; `False`: no gateway ([gateway.md](gateway.md)) |
| `governance` | the memory service's tool catalog, per tenant (with memory), else the tools' own risks | a `Governance`, used for every tenant ([governance.md](governance.md)) |
| `prompts` | `PROMPTS_DIR`, Langfuse (its keys set), the gateway | the prompt sources, asked in order: `Prompt(...)`, `prompts_dir(...)`, `langfuse_prompts(...)`; `[]`: none ([prompts.md](prompts.md)) |
| `skills` | `SKILLS_DIR`, the gateway | the skill sources: `Skill(...)`, `skills_dir(...)`; `[]`: none ([skills.md](skills.md)) |
| `judges` | none | online evaluators run on a sampled share of successful runs ([evaluation.md](evaluation.md#online-judges)) |
| `hooks` | none | `Hooks` around every wrapped agent's runs, model calls and tool calls ([hooks.md](hooks.md)) |

`h.agents` is every agent it wraps, by id; `h.runs` the run store; `h.memory` the memory client
(or `None`); `h.writes` the background writes (`await h.writes.drain()`).

### h.wrap

```text
h.wrap(target, *, id, tools=(), version=None, mcp=None, skills=(), timeout=None, without=(),
       hooks=(), framework_options=None) -> Agent
```

Attach the harness to `target`: a compiled LangGraph graph (Deep Agents and `ReAct` included),
an OpenAI Agents `Agent`, `ClaudeAgentOptions`, or an async function `(input, agent)`. The
framework is detected from its type and not modified.

| Argument | Default | What it is |
|---|---|---|
| `id` | required | the agent's id: its runs, records, traces, catalog entries and inbox entries are under it |
| `tools` | `()` | the agent's own tools, run in this process: functions, [`tool`](#tool), [`a2a`](#a2a), [`openapi`](#openapi), [`sandbox`](#sandbox), [`skills`](#skills), `agent.as_tool()`; refused for a compiled graph, which takes `h.tools(...)` ([tools.md](tools.md)) |
| `version` | `TRELLIS_AGENT_VERSION` | the agent's code version, recorded with each run it starts ([reliability.md](reliability.md#agent-version)) |
| `mcp` | every MCP tool the virtual key allows | the gateway's Virtual MCPs (by slug) whose tools are the agent's; `[]`: none ([gateway.md](gateway.md)) |
| `skills` | `()` | skills by name (`"name"`, `"name@version"`) or given (`Skill(...)`) ([skills.md](skills.md)) |
| `timeout` | no limit | the most working time of each run, in seconds, however it starts ([reliability.md](reliability.md#run-time-limit-and-deadline)) |
| `without` | `()` | what the harness does not do for any of its runs ([configuration.md](configuration.md#what-is-on-and-how-to-turn-it-off)) |
| `hooks` | `()` | this agent's hooks, after the harness's ([hooks.md](hooks.md)) |
| `framework_options` | none | the framework's own options for every run's call, passed through ([configuration.md](configuration.md#the-frameworks-own-run-options)) |

A framework installed outside the range this release was tested with is said once in the log
when its first target is wrapped ([versioning.md](versioning.md)).

### h.tools

```text
await h.tools(*sources, framework, mcp=None, without=()) -> list
```

The toolbox as the framework's own tools, for an agent built with them before it is wrapped:
LangChain tools (`framework="langgraph"` or `"deepagents"`), `FunctionTool`s
(`"openai_agents"`), one in-process MCP server (`"claude_agent_sdk"`). It holds `sources`
(functions, `skills(...)`, `sandbox()`, `agent.as_tool()`...), the MCP tools (`mcp=`: those of
these Virtual MCPs) and the memory tools; `without=` leaves features' tools out
([tools.md](tools.md#tools-built-into-the-agent-htools)).

### h.prompt, h.prompt_messages, h.model_headers

```text
await h.prompt(ref, /, **values) -> str
await h.prompt_messages(ref, /, **values) -> list[dict]
await h.model_headers(*, prompt=None) -> Mapping[str, str]
```

`prompt` is the prompt `ref` names (`"name"`, `"name@version"`, a `Prompt`) from the first
prompt source that has it, its `{{variables}}` filled; inside a run it is pinned and journaled
([prompts.md](prompts.md)). `model_headers` are the headers for a framework's own model client
pointed at the gateway: the deny-all MCP scope and, with `prompt=`, that stored prompt's
selection, pinned per run — every run of the harness pins it at its start, and the headers,
read at each request, select the version the run executing pinned (outside a run: the one
resolved when they were made) ([gateway.md](gateway.md#prompts)).

### h.worker

```text
h.worker(agents, *, concurrency=None) -> Worker
```

Claims queued runs of these agents and executes them, `concurrency` at a time (default
`TRELLIS_WORKER_CONCURRENCY`, else the CPU count from 1 to 8): `await worker.run()` until
`worker.stop()`, `await worker.serve()` (stopped by `SIGTERM`/`SIGINT`), `await
worker.run_once()` ([runs.md](runs.md#workers)). From a shell:

```bash
python -m trellis.harness.worker app.agents:h --concurrency 4
```

### h.inbox, h.feedback, h.add_document, h.evaluate

```text
await h.inbox(assignee=None, *, tenant=None) -> list[RunSummary]
await h.feedback(run_id, verdict, correction=None, *, tenant=None) -> Feedback | None
await h.add_document(file, *, user, tenant=None, thread=None, title=None, visibility=None,
                     wait=60) -> DocumentInfo
await h.evaluate(agent, dataset, evaluators, *, run_name=None, description=None, metadata=None,
                 concurrency=4, limit=None, user=None) -> EvalReport
```

| Call | What it does | Page |
|---|---|---|
| `inbox` | the paused runs waiting on `assignee` (`user:…`, `role:…`) or anyone, newest first; answer one with `agent.resume` | [interrupts.md](interrupts.md#an-inbox-of-your-own) |
| `feedback` | a person's verdict on a run (`confirm`, `approve`, `reject`, `correct`, `edit`): a score on its trace, and with memory the run's `human` feedback, pending the tenant administrator | [memory.md](memory.md) |
| `add_document` | a file into a user's (or a thread's) document memory, waiting until it is indexed (`wait=None`: at once); needs `MEMORY_URL` | [memory.md](memory.md) |
| `evaluate` | the agent over a dataset (a Langfuse dataset's name, or a list), each answer scored onto its run's trace | [evaluation.md](evaluation.md) |

`tenant=` is only for a platform key (one with no tenant of its own).

## Agent

What `h.wrap` returns.

### agent.run, agent.stream, agent.start

```text
await agent.run(input, *, user, thread=None, tenant=None, timeout=None, deadline=None,
                without=(), framework_options=None) -> Result
agent.stream(...same...) -> AsyncIterator[RunEvent]
await agent.start(...same..., priority=0, concurrency_key=None) -> RunHandle
```

| Argument | Default | What it is |
|---|---|---|
| `input` | required | the framework's input: a string, a message list, a dict (queued runs: JSON) |
| `user` | required | whom the run is for: its memory scope, its approvals' default assignee |
| `thread` | none (the run is its own) | the conversation: memory's thread, LangGraph's `thread_id`, the default `concurrency_key` |
| `tenant` | the key's tenant | only for a platform key |
| `timeout` | the agent's (`h.wrap(timeout=)`) | the most working time in seconds, across attempts, pauses not counted → `TIMEOUT` ([reliability.md](reliability.md#run-time-limit-and-deadline)) |
| `deadline` | none | when the run must have ended → `TIMEOUT` |
| `without` | `()` | parts off for this run, on top of the agent's; kept with the run ([configuration.md](configuration.md#what-is-on-and-how-to-turn-it-off)) |
| `framework_options` | the agent's | the framework's own options for this run, over the agent's ([configuration.md](configuration.md#the-frameworks-own-run-options)) |
| `priority` (`start`) | `0` | -1000 to 1000: a higher one is claimed first ([runs.md](runs.md#queue-order-and-busy-conversations)) |
| `concurrency_key` (`start`) | `thread:<thread>` | runs sharing one run one at a time |

`run` runs to the end or the first pause. `stream` yields the run's contracts `RunEvent`s up to
`RUN_FINISHED`; closing it early cancels the run. `start` queues the run for a worker and
returns a [RunHandle](#runhandle).

### agent.resume

```text
await agent.resume(interrupt_id, decision, *, answer=None, reviewer, comment=None,
                   remember="once", tenant=None) -> Result
```

Answer a pause: `interrupt_id` (or the run's id), `decision` one of `answer`, `approve`,
`reject`, `edit` (the edited arguments as `answer`), `cancel`; `reviewer` who decided;
`comment` kept with the decision; `remember="run"` on an approval approves that tool's later
calls in this run. An answer that does not fit is refused (`ConfigurationError`) and the run
keeps waiting ([interrupts.md](interrupts.md#answering)).

### agent.execute, agent.cancel, agent.events

```text
await agent.execute(job) -> Result
await agent.cancel(run_id, *, reason=None, tenant=None) -> RunRecord
agent.events(run_id, *, after=0, tenant=None) -> AsyncIterator[RunEvent]
```

`execute` runs the next attempt of a run a worker claimed (a `trellis.runs.Job`): any worker
of yours calls it ([runs.md](runs.md#workers),
[examples/05_features/own_scheduler.py](../examples/05_features/own_scheduler.py)). `cancel`
stops a run wherever it is ([reliability.md](reliability.md#cancel)). `events` is a run's
events until it ends, from any replica with `RUNS_URL`
([runs.md](runs.md#a-runs-events-from-anywhere)).

### agent.schedule

```text
await agent.schedule(cron, input, *, on_behalf_of, tz="UTC", tenant=None, timeout=None,
                     without=(), framework_options=None, priority=0, concurrency_key=None)
    -> Schedule
```

Queue a run on a cadence (cron, or `hourly`, `daily`, `weekly`, `weekdays`, `manual`), acting
for `on_behalf_of`; each fired run takes the options `start` takes, kept in the schedule's
metadata ([runs.md](runs.md#schedules),
[examples/06_scenarios/scheduled_run_selection_limit.py](../examples/06_scenarios/scheduled_run_selection_limit.py)).

### agent.as_tool, agent.serve_chat, agent.serve_a2a

```text
agent.as_tool(*, name=None, description=None, side_effects=None) -> SubAgent
agent.serve_chat(app, *, path="/agui", identity=None)
agent.serve_a2a(app, url, *, identity=None)
```

`as_tool` is this agent as another agent's tool: each call a child run
([subagents.md](subagents.md)). `serve_chat` mounts AG-UI routes on a FastAPI app;
`serve_a2a` the agent card and A2A JSON-RPC at `url` ([surfaces.md](surfaces.md)).

## RunHandle

`RunHandle` — what `agent.start` returns: `run_id`; `await handle.status()` (the `RunRecord`), `await handle.cancel(reason=None)`,
`await handle.result(timeout=None)` (waits for a pause or an ending; a `Result`).

## Result

`run_id`, `status` (`SUCCESS`, `PAUSED`, `ERROR`, `TIMEOUT`, `QUEUED`, `CANCELLED`), `answer`,
`interrupt` (a contracts `Interrupt` when paused), `error` (a contracts `AgentError`).

## Runtime

`Runtime` — `trellis.current()` inside a tool or a node (`None` outside a run), and the second argument of
a function target.

| Name | What it is |
|---|---|
| `run_id`, `agent_id`, `user`, `thread`, `tenant`, `attempt`, `task` | the run |
| `context` | the pushed memory context (`None` without memory) |
| `memory` | the memory SDK's verbs in the run's scope (needs memory) |
| `idempotency_key` | inside a tool call: that call's key, the same in every attempt |
| `remaining()` | the seconds the code may still take, or `None` |
| `uses(feature)` | whether the run has that part on (`without=`) |
| `await tools.call(name, **args)`, `await tools.hints(task)` | a tool call through the harness; the tool hints |
| `await ask(question, *, expects, form, table, diff, options, multiple, ui_schema, component, props, assignee, deadline, escalate_to)` | pause the run and return the answer on resume ([interrupts.md](interrupts.md#asking)) |
| `log(message, **fields)` | a log line and a `log` event |

## ReAct

```text
ReAct(system, model, *, output=None, max_steps=None, max_repeats=None, model_timeout=None,
      context_window=None, prompt=None, prompt_vars=None, middleware=(), checkpointer=None)
```

A tool-calling agent for teams with no framework: a LangChain v1 `create_agent` graph with the
native context middleware and the harness's ([frameworks/react.md](frameworks/react.md)).

| Argument | Default | What it is |
|---|---|---|
| `system` | required | the instructions (may be `""` with a `prompt`) |
| `model` | required | a Bifrost model name (a `ChatOpenAI` on `BIFROST_URL`) or any LangChain chat model |
| `output` | none | a pydantic model the answer is parsed into |
| `max_steps` | 12 | model calls with tools, then one more without |
| `max_repeats` | 3 | identical calls in a row that stop the run |
| `model_timeout` | none | the most one model call may take, in seconds |
| `context_window` | the model's profile, else 128,000 | the model's window in tokens |
| `prompt`, `prompt_vars` | none | a prompt of the prompt sources, pinned for the run, and its variables ([prompts.md](prompts.md)) |
| `middleware` | `()` | more LangChain or Deep Agents middleware; one named like a default replaces it |
| `checkpointer` | `RunCheckpointer()` | the graph's checkpointer (the run's journal keeps the checkpoint) |

## Tools

### tool

```text
tool(fn, *, name=None, description=None, side_effects="write", idempotent=False, timeout=None)
@tool  /  @tool(...)
```

A Python function (sync or async) as a tool: the schema from its signature, the description
from its docstring; `side_effects` `"read"`, `"write"` or `"irreversible"`; `idempotent` when a
repeated call with its idempotency key has its effect once; `timeout` seconds per call
([tools.md](tools.md)).

### a2a

`a2a(url, *, name=None, timeout=120)` — a remote A2A agent as one `write` tool
([surfaces.md](surfaces.md#calling-a2a-agents-a2aurl--namenone-timeout120)).

### openapi

`openapi(spec, *, only=None, base_url=None, headers=None, timeout=120)` — an OpenAPI 3
document's operations as tools, one per `operationId` ([tools.md](tools.md)).

### sandbox

`sandbox(provider=None, spec=None, *, timeout=120)` — `sandbox_exec`, `sandbox_read`,
`sandbox_write` in a sandbox of the run's own (`SANDBOX=docker`, or a provider of yours)
([sandbox.md](sandbox.md)).

### skills

`skills(*refs)` — skills by name or given as a tool source (`load_skill`, `read_skill_file`)
([skills.md](skills.md)). `Prompt`, `prompts_dir`, `langfuse_prompts` are in
`trellis.harness.prompts`; `Skill`, `skills_dir` in `trellis.harness.skills`.

## Hooks

`Hooks` — subclass and override `on_run_start(run)`, `on_run_end(run, result)`,
`before_model(call) -> ModelCall | None`, `after_model(call, reply)`, `before_tool(call) ->
None | Deny | Ask | Rewrite`, `after_tool(call, outcome) -> outcome`, `on_error(stage, error)`.
`Deny(reason)`, `Ask(question, assignee=None, component=None, props=None)`, `Rewrite(args)`,
`ModelCall(framework, messages, model=None, system=None)` ([hooks.md](hooks.md)).

## current

`current() -> Runtime | None` — the run the calling code executes in.

## Settings

`Settings(bifrost_url=, bifrost_virtual_key=, api_key=, memory_url=, runs_url=, otlp_endpoint=,
otlp_headers=, spool_dir=, worker_concurrency=, agent_version=, grounding_sample=,
judge_model=, judge_virtual_key=, judge_sample=, prompts_dir=, skills_dir=, langfuse_host=,
langfuse_public_key=, langfuse_secret_key=, sandbox=, sandbox_image=)` — every field optional;
`Settings.from_env()` is what `Harness()` reads. Each field, its variable, default and an
example: [configuration.md](configuration.md#settings-and-the-environment).

## The blocks (Way 2)

| Block | Import | Page |
|---|---|---|
| Memory | `from trellis.memory import MemoryClient` | [blocks/memory.md](blocks/memory.md) |
| Runs | `from trellis.runs import RunsClient, Worker` | [blocks/runs.md](blocks/runs.md) |
| Governance | `from trellis.harness.governance import Governance, governed` | [blocks/governance.md](blocks/governance.md) |
| Evaluation | `from trellis.harness.evals import EvalServices, evaluate, judge` | [blocks/evaluation.md](blocks/evaluation.md) |
| A2A client | `from trellis.harness.a2a import remote` | [blocks/a2a.md](blocks/a2a.md) |
| Sandbox | `from trellis.harness.sandbox import DockerSandbox, SandboxSpec` | [sandbox.md](sandbox.md#way-2-without-a-harness) |
| Contracts | `from trellis.contracts import RunStart, Interrupt, InterruptResolution` | [blocks/mixing.md](blocks/mixing.md#the-records-they-share) |
| Gateway | `from bifrost_sdk import Bifrost, NO_GATEWAY_TOOLS` | [gateway.md](gateway.md) |
| Test helpers | `from trellis.testing import Reviewer, Decide` | [interrupts.md](interrupts.md) |
