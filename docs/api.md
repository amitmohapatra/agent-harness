# API reference

Every public name of the wrapped API (Way 1), with its arguments, their defaults and the page
that explains them. The names import from `trellis`; the blocks (Way 2) import from their own
modules (last section). What each environment variable does, and what is automatic, is
[configuration.md](configuration.md); how it is built is [architecture.md](architecture.md).

```python
from trellis import (
    Agent,
    Ask,
    Budget,
    Deny,
    Harness,
    Hooks,
    Output,
    OutputKind,
    ReAct,
    Result,
    Rewrite,
    RunHandle,
    Runtime,
    Settings,
    a2a,
    current,
    llm_judge,
    mcp,
    openapi,
    sandbox,
    stdio,
    stored,
    tool,
)
```

Each setting lives in one place: the environment says where the services are (and their
keys); `Harness()` connects to them; the builder (`ReAct`, or your framework's own) says what
the agent is — its model, its instructions, its tools; `h.wrap` says who it is and how every
run of it goes; `agent.run` says what this call is
([configuration.md](configuration.md#settings-and-the-environment)).

## Harness

```text
Harness(config=None, *, memory=None, runs=None, gateway=None)
```

The connections to the deployment's services: the ones given, the rest built from the
environment. `async with Harness() as h:` (or `await h.aclose()`) drains the background writes,
exports the queued spans and closes the clients it built; a client you give is yours to close.

| Argument | Default | What it is |
|---|---|---|
| `config` | `Settings.from_env()` | the deployment as fields, instead of the environment ([Settings](#settings)) |
| `memory` | a `MemoryClient` with `MEMORY_URL`, else none | `trellis.memory.MemoryClient`; `False`: no memory whatever `MEMORY_URL` says — for the whole deployment: no client, the local tenant, no `h.add_document` (and `RUNS_URL` refused, agent-runs needing the key's tenant). To leave memory out for one agent or one run while the rest keep it, use `without={"memory"}` ([memory.md](memory.md)) |
| `runs` | agent-runs' `RunsClient` with `RUNS_URL`, else the in-process `LocalRuns` | the run store: `trellis.runs.RunsClient`, `trellis.harness.runs.LocalRuns(keep_finished=1000, keep_finished_seconds=3600)` (the finished runs it keeps, and how long), or yours with the same calls; `False`: in process whatever `RUNS_URL` says ([runs.md](runs.md)) |
| `gateway` | a `Gateway` with `BIFROST_URL`, else none | `trellis.harness.clients.bifrost.Gateway(url, virtual_key)`; `False`: no gateway ([gateway.md](gateway.md)) |

Governance is the memory service's tool catalog, per tenant (with memory), else each tool's own
risk ([governance.md](governance.md)). `h.agents` is every agent it wraps, by id; `h.runs` the
run store; `h.memory` the memory client (or `None`); `h.writes` the background writes (`await
h.writes.drain()`).

### h.wrap

```text
h.wrap(agent, *, id, version=None, description=None, hooks=(), judges=(), budget=None,
       run_timeout=None, without=(), framework_options=None) -> Agent
```

Attach the harness to `agent`: a `ReAct`, a compiled LangGraph graph (Deep Agents included),
an OpenAI Agents `Agent`, `ClaudeAgentOptions`, or an async function `(input, agent)`. The
framework is detected from its type and not modified; its own tools are governed as it runs
them ([tools.md](tools.md)).

| Argument | Default | What it is |
|---|---|---|
| `id` | required | the agent's id: its runs, records, traces, catalog entries and inbox entries are under it |
| `version` | `service.version` of `OTEL_RESOURCE_ATTRIBUTES` | the agent's code version, recorded with each run it starts ([reliability.md](reliability.md#agent-version)) |
| `description` | a function's docstring | what the agent does, as another agent (a sub-agent's tool) and the A2A card read it |
| `hooks` | `()` | `Hooks` around its runs, model calls and tool calls, in order ([hooks.md](hooks.md)) |
| `judges` | `()` | online judges, each on the share of successful runs it samples (`llm_judge(sample=)`; else a tenth) ([evaluation.md](evaluation.md#online-judges)); not `grounding()`, which every wrapped run already gets on a sampled share (refused, saying so) |
| `budget` | none: no cap | the most each run may spend, a sub-agent's run counted toward its parent's ([`Budget`](#budget)) |
| `run_timeout` | no limit | the most working time of each run, in seconds, however it starts ([reliability.md](reliability.md#run-time-limit-and-deadline)) |
| `without` | `()` | what the harness does not do for any of its runs: `memory`, `tool_search`, `mcp`, `code_mode`, `grounding`, `judges` ([configuration.md](configuration.md#what-is-on-and-how-to-turn-it-off)) |
| `framework_options` | none | the framework's own options for every run's call, passed through ([configuration.md](configuration.md#the-frameworks-own-run-options)) |

A framework installed outside the range this release was tested with is said once in the log
when its first target is wrapped ([versioning.md](versioning.md)).

### h.prompt

```text
await h.prompt(ref, /, **values) -> Rendered
```

The gateway's stored prompt `ref` (`"name"`, `"name@3"`, or `stored(...)`), its
`{{variables}}` filled with `values`: `.text` and `.messages` (its chat messages), for any
framework. Inside a run it is pinned and journaled: a resumed run reads the same version
([prompts.md](prompts.md)).

### h.memory_tools

```text
await h.memory_tools(*, tenant=None) -> list
```

`memory_tools` are the memory service's agent tools as LangChain tools, for a graph built by
hand: each call in the scope of the run that makes it ([memory.md](memory.md)). A wrapped
graph's own tools are governed with no middleware of yours ([governance.md](governance.md)).

### h.worker

```text
h.worker(agents, *, concurrency=None) -> Worker
```

Claims queued runs of these agents and executes them, `concurrency` at a time (default: the
CPU count, from 1 to 8): `await worker.run()` until
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
await h.evaluate(agent, dataset, judges, *, run_name=None, description=None, metadata=None,
                 concurrency=4, limit=None, user=None) -> EvalReport
```

| Call | What it does | Page |
|---|---|---|
| `inbox` | the paused runs waiting on `assignee` (`user:…`, `role:…`) or anyone, newest first; answer one with `agent.resume` | [interrupts.md](interrupts.md#an-inbox-of-your-own) |
| `feedback` | a person's verdict on a run (`confirm`, `reject`, `correct`): a score on its trace, and with memory the run's `human` feedback citing the memories its answer rested on (the judge's), pending the tenant administrator's review (accepted or dismissed). `approve` and `edit` are refused: they record a decision on a paused action, which `agent.resume` makes | [memory.md](memory.md) |
| `add_document` | a file into a user's (or a thread's) document memory, waiting until it is indexed (`wait=None`: at once); needs `MEMORY_URL`; a run scoped to it: `agent.run(..., document_ids=[doc.document_id])` | [memory.md](memory.md) |
| `evaluate` | the agent over a dataset (a Langfuse dataset's name, or a list), each answer scored by `judges` (the same kind as `h.wrap(judges=)`) onto its run's trace | [evaluation.md](evaluation.md) |

`tenant=` is only for a platform key (one with no tenant of its own).

## Agent

What `h.wrap` returns.

### agent.run, agent.stream, agent.start

```text
await agent.run(input, *, user, thread=None, tenant=None, run_timeout=None, deadline=None,
                budget=None, without=(), framework_options=None, document_ids=None) -> Result
agent.stream(...same...) -> AsyncIterator[RunEvent]
await agent.start(...same..., priority=0, concurrency_key=None) -> RunHandle
```

| Argument | Default | What it is |
|---|---|---|
| `input` | required | the framework's input: a string, a message list, a dict (queued runs: JSON) |
| `user` | required | whom the run is for: its memory scope, its approvals' default assignee |
| `thread` | none (the run is its own) | the conversation: memory's thread, LangGraph's `thread_id`, the default `concurrency_key` |
| `tenant` | the key's tenant | only for a platform key |
| `run_timeout` | the agent's (`h.wrap(run_timeout=)`) | the most working time in seconds, across attempts, pauses not counted → `TIMEOUT` ([reliability.md](reliability.md#run-time-limit-and-deadline)) |
| `deadline` | none | the clock time (a `datetime`) the run must have ended by, time queued and paused for a person counted (agent-runs ends it even while it is queued or paused) → `TIMEOUT`; a child run keeps the earlier of its own and its parent's |
| `budget` | the agent's (`h.wrap(budget=)`) | the most this run may spend; kept with the run ([`Budget`](#budget)) |
| `without` | `()` | parts off for this run, on top of the agent's; kept with the run ([configuration.md](configuration.md#what-is-on-and-how-to-turn-it-off)) |
| `framework_options` | the agent's | the framework's own options for this run, over the agent's ([configuration.md](configuration.md#the-frameworks-own-run-options)) |
| `document_ids` | none (every document the person's scope reads) | the documents (`h.add_document(...).document_id`) the run's memory context draws its passages from, none else; kept with the run and its sub-agents' runs ([memory.md](memory.md#push)) |
| `priority` (`start`) | `0` | -1000 to 1000: a higher one is claimed first ([runs.md](runs.md#queue-order-and-busy-conversations)) |
| `concurrency_key` (`start`) | `thread:<thread>` | runs sharing one run one at a time |

`run_timeout` is working time and `deadline` is clock time: a run that waits a day for an
approver is not past a `run_timeout` of an hour, and is past a `deadline` an hour away. The
`deadline` of `ask(...)` is a third thing, the time one question must be answered by
([interrupts.md](interrupts.md)).

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
await agent.schedule(cron, *, user, input, tz="UTC", tenant=None, run_timeout=None,
                     budget=None, without=(), framework_options=None, document_ids=None,
                     priority=0, concurrency_key=None) -> Schedule
```

Queue a run on a cadence (cron, or `hourly`, `daily`, `weekly`, `weekdays`, `manual`), for
`user` with `input`; each fired run takes the options `start` takes, kept in the schedule's
metadata ([runs.md](runs.md#schedules),
[examples/06_scenarios/scheduled_run_selection_limit.py](../examples/06_scenarios/scheduled_run_selection_limit.py)).

### agent.serve_chat, agent.serve_a2a

```text
agent.serve_chat(app, *, path="/agui", identity=None)
agent.serve_a2a(app, url, *, identity=None)
```

`serve_chat` mounts AG-UI routes on a FastAPI app; `serve_a2a` the agent card and A2A JSON-RPC
at `url` ([surfaces.md](surfaces.md)). Both take the caller from the trusted
`X-Trellis-Identity` header the platform's edge sets, unless `identity=` names it another way;
a call without either is `401` on both surfaces. A wrapped agent in a `ReAct`'s `subagents=[...]` is its
sub-agent: each call a child run ([subagents.md](subagents.md)).

## RunHandle

`RunHandle` — what `agent.start` returns: `id` (the run's); `await handle.status()` (the
`RunRecord`); `await handle.result(wait=None)` (waits for a pause or an ending, at most `wait`
seconds; a `Result`). Cancel the run with `agent.cancel(handle.id)`.

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
| `await tools.call(tool, **args)` | a tool call through the harness: a tool of the run by name, or a tool itself (`tool(fn)`, `mcp(...)`, `a2a(url)`...: it joins the run first) |
| `await tools.use(*sources)` | a function target's own tools added to the run (`sandbox().tools()`, `mcp(...)`...): the names they add |
| `await tools.hints(task)` | the memory service's tool hints for `task` |
| `await ask(question, *, expects, form, table, diff, options, multiple, ui_schema, component, props, assignee, deadline, escalate_to)` | pause the run and return the answer on resume ([interrupts.md](interrupts.md#asking)) |
| `log(message, **fields)` | a log line and a `log` event |

## ReAct

```text
ReAct(system_prompt, model, *, tools=(), subagents=(), skills=None, memory_files=None,
      permissions=None, backend=None, response_format=None, state_schema=None,
      context_schema=None, cache=None, max_steps=12, max_repeats=3, call_timeout=None,
      context_window=None, effort=None, tool_guidance=None, answer_check=None, middleware=())
```

A tool-calling agent for teams with no framework: LangChain's `create_agent` with the harness's
[`middleware()`](#middleware) — Deep Agents' `create_deep_agent`, every feature of it native, when
a Deep Agents capability is asked for (`backend`, a sub-agent of its own, `skills`,
`memory_files`, `permissions`)
([frameworks/react.md](frameworks/react.md)). The parameter names are Deep Agents' where both have
one. Its checkpoint is kept in the run (`RunCheckpointer`) and its store is the memory service:
for a graph with a checkpointer or a store of your own, build it with
`create_agent(..., middleware=middleware(), checkpointer=..., store=...)`. What the assistant
can and cannot do beyond its tools ("you cannot accept an image upload") goes in
`system_prompt`.

**Must**

| Argument | Default | What it is |
|---|---|---|
| `system_prompt` | required | the instructions: a string, or `stored("name", **values)` — the gateway's stored prompt, pinned per run ([prompts.md](prompts.md)) |
| `model` | required | a model name — through Bifrost with `BIFROST_URL`, else the provider's own SDK with its own key (`openai/...`, `anthropic/...`) — or any LangChain chat model ([configuration.md](configuration.md#models)) |

**Usual**

| Argument | Default | What it is |
|---|---|---|
| `tools` | `()` | what the agent can call: functions, [`tool`](#tool), LangChain tools, [`mcp`](#mcp-stdio) servers, [`openapi`](#openapi) ([tools.md](tools.md)); every run also has the MCP tools the virtual key allows and the memory tools. A sandbox is a `backend`, an agent a sub-agent: either given here is refused |
| `response_format` | none | a pydantic model or a JSON schema the answer is parsed into |

**Optional**

| Argument | Default | What it is |
|---|---|---|
| `subagents` | `()` | what it hands tasks to, each a child run: Deep Agents' own — a `SubAgent` dict (`name`, `description`, `system_prompt`, `mode` isolated or fork, `tools`, `model`, `middleware`, `skills`, `permissions`, `response_format`; its `interrupt_on` is the catalog's: a tool of its own named there asks a person) or a `CompiledSubAgent` `{"name", "description", "runnable"}` (another `ReAct` among them), called through Deep Agents' `task` tool — and wrapped agents (`h.wrap(...)`) and [`a2a(url)`](#a2a), each a tool ([subagents.md](subagents.md)) |
| `skills` | none | Deep Agents' skill sources (`["/skills/"]`), read through `backend` |
| `memory_files` | none | Deep Agents' `memory`: `AGENTS.md`-style files loaded into the instructions (named so it is not taken for the memory service) |
| `backend` | the graph's state, kept in the run's journal | where Deep Agents' file tools work: a `StateBackend`, `StoreBackend` (the memory service when wrapped), `FilesystemBackend`, `CompositeBackend`, or the run's sandbox: [`sandbox()`](#sandbox) (`execute` runs code there). Files outside the graph's state are the agent's from the start and governed as any tool |
| `permissions` | none | Deep Agents' `FilesystemPermission` rules for its file tools |
| `max_steps` | 12 | model calls with tools in one invocation, then one more without (1 or more): a loop guard, counted again each invocation. For a spending cap over the whole run (its attempts and sub-agents included) use [`Budget(max_tool_calls=)`](#budget) |
| `call_timeout` | none | the most one model call may take, in seconds (a tool's own `call_timeout` bounds a tool call) |
| `effort` | none | the reasoning effort sent with every call of a model name (`reasoning_effort`) |

**Advanced**

| Argument | Default | What it is |
|---|---|---|
| `middleware` | `()` | your LangChain or Deep Agents middleware, added before `ModelHooks`; one named as the harness's is refused, naming the parameter that sets it (`middleware()` itself: ReAct already has it), one named as Deep Agents' replaces it |
| `state_schema`, `context_schema` | none | Deep Agents' graph state (a `DeepAgentState` subclass) and run context schemas |
| `cache` | none | LangGraph's node cache |
| `context_window` | the model's LangChain profile (`model.profile`); else unknown | the model's window in tokens; every threshold is scaled by it. Unknown: clearing old tool results and the long-task tools are off (said once) |
| `max_repeats` | 3 | identical calls in a row that stop the run (2 or more) |
| `tool_guidance` | `None`: automatic | how to choose among tools, added to the instructions (`ToolGuidance`); `None`: off, as measured ([tool choice](frameworks/react.md#tool-choice)), `True` always, `False` never |
| `answer_check` | `None`: automatic | the draft answer checked against the request once, and the first change against the rules (`AnswerCheck`: one more model call each); `None`: off, as measured ([tool choice](frameworks/react.md#tool-choice)), `True` always, `False` never |

Not exposed: Deep Agents' `interrupt_on` (approvals are the [catalog](governance.md)'s), `name`
(the wrap's `id`), `debug`.

## middleware

```text
middleware(*, max_steps=12, max_repeats=3, call_timeout=None, context_window=None,
           tool_guidance=None, answer_check=None, system_prompt=None)
```

The harness's part of `ReAct` for a LangChain `create_agent` or a Deep Agents
`create_deep_agent` graph of your own — `create_deep_agent(model, tools,
middleware=middleware(), checkpointer=RunCheckpointer())` — the settings meaning what
[`ReAct`](#react)'s do: the run's tools per model call (narrowed per turn; the memory and gateway
MCP tools), the graph's own tools governed, `max_repeats`, `max_steps` then a final answer, the
run's budget, planning on a long task, older results cleared and read back (with a
`context_window`), the tool guidance and the answer check, and `ModelHooks` (the span, a failed
call's `on_error`, `call_timeout`, a `stored(...)` `system_prompt` pinned per run). Give the graph
`checkpointer=RunCheckpointer()` (`trellis.harness.middleware`) for its checkpoint to live in the
run. A `create_agent` or Deep Agents graph wrapped without it still runs governed and narrowed,
said once ([frameworks/react.md](frameworks/react.md#middleware-on-your-own-graph)).

## Budget

```text
Budget(max_tokens=None, max_cost=None, max_tool_calls=None)
```

The most one run may spend, set per agent (`h.wrap(budget=)`) or per run
(`agent.run(budget=)`; none given: no cap, spending still counted): tokens (input and output, every model call), US dollars (what the gateway says
each call cost — Bifrost's `usage.cost`; a call it put no cost on is counted in tokens, said
once when a dollar cap is set) and harness tool calls; `None` is no limit. A sub-agent's
spending counts toward its parent's, and a resumed run goes on from what it spent. At a limit
no new tool call starts (the calls under way finish), a `ReAct` is asked once more, with no
tools, for its best answer, and the run ends `PARTIAL`, its result's error `BUDGET_EXCEEDED`.
What is counted: the model replies of a `ReAct` (and any graph with `ModelHooks`) and of an
OpenAI Agents run (tokens; its dollars are not reported), a Claude Agent SDK run (each query's
tokens and `total_cost_usd`, from its `ResultMessage`, subagents included), and every harness
tool call of any framework. A tenant's or a team's budget is
the gateway's (Bifrost's virtual-key budgets).

## Tools

### tool

```text
tool(fn, *, name=None, description=None, side_effects="write", idempotent=False,
     call_timeout=None)
@tool  /  @tool(...)
```

A Python function (sync or async) as a tool: the schema from its signature (each parameter
described by the docstring's `Args:`), the description from its docstring without its sections; `side_effects` `"read"`, `"write"` or `"irreversible"`; `idempotent` when a
repeated call with its idempotency key has its effect once; `call_timeout` seconds per call
([tools.md](tools.md)).

### mcp, stdio

```text
mcp(target, *, name=None, call_timeout=120, headers=None)
stdio(command, *args, env=None, cwd=None)
```

An MCP server's tools as a `tools=[...]` entry: `target` is its URL (`"https://host/mcp"`, over
streamable HTTP, `headers=` sent with every request) or a local server (`stdio(...)`: the
harness starts it and talks over its stdin and stdout). Tools are named `<name>-<tool>` (else
the name the server gives itself); their side effects come from their annotations. The MCP
tools the virtual key allows are the gateway's: every run has them ([tools.md](tools.md)). The
`mcp` extra (the official MCP SDK).

### a2a

`a2a(url, *, name=None, call_timeout=120)` — a remote A2A agent: a sub-agent
(`ReAct(subagents=[a2a(url)])`), a tool, or an async callable anywhere (`await
a2a(url)(message, user=, tenant=)` outside a run, as the run inside one)
([surfaces.md](surfaces.md#calling-a2a-agents)).

### openapi

`openapi(spec, *, only=None, base_url=None, headers=None, call_timeout=120)` — an OpenAPI 3
document's operations as tools, one per `operationId` ([tools.md](tools.md)).

### sandbox

```text
sandbox(provider=None, *, image=None, spec=None, call_timeout=120)
```

A sandbox of the run's own — Docker (the daemon `DOCKER_HOST` names; `image`, else
`python:3.12-slim`), or a provider of yours — made at the run's first call and named in its
journal, attached again by a resume, snapshotted and paused while it waits for a person,
deleted at its end ([sandbox.md](sandbox.md)). One entry, given where each engine takes it:

| | |
|---|---|
| `ReAct(backend=sandbox())` | `ReAct`'s backend: Deep Agents' file tools and `execute` work in it |
| `sandbox().deep_agent()` | `create_deep_agent`'s arguments, `{"backend": ...}`: `create_deep_agent(model, **sandbox().deep_agent())` |
| `sandbox().tools()` | `sandbox_exec`, `sandbox_read`, `sandbox_write`, a tool source, for an engine with no sandbox of its own (`agent.tools.use(...)`) |

### stored

`stored(ref, /, **values)` — the gateway's stored prompt `ref` (`"name"`, `"name@3"`), its
`{{variables}}` filled with `values`: a `ReAct`'s `system_prompt`, or `h.prompt(...)`'s
reference ([prompts.md](prompts.md)).

### llm_judge

`llm_judge(criteria, name="llm_judge", *, model=None, sample=0.1)` — a judge model grades an
answer against `criteria`: offline in `h.evaluate`, online in `h.wrap(judges=[...])` on the
`sample` share of runs; `model` its own (else the gateway's judge alias)
([evaluation.md](evaluation.md)).

## Hooks

`Hooks` — subclass and override `on_run_start(run)`, `on_run_end(run, result)`,
`before_tool(call) -> None | Deny | Ask | Rewrite`, `after_tool(call, outcome) -> outcome`,
`on_error(stage, error)`, `on_output(item) -> item` (synchronous: what a user is shown). A model
call is each framework's own (its middleware or callbacks: [hooks.md](hooks.md)); a hook class
defining `before_model` or `after_model` is refused.
`Deny(reason)`, `Ask(question, assignee=None, component=None, props=None)`, `Rewrite(args)`,
`Output(kind, value)` with
`replace(value)`, `OutputKind` (`ANSWER`, `EVENT`) ([hooks.md](hooks.md#mask-what-the-ui-sees)).

## current

`current() -> Runtime | None` — the run the calling code executes in.

## Settings

`Settings(bifrost_url=, bifrost_virtual_key=, api_key=, memory_url=, runs_url=, spool_dir=,
judge_virtual_key=)` — every field optional; `Settings.from_env()` is what `Harness()` reads:
`BIFROST_URL`, `BIFROST_VIRTUAL_KEY`, `TRELLIS_API_KEY`, `MEMORY_URL`, `RUNS_URL`,
`TRELLIS_SPOOL_DIR`, `TRELLIS_JUDGE_VIRTUAL_KEY`. The tools' own variables (`OTEL_*`,
`LANGFUSE_*`, the providers' keys, `DOCKER_HOST`) are read by their own SDKs. Each field, its
variable, default and an example: [configuration.md](configuration.md#settings-and-the-environment).

## The blocks (Way 2)

| Block | Import | Page |
|---|---|---|
| Memory | `from trellis.memory import MemoryClient` | [blocks/memory.md](blocks/memory.md) |
| Runs | `from trellis.runs import RunsClient, Worker` | [blocks/runs.md](blocks/runs.md) |
| Governance | `from trellis.harness.governance import Governance, governed` | [blocks/governance.md](blocks/governance.md) |
| Evaluation | `from trellis.harness.evals import EvalServices, evaluate, judge` | [blocks/evaluation.md](blocks/evaluation.md) |
| A2A client | `from trellis import a2a` (`trellis.harness.a2a.RemoteAgent` under it) | [blocks/a2a.md](blocks/a2a.md) |
| Sandbox | `from trellis.harness.sandbox import DockerSandbox, SandboxSpec` | [sandbox.md](sandbox.md#way-2-without-a-harness) |
| Contracts | `from trellis.contracts import RunStart, Interrupt, InterruptResolution` | [blocks/mixing.md](blocks/mixing.md#the-records-they-share) |
| Gateway | `from bifrost_sdk import Bifrost, NO_GATEWAY_TOOLS` | [gateway.md](gateway.md) |
| Test helpers | `from trellis.testing import Reviewer, Decide` | [interrupts.md](interrupts.md) |
