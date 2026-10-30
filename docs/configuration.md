# Configuration

The harness reads the environment and nothing else — no YAML. A variable says *where* a service
is, *whether* it exists in this deployment and the credentials it is reached with. Unset means
"not in this deployment". Code passes only what code knows — the agent's builder (its model,
instructions and tools), its judges, hooks and budget ([below](#what-code-passes)) — and turns
parts off with one switch, `without=`.

Every variable is in [`.env.example`](../.env.example), which `tests/unit/test_settings.py`
checks lists exactly what is read.

## Where each thing goes

One place for each thing, decided by who owns it and how often it changes — never two
(the README's [short table](../README.md#where-things-go) is this one, cut down).

| Place | Owner | What goes there | Example |
|---|---|---|---|
| the environment | the deployment | where the models and each service are, their keys, the spool: the nine [settings](#settings-and-the-environment) | `LLM_BASE_URL=http://localhost:8091/v1` |
| the tools' own variables | their SDKs | `OTEL_*`, `LANGFUSE_*`, `DOCKER_HOST`, the providers' keys and credentials (without `LLM_BASE_URL`; Google's and AWS's for `LLM_AUTH=google`, `aws`) | `OPENAI_API_KEY=sk-...` |
| Bifrost | the gateway's administrator | model routing, fallbacks, model retries, rate limits, key and team budgets, caching, the MCP allow-list of a key, stored prompts, the judge's model alias | a virtual key allowed the `erp` MCP server |
| the memory service | its administrator, or a deploy script | the tool catalog: each tool's `risk` and `approve_when` rules; learned approvals | `refund`: `approve_when amount > 100` |
| agent-runs | the people and code that schedule | schedules and their ticker, the inbox, webhooks | `await agent.schedule("0 9 * * 1-5", user="ada", input="Digest")` |
| Langfuse | the team | datasets, scores and traces (when no `OTEL_EXPORTER_OTLP_*` is set) | `LANGFUSE_PUBLIC_KEY=pk-...` |
| [`Harness(...)`](api.md#harness) | the application | settings in code instead of the environment, a block of your own, a service turned off | `Harness(config=Settings(...), memory=False)` |
| the builder (`ReAct` or your framework's) | the agent's author | the model, the instructions, the tools and their `call_timeout`, the output schema, sub-agents | `ReAct(system_prompt="...", model="openai/gpt-4.1-mini", tools=[lookup])` |
| [`h.wrap(...)`](api.md#hwrap) | the agent's author | every run of the agent: `hooks`, `judges`, `budget`, `run_timeout`, `without`, `framework_options`, `version`, `description` | `h.wrap(graph, id="triage", judges=[llm_judge("Polite.")], run_timeout=600)` |
| [`agent.run(...)`](api.md#agentrun-agentstream-agentstart) (`stream`, `start`, `schedule`) | the caller | this run: `user`, `thread`, `tenant`, `run_timeout`, `deadline`, `budget`, `without`, `framework_options`, `document_ids`; queued: `priority`, `concurrency_key` | `agent.run(q, user="ada", without={"memory"})` |
| [`h.worker(...)`](api.md#hworker) | the operator | how many runs a worker process takes at once | `h.worker(concurrency=8)` |
| constants | the harness | thresholds, limits, cache lengths ([below](#what-is-not-a-setting)) | `MAX_STEPS` |

When two places could say the same, the narrower wins: a run's over its agent's (`without=` is
added to, `framework_options=` merged key by key, `run_timeout`/`budget` replaced), code over
the environment, the environment over what is detected — and `LLM_BASE_URL` over a provider
key ([below](#where-a-model-name-goes)).

## Settings and the environment

`Harness()` reads `Settings.from_env()`; `Harness(config=Settings(...))` takes the same fields
in code, for tests and embedding (`Settings` is frozen and refuses unknown fields;
`Settings.from_env(environ)` reads a mapping instead of `os.environ`, and blank values count as
unset). **Automatic** says what switches on by itself when the variable is set — nothing else to
call.

Each setting lives in one layer, decided by who owns it and how often it changes:

* **the environment** — what differs per deployment: the seven endpoints, keys and the spool
  (the table below). The tools' own standard variables are read
  under their own names, never mirrored: `OTEL_*` by the OpenTelemetry SDK (the exporter's
  `OTEL_EXPORTER_OTLP_ENDPOINT`, `..._TRACES_ENDPOINT`, `..._HEADERS`; `OTEL_SERVICE_NAME`;
  `OTEL_RESOURCE_ATTRIBUTES`, whose `service.version` is the agents' version — bake it in at
  build time), `LANGFUSE_*`, `DOCKER_HOST`, and — without Bifrost — the providers' own keys
  (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`...), which each provider's SDK
  reads ([below](#where-a-model-name-goes));
* **the code** — the agent's behaviour: the builder (`ReAct` or your framework's: the model, the
  instructions, the tools, the output schema), a budget, `without=`, hooks, judges and their
  share of runs (`llm_judge(sample=)`), the worker's concurrency (`h.worker(concurrency=)`,
  `--concurrency`), the sandbox image (`sandbox(image=)`);
* **the owning services** — Bifrost: model routing, fallbacks, model retries, rate limits, key
  and team budgets, caching, the MCP allow-list on the key, stored prompts, the judge's model
  alias; the memory catalog: approval rules and tool risk; agent-runs: schedules;
* **internal** — thresholds (constants: a fixed tenth of runs gets its grounding score), a
  model's window (its metadata), an OpenTelemetry provider the application installed (kept),
  the tenant (its key's).

Explicit code beats the environment, and the environment beats what is detected. At start the
harness logs one line naming what it is connected to; `trellis doctor` (`python -m
trellis.harness.doctor`) prints every setting, where it came from (secrets redacted), whether
each service answers and takes its key — for the memory service also the key's role (a service
key; an admin key fails the check: the verdicts it sends for people would skip their review),
the agents' model key, its readiness (`/health/ready`) and version (`/version`) — whether the
spool can be written and the Docker daemon answers, the approval rules the memory service
accepted on its own (in effect or revoked), and what to set — it changes nothing.

### Models

#### Where a model name goes

Every model call of every framework is routed one way, chosen from the settings — never a
chain of fallbacks (`trellis.harness.gateways`, [ADR 0004](adr/0004-provider-neutral-models.md)).
`LLM_GATEWAY` names the route — a strategy from one registry —, `LLM_AUTH` how its requests
are authenticated; the settings given in code (`Harness(config=Settings(...))`) route the
runs' models over the environment's, whenever the agent was built:

| Set | Route (`LLM_GATEWAY`) | Model calls go | What else it can do |
|---|---|---|---|
| `LLM_BASE_URL` | `openai-compatible` (unset) | to that OpenAI-compatible endpoint — OpenAI, OpenRouter, a LiteLLM proxy, Ollama, vLLM, llama.cpp, Anthropic's and Gemini's OpenAI-compatible endpoints —, the name as given, authenticated by `LLM_AUTH` (`LLM_API_KEY` as the bearer key) | models only: the client's own retries; a judge by name on the same endpoint |
| `LLM_BASE_URL` | `bifrost` | through Bifrost, with `LLM_API_KEY` as the virtual key, the name as Bifrost's `provider/model` | everything: the MCP tools, Code Mode, the cost of each call (dollar budgets), fallbacks and retries, stored prompts, skills, the judge alias, each call's run as its dimensions, the Claude CLI on its Anthropic route |
| no `LLM_BASE_URL` | `provider-sdk` | to the model's provider, through the provider's own SDK and key — LangChain's `init_chat_model` (`OPENAI_API_KEY` and `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, Google's and AWS's own credentials for Vertex and Bedrock...) | models only: the client's own retries; a judge by name is an OpenAI model (`llm_judge(model="openai:...")`); MCP servers are `mcp(...)` entries of your own ([tools.md](tools.md)) |

A URL route wins over a provider key also set (said once at start, and by `trellis doctor`).
A route of your own is a `ModelGateway` subclass a package registers under the entry-point
group `trellis.gateways` (its name is its `LLM_GATEWAY` value); a feature asks the route
whether it has a capability (`gateway.has(Capability.MCP)`), never which route it is.

`LLM_AUTH` picks bifrost-sdk's auth strategy by name: `bearer` (unset: `LLM_API_KEY`, or a
placeholder for a server that takes none), `google` (a Google OAuth token from the application
default credentials, refreshed when it expires: `pip install 'trellis-harness[vertex]'`) or
`aws` (each request signed with SigV4 from the AWS environment, for Bedrock's
OpenAI-compatible endpoint: `trellis-harness[bedrock]`). A login takes no `LLM_API_KEY`; Bifrost
authenticates with its virtual key only (`bearer`). A combination that fits no route —
`LLM_GATEWAY=bifrost` without `LLM_BASE_URL`, `LLM_API_KEY` with `google`, an unknown name — is
refused when the `Harness` is built (`ConfigurationError`).

A name says its provider — `openai/gpt-4o` or `openai:gpt-4o` — or starts the way its
provider's do (`gpt-`, `o3`, `claude-`, `gemini-`, `mistral-`): each route renders it in its
own form — as given on an OpenAI-compatible endpoint, `openai/gpt-4o` on Bifrost,
`openai:gpt-4o` for LangChain's `init_chat_model` (`gemini:` is `google_genai:`, `vertex:` is
`google_vertexai:`, `bedrock:` is `bedrock_converse:`). The provider SDKs other than OpenAI's
are optional extras: `trellis-harness[anthropic]`, `[gemini]`, `[vertex]`, `[bedrock]`. Per
framework:

| Framework | How its models are routed |
|---|---|
| `ReAct(model="name")` | its chat model routes each call as the harness running it says |
| OpenAI Agents | with Bifrost, an agent's model name (or the SDK's default) through Bifrost's chat completions (`RunConfig.model_provider`), unless the run's own `run_config` names a provider |
| Claude Agent SDK | with Bifrost (`Capability.ANTHROPIC_ROUTE`), the CLI through its Anthropic route (`ANTHROPIC_BASE_URL`, the virtual key), its headers in `ANTHROPIC_CUSTOM_HEADERS`, unless the options' own `env` names a base URL |
| a model object you built (a `ChatOpenAI`, an OpenAI `Model`) | as you built it: with Bifrost set, a call that does not go through it is said once per agent and model |

`trellis doctor` names the route, its auth and where each model's profile comes from, checks
the endpoint against the route declared (a `bifrost` route must answer MCP at the endpoint's
root; an endpoint that does under `openai-compatible` is said to have more to give), names the
provider keys it leaves unused, and what is off on the route. Runnable, one agent on each
route:
[examples/05_features/model_routing.py](../examples/05_features/model_routing.py).

| Variable | `Settings` field | Unset (default) | Example | Automatic when set |
|---|---|---|---|---|
| `LLM_BASE_URL` | `llm_base_url` | no endpoint: model names go to their provider's own SDK; no MCP tools, no stored prompts | `http://localhost:8091/v1` | every model name routed there; told to the memory service per agent (its own model work goes there too) |
| `LLM_API_KEY` | `llm_api_key` | no key sent (a placeholder, for a server that takes none); with Bifrost, the gateway's own policy decides | `vk-support` | the bearer key of every model call; Bifrost's virtual key; sent to the memory service with the endpoint |
| `LLM_GATEWAY` | `llm_gateway` | `openai-compatible` with `LLM_BASE_URL`, `provider-sdk` without | `bifrost` | the route's capabilities: Bifrost's MCP tools, Code Mode, stored prompts, skills, cost, the Anthropic route |
| `LLM_AUTH` | `llm_auth` | `bearer` (`LLM_API_KEY`) | `google` | each model request authenticated by that strategy (a refreshed Google token, AWS SigV4); no secret sent to the memory service |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | `judge_virtual_key` | `LLM_API_KEY` (the agents' budget) | `vk-evals` | the judge's calls on their own key: evaluation's own budget, rate limits and models in Bifrost, so its spend never exhausts production's |
| `TRELLIS_API_KEY` | `api_key` | only allowed without `MEMORY_URL` and `RUNS_URL` (`ConfigurationError` otherwise) | `trk_...` | the tenant: the memory service says who the key is (`GET /v1/keys/self`) |
| `MEMORY_URL` | `memory_url` | memory off: no context, no memory tools, no records, no catalog (the tools' own risks decide) | `http://localhost:8080` | memory push, pull and records; the tool catalog; grounding; documents; feedback in memory |
| `RUNS_URL` | `runs_url` | runs, the queue and schedules kept in process (`LocalRuns`: the last 1,000 finished runs, an hour each), lost on restart | `http://localhost:8090` | durable runs, workers across processes, the ticker's schedules and deadlines, artifacts, the event log (needs `MEMORY_URL`) |
| `TRELLIS_SPOOL_DIR` | `spool_dir` | the XDG state folder (`$XDG_STATE_HOME/trellis`, else `~/.local/state/trellis`) when it can be written and the process is not in a container or a serverless runtime (`K_SERVICE`, `KUBERNETES_SERVICE_HOST`, `AWS_LAMBDA_FUNCTION_NAME`, `/.dockerenv`); there, none: a write the service could not take is lost at a restart (said once) | `/var/spool/trellis` | undelivered writes kept in `<dir>/trellis-writes.jsonl`, replayed at the next start ([memory.md](memory.md#background-writes-what-is-guaranteed)) |

The tools' own variables, read by their own SDKs (not `Settings` fields): `LANGFUSE_PUBLIC_KEY`
and `LANGFUSE_SECRET_KEY` (and `LANGFUSE_BASE_URL`, unset: Langfuse Cloud) — its scores and
datasets, and its traces when no `OTEL_EXPORTER_OTLP_*` variable is set (to
`<LANGFUSE_BASE_URL>/api/public/otel` with `Authorization: Basic base64(pk:sk)` and
`x-langfuse-ingestion-version: 4`, as Langfuse's SDK sends them); `OTEL_*`; `DOCKER_HOST`
(where `sandbox()` makes its containers); the providers' keys.

Retired with no alias (a variable left set is not read; `trellis doctor` lists what it reads):
`PROMPTS_DIR`, `SKILLS_DIR` (stored prompts are the gateway's), `TRELLIS_AGENT_VERSION`
(`OTEL_RESOURCE_ATTRIBUTES=service.version=`, or `h.wrap(version=)`),
`TRELLIS_WORKER_CONCURRENCY` (`h.worker(concurrency=)`), `SANDBOX`, `SANDBOX_IMAGE`
(`sandbox(image=)`: Docker unless a provider is given), `TRELLIS_JUDGE_MODEL`
(`llm_judge(model=)`, or the gateway's judge alias), `TRELLIS_EVAL_SAMPLE`,
`TRELLIS_GROUNDING_SAMPLE`, `TRELLIS_JUDGE_SAMPLE` (a fixed tenth of runs is grounded; each
judge's share is `llm_judge(sample=)`), `LANGFUSE_HOST` (`LANGFUSE_BASE_URL`, read by Langfuse's
own names).

A working local setup (the memory service's and agent-runs' development stacks; the key is the
memory service's development key, which speaks for the tenant `default`):

```bash
export MEMORY_URL=http://localhost:8080 TRELLIS_API_KEY=dev-key
export RUNS_URL=http://localhost:8090
export LLM_BASE_URL=http://localhost:8091/v1 LLM_GATEWAY=bifrost LLM_API_KEY=vk-local
python -m examples.02_way1_react.agent      # the same example, now on the real services
```

The same in code:

```python
from trellis import Harness, Settings

settings = Settings(
    memory_url="http://localhost:8080",
    api_key="dev-key",
    runs_url="http://localhost:8090",
)
h = Harness(config=settings)
```

`RUNS_URL` without `MEMORY_URL`, and `MEMORY_URL` (or `RUNS_URL`) without `TRELLIS_API_KEY`,
are refused when the `Harness` is built (`ConfigurationError`), rather than a `401` at the
first run. A field `Settings` does not have is refused (`ValidationError`). The names are the
platform's: agent-runs reads the memory service at `MEMORY_URL` too, the memory SDK defaults to
`MEMORY_URL` and `TRELLIS_API_KEY`, and the models are `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_GATEWAY` and `LLM_AUTH` everywhere (the memory service reads the same four for its own
settings). The blocks read
the same names: `MemoryClient()` `MEMORY_URL` and `TRELLIS_API_KEY`, `RunsClient()` `RUNS_URL`
and `TRELLIS_API_KEY`, `Governance.from_env()` `MEMORY_URL` and `TRELLIS_API_KEY`,
`EvalServices.from_env()` the Langfuse keys, the `LLM_*` settings and `TRELLIS_JUDGE_VIRTUAL_KEY`. A
team on Langfuse sets only Langfuse's own names (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_BASE_URL` unless it is on Langfuse Cloud): with no OpenTelemetry exporter
variable set, the harness exports its traces there too, so traces, scores and datasets all
reach the one project.

## What is automatic

With the services configured, every wrapped agent gets all of this with no code.

| | |
|---|---|
| **Memory** | With `MEMORY_URL`: the context for the question is pushed into the framework's input (without the recent conversation when the framework keeps the thread itself), the memory tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are added, and the transcript, every tool call and the run's outcome are recorded ([memory.md](memory.md)) |
| **MCP tools** | Every tool the virtual key allows, beside the agent's own `mcp(...)` servers, each executed through the gateway saying who the run is for; a tool the gateway would run itself (Agent Mode) is not offered; no model call gets the gateway's MCP tools ([gateway.md](gateway.md)) |
| **Governance** | Each call runs, runs and is announced, or asks a person: by the tool's risk — the MCP server's annotations or a local tool's declaration, overridden by the catalog's `risk`; the catalog's `approve_when` replaces that; a catalog that cannot be read fails closed; every tool is published to the catalog ([governance.md](governance.md)) |
| **Tool hints** | The context comes back with the skills the agent learned for the task; from 5 tools on, also with the tools that fit, and the model is offered those, the memory tools and every tool the run already used ([tools.md](tools.md)) |
| **Tool set check** | When an agent's tools change, what makes them hard for a model to choose among — a missing or short description, an undescribed parameter, a name the OpenAI API refuses, look-alike names or descriptions, 20 or more tools at once — is logged as a warning (`trellis.tools`); nothing is refused ([tools.md](tools.md#tool-choice-what-helps-the-model-pick-the-right-tool)) |
| **Code Mode** | The read-only Code Mode servers, from 3 servers or 20 tools, become Code Mode meta-tools under the harness's names; their nested calls are recorded from the gateway's log |
| **Stored prompts** | A stored prompt (`stored(...)`, `h.prompt(...)`) is pinned and journaled per run: a resumed run reads the version it started with ([prompts.md](prompts.md)) |
| **Outcome and grounding** | The run's ending is its `system` feedback; a tenth of the runs given a context, chosen by the run id, is checked against it (`/v1/verify`), the judge's verdict outranking the ending and its score on the run's trace |
| **People** | A pause is delivered to agent-runs' webhooks (`run.paused`), with the question and whose it is ([interrupts.md](interrupts.md#telling-people)) |
| **Run events and queues** | With `RUNS_URL` every run's events go to agent-runs' event log, so any replica streams any run; a second message to a busy conversation waits for the first ([runs.md](runs.md)) |
| **Sub-agents** | A wrapped agent another calls — a `ReAct`'s `subagents=`, a nested `agent.run` inside a run, a framework's own helper (an OpenAI Agents `Agent.as_tool`, a Deep Agents `task`, a Claude subagent) — is a child run: tenant, user, thread, time and trace inherited, its pauses answered through the parent ([subagents.md](subagents.md)) |
| **Sandboxes** | Made at a run's first call, recorded, attached again after a pause or a crash, paused while a person answers, deleted at its end; leftovers reaped ([sandbox.md](sandbox.md)) |
| **Reliability** | Every tool call bounded by its `call_timeout` and the run's; reads retried; writes run once with an idempotency key; a write of unknown outcome never run blind again; a stopping worker hands its runs back ([reliability.md](reliability.md)) |
| **Traces** | OTel GenAI spans with Langfuse's trace attributes, every attempt of a run in one trace, credentials masked ([observability.md](observability.md#redaction)) |
| **Version check** | A framework outside the range this release was tested with is said once in the log when its first target is wrapped ([versioning.md](versioning.md)) |

## What is on, and how to turn it off

Everything the deployment configures is on for every agent. One switch turns parts off,
`without=` (`trellis.harness.features.Feature`): on [`h.wrap`](api.md#hwrap) for every run of the
agent; on [`agent.run`, `stream`, `start`](api.md#agentrun-agentstream-agentstart) and
[`schedule`](api.md#agentschedule) for that run, on top of the agent's — kept with the run's
record, so its resume, the worker that continues it and its sub-agents' runs are without them
too.

```python
agent = h.wrap(graph, id="triage", without={"judges"})  # every run of this agent
await agent.run(question, user="ada", without={"memory"})  # this run: no memory at all
```

| Feature (`without=` name) | On when | What it is | Turned off |
|---|---|---|---|
| `memory` | `MEMORY_URL` | the memory context pushed into the framework's input, the memory tools, the records (transcript, tool calls, outcome, decisions as feedback) and the tool narrowing | the run has no memory scope: no context, no memory tools, nothing recorded, `trellis.current().memory` refused |
| `tool_search` | `MEMORY_URL`, from 5 tools | the one owner of tool narrowing: the memory service's tool hints narrow the tools the model is offered, and its `tool_search` tool widens them | every tool offered (the learned skills stay) |
| `mcp` | `LLM_GATEWAY=bifrost`, or an `mcp(...)` entry | the MCP tools the virtual key allows and the agent's own `mcp(...)` servers, Code Mode included | no MCP tools: the gateway is not asked for them and no server is started |
| `code_mode` | `LLM_GATEWAY=bifrost`, enough read-only Code Mode servers | their tools behind Bifrost's Code Mode meta-tools (one script instead of many calls) | those servers' tools offered one by one |
| `judges` | `h.wrap(judges=[...])`, each on its share of runs (`llm_judge(sample=)`) | the online judges | not judged |
| `grounding` | `MEMORY_URL`, a fixed tenth of the runs given a context | the grounding check (`/v1/verify`) and its score on the run's trace | not checked |

These six, and only these, are names: any other is refused (`ConfigurationError`, naming
them). `without={"memory"}` is per agent or per run: the rest of the deployment keeps memory,
and the tenant, `h.add_document` and `h.feedback` keep working. `Harness(memory=False)` is the
whole deployment with no memory service: no client even with `MEMORY_URL` set, the local
tenant, no `h.add_document`, and `RUNS_URL` refused ([api.md](api.md#harness)). A `ReAct`'s long-task tools (`write_todos`, scratch files) are its own, offered on a
long task with room in the window ([react.md](frameworks/react.md#the-middleware)). Not switchable, because they are correctness or have an owner elsewhere:
governance and approvals, the journal and replay, tool retries and time limits, tracing and
redaction, the run record, a budget (opt-in: give one), structured output, model retries,
fallbacks and caching (the gateway's). Inside a run, `trellis.current().uses(feature)` says
whether a part is on. Runnable: [examples/05_features/without_combinations.py](../examples/05_features/without_combinations.py)
(each combination, and what it leaves on), [examples/02_way1_react/agent.py](../examples/02_way1_react/agent.py),
[examples/06_scenarios/scheduled_run_selection_limit.py](../examples/06_scenarios/scheduled_run_selection_limit.py).

## The framework's own run options

`framework_options=` hands the framework's own options to its run call, unchanged: on
[`h.wrap`](api.md#hwrap) for every run of the agent (`serve_chat`, `serve_a2a` and `h.evaluate`
have these), on [`agent.run`, `stream`, `start`](api.md#agentrun-agentstream-agentstart) and
[`schedule`](api.md#agentschedule) for that run (each run a schedule fires) — merged over the
agent's, key by key: a mapping both give (a graph's `configurable`, `metadata`) is merged the
same way, and any other value the run gives replaces the agent's.

```python
agent = h.wrap(graph, id="support", framework_options={"recursion_limit": 50})
await agent.run(question, user="ada", framework_options={"tags": ["vip"]})
```

| Target | Where they go | The harness keeps for itself |
|---|---|---|
| LangGraph, Deep Agents, `ReAct` | the `config` of `ainvoke`/`astream` (`RunnableConfig` keys: `recursion_limit`, `configurable`, `tags`, `metadata`, `callbacks`, `max_concurrency`, `run_name`) | `configurable.thread_id`: refused (the thread is `run(thread=)`) |
| OpenAI Agents | keyword arguments of `Runner.run`/`run_streamed` (`max_turns`, `run_config`, `context`, `session`...) | `starting_agent`, `input`, `hooks`: refused |
| Claude Agent SDK | `ClaudeAgentOptions` fields set on the run's copy of the options (`max_turns`, `model`, `allowed_tools`...) | `resume`, `continue_conversation`, `permission_prompt_tool_name`: refused; `system_prompt`, `can_use_tool` and `mcp_servers` merged (the context appended, the callback asked after governance, the `trellis` server added beside — a server of that name refused); `env` routed through Bifrost's Anthropic route when it serves the harness, its headers added to `ANTHROPIC_CUSTOM_HEADERS` |
| a function | — (no framework run call) | any: refused |

An option the framework cannot take (a key it does not know, one the harness keeps) is refused
when the agent is wrapped or the run is called (`ConfigurationError`, naming it), never in the
middle of a run. A run keeps its own options with its record (`RunStart.metadata`, as
`without=`; a schedule's in its `ScheduleSpec.metadata`, copied into each run it fires), so its
resume and a worker that continues it run with the same: a queued or scheduled run's must be
JSON, and an object (a `RunConfig`, a callback) is refused there — give it on `h.wrap`, which
every worker's agent has; on `run`/`stream` an object reaches that call only, and the record
keeps the JSON ones.

## What code passes

The arguments, each linked to its reference: what they are, their defaults, what they accept.

| Where | Arguments |
|---|---|
| [`Harness(...)`](api.md#harness) | `config`, `memory`, `runs`, `gateway` (a client given is used as it is; `False` turns the service off whatever the environment says: [docs/README.md](README.md#composition-a-harness-is-the-blocks-you-give-it)) |
| [`ReAct(...)`](api.md#react) | must: `system_prompt`, `model`; usual: `tools`, `response_format`; optional: `subagents`, `skills`, `memory_files`, `backend`, `permissions`, `max_steps`, `call_timeout`, `effort`; advanced: `middleware`, `state_schema`, `context_schema`, `cache`, `context_window`, `max_repeats` |
| [`h.wrap(...)`](api.md#hwrap) | `id`, `version`, `description`, `hooks`, `judges`, `budget`, `run_timeout`, `without`, `framework_options` |
| [`agent.run` / `stream` / `start`](api.md#agentrun-agentstream-agentstart) | `user`, `thread`, `tenant`, `run_timeout`, `deadline`, `budget`, `without`, `framework_options`, `document_ids`; `start`: `priority`, `concurrency_key` |
| [`agent.schedule(...)`](api.md#agentschedule) | `cron`, `user`, `input`, `tz`, `tenant`, `run_timeout`, `budget`, `without`, `framework_options`, `document_ids`, `priority`, `concurrency_key` |
| [`agent.resume(...)`](api.md#agentresume) | `decision`, `answer`, `reviewer`, `comment`, `remember`, `tenant` |
| [`h.worker(...)`](api.md#hworker) | `agents`, `concurrency` |
| [`tool(...)`](api.md#tool), [`mcp(...)`](api.md#mcp-stdio), [`a2a(...)`](api.md#a2a), [`openapi(...)`](api.md#openapi), [`sandbox(...)`](api.md#sandbox) | `side_effects`, `idempotent`, `call_timeout`, ... |

## Who the deployment is

Not configured: the memory service says it about `TRELLIS_API_KEY` (`GET /v1/keys/self` →
`{key_id, tenant_id, principal, role, may_act_as}`), asked at first use and again every 10
minutes. While the service cannot be reached the last answer stands (asked again after 30 s),
so an outage does not stop runs; a key the service refuses (`401`/`403`), or one it could never
be asked about (the first ask failed), raises `ConfigurationError` with what went wrong.

* **Tenant** — the key's own. A platform key (no tenant) names one per call (`tenant=` on
  `run`/`stream`/`start`/`schedule`); a tenant key refuses any other. A development key (the
  memory service's `trusted_dev` mode) speaks for `default`, and without a memory service the
  tenant is `default` too — so local development needs no `tenant=`.
* **What it may write** — not the key's role: the memory service decides per scope (its
  relationship checks) and a refused write is a reported warning.
* **Memory model endpoint** — with `LLM_BASE_URL`, each agent tells the memory service its model
  endpoint once per process (idempotently): the route's `trellis.contracts.Endpoint` — the
  base URL, the route and auth names, the agent's model as the route names it, its LangChain
  profile (reasoning, temperature, output tokens, tool calling) and `LLM_API_KEY` when it is a
  bearer key (a Google or AWS login sends no secret). The memory service's LLM work for the
  agent then goes where the agent's does: configure the harness only, memory follows. Without
  `LLM_BASE_URL` nothing is told and the service uses its own `LLM_*` settings; an operator can
  make it always do so (`models.llm.accept_agent_endpoints: false`). The service checks the
  endpoint answers before it keeps it, so an endpoint only the harness's machine reaches
  (`localhost`) is refused, said once in the log. A memory service that takes no endpoints is
  logged once per process and not asked again.

## What is not a setting

Limits, retries, the tool-hint threshold, lease length and the like are named constants next to
the code that uses it (`MAX_STEPS`, `TOOLS_TTL_SECONDS`, `GOVERNANCE_TTL_SECONDS`,
`WRITE_ATTEMPTS`...). The timeouts only the agent's author knows are arguments where the thing
is defined: a tool call's `@tool(call_timeout=)`, `openapi(call_timeout=)`,
`a2a(call_timeout=)`; a model call's `ReAct(call_timeout=)` (each model call, not its tools);
a run's `h.wrap(run_timeout=)` and `agent.run(run_timeout=, deadline=)` — `run_timeout` working
time (a pause for a person and time queued not counted), `deadline` clock time (both counted)
([reliability.md](reliability.md)). Which framework versions a release supports is a release's
too ([versioning.md](versioning.md)).
