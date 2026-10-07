# Configuration

The harness reads the environment and nothing else — no YAML. A variable says *where* a service
is, *whether* it exists in this deployment, the credentials it is reached with, and the few
costs the deployment chooses (how much traffic is checked, by which model). Unset means "not in
this deployment". Code passes only what code knows — the blocks it owns, its prompt and skill
sources, its judges and hooks ([below](#what-code-passes)) — and turns parts off with one switch,
`without=`.

Every variable is in [`.env.example`](../.env.example), which `tests/unit/test_settings.py`
checks lists exactly what is read.

## Settings and the environment

`Harness()` reads `Settings.from_env()`; `Harness(config=Settings(...))` takes the same fields
in code, for tests and embedding (`Settings` is frozen and refuses unknown fields;
`Settings.from_env(environ)` reads a mapping instead of `os.environ`, and blank values count as
unset). **Automatic** says what switches on by itself when the variable is set — nothing else to
call.

| Variable | `Settings` field | Unset (default) | Example | Automatic when set |
|---|---|---|---|---|
| `BIFROST_URL` | `bifrost_url` | no gateway: no MCP tools, no `ReAct` model names, no gateway prompts or skills | `http://localhost:8091/v1` | the MCP tools the key allows, Code Mode, gateway prompts and skills as sources, `ReAct(model="name")` |
| `BIFROST_VIRTUAL_KEY` | `bifrost_virtual_key` | gateway calls without a key (the gateway's own policy decides); no memory model key | `vk-support` | registered once per process as each agent's model key in the memory service |
| `TRELLIS_API_KEY` | `api_key` | only allowed without `MEMORY_URL` and `RUNS_URL` (`ConfigurationError` otherwise) | `trk_...` | the tenant: the memory service says who the key is (`GET /v1/keys/self`) |
| `MEMORY_URL` | `memory_url` | memory off: no context, no memory tools, no records, no catalog (the tools' own risks decide) | `http://localhost:8080` | memory push, pull and records; the tool catalog; grounding; documents; feedback in memory |
| `RUNS_URL` | `runs_url` | runs, the queue and schedules kept in process (`LocalRuns`), lost on restart | `http://localhost:8090` | durable runs, workers across processes, the ticker's schedules and deadlines, artifacts, the event log (needs `MEMORY_URL`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `otlp_endpoint` | no export (an application's own provider is still used) | `https://cloud.langfuse.com/api/public/otel` | an OTLP exporter, unless the application installed a provider |
| `OTEL_EXPORTER_OTLP_HEADERS` | `otlp_headers` | no headers; no Langfuse scores API | `Authorization=Basic%20<base64 pk:sk>` | Langfuse's scores, datasets and dataset runs (parsed as the OTel spec writes it: `k1=v1,k2=v2`, URL-decoded, keys lower-cased) |
| `TRELLIS_SPOOL_DIR` | `spool_dir` | a memory write this process cannot deliver is logged, counted and lost | `/var/spool/trellis` | undelivered writes kept in `<dir>/trellis-writes.jsonl`, replayed at the next start ([memory.md](memory.md#background-writes-what-is-guaranteed)) |
| `TRELLIS_WORKER_CONCURRENCY` | `worker_concurrency` | the CPU count, from 1 to 8 | `4` | runs one worker process executes at once (`h.worker(concurrency=)` and `--concurrency` win) |
| `TRELLIS_AGENT_VERSION` | `agent_version` | runs carry no version unless `h.wrap(version=)` names one | `2026.10.6-a1b2c3` | recorded with every run (`RunStart.agent_version`) and on its spans; a resume on another version warns ([reliability.md](reliability.md#agent-version)) |
| `TRELLIS_GROUNDING_SAMPLE` | `grounding_sample` | `0.1` | `0.25` | that share of successful runs checked against their memory context (`/v1/verify`), by run id; `0` off, `1` every run |
| `TRELLIS_JUDGE_MODEL` | `judge_model` | the gateway model a `ReAct` was built with (logged once), else no judge score | a Bifrost model name | `llm_judge` asks it, unless a judge names its own; a different, stronger model than the agent's ([evaluation.md](evaluation.md#the-judges-model)) |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | `judge_virtual_key` | `BIFROST_VIRTUAL_KEY` (the agents' budget) | `vk-evals` | the judge's calls on their own budget |
| `TRELLIS_JUDGE_SAMPLE` | `judge_sample` | `0.1` when the harness has online judges | `1` | that share of successful runs scored by `Harness(judges=[...])`, by run id |
| `PROMPTS_DIR` | `prompts_dir` | no folder of prompts | `./prompts` | `<name>.md` files there are a prompt source, unless the code passes `Harness(prompts=)` ([prompts.md](prompts.md)) |
| `SKILLS_DIR` | `skills_dir` | no folder of skills | `./skills` | `<name>/SKILL.md` folders there are a skill source, unless the code passes `Harness(skills=)` ([skills.md](skills.md)) |
| `LANGFUSE_HOST` | `langfuse_host` | Langfuse Cloud (`https://cloud.langfuse.com`), when the keys are set | `https://langfuse.example.com` | where Langfuse's prompt management is read |
| `LANGFUSE_PUBLIC_KEY` | `langfuse_public_key` | prompts are not read from Langfuse | `pk-lf-...` | with the secret key: Langfuse's prompts are a source, after `PROMPTS_DIR` and before the gateway |
| `LANGFUSE_SECRET_KEY` | `langfuse_secret_key` | as above | `sk-lf-...` | as above |
| `SANDBOX` | `sandbox` | `sandbox()` given no provider has none: its tools tell the model so | `docker` | each run's sandbox is a container of the Docker daemon on this machine ([sandbox.md](sandbox.md)); any other value is refused |
| `SANDBOX_IMAGE` | `sandbox_image` | the provider's own image (`python:3.12-slim` for Docker) | `python:3.12-slim` | the image those sandboxes are made from, unless a `SandboxSpec(image=)` names one |

A working local setup (the memory service's and agent-runs' development stacks; the key is the
memory service's development key, which speaks for the tenant `default`):

```bash
export MEMORY_URL=http://localhost:8080 TRELLIS_API_KEY=dev-key
export RUNS_URL=http://localhost:8090
export BIFROST_URL=http://localhost:8091/v1 BIFROST_VIRTUAL_KEY=vk-local
python -m examples.02_way1_react.agent      # the same example, now on the real services
```

The same in code:

```python
from trellis import Harness, Settings

settings = Settings(
    memory_url="http://localhost:8080",
    api_key="dev-key",
    runs_url="http://localhost:8090",
    grounding_sample=0.25,
    worker_concurrency=4,
)
h = Harness(config=settings)
```

`RUNS_URL` without `MEMORY_URL`, and `MEMORY_URL` (or `RUNS_URL`) without `TRELLIS_API_KEY`,
are refused when the `Harness` is built (`ConfigurationError`), rather than a `401` at the
first run. A sample outside 0 to 1, a concurrency under 1, a version over 128 characters or a
`SANDBOX` other than `docker` is refused by `Settings` (`ValidationError`). The names are the
platform's: agent-runs reads the memory service at `MEMORY_URL` too, the memory SDK defaults to
`MEMORY_URL` and `TRELLIS_API_KEY`, and the gateway is `BIFROST_URL` everywhere. The blocks read
the same names: `MemoryClient()` `MEMORY_URL` and `TRELLIS_API_KEY`, `RunsClient()` `RUNS_URL`
and `TRELLIS_API_KEY`, `Governance.from_env()` `MEMORY_URL` and `TRELLIS_API_KEY`,
`EvalServices.from_env()` the OTLP variables, `BIFROST_URL` and the `TRELLIS_JUDGE_*` ones.

## What is automatic

With the services configured, every wrapped agent gets all of this with no code.

| | |
|---|---|
| **Memory** | With `MEMORY_URL`: the context for the question is pushed into the framework's input (without the recent conversation when the framework keeps the thread itself), the memory tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are added, and the transcript, every tool call and the run's outcome are recorded ([memory.md](memory.md)) |
| **MCP tools** | Every tool the virtual key allows (`mcp=`: those of the named Virtual MCPs), each executed through the gateway saying who the run is for; a tool the gateway would run itself (Agent Mode) is not offered; no model call gets the gateway's MCP tools ([gateway.md](gateway.md)) |
| **Governance** | Each call runs, runs and is announced, or asks a person: by the tool's risk — the MCP server's annotations or a local tool's declaration, overridden by the catalog's `risk`; the catalog's `approve_when` replaces that; a catalog that cannot be read fails closed; every tool is published to the catalog ([governance.md](governance.md)) |
| **Tool hints** | The context comes back with the skills the agent learned for the task; from 5 tools on, also with the tools that fit, and the model is offered those, the memory tools and every tool the run already used ([tools.md](tools.md)) |
| **Code Mode** | The read-only Code Mode servers, from 3 servers or 20 tools, become Code Mode meta-tools under the harness's names; their nested calls are recorded from the gateway's log |
| **Prompts and skills** | A prompt or skill named once is looked up in code, the folder, Langfuse, the gateway; pinned and journaled per run ([prompts.md](prompts.md), [skills.md](skills.md)) |
| **Outcome and grounding** | The run's ending is its `system` feedback; a sampled share is checked against its context, a score on its trace |
| **People** | A pause is delivered to agent-runs' webhooks (`run.paused`), with the question and whose it is ([interrupts.md](interrupts.md#telling-people)) |
| **Run events and queues** | With `RUNS_URL` every run's events go to agent-runs' event log, so any replica streams any run; a second message to a busy conversation waits for the first ([runs.md](runs.md)) |
| **Sub-agents** | A wrapped agent in another's tools is a child run: tenant, user, thread, time and trace inherited, its pauses answered through the parent ([subagents.md](subagents.md)) |
| **Sandboxes** | Made at a run's first call, recorded, attached again after a pause or a crash, paused while a person answers, deleted at its end; leftovers reaped ([sandbox.md](sandbox.md)) |
| **Reliability** | Every tool call bounded by its `timeout` and the run's; reads retried; writes run once with an idempotency key; a write of unknown outcome never run blind again; a stopping worker hands its runs back ([reliability.md](reliability.md)) |
| **Traces** | OTel GenAI spans with Langfuse's trace attributes, every attempt of a run in one trace, every attribute redacted ([observability.md](observability.md)) |
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
| `memory` | `MEMORY_URL` | `memory_push`, `memory_pull` and `records` together | the run has no memory scope: no context, no memory tools, nothing recorded, `trellis.current().memory` refused |
| `memory_push` | `MEMORY_URL` | the memory context pushed into the framework's input (and the tool hints with it) | no context, no `/v1/context` call |
| `memory_pull` | `MEMORY_URL` | the memory tools (`memory_search`, `tool_search`, ...) | not offered; a graph's, bound at build, are hidden by its `ModelHooks()` middleware (a `ReAct`'s included) and otherwise offered and answer that they are off; `h.tools(without=)` leaves them out |
| `records` | `MEMORY_URL` | the transcript, every tool call, the outcome, decisions as feedback | nothing written to memory about the run |
| `hints` | `MEMORY_URL`, from 5 tools | the tool hints narrow the tools the model is offered | every tool offered (the learned skills stay) |
| `grounding` | `MEMORY_URL`, a sampled share (`TRELLIS_GROUNDING_SAMPLE`) | the answer checked against the context it was given | not checked |
| `judges` | `Harness(judges=[...])`, a sampled share (`TRELLIS_JUDGE_SAMPLE`) | the online judges | not judged |
| `mcp` | `BIFROST_URL` | the MCP tools the virtual key allows (or those of `mcp=`'s Virtual MCPs), Code Mode included | no MCP tools, and the gateway is not asked for them (`h.tools(without={"mcp"})` for a graph's; `mcp=[]` on `wrap` or `h.tools` says the same for every run of the agent) |
| `code_mode` | `BIFROST_URL`, enough read-only Code Mode servers | their tools behind Bifrost's Code Mode meta-tools (one script instead of many calls) | those servers' tools offered one by one |
| `skills` | `skills=` / `skills(...)` | the skills' section in the context and `load_skill`, `read_skill_file` | neither |

A name not in the table is refused (`ConfigurationError`, naming them). Not switchable, because
they are automatic and deterministic: governance and approvals, the journal and replay, retries
and time limits, tracing and redaction, the run record. Inside a run,
`trellis.current().uses(feature)` says whether a feature (a row's name, not `memory`, which is
three) is on. Runnable: [examples/02_way1_react/agent.py](../examples/02_way1_react/agent.py),
[examples/06_scenarios/scheduled_run_selection_limit.py](../examples/06_scenarios/scheduled_run_selection_limit.py).

## The framework's own run options

`framework_options=` hands the framework's own options to its run call, unchanged: on
[`h.wrap`](api.md#hwrap) for every run of the agent (`serve_chat`, `serve_a2a` and `h.evaluate`
have these), on [`agent.run`, `stream`, `start`](api.md#agentrun-agentstream-agentstart) and
[`schedule`](api.md#agentschedule) for that run (each run a schedule fires) — merged over the
agent's, key by key (a key the run gives replaces the agent's, a `configurable` dict included).

```python
agent = h.wrap(graph, id="support", framework_options={"recursion_limit": 50})
await agent.run(question, user="ada", framework_options={"tags": ["vip"]})
```

| Target | Where they go | The harness keeps for itself |
|---|---|---|
| LangGraph, Deep Agents, `ReAct` | the `config` of `ainvoke`/`astream` (`RunnableConfig` keys: `recursion_limit`, `configurable`, `tags`, `metadata`, `callbacks`, `max_concurrency`, `run_name`) | `configurable.thread_id`: the run's thread wins over one given |
| OpenAI Agents | keyword arguments of `Runner.run`/`run_streamed` (`max_turns`, `run_config`, `context`, `session`...) | `starting_agent`, `input`, `hooks`: refused |
| Claude Agent SDK | `ClaudeAgentOptions` fields set on the run's copy of the options (`max_turns`, `model`, `allowed_tools`...) | `resume`, `continue_conversation`, `permission_prompt_tool_name`: refused; `system_prompt`, `can_use_tool` and `mcp_servers` merged (the context appended, the callback asked after governance, the `trellis` server added beside — a server of that name refused) |
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
| [`Harness(...)`](api.md#harness) | `config`, `runs`, `memory`, `gateway`, `governance` (a block given is used as it is; `False` turns it off whatever the environment says: [docs/README.md](README.md#composition-a-harness-is-the-blocks-you-give-it)), `prompts`, `skills`, `judges`, `hooks` |
| [`h.wrap(...)`](api.md#hwrap) | `id`, `tools`, `version`, `mcp`, `skills`, `timeout`, `without`, `hooks`, `framework_options` |
| [`agent.run` / `stream` / `start`](api.md#agentrun-agentstream-agentstart) | `user`, `thread`, `tenant`, `timeout`, `deadline`, `without`, `framework_options`; `start`: `priority`, `concurrency_key` |
| [`agent.schedule(...)`](api.md#agentschedule) | `cron`, `input`, `on_behalf_of`, `tz`, `tenant`, `timeout`, `without`, `framework_options`, `priority`, `concurrency_key` |
| [`agent.resume(...)`](api.md#agentresume) | `decision`, `answer`, `reviewer`, `comment`, `remember`, `tenant` |
| [`h.worker(...)`](api.md#hworker) | `agents`, `concurrency` |
| [`ReAct(...)`](api.md#react) | `system`, `model`, `output`, `max_steps`, `max_repeats`, `model_timeout`, `context_window`, `prompt`, `prompt_vars`, `middleware`, `checkpointer` |
| [`tool(...)`](api.md#tool), [`a2a(...)`](api.md#a2a), [`openapi(...)`](api.md#openapi), [`sandbox(...)`](api.md#sandbox) | `side_effects`, `idempotent`, `timeout`, ... |

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
* **Memory model key** — `BIFROST_VIRTUAL_KEY` is registered as each agent's model key once per
  process (idempotently): the memory service's LLM work for the agent runs on it. A memory
  service that takes no model keys is logged once per process and not asked again.

## What is not a setting

Limits, retries, the tool-hint threshold, lease length and the like are named constants next to
the code that uses it (`MAX_STEPS`, `TOOLS_TTL_SECONDS`, `GOVERNANCE_TTL_SECONDS`,
`WRITE_ATTEMPTS`...). The timeouts only the agent's author knows are arguments where the thing
is defined: `@tool(timeout=)`, `openapi(timeout=)`, `a2a(timeout=)`,
`ReAct(model_timeout=)`, `h.wrap(timeout=)`, `agent.run(timeout=, deadline=)`
([reliability.md](reliability.md)). Which framework versions a release supports is a release's
too ([versioning.md](versioning.md)).
