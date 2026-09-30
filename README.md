# trellis-harness

Attach memory, tools, approvals, durable runs and tracing to an agent you already built — a
LangGraph graph (Deep Agents included), an OpenAI Agents SDK `Agent`, a Claude Agent SDK
`ClaudeAgentOptions`, the harness's own `ReAct` loop, or a plain async function. The framework
is not modified and not re-implemented: the harness wraps the object you built and speaks to
it through the framework's public API.

```python
from trellis import Harness

h = Harness()                               # the deployment is the environment
agent = h.wrap(graph, id="procurement")     # nothing else to configure

result = await agent.run("Reorder SKU-1 if low", user="ada")
if result.interrupt:                        # a person has to approve something
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
```

Everything else is automatic: memory is on when the deployment has a memory service; the MCP
tools are the ones the agent's Bifrost virtual key allows; which calls run, which are
announced and which wait for a person follows from the tools' own annotations and the tool
catalog; the tool schemas the model sees are narrowed to what fits the task; the tenant and
whether the agent may write memory come from the key.

## Install

One distribution; each framework is an extra (the core imports none of them).

```bash
pip install 'trellis-harness[langgraph]'          # LangGraph
pip install 'trellis-harness[deepagents]'         # Deep Agents (brings langgraph)
pip install 'trellis-harness[openai-agents]'      # OpenAI Agents SDK
pip install 'trellis-harness[claude-agent-sdk]'   # Claude Agent SDK
pip install 'trellis-harness[agui]'               # serve_chat (FastAPI)
pip install 'trellis-harness[a2a]'                # serve_a2a and a2a() tools
pip install 'trellis-harness[otel]'               # OTLP export (Langfuse, a collector)
pip install 'trellis-harness[all]'
```

`trellis` is shared with `trellis-contracts` (`trellis.contracts`) and `trellis-memory`
(`trellis.memory`); importing either does not load the harness.

## Configuration

The environment, and nothing else ([`.env.example`](.env.example)):

| Variable | |
|---|---|
| `BIFROST_URL` | Bifrost's `/v1` base: the MCP tools, `ReAct` model names |
| `BIFROST_VIRTUAL_KEY` | the agent's virtual key: its models, its MCP tools, its budget — and, registered automatically, the key the memory service's LLM work for the agent is billed to |
| `TRELLIS_API_KEY` | the one Trellis key, for the memory service and agent-runs; its tenant and role are asked of the memory service (`GET /v1/keys/self`) |
| `MEMORY_URL` | the memory service; memory is on exactly when it is set (read-only when the key's role is) |
| `RUNS_URL` | agent-runs; unset keeps runs, the queue and schedules in process |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` | OTLP traces: Langfuse's endpoint, or a collector ([deploy/otel-collector.yaml](deploy/otel-collector.yaml)) |

## The API

Everything public is importable from `trellis`:

| Name | What it is |
|---|---|
| `Harness(config=None)` | Reads the environment; `config=Settings(...)` instead of it. `async with` closes it. |
| `h.wrap(target, *, id, tools=()) -> Agent` | Attach the harness. The framework is detected from the target's type. `tools` are the agent's own, run in this process: functions, `a2a(url)`, `openapi(spec)`. |
| `await h.tools(*sources, framework=...)` | The toolbox as the framework's own tools, for an agent built with them before wrapping (a compiled LangGraph graph binds its tools): LangChain tools (`"langgraph"`), `FunctionTool`s (`"openai-agents"`), one in-process MCP server (`"claude-agent-sdk"`). It holds `sources`, the MCP tools and the memory tools. |
| `h.worker(agents, *, concurrency=4)` | Claims queued runs of these agents and executes them: `await worker.run()` (until cancelled) or `await worker.run_once()`. |
| `await h.inbox(assignee=None) -> list[RunSummary]` | The paused runs waiting on `assignee` (`user:…`, `role:…`) or on anyone, newest first. |
| `await h.feedback(run_id, verdict, correction=None)` | What a person said about a run: a score on its trace (Langfuse) and the run's `human` feedback in the memory service. |
| `tool(fn)` / `@tool(name=, description=, side_effects=)` | A Python function as a tool (`side_effects`: `"read"`, `"write"` (default), `"irreversible"`). A bare function in `tools=[...]` is `tool(fn)`. |
| `a2a(url, *, name=None)` | A remote A2A agent as one tool. |
| `openapi(spec, *, only=None, base_url=None, headers=None)` | The operations of an OpenAPI 3 document as tools. |
| `ReAct(system, model, output=None, max_steps=12)` | A tool-calling loop over chat completions, for teams with no framework. |
| `current() -> Runtime \| None` | Inside a tool or a node: the run it executes in. |

`Agent` — what `wrap` returns:

| Method | |
|---|---|
| `await run(input, *, user, thread=None, tenant=None) -> Result` | Run to the end or the first pause. `tenant` only for a platform key (one with no tenant of its own). |
| `stream(...) -> AsyncIterator[RunEvent]` | The run's events (contracts `RunEvent`s) up to `RUN_FINISHED`. Closing it early cancels the run. |
| `await start(...) -> RunHandle` | Queue the run for a worker; `handle.status()`, `await handle.result(timeout=)`. |
| `await resume(interrupt_id, decision, *, answer=None, reviewer) -> Result` | Answer the pause (`answer`, `approve`, `reject`, `edit` with the edited arguments as `answer`, `cancel`). A run started in process continues here; a queued run goes back to the queue. |
| `await schedule(cron, input, *, on_behalf_of, tz="UTC") -> Schedule` | Queue a run on a cadence, acting for `on_behalf_of`. The same agent, person, cadence and input are one schedule. |
| `serve_chat(app, *, path="/agui", identity=None)` | AG-UI routes on a FastAPI app (run, reconnect/replay, artifacts). |
| `serve_a2a(app, url, *, identity=None)` | The agent card and A2A JSON-RPC routes at `url`. |

`Runtime` — `trellis.current()` inside a tool or a node, and the second argument of a function
target: `run_id`, `agent_id`, `user`, `thread`, `tenant`, `context` (the pushed memory
context), `memory` (the memory SDK's verbs in the run's scope), `tools.call(name, **args)`,
`tools.hints(task)`, `await ask(question, *, expects, table, diff, options, assignee, deadline,
escalate_to)` (pauses the run; returns the answer on resume) and `log(message, **fields)`.

`Result`: `run_id`, `status` (`SUCCESS`, `PAUSED`, `ERROR`, `QUEUED`, `CANCELLED`), `answer`,
`interrupt`, `error`. `RunSummary`: `run_id`, `agent_id`, `status`, `awaiting`, `assignee`,
`deadline`, `updated_at`.

Run a worker for every agent a module's harness wraps:

```bash
python -m trellis.worker app.agents:h
```

## What happens automatically

| | |
|---|---|
| **Memory** | With `MEMORY_URL`: the context for the question is pushed into the framework's input (without the recent conversation when the framework keeps the thread itself), the memory tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are added, and the transcript, every tool call and the run's outcome are recorded. A read-only key gets the reading tools and records nothing. `BIFROST_VIRTUAL_KEY` is registered as the agent's memory model key. |
| **MCP tools** | Every tool the virtual key allows (the gateway's own listing for the key), each executed through the gateway, one call at a time. |
| **Risk tiers** | From the MCP server's annotations (`readOnlyHint` → read, `destructiveHint` → irreversible, else write) or a local tool's declaration, overridden by the catalog's `risk`. `read` runs; `write` runs and is announced (`tool_notice` event); `irreversible` asks a person. The catalog's `approve_when` (an administrator's rule, or an accepted approval suggestion, in the memory service's own expression language) replaces the tier: the call asks exactly when it holds. Every tool is published to the catalog. |
| **Tool hints** | From 5 tools on, the context is asked for with the run's tool names and comes back with the candidates that fit the task; the model is offered the memory tools, those candidates and every tool the run already used — per model call for `ReAct` and OpenAI Agents, per run for Claude; a `tool_search` call offers what it finds. |
| **Code Mode** | The Code Mode servers whose tools all only read, from 3 servers or 20 tools, become Bifrost's Code Mode meta-tools (one script instead of many calls); their nested calls are recorded from the gateway's log. |
| **Outcome** | From how the run ended (`SUCCESS` confirm, `ERROR` reject) as the run's `system` feedback; the judge's and a person's feedback outrank it. |
| **Grounding** | On a sampled 10 % of successful runs, the answer is checked against the context it was given (`/v1/verify` with its `bundle_id`); the score goes on the run's trace. |
| **Traces** | OTel GenAI spans (`invoke_agent`, `execute_tool`, `chat`) with Langfuse's trace attributes (agent, user, session = thread, run, tenant); every attempt of a run in one trace. |

## Frameworks

| Target | Stream | Pause / resume | Harness tools | Tools narrowed | Memory push |
|---|---|---|---|---|---|
| LangGraph graph, Deep Agents | text deltas and tool events | native `interrupt` / `Command(resume=)` with a checkpointer; re-run against the journal without | built in with `h.tools(..., framework="langgraph")` (a compiled graph refuses `tools=`) | no (bound at build) | leading system message, one per checkpointed thread |
| OpenAI Agents `Agent` | text deltas and tool events | `ask` → re-run against the journal; the SDK's own `needs_approval` → its `RunState` approved or rejected and continued | added to a copy per run | per turn (`FunctionTool.is_enabled`) | leading `system` message |
| Claude Agent SDK `ClaudeAgentOptions` | assistant text blocks and tool events | `ask` → the CLI is stopped, re-run against the journal | in-process MCP server `trellis` (`mcp__trellis__*`, pre-allowed) | per run | appended to `system_prompt` |
| `ReAct` | per step, and tool events | re-run against the journal | per run | per model call | appended to `system` |
| async function `(input, agent)` | tool events | re-run against the journal | `agent.tools.call(...)` | n/a | `agent.context` (and a leading system message for a message list) |

Teams bring their own model objects pointed at Bifrost's OpenAI-compatible endpoint
(`ChatOpenAI(base_url=BIFROST_URL)`, `OpenAIChatCompletionsModel(AsyncOpenAI(base_url=...))`);
the harness does not wrap models.

## Observability and evaluation

Langfuse is the eval system of record — LLM-as-judge evaluators, annotation queues, datasets,
experiments, dashboards — per agent and per request through the trace attributes. The harness
emits the traces, the grounding score and people's feedback; it builds no evaluator of its
own. See [docs/observability.md](docs/observability.md).

## Examples

Each runs with no services (scripted models, runs in process) and uses the real ones when
`BIFROST_URL` / `MEMORY_URL` / `RUNS_URL` are set: `make examples`.

`langgraph_agent.py`, `deepagents_agent.py`, `openai_agents_agent.py`,
`claude_agent_sdk_agent.py`, `react_agent.py`, `cowork.py` (start → worker → ask with a diff →
inbox → resume), `schedule.py`, `serve_chat.py` (AG-UI and A2A on one FastAPI app).

## Development

```bash
make install     # uv sync (the sibling trellis checkouts are path sources)
make check       # ruff, pyright, tests
make examples
make bench       # the overhead benchmark against benchmark-results.json
make test-live   # opt-in tests against BIFROST_URL / MEMORY_URL / RUNS_URL / TRELLIS_API_KEY
```

`make test-live` reads the deployment environment and skips whatever is unset or
unreachable; see `tests/live/conftest.py` for what it registers in the gateway for the session.

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/](docs/README.md).
