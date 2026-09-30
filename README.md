# trellis-harness

Attach memory, tools, approvals, durable runs and evaluation to an agent you already built —
a LangGraph graph (Deep Agents included), an OpenAI Agents SDK `Agent`, a Claude Agent SDK
`ClaudeAgentOptions`, the harness's own `ReAct` loop, or a plain async function. The framework
is not modified and not re-implemented: the harness wraps the object you built and speaks to
it through the framework's public API.

```python
from trellis import Harness, mcp, tool, a2a, openapi, ReAct

h = Harness()  # reads the environment (.env.example)
agent = h.wrap(
    graph,
    id="procurement",
    tools=[mcp("erp", only=["get_stock"]), my_fn, a2a(url)],
    memory="read_write",  # "off" | "read" | "read_write"
    approve={"erp-create_po": "amount > 10000"},
    tool_hints=False,
)

result = await agent.run("Reorder SKU-1 if low", user="ada")
if result.interrupt:  # a person has to approve something
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
```

## Install

One distribution; each framework is an extra (the core imports none of them).

```bash
pip install 'trellis-harness[langgraph]'          # LangGraph
pip install 'trellis-harness[deepagents]'         # Deep Agents (brings langgraph)
pip install 'trellis-harness[openai-agents]'      # OpenAI Agents SDK
pip install 'trellis-harness[claude-agent-sdk]'   # Claude Agent SDK
pip install 'trellis-harness[agui]'               # serve_chat (FastAPI)
pip install 'trellis-harness[a2a]'                # serve_a2a and a2a() tools
pip install 'trellis-harness[otel]'               # OTLP export (and Langfuse through it)
pip install 'trellis-harness[all]'
```

`trellis` is shared with `trellis-contracts` (`trellis.contracts`) and `trellis-memory`
(`trellis.memory`); importing either does not load the harness.

## The API

Everything public is importable from `trellis`:

| Name | What it is |
|---|---|
| `Harness(config=None)` | Reads the environment; `config=Settings(...)` instead of it. `async with` closes it. |
| `h.wrap(target, *, id, tools=(), memory="off", approve=None, tool_hints=False) -> Agent` | Attach the harness. The framework is detected from the target's type. |
| `await h.tools(*sources, framework=..., memory=False)` | The sources as the framework's own tools, for building an agent with them before wrapping: LangChain tools (`"langgraph"`), `FunctionTool`s (`"openai-agents"`), one in-process MCP server (`"claude-agent-sdk"`). `memory=True` adds the memory service's agent tools. |
| `h.worker(agents, *, concurrency=4)` | Claims queued runs of these agents and executes them: `await worker.run()` (until cancelled) or `await worker.run_once()`. |
| `await h.feedback(run_id, verdict, correction=None, *, reviewer=None)` | What a person said about a run, stored in the memory service. |
| `mcp(*servers, only=None)` | Tools of MCP servers registered in Bifrost. |
| `tool(fn)` / `@tool(name=, description=, side_effects=)` | A Python function as a tool (`side_effects`: `"read"`, `"write"` (default), `"irreversible"`). A bare function in `tools=[...]` is `tool(fn)`. |
| `a2a(url, *, name=None)` | A remote A2A agent as one tool. |
| `openapi(spec, *, only=None, base_url=None, headers=None)` | The operations of an OpenAPI 3 document as tools. |
| `ReAct(system, model, output=None, max_steps=12)` | A tool-calling loop over chat completions, for teams with no framework. |
| `current() -> Runtime \| None` | Inside a tool or a node: the run it executes in. |

`Agent` — what `wrap` returns:

| Method | |
|---|---|
| `await run(input, *, user, thread=None, tenant=None) -> Result` | Run to the end or the first pause. |
| `stream(...) -> AsyncIterator[RunEvent]` | The run's events (contracts `RunEvent`s) up to `RUN_FINISHED`. Closing it early cancels the run. |
| `await start(...) -> RunHandle` | Queue the run for a worker; `handle.status()`, `await handle.result(timeout=)`. |
| `await resume(interrupt_id, decision, *, answer=None, reviewer) -> Result` | Answer the pause (`answer`, `approve`, `reject`, `edit` with the edited arguments as `answer`, `cancel`). A run started in process continues here; a queued run goes back to the queue. |
| `await schedule(cron, input, *, on_behalf_of, tz="UTC") -> Schedule` | Queue a run on a cron, acting for `on_behalf_of`. |
| `serve_chat(app, *, path="/agui", identity=None)` | AG-UI routes on a FastAPI app (run, reconnect/replay, artifacts). |
| `serve_a2a(app, url, *, identity=None)` | The agent card and A2A JSON-RPC routes at `url`. |

`Runtime` — `trellis.current()` inside a tool or a node, and the second argument of a
function target: `run_id`, `agent_id`, `user`, `thread`, `tenant`, `context` (the pushed
memory context), `memory` (the memory SDK's verbs in the run's scope), `tools.call(name,
**args)`, `tools.hints(task)`, `await ask(question, *, ui, expects, table, options, assignee,
deadline, escalate_to)` (pauses the run; returns the answer on resume) and `log(message,
**fields)`.

`Result`: `run_id`, `status` (`SUCCESS`, `PAUSED`, `ERROR`, `QUEUED`, `CANCELLED`), `answer`,
`interrupt`, `error`.

Run a worker for every agent a module's harness wraps:

```bash
python -m trellis.worker app.agents:h
```

## Frameworks

| Target | Stream | Pause / resume | Harness tools | Memory push | Memory pull |
|---|---|---|---|---|---|
| LangGraph graph, Deep Agents | text deltas and tool events | native `interrupt` / `Command(resume=)` with a checkpointer; re-run against the journal without | built in with `h.tools(..., framework="langgraph")` (a compiled graph refuses `tools=`) | leading system message | `h.tools(..., memory=True)` at build time |
| OpenAI Agents `Agent` | text deltas and tool events | `ask` → re-run against the journal; the SDK's own `needs_approval` → its `RunState` approved or rejected and continued | `wrap(tools=)` (added to a copy per run) or `h.tools` | leading `system` message | added per run |
| Claude Agent SDK `ClaudeAgentOptions` | assistant text blocks and tool events | `ask` → the CLI is stopped, re-run against the journal | in-process MCP server `trellis` (`mcp__trellis__*`, pre-allowed) | appended to `system_prompt` | added per run |
| `ReAct` | per step, and tool events | re-run against the journal | `wrap(tools=)` | appended to `system` | added per run |
| async function `(input, agent)` | tool events | re-run against the journal | `agent.tools.call(...)` | `agent.context` (and a leading system message for a message list) | `agent.memory` |

Teams bring their own model objects pointed at Bifrost's OpenAI-compatible endpoint
(`ChatOpenAI(base_url=BIFROST_URL)`, `OpenAIChatCompletionsModel(AsyncOpenAI(base_url=...))`);
the harness does not wrap models.

## Configuration

The environment, and nothing else. Every variable is in [`.env.example`](.env.example):

| Variable | |
|---|---|
| `TRELLIS_TENANT` | tenant a single-tenant deployment runs as (default `default`) |
| `BIFROST_URL`, `BIFROST_VIRTUAL_KEY` | Bifrost's `/v1` base and virtual key: MCP tools, `ReAct` model names, the judge |
| `MEMORY_URL`, `MEMORY_API_KEY` | the memory service |
| `TRELLIS_MEMORY_MODEL_KEY` | LLM key the memory service uses for these agents (registered once per agent) |
| `RUNS_URL`, `RUNS_API_KEY` | agent-runs; unset keeps runs, the queue and schedules in process |
| `TRELLIS_EVAL_SAMPLE` | fraction of runs the online judge scores (default `0.1`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP traces |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | traces to Langfuse, through its OTLP endpoint |

## Evaluation

`trellis.eval`: the sampled online judge (`GroundedJudge` with a `JudgeBudget`), offline
`DatasetBuilder` / `ExperimentRunner` (the item's `expected_output` and `evidence` reach the
judge), and the regression gate CI runs:

```bash
python -m trellis.eval gate --baseline benchmark-results.json --current build/benchmark-results.json
```

## Examples

Each runs with no services (scripted models, runs in process) and uses the real ones when
`BIFROST_URL` / `MEMORY_URL` / `RUNS_URL` are set: `make examples`.

`langgraph_agent.py`, `deepagents_agent.py`, `openai_agents_agent.py`,
`claude_agent_sdk_agent.py`, `react_agent.py`, `cowork.py` (start → worker → ask → inbox →
resume), `schedule.py`, `serve_chat.py` (AG-UI and A2A on one FastAPI app).

## Development

```bash
make install     # uv sync (the sibling trellis checkouts are path sources)
make check       # ruff, pyright, tests
make examples
make test-live   # opt-in tests against BIFROST_URL / MEMORY_URL / RUNS_URL
make gate        # the benchmark and the regression gate
```

`make test-live` reads the deployment environment (`.env.example`) and skips whatever is unset
or unreachable. Against all three services it runs every target (LangGraph, Deep Agents, OpenAI
Agents, `ReAct`, a function, Claude through a scripted CLI) with memory and an MCP server plus
a local tool, Code Mode, the agent-runs wire (queue, lease loss, escalation), a run continued by
three worker processes, a schedule fired by the ticker, AG-UI with replay, an A2A round trip,
and what the memory service learns from it all. It registers public MCP servers in the gateway
for the session (`tests/live/conftest.py`) and removes them after.

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/](docs/README.md).

## Removed in 0.4.0

`AgentHarness` and its keyword arguments, `@agent`, `execution()`, the per-framework harness
classes and their middleware, hooks, sessions, model wrappers and memory backends, the
`trellis-harness-*` integration distributions (now extras), Temporal, YAML and
`pydantic-settings` configuration, listeners, the Langfuse SDK providers, the registry client
and `react()` (replaced by `ReAct`).
