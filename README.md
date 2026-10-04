# trellis-harness

Attach memory, tools, approvals, durable runs and tracing to an agent you already built — a
LangGraph graph (Deep Agents included), an OpenAI Agents SDK `Agent`, a Claude Agent SDK
`ClaudeAgentOptions`, the harness's own `ReAct` loop, or a plain async function. The framework
is not modified and not re-implemented: the harness wraps the object you built and speaks to
it through the framework's public API.

```python
from trellis import Harness

h = Harness()  # the deployment is the environment
agent = h.wrap(graph, id="procurement")  # nothing else to configure

result = await agent.run("Reorder SKU-1 if low", user="ada")
if result.interrupt:  # a person has to approve something
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
```

Everything else is automatic: memory is on when the deployment has a memory service; the MCP
tools are the ones the agent's Bifrost virtual key allows; which calls run, which are
announced and which wait for a person follows from the tools' own annotations and the tool
catalog; the tool schemas the model sees are narrowed to what fits the task; the tenant comes
from the key. (What a run may write in memory is the memory service's to decide, per scope —
its relationship checks — not a property of the key's role: with memory on, every run records
its transcript, tool calls and outcome, and a refused write is a reported warning.)

**Where to start reading:** [docs/README.md](docs/README.md) — every page, and which feature to
use when (target, run mode, surface, tools, approvals, memory, evaluation, local or agent-runs,
what each variable turns on) — and one page per framework, each with the lines to add to an
existing project: [LangGraph and LangChain](docs/frameworks/langgraph.md),
[Deep Agents](docs/frameworks/deepagents.md), [OpenAI Agents SDK](docs/frameworks/openai-agents.md),
[Claude Agent SDK](docs/frameworks/claude-agent-sdk.md), [ReAct](docs/frameworks/react.md),
[plain functions](docs/frameworks/functions.md).

## Install

The platform is not published to PyPI yet: install from source, with the sibling repositories
checked out next to this one — they are path dependencies (`[tool.uv.sources]` in
`pyproject.toml`):

```bash
mkdir trellis && cd trellis
for repo in agent-contracts bifrost-sdk agent-memory-service agent-harness; do
  git clone https://github.com/amitmohapatra/$repo.git
done
cd agent-harness
uv sync --all-extras    # the core, every framework extra and the dev tools, in .venv
```

Into an environment of your own, with pip, the siblings first:

```bash
pip install -e ../agent-contracts -e ../bifrost-sdk -e ../agent-memory-service/sdk/python
pip install -e '.[langgraph]'    # or any extras, below
```

One distribution; each framework is an extra (the core imports none of them): `langgraph`
(LangGraph, and `langchain` for `create_agent` and its middleware), `deepagents` (brings
`langgraph`), `openai-agents`, `claude-agent-sdk`, `agui` (`serve_chat`,
FastAPI), `a2a` (`serve_a2a`, `a2a()` tools and `remote()`), `otel` (OTLP export to Langfuse or a
collector), `all`.

`trellis` is shared with `trellis-contracts` (`trellis.contracts`) and `trellis-memory`
(`trellis.memory`); importing either does not load the harness.

### Quickstart

With nothing configured everything runs in process (no memory, runs kept in memory, tenant
`default`). With the memory service's development stack the agent remembers — and nothing else
changes: a development key speaks for the tenant `default`, so no `tenant=` anywhere.

```bash
(cd ../agent-memory-service && docker compose up -d)   # http://localhost:8080, key "dev-key"
export MEMORY_URL=http://localhost:8080 TRELLIS_API_KEY=dev-key
```

```python
import asyncio

from trellis import Harness, Runtime


async def answer(question: str, agent: Runtime) -> str:
    return f"{agent.context or 'Nothing remembered yet.'}\n\n(asked: {question})"


async def main() -> None:
    async with Harness() as h:  # the environment above, or nothing
        agent = h.wrap(answer, id="hello")
        result = await agent.run("How do I like to be contacted?", user="ada")
        print(result.status, result.answer)


asyncio.run(main())
```

Swap `answer` for your LangGraph graph, OpenAI Agents `Agent`, `ClaudeAgentOptions` or a
`ReAct` and keep the rest; `make examples` runs one of each with no services.

## Configuration

The environment, and nothing else ([`.env.example`](.env.example)):

| Variable | |
|---|---|
| `BIFROST_URL` | Bifrost's `/v1` base: the MCP tools, `ReAct` model names |
| `BIFROST_VIRTUAL_KEY` | the agent's virtual key: its models, its MCP tools, its budget — and, registered automatically, the key the memory service's LLM work for the agent is billed to |
| `TRELLIS_API_KEY` | the one Trellis key, for the memory service and agent-runs (required with `MEMORY_URL`); its tenant is asked of the memory service (`GET /v1/keys/self`) |
| `MEMORY_URL` | the memory service; memory is on exactly when it is set |
| `RUNS_URL` | agent-runs; unset keeps runs, the queue and schedules in process |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` | OTLP traces: Langfuse's endpoint, or a collector ([deploy/otel-collector.yaml](deploy/otel-collector.yaml)) |
| `TRELLIS_SPOOL_DIR` | where memory writes this process could not deliver are kept and replayed from at the next start |
| `TRELLIS_WORKER_CONCURRENCY` | runs a worker executes at once (default: the CPU count, 1 to 8) |
| `TRELLIS_GROUNDING_SAMPLE` | the share of successful runs (0 to 1, by run id) checked against their memory context (default 0.1) |
| `TRELLIS_JUDGE_MODEL` | the Bifrost model `llm_judge` asks — a different, stronger model than the agent's (unset: the judged agent's own model, logged) |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the virtual key the judge's calls go through, so evaluation has its own budget (unset: `BIFROST_VIRTUAL_KEY`) |
| `TRELLIS_JUDGE_SAMPLE` | the share of successful runs the online judges score (0 to 1, by run id; unset: 0.1 with judges) |

## The API

The wrapped API is importable from `trellis`; the blocks usable without `Harness` from their own
modules — `trellis.harness.governance` ([docs/governance.md](docs/governance.md)) and
`trellis.harness.evals` (below):

| Name | What it is |
|---|---|
| `Harness(config=None, *, judges=())` | Reads the environment; `config=Settings(...)` instead of it. `judges` are online evaluators that score a sampled share of successful runs in the background ([docs/evaluation.md](docs/evaluation.md)). `async with` (or `await h.aclose()`) drains the background writes and closes the clients. `h.agents` is every agent it wraps, by id. |
| `Settings(bifrost_url=, bifrost_virtual_key=, api_key=, memory_url=, runs_url=, otlp_endpoint=, otlp_headers=, spool_dir=, worker_concurrency=, grounding_sample=, judge_model=, judge_virtual_key=, judge_sample=)` | The deployment as fields (every one optional); `Settings.from_env()` is what `Harness()` reads ([docs/configuration.md](docs/configuration.md)). |
| `h.wrap(target, *, id, tools=()) -> Agent` | Attach the harness. The framework is detected from the target's type. `tools` are the agent's own, run in this process: functions, `a2a(url)`, `openapi(spec)`. |
| `await h.tools(*sources, framework=...)` | The toolbox as the framework's own tools, for an agent built with them before wrapping (a compiled LangGraph graph binds its tools): LangChain tools (`"langgraph"`), `FunctionTool`s (`"openai-agents"`), one in-process MCP server (`"claude-agent-sdk"`). It holds `sources`, the MCP tools and the memory tools. |
| `h.worker(agents, *, concurrency=None)` | Claims queued runs of these agents and executes them, `concurrency` at a time (default `TRELLIS_WORKER_CONCURRENCY`, else the CPU count from 1 to 8): `await worker.run()` (until `worker.stop()`: the runs held finish, or are released after a grace period), `await worker.serve()` (the same, stopped by `SIGTERM`/`SIGINT`) or `await worker.run_once()`. The loop is agent-runs' SDK's `trellis.runs.Worker`, which code on its own framework runs with its own handler ([docs/runs.md](docs/runs.md#workers)). |
| `await h.inbox(assignee=None, *, tenant=None) -> list[RunSummary]` | The paused runs waiting on `assignee` (`user:…`, `role:…`) or on anyone, newest first, as `trellis.runs.RunSummary` (pip `trellis-runs`, the agent-runs SDK). `tenant` only for a platform key. |
| `await h.feedback(run_id, verdict, correction=None, *, tenant=None) -> Feedback \| None` | What a person said about a run (`verdict`: `confirm`, `approve`, `reject`, `correct` or `edit`, the last two with a `correction`): a `feedback` score on its trace (Langfuse; 1.0, 0.5 for `edit`, 0.0 for `correct`/`reject`) and — memory on — the run's `human` feedback in the memory service, returned as stored (`None` with memory off). It waits for the tenant administrator (`review.state == "pending"`) before it changes what memory learned. |
| `await h.evaluate(agent, dataset, evaluators, *, run_name=None, description=None, metadata=None, concurrency=4, limit=None, user=None) -> EvalReport` | Run the agent on every item of a dataset — a Langfuse dataset's name, or `[{"input", "expected"?, "metadata"?}]` / `EvalItem`s — through the normal pipeline, `concurrency` at a time, score each answer with `evaluators` onto its run's trace, and make each run an item of the Langfuse experiment `run_name` (the dataset run link on Langfuse v3, the `langfuse.experiment.*` span attributes on v4). A failing or pausing item is reported (`error`, `interrupted`), never fatal. The `EvalReport` has every item in order and each evaluator's mean, count and failures. The evaluation names (the evaluators, `EvalItem`, `EvalReport`...) are imported from `trellis.harness.evals`, which also evaluates and judges code that is not wrapped (below, and [docs/evaluation.md](docs/evaluation.md)). |
| `await h.add_document(file, *, user, tenant=None, thread=None, title=None, visibility=None, wait=60) -> DocumentInfo` | Add a file (bytes, a path, or `(filename, bytes, media_type)`) to a user's document memory (or one thread's), waiting until it is indexed (`wait=None`: return at once): the user's next context cites it. `visibility` widens it (`WORKSPACE`, `TENANT`); `tenant` only for a platform key. Needs `MEMORY_URL`. |
| `tool(fn, *, name=None, description=None, side_effects="write")` / `@tool` / `@tool(...)` | A Python function (sync or async) as a tool: the schema from its signature (pydantic validates the arguments), the description from its docstring's first paragraph, `side_effects` `"read"`, `"write"` (default) or `"irreversible"`. It stays callable as the function. A bare function in `tools=[...]` is `tool(fn)`. |
| `a2a(url, *, name=None)` | A remote A2A agent (its card at `{url}/.well-known/agent-card.json`) as one `write` tool, `{"message": string}` in, its answer out; `name` overrides the card's. |
| `trellis.harness.a2a.remote(url, *, tenant, user, thread=None, on_input=None, ...)` | The same A2A client for any code, no `Harness`: a `RemoteAgent` — `await agent(message)` is the answer; `card`, `spec` (a `ToolSpec`); a remote question goes to `on_input(question)`, else raises `InputRequired` and `reply(task_id, answer)` continues it ([docs/surfaces.md](docs/surfaces.md#calling-an-a2a-agent-from-your-own-code)). |
| `openapi(spec, *, only=None, base_url=None, headers=None)` | The operations of an OpenAPI 3 document (a URL or the parsed document) as tools, one per `operationId` (`only` keeps those named); `base_url` when the document names no server; `headers` on every request. |
| `ReAct(system, model, output=None, max_steps=12, *, max_result_chars=20000, max_repeats=3)` | A tool-calling loop over chat completions, for teams with no framework: `model` is a Bifrost model name (needs `BIFROST_URL`) or any object with `async complete(messages, **body)`; `output` a pydantic model for a structured answer. Arguments that are not JSON or do not fit the tool's schema are an error the model reads (the tool does not run); a result longer than `max_result_chars` is cut with a marker (the whole of it kept as a run artifact); the same call in `max_repeats` consecutive steps stops the run; a resume replays the model steps already taken. |
| `current() -> Runtime \| None` | Inside a tool or a node: the run it executes in. |

`Agent` — what `wrap` returns:

| Method | |
|---|---|
| `await run(input, *, user, thread=None, tenant=None) -> Result` | Run to the end or the first pause. `tenant` only for a platform key (one with no tenant of its own). |
| `stream(...) -> AsyncIterator[RunEvent]` | The run's events (contracts `RunEvent`s) up to `RUN_FINISHED`. Closing it early cancels the run. |
| `await start(...) -> RunHandle` | Queue the run for a worker (its input must be JSON). The `RunHandle`: `run_id`, `await handle.status()` (the `RunRecord`) and `await handle.result(timeout=None)`, which waits for a pause or an ending and returns a `Result`. |
| `await resume(interrupt_id, decision, *, answer=None, reviewer, tenant=None) -> Result` | Answer the pause (`decision`, a string or contracts `InterruptDecision`: `answer`, `approve`, `reject`, `edit` with the edited arguments as `answer`, `cancel`). A run started in process continues here; a queued run goes back to the queue (`QUEUED`). `tenant` only for a platform key. |
| `await schedule(cron, input, *, on_behalf_of, tz="UTC", tenant=None) -> Schedule` | Queue a run on a cadence (cron, or `hourly`/`daily`/`weekly`/`weekdays`/`manual`), acting for `on_behalf_of`. The same agent, person, cadence and input are one schedule. |
| `serve_chat(app, *, path="/agui", identity=None)` | AG-UI routes on a FastAPI app (run, reconnect/replay, artifacts); `identity(request) -> user` (sync or async), else every caller is `anonymous` ([docs/surfaces.md](docs/surfaces.md)). |
| `serve_a2a(app, url, *, identity=None)` | The agent card and A2A JSON-RPC routes at `url`; `identity(call_context) -> user`, else the trusted `x-trellis-identity` header ([docs/surfaces.md](docs/surfaces.md)). |

`Runtime` — `trellis.current()` inside a tool or a node (`None` outside a run), and the second
argument of a function target: `run_id`, `agent_id`, `user`, `thread`, `tenant`, `attempt`,
`task` (the question), `context` (the pushed memory context), `memory` (the memory SDK's
verbs in the run's scope; needs `MEMORY_URL`), `await tools.call(name, **args)`,
`await tools.hints(task)`, `await ask(question, *, expects, table, diff, options, assignee,
deadline, escalate_to)` (pauses the run; returns the answer on resume —
[docs/interrupts.md](docs/interrupts.md)) and `log(message, **fields)`.

`Result`: `run_id`, `status` (`SUCCESS`, `PAUSED`, `ERROR`, `QUEUED`, `CANCELLED`), `answer`,
`interrupt`, `error`. `trellis.runs.RunSummary`: `run_id`, `agent_id`, `status`, `awaiting`,
`assignee`, `deadline`, `updated_at`. `h.runs` is the run store: agent-runs' client
(`trellis.runs.RunsClient`) with `RUNS_URL`, else the in-process `LocalRuns`; code on its own
framework uses `RunsClient` directly ([docs/runs.md](docs/runs.md)).

Run a worker for every agent a module's harness wraps:

```bash
python -m trellis.harness.worker app.agents:h [--concurrency N]   # SIGTERM stops it gracefully
```

## What happens automatically

| | |
|---|---|
| **Memory** | With `MEMORY_URL`: the context for the question is pushed into the framework's input (without the recent conversation when the framework keeps the thread itself), the memory tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are added, and the transcript, every tool call and the run's outcome are recorded. `BIFROST_VIRTUAL_KEY` is registered as the agent's memory model key. |
| **MCP tools** | Every tool the virtual key allows (the gateway's own listing for the key), each executed through the gateway, one call at a time. |
| **Governance** | Each call runs, runs and is announced (`tool_notice` event), or asks a person: by the tool's risk — the MCP server's annotations (`readOnlyHint` → read, `destructiveHint` → irreversible, else write) or a local tool's declaration, overridden by the catalog's `risk`; `read` runs, `write` is announced, `irreversible` asks. The catalog's `approve_when` (an administrator's rule, or an accepted approval suggestion, in the memory service's own expression language) replaces that: the call asks exactly when it holds. A catalog that cannot be read fails closed. Every tool is published to the catalog. The same decisions are available without `h.wrap` (`trellis.harness.governance`, [docs/governance.md](docs/governance.md)). |
| **Tool hints** | From 5 tools on, the context is asked for with the run's tool names and comes back with the tools that fit the task (each with a 0–1 confidence); the model is offered the memory tools, those and every tool the run already used — per model call for `ReAct` and OpenAI Agents, per run for Claude; a `tool_search` call offers what it finds. |
| **Code Mode** | The Code Mode servers whose tools all only read, from 3 servers or 20 tools, become Bifrost's Code Mode meta-tools (one script instead of many calls); their nested calls are recorded from the gateway's log. |
| **Outcome** | From how the run ended (`SUCCESS` confirm, `ERROR` reject) as the run's `system` feedback, applied as it arrives; the judge's verdict outranks it, and so does a person's once the tenant administrator approves it. |
| **Grounding** | On a sampled share of successful runs (`TRELLIS_GROUNDING_SAMPLE`, 10 % by default), the answer is checked against the context it was given (`/v1/verify` with its `bundle_id`); the score goes on the run's trace. |
| **Traces** | OTel GenAI spans (`invoke_agent`, `execute_tool`, `chat`) with Langfuse's trace attributes (agent, user, session = thread, run, tenant); every attempt of a run in one trace. |

## Frameworks

| Target | Stream | Pause / resume | Harness tools | Tools narrowed | Memory push |
|---|---|---|---|---|---|
| LangGraph graph, Deep Agents | text deltas and tool events | native `interrupt` / `Command(resume=)` with a checkpointer; re-run against the journal without; `HumanInTheLoopMiddleware` / `interrupt_on` pauses are approvals answered with the harness's decisions | built in with `h.tools(..., framework="langgraph")` (a compiled graph refuses `tools=`) | no (bound at build) | leading system message, one per checkpointed thread |
| OpenAI Agents `Agent` | text deltas and tool events | `ask` → re-run against the journal; the SDK's own `needs_approval` → its `RunState` approved or rejected (with the reason; an edit or an answer as the message the model reads) and continued | added to a copy per run | per turn (`FunctionTool.is_enabled`) | leading `system` message |
| Claude Agent SDK `ClaudeAgentOptions` | assistant text blocks and tool events | `ask` → the CLI is stopped, re-run against the journal | in-process MCP server `trellis` (`mcp__trellis__*`, pre-allowed) | per run | appended to `system_prompt` |
| `ReAct` | per step, and tool events | re-run against the journal, model steps included (no repeated model call) | per run | per model call | appended to `system` |
| async function `(input, agent)` | tool events | re-run against the journal | `agent.tools.call(...)` | n/a | `agent.context` (and a leading system message for a message list) |

Teams bring their own model objects pointed at Bifrost's OpenAI-compatible endpoint
(`ChatOpenAI(base_url=BIFROST_URL)`, `OpenAIChatCompletionsModel(AsyncOpenAI(base_url=...))`);
the harness does not wrap models.

Each target has a page — how to add the harness to an existing project, what is automatic,
approvals, streaming, durable runs, surfaces, evaluation and limits:
[docs/frameworks/langgraph.md](docs/frameworks/langgraph.md) (`create_agent`, a hand-built
`StateGraph`, checkpointers, `HumanInTheLoopMiddleware`),
[docs/frameworks/deepagents.md](docs/frameworks/deepagents.md) (sub-agents, planning,
`interrupt_on`, its built-in tools), [docs/frameworks/openai-agents.md](docs/frameworks/openai-agents.md)
(handoffs, `needs_approval`), [docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md)
(the MCP server, `allowed_tools`, Claude Code's built-in tools),
[docs/frameworks/react.md](docs/frameworks/react.md) and
[docs/frameworks/functions.md](docs/frameworks/functions.md). The framework's own entry point
(`graph.ainvoke`, `Runner.run`, `query`) is not intercepted: call `agent.run`/`stream`/`resume`.

## Observability and evaluation

Langfuse is the eval system of record — scores, datasets and dataset runs, annotation queues,
dashboards — per agent and per request through the trace attributes. The harness emits the
traces, the grounding score and people's feedback, and makes both kinds of evaluation one call
([docs/evaluation.md](docs/evaluation.md)):

```python
from trellis.harness.evals import grounding, llm_judge

report = await h.evaluate(agent, "support-golden", [grounding(), llm_judge("Cites the policy.")])
print(report)  # offline: a Langfuse dataset (or a list), every answer scored, runs linked

h = Harness(judges=[llm_judge("Polite and correct.")])  # online: sampled runs, in the background
```

Evaluation is also a block for code that is not wrapped — a LangGraph graph, an OpenAI Agents
`Runner`, a Claude Agent SDK `query`, plain functions — from `trellis.harness.evals`:

| Name | What it is |
|---|---|
| `EvalServices.from_env()` | Langfuse (the OTLP variables) and the judge (`BIFROST_URL`, `TRELLIS_JUDGE_MODEL`, `TRELLIS_JUDGE_VIRTUAL_KEY`) the environment names; `async with` closes them. A wrapped agent's are `agent.evals`. |
| `await evaluate(target, dataset, evaluators, *, services=None, user=None, run_name=None, description=None, metadata=None, concurrency=4, limit=None) -> EvalReport` | `h.evaluate` for a wrapped `Agent`, or any `async (input) -> answer` — each item one call in a root span of its own, an item of the Langfuse experiment the same way. Return `EvalOutput(answer, bundle_id, memory)` for grounding. |
| `await judge(case, judges, *, services, sample=None)` | Score one `EvalCase` on-line, on its trace (`trace_id`, else its run's); `sample` judges a stable share of runs. A judge that fails is reported, never raised. |
| `EvalItem`, `EvalCase`, `EvalScore`, `EvalReport`, `EvalResult`, `EvalOutput`, `Evaluator` | A dataset item; what an evaluator is given (input, output, expected, run and trace, memory context and scope); what it returns (`EvalScore(name, value, comment=None)`); the report and each item's result; what a callable target may return; an `Evaluator` is any `async (EvalCase) -> EvalScore \| None`. |
| `grounding()`, `exact_match()`, `contains()`, `llm_judge(criteria, *, name="llm_judge")` | The built-in evaluators: grounding against the run's memory context (`grounding_score`: `/v1/verify`), against `expected`, and a judge model (`TRELLIS_JUDGE_MODEL` through Bifrost with `TRELLIS_JUDGE_VIRTUAL_KEY`). |

The judge's model and virtual key are configuration (`TRELLIS_JUDGE_MODEL`,
`TRELLIS_JUDGE_VIRTUAL_KEY`): pick a stronger model than the agent's, on its own budget. See
also [docs/observability.md](docs/observability.md).

## Examples

Each runs with no services (scripted models, runs in process) and uses the real ones when
`BIFROST_URL` / `MEMORY_URL` / `RUNS_URL` are set: `make examples`.

| Example | Shows |
|---|---|
| `langgraph_agent.py` | `create_agent` with `h.tools`, no checkpointer: an approval re-run from the journal |
| `langgraph_stategraph.py` | a hand-built `StateGraph` with a checkpointer: a harness approval in the tool node, then the graph's own `interrupt()`, both resumed in place |
| `langchain_hitl_middleware.py` | LangChain's `HumanInTheLoopMiddleware`: an edit, then a reject with a reason |
| `deepagents_agent.py` | Deep Agents with `TodoListMiddleware`: a plan, an approval resumed in place |
| `openai_agents_agent.py` | harness tools added to an OpenAI Agents `Agent` per run, an approval |
| `openai_agents_handoff.py` | a handoff to a specialist built with `h.tools(framework="openai-agents")` |
| `claude_agent_sdk_agent.py` | `ClaudeAgentOptions` with a harness tool (a scripted CLI offline) |
| `react_agent.py` | `ReAct` with a tool and a structured answer |
| `cowork.py` | start → worker → ask with a diff → inbox → resume → worker |
| `schedule.py` | a schedule a worker runs (memory on: the person's context) |
| `serve_chat.py` | AG-UI and A2A on one FastAPI app |
| `a2a_agents.py` | an agent served over A2A and consumed by another as a tool; the remote question answered; the same agent called from plain code with `remote()` |
| `memory_features.py` | a document the next context cites, `agent.memory`, a person's feedback (memory on) |
| `evaluate_offline.py` | a dataset scored by exact match, contains and a judge, for a wrapped agent (`h.evaluate`) and a plain function (`evaluate`); the reports |
| `online_judges.py` | judges on live runs, in the background; `judge()` on a run of your own code |

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
`tests/live/test_live_matrix.py` needs only the memory service and agent-runs (with its ticker):
every model in it is planned, everything else is real — each framework against both services
(memory pushed and pulled, approvals in agent-runs, the records read back), then workers,
schedules, AG-UI, A2A, documents, feedback and evaluation with memory on.

The memory service in the tests is an in-process fake (`tests/support/memory.py`) behind the
real SDK, and every request the harness sends it and every answer it gives is checked against
the memory service's committed `docs/openapi.json` (a test with a mismatch fails);
`tests/contract/test_openapi.py` drives every call once and does the same for every call of
the run store (`trellis.runs.RunsClient`) against agent-runs' `docs/openapi.json`;
`tests/contract/test_runs_wire.py` checks what real runs (approvals, artifacts, failures, a
worker's progress, schedules, a platform key's tenant on every call) send agent-runs and that
its schemas and enums are the contracts' models; `tests/contract/test_types.py` that every
result, event, record and error handed back is the contracts type. The documents are read from
the sibling checkouts (CI checks out `main` of each), or from `TRELLIS_MEMORY_OPENAPI` /
`TRELLIS_RUNS_OPENAPI`; an agent-runs checkout without its document fails the tests rather
than skipping them.

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) (diagrams: system context, components, a run, a
pause through agent-runs, an A2A call, run states), [docs/scenarios.md](docs/scenarios.md)
(which to use when) and [docs/](docs/README.md).
