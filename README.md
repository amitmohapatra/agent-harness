# trellis-harness

Trellis gives an agent you already built the platform around it: memory of the people it works
for, durable runs that wait for a person and survive a restart, governance of its tool calls,
evaluation, models and MCP tools through the Bifrost gateway, and AG-UI and A2A serving. The
agent stays what it is: a LangGraph graph (Deep Agents included), an OpenAI Agents SDK
`Agent`, a Claude Agent SDK `query()`, the harness's own `ReAct` loop, or a plain async
function.

## Two ways to use Trellis

**Way 1, wrapped: the harness runs your agent.** `h.wrap(agent)` attaches every piece at once,
and you call `agent.run` where you called the framework. The framework is not modified and not
re-implemented: the harness speaks to the object you built through its public API, and around
each run it pushes the memory context in, governs every tool call, pauses for approvals in
agent-runs and resumes them without repeating a side effect, records the transcript and every
call, judges a sample of answers and traces it all. Each piece switches on with its environment
variable; there is nothing else to configure.

**Way 2, pluggable blocks: your framework, our pieces.** Your framework keeps running the agent,
untouched, and your code imports only the blocks it wants and calls them where it chooses:
`trellis.memory` for the context and the records, `trellis.runs` for durable runs, the inbox,
workers and webhooks, `trellis.harness.governance` for the run, announce or ask decision on your
own tools, `trellis.harness.evals` for datasets and judges, `trellis.harness.a2a.remote` to call
other agents. Your framework's own pause (a LangGraph checkpointer and `interrupt`, an OpenAI
Agents `RunState`, a Claude session) stays the pause; agent-runs keeps the run and the inbox.

Both ways use the same services and the same records, so one deployment can mix them
([mixing both ways](docs/blocks/mixing.md)).

| Question | Way 1, wrapped | Way 2, blocks |
|---|---|---|
| Should Trellis run your agent loop? | Yes: `agent.run`, `agent.stream`, `agent.start` call the framework for you | No: you call `graph.ainvoke`, `Runner.run`, `query()` as today |
| Do you need to serve the agent over AG-UI or A2A? | `agent.serve_chat(app)`, `agent.serve_a2a(app, url)` | Not a block: serving is the harness's pipeline. Wrap the function that calls your agent and serve that; *calling* A2A agents is a block (`remote`) |
| Should durable pause and resume be handled for you? | Yes: an approval or `ask` pauses the run in agent-runs, and the resumed run replays its journal, so no question is asked twice and no side effect repeats | You keep your framework's state and hand agent-runs its checkpoint; the recipes show how for each framework |
| Is the framework code something you cannot change, or call elsewhere directly? | The framework is unchanged, but its runs go through `agent.run` | Nothing changes: the blocks are calls your code makes around it |
| Do you want only one capability (just memory, just the inbox, just evaluation)? | You get all of them, each on when its service is configured; pass the blocks you want (`Harness(memory=False, runs=...)`) and turn parts off per agent or run (`without={"memory", "judges"}`) | Import that one block |

### Way 1 quickstart

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

One page per framework, each with the lines to add to an existing project:
[LangGraph and LangChain](docs/frameworks/langgraph.md),
[Deep Agents](docs/frameworks/deepagents.md), [OpenAI Agents SDK](docs/frameworks/openai-agents.md),
[Claude Agent SDK](docs/frameworks/claude-agent-sdk.md), [ReAct](docs/frameworks/react.md),
[plain functions](docs/frameworks/functions.md). A runnable quickstart is [below](#quickstart).

### Way 2: the blocks

| Block | Import | What it does | Page |
|---|---|---|---|
| Memory | `from trellis.memory import MemoryClient` (pip `trellis-memory`) | the context for a question, into your prompt; the turn and each tool call recorded; feedback on how the run went | [docs/blocks/memory.md](docs/blocks/memory.md) |
| Runs | `from trellis.runs import RunsClient, Worker` (pip `trellis-runs`) | a durable run record; a pause with your framework's checkpoint, the inbox, the answer; a queue and a worker loop; schedules; webhooks and `trellis.runs.webhooks.verify_signature` | [docs/blocks/runs.md](docs/blocks/runs.md) |
| Governance | `from trellis.harness.governance import Governance, governed` | whether each call of your tools runs, is announced or asks a person, from the tool catalog an administrator governs | [docs/blocks/governance.md](docs/blocks/governance.md) |
| Evaluation | `from trellis.harness.evals import EvalServices, evaluate, judge` | a dataset run through your agent and scored; one run judged on-line; every score on the run's trace in Langfuse | [docs/blocks/evaluation.md](docs/blocks/evaluation.md) |
| A2A client | `from trellis.harness.a2a import remote` | another agent (any A2A server) as an async callable; its questions to your `on_input` | [docs/blocks/a2a.md](docs/blocks/a2a.md) |
| Contracts | `from trellis.contracts import RunStart, Interrupt, InterruptResolution, ...` (pip `trellis-contracts`) | the records every block takes and returns | [docs/blocks/contracts.md](docs/blocks/contracts.md) |
| Models and MCP tools | `from bifrost_sdk import ...` (pip `bifrost-sdk`) | completions, MCP tools and the tool log through the Bifrost gateway | [bifrost-sdk](https://github.com/amitmohapatra/bifrost-sdk/blob/main/README.md) |

End to end, an unmodified agent with memory, governed tools, a durable pause in agent-runs
answered from the inbox, and a judge: [LangGraph](docs/blocks/langgraph.md),
[OpenAI Agents SDK](docs/blocks/openai-agents.md), [Claude Agent SDK](docs/blocks/claude-agent-sdk.md).

**Where to read next:** [docs/README.md](docs/README.md), every page in both ways and which
feature to use when; [ARCHITECTURE.md](ARCHITECTURE.md), how it is built.

## Install

The platform is not published to PyPI yet: install from source, with the sibling repositories
checked out next to this one — they are path dependencies (`[tool.uv.sources]` in
`pyproject.toml`):

```bash
mkdir trellis && cd trellis
for repo in agent-contracts bifrost-sdk agent-memory-service agent-runs agent-harness; do
  git clone https://github.com/amitmohapatra/$repo.git
done
cd agent-harness
uv sync --all-extras    # the core, every framework extra and the dev tools, in .venv
```

Into an environment of your own, with pip, the siblings first:

```bash
pip install -e ../agent-contracts -e ../bifrost-sdk -e ../agent-memory-service/sdk/python \
  -e ../agent-runs/sdk/python
pip install -e '.[langgraph]'    # or any extras, below
```

A Way 2 team that wants only `trellis.memory`, `trellis.runs` or `trellis.contracts` installs
that one sibling; governance, evaluation and the A2A client come with this distribution.

One distribution; each framework is an extra (the core imports none of them): `langgraph`
(LangGraph, and `langchain` for `create_agent` and its middleware), `deepagents` (brings
`langgraph`), `openai-agents`, `claude-agent-sdk`, `agui` (`serve_chat`,
FastAPI), `a2a` (`serve_a2a`, `a2a()` tools and `remote()`), `otel` (OTLP export to Langfuse or a
collector), `all`.

`trellis` is a namespace shared with `trellis-contracts` (`trellis.contracts`), `trellis-memory`
(`trellis.memory`) and `trellis-runs` (`trellis.runs`); importing any of them does not load the
harness.

### Quickstart

Way 1. With nothing configured everything runs in process (no memory, runs kept in memory,
tenant `default`). With the memory service's development stack the agent remembers — and
nothing else changes: a development key speaks for the tenant `default`, so no `tenant=`
anywhere.

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

For a real deployment, [docs/onboarding.md](docs/onboarding.md) is the path from running
services to a configured application: the operator creates the tenant, its admin issues the
application's key, the environment is set, and (optionally) people get keys of their own for an
approvals UI.

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
| `TRELLIS_AGENT_VERSION` | the version of the agents' code (a release or deploy id), recorded with every run they start (`h.wrap(..., version=)` wins) |
| `TRELLIS_GROUNDING_SAMPLE` | the share of successful runs (0 to 1, by run id) checked against their memory context (default 0.1) |
| `TRELLIS_JUDGE_MODEL` | the Bifrost model `llm_judge` asks — a different, stronger model than the agent's (unset: the judged agent's own model, logged) |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the virtual key the judge's calls go through, so evaluation has its own budget (unset: `BIFROST_VIRTUAL_KEY`) |
| `TRELLIS_JUDGE_SAMPLE` | the share of successful runs the online judges score (0 to 1, by run id; unset: 0.1 with judges) |

The blocks read the same names: `MemoryClient()` `MEMORY_URL` and `TRELLIS_API_KEY`,
`RunsClient()` `RUNS_URL` and `TRELLIS_API_KEY`, `Governance.from_env()` `MEMORY_URL` and
`TRELLIS_API_KEY`, `EvalServices.from_env()` the OTLP variables, `BIFROST_URL` and the
`TRELLIS_JUDGE_*` ones.

## The API (Way 1)

The wrapped API is importable from `trellis`. The blocks are imported from their own modules
([Way 2: the blocks](#way-2-the-blocks)).

| Name | What it is |
|---|---|
| `Harness(config=None, *, runs=None, memory=None, gateway=None, governance=None, prompts=None, skills=None, judges=(), hooks=())` | The blocks it is given, the rest from the environment (`config=Settings(...)` instead of it): `runs` a run store (`trellis.runs.RunsClient`, `LocalRuns`), `memory` a `trellis.memory.MemoryClient`, `gateway` a `trellis.harness.clients.bifrost.Gateway`, `governance` a `Governance` — each used as it is (and closed by you); `False` leaves memory, the gateway or agent-runs off whatever the environment names ([docs/README.md](docs/README.md#composition-a-harness-is-the-blocks-you-give-it)). `prompts` and `skills` are the prompt and skill sources (`Prompt`, `prompts_dir`, `langfuse_prompts`, `Skill`, `skills_dir`...), asked in their order — else the ones the environment names ([docs/prompts.md](docs/prompts.md), [docs/skills.md](docs/skills.md)). `hooks` run around every agent's runs, model calls and tool calls. `judges` are online evaluators that score a sampled share of successful runs in the background ([docs/evaluation.md](docs/evaluation.md)). `async with` (or `await h.aclose()`) drains the background writes and closes the clients. `h.agents` is every agent it wraps, by id. |
| `Settings(bifrost_url=, bifrost_virtual_key=, api_key=, memory_url=, runs_url=, otlp_endpoint=, otlp_headers=, spool_dir=, worker_concurrency=, agent_version=, grounding_sample=, judge_model=, judge_virtual_key=, judge_sample=, prompts_dir=, skills_dir=, langfuse_host=, langfuse_public_key=, langfuse_secret_key=)` | The deployment as fields (every one optional); `Settings.from_env()` is what `Harness()` reads ([docs/configuration.md](docs/configuration.md)). |
| `h.wrap(target, *, id, tools=(), version=None, mcp=None, skills=(), timeout=None, without=(), hooks=()) -> Agent` | Attach the harness. The framework is detected from the target's type. `tools` are the agent's own, run in this process: functions, `a2a(url)`, `openapi(spec)`. `mcp`: the gateway's Virtual MCPs (by slug) whose tools are the agent's MCP tools (else every tool its virtual key allows); `skills`: skills by name (`"name"`, `"name@version"`) from the skill sources — `Harness(skills=)`, else `SKILLS_DIR` and the gateway's Skills Repository — or given (`Skill(...)`), disclosed in the context and read with two tools ([docs/skills.md](docs/skills.md)). `version` is the agent's code version (else `TRELLIS_AGENT_VERSION`): recorded with each run it starts and on its spans; a run resumed on another version goes on with a warning naming both. `timeout`: the most working time of each of its runs, in seconds, however it starts (`run`, `stream`, `start`, chat, A2A, `h.evaluate`, a schedule), unless the run names its own ([docs/reliability.md](docs/reliability.md#run-time-limit-and-deadline)). `without`: what the harness does not do for any of its runs — `memory` (or `memory_push`, `memory_pull`, `records`), `judges`, `grounding`, `hints`, `code_mode`, `skills`, `mcp` ([docs/README.md](docs/README.md#what-is-on-and-how-to-turn-it-off)); `mcp=[]` is no MCP tools. `hooks`: this agent's hooks, after the harness's. |
| `await h.tools(*sources, framework=..., mcp=None)` | The toolbox as the framework's own tools, for an agent built with them before wrapping (a compiled LangGraph graph binds its tools): LangChain tools (`"langgraph"`, `"deepagents"`), `FunctionTool`s (`"openai_agents"`), one in-process MCP server (`"claude_agent_sdk"`). It holds `sources` (`skills(...)` among them), the MCP tools (`mcp=`: those of these Virtual MCPs) and the memory tools. |
| `await h.prompt(ref, /, **vars) -> str`, `await h.prompt_messages(ref, /, **vars) -> list[dict]` | The prompt `ref` names (`"name"`, `"name@version"`, a `Prompt`) from the first prompt source that has it — `Harness(prompts=)`, else `PROMPTS_DIR`, Langfuse, the gateway — with its `{{variables}}` filled: any framework's instructions. Inside a run it is pinned (journaled, so a resume reads the same text) and said (a `prompt` event) ([docs/prompts.md](docs/prompts.md)). |
| `await h.model_headers(*, prompt=None) -> dict[str, str]` | The headers for a framework's own model client pointed at the gateway (`default_headers=`): the deny-all MCP scope (the gateway adds no MCP tool to its requests and runs none itself) and, with `prompt` (`"name"`, `"name@version"`), that stored prompt's selection ([docs/gateway.md](docs/gateway.md)). |
| `h.worker(agents, *, concurrency=None)` | Claims queued runs of these agents and executes them, `concurrency` at a time (default `TRELLIS_WORKER_CONCURRENCY`, else the CPU count from 1 to 8): `await worker.run()` (until `worker.stop()`: the runs held finish, or are released after a grace period), `await worker.serve()` (the same, stopped by `SIGTERM`/`SIGINT`) or `await worker.run_once()`. The loop is agent-runs' SDK's `trellis.runs.Worker` running wrapped agents ([docs/runs.md](docs/runs.md#workers)). |
| `await h.inbox(assignee=None, *, tenant=None) -> list[RunSummary]` | The paused runs waiting on `assignee` (`user:…`, `role:…`) or on anyone, newest first, as `trellis.runs.RunSummary` (pip `trellis-runs`, the agent-runs SDK); a sub-agent's question once, on its parent's run. `tenant` only for a platform key. |
| `await h.feedback(run_id, verdict, correction=None, *, tenant=None) -> Feedback \| None` | What a person said about a run (`verdict`: `confirm`, `approve`, `reject`, `correct` or `edit`, the last two with a `correction`): a `feedback` score on its trace (Langfuse; 1.0, 0.5 for `edit`, 0.0 for `correct`/`reject`) and — memory on — the run's `human` feedback in the memory service, returned as stored (`None` with memory off). It waits for the tenant administrator (`review.state == "pending"`) before it changes what memory learned. |
| `await h.evaluate(agent, dataset, evaluators, *, run_name=None, description=None, metadata=None, concurrency=4, limit=None, user=None) -> EvalReport` | Run the agent on every item of a dataset — a Langfuse dataset's name, or `[{"input", "expected"?, "metadata"?}]` / `EvalItem`s — through the normal pipeline, `concurrency` at a time, score each answer with `evaluators` onto its run's trace, and make each run an item of the Langfuse experiment `run_name` (the dataset run link on Langfuse v3, the `langfuse.experiment.*` span attributes on v4). A failing or pausing item is reported (`error`, `interrupted`), never fatal. The `EvalReport` has every item in order and each evaluator's mean, count and failures. The evaluation names (the evaluators, `EvalItem`, `EvalReport`...) are imported from `trellis.harness.evals` ([docs/evaluation.md](docs/evaluation.md)). |
| `await h.add_document(file, *, user, tenant=None, thread=None, title=None, visibility=None, wait=60) -> DocumentInfo` | Add a file (bytes, a path, or `(filename, bytes, media_type)`) to a user's document memory (or one thread's), waiting until it is indexed (`wait=None`: return at once): the user's next context cites it. `visibility` widens it (`WORKSPACE`, `TENANT`); `tenant` only for a platform key. Needs `MEMORY_URL`. |
| `tool(fn, *, name=None, description=None, side_effects="write", idempotent=False, timeout=None)` / `@tool` / `@tool(...)` | A Python function (sync or async) as a tool: the schema from its signature (pydantic validates the arguments), the description from its docstring's first paragraph, `side_effects` `"read"`, `"write"` (default) or `"irreversible"`, `idempotent` when a call repeated with its idempotency key has its effect once (retried like a read, run again after a crash), `timeout` the most seconds one call may take (a sync function runs in a worker thread). It stays callable as the function. A bare function in `tools=[...]` is `tool(fn)`. |
| `a2a(url, *, name=None, timeout=120)` | A remote A2A agent (its card at `{url}/.well-known/agent-card.json`) as one `write` tool, `{"message": string}` in, its answer out; `name` overrides the card's; `timeout` bounds one exchange (the one default of a remote tool, `openapi`'s too). |
| `openapi(spec, *, only=None, base_url=None, headers=None, timeout=120)` | The operations of an OpenAPI 3 document (a URL or the parsed document) as tools, one per `operationId` (`only` keeps those named); `base_url` when the document names no server; `headers` on every request; `timeout` seconds per operation. An operation that does more than read sends the call's `Idempotency-Key`. |
| `ReAct(system, model, output=None, max_steps=12, *, max_result_chars=20000, max_repeats=3, model_timeout=None, context_window=None, prompt=None, prompt_vars=None)` | A tool-calling loop over chat completions, for teams with no framework: `model` is a Bifrost model name (needs `BIFROST_URL`) or any object with `async complete(messages, **body)`; `output` a pydantic model for a structured answer. Several calls in one step: the reads at once, then the writes one at a time in the model's order. Arguments that are not JSON or do not fit the tool's schema are an error the model reads (the tool does not run); a result longer than `max_result_chars` keeps its head and tail (the whole of it kept as a run artifact, read back with `read_result`); past half of the model's window (`context_window`, else the model object's, else 128k tokens) older results are cleared, past three quarters older turns compacted into one summary; at `max_steps` the model answers once more without tools; the same call in `max_repeats` consecutive steps, or 3 steps in which every call failed, stop the run; a resume replays the model steps already taken; a model call takes at most `model_timeout` seconds (else a retryable `ModelError`); `prompt` names a prompt (`"name"`, `"name@version"`, a `Prompt`) from the prompt sources, pinned for the run: rendered into the instructions with `prompt_vars`, or — a stored prompt of the gateway's — selected by every model call ([docs/frameworks/react.md](docs/frameworks/react.md), [docs/prompts.md](docs/prompts.md)). |
| `skills(*refs) -> Skills` | Skills by name or given (`Skill(...)`) as a tool source (`h.tools(skills("sql-review"), framework=...)` for a graph; `h.wrap(..., skills=[...])` otherwise): `load_skill(name)` and `read_skill_file(name, path)`, each skill pinned at a run's start ([docs/skills.md](docs/skills.md)). `Prompt`, `prompts_dir`, `langfuse_prompts`, `PromptSources` are in `trellis.harness.prompts`; `Skill`, `skills_dir`, `SkillSources` in `trellis.harness.skills`. |
| `current() -> Runtime \| None` | Inside a tool or a node: the run it executes in. |
| `Hooks`, `Deny(reason)`, `Ask(question, *, assignee=None)`, `Rewrite(args)`, `ModelCall` | Your code around runs, model calls and tool calls: subclass `Hooks` and override `on_run_start(run)`, `on_run_end(run, result)`, `before_model(call) -> ModelCall \| None`, `after_model(call, reply)`, `before_tool(call) -> None \| Deny \| Ask \| Rewrite`, `after_tool(call, outcome) -> outcome`, `on_error(stage, error)`. Tool hooks run in the bridge (every adapter; `governed(..., hooks=)` in Way 2), run hooks in the pipeline, model hooks through each framework's own mechanism ([docs/hooks.md](docs/hooks.md)). |

`Agent` — what `wrap` returns:

| Method | |
|---|---|
| `await run(input, *, user, thread=None, tenant=None, timeout=None, deadline=None, without=()) -> Result` | Run to the end or the first pause. `without`: what the harness does not do for this run, on top of the agent's (kept with the run: its resume and sub-agents too). `tenant` only for a platform key (one with no tenant of its own). `timeout`: the most working time in seconds, across attempts (pauses not counted; else the agent's `h.wrap(timeout=)`); `deadline`: when it must have ended; past either it ends `TIMEOUT` ([docs/reliability.md](docs/reliability.md)). |
| `stream(...) -> AsyncIterator[RunEvent]` | The run's events (contracts `RunEvent`s) up to `RUN_FINISHED`. Closing it early cancels the run. |
| `await start(...) -> RunHandle` | Queue the run for a worker (its input must be JSON). The `RunHandle`: `run_id`, `await handle.status()` (the `RunRecord`), `await handle.cancel(reason=None)` and `await handle.result(timeout=None)`, which waits for a pause or an ending and returns a `Result`. |
| `await execute(job) -> Result` | The next attempt of a run of this agent a worker claimed (`job`: a `trellis.runs.Job`) — with its journal, governance, memory and time limit. What `h.worker` does with each claimed run; any worker of yours does the same: `trellis.runs.Worker(runs, agent.execute, [agent.id])`, or a loop around `runs.claim` ([docs/runs.md](docs/runs.md#workers)). Another agent's job is refused. |
| `await cancel(run_id, *, reason=None, tenant=None) -> RunRecord` | Cancel a run whatever it is doing: queued or paused, at once; running here, now; on a worker elsewhere, by that worker at its next heartbeat; its sub-agents' runs with it. An ended run is a `ConflictError`. |
| `await resume(interrupt_id, decision, *, answer=None, reviewer, tenant=None) -> Result` | Answer the pause (`decision`, a string or contracts `InterruptDecision`: `answer`, `approve`, `reject`, `edit` with the edited arguments as `answer`, `cancel`). An answer that does not fit the question (`expects`, `options`) or an edit that does not fit the tool's schema is refused (`ConfigurationError`, saying why) and the run keeps waiting. A run started in process continues here; a queued run goes back to the queue (`QUEUED`). `tenant` only for a platform key. |
| `as_tool(*, name=None, description=None, side_effects=None)` | This agent as a tool of other agents, any framework (`tools=[agent.as_tool()]`, `h.tools(agent.as_tool(), framework=...)`): each call is a child run of it — its parent, tenant, user, thread, time and trace inherited; its question pauses the parent, whose `resume` answers it; continued after a crash; cancelled with the parent. The tool reads when every tool the agent declares reads, else writes ([docs/subagents.md](docs/subagents.md)). |
| `await schedule(cron, input, *, on_behalf_of, tz="UTC", tenant=None) -> Schedule` | Queue a run on a cadence (cron, or `hourly`/`daily`/`weekly`/`weekdays`/`manual`), acting for `on_behalf_of`. The same agent, person, cadence and input are one schedule. |
| `serve_chat(app, *, path="/agui", identity=None)` | AG-UI routes on a FastAPI app (run, reconnect/replay, artifacts); `identity(request) -> user` (sync or async), else every caller is `anonymous` ([docs/surfaces.md](docs/surfaces.md)). |
| `serve_a2a(app, url, *, identity=None)` | The agent card and A2A JSON-RPC routes at `url`; `identity(call_context) -> user`, else the trusted `x-trellis-identity` header ([docs/surfaces.md](docs/surfaces.md)). |

`Runtime` — `trellis.current()` inside a tool or a node (`None` outside a run), and the second
argument of a function target: `run_id`, `agent_id`, `user`, `thread`, `tenant`, `attempt`,
`task` (the question), `context` (the pushed memory context), `memory` (the memory SDK's
verbs in the run's scope; needs `MEMORY_URL`), `idempotency_key` (inside a tool call: that
call's key, the same in every attempt), `remaining()` (the seconds the code may still take, or
`None`), `uses(feature)` (whether the run has that part of the harness on: `without=`),
`await tools.call(name, **args)`,
`await tools.hints(task)`, `await ask(question, *, expects, table, diff, options, assignee,
deadline, escalate_to)` (pauses the run; returns the answer on resume —
[docs/interrupts.md](docs/interrupts.md)) and `log(message, **fields)`.

`Result`: `run_id`, `status` (`SUCCESS`, `PAUSED`, `ERROR`, `TIMEOUT`, `QUEUED`, `CANCELLED`), `answer`,
`interrupt`, `error`. `trellis.runs.RunSummary`: `run_id`, `agent_id`, `status`, `awaiting`,
`assignee`, `deadline`, `updated_at`. `h.runs` is the run store: the one given
(`Harness(runs=)`), else agent-runs' client (`trellis.runs.RunsClient`) with `RUNS_URL`, else the
in-process `LocalRuns` ([docs/runs.md](docs/runs.md)).

Run a worker for every agent a module's harness wraps:

```bash
python -m trellis.harness.worker app.agents:h [--concurrency N]   # SIGTERM stops it gracefully
```

## What happens automatically

| | |
|---|---|
| **Memory** | With `MEMORY_URL`: the context for the question is pushed into the framework's input (without the recent conversation when the framework keeps the thread itself), the memory tools (`memory_search`, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) are added, and the transcript, every tool call and the run's outcome are recorded. `BIFROST_VIRTUAL_KEY` is registered as the agent's memory model key. |
| **MCP tools** | Every tool the virtual key allows (the gateway's own listing for the key; `mcp=`: those of the named Virtual MCPs), each executed through the gateway, one call at a time, saying who the run is for (the trusted `x-trellis-identity` header, forwarded to the servers that accept it). A tool the gateway would run itself (Agent Mode) is not offered; no model call gets the gateway's MCP tools ([docs/gateway.md](docs/gateway.md)). |
| **Governance** | Each call runs, runs and is announced (`tool_notice` event), or asks a person: by the tool's risk — the MCP server's annotations (`readOnlyHint` → read, `destructiveHint` → irreversible, else write) or a local tool's declaration, overridden by the catalog's `risk`; `read` runs, `write` is announced, `irreversible` asks. The catalog's `approve_when` (an administrator's rule, or an accepted approval suggestion, in the memory service's own expression language) replaces that: the call asks exactly when it holds. A catalog that cannot be read fails closed. Every tool is published to the catalog ([docs/governance.md](docs/governance.md)). |
| **Tool hints** | From 5 tools on, the context is asked for with the run's tool names and comes back with the tools that fit the task (each with a 0–1 confidence); the model is offered the memory tools, those and every tool the run already used — per model call for `ReAct` and OpenAI Agents, per run for Claude; a `tool_search` call offers what it finds. |
| **Code Mode** | The Code Mode servers whose tools all only read, from 3 servers or 20 tools, become Bifrost's Code Mode meta-tools, offered under the harness's names so every call goes through the bridge (one script instead of many calls); their nested calls are recorded from the gateway's log. |
| **Prompts** | A prompt named once (`"triage@3"`) is looked up in code (`Harness(prompts=)`), then `PROMPTS_DIR` (`.md` files), then Langfuse (its keys set), then the gateway (`BIFROST_URL`): `ReAct(prompt=)` renders it into the instructions (a gateway prompt is selected on every model call instead), `h.prompt(...)` gives any framework its text, `llm_judge(prompt=)` the judge's; the version and text are pinned per run and journaled, on the `prompt` event and the spans; Langfuse's last good copy stands while it is down ([docs/prompts.md](docs/prompts.md)). |
| **Skills** | Skills from code (`Skill`), `SKILLS_DIR` (`<name>/SKILL.md` folders) and the gateway mix in one run: one `## Skills` section in the pushed context, `load_skill` and `read_skill_file` through the bridge, each skill pinned and journaled with its body for the run; a skill no source can give is a `skills_unavailable` warning ([docs/skills.md](docs/skills.md)). |
| **Outcome** | From how the run ended (`SUCCESS` confirm, `ERROR` reject) as the run's `system` feedback, applied as it arrives; the judge's verdict outranks it, and so does a person's once the tenant administrator approves it. |
| **Grounding** | On a sampled share of successful runs (`TRELLIS_GROUNDING_SAMPLE`, 10 % by default), the answer is checked against the context it was given (`/v1/verify` with its `bundle_id`); the score goes on the run's trace. |
| **Sub-agents** | A wrapped agent in another's `tools=[agent.as_tool()]` runs as a child run of the caller: named by its call, inheriting tenant, user, thread, time and trace, its pauses answered through the parent, its journal saved with the parent's progress, cancelled with it ([docs/subagents.md](docs/subagents.md)). |
| **Reliability** | Every tool call takes at most its `timeout` and what is left of the run's; a call that only reads is retried after an error that may pass, a write runs once with an idempotency key the tool can hand on; a write that timed out, or was running when its worker died, is told to the model as of unknown effect and never run again blind; a stopping worker hands its runs back at once; the last good tool list stands while the gateway is down ([docs/reliability.md](docs/reliability.md)). |
| **Traces** | OTel GenAI spans (`invoke_agent`, `execute_tool`, `chat`) with Langfuse's trace attributes (agent, user, session = thread, run, tenant); every attempt of a run in one trace. |

## Frameworks

| Target | Stream | Pause / resume | Harness tools | Tools narrowed | Memory push |
|---|---|---|---|---|---|
| LangGraph graph, Deep Agents | text deltas and tool events | native `interrupt` / `Command(resume=)` with a checkpointer; re-run against the journal without; `HumanInTheLoopMiddleware` / `interrupt_on` pauses are approvals answered with the harness's decisions | built in with `h.tools(..., framework="langgraph")` (a compiled graph refuses `tools=`) | no (bound at build) | leading system message, one per checkpointed thread |
| OpenAI Agents `Agent` | text deltas and tool events | `ask` → re-run against the journal; the SDK's own `needs_approval` → its `RunState` approved or rejected (with the reason; an edit or an answer as the message the model reads) and continued | added to a copy per run | per turn (`FunctionTool.is_enabled`) | leading `system` message |
| Claude Agent SDK `ClaudeAgentOptions` | assistant text blocks and tool events | `ask` → the CLI is stopped; the resume continues its session (built-ins not run again), the journal answering | in-process MCP server `trellis` (`mcp__trellis__*`); built-ins governed through `can_use_tool` | per run | appended to `system_prompt` |
| `ReAct` | per step, and tool events | re-run against the journal, model steps included (no repeated model call) | per run | per model call | appended to `system` |
| async function `(input, agent)` | tool events | re-run against the journal | `agent.tools.call(...)` | n/a | `agent.context` (and a leading system message for a message list) |

Teams bring their own model objects pointed at Bifrost's OpenAI-compatible endpoint
(`ChatOpenAI(base_url=BIFROST_URL, default_headers=await h.model_headers())`,
`OpenAIChatCompletionsModel(AsyncOpenAI(base_url=..., default_headers=...))`); the harness does
not wrap models. `h.model_headers()` keeps the gateway from adding its MCP tools to them, and
selects a stored prompt (`prompt=`).

Each target has a page — how to add the harness to an existing project, what is automatic,
approvals, streaming, durable runs, surfaces, evaluation and limits:
[docs/frameworks/langgraph.md](docs/frameworks/langgraph.md) (`create_agent`, a hand-built
`StateGraph`, checkpointers, `HumanInTheLoopMiddleware`),
[docs/frameworks/deepagents.md](docs/frameworks/deepagents.md) (sub-agents, planning,
`interrupt_on`, its built-in tools), [docs/frameworks/openai-agents.md](docs/frameworks/openai-agents.md)
(handoffs, `needs_approval`), [docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md)
(the MCP server, Claude Code's built-in tools governed through `can_use_tool`, the session resumed),
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

The evaluation names are imported from `trellis.harness.evals`: the evaluators `grounding()`
(against the run's memory context), `exact_match()` and `contains()` (against `expected`),
`llm_judge(criteria, *, name="llm_judge", prompt=None)` and your own (any `async (EvalCase) -> EvalScore |
None`), and `EvalItem`, `EvalReport`. The same module evaluates and judges code that is not
wrapped ([docs/blocks/evaluation.md](docs/blocks/evaluation.md)).

The judge's model and virtual key are configuration (`TRELLIS_JUDGE_MODEL`,
`TRELLIS_JUDGE_VIRTUAL_KEY`): pick a stronger model than the agent's, on its own budget. See
also [docs/observability.md](docs/observability.md).

## Examples

Each runs with no services (scripted models, runs in process) and uses the real ones when
`BIFROST_URL` / `MEMORY_URL` / `RUNS_URL` are set: `make examples`.

Way 1, wrapped:

| Example | Shows |
|---|---|
| `langgraph_agent.py` | `create_agent` with `h.tools`, no checkpointer: an approval re-run from the journal |
| `langgraph_stategraph.py` | a hand-built `StateGraph` with a checkpointer: a harness approval in the tool node, then the graph's own `interrupt()`, both resumed in place |
| `langchain_hitl_middleware.py` | LangChain's `HumanInTheLoopMiddleware`: an edit, then a reject with a reason |
| `deepagents_agent.py` | Deep Agents with `TodoListMiddleware`: a plan, an approval resumed in place |
| `openai_agents_agent.py` | harness tools added to an OpenAI Agents `Agent` per run, an approval |
| `openai_agents_handoff.py` | a handoff to a specialist built with `h.tools(framework="openai_agents")` |
| `claude_agent_sdk_agent.py` | `ClaudeAgentOptions` with a harness tool (a scripted CLI offline) |
| `react_agent.py` | `ReAct` with a tool and a structured answer |
| `prompts_and_skills.py` | prompts from a folder and from code, skills from a `SKILL.md` folder and from code, read by a `ReAct` and by a function |
| `hooks.py` | hooks on a `ReAct` agent: a guardrail (deny, ask, rewrite) and card numbers masked before the model reads them |
| `react_with_blocks.py` | `ReAct` on blocks of your own: your run store, your scheduler loop (`agent.execute`), your governance, memory off |
| `react_subagents.py` | `agent.as_tool()`: a `ReAct` planner delegating to two agents at once, one asking a person through the planner |
| `cowork.py` | start → worker → ask with a diff → inbox → resume → worker |
| `schedule.py` | a schedule a worker runs (memory on: the person's context) |
| `serve_chat.py` | AG-UI and A2A on one FastAPI app |
| `memory_features.py` | a document the next context cites, `agent.memory`, a person's feedback (memory on) |
| `evaluate_offline.py` | `h.evaluate`: a dataset scored by exact match, contains and a judge; the report |
| `online_judges.py` | `Harness(judges=[...])`: judges on live runs, in the background |
| `reliability.py` | a read retried, a write past its `timeout` reported as of unknown effect (with its idempotency key), a run past its `timeout`, a cancel |

Way 2, blocks (your framework's own objects, not wrapped):

| Example | Shows |
|---|---|
| `blocks_langgraph.py` | a plain LangGraph graph: memory context and recording, `governed` tools asking through `interrupt`, the pause in agent-runs answered from the inbox, `Command(resume=)`, a judge |
| `blocks_openai_agents.py` | a plain OpenAI Agents `Agent`: `needs_approval` from governance, its `RunState` as the run's checkpoint in agent-runs, the inbox, a judge |
| `blocks_claude.py` | a plain Claude Agent SDK `query()`: `can_use_tool` from governance, the session as the checkpoint, the inbox, a resumed session (a scripted CLI offline) |
| `blocks_worker.py` | `trellis.runs.Worker` with a handler of your own: queued runs, a pause, the inbox, the resumed run claimed again; a webhook delivery checked with `verify_signature` |
| `blocks_evaluate.py` | `evaluate` on a plain function over a dataset, and `judge` on one run of your own |

Both ways: `a2a_agents.py` serves a wrapped agent over A2A and consumes it as another wrapped
agent's tool (Way 1), then calls the same agent from plain code with `remote()` (Way 2).

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
`tests/live/test_live_pluggable.py` is Way 2 against the same two services, with no `h.wrap`
anywhere: a plain LangGraph graph using memory, governance (an administrator's rule, LangGraph's
`interrupt`), a durable pause in agent-runs answered from the inbox, signed webhooks, a schedule
run by `trellis.runs.Worker`, evaluation and `remote()`. `tests/live/test_live_mixed.py` puts a
wrapped agent and a plain graph under one rule, one inbox and one memory.

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
