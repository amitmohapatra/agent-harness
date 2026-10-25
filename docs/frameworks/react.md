# ReAct: a tool-calling agent with no framework of yours

For a team with no agent framework: `ReAct(system_prompt, model)` returns a Deep Agents
`create_deep_agent` graph — every Deep Agents feature native (file tools on a backend, large
results saved as files, summarization near the window's end, sub-agents, skills, memory files,
permissions, dangling-call repair, prompt caching) — with the harness's own middleware on top:
[`middleware()`](#middleware-on-your-own-graph), what Deep Agents has no piece for. `h.wrap` runs
it as it runs any LangGraph graph ([langgraph.md](langgraph.md)): nothing of the loop is the
harness's. Install the `react` extra (`pip install 'trellis-harness[react]'`: LangGraph,
LangChain, `langchain-openai` and Deep Agents).

`ReAct` runs wrapped, with the blocks of the harness that wraps it — the deployment's
(`Harness()`), or your own: `Harness(runs=..., governance=..., memory=False).wrap(ReAct(...))`,
its queued runs continued by your own scheduler with `agent.execute(job)`
([composition](../README.md#composition-a-harness-is-the-blocks-you-give-it),
[`examples/05_features/own_scheduler.py`](../../examples/05_features/own_scheduler.py)). A team with a loop of
its own that wants only some pieces uses the blocks instead
([docs/README.md](../README.md#way-2-pluggable-blocks-your-framework-our-pieces)), or gives its
own `create_agent` or `create_deep_agent` graph `middleware=trellis.middleware()`
([below](#middleware-on-your-own-graph)).

```python
from pydantic import BaseModel
from trellis import Harness, ReAct, tool


class Forecast(BaseModel):
    city: str
    celsius: float
    advice: str


@tool(side_effects="read")
def temperature(city: str) -> float:
    """Today's temperature in a city."""
    ...


h = Harness()
target = ReAct(
    system_prompt="You give weather advice. Use the tool, then answer.",
    model="provider/model",  # through Bifrost with BIFROST_URL, else the provider's own key
    tools=[temperature],
    response_format=Forecast,  # optional: the answer parsed into it
)
agent = h.wrap(target, id="weather")
result = await agent.run("Do I need a coat in Oslo?", user="ada")  # result.answer: a Forecast
```

```text
ReAct(system_prompt, model, *, tools=(), subagents=(), skills=None, memory_files=None,
      permissions=None, backend=None, store=None, response_format=None, state_schema=None,
      context_schema=None, cache=None, max_steps=12, max_repeats=3, call_timeout=None,
      context_window=None, effort=None, tool_guidance=None, answer_check=None,
      capabilities=None, middleware=(), checkpointer=None)
```

The names are Deep Agents' where both have one. In four tiers:

| Tier | Arguments | When |
|---|---|---|
| **must** | `system_prompt`, `model` | every agent |
| **usual** | `tools`, `response_format` | most agents |
| **optional** | `subagents`, `skills`, `memory_files`, `backend`, `permissions`, `max_steps`, `call_timeout`, `effort`, `capabilities` | when the agent needs them |
| **advanced** | `middleware`, `state_schema`, `context_schema`, `cache`, `context_window`, `max_repeats`, `tool_guidance`, `answer_check`, `store`, `checkpointer` | rarely: the defaults are measured |

* `system_prompt` — the instructions: a string, or `stored("triage", team="EU")`, the
  gateway's stored prompt looked up at the run's start and pinned for it (a resume reads the
  same one): with no variables to fill and a model name routed through Bifrost, every model
  call selects it (the gateway prepends it); any other is rendered into the system message
  ([prompts.md](../prompts.md)).
* `model` — a model name, routed per call as the harness running the agent says — its
  `Harness(config=)` settings over the environment's: through Bifrost (the agent's virtual key,
  the gateway's deny-all MCP scope — the harness gives the tools — and no retries of its own:
  the gateway owns retries, fallbacks and prompt caching), else to the provider with its own key
  ([configuration.md](../configuration.md#where-a-model-name-goes)). Or any LangChain chat
  model, used as it is (never replaced).
* `tools` — what the agent can call: functions, `tool(...)`, LangChain tools (the graph's own,
  governed as its tool node runs them), `mcp(url)` or `mcp(stdio(...))`, `openapi(...)`
  ([tools.md](../tools.md)). Every run also has the MCP tools the virtual key allows and,
  memory on, the memory tools. A sandbox is a `backend` and an agent a sub-agent: either given
  here is refused, saying where it goes.
* `response_format` — a Pydantic model or a JSON schema the answer is parsed into (LangChain's:
  the provider's structured output when LangChain knows the model has it, else a tool the
  model answers with — within `max_steps`).
* `subagents` — what it hands tasks to, each call a child run ([below](#sub-agents-and-child-runs)):
  Deep Agents' own, through its `task` tool — a `SubAgent` dict (`name`, `description`,
  `system_prompt`, `mode` isolated or fork, `tools`, `model`, `middleware`, `skills`,
  `permissions`, `response_format`) or a `CompiledSubAgent` `{"name", "description",
  "runnable"}` (another `ReAct` among them) — and wrapped agents (`h.wrap(...)`) and `a2a(url)`,
  each a tool of its own.
* `skills`, `memory_files`, `permissions` — Deep Agents' skill sources, its `memory`
  (`AGENTS.md`-style files loaded into the instructions; named so it is not taken for the memory
  service) and its file permission rules, as they are; skills and memory files are read through
  the `backend`.
* `backend` — where the file tools work: by default the graph's state (a `JournalBackend`: the
  run's journal keeps it across a pause, a crash and a move to another worker, a file at most
  100,000 characters and a run's files at most 1,000,000); or a Deep Agents backend of yours —
  `StoreBackend` (the memory service's store when wrapped), `FilesystemBackend`,
  `CompositeBackend` — or the run's sandbox, `docker_sandbox()`, where `execute` runs code
  ([sandbox.md](../sandbox.md)). Files outside the graph's state are the agent's from the start
  and governed as any tool (the reads read, the others write; the catalog may say more).
* `max_steps` (12), `max_repeats` (3, at least 2), `call_timeout` — the loop's bounds: model
  calls with tools then one without, identical calls in a row that stop the run, the most one
  model call may take (within what is left of the run's `run_timeout=`/`deadline=`).
* `context_window` — the model's window in tokens; given, it wins. Unset, the model's
  LangChain profile (`model.profile["max_input_tokens"]`). When neither says, the window-scaled
  parts are off — clearing old tool results and the long-task tools — said once in the log, and
  Deep Agents' summarization uses its own thresholds: give it for a local model (a llama.cpp
  server's `-c`) or a prefixed vendor name.
* `effort` — the reasoning effort sent with every call of a model name (`reasoning_effort`).
* `tool_guidance`, `answer_check`, `capabilities` — how to choose among tools, a check of the
  answer before it is final and of a change before it is made, and what the assistant can do
  beyond its tools; `None` (the default) switches the first two on by a measured rule
  ([below](#tool-choice)).
* `middleware` — more middleware, LangChain's, Deep Agents' or yours, added before `ModelHooks`
  (so its span shows what the model is sent): one named as the harness's is refused, naming the
  parameter that sets it (`StepLimit`: `max_steps`, `StallGuard`: `max_repeats`, `ModelHooks`:
  `call_timeout` and `system_prompt`...); one named as Deep Agents' replaces it, as Deep Agents
  does.
* `state_schema`, `context_schema`, `cache` — Deep Agents' and LangGraph's, as they are.
* `store`, `checkpointer` — for the graph run on its own, outside the harness; wrapped, its runs
  keep their checkpoint in the run itself (`RunCheckpointer()`) and its store is the memory
  service, and `h.wrap` refuses a `ReAct` given either.

Not exposed: Deep Agents' `interrupt_on` — approvals are the [catalog](../governance.md)'s (a
sub-agent's `interrupt_on` becomes its tools' declared risk: a tool named there asks a person) —
`name` (the wrap's `id`) and `debug`.

## Lean by default: Deep Agents' capabilities on demand

A simple task costs what it cost before Deep Agents: the same instructions (Deep Agents' main
agent adds no base prompt), the run's tools and nothing else. Each capability is offered on the
turn that needs it — by the harness's per-call tool set, never always on:

| Capability | Offered when |
|---|---|
| planning (`write_todos`) | a long task: the model's third call, or a request asking for steps, on a window of 32,000 tokens or more |
| scratch files (`ls`, `write_file`, `edit_file`, `delete`, `glob`, `grep`) and the general-purpose helper (`task`) | the same long task (on the default backend) |
| `read_file` | once a result was saved as a file (Deep Agents' eviction of a large result) or a summary saved the history; from the start on a backend of yours, with skills or memory files |
| `read_result` | once an older result was cleared past half the window |
| summarization | near the window's end (Deep Agents' thresholds) |
| `task` with your sub-agents | when you give `subagents=` |
| `execute` | when the backend is a sandbox (`docker_sandbox()`) |
| skills, memory files | when you give `skills=`, `memory_files=` |

The guidance for a capability travels in its tool's description, offered that turn — never in
the instructions — so the system prompt is the same from the first call to the last and a
provider's prompt cache keeps its prefix.

## The middleware

**Deep Agents', native** — `create_deep_agent`'s own stack, as it builds it:

| Middleware | What it does |
|---|---|
| `FilesystemMiddleware` | the file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute` on a sandbox backend) on `backend`; a tool result over 20,000 tokens saved as a file (`/large_tool_results/<call id>`), a head-and-tail preview in its place, `read_file` reading it by lines |
| `SubAgentMiddleware` | the `task` tool: your `SubAgent`/`CompiledSubAgent` sub-agents, and its general-purpose helper |
| summarization (`create_summarization_middleware`) | past 85% of the window (its own thresholds when the window is not known), the older turns summarized by one model call, the turns it replaced saved where `read_file` reads them (`/conversation_history/...`), oversized tool arguments in older turns truncated first, a context overflow summarized and retried. `ReAct` places it after the harness's middleware that narrow the tools, so it counts what the model is sent |
| `PatchToolCallsMiddleware` | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `SkillsMiddleware`, `MemoryMiddleware` | `skills=`, `memory_files=` |
| prompt caching | Anthropic's (and Bedrock's, Fireworks') cache breakpoints, when the model is theirs |
| `HumanInTheLoopMiddleware` | only for a `permissions` rule in `interrupt` mode (the harness asks it once, as an approval) |
| `response_format`, parallel tool calls | LangChain's and LangGraph's |

Deep Agents keeps `messages` and `files` as deltas (LangGraph's `DeltaChannel`); the run's
checkpointer keeps the latest checkpoint only, with each delta channel's last snapshot and the
writes since beside it, so a resume on any worker rebuilds them whole.

**The harness's** — [`middleware()`](#middleware-on-your-own-graph), in this order:

| Middleware | What it does |
|---|---|
| `TodoListMiddleware` (LangChain), `LongTaskTools` | a long task — from the model's third call of an invocation, or at once when the request asks for steps (a list of three items or more, "step by step", "a plan"), on a known window of at least 32,000 tokens — is offered planning (`write_todos`) and, on the default backend, Deep Agents' scratch file tools and its general-purpose helper; a shorter task is sent none of them (together about 2,600 tokens a call) |
| `AnswerCheck` (`answer_check`) | the draft answer checked against the request once, the tools still needed called; the first change checked against the rules ([below](#tool-choice)) |
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (narrowed per turn by the tool hints; sorted by name, so a prompt cache keeps working) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order |
| `GovernTools` | the graph's own tools — your LangChain tools, Deep Agents' `task`, its file tools on a backend outside the state, a sub-agent's tools — governed as the harness's are (catalog, approval, journal, records) |
| `StallGuard` | the same call in `max_repeats` consecutive steps, or 3 consecutive steps in which every call failed, stop the run with a `ModelError` |
| `StepLimit` | after `max_steps` model calls one more, with `tool_choice: "none"`, for the best answer with what it has; a `warning` event (`max_steps`) says so |
| `Budgeted` | with a budget (`h.wrap(budget=)`): once it — the run's or its parent's — is spent, one more call without tools; the run ends `PARTIAL` |
| `ContextEditingMiddleware` (LangChain) + `read_result` | a known window only: past half of it, the older tool results (all but the last 3) cleared from what the model is sent, a placeholder naming `read_result` in their place — only when it frees a tenth of the window, so a prompt cache breaks rarely |
| `ReadTools` | `read_result` offered once a result was cleared, `read_file` once the graph's state holds a file (always, on a backend outside it): a small model offered them from the start reads nothing, again and again |
| `ToolGuidance` (`tool_guidance`, `capabilities`) | how to choose among tools, added to the instructions ([below](#tool-choice)) |
| `ModelHooks` | the run's `before_model`/`after_model`/`on_error` hooks around every model call, its `chat` span (redacted, usage, finish reasons), `call_timeout` and what is left of the run's time, the stored `system_prompt` pinned for the run, the gateway's headers on a routed model, the tools of a part the run is `without=` left out, and the first user message kept ahead of a summary; a call whose arguments are not JSON answered with what was wrong. A model error is a `ModelError`, retryable for a timeout, a lost connection, a 408, 409, 429 or 5xx |

The graph's checkpointer is `RunCheckpointer()`: the latest checkpoint, in the run's journal — a
resume after a pause, a crash or a requeue continues the graph where it stopped. A declarative
sub-agent is given `StallGuard`, `StepLimit` and `ModelHooks` too (its own of those names kept),
and a model name of its own is routed as `ReAct`'s.

**Yours, opt-in** — `ReAct(..., middleware=[...])`. Each works the same in any `create_agent`
or Deep Agents graph:

| Add | For |
|---|---|
| `HumanInTheLoopMiddleware(...)` (LangChain) | the framework's own approvals (the harness's are the catalog's and a tool's `ask`, [interrupts.md](../interrupts.md)) |
| `ToolCallLimitMiddleware(...)` (LangChain) | a cap on the calls of a tool, or of all |
| `PIIMiddleware(...)` (LangChain) | PII redacted, masked or refused in what the model reads |
| `LLMToolSelectorMiddleware(...)` (LangChain) | many tools: a model call of its own picks those that fit the request, per model call (`max_tools=`, `on_parsing_failure="all"`) |
| a Deep Agents middleware by its name (`SummarizationMiddleware`...) | replaces Deep Agents' own, as `create_deep_agent` does |

## middleware() on your own graph

`trellis.middleware()` is the harness's part on its own: give it to a `create_agent` or a
`create_deep_agent` graph of yours, with the run's checkpointer, and `h.wrap` runs it as it runs
`ReAct` — the run's tools per model call (the memory and gateway MCP tools among them), narrowed
per turn, its own tools governed, the step limit and the stall guard, the hooks and spans,
`call_timeout`, the budget's stop, the checkpoint in the run. The settings are `ReAct`'s, meaning
the same: `max_steps`, `max_repeats`, `call_timeout`, `context_window`, `tool_guidance`,
`answer_check`, `capabilities`, and `system_prompt=stored(...)` (a text goes to the graph's own
`system_prompt`).

```python
from deepagents import create_deep_agent
from langchain.agents import create_agent
from trellis import middleware
from trellis.harness.middleware import RunCheckpointer

deep = create_deep_agent(
    model, tools=[quote], middleware=middleware(max_steps=20), checkpointer=RunCheckpointer()
)
plain = create_agent(
    model, tools=[quote], middleware=middleware(call_timeout=30), checkpointer=RunCheckpointer()
)
agent = h.wrap(deep, id="quotes")
```

A `create_agent` or Deep Agents graph wrapped without it still runs governed and narrowed (the
harness runs a copy of its model node that offers only the run's tools), said once: without the
memory and gateway MCP tools, the step limit, the stall guard, the hooks, `call_timeout` and the
checkpoint in the run. Runnable:
[`examples/02_way1_deepagents/middleware.py`](../../examples/02_way1_deepagents/middleware.py).

## Tool choice

A model given several tools goes wrong in typical ways, whatever the tools: it reaches for a
broad tool (a dashboard) when a specific one answers; it answers one part of a several-part
request and drops another; it answers a request to change something in text, never calling the
tool — so the change is not made and its approval never asked; it draws a conclusion the
results do not support. Two switches guard against them — both LangChain middleware, in
[`middleware()`](#middleware-on-your-own-graph) too. Each is `None` by default: on by a rule
measured on a small model (below), `True` always, `False` never.

| | What it does | Costs | `None` (automatic) |
|---|---|---|---|
| `tool_guidance` — `ToolGuidance()` | adds to the instructions of a call that offers tools (`TOOL_GUIDANCE`): address every part of the request, each with the tool made for it; the most specific tool over a general one; current figures from a tool, not the remembered context, which gives background and rules; a change made by calling its tool — the system asks for approval itself — never claimed without its success; conclusions from the results, and a plain "cannot" for a part no tool answers. Inside a run it names the offered tools that change data | a few hundred tokens of instructions, the same on every call (a prompt cache holds) | on a call offering 20 tools or more (`GUIDE_FROM`), on a known window of 32,000 tokens or more (room for its ~450 tokens) |
| `capabilities={...}` — `ToolGuidance(capabilities=)` | what the assistant can and cannot do beyond its tools (`{"accept an image upload": False}`), one line each in every call's instructions, so "can you ...?" gets a plain yes or no | a line per capability | always, when given |
| `answer_check` — `AnswerCheck()` | when the model answers without calling a tool, its draft is not final (nor streamed): the model is shown it and asked once to check it against the request (`ANSWER_CHECK`), and either calls the tools still needed or gives the final answer in its place; when nothing found is about what was asked it asks for a plain "no record". **A change, before it is made**: the first step calling a tool that changes data is checked against the rules in the instructions and context (`WRITE_CHECK`) before governance asks anyone to approve it | one more model call per invocation for the answer, one for the first change | when the run's context holds a rule — the memory context pushed for it, or a system message of the input ("never", "always", "must", a rule, a policy, something forbidden) |

<!-- measured: the rules above, from the Phase 2 Granite batch -->

Both are generic: they name no tool, and read only what the run already knows (the request,
the results, which tools change data). What they cannot do is tell two tools apart that are
described alike: that is the tools' descriptions ([tools.md](../tools.md#tool-choice-what-helps-the-model-pick-the-right-tool),
where the toolbox's check of the tool set, and the native tool selector for many tools, are).
`tool_coverage()` measures the result per item ([evaluation.md](../evaluation.md#trajectories)).

```python
target = ReAct(
    system,
    "provider/model",
    tool_guidance=True,  # always (None: from 20 tools, on a window with room)
    answer_check=False,  # never (None: when the context holds a rule)
    capabilities={"accept an image upload": False},
)
```

Native pieces used rather than rebuilt: LangChain's `TodoListMiddleware` for a plan the model
keeps on a long request, `LLMToolSelectorMiddleware` for many tools, `tool_choice` (the step
limit's last call is `"none"`), Deep Agents' file tools and sub-agents. A first call forced to
`"required"` is not offered: a request that needs no tool (a greeting, a rule the context
holds) would be made to call one.

## What else is automatic

| | |
|---|---|
| Memory push | the context is a system message after `system` |
| Memory pull | the memory tools are offered to every model call |
| Tool hints | each model call is offered the tools offered at that moment (the hinted ones, the memory tools, those already used, what a `tool_search` found) |
| Model steps | the graph's checkpoint is the run's: a resume continues from the last step, so the model is not asked again for what it already decided |
| Several calls in one step | the read-only calls run at once; the others one at a time, in the model's order; a call that pauses lets the others finish (journaled) before the run pauses, and the pauses of one step come one by one |
| Sub-agents | each call a child run, Deep Agents' through `task` and the wrapped and A2A agents as tools ([above](#sub-agents-and-child-runs)) |
| Learned skills | the ones the agent learned, with its memory context ([skills.md](../skills.md#learned-skills)); Deep Agents' `skills=` beside them |
| Store | a `StoreBackend` (and `runtime.store`) is the memory service's, wrapped |
| Records, grounding, judges, tracing, trajectory | as for every target |

## Approvals, streaming, durable runs, surfaces, evaluation

A tool that asks pauses the run; `agent.resume(...)` continues the graph from its checkpoint
and the approved call runs once. `agent.stream(...)` yields each step's text and the tool
events. `start` + workers, `schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` tools and
`h.evaluate` work as for every target. `llm_judge` asks its own `model=`, else the gateway's
judge alias — never the agent's own model ([evaluation.md](../evaluation.md#the-judges-model)).

`framework_options=` go into the graph's run config as for any graph (`recursion_limit`,
`configurable`...: [langgraph.md](langgraph.md)); the thread is the run's. Every framework's runs
are queued, scheduled and cancelled the same way:
[examples/06_scenarios/runs_every_framework.py](../../examples/06_scenarios/runs_every_framework.py).

## Sub-agents and child runs

Every sub-agent's call is a child run of the run — its own record, its memory scope derived from
the run's, its pauses asked through the run, its crash continued, its cancel with the run
([subagents.md](../subagents.md)):

* **Deep Agents' own** — a `SubAgent` dict or a `CompiledSubAgent` (a `ReAct` or any graph, with
  its `name` and `description`), called through Deep Agents' `task` tool; the call is governed
  and becomes a child run (`bridges.helpers`): given the memory context for its task, its tool
  calls governed and recorded under its run. A declarative one gets the loop's guards and its
  model name routed as `ReAct`'s; a tool it names in `interrupt_on` asks a person — its risk,
  as the catalog says (a name that is none of its tools stays Deep Agents' approval, asked once
  through the run). With no sub-agent of its own, the general-purpose helper is a long task's.
* **Wrapped agents and `a2a(url)`** — each a tool named after the agent's id (described by
  `h.wrap(description=)`, else its docstring; an A2A agent by its card), any framework, local or
  remote. The reads among them run at once, the writes one at a time.

```python
from trellis import ReAct, a2a

researcher = {
    "name": "researcher",
    "description": "Finds facts on the web.",
    "system_prompt": "You research one question and answer with sources.",
    "tools": [search],
}
lead = ReAct(
    "You plan the trip.",
    "provider/model",
    subagents=[researcher, h.wrap(booker, id="booker"), a2a("https://pricing.example/a2a")],
)
```

Runnable: [examples/02_way1_react/subagents.py](../../examples/02_way1_react/subagents.py),
[examples/05_features/subagents.py](../../examples/05_features/subagents.py),
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Limits

* `middleware=` adds: one named as the harness's is refused (its parameter sets it).
* `store=` and `checkpointer=` are for the graph run on its own: wrapped, the memory service and
  the run keep them, and `h.wrap` refuses a `ReAct` given either.
* `tools=` takes no sandbox (`backend=docker_sandbox()`) and no agent (`subagents=`); a graph is
  a sub-agent with its name and description (`{"name", "description", "runnable"}`).
* Building a `ReAct` costs what `create_deep_agent` costs (it compiles its general-purpose helper
  as well): about 26 ms against 13 ms for `create_agent` — once per agent, not per run.
* Without a known window (`context_window=`, or the model's LangChain profile), clearing old
  tool results and the long-task tools are off, said once.
* Skills and memory files on the default backend must be in the graph's state; give a
  `FilesystemBackend` (or a store) that holds them.
* The harness runs the graph asynchronously (`ainvoke`): `docker_sandbox()` has no synchronous
  operations.
* A model name needs Bifrost or a provider prefix (`openai/...`) whose key the provider's SDK
  reads ([configuration.md](../configuration.md#where-a-model-name-goes)).

## Run it

* [`examples/01_start/first_agent.py`](../../examples/01_start/first_agent.py) — two tools and an approval.
* [`examples/02_way1_react/agent.py`](../../examples/02_way1_react/agent.py) — every piece on: memory, MCP
  tools, an approval resumed in place, a hook, `run_timeout`, `framework_options`, `without=`.
* [`examples/02_way1_react/subagents.py`](../../examples/02_way1_react/subagents.py) — a Deep Agents
  sub-agent, a wrapped agent and an A2A agent, each a child run.
* [`examples/05_features/sandbox.py`](../../examples/05_features/sandbox.py) — `backend=docker_sandbox()`:
  a script written and run in the run's sandbox.
* [`examples/02_way1_deepagents/middleware.py`](../../examples/02_way1_deepagents/middleware.py) —
  `middleware()` on a `create_deep_agent` and a `create_agent` graph of your own.
* [`examples/02_way1_react/structured_output.py`](../../examples/02_way1_react/structured_output.py) — a tool, then a structured answer.
* [`examples/06_scenarios/tool_choice.py`](../../examples/06_scenarios/tool_choice.py) — the four
  ways tool choice goes wrong, the two switches, `tool_coverage` per item.
* [`examples/05_features/own_scheduler.py`](../../examples/05_features/own_scheduler.py) — your run store, your
  scheduler loop, your governance, memory off.
* Tests: `tests/integration/test_react.py`, `test_react_deep_agents.py`, `test_react_parallel.py`,
  `test_react_context.py`, `test_react_long_tasks.py`, `test_sandbox_backend.py`,
  `test_middleware.py` (each middleware on a plain `create_agent`), and against the real
  services `tests/live/test_live_matrix.py` (`react`) and `tests/live/test_live_subagents.py`.
