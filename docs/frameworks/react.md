# ReAct: a tool-calling agent with no framework of yours

For a team with no agent framework: `ReAct(system, model)` returns a LangChain v1
`create_agent` graph — LangChain's loop, with the native middleware a long run needs and the
harness's own on top. `h.wrap` runs it as it runs any LangGraph graph
([langgraph.md](langgraph.md)): nothing of the loop is the harness's. Install the `react` extra
(`pip install 'trellis-harness[react]'`: LangGraph, LangChain, `langchain-openai` and Deep
Agents' middleware).

`ReAct` runs wrapped, with the blocks of the harness that wraps it — the deployment's
(`Harness()`), or your own: `Harness(runs=..., governance=..., memory=False).wrap(ReAct(...))`,
its queued runs continued by your own scheduler with `agent.execute(job)`
([composition](../README.md#composition-a-harness-is-the-blocks-you-give-it),
[`examples/05_features/own_scheduler.py`](../../examples/05_features/own_scheduler.py)). A team with a loop of
its own that wants only some pieces uses the blocks instead
([docs/README.md](../README.md#way-2-pluggable-blocks-your-framework-our-pieces)); the harness
middleware below also works in any `create_agent` or Deep Agents graph of its own.

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
ReAct(system_prompt, model, *, tools=(), subagents=(), response_format=None, max_steps=12,
      max_repeats=3, call_timeout=None, context_window=None, effort=None, tool_guidance=False,
      answer_check=False, capabilities=None, middleware=(), checkpointer=None)
```

* `system_prompt` — the instructions: a string, or `stored("triage", team="EU")`, the
  gateway's stored prompt looked up at the run's start and pinned for it (a resume reads the
  same one): with no variables to fill and a model name routed through Bifrost, every model
  call selects it (the gateway prepends it); any other is rendered into the system message
  ([prompts.md](../prompts.md)).
* `tools` — what the agent can call: functions, `tool(...)`, LangChain tools (the graph's own,
  governed as its tool node runs them), `mcp(...)`, `openapi(...)`, `sandbox()`
  ([tools.md](../tools.md)). Every run also has the MCP tools the virtual key allows and,
  memory on, the memory tools.
* `subagents` — agents it hands tasks to: wrapped agents (`h.wrap(...)`) and `a2a(url)`, each a
  tool whose call is a child run ([subagents.md](../subagents.md)).

* `model` — a Bifrost model name: a `ChatOpenAI` on the gateway's chat completions
  (`BIFROST_URL`, needed when `ReAct` is called) with the agent's virtual key
  (`BIFROST_VIRTUAL_KEY`), the gateway's deny-all MCP scope (the harness gives the tools),
  `timeout=call_timeout` and no retries of its own: the gateway owns retries (a provider's
  `network_config.max_retries`, 0 by default — set it), fallbacks (the virtual key's) and
  prompt caching (`prompt_cache.auto_inject`). Or any LangChain chat model, used as it is
  (never replaced).
* `response_format` — a Pydantic model or a JSON schema the answer is parsed into (LangChain's:
  the provider's structured output when LangChain knows the model has it,
  else a tool the model answers with, asked again when its answer does not fit — within
  `max_steps`).
* `call_timeout` — the most one model call may take, in seconds, the gateway's retries and
  fallbacks included; within what is left of the run's `run_timeout=`/`deadline=`.
* `context_window` — the model's window in tokens; given, it wins. Unset, the model's
  LangChain profile (`model.profile["max_input_tokens"]`: a chat model of LangChain's that knows
  the model, or a `ChatOpenAI` whose model name LangChain knows). When neither says, the
  window-scaled parts are off — clearing old tool results and the long-task tools — said once
  in the log, and Deep Agents' summarization uses its own thresholds: give it for a local model
  (a llama.cpp server's `-c`) or a prefixed vendor name. Where the number came from is logged.
* `effort` — the reasoning effort sent with every call of a model name (`reasoning_effort`, the
  same on every step; the gateway passes it on to the provider). For a chat model of your own,
  set it on that model.
* a model name is routed per call as the harness running the agent says — its
  `Harness(config=)` settings over the environment's: through Bifrost, else to the provider
  with its own key ([configuration.md](../configuration.md#where-a-model-name-goes)).
* `middleware` — more middleware, LangChain's, Deep Agents' or yours (below), added to the
  defaults; one with the name of a default one is refused, naming the parameter that sets it
  (`StepLimit`: `max_steps`, `StallGuard`: `max_repeats`, `ModelHooks`: `call_timeout` and
  `system_prompt`...). `max_steps` and `max_repeats` are 1 or more (12 and 3 by default).
* `checkpointer` — for the graph run on its own, outside the harness; wrapped, its runs keep
  their checkpoint in the run itself (`RunCheckpointer()`), and `h.wrap` refuses a `ReAct`
  given one.
* `tool_guidance`, `answer_check`, `capabilities` — how to choose among tools, a check of the
  answer before it is final and of a change before it is made, and what the assistant can do
  beyond its tools ([below](#tool-choice)).

## The middleware

**Native, on by default** — LangChain's and Deep Agents' own:

| Middleware | What it does |
|---|---|
| `FilesystemMiddleware` (Deep Agents) on a journal backend | a tool result over 20,000 tokens is saved as a file in the graph's state (`/large_tool_results/<call id>`), a head-and-tail preview in its place; `read_file` reads it by lines (OpenAI Agents and Claude get the same from the harness: [tools.md](../tools.md#large-results)). The files are in a channel the run's journal keeps whole, so they are still there after a pause, a crash or a move to another worker |
| `ContextEditingMiddleware` + `ClearToolUsesEdit` (LangChain) | a known window only: past half of it, the older tool results (all but the last 3) are cleared from what the model is sent, a placeholder naming `read_result` in their place — only when it frees a tenth of the window, so a provider's prompt cache breaks rarely |
| `create_summarization_middleware` (Deep Agents) | past 85% of the window (Deep Agents' own thresholds when the window is not known), the older turns are summarized by one model call, the last 10% kept; the turns it replaced are saved where `read_file` reads them (`/conversation_history/...`); oversized tool arguments in older turns are truncated first; a context overflow is summarized and retried. |
| `PatchToolCallsMiddleware` (Deep Agents) | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `response_format` (LangChain) | `response_format=` |
| parallel tool calls (LangGraph) | the calls of one step run at once, each in its own task |

**Native, on when a task needs them** (`trellis.harness.automatic`) — a long task: from the
model's third call of a run's invocation, or at once when the request asks for steps (a list of
three items or more, "step by step", "a plan"), on a model whose window is known and at least
32,000 tokens (together they cost about 2,600 tokens a call). A shorter task is sent exactly what it
was before they existed:

| Middleware | What it does |
|---|---|
| `TodoListMiddleware` (LangChain) | the `write_todos` tool and its instructions: a plan the model keeps |
| `FilesystemMiddleware`'s scratch tools (Deep Agents) | `ls`, `write_file` and `edit_file` beside `read_file`: notes and drafts kept as files, one at most 100,000 characters, a run's at most 1,000,000 |

They are `ReAct`'s own: a `TodoListMiddleware` or `FilesystemMiddleware` given to
`ReAct(middleware=)` is refused. Delegation is `subagents=[...]`
([subagents.md](../subagents.md)).

**With a budget** (`h.wrap(budget=Budget(...))`, [api.md](../api.md#budget)): once the run's —
or its parent's — is spent, one more call without tools for the best answer with what it has;
the run ends `PARTIAL`. No budget given, no cap.

**The harness's** (`trellis.harness.middleware`) — only what the native middleware does not do:

| Middleware | What it does |
|---|---|
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (sorted by name; the set only grows within a run, so a prompt cache keeps working) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order. Arguments that do not fit the tool's schema are an error the model reads; the tool does not run |
| `ModelHooks` | the tools of a part the run is `without=` left out of every model call (a graph's own, bound when it was built, included); the `before_model`/`after_model` hooks around every model call (a `before_model` call is the call sent), its `chat` span (the messages redacted, the model, usage, finish reasons), `call_timeout` and what is left of the run's time, the stored `system_prompt` pinned for the run, and the first user message kept ahead of a summary; a call whose arguments are not JSON answered with what was wrong (the model asked again), the broken arguments sent back as `{}`. A model error is a `ModelError`, retryable for a timeout, a lost connection, a 408, 409, 429 or 5xx (a queued run is queued again) |
| `StepLimit` | after `max_steps` model calls the model is asked once more, with `tool_choice: "none"`, for its best answer with what it has; a `warning` event (`max_steps`) and a log line say so. No answer then fails the run |
| `StallGuard` | the same call in `max_repeats` consecutive steps, or 3 consecutive steps in which every call failed (no such tool, arguments that do not fit, an error, a timeout), stop the run with a `ModelError` |
| `read_result` (a tool) | reads a cleared tool result again, by its call's id, in pieces |
| `ReadTools` | `read_file` offered only once there is a file to read, `read_result` once a result was cleared: a small model offered them from the start reads nothing, again and again (both stay callable) |
| `RunCheckpointer` | the graph's checkpoint kept in the run's journal, the latest only: a resume — after a pause, a crash or a requeue — continues the graph where it stopped |
| `ToolGuidance` (`tool_guidance=True`) | how to choose among tools, added to the instructions of every call that offers tools ([below](#tool-choice)) |
| `AnswerCheck` (`answer_check=True`) | the draft answer checked against the request once, the tools still needed called ([below](#tool-choice)) |

**Yours, opt-in** — `ReAct(..., middleware=[...])`, placed before `ModelHooks` (so its span
shows what the model is sent). Each works the same in any `create_agent` or Deep Agents graph:

| Add | For |
|---|---|
| `FilesystemMiddleware()` (Deep Agents) | the whole virtual filesystem (`ls`, `write_file`, `edit_file`, `glob`, `grep`): in a `create_agent` or Deep Agents graph of your own (`ReAct` has the `read_file`-only one and refuses a second of that name) |
| `SubAgentMiddleware(...)` (Deep Agents) | the `task` tool with sub-agents of the graph's own, run by Deep Agents, each call a child run ([subagents.md](../subagents.md)) |
| `HumanInTheLoopMiddleware(...)` (LangChain) | the framework's own approvals (the harness's are a tool's `ask`, [interrupts.md](../interrupts.md)) |
| `ToolCallLimitMiddleware(...)` (LangChain) | a cap on the calls of a tool, or of all |
| `PIIMiddleware(...)` (LangChain) | PII redacted, masked or refused in what the model reads |
| `SummarizationMiddleware(...)` (LangChain) | LangChain's summarization, in a graph of your own (`ReAct` has Deep Agents' under that name, and refuses a second) |
| `LLMToolSelectorMiddleware(...)` (LangChain) | many tools: a model call of its own picks those that fit the request, per model call, among the run's tools (`max_tools=`, `on_parsing_failure="all"` so a reply it cannot read offers every tool) |

The harness middleware in a graph of your own: `HarnessTools()` makes the run's tools — the
gateway's MCP tools, memory's — the graph's, beside its own, `ModelHooks()` gives the hooks and the spans, and
`StepLimit`, `StallGuard`, `read_result()`, `ReadTools` and `RunCheckpointer()` work as above.

```python
from langchain.agents import create_agent
from trellis.harness.middleware import HarnessTools, ModelHooks, RunCheckpointer, StepLimit

graph = create_agent(
    model,
    tools=[quote],  # LangChain tools: governed as the graph runs them
    system_prompt="...",
    middleware=[HarnessTools(), StepLimit(20), ModelHooks()],
    checkpointer=RunCheckpointer(),
)
agent = h.wrap(graph, id="mine")
```

## Tool choice

A model given several tools goes wrong in typical ways, whatever the tools: it reaches for a
broad tool (a dashboard) when a specific one answers; it answers one part of a several-part
request and drops another; it answers a request to change something in text, never calling the
tool — so the change is not made and its approval never asked; it draws a conclusion the
results do not support. Two switches, off by default, guard against them — both LangChain
middleware, which a `create_agent` or Deep Agents graph of your own takes too:

| | What it does | Costs |
|---|---|---|
| `tool_guidance=True` — `ToolGuidance()` | adds to the instructions of every call that offers tools (`TOOL_GUIDANCE`): address every part of the request, each with the tool made for it; the most specific tool over a general one; current figures from a tool, not the remembered context, which gives background and rules; a change made by calling its tool — the system asks for approval itself — never claimed without its success; conclusions from the results, and a plain "cannot" for a part no tool answers. Inside a run it names the offered tools that change data (their declared side effects) | a few hundred tokens of instructions, the same on every call (a prompt cache holds) |
| `capabilities={...}` — `ToolGuidance(capabilities=)` | what the assistant can and cannot do beyond its tools (`{"accept an image upload": False}`), one line each in every call's instructions, so "can you ...?" gets a plain yes or no; with `tool_guidance` off, only these lines | a line per capability |
| `answer_check=True` — `AnswerCheck()` | when the model answers without calling a tool, its draft is not final (nor streamed): the model is shown it and asked once to check it against the request (`ANSWER_CHECK`) — every part answered by a tool result, the context, or a plain "cannot"; a change made by its tool (the run's tools that change data named); the conclusions following from the results — and either calls the tools still needed, the loop going on, or gives the final answer in its place. A run whose tool hints narrowed its tools is offered all of them from the check on (only those that change data past 128), so a part the hints did not foresee is still served. When nothing found is about what was asked — every tool result of the request empty or "not found", or no tool called and memory `INSUFFICIENT` — the check says so and asks for a plain "no record" (`NO_RECORD`). **A change, before it is made**: the first step of a run calling a tool that changes data is drafted (not streamed) and the model asked to check the call against the rules in its instructions and context (`WRITE_CHECK`) — before governance asks anyone to approve it: it makes the call again, or tells the user which rule forbids it (`AnswerCheck(writes=False)` turns this half off) | one more model call per run for the answer (`AnswerCheck(checks=n)`: up to n), one for the first change; after the step limit's last call the check can only correct the text (no tool is offered) |

Both are generic: they name no tool, and read only what the run already knows (the request,
the results, which tools change data). What they cannot do is tell two tools apart that are
described alike: that is the tools' descriptions ([tools.md](../tools.md#tool-choice-what-helps-the-model-pick-the-right-tool),
where the toolbox's check of the tool set, and the native tool selector for many tools, are).
`tool_coverage()` measures the result per item ([evaluation.md](../evaluation.md#trajectories)).

```python
target = ReAct(
    system,
    "provider/model",
    tool_guidance=True,
    answer_check=True,
    capabilities={"accept an image upload": False},
)
```

Native pieces used rather than rebuilt: LangChain's `TodoListMiddleware` for a plan the model
keeps on a long request, `LLMToolSelectorMiddleware` for many tools, `tool_choice` (the step
limit's last call is `"none"`). A first call forced to `"required"` is not offered: a request
that needs no tool (a greeting, a rule the context holds) would be made to call one.

## What else is automatic

| | |
|---|---|
| Memory push | the context is a system message after `system` |
| Memory pull | the memory tools are offered to every model call |
| Tool hints | each model call is offered the tools offered at that moment (the hinted ones, the memory tools, those already used, what a `tool_search` found) |
| Model steps | the graph's checkpoint is the run's: a resume continues from the last step, so the model is not asked again for what it already decided |
| Several calls in one step | the read-only calls run at once; the others one at a time, in the model's order; a call that pauses lets the others finish (journaled) before the run pauses, and the pauses of one step come one by one |
| Sub-agents | another wrapped agent in `subagents=[agent]` is a call like any other ([subagents.md](../subagents.md)) |
| Skills | the ones the agent learned, with its memory context ([skills.md](../skills.md#learned-skills)) |
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

`subagents=[...]` takes wrapped agents of any framework and `a2a(url)`: each is a tool named
after the agent's id and described by `h.wrap(description=)` (else its docstring), and each
call is a child run of the run — its own record, journal and memory scope, its pauses asked
through the run, its crash continued, its cancel with the run. The reads among them run at
once, the writes one at a time ([subagents.md](../subagents.md)). A `ReAct` is any other
parent's sub-agent the same way. Runnable:
[examples/05_features/subagents.py](../../examples/05_features/subagents.py),
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Limits

* `middleware=` adds to the defaults: one of a default's name is refused (its parameter sets
  it), and so are `TodoListMiddleware` and `FilesystemMiddleware`, which are `ReAct`'s own.
* `checkpointer=` is for the graph run on its own: wrapped, the run keeps the checkpoint, and
  `h.wrap` refuses a `ReAct` given one.
* `subagents=` takes wrapped agents and `a2a(url)`; a Deep Agents sub-agent spec or a graph is
  wrapped first (`h.wrap(graph, id=...)`).
* Without a known window (`context_window=`, or the model's LangChain profile), clearing old
  tool results and the long-task tools are off, said once.
* A model name needs Bifrost or a provider prefix (`openai/...`) whose key the provider's SDK
  reads ([configuration.md](../configuration.md#where-a-model-name-goes)).

## Run it

* [`examples/01_start/first_agent.py`](../../examples/01_start/first_agent.py) — two tools and an approval.
* [`examples/02_way1_react/agent.py`](../../examples/02_way1_react/agent.py) — every piece on: memory, MCP
  tools, an approval resumed in place, a hook, `run_timeout`, `framework_options`, `without=`.
* [`examples/02_way1_react/structured_output.py`](../../examples/02_way1_react/structured_output.py) — a tool, then a structured answer.
* [`examples/06_scenarios/tool_choice.py`](../../examples/06_scenarios/tool_choice.py) — the four
  ways tool choice goes wrong, the two switches, `tool_coverage` per item.
* [`examples/05_features/own_scheduler.py`](../../examples/05_features/own_scheduler.py) — your run store, your
  scheduler loop, your governance, memory off.
* [`examples/05_features/subagents.py`](../../examples/05_features/subagents.py) — two sub-agents at once,
  one asking a person.
* Tests: `tests/integration/test_react.py`, `test_react_parallel.py`, `test_react_context.py`,
  `test_middleware.py` (each middleware on a plain `create_agent`), and against the real
  services `tests/live/test_live_matrix.py` (`react`) and `tests/live/test_live_subagents.py`.
