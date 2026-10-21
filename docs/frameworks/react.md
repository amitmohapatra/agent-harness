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
    system="You give weather advice. Use the tool, then answer.",
    model="provider/model",  # a Bifrost model name (needs BIFROST_URL)
    output=Forecast,  # optional: the answer parsed into it
)
agent = h.wrap(target, id="weather", tools=[temperature])
result = await agent.run("Do I need a coat in Oslo?", user="ada")  # result.answer: a Forecast
```

```text
ReAct(system, model, *, output=None, max_steps=12, max_repeats=3, model_timeout=None,
      context_window=None, prompt=None, prompt_vars=None, middleware=(), checkpointer=None,
      tool_guidance=False, answer_check=False, capabilities=None, effort=None,
      fallback_models=())
```

* `model` — a Bifrost model name: a `ChatOpenAI` on `BIFROST_URL` (needed when `ReAct` is
  called) with the agent's virtual key (`BIFROST_VIRTUAL_KEY`), the gateway's deny-all MCP
  scope (the harness gives the tools), `timeout=model_timeout`, 2 retries of its own and the
  window as its profile. A vendor's model (`anthropic/...`, `openai/...`) is asked through the
  vendor's own API on the gateway when the gateway serves it — `ChatAnthropic` on its
  Anthropic endpoint, `ChatOpenAI` on its Responses API — so the vendor's caching, effort and
  tool search work (`native_client`; an empty request probes the endpoint once: nothing is
  generated); otherwise the `ChatOpenAI` above, said once in the log. Or any LangChain chat
  model, used as it is (never replaced).
* `output` — a Pydantic model or a JSON schema the answer is parsed into (LangChain's
  `response_format`: the provider's structured output when LangChain knows the model has it,
  else a tool the model answers with). An answer that does not fit is asked for once more,
  saying what was wrong; a second is a `ModelError` saying so (`structured_output`).
* `model_timeout` — the most one model call may take, in seconds, its client's retries
  included; within what is left of the run's `timeout=`/`deadline=`.
* `context_window` — the model's window in tokens. Unset, it is detected: the model object's
  profile (`max_input_tokens`); for a model name, the gateway's model listing (a listed
  `max_input_tokens`, `context_window` or `context_length`, or the `n_ctx` an OpenAI-compatible
  llama.cpp server reports through it), then the table of known models
  (`trellis.harness.models.WINDOWS`, sourced and dated), else 128k, said once in the log. Every
  threshold below is a share of it.
* `effort` — the reasoning effort of every step of a model that takes one; unset, each step's
  own (`effort`, below).
* `fallback_models` — models tried in order when a model call fails (`fallback`). For a model
  name the gateway falls back: the names are sent with every call as Bifrost's own
  `fallbacks` (as the fallbacks configured on the virtual key are), never a second layer in
  the graph; the model that answered is in the `chat` span and the run's journal
  (`answered_by`). A chat model of your own (or for a model of your own) is tried by
  LangChain's `ModelFallbackMiddleware`. The last one's error is the run's.
* `prompt` — a prompt (`"triage"`, `"triage@3"`, a `Prompt`) from the prompt sources — code,
  `PROMPTS_DIR`, Langfuse, the gateway ([prompts.md](../prompts.md)) — looked up at the run's
  start and pinned for it (a resume reads the same one): a stored prompt of the gateway's is
  selected by every model call, which the gateway prepends (a model name only,
  [gateway.md](../gateway.md#prompts)); any other is rendered (`prompt_vars` fill its
  `{{variables}}`) into the system message before `system`, which may then be `""`.
* `middleware` — more middleware, LangChain's, Deep Agents' or yours (below); one with the name
  of a default one replaces it.
* `checkpointer` — the graph's checkpointer; by default `RunCheckpointer()`, which keeps the
  graph's checkpoint in the run itself.
* `tool_guidance`, `answer_check`, `capabilities` — how to choose among tools, a check of the
  answer before it is final and of a change before it is made, and what the assistant can do
  beyond its tools ([below](#tool-choice)).

## The middleware

**Native, on by default** — LangChain's and Deep Agents' own:

| Middleware | What it does |
|---|---|
| `FilesystemMiddleware` (Deep Agents) on a journal backend | a tool result over 20,000 tokens is saved as a file in the graph's state (`/large_tool_results/<call id>`), a head-and-tail preview in its place; `read_file` reads it by lines (OpenAI Agents and Claude get the same from the harness: [tools.md](../tools.md#large-results)). The files are in a channel the run's journal keeps whole, so they are still there after a pause, a crash or a move to another worker |
| `ContextEditingMiddleware` + `ClearToolUsesEdit` (LangChain) | past half the window, the older tool results (all but the last 3) are cleared from what the model is sent, a placeholder naming `read_result` in their place — only when it frees a tenth of the window, so a provider's prompt cache breaks rarely |
| `create_summarization_middleware` (Deep Agents) | past 85% of the window, the older turns are summarized by one model call, the last 10% kept; the turns it replaced are saved where `read_file` reads them (`/conversation_history/...`); oversized tool arguments in older turns are truncated first; a context overflow is summarized and retried |
| `PatchToolCallsMiddleware` (Deep Agents) | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `response_format` (LangChain) | `output=` |
| parallel tool calls (LangGraph) | the calls of one step run at once, each in its own task |

**Native, on when a task needs them** (`trellis.harness.automatic`) — a long task: from the
model's third call of a run's invocation, or at once when the request asks for steps (a list of
three items or more, "step by step", "a plan"), on a model whose window is at least 32,000
tokens (together they cost about 2,600 tokens a call). A shorter task is sent exactly what it
was before they existed. Each is off with `without=` (the model is not offered it; a call of it
answers that it is not available):

| Middleware (`without=` name) | What it does |
|---|---|
| `TodoListMiddleware` (LangChain), `planning` | the `write_todos` tool and its instructions: a plan the model keeps |
| `FilesystemMiddleware`'s scratch tools (Deep Agents), `files` | `ls`, `write_file` and `edit_file` beside `read_file`: notes and drafts kept as files, one at most 100,000 characters, a run's at most 1,000,000 |
| `SubAgentMiddleware` (Deep Agents), `subagent_task` | the `task` tool: a general-purpose helper with the same tools does a self-contained part in a context of its own. Each call is a child run of the same agent, as `agent.as_tool()` makes one ([subagents.md](../subagents.md)): governed, journaled, continued from its journal after a crash, its question the parent's, cancelled with the parent. A helper starts no helper; at most 4 work at once per run |

A `TodoListMiddleware`, `FilesystemMiddleware` or `SubAgentMiddleware` given to
`ReAct(middleware=)` replaces the automatic one (it is then always on).

**Automatic for every task**, each off with `without=`:

| What (`without=` name) | What it does |
|---|---|
| tool search, `tool_search` | 20 tools or more, or definitions over 40% of the window, narrowed to the tools the task is about, searched again at every step ([tools.md](../tools.md#tool-search)) |
| the vendor's own API, `native_client` | a vendor's model named by name asked through its own API on the gateway (above) |
| effort, `effort` | a model that reasons (its LangChain profile lists effort levels, or `trellis.harness.models.EFFORT_LEVELS`) thinks hard on the steps that decide the run — its first call, the step after a tool failed, an `AnswerCheck` check — and little on the others; `effort=` fixes it. A model that takes no effort (a local one) is sent none |
| the prompt cache, `prompt_cache` | an Anthropic model's calls have their stable start — the instructions, the tools, the history — cached (LangChain's `AnthropicPromptCachingMiddleware`); OpenAI's caching is automatic. Cached tokens are on the `chat` span and the run's budget |
| fallback models, `fallback` | with `fallback_models=`: a failed call tried on each in order — by the gateway for its models, by `ModelFallbackMiddleware` for your own |
| structured output repair, `structured_output` | with `output=`: an answer that does not fit asked for once more |
| the budget, `budget` | once the run's budget ([api.md](../api.md#budget)) — or its parent's — is spent: one more call, without tools, for the best answer with what it has; the run ends `PARTIAL` (`BUDGET_EXCEEDED`) |

**The harness's** (`trellis.harness.middleware`) — only what the native middleware does not do:

| Middleware | What it does |
|---|---|
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (sorted by name; the set only grows within a run, so a prompt cache keeps working) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order. Arguments that do not fit the tool's schema are an error the model reads; the tool does not run |
| `ModelHooks` | the tools of a part the run is `without=` left out of every model call (a graph's own, bound when it was built, included); the `before_model`/`after_model` hooks around every model call (a `before_model` call is the call sent), its `chat` span (the messages redacted, the model, usage, finish reasons), `model_timeout` and what is left of the run's time, the `prompt` pinned for the run, and the first user message kept ahead of a summary; a call whose arguments are not JSON answered with what was wrong (the model asked again), the broken arguments sent back as `{}`. A model error is a `ModelError`, retryable for a timeout, a lost connection, a 408, 409, 429 or 5xx (a queued run is queued again) |
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
| `TodoListMiddleware()` (LangChain) | a `write_todos` plan the model keeps from the first call (by default only a long task gets one) |
| `FilesystemMiddleware()` (Deep Agents) | the whole virtual filesystem (`ls`, `write_file`, `edit_file`, `glob`, `grep`): replaces the default `read_file`-only one |
| `SubAgentMiddleware(...)` (Deep Agents) | the `task` tool with sub-agents of the graph's own, run by Deep Agents (the automatic helper's calls are child runs; a wrapped agent's `as_tool()` is the harness's way, [subagents.md](../subagents.md)) |
| `HumanInTheLoopMiddleware(...)` (LangChain) | the framework's own approvals (the harness's are a tool's `ask`, [interrupts.md](../interrupts.md)) |
| `ToolCallLimitMiddleware(...)` (LangChain) | a cap on the calls of a tool, or of all |
| `PIIMiddleware(...)` (LangChain) | PII redacted, masked or refused in what the model reads |
| `SummarizationMiddleware(...)` (LangChain) | LangChain's summarization in place of Deep Agents' (same name: it replaces it) |
| `LLMToolSelectorMiddleware(...)` (LangChain) | many tools: a model call of its own picks those that fit the request, per model call, among the run's tools (`max_tools=`, `on_parsing_failure="all"` so a reply it cannot read offers every tool) |

The harness middleware in a graph of your own: `HarnessTools()` makes the run's tools the
graph's (`tools=[]` is then enough), `ModelHooks()` gives the hooks and the spans, and
`StepLimit`, `StallGuard`, `read_result()`, `ReadTools` and `RunCheckpointer()` work as above.

```python
from langchain.agents import create_agent
from trellis.harness.middleware import HarnessTools, ModelHooks, RunCheckpointer, StepLimit

graph = create_agent(
    model,
    tools=[],
    system_prompt="...",
    middleware=[HarnessTools(), StepLimit(20), ModelHooks()],
    checkpointer=RunCheckpointer(),
)
agent = h.wrap(graph, id="mine", tools=[quote])
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
| Sub-agents | another wrapped agent in `tools=[agent.as_tool()]` is a call like any other ([subagents.md](../subagents.md)) |
| Skills | with `h.wrap(..., skills=[...])`: their names and descriptions in the context, `load_skill` and `read_skill_file` offered ([skills.md](../skills.md)) |
| Records, grounding, judges, tracing, trajectory | as for every target |

## Approvals, streaming, durable runs, surfaces, evaluation

A tool that asks pauses the run; `agent.resume(...)` continues the graph from its checkpoint
and the approved call runs once. `agent.stream(...)` yields each step's text and the tool
events. `start` + workers, `schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` tools and
`h.evaluate` work as for every target. `llm_judge` asks its own `model=`, else
`TRELLIS_JUDGE_MODEL`; with neither, a `ReAct` built with a model name is judged by that model
(said once: a model grading itself is biased — set a separate, stronger one), and one built
with a model object needs a judge model ([evaluation.md](../evaluation.md#the-judges-model)).

`framework_options=` go into the graph's run config as for any graph (`recursion_limit`,
`configurable`...: [langgraph.md](langgraph.md)); the thread is the run's.

## Run it

* [`examples/01_start/first_agent.py`](../../examples/01_start/first_agent.py) — two tools and an approval.
* [`examples/02_way1_react/agent.py`](../../examples/02_way1_react/agent.py) — every piece on: memory, MCP
  tools, an approval resumed in place, a hook, `timeout`, `framework_options`, `without=`.
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
