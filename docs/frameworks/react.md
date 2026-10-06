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
[`examples/react_with_blocks.py`](../../examples/react_with_blocks.py)). A team with a loop of
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

```python
ReAct(system, model, *, output=None, max_steps=12, max_repeats=3, model_timeout=None,
      context_window=None, prompt=None, prompt_vars=None, middleware=(), checkpointer=None)
```

* `model` — a Bifrost model name: a `ChatOpenAI` on `BIFROST_URL` (needed when `ReAct` is
  called) with the agent's virtual key (`BIFROST_VIRTUAL_KEY`), the gateway's deny-all MCP
  scope (the harness gives the tools), `timeout=model_timeout`, 2 retries of its own and the
  window as its profile. Or any LangChain chat model, used as it is.
* `output` — a Pydantic model the answer is parsed into (LangChain's `response_format`: the
  provider's structured output when LangChain knows the model has it, else a tool the model
  answers with).
* `model_timeout` — the most one model call may take, in seconds, its client's retries
  included; within what is left of the run's `timeout=`/`deadline=`.
* `context_window` — the model's window in tokens, when its profile does not say
  (`max_input_tokens`) and the 128k assumed is wrong for it.
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

## The middleware

**Native, on by default** — LangChain's and Deep Agents' own:

| Middleware | What it does |
|---|---|
| `FilesystemMiddleware(tools=["read_file"])` (Deep Agents) | a tool result over 20,000 tokens is saved as a file in the graph's state (`/large_tool_results/<call id>`), a head-and-tail preview in its place; `read_file` reads it by lines |
| `ContextEditingMiddleware` + `ClearToolUsesEdit` (LangChain) | past half the window, the older tool results (all but the last 3) are cleared from what the model is sent, a placeholder naming `read_result` in their place — only when it frees a tenth of the window, so a provider's prompt cache breaks rarely |
| `create_summarization_middleware` (Deep Agents) | past 85% of the window, the older turns are summarized by one model call, the last 10% kept; the turns it replaced are saved where `read_file` reads them (`/conversation_history/...`); oversized tool arguments in older turns are truncated first; a context overflow is summarized and retried |
| `PatchToolCallsMiddleware` (Deep Agents) | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `response_format` (LangChain) | `output=` |
| parallel tool calls (LangGraph) | the calls of one step run at once, each in its own task |

**The harness's** (`trellis.harness.middleware`) — only what the native middleware does not do:

| Middleware | What it does |
|---|---|
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (sorted by name; the set only grows within a run, so a prompt cache keeps working) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order. Arguments that do not fit the tool's schema are an error the model reads; the tool does not run |
| `ModelHooks` | the `before_model`/`after_model` hooks around every model call (a `before_model` call is the call sent), its `chat` span (the messages redacted, the model, usage, finish reasons), `model_timeout` and what is left of the run's time, the `prompt` pinned for the run, and the first user message kept ahead of a summary. A model error is a `ModelError`, retryable for a timeout, a lost connection, a 408, 409, 429 or 5xx (a queued run is queued again) |
| `StepLimit` | after `max_steps` model calls the model is asked once more, with `tool_choice: "none"`, for its best answer with what it has; a `warning` event (`max_steps`) and a log line say so. No answer then fails the run |
| `StallGuard` | the same call in `max_repeats` consecutive steps, or 3 consecutive steps in which every call failed (no such tool, arguments that do not fit, an error, a timeout), stop the run with a `ModelError`; a step of malformed calls only is answered and the model asked again |
| `read_result` (a tool) | reads a cleared tool result again, by its call's id, in pieces |
| `ReadTools` | `read_file` offered only once there is a file to read, `read_result` once a result was cleared: a small model offered them from the start reads nothing, again and again (both stay callable) |
| `RunCheckpointer` | the graph's checkpoint kept in the run's journal, the latest only: a resume — after a pause, a crash or a requeue — continues the graph where it stopped |

**Yours, opt-in** — `ReAct(..., middleware=[...])`, placed before `ModelHooks` (so its span
shows what the model is sent). Each works the same in any `create_agent` or Deep Agents graph:

| Add | For |
|---|---|
| `TodoListMiddleware()` (LangChain) | a `write_todos` plan the model keeps |
| `FilesystemMiddleware()` (Deep Agents) | the whole virtual filesystem (`ls`, `write_file`, `edit_file`, `glob`, `grep`): replaces the default `read_file`-only one |
| `SubAgentMiddleware(...)` (Deep Agents) | the `task` tool: sub-agents of the graph's own (a wrapped agent's `as_tool()` is the harness's way, [subagents.md](../subagents.md)) |
| `HumanInTheLoopMiddleware(...)` (LangChain) | the framework's own approvals (the harness's are a tool's `ask`, [interrupts.md](../interrupts.md)) |
| `ToolCallLimitMiddleware(...)` (LangChain) | a cap on the calls of a tool, or of all |
| `PIIMiddleware(...)` (LangChain) | PII redacted, masked or refused in what the model reads |
| `SummarizationMiddleware(...)` (LangChain) | LangChain's summarization in place of Deep Agents' (same name: it replaces it) |

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

* [`examples/react_agent.py`](../../examples/react_agent.py) — a tool, then a structured answer.
* [`examples/react_with_blocks.py`](../../examples/react_with_blocks.py) — your run store, your
  scheduler loop, your governance, memory off.
* [`examples/react_subagents.py`](../../examples/react_subagents.py) — two sub-agents at once,
  one asking a person.
* Tests: `tests/integration/test_react.py`, `test_react_parallel.py`, `test_react_context.py`,
  `test_middleware.py` (each middleware on a plain `create_agent`), and against the real
  services `tests/live/test_live_matrix.py` (`react`) and `tests/live/test_live_subagents.py`.
