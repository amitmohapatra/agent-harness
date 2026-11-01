# ReAct: a tool-calling agent with no framework of yours

For a team with no agent framework: `ReAct(system_prompt, model)` returns LangChain's
`create_agent` graph with the harness's own middleware — [`middleware()`](#middleware-on-your-own-graph)
— and dangling-call repair. Ask for a Deep Agents capability — `backend=` (file tools), a
sub-agent of its own (`subagents=`), `skills=`, `memory_files=`, `permissions=` — and it is a
Deep Agents `create_deep_agent` graph instead, every Deep Agents feature native (large results
saved as files, summarization near the window's end, sub-agents, skills, memory files,
permissions, prompt caching), the same middleware on top. Both bases were measured; the plain one
was as good or better on every class of task ([the base](#the-base)). `h.wrap` runs it as it runs
any LangGraph graph ([langgraph.md](langgraph.md)): nothing of the loop is the harness's. Install
the `react` extra (`pip install 'trellis-harness[react]'`: LangGraph, LangChain,
`langchain-openai` and Deep Agents).

`ReAct` runs wrapped, with the blocks of the harness that wraps it — the deployment's
(`Harness()`), or your own: `Harness(runs=..., memory=False).wrap(ReAct(...))`,
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
    model="provider/model",  # to LLM_BASE_URL when set, else the provider's own key
    tools=[temperature],
    response_format=Forecast,  # optional: the answer parsed into it
)
agent = h.wrap(target, id="weather")
result = await agent.run("Do I need a coat in Oslo?", user="ada")  # result.answer: a Forecast
```

```text
ReAct(system_prompt, model, *, tools=(), subagents=(), skills=None, memory_files=None,
      permissions=None, backend=None, response_format=None, state_schema=None,
      context_schema=None, cache=None, max_steps=12, max_repeats=3, call_timeout=None,
      context_window=None, effort=None, middleware=())
```

The names are Deep Agents' where both have one. In four tiers:

| Tier | Arguments | When |
|---|---|---|
| **must** | `system_prompt`, `model` | every agent |
| **usual** | `tools`, `response_format` | most agents |
| **optional** | `subagents`, `skills`, `memory_files`, `backend`, `permissions`, `max_steps`, `call_timeout`, `effort` | when the agent needs them |
| **advanced** | `middleware`, `state_schema`, `context_schema`, `cache`, `context_window`, `max_repeats` | rarely: the defaults are measured |

* `system_prompt` — the instructions: a string, or `stored("triage", team="EU")`, the
  gateway's stored prompt looked up at the run's start and pinned for it (a resume reads the
  same one): with no variables to fill and a model name routed through Bifrost, every model
  call selects it (the gateway prepends it); any other is rendered into the system message
  ([prompts.md](../prompts.md)). What the assistant can and cannot do beyond its tools ("You
  cannot accept an image upload.") is said here, in the instructions.
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
  `CompositeBackend` — or the run's sandbox, `sandbox()`, where `execute` runs code
  ([sandbox.md](../sandbox.md)). Files outside the graph's state are the agent's from the start
  and governed as any tool (the reads read, the others write; the catalog may say more).
* `max_steps` (12), `max_repeats` (3, at least 2), `call_timeout` — the loop's bounds: model
  calls the run makes that may call tools (LangChain's `ModelCallLimitMiddleware` counts them;
  at the limit the model is called once more, told the limit is reached and offered no tool, and
  its answer — what it found, what is left — is the run's, a `step_limit` warning on the
  stream), identical calls in a row that stop the run, the
  most one model call may take (within what is left of the run's `run_timeout=`/`deadline=`; a
  tool's own `call_timeout` bounds a tool call). A cap on the whole run's tool calls, across
  its attempts and sub-agents, is `Budget(max_tool_calls=)` ([runs.md](../runs.md)); a spent
  budget asks the model for one last answer.
* `context_window` — the model's window in tokens; given, it wins. Unset, the model's
  LangChain profile (`model.profile["max_input_tokens"]`). When neither says, the window-scaled
  parts are off — clearing old tool results and the long-task tools — said once in the log, and
  Deep Agents' summarization uses its own thresholds: give it for a local model (a llama.cpp
  server's `-c`) or a prefixed vendor name.
* `effort` — the reasoning effort sent with every call of a model name (`reasoning_effort`).
* `middleware` — your middleware, LangChain's, Deep Agents' or your own, added to the harness's
  before `ModelHooks` (so its span shows what the model is sent): one named as the harness's is
  refused, naming the parameter that sets it (`ModelCallLimitMiddleware`: `max_steps`, `StallGuard`:
  `max_repeats`, `ModelHooks`: `call_timeout` and `system_prompt`, `TodoListMiddleware`: the
  planning a long task gets...), and `middleware()` itself is refused: ReAct already has it.
  One named as Deep Agents' replaces it, as Deep Agents does.
* `state_schema`, `context_schema`, `cache` — Deep Agents' and LangGraph's, as they are.

Its runs keep their checkpoint in the run itself (`RunCheckpointer()`) and its store is the
memory service. A graph with a checkpointer or a store of your own is a `create_agent(...,
middleware=middleware(), checkpointer=..., store=...)` ([below](#middleware-on-your-own-graph)).

Not exposed: Deep Agents' `interrupt_on` — approvals are the [catalog](../governance.md)'s (a
sub-agent's `interrupt_on` becomes its tools' declared risk: a tool named there asks a person) —
`name` (the wrap's `id`) and `debug`.

## The base

Three bases were measured on Granite 3.3 2B (CPU, llama.cpp behind Bifrost, an 8,192-token
window), each with `middleware()` and `FitWindow`, ten different tasks per class:
**A** `create_agent` (today's `ReAct`), **B** `create_deep_agent`, **C** `create_agent` with Deep
Agents' pieces attached and offered per call.

| Class (ten tasks each) | Base | Completed | Answered right | Input tokens / run (± 95% CI) | Model calls | p50 s |
|---|---|---|---|---|---|---|
| simple, 10 tools | A | 9/10 | 8/10 | 2,860 ± 1,400 | 1.8 | 14.7 |
| | B | 10/10 | 8/10 | 2,450 ± 610 | 1.7 | 16.7 |
| | C | 10/10 | 8/10 | 2,450 ± 610 | 1.7 | 15.3 |
| multi-step, 10 tools | A | 10/10 | 10/10 | 2,930 ± 150 | 2.0 | 26.3 |
| | B | 10/10 | 10/10 | 2,930 ± 150 | 2.0 | 26.2 |
| | C | 10/10 | 10/10 | 2,950 ± 150 | 2.0 | 34.8 |
| wide (4–5 parts), 10 tools | A | 9/10 | 8/10 | 3,470 ± 1,100 | 2.1 | 57.6 |
| | B | 9/10 | 8/10 | 3,630 ± 1,200 | 2.2 | 58.0 |
| | C | 8/10 | 7/10 | 3,610 ± 1,200 | 2.2 | 58.9 |
| a ~100k-character result | A | **10/10** | 2/10 | 4,540 ± 2,300 | 2.1 | 67.6 |
| | B | 5/10 | 2/10 | 10,400 ± 7,500 | 2.9 | 175.9 |
| | C | 6/10 | 0/10 | 11,600 ± 7,700 | 3.5 | 125.0 |
| mixed, 50 tools | A | 10/10 | 9/10 | 16,400 ± 2,400 | 2.2 | 160.1 |
| | B | 9/10 | 8/10 | 16,000 ± 2,000 | 2.3 | 159.4 |
| | C | 8/10 | 8/10 | 15,400 ± 2,700 | 2.2 | 184.4 |
| mixed, 100 tools | A | 8/10 | 8/10 | 15,400 ± 2,600 | 2.1 | 336.8 |
| | B | 7/10 | 7/10 | 13,700 ± 1,400 | 2.1 | 311.6 |
| | C | 8/10 | 8/10 | 13,800 ± 2,700 | 2.1 | 348.4 |
| mixed, 50 tools, memory on (hints) | A | 9/10 | 7/10 | 5,590 ± 1,200 | 2.0 | 96.7 |
| | B | 9/10 | 7/10 | 6,350 ± 1,300 | 2.1 | 117.5 |
| | C | 9/10 | 7/10 | 5,840 ± 950 | 2.1 | 94.5 |

Runs that did not complete were stopped by `StallGuard` (the model repeating a call). Tool
coverage was 0.9–1.0 for every base and class; peak memory 145–151 MB for each; building and
wrapping the graph took 15–23 ms for A and C, 39–49 ms for B (Deep Agents compiles its
general-purpose helper). Cached-token share, cost at hosted prices and CPU seconds were within
each other's intervals in every class but the large result, where A read a third of B's tokens.

A simple task costs the same on all three — the same instructions (Deep Agents' main agent
adds no base prompt) and tools. A is `ReAct`: no less accurate on any class, and on a large
result it completed every run where B and C stalled half of theirs, after Deep Agents saved the
result as a file and the model went looking. Deep Agents' capabilities stay a parameter away:

| Capability | When |
|---|---|
| planning (`write_todos`) | a long task: the model's third call, or a request asking for steps, on a window of 32,000 tokens or more |
| `read_file` | once an older result was cleared past half the window (by its call's id: `/large_tool_results/<call id>`), a large result was kept, or the graph's state holds a file |
| file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`), a large result saved as a file, summarization | `backend=` — `JournalBackend()` for the graph's state kept by the run's journal, a disk, a store, or `sandbox()` (with `execute`) |
| `task` with your sub-agents | `subagents=` with a `SubAgent` or `CompiledSubAgent` dict; on a long task, Deep Agents' general-purpose helper too |
| skills, memory files, permissions | `skills=`, `memory_files=`, `permissions=` |

On Deep Agents, the scratch files and the general-purpose helper are offered on a long task only
(on the graph's state), `read_file` once a result was saved as a file, and each capability's
guidance travels in its tool's description, offered that turn — never in the instructions — so
the system prompt is the same from the first call to the last and a provider's prompt cache
keeps its prefix. A run's first scratch write on the graph's state says, in its result, that the file is the
run's own scratch copy, which the user cannot open or download (`middleware.SCRATCH_NOTE`):
asked for a chart, a model had saved an SVG there and told the user to open it.

## The middleware

**Deep Agents', native** — `create_deep_agent`'s own stack, as it builds it:

| Middleware | What it does |
|---|---|
| `FilesystemMiddleware` | the file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute` on a sandbox backend) on `backend`; a tool result over 20,000 tokens saved as a file (`/large_tool_results/<call id>`), a head-and-tail preview in its place, `read_file` reading it by lines |
| `SubAgentMiddleware` | the `task` tool: your `SubAgent`/`CompiledSubAgent` sub-agents, and its general-purpose helper |
| summarization (`create_summarization_middleware`) | past 85% of the window (its own thresholds when the window is not known), the older turns summarized by one model call, the turns it replaced saved where `read_file` reads them (`/conversation_history/...`), oversized tool arguments in older turns truncated first, a context overflow summarized and retried. `ReAct` places it after the harness's middleware that narrow the tools, so it counts what the model is sent |
| `PatchToolCallsMiddleware` | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `SkillsMiddleware`, `MemoryMiddleware` | `skills=`, `memory_files=` |
| prompt caching | Anthropic's (and Bedrock's, Fireworks') cache breakpoints, when the model is theirs; `ReAct` puts `PromptCache` in the Anthropic one's place ([below](#the-middleware)) |
| `HumanInTheLoopMiddleware` | only for a `permissions` rule in `interrupt` mode (the harness asks it once, as an approval) |
| `response_format`, parallel tool calls | LangChain's and LangGraph's |

Deep Agents keeps `messages` and `files` as deltas (LangGraph's `DeltaChannel`); the run's
checkpointer keeps the latest checkpoint only, with each delta channel's last snapshot and the
writes since beside it, so a resume on any worker rebuilds them whole.

**The harness's** — [`middleware()`](#middleware-on-your-own-graph), in this order:

| Middleware | What it does |
|---|---|
| `TodoListMiddleware` (LangChain), `LongTaskTools` | a long task — from the model's third call of an invocation, or at once when the request asks for steps (a list of three items or more, "step by step", "a plan"), on a known window of at least 32,000 tokens — is offered planning (`write_todos`) and, on the default backend, Deep Agents' scratch file tools and its general-purpose helper; a shorter task is sent none of them (together about 2,600 tokens a call) |
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (narrowed once per run by the tool hints, then only growing: a `tool_search` find, a long task's tools; in the order first offered — the first call's by name, the later ones after them — so a provider's prompt cache keeps its prefix) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order. With no `read_file` of the graph's own (no backend), a result over 80,000 characters is kept in the run and previewed, and `read_file` and `query_result` (read-only SQL over its rows) are offered once one is ([tools.md](../tools.md#large-results)); with a backend, `query_result` runs over the result Deep Agents saved in the state |
| `GovernTools` | the graph's own tools — your LangChain tools, Deep Agents' `task`, its file tools on a backend outside the state, a sub-agent's tools — governed as the harness's are (catalog, approval, journal, records) |
| `StallGuard` | the same call in `max_repeats` consecutive steps, or 3 consecutive steps in which every call failed and a failure was met again (the same tool, the same error): the model is told so and called once more with no tool (`tool_choice="none"`), so the user is told what went wrong; no answer then stops the run with a `ModelError`. Failures that each differ are a model correcting itself, bounded by `max_steps` |
| `ModelCallLimitMiddleware` (LangChain, as `AnswerAtLimit`) | `max_steps` model calls; then one more, offered no tool, whose answer ends the run (one extra call at most; LangChain's own limit one call later is the backstop) |
| `Budgeted` | with a budget (`h.wrap(budget=)`): once it — the run's or its parent's — is spent, one more call without tools; the run ends `PARTIAL` |
| `ContextEditingMiddleware` (LangChain) | a known window only: past half of it, the older tool results (all but the last 3) cleared from what the model is sent, a placeholder naming `read_file` (the call's id as its path) in their place — the one clearing pass — only when it frees a tenth of the window, so a prompt cache breaks rarely |
| `ReadTools` | `read_file` offered once a result was cleared, the graph's state holds a file or the run keeps a result (always, on a backend outside it): a small model offered it from the start reads nothing, again and again; a cleared result is paged from the graph's state as a kept one is (100 lines, 20,000 characters a page) |
| `FitWindow` | a request over the window (95% of it, less the reply's tokens; `context_window=`, else the model's profile) made to fit before the call, never a failure for its size ([below](#when-a-request-does-not-fit)) |
| `ModelHooks` | the run's `on_error` hooks for a model call that failed, its `chat` span (redacted, usage, finish reasons), `call_timeout` and what is left of the run's time, the stored `system_prompt` pinned for the run, the gateway's headers on a routed model, the tools of a part the run is `without=` left out, and the first user message kept ahead of a summary; a call whose arguments are not JSON answered with what was wrong. A model error is a `ModelError`, retryable for a timeout, a lost connection, or a status in bifrost-sdk's `RETRYABLE` (408, 409, 425, 429, 500, 502, 503, 504) |

The graph's checkpointer is `RunCheckpointer()`: the latest checkpoint, in the run's journal — a
resume after a pause, a crash or a requeue continues the graph where it stopped.

Last of what changes a request, where Deep Agents puts its own, both bases have `PromptCache`:
LangChain's `AnthropicPromptCachingMiddleware` (the one you give `ReAct(middleware=)`, its
settings kept) run on the chat model the call is made with — a `ChatAnthropic`, or a model name
routed to Anthropic's own SDK or to Anthropic's API as an OpenAI-compatible endpoint — so the tools, the instructions, the request's end and the
answer that ends the history are marked (4 breakpoints). Any other model's request is sent as
it was. Through Bifrost the gateway caches (`prompt_cache.auto_inject`); any other
OpenAI-compatible endpoint's chat completions carry no breakpoints.

A declarative sub-agent is given `StallGuard`, `ModelCallLimitMiddleware`, `ModelHooks` and
`PromptCache` too (its own of those names kept), and a model name of its own is routed as
`ReAct`'s.

**Yours, opt-in** — `ReAct(..., middleware=[...])`, added to the harness's (one named as the
harness's is refused, `middleware()` itself among them). Each works the same in any
`create_agent` or Deep Agents graph:

| Add | For |
|---|---|
| `HumanInTheLoopMiddleware(...)` (LangChain) | the framework's own approvals (the harness's are the catalog's and a tool's `ask`, [interrupts.md](../interrupts.md)) |
| `ToolCallLimitMiddleware(...)` (LangChain) | a cap on the calls of a tool, or of all |
| `PIIMiddleware(...)` (LangChain) | PII redacted, masked or refused in what the model reads |
| `LLMToolSelectorMiddleware(...)` (LangChain) | many tools: a model call of its own picks those that fit the request, per model call (`max_tools=`, `on_parsing_failure="all"`) |
| a Deep Agents middleware by its name (`SummarizationMiddleware`...) | replaces Deep Agents' own, as `create_deep_agent` does |

## middleware() on your own graph

`trellis.middleware()` is the harness's part on its own: give it to a `create_agent` or a
`create_deep_agent` graph of yours (compiled without a checkpointer, it is run with the run's,
`RunCheckpointer()`), and `h.wrap` runs it as it runs
`ReAct` — the run's tools per model call (the memory and gateway MCP tools among them), narrowed
per turn, its own tools governed, the step limit and the stall guard, the hooks and spans,
`call_timeout`, the budget's stop, the checkpoint in the run. The settings are `ReAct`'s, meaning
the same: `max_steps`, `max_repeats`, `call_timeout`, `context_window`, and
`system_prompt=stored(...)` (a text goes to the graph's own
`system_prompt`). A graph built so may have a checkpointer or a store of its own (a `ReAct`
cannot): give it `checkpointer=`/`store=` as LangChain takes them.

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
results do not support. `ReAct` adds no rule of its own to the loop against them: what guides
the choice is the tools' own descriptions and the instructions, and what helps is the
harness's tool set check, the tool hints and, for many tools, the native tool selector
([tools.md](../tools.md#tool-choice-what-helps-the-model-pick-the-right-tool)).
`tool_coverage()` measures the result per item ([evaluation.md](../evaluation.md#trajectories));
why the loop has no such rule is [ADR 0002](../adr/0002-tool-choice.md).

Native pieces used rather than rebuilt: LangChain's `TodoListMiddleware` for a plan the model
keeps on a long request, `LLMToolSelectorMiddleware` for many tools, `tool_choice` (the step
limit's last call is `"none"`), Deep Agents' file tools and sub-agents. A first call forced to
`"required"` is not offered: a request that needs no tool (a greeting, a rule the context
holds) would be made to call one. What the assistant can and cannot do beyond its tools
("You cannot accept an image upload.") goes in `system_prompt`, so a "can you ...?" gets a
plain yes or no (the live tool-choice test says it so).

## When a request does not fit

A run never fails because its request does not fit the model's window. Before each model call,
`FitWindow` counts the request (messages, instructions, tools, as Deep Agents counts them); a
request that fits is sent unchanged — the same object, byte for byte. One that does not is made
smaller, step by step, only as far as it must, in this order:

| Step | What goes | Kept |
|---|---|---|
| `tools` | the offered tools narrowed to the most relevant | those the run used; then the hinted ones; then by the words they share with the request; at least 8 |
| `descriptions` | the kept tools' descriptions cut to their first sentence, no parameter descriptions | the names and parameters |
| `results` | the longest tool results cut to head and tail (clearing is `ContextEditingMiddleware`'s alone; on Deep Agents, none: its summarization offloads them whole to its backend), each kept whole in the run and its cut naming the `read_file` call that reads on from where it starts | the latest |
| `turns` | the older turns (`ReAct`: Deep Agents' summarization summarizes them, the turns saved for `read_file`) | the latest turn whole |
| `memory` | last, only when the latest turn alone still does not fit: the memory context's lowest-ranked items, those its `Most relevant` does not name first | the rules, the standing instructions (profile), the summary, the recent conversation, the learned plan (its skills and tools) |

Each step taken is said on the run (a `warning` event, code `context_fit`, naming the steps) and
in the log. The count is an estimate (Deep Agents'): when the model's server still refuses the
request as over its window — its tokenizer makes more tokens of it — the request is fitted again
by the server's own count and sent once more, and — when the server stated its count — that
model's later calls are fitted by the learned ratio from the start. The ratio is also learned
from what each reply says its request was, so a request that fits by the model's own count is
sent as it is; and the tools a run's narrowing kept are kept first on its later calls (the
same set while it fits: a prompt cache keeps its prefix). Only when the instructions and the user's own message alone do not fit does the call
fail: a `ContextTooLarge` (`CONTEXT_TOO_LARGE`) naming the window and the size. The memory
context's own budget is unchanged (5% of the window, 2,000 to 8,000 tokens): it is the last
thing shrunk, and only for a request that would overflow.

Measured on Granite 3.3 2B (CPU, llama.cpp behind Bifrost), the cost benchmark's ten tasks
(simple, several-part, a ~100k-character result) among 50 tools, memory off:

| | Window | Runs that completed | Tool coverage | Input tokens / run | Seconds / run | Steps that fired |
|---|---|---|---|---|---|---|
| without `FitWindow` | 8,192 | 7/10 (3 refused as over the window) | 1.00 | 13,339 | 162 | — |
| `FitWindow` | 8,192 | 8/10 (2 stalls, the model's: `StallGuard` stopped them) | 1.00 | 15,207 | 189 | `tools` 6, `refit` 3 |
| no need to fit | 16,384 | 9/10 (1 stall) | 1.00 | 15,802 | 181 | — |

On the simple tasks the requests were the same as without it — the same tokens (13,518 a
run) and the same time: a request that fits is sent as it is. With 100 tools on 8,192 tokens
every run completed (coverage 0.96). The memory step, on ten requests whose user message
plus memory context did not fit 8,192 tokens (the server refused them; the fact asked for was
in the context's best-ranked items): 9/10 answered right with the memory context trimmed, 10/10
with it whole on 16,384; without the step none of the ten fits the window. LangChain's
`LLMToolSelectorMiddleware` on the same 50 tools completed 6/10 (coverage 0.60): it is not a
default, and the harness has no fallback to it; a team adds it through `middleware=`.

## What else is automatic

| | |
|---|---|
| Memory push | the context is a system message after `system` |
| Memory pull | the memory tools are offered to every model call |
| Tool hints | each model call is offered the tools offered at that moment (the hinted ones, the memory tools, those already used, what a `tool_search` found) |
| Model steps | the graph's checkpoint is the run's: a resume continues from the last step, so the model is not asked again for what it already decided |
| Several calls in one step | the read-only calls run at once; the others one at a time, in the model's order; a call that pauses lets the others finish (journaled) before the run pauses, and the pauses of one step come one by one; a call the step repeats (same tool, same arguments) runs once, each repeat told to read the first one's result ([reliability.md](../reliability.md#calls-made-at-once)) |
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

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | `ReAct(backend=sandbox())`: Deep Agents' file tools and `execute` in the run's own sandbox ([sandbox.md](../sandbox.md)) |
| Skills | `ReAct(skills=src)`: a skill source (`trellis.skills`), pinned per run, read by Deep Agents' own skills middleware through a `/skills/` route of the backend — with `backend=sandbox()` too; or Deep Agents' own source paths ([skills.md](../skills.md)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks ([hooks.md](../hooks.md)); a model call is LangChain middleware: `ReAct(middleware=[...])` |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)); the loop's own turn limit is `max_steps`, and LangGraph's `recursion_limit` goes in `framework_options` |

## Limits

* `middleware=` adds: one named as the harness's is refused (its parameter sets it).
* No `store=` or `checkpointer=`: the memory service and the run keep them; a graph with its
  own is `create_agent(..., middleware=middleware())`.
* `tools=` takes no sandbox (`backend=sandbox()`) and no agent (`subagents=`); a graph is
  a sub-agent with its name and description (`{"name", "description", "runnable"}`).
* Building a `ReAct` costs what `create_deep_agent` costs (it compiles its general-purpose helper
  as well): about 26 ms against 13 ms for `create_agent` — once per agent, not per run.
* Without a known window (`context_window=`, or the model's LangChain profile), clearing old
  tool results and the long-task tools are off, said once.
* Skills and memory files on the default backend must be in the graph's state; give a
  `FilesystemBackend` (or a store) that holds them.
* The harness runs the graph asynchronously (`ainvoke`): the sandbox backend has no synchronous
  operations.
* A model name needs Bifrost or a provider prefix (`openai/...`) whose key the provider's SDK
  reads ([configuration.md](../configuration.md#where-a-model-name-goes)).

## Run it

* [`examples/01_start/first_agent.py`](../../examples/01_start/first_agent.py) — two tools and an approval.
* [`examples/02_way1_react/agent.py`](../../examples/02_way1_react/agent.py) — every piece on: memory, MCP
  tools, an approval resumed in place, a hook, `run_timeout`, `framework_options`, `without=`.
* [`examples/02_way1_react/subagents.py`](../../examples/02_way1_react/subagents.py) — a Deep Agents
  sub-agent, a wrapped agent and an A2A agent, each a child run.
* [`examples/05_features/sandbox.py`](../../examples/05_features/sandbox.py) — `backend=sandbox()`:
  a script written and run in the run's sandbox.
* [`examples/02_way1_deepagents/middleware.py`](../../examples/02_way1_deepagents/middleware.py) —
  `middleware()` on a `create_deep_agent` and a `create_agent` graph of your own.
* [`examples/02_way1_react/structured_output.py`](../../examples/02_way1_react/structured_output.py) — a tool, then a structured answer.
* [`examples/06_scenarios/tool_choice.py`](../../examples/06_scenarios/tool_choice.py) — the four
  ways tool choice goes wrong, `tool_coverage` per item.
* [`examples/05_features/own_scheduler.py`](../../examples/05_features/own_scheduler.py) — your run store, your
  scheduler loop, your governance, memory off.
* Tests: `tests/integration/test_react.py`, `test_react_deep_agents.py`, `test_react_parallel.py`,
  `test_react_context.py`, `test_react_long_tasks.py`, `test_sandbox_backend.py`,
  `test_middleware.py` (each middleware on a plain `create_agent`), and against the real
  services `tests/live/test_live_matrix.py` (`react`) and `tests/live/test_live_subagents.py`.
