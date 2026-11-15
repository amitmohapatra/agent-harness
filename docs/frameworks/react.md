# ReAct: a tool-calling agent with no framework of yours

For a team with no agent framework: `ReAct(system_prompt, model)` returns a Deep Agents
`create_deep_agent` graph with the harness's own middleware —
[`middleware()`](#middleware-on-your-own-graph) — on top. There are two ways to run it, and
`ReAct()` already takes what says which ([two ways](#two-ways-to-run-it)):

* **`ReAct` alone** is lean: an ordinary task is sent the agent's tools and nothing else, a
  large result is kept by the run and the window is fitted by `FitWindow`, as on LangChain's
  `create_agent` with `middleware()`. Deep Agents' machinery — planning, scratch files, its
  general-purpose helper — is held, and offered by itself once the run shows a long task: its
  conversation has outgrown half the model's window.
* **`ReAct` on Deep Agents, on purpose** — ask for a Deep Agents capability: `backend=` (file
  tools), a sub-agent of its own (`subagents=`), `skills=`, `memory_files=`, `permissions=` —
  is Deep Agents from its first step, every feature native (its tools and planning from the
  first call, large results saved as files, summarization near the window's end, sub-agents,
  skills, memory files, permissions, prompt caching).

The lean base was measured as good or better on every class of task ([the base](#the-base)). `h.wrap` runs it as it runs
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
  each a tool of its own; in a chat each also takes tasks in the background, with no server to
  host ([a task in the background](#a-task-in-the-background)). Deep Agents' `AsyncSubAgent`
  (`graph_id`, `url`) runs a task on a server of yours instead.
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
  LangChain profile (`model.profile["max_input_tokens"]`); for a model name LangChain has no
  profile of (one released after the installed package, a local one) the route is asked once,
  when the agent is built: Anthropic's Models API (`max_input_tokens`), an OpenAI-compatible
  endpoint's `/models` (the field its server names the window with: Bifrost, OpenRouter, Groq,
  vLLM, Mistral), within 5 seconds, with the route's auth. When none says, the window-scaled
  parts are off — clearing old tool results and the long-task tools — said once in the log, and
  Deep Agents' summarization uses its own thresholds: give it for a server whose model list
  names no window (Ollama, llama.cpp's `-c`) and for a chat model object without a profile.
  A model whose window no one said is still not sent a request its server refuses twice:
  the refusal names the window (OpenAI-compatible servers', Anthropic's, llama.cpp's
  wording), the request is fitted to it and sent again, and later requests are fitted at
  once. A request is fitted to 95% of the window, and to no more than the model's own
  window leaves beside the reply the call will send (its `max_tokens`).
  The same answer names the model's output limit (Anthropic's `max_tokens`, a server's
  `max_output_tokens` or `max_completion_tokens`): it is used where LangChain knows none of
  the model and filled in a fallback of its own (`ChatAnthropic`: 4,096 tokens), so a model
  newer than the installed package does not stop mid-call or mid-answer. It is used up to
  the share of the window the library gives the models it knows (64,000 of 200,000 today,
  read from its installed profiles), since input and reply must fit the window together. A
  model LangChain knows keeps its profile's limit, a chat model object or a call's own
  `max_tokens` is yours, and a route that names no limit, or no window, leaves the library's
  fallback.
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
sub-agent's own `interrupt_on` is kept as Deep Agents' approval gate, asked once) —
`name` (the wrap's `id`) and `debug`.

## Two ways to run it

**`ReAct` alone: lean, Deep Agents' machinery by itself when the run needs it.**

```python
from trellis import Harness, ReAct

h = Harness()
agent = h.wrap(ReAct("You answer stock questions.", "provider/model", tools=[quote]), id="stock")
```

An ordinary task costs what `create_agent` with `middleware()` costs: the same tools sent, the
same instructions. The switch is the run's own behaviour, never the request's wording and never
a model call made to classify: once the conversation is past half the model's window — where
old tool results start being cleared, so the request is rewritten anyway and a provider's prompt
cache is not broken a second time — the model is offered `write_todos`, the scratch file tools
(`ls`, `write_file`, `edit_file`, `delete`, `glob`, `grep`) and Deep Agents' general-purpose
`task` helper, for the rest of the thread. It needs a known window of 32,000 tokens or more.

**`ReAct` on Deep Agents, on purpose: Deep Agents from the first step.**

```python
from trellis import Harness, ReAct
from trellis.harness.automatic import JournalBackend

h = Harness()
target = ReAct(
    "You research and write reports.",
    "provider/model",
    tools=[quote],
    backend=JournalBackend(),  # or a disk, a store, sandbox(); or subagents=, skills=, ...
)
agent = h.wrap(target, id="reports")
```

Its tools, planning and `read_file` are offered from the first call (about 2,600 tokens a
call), a large result is saved as a file by Deep Agents (kept by the run instead when the
backend's files are outside the graph's state — a sandbox, a disk, a store — so `query_result`
reads its rows there as anywhere), and the older turns are summarized near the window's end.
On a window known to be under 32,000 tokens the long-task tools are held: there is no room
for them. The summarization is not placed there either: `FitWindow` fits the request, as on
`ReAct` alone (one owner of the window; measured on windows of 6,000 to 16,000 tokens, where
the summarization saved again the results the run keeps and spent model calls on summaries).

## The base

Three bases were measured on Granite 3.3 2B (CPU, llama.cpp behind Bifrost, an 8,192-token
window), each with `middleware()` and `FitWindow`, ten different tasks per class:
**A** `create_agent` (what `ReAct` alone sends), **B** `create_deep_agent`, **C** `create_agent` with Deep
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
adds no base prompt) and tools. A is what `ReAct` alone sends (the same tools, large results kept by the run, the window
fitted by `FitWindow`; Deep Agents' own tools held): no less accurate on any class, and on a large
result it completed every run where B and C stalled half of theirs, after Deep Agents saved the
result as a file and the model went looking. Deep Agents' capabilities stay a parameter away:

| Capability | When |
|---|---|
| planning (`write_todos`), the scratch file tools, the general-purpose `task` helper | `ReAct` alone: a long task — the conversation past half the window, on a known window of 32,000 tokens or more. Asked for a Deep Agents capability: from the first call |
| `read_file` | once an older result was cleared past half the window (by its call's id: `/large_tool_results/<call id>`), a large result was kept, or the graph's state holds a file |
| `query_result` | once a result is held anywhere: kept by the run (a large one, a table read whole, one cut to fit the window), saved in the graph's state, or cleared from the window with rows in it; the result's message names its path and table |
| file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`), a large result saved as a file, summarization | `backend=` — `JournalBackend()` for the graph's state kept by the run's journal, a disk, a store, or `sandbox()` (with `execute`) |
| `task` with your sub-agents | `subagents=` with a `SubAgent` or `CompiledSubAgent` dict |
| skills, memory files, permissions | `skills=`, `memory_files=`, `permissions=` |

Each capability's guidance travels in its tool's description, offered that turn — never in the
instructions — so
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
| summarization (`create_summarization_middleware`) | past 85% of the window (its own thresholds when the window is not known), the older turns summarized by one model call, the turns it replaced saved where `read_file` reads them (`/conversation_history/...`), oversized tool arguments in older turns truncated first, a context overflow summarized and retried. `ReAct` places it after the harness's middleware that narrow the tools, so it counts what the model is sent; with a capability and a window known to be under 32,000 tokens it is left out and `FitWindow` leaves the older turns out instead |
| `PatchToolCallsMiddleware` | a tool call left with no answer (a crash, a cancel) gets one, so the conversation stays valid |
| `SkillsMiddleware`, `MemoryMiddleware` | `skills=`, `memory_files=` |
| prompt caching | Anthropic's (and Bedrock's, Fireworks') cache breakpoints, when the model is theirs; `ReAct` puts `PromptCache` in the Anthropic one's place ([below](#the-middleware)) |
| `HumanInTheLoopMiddleware` | only for a `permissions` rule in `interrupt` mode (the harness asks it once, as an approval) |
| `response_format`, parallel tool calls | LangChain's and LangGraph's |

Deep Agents keeps `messages` and `files` as deltas (LangGraph's `DeltaChannel`); the run's
checkpointer keeps each delta channel's last snapshot and the writes since beside the latest
checkpoint, so a resume on any worker rebuilds them whole — and at any earlier checkpoint of
the run's history ([below](#checkpoint-history-langgraphs-time-travel)).

**The harness's** — [`middleware()`](#middleware-on-your-own-graph), in this order:

| Middleware | What it does |
|---|---|
| `TodoListMiddleware` (LangChain), `LongTaskTools` | `ReAct` alone: a long task — a conversation past half the window (counted as LangChain's context editing counts it; where old results start being cleared), on a known window of at least 32,000 tokens — is offered planning (`write_todos`), Deep Agents' scratch file tools and its general-purpose helper, for the rest of the thread; a shorter task is sent none of them (together about 2,600 tokens a call), whatever its wording or the calls it made. `ReAct` asked for a Deep Agents capability offers them from the first call. On your own graph, `middleware()` offers planning the same way |
| `HarnessTools` | the run's tools — your own, memory, skills, the MCP tools, what a `tool_search` found — offered per model call (narrowed once per run by the tool hints, then only growing: a `tool_search` find, a long task's tools; in the order first offered — the first call's by name, the later ones after them — so a provider's prompt cache keeps its prefix) and run through the harness: governed, journaled (a resume does not run a finished call again), traced. The calls whose tool only reads run at once; the others in the model's order. With no `read_file` of the graph's own (no backend), a result over 80,000 characters is kept in the run and previewed, and `read_file` and `query_result` (read-only SQL over its rows) are offered once one is ([tools.md](../tools.md#large-results)); with a backend, `query_result` runs over the result Deep Agents saved in the state |
| `GovernTools` | the graph's own tools — your LangChain tools, Deep Agents' `task`, its file tools on a backend outside the state, a sub-agent's tools — governed as the harness's are (catalog, approval, journal, records) |
| `StallGuard` | the same call in `max_repeats` consecutive steps, or 3 consecutive steps in which every call failed and a failure was met again (the same tool, the same error): the model is told so and called once more with no tool (`tool_choice="none"`), so the user is told what went wrong; no answer then stops the run with a `ModelError`. Failures that each differ are a model correcting itself, bounded by `max_steps` |
| `ModelCallLimitMiddleware` (LangChain, as `AnswerAtLimit`) | `max_steps` model calls; then one more, offered no tool, whose answer ends the run (one extra call at most; LangChain's own limit one call later is the backstop) |
| `Budgeted` | with a budget (`h.wrap(budget=)`): once it — the run's or its parent's — is spent, one more call without tools; the run ends `PARTIAL` |
| `ContextEditingMiddleware` (LangChain) | a known window only: past half of it, the turn's older tool results cleared from what the model is sent, oldest first, a placeholder naming `read_file` (the call's id as its path) in each one's place — never the last 3 nor a `read_file` result — stopping once a tenth of the window is freed (LangChain's `clear_at_least`), so more than the last 3 may stay and a prompt cache breaks rarely |
| `ReadTools` | `read_file` offered once a result was cleared, the graph's state holds a file or the run keeps a result (always, on a backend outside it): a small model offered it from the start reads nothing, again and again; a cleared result is paged from the graph's state as a kept one is (100 lines, 20,000 characters a page) |
| `FitWindow` | a request over the window (95% of it, less the reply's tokens; `context_window=`, else the model's profile) made to fit before the call, never a failure for its size ([below](#when-a-request-does-not-fit)) |
| `ModelHooks` | the run's `on_error` hooks for a model call that failed, its `chat` span (redacted, usage, finish reasons), `call_timeout` and what is left of the run's time, the stored `system_prompt` pinned for the run, the gateway's headers on a routed model, the tools of a part the run is `without=` left out, the run's question kept ahead of a summary (not an earlier turn's replayed before it), and the day it is after the instructions ([tools.md](../tools.md#what-day-it-is)); a call whose arguments are not JSON answered with what was wrong. A model error is a `ModelError`, retryable for a timeout, a lost connection, or a status in bifrost-sdk's `RETRYABLE` (408, 409, 425, 429, 500, 502, 503, 504) |

The graph's checkpointer is `RunCheckpointer()`: every checkpoint, in the run's journal — a
resume after a pause, a crash or a requeue continues the graph where it stopped.

Last of what changes a request, where Deep Agents puts its own, both bases (and `middleware()`
on your own graph) have `PromptCache`:
LangChain's `AnthropicPromptCachingMiddleware` (the one you give `ReAct(middleware=)`, its
settings kept) run on the chat model the call is made with — a `ChatAnthropic`, or a model name
routed to Anthropic's own SDK or to Anthropic's API as an OpenAI-compatible endpoint — so the tools, the instructions, the request's end and the
answer that ends a history the harness replays (memory's window) are marked (4 breakpoints). An
answer the model itself gave in a graph's own thread is sent exactly as it gave it: marking one
changed the answers after it. Any other model's request is sent as
it was. Through Bifrost the gateway caches (`prompt_cache.auto_inject`); any other
OpenAI-compatible endpoint's chat completions carry no breakpoints.

A declarative sub-agent is given `StallGuard`, `ModelCallLimitMiddleware`, `ModelHooks` and
`PromptCache` too (its own of those names kept), and a model name of its own is routed as
`ReAct`'s. One that names no `tools` has the run's tools, as Deep Agents gives such a sub-agent
the main agent's (`HarnessTools` and `GovernTools` are given to it: `ReAct`'s tools are offered
per model call, not built into the graph); one that names its own `tools` has exactly those.
Deep Agents' general-purpose helper is given the same way — with the agent's `skills`, as
Deep Agents gives the one it builds itself — unless you name a sub-agent `general-purpose`
yourself.

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
| `tools` | the offered tools narrowed to the most relevant; one not kept is still offered by its signature (its name and arguments, the description left out) while that fits, and always when the tool hints chose it | those the run used, the graph's own state tools and `tool_search`; then the hinted ones; then by the words they share with the request; 8 of them at least besides |
| `descriptions` | the kept tools' descriptions cut to their first sentence, no parameter descriptions | the names and parameters |
| `results` | the longest tool results cut to head and tail (clearing is `ContextEditingMiddleware`'s alone; on Deep Agents, none: its summarization offloads them whole to its backend), each kept whole in the run and its cut naming the `read_file` call that reads on from where it starts | the latest |
| `turns` | the older turns, in layers: their tool calls and results, the oldest turn's first, then the turns whole (`ReAct`: Deep Agents' summarization summarizes them, the turns saved for `read_file`); then the tools past the first step's 8, only as far as the latest turn leaves no room | the older turns' questions and answers while they fit; the latest turn whole, from the user's question (not a note the step limit or the budget appends); the tools it must keep, and the first 8 besides by their signature whatever the room (only a request the model's server itself refused gives those up) |
| `memory` | last, only when the latest turn alone still does not fit: the memory context's lowest-ranked items, those its `Most relevant` does not name first | the rules, the standing instructions (profile), the summary, the recent conversation, the learned plan (its skills and tools) |

No tool is held back in silence: a tool the request no longer offers at all is named after the
instructions ("Tools held back to fit the window (...)"), with what finds one (`tool_search`,
where the run has it, which is never held itself). So at any window size a tool the model has
not called yet is there when it reaches for it, or it knows the tool exists.

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
  model name routed as `ReAct`'s; its `interrupt_on` is Deep Agents' own approval gate (LangChain's
  `HumanInTheLoopMiddleware` in the sub-agent: `when`, `allowed_decisions` and `description`
  as you give them), asked once through the run — governance takes the gate's decision for
  its own. With no Deep Agents capability asked for, the general-purpose helper is a long task's.
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

### A task in the background

`ReAct`'s own sub-agents — its wrapped agents and Deep Agents' `SubAgent`/`CompiledSubAgent`
dicts — also take tasks in the background, with no server to host: a turn starts a task and
replies at once, the chat takes new messages while it works, and its result comes back to the
chat. The tools are Deep Agents' async sub-agent tools, by name, arguments and record —
`start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`,
`list_async_tasks` — so the model sees one interface whether a task runs here or on a server
(below). Nothing to configure:

```python
research = h.wrap(ReAct("You research.", model, tools=[search]), id="research")
chat = h.wrap(ReAct("You help.", model, subagents=[research]), id="chat")
chat.serve_chat(app)  # a chat a person reads: the five tools are offered
```

* **Where it is offered** — where `ask_user` is: a chat a person reads (`serve_chat`), where an
  outcome can be delivered and a question answered. A run started from code, queued, fired or
  called by another run hands its sub-agents work in the foreground (`task`, a wrapped agent's
  own tool), as before. `start_async_task`'s description says when a task in the background is
  the right call, and that the result comes back by itself.
* **The task** — a child run of the turn that started it (its tenant, user, documents and
  what it is without), a run of the sub-agent on a conversation of its own (the task's id is
  that thread's; a follow-up is a turn of it and sees what it did), its metadata
  `trellis_background` naming the sub-agent and the chat. A Deep Agents dict runs as the graph
  its `task` tool runs, as the agent `<id>.<name>` (the parent's tools, budget, time limit).
  It works in the process that started it, apart from the turn, through the pipeline as any
  run — its tool calls governed and recorded under its run, its own budget and `max_steps` —
  and is held in the run store meanwhile: a server killed with it leaves a run the store ends
  (`holder_lost`), the harness closing cuts it off (`ERROR`, `BackgroundUnfinished`).
* **Its result** — Deep Agents' notification: a user-role message the user did not type, marked
  with langchain-core's notification metadata (source `async_subagent`, its data the task's
  `TaskEvent`: id, name, status, run). The conversation's model is given it before its next
  model call, at its next turn, or after its final answer (another model turn, within
  `max_steps`); when no turn works in the chat, a turn of the chat's own — a run of the agent
  in the chat's thread whose input is the notification — answers it, and the chat's thread
  stream (`GET /agui/threads/{thread}/events`) delivers that turn as it delivers a scheduled
  run's. That turn is one run per outcome (its id is the outcome's), run by the server whose
  start made it: two servers that find the chat idle at once start one turn. An outcome the
  conversation was told already is not told again: such a turn answers `NO_REPORT` without a
  model call, and the stream leaves it out.
* **Its questions** — an approval or a question inside the task pauses its run (`waiting`):
  the chat's next turn, or a turn of its own when it is idle, asks it as a sub-agent's question
  is asked through its parent (`payload["subagent"]` names the task's run, `background: true`),
  and the answer resumes the task where the answer reached — on another server sharing the run
  store too.
* **Check, list, update, cancel** — read the run store from any server (the list, only the
  unfinished tasks). The turn's own reads of its tasks cost at most one request of the run
  store per model call: a task at work, or ended, in the turn's process is known there, and
  the others — at work on another server, waiting for a person — are read one per model
  call, in turn, so with several of them an outcome may be told a model call or two later. `update_async_task`
  follows Deep Agents' `update_strategy` (a `SubAgent` dict may carry it): `interrupt`, the
  default, cancels the task's run at work and continues its conversation
  (`conversation.Mode.INTERRUPT`); `enqueue` waits for it (`Mode.QUEUE`).
* **Approvals** — as for a task on a server (below): a check and the list
  are reads; a start, an update and a cancel are writes, announced on the stream and not
  asked about (each call the sub-agent makes is governed in its own run). To have a person
  approve one, give it its risk in the tool catalog ([governance.md](../governance.md)).
* **Bounds** — a chat has at most as many unfinished tasks as a run may have children working
  at once (the admission limit of [subagents.md](../subagents.md): 4 at first, less under
  pressure): a start past it is refused with the reason, and the model is told to wait or
  cancel one. A task's outcome that reaches an idle chat may start or update a task three
  times in a row with no message from the user between; the fourth is refused, so a result
  cannot keep work going by itself. A process keeps at most 1,024 attempts at work.
* **The list** — the conversation carries its unfinished tasks and the 50 that ended last
  (a task's record says it ended once the conversation was told); an older one is no longer
  tracked.
* **Whose task** — a task belongs to the conversation that started it: its tenant, its user,
  the agent and the chat thread. The tools, the questions asked through the chat and each
  read act on a task's run only then; a task id from another conversation is answered as one
  this conversation does not track. A run's input names no task list (`async_tasks` in a
  mapping input is left out), and a message a chat client gives is never a notification.
* **With the turn** — a turn that is cancelled cancels the tasks it started (their attempts
  stop); one out of time leaves them working: the turn's clock is not theirs.
* **Notifications in the conversation** — the turn's question stays the user's message; a
  notification is recorded in memory as one (its role `USER`, its notification metadata in its
  run record, so a replay marks it again), the earlier-results and turn logic pass over it,
  and a turn of the chat's own keeps its notification as its question in the conversation's
  earlier turns.

Deep Agents' async sub-agent runs a task on a server of yours instead. Declare it as Deep Agents
does, an `AsyncSubAgent` dict with a `graph_id` (with one, the five tools are Deep Agents' own,
for its server's tasks: the two kinds are not mixed in one `ReAct`):

```python
optimizer = {
    "name": "optimizer",
    "description": "Runs the slow optimization, in the background.",
    "graph_id": "optimizer",
    "url": "https://agents.example",  # and "headers": {...} for its authentication
}
agent = h.wrap(ReAct("You help.", "provider/model", subagents=[optimizer]), id="chat")
```

* **What you host** — an Agent Protocol server (a LangGraph server) at `url` that serves the
  graph `graph_id`: Deep Agents' tools create a thread and a run there, read them and cancel
  the run. Give the `url`: without one Deep Agents looks for a LangGraph server in its own
  process, which a harness process is not, and the task cannot start. The spec reaches Deep
  Agents as you wrote it; the server is sent the task's text, your `headers` and the id of the
  run that started it (Deep Agents' `parent_thread_id`: a `ReAct`'s LangGraph thread is its
  run), nothing of the run's user or chat thread — and, with `mode="fork"`, the conversation at
  launch. A `webhook` is checked as every URL the harness hands on (https, a public host; one
  refused is a `ConfigurationError` when `ReAct` is built) and again each time a task is
  started, by where its host resolves: one that leads to a private address fails the start,
  and the server is sent nothing. A start of a sub-agent that forks
  is a write as any start (Deep Agents asks nobody); to have a person approve it, give
  `start_async_task` that risk in the tool catalog.
* **In the chat** — the model starts a task with `start_async_task` and the turn ends at once
  with the task's id; the next messages are answered while it runs. Deep Agents reads the
  task's status itself (0.7.27 on) when a turn starts, before each model call and after a
  final answer (then giving the model another turn, a model call as any: against `max_steps`
  and the budget; past `max_steps` the run ends with the answer it had, and the outcome read
  for that turn is not marked read — the next turn is told it, memory on or off): each new outcome is a
  notification after the conversation's messages — a user-role message the user did not type,
  marked with langchain-core's notification metadata, which the harness treats so (below). When
  no turn works, the server holding the task's run reads it — after 10 s at most at first, then
  less and less often, up to every two minutes, never two servers in step —: ended, its run ends
  and the chat is told by a turn of its own, as a harness task's outcome is
  ([above](#a-task-in-the-background)). `update_async_task` and `cancel_async_task` reach it by
  its id.
* **Approvals** — a check or the list is a read; a start, an update and a cancel are writes of
  Deep Agents' own tools, announced on the stream and not asked about. To have a person
  approve one, give it its risk in the tool catalog ([governance.md](../governance.md)).
* **Its own run** — each task is a child run of the turn that started it, as a `task`
  sub-agent is ([above](#sub-agents-and-child-runs)), and outlives that turn as the task does: its input
  the task's text, its metadata (`native_helper`) the task's id, the server's `url` and the
  `graph_id` (never your `headers`), `helper_started` on the turn's stream. While the task
  runs, the run is `RUNNING` and held in the run store from the server that started it: a
  server that dies leaves a run the store ends (cut off). It ends when the task's end is
  learned — a later `check_async_task` or `list_async_tasks` that reads its final status, the
  model's `cancel_async_task`, or the holding server's read of it: `SUCCESS` with the task's
  result, `ERROR` saying why it failed,
  `CANCELLED` — with `helper_finished` on the stream of the turn that learned it, on any
  server that shares the run store.
* **Cancelled with its turn** — a turn that is cancelled (while it works, or where it waits
  for an approval, or taken over by a newer message: `mode="interrupt"`) cancels the tasks it
  started, as a cancelled run's other children are: on their server, through Deep Agents'
  own `cancel_async_task` (the Agent Protocol's cancel of the task's run of now), and their
  runs end `CANCELLED`. A server that refuses the cancel is said in the log. The task stays in
  the list, cancelled. A task a turn did not start (one of an earlier turn) goes on. A turn
  cancelled from another server (agent-runs' cancel of a worker's turn) ends the task's run
  there, and the server holding that run cancels the task on its server at its next read.
  New instructions to a task that ended (`update_async_task` after its result) run it again
  on its server: a run of its own, the child of the turn that sent them.
* **Across turns** — Deep Agents keeps its list of tasks in the graph's state, and a `ReAct`
  graph's state is its run's, so the harness carries that list from a conversation's turn to
  its next one: per user, thread and agent, a cancelled turn's too, and that of a turn that
  waits for an approval (the task it started is listed meanwhile, and still is, cancelled,
  when the turn is cancelled there). Two turns that overlap (one waiting for an approval, a worker's still
  running while another ends, two ending at once) each keep the tasks they started. A graph with a checkpointer
  of its own (`create_deep_agent(..., middleware=middleware(), checkpointer=...)`) keeps the
  list in its thread instead, and it lasts as that checkpointer does.
* **Across servers and restarts** — the list is kept in the process that ran the turn and in
  the file store (`TRELLIS_FILES_URL`, [configuration.md](../configuration.md)), where the
  next turn reads it wherever it runs: after a restart, and on another server when the
  servers share that store (a bucket, or a folder they all mount). Unset, the store is a
  folder on the server's own disk: it survives a restart there, another server does not see
  it, and a container loses it with its filesystem. What is kept is Deep Agents' record of
  each task (its ids, the sub-agent's name, a status, three times), at most 256 KiB for a
  conversation, in a folder that is none of the user's files. A store that cannot be read or
  written is said in the log, and the list stays in the process as before. The store has no
  conditional write, so no turn writes over another's: each keeps a file of its own in the
  conversation's folder (what it read and its own state, merged), a turn reads every file
  there, and the files a turn read are removed once its own file holds them — two servers
  ending turns of one conversation at the same moment each keep their task, in any order.
* **Earlier results** — with memory on, a later turn's history holds the earlier calls and
  their results. A start's, an update's and a cancel's are left as they were said (the record
  of what was done: nothing advises starting again); a status and the list are values that
  change, and an earlier one is marked out of date.
* **Your own graph** — the same holds for a graph you build with `middleware()` and no
  checkpointer: `create_deep_agent(..., subagents=[optimizer], middleware=middleware())`, or
  LangChain's `create_agent` with `middleware=[*middleware(), AsyncSubAgentMiddleware(...)]`
  (Deep Agents' `deepagents.middleware.async_subagents`). A graph with neither `middleware()`
  nor a checkpointer keeps no state between turns; its list is carried too, read from the
  state each turn ends with — so a turn cancelled or failed after a start leaves that task
  out of the list (with `middleware()` each step is kept, and it is in).
* **Overlapping turns** — two turns that overlap and both change one task lose neither's
  update: the conversation's list is merged task by task, the record updated last standing
  (at the same second, the further status), whichever turn ends last.

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | `ReAct(backend=sandbox())`: Deep Agents' file tools and `execute` in the run's own sandbox ([sandbox.md](../sandbox.md)) |
| Skills | `ReAct(skills=src)`: a skill source (`trellis.skills`), pinned per run, read by Deep Agents' own skills middleware through a `/skills/` route of the backend — with `backend=sandbox()` too; or Deep Agents' own source paths ([skills.md](../skills.md)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks ([hooks.md](../hooks.md)); a model call is LangChain middleware: `ReAct(middleware=[...])` |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)); the loop's own turn limit is `max_steps`, and LangGraph's `recursion_limit` goes in `framework_options` |

## What the other libraries have, in ReAct

Each is the library's own piece, switched on the library's own way (its tool, its middleware,
`framework_options`): the harness governs it, records it and counts it. None changes what a
`ReAct` built without it sends its model (`tests/integration/test_react_golden.py`).

### A provider's own tools: web search, web fetch, code execution

A tool the provider runs itself, inside the model's step, is given as LangChain gives it — a
`dict` in `tools=`, bound to the model as it is:

```python
ReAct(
    "You answer with sources.",
    "anthropic/claude-sonnet-4-5",
    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
)
```

`create_agent` binds it and runs none of it: the provider does, and its reply says what each
call asked and found (LangChain's `server_tool_call` and `server_tool_result` content blocks).
The harness adds (`trellis.harness.provider_tools`):

* **the catalog's risk.** Each is a tool of the run's set and of the catalog. Before each model
  call governance decides it: the catalog's risk, else what it is known to do (a search, a
  fetch and a file search read; code runs in the provider's own sandbox, a write announced as
  `sandbox_exec` is). No approval can stand before a call the provider makes inside its step,
  so a tool whose decision asks a person (`irreversible`, an approval rule, a tool nobody
  declared) is not offered to that call, said once on the run (`PROVIDER_TOOL_HELD`): the
  catalog offers it by declaring a risk that needs none.
* **records.** Each server call is the run's tool call: on its stream, traced, counted,
  recorded in the memory service, with what it was asked and what it found (never the
  provider's encrypted state). A sub-agent's are recorded the same way.

A `dict` a client would have to run (a function's schema, Anthropic's `bash`, `text_editor`,
`computer`, `memory`) is refused: the graph has no code for it — give a function. Whether a
route passes a provider's tool on is the route's: Anthropic's tools need an Anthropic chat
model (a name routed to its API, or `ChatAnthropic`), OpenAI's the Responses API.

### Plan mode

Read-only exploration, then a plan a person approves, then the work — Claude Code's and Kimi's
plan mode, from LangChain's pieces (`trellis.harness.plan_mode`):

```python
from trellis.harness.plan_mode import PlanMode

agent = h.wrap(
    ReAct("You handle refunds.", "gpt-4o", tools=[order, refund], middleware=[PlanMode()]),
    id="refunds",
)
await agent.run("refund o1", user="u", framework_options={"configurable": {"plan_mode": True}})
```

It is **entered on request, never by default**, as theirs is: the Claude Agent SDK's
`permission_mode` is unset unless a caller sets `"plan"` (`ClaudeAgentOptions.permission_mode:
PermissionMode | None = None`), and Kimi's plan mode is the user's to enter (from 2.4.0 the
agent can no longer enter it itself; `default_plan_mode` is off). A run enters it through its
`framework_options` — LangGraph's `configurable`, per run or for every run at `h.wrap` — where
Claude's `permission_mode` and Kimi's `mode` go. A run that did not enter it is sent what it
always was.

* The plan is LangChain's `write_todos`, offered from the run's first call.
* Its approval is LangChain's `HumanInTheLoopMiddleware` on `write_todos`, with its own `when`
  predicate: one question (approve, edit the plan, or reject saying what to change — the model
  then revises it and asks again), the harness's approval like any other.
* Until it is approved, a call runs only if governance rates it a read (the catalog's risk,
  else the tool's own). Any other — a write, a tool that says nothing, a hand-over to another
  agent (`task`, a wrapped agent: its own calls are not this run's to bind) — is answered with
  what to do instead, never run, and recorded as the run's rejected call.
* Approved, the plan is in the graph's state (kept in the run: a resume in any worker keeps
  it) and each call is governed as ever: a write still asks.

### Checkpoint history: LangGraph's time travel

`RunCheckpointer` keeps every checkpoint, as LangGraph's own savers do (`InMemorySaver` stores
each under its id and prunes none; the Postgres saver likewise) — the Claude Agent SDK's file
checkpoints are the opt-in one (`enable_file_checkpointing: bool = False`, which
`rewind_files` needs). So LangGraph's own API works on a `ReAct` graph, and on a
`middleware()` graph, as on any saver:

```python
graph = ReAct("...", model, tools=[...])
config = {"configurable": {"thread_id": "t"}}
states = [s async for s in graph.aget_state_history(config)]  # newest first
await graph.ainvoke(None, states[3].config)  # continue from an earlier one
await graph.aupdate_state(states[3].config, {"messages": [...]})  # or fork it
```

In a harness run the thread is the run, and its history is the run's: kept in its journal,
saved with its pause and with a worker's progress, until the run ends — the run store clears a
run's checkpoint with its ending, so there is no setting of its own, and nothing kept past the
run. It is read with the same `aget_state_history` from code in the run (a tool, a hook, a
node), on the run's id, and from outside it with `agent.checkpoints` (below).

What it costs, measured on a scripted run (`ReAct`, no sub-agent): about 7 checkpoints a tool
step and 10 a turn, each about 2 KB in the journal (its checkpoint and metadata, base64 in
JSON), and what each step wrote, once. One tool call and an answer: 17 checkpoints, 36 KB
against 4.5 KB for the latest alone; eleven tool calls: 87 checkpoints, 217 KB against 17 KB.
A save still sends what the step added; the journal is compared in memory at each save, and
written whole every 16th.

#### Rewinding a paused run

```python
# newest first; only the run's own user reads them
states = [s async for s in agent.checkpoints(run_id, reader="ada")]
result = await agent.rewind(
    run_id, states[5].config["configurable"]["checkpoint_id"], reviewer="ada"
)
```

`agent.checkpoints(run_id, reader=...)` reads, for the run's own user only, the history the run store keeps of a paused run (or of one a
worker is running: what it last saved; or, for an agent wrapped `h.wrap(..., rewindable=True)`,
of one that has ended — the run store keeps its history past the end, its last state included,
for its artifacts' retention), read-only, a page of 20 at a time. `agent.rewind`
continues the run's conversation from one of those checkpoints, as LangGraph's time travel
does — `values=` applied to that state first, as LangGraph's `update_state` applies them — in
a **new run** of the same user and thread, to its end or its first pause:

* **on record.** The new run's `metadata["rewound"]` names the run and the checkpoint it
  continues, who rewound it (`reviewer`) and when. The run rewound is left as it was — paused
  on its question, its history readable: answer it, or cancel it.
* **nothing done twice unasked.** What the run did in the world is not undone. A call that did
  more than read past the checkpoint asks a person before the new run makes it again —
  whatever "allow for this conversation" or "always allow" said before, since it is a second
  effect — and runs once when approved; rejected, it does not run. A read is made again
  freely. Asked about once, the call is decided as any call is from then on.
* **the budget.** What the run spent before the rewind stays counted in the new run.
* **refused**, changing nothing: a run at work (queued or running, on any server); a run
  that has ended; a `reviewer` who is not the run's user; a checkpoint that is none of the
  run's (another run's, another thread's); a run that is not this agent's in the caller's
  tenant, as for every read of a run; and an agent whose runs keep no checkpoints (Claude,
  Kimi, OpenAI Agents, a function, a graph with a checkpointer of its own), which answers
  that checkpoints and rewind are not supported for it.

**An ended run cannot be rewound.** The run store clears a run's checkpoint when the run
ends (agent-runs: "the checkpoint ... is cleared when the run finishes"), so only a paused
run has a history to go back in. A graph with a checkpointer of its own keeps LangGraph's
time travel across runs ([langgraph.md](langgraph.md)). From code inside a run, a fork of the
run's own thread is still refused (`RewindRefused`): a rewind starts from outside the run.

Claude's own file checkpoints (`enable_file_checkpointing`, `rewind_files`) are a different
feature — files on disk restored to a user message, opt-in in the Claude Agent SDK — and this
API does not drive them.

### Handoffs: `langgraph-swarm`

Control passes to another agent, which answers the user — an OpenAI agent's handoff — with
LangChain's own library for it, `langgraph-swarm` (in the `react` extra), used as it is: its
`create_handoff_tool` among an agent's tools, its `SwarmState` and its router:

```python
from langgraph.graph import StateGraph
from langgraph_swarm import SwarmState, add_active_agent_router, create_handoff_tool
from trellis.harness.middleware import RunCheckpointer

desk = ReAct("You are the front desk.", model, tools=[create_handoff_tool(agent_name="billing")])
billing = ReAct("You do refunds.", model, tools=[refund, create_handoff_tool(agent_name="desk")])
builder = (
    StateGraph(SwarmState)
    .add_node("desk", desk, destinations=("billing",))
    .add_node("billing", billing, destinations=("desk",))
)
team = add_active_agent_router(builder, route_to=["desk", "billing"], default_active_agent="desk")
agent = h.wrap(team.compile(checkpointer=RunCheckpointer()), id="support")
```

(`create_swarm([...])` builds the same graph from agents that have names of their own:
`create_agent(..., middleware=middleware(), name="desk")`.) What the harness adds:

* **governed and recorded.** A handoff tool is a hand-over, as Deep Agents' `task` is: the
  call is the run's tool call (`transfer_to_billing`: its stream, its trajectory, the memory
  record, a span), it only reads, so it asks nobody unless the catalog says otherwise — and
  each call the agent handed to then makes is governed on its own (its refund still asks).
* **whose turn it is.** The hand-over call says who takes over within a turn; the
  conversation's next turn starts with the agent it was last handed to (langgraph-swarm's
  `active_agent`, carried from turn to turn for the tenant's, the user's conversation as Deep
  Agents' task list is — kept in the file store, so on any server) and says so on its stream
  (the `CUSTOM` event `active_agent`).
* **memory.** The conversation, its memory and its records are the wrapped agent's
  (`id="support"`): the agents of the graph share them, as they share langgraph-swarm's
  `messages`.
* **a pause resumes in place.** With `RunCheckpointer()` the graph's checkpoints are the
  run's: after an approval neither the hand-over nor the steps before it are asked of a model
  again, on any worker.

The agents' own tools are LangChain tools (their tool nodes run them, governed by what each
declares in its `metadata`, else the catalog); the functions given to an inner `ReAct`'s
`tools=` are the run's only when the graph wrapped is that `ReAct`.

### Fan-out: many sub-agents at once (a recipe)

Kimi's `AgentSwarm` — one sub-agent an item, all at once — is LangGraph's `Send` over the
items, given to `ReAct` as a sub-agent of its own (Deep Agents' `CompiledSubAgent`). It needs
no harness code: a wrapped agent run from a node is the run's child
([Sub-agents and child runs](#sub-agents-and-child-runs)), so each item has a run record of
its own (its parent, its thread, its spend counted to the parent's budget), governed as any
run, and they work at once only as far as the harness's admission limit lets them (4 at
first, up to 16, fewer under load: `trellis.harness.admission`).

```python
import operator
from typing import Annotated, TypedDict

import trellis
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

researcher = h.wrap(ReAct("You research one item.", model, tools=[search]), id="researcher")


class Fan(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    item: str
    found: Annotated[list[str], operator.add]


def items(state: Fan) -> list[Send]:  # one Send a line of the task
    return [
        Send("work", {"item": line}) for line in str(state["messages"][-1].content).splitlines()
    ]


async def work(state: Fan) -> Fan:
    result = await researcher.run(state["item"], user=trellis.current().user)  # a child run
    return {"found": [str(result.answer)]}


def gather(state: Fan) -> Fan:
    return {"messages": [AIMessage("\n".join(state["found"]))]}


builder = StateGraph(Fan)
builder.add_node("work", work)
builder.add_node("gather", gather)
builder.add_conditional_edges(START, items, ["work"])
builder.add_edge("work", "gather")
builder.add_edge("gather", END)

lead = ReAct(
    "You lead the research.",
    model,
    subagents=[
        {
            "name": "fan_out",
            "description": "Researches many items at once: give one item a line.",
            "runnable": builder.compile(),
        }
    ],
)
```

Tested in `tests/integration/test_react_fan_out.py`.

### Compaction on request

Claude's and Kimi's `/compact`, the model's own way: Deep Agents' `SummarizationToolMiddleware`
(its `compact_conversation` tool), opt-in through `middleware=`:

```python
from deepagents.middleware.summarization import create_summarization_tool_middleware
from trellis.harness.automatic import JournalBackend

backend = JournalBackend()
ReAct(
    "...", model, backend=backend, middleware=[create_summarization_tool_middleware(model, backend)]
)
```

The tool is the graph's own machinery — a summary in the history's place, the history kept in
the backend as Deep Agents' automatic summarization keeps it — so it is declared a read (on
any wrapped graph that has it, `middleware()`'s too): the model compacts without a person
approving it. Deep Agents refuses a compaction that is too early (under half its automatic
trigger); the run's question stays ahead of the summary, as with the automatic one.

### A goal loop: the answer graded and revised (a recipe)

Kimi's goal mode and Claude's `/loop`, as Deep Agents has it: `RubricMiddleware`, opt-in
through `middleware=`. A run whose input names a `rubric` has its answer graded by a model of
its own and revised until the grader is satisfied; a run that names none is sent what it
always was.

```python
from deepagents.middleware.rubric import RubricMiddleware

agent = h.wrap(
    ReAct(
        "You answer with units.", model, tools=[mast], middleware=[RubricMiddleware(model=grader)]
    ),
    id="surveyor",
)
await agent.run(
    {"messages": [{"role": "user", "content": "how high is the mast?"}], "rubric": "Says the unit"},
    user="u",
)
```

It needs no harness code: the grader's calls count to the run's budget and spend as every
model call of the graph does, and the revisions are model steps under `max_steps`. It adds
model calls to every graded run, so it is never a default
(`tests/integration/test_react_rubric.py`).

### Guardrails: input and output (a recipe)

An OpenAI agent's input and output guardrails are LangChain's `@before_agent` and
`@after_agent` hooks with `jump_to`, given through `middleware=`:

```python
from langchain.agents.middleware import after_agent, before_agent
from langchain_core.messages import AIMessage


@before_agent(can_jump_to=["end"])
def no_card_numbers(state, runtime):
    if holds_a_card_number(state["messages"][-1].content):
        return {"messages": [AIMessage("I cannot help with card numbers.")], "jump_to": "end"}


@after_agent
def no_cards_out(state, runtime):
    answer = state["messages"][-1]
    if holds_a_card_number(answer.content):
        return {"messages": [AIMessage("The answer was withheld.", id=answer.id)]}


ReAct("You help with accounts.", model, tools=[account], middleware=[no_card_numbers, no_cards_out])
```

A blocked input reaches no model and no tool; the guard's message is the turn's answer. It
needs no harness code, and the blocked turn is the run's as any other: its record (`SUCCESS`,
the guard's answer its output), its stream, the conversation's memory — so the next turn's
model reads that the request was refused — and every tool call made before an output guard
stays recorded. For a policy model as the guard, LangChain's `OpenAIModerationMiddleware` and
`PIIMiddleware` go in `middleware=` the same way (`tests/integration/test_react_guardrails.py`).

### Model fallback and routing, through the gateway

LangChain's `ModelFallbackMiddleware` and `ModelRouterMiddleware` take chat models; a name
given to them as text is LangChain's to make (`init_chat_model`: the provider's own SDK and
key, past the gateway). Give them the harness's routed chat model of the name — what `ReAct`
makes of its own `model` name — and the fallback, or the model the router picks, goes through
the gateway as the main model does: the route's form of the name, its key, the gateway's
headers for the run (its MCP tools, its attribution), the run's budget:

```python
from langchain.agents.middleware import ModelFallbackMiddleware
from trellis.harness.gateways import routed_chat_model

ReAct(
    "...",
    "anthropic/claude-sonnet-4-5",
    middleware=[ModelFallbackMiddleware(routed_chat_model("openai/gpt-4o"))],
)
```

(`tests/integration/test_react_routed_fallback.py`.)

### The capabilities, against the other libraries

What each library has, and where `ReAct` gets it. Columns: C = Claude Agent SDK / Claude Code,
K = Kimi Code, O = OpenAI Agents, L = LangChain / LangGraph / Deep Agents. The source of each
row is the library's own class or tool and, for `ReAct`, the code and its test.

| Capability | C | K | O | L | `ReAct` | Source |
|---|---|---|---|---|---|---|
| Function tools, governed | yes | yes | yes | yes | yes: `tools=` | `react._split`; `test_react.py` |
| MCP tools, resources, elicitation | yes | yes | yes | yes | yes: `mcp(...)`, the gateway's | [tools.md](../tools.md); `test_mcp_resources.py`, `test_mcp_elicitation.py` |
| File tools | `Read`/`Write`/`Edit` | `Read`/`Write` | sandbox | Deep Agents `FilesystemMiddleware` | yes: Deep Agents', held until a long task | `automatic.LongTaskTools`; `test_react_long_tasks.py` |
| Shell / code in a sandbox | `Bash` | `Bash` | `ShellTool`, sandbox | Deep Agents `execute` | yes: `backend=sandbox()` | `sandbox.backend`; `test_sandbox_backend.py` |
| Provider tools: web search, web fetch, code execution | `WebSearch`, `WebFetch` | `WebSearch`, `FetchURL` | `WebSearchTool`, `CodeInterpreterTool` | provider `dict` tools | yes: `tools=[{...}]`, governed and recorded | `provider_tools.py`; `test_react_provider_tools.py` |
| Computer use | Chrome | no | `ComputerTool` | no executor (the tool is only formatted) | no: refused, saying to give a function | `provider_tools.CLIENT_RUN`; `test_react_provider_tools.py` |
| Sub-agents | `Agent` | `Agent` | agent tools | Deep Agents `task` | yes: `subagents=`, each a child run | `react._subagents`; `test_react_helpers.py` |
| Background sub-agents | yes | yes | no | Deep Agents async sub-agents | yes: Deep Agents' five task tools | `bridges.background_tasks`; `test_react_background.py` |
| Many sub-agents at once | `Workflow` | `AgentSwarm` | no | `Send` | yes: the `Send` recipe, each a child run | [above](#fan-out-many-sub-agents-at-once-a-recipe); `test_react_fan_out.py` |
| Handoffs | no | no | `handoffs=` | `langgraph-swarm` | yes: `create_handoff_tool` | `middleware._hands_off`, `adapters.langgraph.ACTIVE_AGENT`; `test_react_handoffs.py` |
| Planning (a todo list) | `TodoWrite` | `TodoList` | no | `TodoListMiddleware` | yes: on a long task | `automatic.Planning`; `test_react_long_tasks.py` |
| Plan mode | `permission_mode="plan"`, on request | `mode=plan`, on request | no | no | yes: `PlanMode()`, on request | `plan_mode.py`; `test_react_plan_mode.py` |
| Skills | yes | yes | sandbox | Deep Agents `SkillsMiddleware` | yes: `skills=`, the helper too | `skills.py`; `test_skills.py` |
| Memory files (`AGENTS.md`) | `CLAUDE.md` | `AGENTS.md` | no | Deep Agents `MemoryMiddleware` | yes: `memory_files=` | `react.ReAct`; `test_react_deep_agents.py` |
| Long-term memory | auto memory | no | sandbox memory | store | yes: the memory service | [memory.md](../memory.md); `test_memory_pipeline.py` |
| Summarization near the window's end | yes | yes | Responses compaction | `SummarizationMiddleware` | yes: Deep Agents', with a capability; `FitWindow` alone | `react._middleware`; `test_react_context.py`, `test_fit_window.py` |
| Compaction on request | `/compact` | `/compact` | no | Deep Agents `SummarizationToolMiddleware` | yes: opt-in, asks nobody | `middleware.COMPACT`; `test_react_compaction_tool.py` |
| Approvals | `can_use_tool` | permission requests | `needs_approval` | `HumanInTheLoopMiddleware` | yes: the catalog, one question | [governance.md](../governance.md); `test_approvals.py`, `test_hitl_middleware.py` |
| Ask the user | `AskUserQuestion` | `AskUserQuestion` | no | `interrupt` | yes: `ask_user` | `tools.ask`; `test_ask_user.py` |
| Guardrails | hooks | hooks | input/output guardrails | `@before_agent`/`@after_agent`, `jump_to` | yes: the recipe, the blocked turn recorded | [above](#guardrails-input-and-output-a-recipe); `test_react_guardrails.py` |
| Goal loop / self-evaluation | `/loop` | goal mode | no | Deep Agents `RubricMiddleware` | yes: opt-in | [above](#a-goal-loop-the-answer-graded-and-revised-a-recipe); `test_react_rubric.py` |
| Model fallback and routing | `fallback_model` | no | no | `ModelFallbackMiddleware`, `ModelRouterMiddleware` | yes: routed through the gateway | `gateways.routed_chat_model`; `test_react_routed_fallback.py` |
| Structured output | yes | no | `output_type` | `response_format` | yes: `response_format=` | `react.ReAct`; `test_react.py` |
| Checkpoint history / time travel | `rewind_files` (opt-in), `fork_session` | `session/fork` | no | `get_state_history`, `update_state` | history kept and read; a run is not rewound yet | `middleware.RunCheckpointer`; `test_react_rewind.py` |
| Steering a running turn | yes | TUI only | no | no | yes: `mode="steer"` | `middleware.HarnessTools`; `test_inbox.py` |
| Step limit, stall guard, budget | `max_turns`, `max_budget_usd` | repeat breaker | `max_turns` | `ModelCallLimitMiddleware` | yes: `max_steps`, `max_repeats`, `budget=` | `middleware.AnswerAtLimit`, `StallGuard`; `test_react.py`, `test_budget.py` |
| Prompt caching | yes | yes | yes | `AnthropicPromptCachingMiddleware` | yes: `PromptCache` | `middleware.PromptCache`; `test_react_spending.py` |
| Scheduling | cron | cron | no | no | yes: the harness's schedule tools | `schedules.py`; `test_schedules.py` |
| Slash commands, agent teams, realtime voice | yes | yes | realtime | no | no: text dispatch is a way past approvals; teams and voice have no LangChain piece to borrow | [claude-agent-sdk.md](claude-agent-sdk.md), [kimi.md](kimi.md) |

## Limits

* The general-purpose `task` helper is built from a spec `ReAct` passes, not by Deep Agents'
  own builder: Deep Agents builds its default helper only when no sub-agent of that name is
  given (`deepagents/graph.py:853-854`, 0.7.38), and that helper takes the model profile's
  settings (`general_purpose_subagent`: enabled, description, prompt) and the middleware of
  Deep Agents' own default slots only — not the harness's guards (`StallGuard`, the step
  limit, `ModelHooks`, the run's tools per call). So the helper `ReAct` gives has the
  loop's guards and the run's tools, and does not read a profile's
  `general_purpose_subagent` settings.
* `middleware=` adds: one named as the harness's is refused (its parameter sets it).
* No `store=` or `checkpointer=`: the memory service and the run keep them; a graph with its
  own is `create_agent(..., middleware=middleware())`.
* `tools=` takes no sandbox (`backend=sandbox()`) and no agent (`subagents=`); a graph is
  a sub-agent with its name and description (`{"name", "description", "runnable"}`).
* Building a `ReAct` costs what `create_deep_agent` costs (it compiles its general-purpose helper
  as well): about 26 ms against 13 ms for `create_agent` — once per agent, not per run.
* Without a known window (`context_window=`, the model's LangChain profile, or the route's
  own record of the model), clearing old tool results and the long-task tools are off, said
  once.
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
