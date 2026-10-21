# Tools

## The toolbox

An agent's tools, per agent and tenant, in `tools/toolbox.py` — their definitions listed again
after `TOOLS_TTL_SECONDS` (300); while a listing fails (the gateway is down), the last good one
stands and is listed again after `TOOLS_RETRY_SECONDS` (30). Whether a call runs, is announced or asks is not the tools':
[governance](governance.md) decides it at each call, from the side effects below and the tool
catalog.

| Where from | Tools | Side effects |
|---|---|---|
| **MCP** — automatic | every tool the agent's Bifrost virtual key allows (the gateway's own MCP listing, asked with the key) — or, with `h.wrap(..., mcp=[slug])`, the tools of those Virtual MCPs — named `<server>-<tool>`, executed through the gateway, each request waiting what is left of the run's time and saying who the run is for; a tool the gateway would run itself (`tools_to_auto_execute`) is left out ([gateway.md](gateway.md)); `mcp=[]` (or `without={"mcp"}`) is no MCP tools | the server's annotations: `readOnlyHint` → read, `destructiveHint` → irreversible, anything else → write; `idempotentHint` makes it idempotent (retried like a read) |
| `tool(fn)`, `@tool(...)`, or a bare function in `tools=[...]` | one; schema from the signature (pydantic validates the model's arguments), each parameter described by the docstring's `Args:` (or `:param x:`) unless `Annotated[..., Field(description=...)]` says; description from the docstring without its sections (`Args:`, `Returns:`, `Raises:`...), every other paragraph kept — a "use it when / not when" paragraph reaches the model; `timeout=` seconds per call (none by default; a sync function runs in a worker thread); `idempotent=True` when a call repeated with its idempotency key has its effect once (retried like a read, run again after a crash); a result that comes from outside the run is an `ask` in the function ([interrupts.md](interrupts.md#results-from-outside-ask-in-the-tool)) | `side_effects=` (`"write"` by default) |
| `a2a(url, *, name=None, timeout=120)` | one: the remote agent, `{"message": string}` in, its answer out; at most `timeout` seconds per exchange | `"write"` |
| `agent.as_tool(*, name=None, description=None, side_effects=None)` | one: another agent wrapped by this harness, `{"message": string}` in, its answer out — each call a child run of it ([subagents.md](subagents.md)) | `"read"` when every tool it declares only reads (and none escapes the harness), else `"write"`; `side_effects=` overrides it |
| `openapi(spec, *, only=None, base_url=None, headers=None, timeout=120)` | one per `operationId`; path and query parameters and a JSON `body` flattened into one argument object; at most `timeout` seconds per operation | by method: GET/HEAD/OPTIONS read, POST/PUT/PATCH write, DELETE irreversible |
| `sandbox(provider=None, spec=None, *, timeout=120)` | `sandbox_exec`, `sandbox_read` and `sandbox_write`, in a sandbox of the run's own (`SANDBOX=docker`, or the provider given), made at its first call and deleted at its end ([sandbox.md](sandbox.md)) | `sandbox_exec` and `sandbox_write` (idempotent) `"write"`, `sandbox_read` `"read"` |
| `h.wrap(..., skills=[...])`, `skills(...)` | `load_skill` and `read_skill_file`: skills from code, `SKILLS_DIR` and the gateway's Skills Repository, each pinned per run ([skills.md](skills.md)) | `"read"` |
| the memory service (memory on) | its agent tools (see [memory.md](memory.md)) | read or write |

`tools=` takes only what runs in this process; which MCP tools an agent has is decided where
its virtual key is configured, in the gateway. Tool names must be unique across the toolbox.

With memory on, every tool is published to the memory service's tool catalog, through
governance, in the background and once per content: MCP tools with their annotations, local,
OpenAPI and A2A tools with their declared side effects. An administrator sets `approve_when` on
them there ([governance.md](governance.md)). Concurrent runs that find the toolbox stale share
one listing.

## Tool choice: what helps the model pick the right tool

What a model chooses from is each tool's name, description and parameter schema — kept as
the source gave them (an MCP server's, through the gateway, as the server lists them; a
function's from its signature and docstring; an OpenAPI operation's), in every framework's own
format. What helps it choose, whatever the source:

| | Where |
|---|---|
| **Descriptions worth choosing by** — what the tool is for, when to use it rather than a broader or narrower one, each parameter described | the tool's author: the docstring (functions), the MCP server, the OpenAPI document. The harness checks, below |
| **A check of the tool set** — `trellis.harness.tools.lint.lint(specs)`: a missing or short description (under 5 words), an undescribed parameter, a name the OpenAI API refuses (letters, digits, `_`, `-`, at most 64), names differing only in case or separators, two tools described the same or nearly (75% of their words shared), 20 tools or more at once (OpenAI's guide suggests fewer), over 128 (the most the OpenAI API takes) | automatic: the toolbox logs each finding as a warning (`trellis.tools`, `tool choice: ...`) when the tools it lists change; in CI, assert `lint([t.spec for t in tools]) == []` |
| **Fewer tools per call** — the memory service's tool hints (from 5 tools, [below](#tool-hints-what-the-model-is-offered)); without them, from 20 tools, a search over the definitions ([tool search](#tool-search)); or LangChain's `LLMToolSelectorMiddleware` in a `ReAct` or `create_agent` graph (its own model call picks the tools for the request) | automatic; `ReAct(middleware=[LLMToolSelectorMiddleware(max_tools=8, on_parsing_failure="all")])` for a model's choice |
| **How to choose** — every part of a request addressed, the most specific tool over a general one, current figures from tools rather than the remembered context, a change made by calling its tool (never claimed without it) | `ReAct(tool_guidance=True)` (`ToolGuidance`) |
| **A check before the answer** — the draft answer checked against the request once; a part left open, or a change only claimed, makes the model call the tools still needed — the write tool's call reaches governance, and its approval; nothing found said plainly as "no record" | `ReAct(answer_check=True)` (`AnswerCheck`) |
| **A check before a change** — the first call of a tool that changes data checked against the rules in the instructions and the memory context, before it runs or anyone is asked to approve it | `ReAct(answer_check=True)` (`AnswerCheck(writes=False)` turns it off) |
| **What it can do** — capabilities beyond the tools, said plainly | `ReAct(capabilities={"accept an image upload": False})` |
| **Measured** — `tool_coverage()` scores each evaluation item by the share of its `needed_tools` the run called | [evaluation.md](evaluation.md#trajectories) |

[frameworks/react.md](frameworks/react.md#tool-choice) has the switches;
[`examples/06_scenarios/tool_choice.py`](../examples/06_scenarios/tool_choice.py) and
`tests/live/test_live_tool_choice.py` measure them. The OpenAI Agents and Claude Agent SDK
targets get the descriptions, the lint and the hints; the guidance and the check are
LangChain middleware (a `create_agent` or Deep Agents graph of your own takes them too).

## Every call

The bridge (`tools/bridge.py`) handles every harness tool call whichever framework makes it:
journal replay, governance (run, announce, or pause for approval:
[governance.md](governance.md#way-1-inside-hwrap)), execution in an `execute_tool` span between
`TOOL_CALL_START/ARGS/END/RESULT` events (arguments and results redacted —
[observability.md](observability.md#redaction) — and results previewed up to 2000 characters), then the
record. A tool that raises becomes an error result the model reads (`"<tool> failed: ..."`); a
pause is never swallowed. A harness tool called outside a run is refused. Calls may come at
once (a step's reads in `ReAct`, a framework running tools concurrently): identical ones take their
turn, each with its own journal entry and idempotency key, and the progress saves go one at a
time ([reliability.md](reliability.md#calls-made-at-once)).

Execution is bounded and counted ([reliability.md](reliability.md)): a call takes at most its
tool's `timeout` and what is left of the run's time; one that only reads (or is idempotent) is
tried again, up to twice, after an error that may pass (an OpenAPI `429` or `503`, a timeout, a
dropped connection) — a write runs once; the tool reads its idempotency key from
`trellis.current().idempotency_key` (OpenAPI writes send it as `Idempotency-Key`, A2A uses it
as the message id); and a write is marked started, and saved, before it runs. A read out of
time reads `"<tool> timed out after 20s"`; a write out of time, or one running when its worker
died, has an *unknown* effect: the model reads that it may or may not have taken effect and
should check, the journal keeps that, and the call is never run again blind.

## Large results

A result longer than 80,000 characters (20,000 tokens) is not what the model reads: it is
saved as a file, and the model reads where (`/large_tool_results/<id>`) with a head-and-tail
preview, then pages the rest with `read_file` (`file_path`, `offset` the line to start from,
`limit` the most lines). One convention everywhere, automatic, nothing to configure:

| Target | Who cuts | Where the whole result is kept |
|---|---|---|
| `ReAct`, Deep Agents | Deep Agents' `FilesystemMiddleware` (on by default: [react.md](frameworks/react.md#the-middleware)) | the graph's state |
| a `create_agent` graph | add `FilesystemMiddleware(tools=["read_file"])` | the graph's state |
| OpenAI Agents, Claude | the harness, as Deep Agents does (`tools/results.py`; Deep Agents is not needed) | the run's journal: it stays readable for the rest of the run, across a pause and on another worker |
| a function target, Way 2 | nobody: your code gets the whole result and decides what its model reads | — |

On OpenAI Agents `read_file` is offered once the run keeps a result (`FunctionTool.is_enabled`);
on Claude it is on the `trellis` server from the start (the CLI lists a server's tools once).
A page holds at most 20,000 characters and says where the next one starts; a line longer than
5,000 characters reads as several, so a one-line JSON result pages too.

## Tool hints: what the model is offered

The memory context is asked for with the run's own tools (not the memory tools): it comes back
with the skills the agent learned for the task ("Learned skills for this task", any number of
tools) and, from five tools (the memory service's threshold), the tools section (the next step,
argument values found in memory, what is missing) and the `tools` that fit the task
(`[{name, confidence}]`, confidence 0–1) — one call, no separate hints request. With five or
more tools, the model is then offered:

* the memory tools,
* the tools the service named (at most 8),
* every tool the run has already called (kept in the journal across a pause),
* whatever a `tool_search` call found during the run.

| Framework | Narrowing |
|---|---|
| `ReAct`, a graph with `HarnessTools` | per model call (each request carries the tools offered at that moment, sorted by name; the set only grows within a run) |
| OpenAI Agents | per turn (`FunctionTool.is_enabled`); the team's own tools are untouched |
| Claude Agent SDK | per run (the CLI lists an MCP server's tools once per query) |
| `create_agent` / Deep Agents graph built with `h.tools` | per model call: the harness runs a copy of the graph whose model calls go through one more `awrap_model_call` handler, offering the tools offered at that moment (your graph object is unchanged) |
| hand-written LangGraph `StateGraph` | none: its own code binds the model's tools (`model.bind_tools(...)`) when it is built, out of the harness's reach |
| function | n/a: it calls tools by name |

A tool outside the offered set still runs if the model calls it. When the context call fails,
or the service names no tools, the [tool search](#tool-search) narrows a catalogue of 20 tools
or more, and every tool of a smaller one is offered.

## Tool search

A catalogue of 20 tools or more that the tool hints did not narrow (memory off, or no
candidates) is narrowed by a search for the task (`tools/search.py`, no model call): BM25 over
each tool's whole definition — its name, its description, and each parameter's name,
description and allowed values — so a tool is found by what its parameters take as well as by
what it is called. The model is offered the 10 best matches, every tool the task names as it is
named, and Code Mode's and the skills' tools; the memory tools are offered as always. It works
wherever the hints do (the table above): per model call in a `ReAct` and a graph built with
`h.tools`, per turn in OpenAI Agents, per run in Claude; a function target calls any tool by
name. A `ReAct` also narrows a smaller catalogue whose definitions would take more than 40% of
its context window — the definitions, not the history, crowding a small window are narrowed
before anything is summarized or refused — and on a narrowed run searches again at every step,
with what the model last said and what the tools last answered, offering up to 3 more tools
(the offered set only grows). `without={"tool_search"}` offers every tool.

## Code Mode

The Code Mode servers of the gateway (`is_code_mode_client`: a script sees only those) whose
every allowed tool only reads — as governance says now: the catalog's risk over the tool's own,
and no `approve_when` — go to Code Mode when there are at
least 3 of them or 20 tools between them: the agent gets Bifrost's meta-tools, under the
harness's names — `list_tool_files`, `read_tool_file`, `get_tool_docs`, `execute_tool_code`,
each run as the gateway's (`listToolFiles`, ...) — scoped to those servers, and writes one
Starlark script (`server.tool(param=value)`, `print(...)`) instead of many
calls. Every other tool stays a normal tool, so no script reaches a tool that writes. Scripts
run under the run id (`x-bf-parent-request-id`); with memory writes on, their nested calls are
read back from Bifrost's MCP log after the run — it is written a few seconds behind, so it is
read until two reads agree (at most 20 s) — and recorded as tool calls. Agent Mode (the
gateway running tools itself) is never used: under its own names the gateway would run a
declared meta-tool itself, inside the completion, so the harness's names keep every call in
the bridge ([gateway.md](gateway.md#what-the-gateway-never-does-for-a-run)). Not through a
Virtual MCP (`mcp=`): a script reaches every tool of a server, a bundle only some.

## Tools built into the agent: `h.tools`

A compiled LangGraph graph (and Deep Agents) binds its tools when it is built, so `wrap(tools=)`
is refused for it. Build it with the toolbox instead — every call still goes through the
bridge, governed for the run that makes it:

```python
graph = create_agent(model, tools=await h.tools(stock, framework="langgraph"))
agent = h.wrap(graph, id="stock")
```

The result holds `stock`, the MCP tools the virtual key allows and (memory on) the memory
tools. `framework` is the adapter's name: `"langgraph"` (and `"deepagents"` for
`create_deep_agent`) returns LangChain tools; `framework="openai_agents"` returns `FunctionTool`s (for an agent reached by a handoff,
whose tools `wrap(tools=)` does not reach); `framework="claude_agent_sdk"` returns one
in-process MCP server config (add it to `mcp_servers` as `"trellis"` and allow its tools,
`mcp__trellis__<tool>`, in `allowed_tools`). Any other name is refused with the valid ones
(`ConfigurationError`). A LangGraph agent's tool hints are asked for among
these tools, and a `create_agent` or `create_deep_agent` graph's model calls are offered only
those the run offers ([tool hints](#tool-hints-what-the-model-is-offered)).

The tools are built once, but governed at each call: governance looks the call up by its tool's
name in the catalog as it is *then*, so an administrator's rule reaches a graph compiled before
it was set, within the 30 s the rules are kept ([governance.md](governance.md#way-1-inside-hwrap)).

## Inside a run

`trellis.current().tools.call(name, **args)` calls any of the run's tools through the bridge;
`tools.hints(task)` asks the memory service which of them fit (and offers the tools it names).
Inside a tool call, `trellis.current().idempotency_key` is that call's key and
`trellis.current().remaining()` the seconds it may still take (`None`: no limit).
