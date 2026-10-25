# Tools

## The toolbox

An agent's tools, per agent and tenant, in `tools/toolbox.py` — their definitions listed again
after `TOOLS_TTL_SECONDS` (300); while a listing fails (the gateway is down), the last good one
stands and is listed again after `TOOLS_RETRY_SECONDS` (30). Whether a call runs, is announced or asks is not the tools':
[governance](governance.md) decides it at each call, from the side effects below and the tool
catalog.

| Where from | Tools | Side effects |
|---|---|---|
| **MCP** — automatic, with a gateway | every tool the agent's Bifrost virtual key allows (the gateway's own MCP listing, asked with the key), named `<server>-<tool>`, executed through the gateway, each request waiting what is left of the run's time and saying who the run is for; a tool the gateway would run itself (`tools_to_auto_execute`) is left out ([gateway.md](gateway.md)); `without={"mcp"}` is no MCP tool at all | the server's annotations: `readOnlyHint` → read, `destructiveHint` → irreversible, anything else → write; `idempotentHint` makes it idempotent (retried like a read) |
| `mcp(target, *, name=None, call_timeout=120, headers=None)` | an MCP server's tools, named `<server>-<tool>` (`name=`, else the name the server gives itself): `mcp("https://host/mcp")` a server over HTTP (`headers=` for its auth), `mcp(stdio("npx", "@mcp/files"))` a local server the harness starts (`stdio(command, *args, env=None, cwd=None)`). A server of your own is reached with the official MCP SDK (the `mcp` extra): one connection per server, kept, opened again after it breaks; a server reached both directly and through the gateway is an error (one path per server). Every call goes through the bridge like any other tool: governance, the journal, retries of reads, paging, the run's records. Code Mode is for the gateway's servers | the server's annotations, as above |
| `tool(fn)`, `@tool(...)`, or a bare function in `tools=[...]` | one; schema from the signature (pydantic validates the model's arguments), each parameter described by the docstring's `Args:` (or `:param x:`) unless `Annotated[..., Field(description=...)]` says; description from the docstring without its sections (`Args:`, `Returns:`, `Raises:`...), every other paragraph kept — a "use it when / not when" paragraph reaches the model; `call_timeout=` seconds per call (none by default; a sync function runs in a worker thread); `idempotent=True` when a call repeated with its idempotency key has its effect once (retried like a read, run again after a crash); a result that comes from outside the run is an `ask` in the function ([interrupts.md](interrupts.md#results-from-outside-ask-in-the-tool)) | `side_effects=` (`"write"` by default) |
| `a2a(url, *, name=None, call_timeout=120)` | one: the remote agent, `{"message": string}` in, its answer out; at most `call_timeout` seconds per exchange | `"write"` |
| a wrapped agent, in a `ReAct`'s `subagents=[...]` | one, named after the agent's id: `{"message": string}` in, its answer out — each call a child run of it ([subagents.md](subagents.md)) | `"read"` when every tool it declares only reads (and none escapes the harness), else `"write"` |
| `openapi(spec, *, only=None, base_url=None, headers=None, call_timeout=120)` | one per `operationId`; path and query parameters and a JSON `body` flattened into one argument object; at most `call_timeout` seconds per operation | by method: GET/HEAD/OPTIONS read, POST/PUT/PATCH write, DELETE irreversible |
| `sandbox(provider=None, *, image=None, spec=None, call_timeout=120)` | `sandbox_exec`, `sandbox_read` and `sandbox_write`, in a sandbox of the run's own (Docker, or the provider given), made at its first call and deleted at its end ([sandbox.md](sandbox.md)) | `sandbox_exec` and `sandbox_write` (idempotent) `"write"`, `sandbox_read` `"read"` |
| your framework's own tools (a graph's, an OpenAI Agents agent's, Claude's built-ins and in-process servers) | the framework runs them; every call is governed as it makes it, published to the catalog, journaled and recorded | the catalog's risk (an MCP server's annotations where it has them), else `"write"` |
| the memory service (memory on) | its agent tools (see [memory.md](memory.md)) | read or write |

What an agent can call is said where it is built: a `ReAct`'s `tools=[...]` (functions,
`tool(...)`, LangChain tools, `mcp(...)`, `openapi`, `sandbox()`) and `subagents=[...]`
(wrapped agents, `a2a`); your framework's agent with its own tools; a function target's code
with `agent.tools.call(tool, ...)`. Which of the gateway's MCP tools an agent has is decided
where its virtual key is configured, in the gateway. Tool names must be unique across the
toolbox.

With memory on, every tool is published to the memory service's tool catalog, through
governance, in the background and once per content: MCP tools (the gateway's and your own
servers') with their annotations, local, OpenAPI and A2A tools with their declared side
effects. An administrator sets `approve_when` on
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
| **Fewer tools per call** — the memory service's tool hints (from 5 tools, [below](#tool-hints-what-the-model-is-offered)) and its `tool_search` tool: the one owner of narrowing. Without memory every tool is offered (the check below warns from 20). A team that wants narrowing without memory adds its own: LangChain's `LLMToolSelectorMiddleware` in a `ReAct` or `create_agent` graph (its own model call per step picks the tools) — the team's code, not a harness fallback | automatic with memory; your opt-in without: `ReAct(middleware=[LLMToolSelectorMiddleware(max_tools=8, on_parsing_failure="all")])` |
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
| a `create_agent` graph | add `FilesystemMiddleware(tools=["read_file"])` (or build it with `create_deep_agent`) | the graph's state |
| OpenAI Agents, Claude | the harness, as Deep Agents does (`tools/results.py`; Deep Agents is not needed) | the run's journal: it stays readable for the rest of the run, across a pause and on another worker |
| a function target, Way 2 | nobody: your code gets the whole result and decides what its model reads | — |

On OpenAI Agents `read_file` is offered once the run keeps a result (`FunctionTool.is_enabled`);
on Claude it is on the `trellis` server from the start (the CLI lists a server's tools once).
A page holds at most 20,000 characters and says where the next one starts; a line longer than
5,000 characters reads as several, so a one-line JSON result pages too.

## The run's tool set

A run's tools are one set: the harness's (the toolbox above) and the framework's own — a graph's
tool nodes' tools (a hand-built `StateGraph`'s `ToolNode` included), a Deep Agent's `task`
sub-agents' tools, an OpenAI agent's function and hosted tools, its MCP servers' tools (as the
servers list them), its handoffs (`transfer_to_...`) and the tools of the agents it hands off to
or runs as tools, Claude Code's built-ins and its MCP servers' tools (an in-process server's as
it lists them, the ones `allowed_tools` names, the ones it was seen calling). The set is sorted
by name, each name once, so the memory context's cache key holds. It is what the tool hints and
`tool_search` choose among, what the tenant's catalog is told about (once per set of names) and
what the narrowing narrows. A tool that hands the work to another agent (Deep Agents' `task`, a
handoff, an agent tool, Claude's `Agent`) is of the set but never narrowed away, and its call is
a `read`: each of the other agent's own calls is governed on its own.

## Tool hints: what the model is offered

The memory context is asked for with the run's tool set (not the memory tools): it comes back
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
| OpenAI Agents | per turn (`FunctionTool.is_enabled`): the harness's tools and the agent's own function tools (theirs is kept, and asked first); its hosted tools are the provider's |
| Claude Agent SDK | per run (the CLI lists an MCP server's tools once per query): the harness's tools, and Claude's own built-ins and MCP tools left out of the query's `disallowed_tools` (yours are kept) |
| `create_agent` / Deep Agents graph | per model call: the harness runs a copy of the graph whose model calls go through one more `awrap_model_call` handler, offering the tools offered at that moment — its own tools of the set too (your graph object is unchanged) |
| hand-written LangGraph `StateGraph` | none: its own code binds the model's tools (`model.bind_tools(...)`) when it is built, out of the harness's reach |
| function | n/a: it calls tools by name |

A tool outside the offered set still runs if the model calls it. When the context call fails,
or the service names no tools, every tool is offered.

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
the bridge ([gateway.md](gateway.md#what-the-gateway-never-does-for-a-run)).

## A framework's own tools

A framework's agent keeps its tools: a LangGraph graph's (bound when it is built), an OpenAI
Agents agent's `function_tool`s and handoffs, Claude Code's built-ins and your in-process MCP
servers. The harness finds them in the target, adds them to the run's tool set (the tool hints
and the narrowing see them), publishes them to the catalog, and governs every call as the
framework makes it — the catalog's risk and `approve_when`, approvals, the journal, the records:

```python
graph = create_agent(model, tools=[stock], middleware=[HarnessTools()])
agent = h.wrap(graph, id="stock")
```

`HarnessTools()` (`trellis.harness.middleware`) adds the harness's tools — the MCP tools the
virtual key allows and (memory on) the memory tools — to every model call of a `create_agent`
or Deep Agents graph; an OpenAI Agents or Claude run gets them on its copy of the agent. A
`create_agent` or `create_deep_agent` graph's model calls are offered only the tools the run
offers ([tool hints](#tool-hints-what-the-model-is-offered)).

The tools are built once, but governed at each call: governance looks the call up by its tool's
name in the catalog as it is *then*, so an administrator's rule reaches a graph compiled before
it was set, within the 30 s the rules are kept ([governance.md](governance.md#way-1-inside-hwrap)).

## Inside a run

`trellis.current().tools.call(tool, **args)` calls one of the run's tools through the bridge —
by name, or a tool itself (`tool(fn)`, `mcp(...)`, `a2a(url)`...: it joins the run first);
`tools.use(*sources)` adds a function target's own tools to the run;
`tools.hints(task)` asks the memory service which of them fit (and offers the tools it names).
Inside a tool call, `trellis.current().idempotency_key` is that call's key and
`trellis.current().remaining()` the seconds it may still take (`None`: no limit).
