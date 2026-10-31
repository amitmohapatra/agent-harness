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
| `mcp(target, *, name=None, call_timeout=120, headers=None)` | an MCP server's tools, named `<server>-<tool>` (`name=`, else the name the server gives itself): `mcp("https://host/mcp")` a server over HTTP (`headers=` for its auth), `mcp(stdio("npx", "@mcp/files"))` a local server the harness starts (`stdio(command, *args, env=None, cwd=None)`). A server of your own is reached with the official MCP SDK (the `mcp` extra): one connection per server, kept, opened again after it breaks; a server reached both directly and through the gateway is an error (one path per server). Every call goes through the bridge like any other tool: governance, the journal, retries of reads, paging, the run's records. A server that asks the person in the middle of a call ([MCP elicitation](#a-server-that-asks-mcp-elicitation)) pauses the run as `ask` does. Code Mode is for the gateway's servers | the server's annotations, as above |
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

### A server that asks: MCP elicitation

An `mcp(...)` server may ask the person in the middle of a call (MCP elicitation): the harness
tells the server it can be asked (the SDK's client capability, so a server asks only then), and
a question pauses the run as `trellis.current().ask` does ([interrupts.md](interrupts.md)):
the run's user's to answer, in their inbox, on AG-UI's interrupt, over A2A. What they are shown
follows from the server's form (`tools/elicit.py`):

| The server asks | The person is shown | The server gets |
|---|---|---|
| one field that picks from values (`enum`, or `oneOf` of `const` with a `title`; an `array` of them picks several) | a choice (`options`, `multiple`): a dropdown | `{field: value}` |
| any other form (strings, numbers, booleans, a `format: date` field) | a form with the server's schema as `expects` (a date picker where the surface renders JSON Schema) | the form's content |
| a link to open (`mode: url`) | a choice: opened it, or declined | `accept` or `decline` |

A rejection is the server's `decline`; a cancel ends the run (the server is told). The answer
is journaled as any `ask`'s: the call is made again when the run resumes (on any worker), and
the server is answered at once. On the MCP protocol of 2026-07-28 the question comes back in
the call's result and the call is made again with the answer; before it, the server's question
is a request of its own that names no call, so a question that comes while several calls to
the same server are under way is refused, the server told why.

## Tool choice: what helps the model pick the right tool

What a model chooses from is each tool's name, description and parameter schema — kept as
the source gave them (an MCP server's, through the gateway, as the server lists them; a
function's from its signature and docstring; an OpenAPI operation's), in every framework's own
format. What helps it choose, whatever the source:

| | Where |
|---|---|
| **Descriptions worth choosing by** — what the tool is for, when to use it rather than a broader or narrower one, each parameter described | the tool's author: the docstring (functions), the MCP server, the OpenAPI document. The harness checks, below |
| **A check of the tool set** — `trellis.harness.tools.lint.lint(specs)`: a missing or short description (under 5 words), an undescribed parameter, a name the OpenAI API refuses (letters, digits, `_`, `-`, at most 64), names differing only in case or separators, two tools described the same or nearly (75% of their words shared), 20 tools or more at once (OpenAI's guide suggests fewer), over 128 (the most the OpenAI API takes) | automatic: the toolbox logs each finding as a warning (`trellis.tools`, `tool choice: ...`) when the tools it lists change; in CI, assert `lint([t.spec for t in tools]) == []` |
| **Fewer tools per call** — the memory service's tool hints (they narrow from 30 tools in the run, [below](#tool-hints-what-the-model-is-offered)) and its `tool_search` tool: the one owner of narrowing. Without memory every tool is offered (the check below warns from 20). A team that wants narrowing without memory adds its own: LangChain's `LLMToolSelectorMiddleware` in a `ReAct` or `create_agent` graph (its own model call per step picks the tools) — the team's code, not a harness fallback | automatic with memory; your opt-in without: `ReAct(middleware=[LLMToolSelectorMiddleware(max_tools=8, on_parsing_failure="all")])` |
| **Tool chains** — which tool's output is another's argument (`find_sku → sku_id → get_stock`), read from the schemas the tools publish ([below](#tool-chains)) | automatic: a function's return annotation, an MCP tool's `outputSchema` |
| **What it can do** — capabilities beyond the tools, said plainly | the instructions: `ReAct("... You cannot accept an image upload.", ...)` |
| **Measured** — `tool_coverage()` scores each evaluation item by the share of its `needed_tools` the run called | [evaluation.md](evaluation.md#trajectories) |

[`examples/06_scenarios/tool_choice.py`](../examples/06_scenarios/tool_choice.py) and
`tests/live/test_live_tool_choice.py` measure tool choice per item; every framework gets the
descriptions, the lint and the hints ([frameworks/react.md](frameworks/react.md#tool-choice)).

### Tool chains

When one tool's answer is another's argument — `find_sku(name)` gives a `sku_id`,
`get_stock(sku_id)` takes it — the harness reads it from the schemas the tools publish
(`trellis.harness.tools.links`): a field of one tool's output schema that a declared argument
of another takes, by name and compatible type (records in a list included), or an `id` by the
tool's noun (`find_sku`'s `id` is a `sku_id`). Generic names (`name`, `status`, `items`...)
link nothing. Each linked tool's description then ends with its chains, in every framework:

```text
Chains: find_sku → sku_id → get_stock; get_stock → supplier_id → supplier_lead_time.
```

so a model with nothing learned yet calls them in order, and the tool hints keep a hinted
tool's chain offered ([below](#tool-hints-what-the-model-is-offered)). The output schema is a
function's return annotation (`-> Stock`, a pydantic model or a `TypedDict`: pydantic's
`TypeAdapter`), an MCP server's `outputSchema`; a tool that declares none (an untyped
function, an OpenAPI operation, a gateway's MCP tool, whose listing carries no output schema)
links nothing — nothing is guessed. Once runs succeed, the memory service learns the chain as
a procedure with its argument bindings (`step 1.sku_id` from step 0's output) and hands the
next step and its arguments over with the hints. `benchmark/cost` measures chains
(`--suite classes --classes chain`: two to four dependent calls, cold and reworded).

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
| `ReAct` given a backend, Deep Agents | Deep Agents' `FilesystemMiddleware` (on by default: [react.md](frameworks/react.md#the-middleware)) | the graph's backend (its state, a disk, the run's sandbox) |
| `ReAct` (no backend), a `create_agent` graph with `middleware()` | the harness (`HarnessTools`, as Deep Agents does: `tools/results.py`), its `read_file` offered once a result is kept | the run's journal: it stays readable for the rest of the run, across a pause and on another worker |
| OpenAI Agents, Claude, Kimi (MCP) | the harness, the same way (Deep Agents is not needed) | the run's journal |
| a `create_agent` graph of your own (no `middleware()`) | add `FilesystemMiddleware(tools=["read_file"])` (or build it with `create_deep_agent`) | the graph's state |
| a function target, Way 2 | nobody: your code gets the whole result and decides what its model reads | — |

On OpenAI Agents `read_file` is offered once the run keeps a result (`FunctionTool.is_enabled`);
on Claude it is on the `trellis` server from the start (the CLI lists a server's tools once).
A page holds at most 20,000 characters and says where the next one starts; a line longer than
5,000 characters reads as several, so a one-line JSON result pages too.

A result under 80,000 characters that still does not fit a small window is cut to its head
and tail by `FitWindow`; the cut is kept the same way and says where the rest is
(`…[N characters cut to fit the window; the rest: read_file(file_path="…", offset=L)]…`, the
line the cut starts on), so no row is silently out of reach.

### Counting, summing, ranking over a large result

A model that pages 10,000 rows into its context to add them up gets the sum wrong, so it never
has to. In order of preference:

1. **the tool's own arguments** — a tool that filters, sorts, limits or aggregates on its side
   (`max_stock`, `category`, `top`, `group_by`) answers exactly and small: say so in its
   parameters' descriptions (the model reads every tool's schema; the lint warns about a
   parameter with no description);
2. **`query_result`**: offered with `read_file` whenever a result is kept, on every adapter
   (ReAct with or without a backend, `middleware()`, Deep Agents, OpenAI Agents, Claude, Kimi):
   a read-only SQL `SELECT` over every row, in process (`tools/query.py`: Python's own SQLite,
   no dependency, no sandbox). Every result the run saved is a table named by the tool that
   returned it (`result` is the file asked about), so one query joins results of several tools;
   a JSON array of objects (or the first one a JSON object holds), CSV with a header
   (`,` tab `;` `|`), else one row per line, with whole-number, number and text columns. Besides
   SQLite's arithmetic, `CASE`, math, date and window functions (a moving average, `lag` for
   growth), it has `stddev`, `variance`, `median`, `percentile_cont(x, p)`, `corr`,
   `covar_samp`, `regr_slope`, `regr_intercept`, `regr_r2` (named as PostgreSQL and DuckDB
   name them). Only reading runs (an authorizer: `SELECT` and functions; never a write,
   `ATTACH`, a `PRAGMA` or an extension), on a database made for the query and closed after it
   whatever happens, in a worker thread (4 at once in a process): a query runs at most 5 s
   (a cancelled run interrupts it), loads at most 64 MB of its own run's results (never
   another run's), holds at most 8 MB of sort memory (more spills to SQLite's temporary
   files), and answers at most 200 rows and 20,000 characters — the model reads the small
   answer, never the rows. A tool declared `paged=True` (a cursor argument; `next_cursor` in
   its result) is fetched to its last page in one call (`tools/pages.py`, at most 200 pages),
   so its rows are queried whole too;
3. **code, when configured** — `ReAct(..., backend=sandbox())` (Deep Agents saves the large
   result in the sandbox, and `execute` runs code over it), or `agent.tools.use(sandbox().tools())`
   on any adapter (`sandbox_exec`): every result the run keeps is written into the run's sandbox
   at the path the model was given, before a command runs there, and the preview says so; or
   **Code Mode** for a gateway's read-only MCP tools ([Code Mode](#code-mode)): the Starlark
   script calls the tool and computes over what it returns inside the gateway.

`benchmark/cost` measures it (`--suite classes --classes aggregate`: a count, a sum, a filter,
a top-N and a group over 10,000 SKUs).

## Values that change: freshness

A tool's data is live by default: any value from an earlier turn, a replayed window or memory is
a dated snapshot. A tool says otherwise beside its other metadata: `tool(fn, max_age=300)`
(seconds a value stays current) or `tool(fn, freshness="static")` (reference data: an earlier
value is as good as a new one); `freshness="live"` also says the default in the tool's
description (`tools.base.CHANGES`). The same keys go in a LangChain tool's `metadata` (beside
`side_effects`) and an MCP tool's `_meta` (`{"freshness": "static"}`, `{"maxAge": 300}`).

Where memory owns the conversation, every tool result a run writes to it starts with
`[snapshot of <tool> from <UTC time>]`, so a later turn replays when (and by which tool) each
value was fetched — the model can say where a number came from. A replayed result past its
tool's age (0 unless declared) is marked out of date in the replay, the run's context says the
same in one line, and the tool stays offered even when the tool hints narrow the rest: the model
fetches the value again instead of answering from the old one, and the fresh result, with the
older one dated beside it, is what the answer uses (`trellis.harness.snapshots`). A repeated
call is answered from an earlier one only within its own model step and run, never across
turns; `memory_search` items carry `observed_on`. The
memory service never turns a tool's result into a long-term fact (it forms memories from what
it was told, not from tool output), so the user's facts and preferences stay and live values do
not.

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
(`[{name, confidence}]`, confidence 0–1) — one call, no separate hints request. With 30 or
more tools in the run (`NARROW_FROM`: its own, its framework's and memory's), the model is then
offered the following; with fewer, every tool, and the tools section still says which fit
(narrowing a set of 20 hid a tool the question needed in about a quarter of the turns, and
those turns went wrong two to three times as often: the holistic small-model run):

* the memory tools,
* the tools the service named (at most 8),
* every tool the run has already called (kept in the journal across a pause),
* whatever a `tool_search` call found during the run,
* the chain of a tool named ([tool chains](#tool-chains)): the tools its output feeds, and
  those whose output gives an argument it requires.

| Framework | Narrowing |
|---|---|
| `ReAct`, a graph with `HarnessTools` | per model call (each request carries the tools offered at that moment, in the order first offered — the first call's by name, those offered later after them — so the set only grows at its end within a run and a provider's prompt cache keeps its prefix) |
| OpenAI Agents | per turn (`FunctionTool.is_enabled`): the harness's tools and the agent's own function tools (theirs is kept, and asked first); its hosted tools are the provider's |
| Claude Agent SDK | per run (the CLI lists an MCP server's tools once per query): the harness's tools, and Claude's own built-ins and MCP tools left out of the query's `disallowed_tools` (yours are kept) |
| `create_agent` / Deep Agents graph | per model call: the harness runs a copy of the graph whose model calls go through one more `awrap_model_call` handler, offering the tools offered at that moment — its own tools of the set too (your graph object is unchanged) |
| hand-written LangGraph `StateGraph` | none: its own code binds the model's tools (`model.bind_tools(...)`) when it is built, out of the harness's reach |
| function | n/a: it calls tools by name |

A tool outside the offered set still runs if the model calls it. When the context call fails,
the service names no tools, or its best hint's confidence is under 0.5 (`NARROW_FLOOR`:
nothing matched the task, as before the index has the tools), every tool is offered — the
window fits them — with `tool_search` among the memory tools. The tools behind Code Mode are
asked about too; when the service names one, the Code Mode meta-tools stay in the offered set,
the only way to reach it.

A tenant's tools are published to the catalog when a worker starts (the key's tenant), and a
tenant's first hints wait — at most 5 s (`INDEX_WAIT_SECONDS`) — until the catalog says tool
search finds them (`GET /v1/tools`, `indexed`): asked before, every hint is a zero. A confident hint (0.5
or more) also means a tool answers the question: the "Memory holds nothing … do not guess" note
is then left out of the context.

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
