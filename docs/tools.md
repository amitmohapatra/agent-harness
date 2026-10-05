# Tools

## The toolbox

An agent's tools, per agent and tenant, in `tools/toolbox.py` — their definitions listed again
after `TOOLS_TTL_SECONDS` (300); while a listing fails (the gateway is down), the last good one
stands and is listed again after `TOOLS_RETRY_SECONDS` (30). Whether a call runs, is announced or asks is not the tools':
[governance](governance.md) decides it at each call, from the side effects below and the tool
catalog.

| Where from | Tools | Side effects |
|---|---|---|
| **MCP** — automatic | every tool the agent's Bifrost virtual key allows (the gateway's own MCP listing, asked with the key) — or, with `h.wrap(..., mcp=[slug])`, the tools of those Virtual MCPs — named `<server>-<tool>`, executed through the gateway, each request waiting what is left of the run's time and saying who the run is for; a tool the gateway would run itself (`tools_to_auto_execute`) is left out ([gateway.md](gateway.md)) | the server's annotations: `readOnlyHint` → read, `destructiveHint` → irreversible, anything else → write; `idempotentHint` makes it idempotent (retried like a read) |
| `tool(fn)`, `@tool(...)`, or a bare function in `tools=[...]` | one; schema from the signature (pydantic validates the model's arguments), description from the docstring's first paragraph; `timeout=` seconds per call (none by default; a sync function runs in a worker thread) | `side_effects=` (`"write"` by default) |
| `a2a(url, *, name=None, timeout=None)` | one: the remote agent, `{"message": string}` in, its answer out; at most `timeout` per exchange (120 s by default) | `"write"` |
| `agent.as_tool(*, name=None, description=None, side_effects=None)` | one: another agent wrapped by this harness, `{"message": string}` in, its answer out — each call a child run of it ([subagents.md](subagents.md)) | `"read"` when every tool it declares only reads (and none escapes the harness), else `"write"`; `side_effects=` overrides it |
| `openapi(spec, *, only=None, base_url=None, headers=None, timeout=30)` | one per `operationId`; path and query parameters and a JSON `body` flattened into one argument object; at most `timeout` seconds per operation | by method: GET/HEAD/OPTIONS read, POST/PUT/PATCH write, DELETE irreversible |
| `h.wrap(..., skills=[...])`, `skills(...)` | `load_skill` and `read_skill_file`: skills of the gateway's Skills Repository, their versions pinned per run ([gateway.md](gateway.md#skills)) | `"read"` |
| the memory service (memory on) | its agent tools (see [memory.md](memory.md)) | read or write |

`tools=` takes only what runs in this process; which MCP tools an agent has is decided where
its virtual key is configured, in the gateway. Tool names must be unique across the toolbox.

With memory on, every tool is published to the memory service's tool catalog, through
governance, in the background and once per content: MCP tools with their annotations, local,
OpenAPI and A2A tools with their declared side effects. An administrator sets `approve_when` on
them there ([governance.md](governance.md)). Concurrent runs that find the toolbox stale share
one listing.

## Every call

The bridge (`tools/bridge.py`) handles every harness tool call whichever framework makes it:
journal replay, governance (run, announce, or pause for approval:
[governance.md](governance.md#way-1-inside-hwrap)), execution in an `execute_tool` span between
`TOOL_CALL_START/ARGS/END/RESULT` events (arguments and results redacted —
[observability.md](observability.md#redaction) — and results previewed up to 2000 characters), then the
record. A tool that raises becomes an error result the model reads (`"<tool> failed: ..."`); a
pause is never swallowed. A harness tool called outside a run is refused. Calls may come at
once (`ReAct`'s reads, a framework running tools concurrently): identical ones take their
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

## Tool hints: what the model is offered

When the run's own tools (not the memory tools) number at least `TOOL_HINTS_MIN` (5), the
memory context is asked for with their names: it comes back with the tools section (the next
step, argument values found in memory, what is missing) and the `tools` that fit the task
(`[{name, confidence}]`, confidence 0–1) — one call, no separate hints request. The model is then offered:

* the memory tools,
* the tools the service named (at most 8),
* every tool the run has already called (kept in the journal across a pause),
* whatever a `tool_search` call found during the run.

| Framework | Narrowing |
|---|---|
| `ReAct` | per model call (each request carries the tools offered at that moment, sorted by name; the set only grows within a run) |
| OpenAI Agents | per turn (`FunctionTool.is_enabled`); the team's own tools are untouched |
| Claude Agent SDK | per run (the CLI lists an MCP server's tools once per query) |
| LangGraph / Deep Agents | none: a compiled graph binds its tools when it is built |
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
tools. `framework="openai-agents"` returns `FunctionTool`s (for an agent reached by a handoff,
whose tools `wrap(tools=)` does not reach); `framework="claude-agent-sdk"` returns one
in-process MCP server config (add it to `mcp_servers` as `"trellis"` and allow its tools,
`mcp__trellis__<tool>`, in `allowed_tools`). A LangGraph agent's tool hints are asked for among
these tools.

The tools are built once, but governed at each call: governance looks the call up by its tool's
name in the catalog as it is *then*, so an administrator's rule reaches a graph compiled before
it was set, within the 30 s the rules are kept ([governance.md](governance.md#way-1-inside-hwrap)).

## Inside a run

`trellis.current().tools.call(name, **args)` calls any of the run's tools through the bridge;
`tools.hints(task)` asks the memory service which of them fit (and offers the tools it names).
Inside a tool call, `trellis.current().idempotency_key` is that call's key and
`trellis.current().remaining()` the seconds it may still take (`None`: no limit).
