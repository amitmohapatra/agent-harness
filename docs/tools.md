# Tools

## The toolbox

An agent's tools, resolved once per agent and tenant and again after `TOOLS_TTL_SECONDS`
(300), in `tools/toolbox.py`:

| Where from | Tools | Side effects |
|---|---|---|
| **MCP** — automatic | every tool the agent's Bifrost virtual key allows (the gateway's own MCP listing, asked with the key), named `<server>-<tool>`, executed through the gateway | the server's annotations: `readOnlyHint` → read, `destructiveHint` → irreversible, anything else → write |
| `tool(fn)`, `@tool(...)`, or a bare function in `tools=[...]` | one; schema from the signature (pydantic validates the model's arguments), description from the docstring's first paragraph | `side_effects=` (`"write"` by default) |
| `a2a(url, *, name=None)` | one: the remote agent, `{"message": string}` in, its answer out | `"write"` |
| `openapi(spec, *, only=None, base_url=None, headers=None)` | one per `operationId`; path and query parameters and a JSON `body` flattened into one argument object | by method: GET/HEAD/OPTIONS read, POST/PUT/PATCH write, DELETE irreversible |
| the memory service (memory on) | its agent tools (see [memory.md](memory.md)) | read or write |

`tools=` takes only what runs in this process; which MCP tools an agent has is decided where
its virtual key is configured, in the gateway. Tool names must be unique across the toolbox.

With memory on, the tool catalog (`GET /v1/tools?names=`) is read for every tool, and every
tool is published to it in the background, once per content: MCP tools with their annotations
(never a `side_effects` of the harness's making, so an administrator's stays), local, OpenAPI
and A2A tools with their declared side effects.

## Tiers and `approve_when`

The catalog's `risk` — what the memory service decided from the annotations it was sent and
what an administrator set — overrides the tool's own side effects. Then:

| Side effects | Tier |
|---|---|
| `read` | runs |
| `write` | runs, announced as a `CUSTOM` `tool_notice` event |
| `irreversible` | pauses the run for approval (`InterruptReason.APPROVAL`, the call attached) |

The catalog's `approve_when` for a tool — an administrator's rule, or an approval suggestion
someone accepted (`POST /v1/tools/approval-suggestions/{id}/accept` in the memory service) —
replaces the tier: the call asks exactly when the expression holds on its arguments.

```text
amount > 10000 and currency in ["EUR", "USD"]
shape == "amount:num:1e4"
```

The expression language is the memory service's own (`trellis.memory.approval`, which writes
and validates these rules and which the harness evaluates them with — one implementation, not
two): comparisons, `and`/`or`/`not`, `in`, literals, lists, dotted argument paths, and `shape`
(the call's argument shape, which approval decisions are pooled by). A rule that does not
parse, or cannot be evaluated on a call, asks: the policy fails closed. When it does not hold
the call runs (announced unless the tool only reads). With memory off there is no catalog: the
tiers are the tools' own.

An approver may approve, reject (the model is told the call was not run), edit (the call runs
with the edited arguments) or cancel (the run ends `CANCELLED`). Each decision is also
`TOOL_CALL` feedback, from which the memory service learns approval suggestions.

## Every call

The bridge (`tools/bridge.py`) handles every harness tool call whichever framework makes it:
journal replay, tier, execution in an `execute_tool` span between `TOOL_CALL_START/ARGS/END/
RESULT` events (results previewed up to 2000 characters), then the record. A tool that raises
becomes an error result the model reads (`"<tool> failed: ..."`); a pause is never swallowed.
A harness tool called outside a run is refused.

## Tool hints: what the model is offered

When the run's own tools (not the memory tools) number at least `TOOL_HINTS_MIN` (5), the
memory context is asked for with their names: it comes back with the tools section (the next
step, argument values found in memory, what is missing) and the `tool_candidates` that fit the
task — one call, no separate hints request. The model is then offered:

* the memory tools,
* the candidates the service named (at most 8),
* every tool the run has already called (kept in the journal across a pause),
* whatever a `tool_search` call found during the run.

| Framework | Narrowing |
|---|---|
| `ReAct` | per model call (each request carries the tools offered at that moment) |
| OpenAI Agents | per turn (`FunctionTool.is_enabled`); the team's own tools are untouched |
| Claude Agent SDK | per run (the CLI lists an MCP server's tools once per query) |
| LangGraph / Deep Agents | none: a compiled graph binds its tools when it is built |
| function | n/a: it calls tools by name |

A tool outside the offered set still runs if the model calls it. When the context call fails,
or the service names no candidates, every tool is offered.

## Code Mode

The Code Mode servers of the gateway (`is_code_mode_client`: a script sees only those) whose
every allowed tool is `read` and carries no `approve_when` go to Code Mode when there are at
least 3 of them or 20 tools between them: the agent gets Bifrost's meta-tools
(`listToolFiles`, `readToolFile`, `getToolDocs`, `executeToolCode`) scoped to those servers
and writes one Starlark script (`server.tool(param=value)`, `print(...)`) instead of many
calls. Every other tool stays a normal tool, so no script reaches a tool that writes. Scripts
run under the run id (`x-bf-parent-request-id`); with memory writes on, their nested calls are
read back from Bifrost's MCP log after the run — it is written a few seconds behind, so it is
read until two reads agree (at most 20 s) — and recorded as tool calls. Agent Mode (the
gateway running tools itself) is never used.

## Tools built into the agent: `h.tools`

A compiled LangGraph graph (and Deep Agents) binds its tools when it is built, so `wrap(tools=)`
is refused for it. Build it with the toolbox instead — every call still goes through the
bridge, under the tiers of the run that makes it:

```python
graph = create_agent(model, tools=await h.tools(stock, framework="langgraph"))
agent = h.wrap(graph, id="stock")
```

The result holds `stock`, the MCP tools the virtual key allows and (memory on) the memory
tools. `framework="openai-agents"` returns `FunctionTool`s; `framework="claude-agent-sdk"`
returns one in-process MCP server config (add it to `mcp_servers` as `"trellis"` and allow its
tools, `mcp__trellis__<tool>`, in `allowed_tools`). A LangGraph agent's tool hints are asked
for among these tools.

## Inside a run

`trellis.current().tools.call(name, **args)` calls any of the run's tools through the bridge;
`tools.hints(task)` asks the memory service which of them fit (and offers the candidates).
