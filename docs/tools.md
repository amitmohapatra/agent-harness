# Tools

## Sources

`tools=[...]` on `h.wrap` (and `h.tools(...)`) accepts:

| Source | Tools | Side effects |
|---|---|---|
| `tool(fn)`, `@tool(...)`, or a bare function | one; schema from the signature (pydantic validates the model's arguments), description from the docstring's first paragraph | `side_effects=` (`"write"` by default) |
| `mcp(*servers, only=None)` | the servers' tools as Bifrost lists them, named `<server>-<tool>` (`only` takes bare or qualified names) | from the memory service's tool catalog; `"write"` when unknown |
| `a2a(url, *, name=None)` | one: the remote agent, `{"message": string}` in, its answer out | `"write"` |
| `openapi(spec, *, only=None, base_url=None, headers=None)` | one per `operationId`; path and query parameters and a JSON `body` flattened into one argument object | by method: GET/HEAD/OPTIONS read, POST/PUT/PATCH write, DELETE irreversible |

Tool names must be unique across an agent's sources. Local, OpenAPI and A2A tools are
published to the memory service's catalog (in the background) when memory is configured.

## Tiers and approve rules

| Side effects | Tier |
|---|---|
| `read` | runs |
| `write` | runs, announced as a `CUSTOM` `tool_notice` event |
| `irreversible` | pauses the run for approval (`InterruptReason.APPROVAL`, the call attached) |

`approve={"tool-name": rule}` replaces the tier for that tool. A rule is `True` (always ask),
`False` (never ask) or a condition over the call's arguments:

```python
approve = {"erp-create_po": "amount > 10000 and currency in ['EUR', 'USD']"}
```

Conditions allow comparisons, `and`/`or`/`not`, `+ - * / %`, `in`, literals, lists/tuples and
argument names; anything else fails at `wrap`. A condition that cannot be evaluated on a call
(a missing argument, a type error) asks. When a rule does not ask, the call runs (announced
unless the tool only reads).

An approver may approve, reject (the model is told the call was not run), edit (the call runs
with the edited arguments) or cancel (the run ends `CANCELLED`).

## Every call

The bridge (`tools/bridge.py`) handles every harness tool call whichever framework makes it:
journal replay, tier, execution in a `trellis.tool` span between `TOOL_CALL_START/ARGS/END/
RESULT` events (results previewed up to 2000 characters), then the record. A tool that raises
becomes an error result the model reads (`"<tool> failed: ..."`); a pause is never swallowed. A
harness tool called outside a run is refused.

## Tools built into the agent: `h.tools`

A compiled LangGraph graph (and Deep Agents) binds its tools when it is built, so `wrap(tools=)`
is refused for it. Build with the harness tools instead — every call still goes through the
bridge, under the policy of the agent that is running:

```python
tools = await h.tools(stock, mcp("erp"), framework="langgraph", memory=True)
graph = create_agent(model, tools=tools)
agent = h.wrap(graph, id="stock", memory="read_write", approve={"erp-create_po": True})
```

`framework="openai-agents"` returns `FunctionTool`s; `framework="claude-agent-sdk"` returns one
in-process MCP server config (add it to `mcp_servers` as `"trellis"` and allow its tools,
`mcp__trellis__<tool>`, in `allowed_tools`).

## Code Mode

An `mcp(...)` source with at least 20 tools or 3 servers, all of them `read` in the catalog and
all served by Code Mode clients of the gateway (`is_code_mode_client`: a script sees no other
server), is given to the agent as Bifrost's Code Mode meta-tools (`listToolFiles`,
`readToolFile`, `getToolDocs`, `executeToolCode`) scoped to its servers: the model writes one
Starlark script (`server.tool(param=value)`, `print(...)`) instead of many calls. Scripts run under the run id (`x-bf-parent-request-id`); with
`memory="read_write"` their nested calls are read from Bifrost's MCP log ten seconds after the
run and recorded as tool calls. One non-read tool keeps the whole source in normal mode, so a
script never reaches a write.

## Inside a run

`trellis.current().tools.call(name, **args)` calls any of the run's tools through the bridge;
`tools.hints(task)` asks the memory service which of them fit (needs memory).
