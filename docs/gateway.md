# The Bifrost gateway

With Bifrost as its route (`LLM_GATEWAY=bifrost`, `LLM_BASE_URL` its `/v1` root, `LLM_API_KEY`
the virtual key) the harness reaches models and MCP tools through it: the tools a virtual key
allows are the agent's ([tools.md](tools.md)), and every model name is routed through it
([configuration.md](configuration.md#models)). Any other OpenAI-compatible endpoint serves the
models alone (`LLM_GATEWAY=openai-compatible`); what follows is Bifrost's. This page is the
rest of what the harness uses of the gateway — stored prompts, who a tool call is for — and
what it makes sure the gateway never does for a run. Nothing here is configured beyond the
three variables.

| You want | Use | Section |
|---|---|---|
| a system prompt kept, versioned and changed in the gateway, not in code | `ReAct(system_prompt=stored("triage"))`, `await h.prompt("triage")` | [Prompts](#prompts) |
| an agent limited to some tools of some MCP servers | the virtual key's MCP allow-list, in Bifrost | [Who a call is for](#who-a-call-is-for) |
| an MCP server acting for the person a run is for | nothing: every call says who | [Who a call is for](#who-a-call-is-for) |
| your framework's own MCP client on the gateway | its client at `/mcp` with the key — ungoverned | [Frameworks' own MCP clients](#frameworks-own-mcp-clients) |

## What the gateway never does for a run

**Automatic.** Every model call the harness routes through the gateway (a model name, the
judge; an OpenAI Agents run, the Claude CLI) sends the gateway's deny-all MCP scope
(`x-bf-mcp-include-clients` and `x-bf-mcp-include-tools`, empty): the gateway adds none of the
key's MCP tools to the request and runs none itself (Agent Mode). A model client of your own
pointed at the gateway sends them itself (`bifrost_sdk.NO_GATEWAY_TOOLS`):

```python
from bifrost_sdk import NO_GATEWAY_TOOLS

model = ChatOpenAI(
    base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=MODEL, default_headers=NO_GATEWAY_TOOLS
)
```

Without them the gateway adds the key's MCP tools to the framework's requests (the model is
offered tools nobody declared) and runs, itself, any tool a client lists in
`tools_to_auto_execute`.

Two more things would let the gateway run a call out of the harness's sight, and the harness
prevents both:

* **Code Mode's meta-tools.** The gateway's agent loop runs a call of `listToolFiles`,
  `readToolFile` or `getToolDocs` (and of `executeToolCode` whose script calls no tool, or
  only auto-executed ones) itself, inside the completion, whenever the request declares the
  tool — whatever the MCP scope says. The harness offers them as `list_tool_files`,
  `read_tool_file`, `get_tool_docs` and `execute_tool_code`: the gateway does not know those
  names, so every call comes back to the harness and goes through the bridge — journaled,
  governed, recorded — which runs it as the gateway's meta-tool. (A model that writes a gateway
  name from a tool's output anyway gets the gateway's answer under the deny-all scope: no
  server, no data, no effect.)
* **Agent Mode tools.** A tool its MCP client lists in `tools_to_auto_execute` is answered by
  the gateway itself when the model calls it, so it is not offered, and the log says why
  (`... is not offered: its MCP client ... lists it in tools_to_auto_execute`). Keep that list
  empty (bifrost-sdk's `MCPClientConfig` does).

**On failure.** The Agent Mode lists are read from the gateway's management API
(`GET /api/mcp/clients`) with the virtual key. A gateway whose admin auth closes `/api` to the
key leaves them unchecked (logged at INFO): its deny-all scope still refuses such a call, so
the tool fails instead of running unseen.

## Prompts

**What.** A stored prompt of the gateway's Prompt Repository: messages committed in versions
(1, 2, 3, ...) by whoever owns the prompt, not by a deploy — the one place the harness reads
stored prompts from ([prompts.md](prompts.md)).

**When.** The instructions change more often than the code, or are owned by someone else
(a support lead's tone, a legal rubric).

**How.**

```python
agent = h.wrap(ReAct(system_prompt=stored("triage"), model=MODEL), id="triage")
rubric = await h.prompt("tone-rubric@4")  # that version, always: rubric.text, rubric.messages
```

`"name"` is the latest committed version, `"name@3"` that one. A `ReAct` whose prompt has no
`{{variables}}`, its model a name routed through the gateway, has every model call select it
(`x-bf-prompt-id`, `x-bf-prompt-version`): the gateway prepends the prompt's messages to the
request's own and applies its `model_params` where the request sets none. Any other is
rendered into the instructions.

**Automatic.** The name is resolved to the prompt's id once and kept fresh
(`REPOSITORY_TTL_SECONDS`, 300 s; the last answer stands while the gateway is down). A run
pins the version at its first read and journals it, so every call of the run — a resume on
another worker included — gets the same version, even after a new commit. The gateway's log
does not record which prompt a request selected, so every `chat` span says it:
`trellis.prompt.name`, `trellis.prompt.id`, `trellis.prompt.version`, and the run says it with a
`prompt` event.

**On failure.** A name the gateway does not have, a prompt with no committed version, a version
past the latest, or two prompts of one name: `ConfigurationError`, saying which — `stored(...)`
refuses a malformed reference (`"triage@"`) where it is written; a version that is not a number
(`"triage@latest"`) or a variable left unfilled fails the run. A prompt that cannot be read at a
run's first call (the gateway unreachable, never read before) fails the run with the gateway's
error.

## Who a call is for

**What.** Every MCP call of a run says who the run is for, so an MCP server that acts per
person acts for the right one.

**Automatic.** Each call carries:

* the trusted identity header, `x-trellis-identity: {"tenant_id": ..., "user_id": ...}` — the
  run's tenant and user, the header the A2A client sends too. The gateway forwards it to a
  server whose MCP client lists it in `allowed_extra_headers`, and to no other;
* `x-bf-mcp-session-id: <tenant>:<user>`, which keys the gateway's own per-user credentials
  (`per_user_headers`, `per_user_oauth`) when the request has no virtual key.

**How**, on the server's side: list the header on its client
(`MCPClientConfig(..., allowed_extra_headers=("x-trellis-identity",))`) and read it in the
tool. Only the gateway can reach the server, and only the harness sets the header, so it is
trusted the way the A2A server trusts it — put the server where nothing else can call it.

```python
@server.tool()
def my_orders(ctx: Context) -> list[dict]:
    user = json.loads(ctx.headers["x-trellis-identity"])["user_id"]
    return orders_of(user)
```

**Credentials the gateway holds per person.** The gateway keys them by the signed-in user, else
the virtual key, else the session id: under one shared virtual key every run is the same
"person". For a credential per person, give each person a virtual key and build their runs'
harness with it — `Harness(config=Settings.from_env().model_copy(update={"llm_api_key":
key_of(person)}))`, the key resolved by your application as it resolves a person's agent-runs
key ([onboarding.md](onboarding.md#4-optional-a-key-per-person-for-an-approvals-ui)) — or have
the server take a token from a header of its own that your gateway forwards.

**On failure.** A server that refuses the call (no identity, no credential yet: the gateway
answers with the URL where it is given) is an error result the model reads; nothing is retried
past a write's single attempt.

## Whose run a model call is, and its trace

Every model call the harness routes through the gateway — a model name's, a judge's, an OpenAI
Agents run's (its client reads them per request), the Claude CLI's — carries
Bifrost's dimension headers, which Bifrost puts on its logs, its OpenTelemetry spans and its
metrics: `x-bf-dim-tenant`, `x-bf-dim-agent`, `x-bf-dim-run` and `x-bf-dim-user` — a hash of the
user id, never the id itself (dimensions are not for personal data); a judge's also
`x-bf-dim-purpose: judge`. Cost per agent, run and tenant is then Bifrost's report. Each call
also carries the W3C `traceparent` of the span it is made in, which Bifrost's OTel plugin keeps:
the gateway's span of the call is in the run's trace, whatever the framework. The harness's
own `chat` spans stay where `ModelHooks` runs (a `ReAct`, a graph given it); for an
OpenAI Agents run its SDK's own tracing is left as you set it (the SDK exports to OpenAI unless
you turn it off: `agents.set_tracing_disabled(True)`), said once when such an agent is wrapped.

## Frameworks' own MCP clients

The gateway is an MCP server itself: `/mcp` (every tool the key allows), authenticated with the
virtual key (`Authorization: Bearer <key>`).

```python
# LangChain (langchain-mcp-adapters 0.3 needs mcp<2: an environment of its own)
client = MultiServerMCPClient(
    {
        "bifrost": {
            "transport": "streamable_http",
            "url": f"{GATEWAY}/mcp",
            "headers": {"Authorization": f"Bearer {VIRTUAL_KEY}"},
        }
    }
)
# OpenAI Agents SDK
server = MCPServerStreamableHttp(
    params={"url": f"{GATEWAY}/mcp", "headers": {"Authorization": f"Bearer {VIRTUAL_KEY}"}}
)
# Claude Agent SDK
options = ClaudeAgentOptions(
    mcp_servers={
        "bifrost": {
            "type": "http",
            "url": f"{GATEWAY}/mcp",
            "headers": {"Authorization": f"Bearer {VIRTUAL_KEY}"},
        }
    }
)
```

| | The framework's own MCP client | the harness's (the key's MCP tools every run has) |
|---|---|---|
| Who runs a call | the framework, straight to the gateway | the harness's bridge, then the gateway |
| Governance (run, announce, ask), approvals | none | every call |
| Journal (no repeat after a pause or a crash), timeouts, retries by side effects | none | every call |
| Records (memory's tool records, spans, events) | none | every call |
| Use it when | the agent is not wrapped and nothing about its calls needs to be governed or recorded | anything wrapped, and anything with side effects |

In a wrapped agent, use the harness's: every run has the key's MCP tools (a graph of yours
gets them with `HarnessTools`), and a framework's own MCP client there makes calls the harness
never sees.

## Reaching the gateway

The harness authenticates with the virtual key: `Authorization: Bearer` on completions, on
`/mcp` (the tool listing) and on tool execution. Browser clients (Claude Desktop, Cursor) can
use the gateway's OAuth 2.1 instead (`mcp_server_auth_mode: both`), which the harness does not
need.

The stored prompts, the MCP clients' Agent Mode lists and Code Mode's log are read through the
gateway's management API (`/api`) with the same key. A gateway with admin auth on closes `/api`
to virtual keys: there a prompt fails its run with the gateway's refusal, the Agent Mode lists
go unchecked and Code Mode's nested calls are not recorded.

Runnable: [examples/03_way2_gateway/mcp_and_model_headers.py](../examples/03_way2_gateway/mcp_and_model_headers.py)
(the gateway's MCP tools and `NO_GATEWAY_TOOLS` with no harness), and
[examples/06_scenarios/gateway_code_mode_governed_evals.py](../examples/06_scenarios/gateway_code_mode_governed_evals.py)
(Code Mode, an `approve_when` rule, evaluation).
