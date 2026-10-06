# The Bifrost gateway

The harness reaches models and MCP tools through Bifrost (`BIFROST_URL`, `BIFROST_VIRTUAL_KEY`):
the tools a virtual key allows are the agent's ([tools.md](tools.md)), and `ReAct` and the LLM
judge call models by name. This page is the rest of what the harness uses of the gateway —
stored prompts, skills, Virtual MCPs, who a tool call is for — and what it makes sure the
gateway never does for a run. Nothing here is configured beyond the two variables: every
feature is off until an agent names it (`prompt=`, `skills=`, `mcp=`).

| You want | Use | Section |
|---|---|---|
| a system prompt kept, versioned and changed in the gateway, not in code | `ReAct(..., prompt="triage")`, `llm_judge(..., prompt=)`, `h.model_headers(prompt=)` | [Prompts](#prompts) |
| instructions the model reads only when a task needs them | `h.wrap(..., skills=["sql-review"])`, `skills(...)` | [Skills](#skills) |
| an agent limited to some tools of some MCP servers | `h.wrap(..., mcp=["finance-tools"])` | [Virtual MCPs](#virtual-mcps) |
| an MCP server acting for the person a run is for | nothing: every call says who | [Who a call is for](#who-a-call-is-for) |
| your framework's own MCP client on the gateway | its client at `/mcp` with the key — ungoverned | [Frameworks' own MCP clients](#frameworks-own-mcp-clients) |

## What the gateway never does for a run

**Automatic.** Every model call the harness makes (a `ReAct` built with a model name, the
judge) sends the gateway's deny-all MCP scope (`x-bf-mcp-include-clients` and
`x-bf-mcp-include-tools`, empty): the gateway adds none of the key's MCP tools to the request
and runs none itself (Agent Mode). A framework's own model client pointed at the gateway gets
the same headers from `await h.model_headers()`:

```python
model = ChatOpenAI(
    base_url=BIFROST_URL, api_key=VIRTUAL_KEY, model=MODEL, default_headers=await h.model_headers()
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

**What.** A stored prompt of the gateway's Prompt Repository: messages the gateway prepends to
a model call, committed in versions (1, 2, 3, ...) by whoever owns the prompt, not by a deploy.
The gateway is one prompt source among others — code, `.md` files, Langfuse — asked last
([prompts.md](prompts.md) has them all, and the order): a name the others do not have is the
gateway's.

**When.** The instructions change more often than the code, or are owned by someone else
(a support lead's tone, a legal rubric for the judge).

**How.**

```python
agent = h.wrap(ReAct(system="You answer tickets.", model=MODEL, prompt="triage"), id="triage")
judge = llm_judge("Follows the tone guide.", prompt="tone-rubric@4")  # that version, always
```

`"name"` is the latest committed version, `"name@3"` that one. The gateway prepends the
prompt's messages to the request's own (the `ReAct`'s `system` and the conversation) and
applies its `model_params` where the request sets none.

**Automatic.** The name is resolved to the prompt's id once and kept fresh
(`REPOSITORY_TTL_SECONDS`, 300 s; the last answer stands while the gateway is down). A run
pins the version at its first model call and journals it, so every call of the run — a resume
on another worker included — selects the same version, even after a new commit. The gateway's
log does not record which prompt a request selected, so every `chat` span says it:
`trellis.prompt.name`, `trellis.prompt.id`, `trellis.prompt.version`.

**Where.** `ReAct` and `llm_judge` call models themselves. Any other framework calls them
through its own model client: `await h.model_headers(prompt="triage")` gives that client the
prompt's headers (`x-bf-prompt-id`, `x-bf-prompt-version`) with the deny-all scope — the
version resolved then, for the client's life, not per run.

**On failure.** A name the gateway does not have, a prompt with no committed version, a version
past the latest, or two prompts of one name: `ConfigurationError`, saying which — `ReAct` and
`llm_judge` refuse a malformed reference (`"triage@"`) when they are built; a version that is
not a number (`"triage@latest"`), a model object or `prompt_vars=` with a gateway prompt (only a
model name's calls go through the gateway, which prepends the prompt as stored) fail the run.
A prompt that cannot be read at a run's first call (the gateway unreachable, never read before)
fails the run with the gateway's error.

## Skills

**What.** [Agent Skills](https://agentskills.io) from the gateway's Skills Repository: a
`SKILL.md` body of instructions and the files it refers to, published in immutable SemVer
versions, one of them served. The gateway is one skill source among others — code, `SKILL.md`
folders — asked last; skills of every source mix in one run ([skills.md](skills.md)).

**When.** Instructions for kinds of task that most runs do not need: loading them all into
every prompt costs tokens and attention; listing them and loading one when it fits does not.

**How.**

```python
agent = h.wrap(target, id="analyst", skills=["sql-review", "refunds@1.2.0"])
# a graph binds its tools when it is built: the same through h.tools
graph = create_agent(model, tools=await h.tools(skills("sql-review"), framework="langgraph"))
```

`"name"` is the version served when a run starts; `"name@1.2.0"` that version.

**Automatic**, for every framework (progressive disclosure):

* at the start of a run each skill is pinned and journaled (a resumed run keeps the versions it
  started with), and the run says which versions it used: a `skills` event
  (`{"versions": {"sql-review": "1.3.0", ...}}`) and the `trellis.skills` attribute of its
  span;
* the context pushed into the framework's input gets a `## Skills` section, after the memory
  context: each skill's name and description, and how to read one;
* two read-only tools, always offered: `load_skill(name)` — the pinned version's `SKILL.md`
  body and its file list — and `read_skill_file(name, path)`, one file. Both are harness tools:
  every call goes through the bridge and is journaled, so a resumed run reads what it read.

**On failure.** The gateway serves a file only of the version it serves now: reading a file of
another pinned version — `"refunds@1.2.0"` while 1.3.0 is served, or a run that pinned 1.2.0
before a rollout — is an error the model reads, saying both versions; the body still loads. A
skill the gateway does not have, or cannot be reached for and never was, is a
`skills_unavailable` warning event and the run goes on without it (the last copy read stands
while the gateway is down). Skills named with no skill source at all (no `BIFROST_URL`, no
`SKILLS_DIR`, none passed) fail the run with a `ConfigurationError`.

## Virtual MCPs

**What.** A Virtual MCP is a bundle of tools from several MCP servers behind one endpoint of the
gateway (`/mcp/<slug>`), attached to virtual keys.

**When.** One virtual key, several agents that should each see only some of its tools: a
finance agent that sees the payment tools and not the CRM's.

**How.** `h.wrap(target, id="payer", mcp=["finance-tools"])` (a graph: `h.tools(...,
framework="langgraph", mcp=[...])`). Without `mcp=` the agent has every tool its key allows,
as before; with it, the tools of those Virtual MCPs (a tool in two of them is listed once, from
the first), each listed and run through its bundle. The key must be attached to each one
(`admin.virtual_mcps.attach`, bifrost-sdk).

**Automatic.** Governance, approvals, the journal and the records are the same as for any MCP
tool. Code Mode is not used through a Virtual MCP: a script reaches every tool of a server, a
bundle only some.

**On failure.** A slug the key is not attached to fails the listing (`PermissionDeniedError`)
and so the run, until the toolbox can list it; after a first good listing the last one stands.

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
harness with it — `Harness(config=Settings.from_env().model_copy(update={"bifrost_virtual_key":
key_of(person)}))`, the key resolved by your application as it resolves a person's agent-runs
key ([onboarding.md](onboarding.md#4-optional-a-key-per-person-for-an-approvals-ui)) — or have
the server take a token from a header of its own that your gateway forwards.

**On failure.** A server that refuses the call (no identity, no credential yet: the gateway
answers with the URL where it is given) is an error result the model reads; nothing is retried
past a write's single attempt.

## Frameworks' own MCP clients

The gateway is an MCP server itself: `/mcp` (every tool the key allows) and `/mcp/<slug>` (one
Virtual MCP), authenticated with the virtual key (`Authorization: Bearer <key>`).

```python
# LangChain (langchain-mcp-adapters)
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

| | The framework's own MCP client | `h.tools(..., framework=...)` / `tools=` |
|---|---|---|
| Who runs a call | the framework, straight to the gateway | the harness's bridge, then the gateway |
| Governance (run, announce, ask), approvals | none | every call |
| Journal (no repeat after a pause or a crash), timeouts, retries by side effects | none | every call |
| Records (memory's tool records, spans, events) | none | every call |
| Use it when | the agent is not wrapped and nothing about its calls needs to be governed or recorded | anything wrapped, and anything with side effects |

In a wrapped agent, use `h.tools`: a framework's own MCP client there makes calls the harness
never sees.

## Reaching the gateway

The harness authenticates with the virtual key: `Authorization: Bearer` on completions, on
`/mcp` (the tool listing) and on tool execution. Browser clients (Claude Desktop, Cursor) can
use the gateway's OAuth 2.1 instead (`mcp_server_auth_mode: both`), which the harness does not
need.

The stored prompts, the skills, the MCP clients' Agent Mode lists and Code Mode's log are read
through the gateway's management API (`/api`) with the same key. A gateway with admin auth on
closes `/api` to virtual keys: there a prompt fails its run with the gateway's refusal, skills
are a warning, the Agent Mode lists go unchecked and Code Mode's nested calls are not recorded.
