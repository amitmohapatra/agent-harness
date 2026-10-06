# Governance: `trellis.harness.governance` (Way 2)

Governance decides, for each tool call, whether it **runs**, runs and is **announced**, or
**asks** a person first. As a block it checks the tools of an agent you do not wrap: your
LangGraph tools, your OpenAI Agents `function_tool`s, the tools Claude calls through your
`can_use_tool`, any function. The decision is the same one a wrapped agent's calls get, read
from the same tool catalog, so an administrator's rule governs both.

How the decision is made (the risks, the catalog's `approve_when`, the catalog kept fresh,
failing closed) is in [governance.md](../governance.md). This page is the API you call.

## Install and set up

Governance ships in the harness distribution (`trellis-harness`, from source:
`pip install -e ../agent-harness`); it reads the catalog through the memory SDK
(`trellis.memory`) and needs no `Harness`.

```python
from trellis.harness.governance import Governance

gov = Governance.from_env(agent_id="procurement", tenant="acme")
...
await gov.aclose()
```

`Governance.from_env(*, agent_id=None, tenant=None, environ=None)` reads two variables:

| Environment | Governance |
|---|---|
| `MEMORY_URL` and `TRELLIS_API_KEY` | the memory service's tool catalog, in `tenant` (or the key's own tenant): risks and `approve_when` rules as an administrator set them |
| `MEMORY_URL` unset | no catalog: each tool's own side effects decide (`read` runs, `write` is announced, `irreversible` asks) |
| `MEMORY_URL` without a key | a `ConfigurationError` |

`agent_id` and `tenant` are whom `decided` attributes a person's decision to; everything else
works with the key's own tenant.

## The API

```python
from trellis.contracts import ToolSpec

# once at start-up: tell the catalog about your tools, so an administrator can govern them
await gov.publish([ToolSpec(name="create_po", side_effects="irreversible")])

decision = await gov.check("create_po", {"sku": "A-1", "qty": 50}, side_effects="irreversible")
if decision.asks:
    ...  # ask a person decision.question; run, edit or drop the call
elif decision.announces:
    ...  # tell whoever watches, then run it
else:
    ...  # run it
```

| Call | What it does |
|---|---|
| `await gov.check(tool, args, *, side_effects="write")` | the `Decision` for one call. `side_effects` is what the tool says it does (`read`, `write`, `irreversible`); the catalog's word overrides it |
| `governed(fn, gov, *, name=None, side_effects="write", timeout=None, on_ask, on_announce=None, hooks=(), approval=None)` | `fn` (sync or async) as an async callable that takes the tool's arguments as keywords and is checked at every call — and run as a harness tool call is: at most `timeout` seconds, retried when it only reads; `hooks` around it as around a harness tool call: a `before_tool` `Deny` raises `Denied`, a `Rewrite` changes the arguments, an `Ask` asks (`on_ask`); `after_tool` changes what it returns ([hooks.md](../hooks.md)); `approval` is the tool's approval function, as `tool(approval=)`'s: `fn(args)` returning `None` (governance decides), `True` (approved: `on_ask` is not called) or an `Ask` (`on_ask` is called; the `Decision` carries its `assignee`, `component` and `props`) |
| `await gov.publish(specs, *, annotations=None)` | tells the catalog about the tools (`ToolSpec`s; an MCP tool's annotations by name), once per content; a failed publish is sent again next time |
| `await gov.decided(decision, verdict, *, reviewer, run_id, user, edited=None)` | records what a person decided (`"approve"`, `"reject"` or `"edit"` with the `edited` arguments) as `TOOL_CALL` feedback, from which the memory service learns approval suggestions; idempotent per run and call; needs `agent_id` and `tenant` |
| `await gov.rules(names)` | the catalog's `Rule` (`risk`, `approve_when`) for each name, `None` where it has none |

A `Decision` has `tool`, `args`, `action` (`Action.RUN`, `ANNOUNCE`, `ASK`; also `.runs`,
`.announces`, `.asks`), `reason` (`"create_po is irreversible."`, or the rule that held),
`risk`, `rule`, and `question`, what an approver reads: `"Approve create_po? create_po is
irreversible."`.

### `governed`

`governed` wraps a tool function so you cannot forget the check:

```python
from trellis.harness.governance import Decision, Rejected, governed


async def create_po(sku: str, qty: int) -> str: ...  # your ERP call


async def ask(decision: Decision) -> bool | dict:
    return await my_approvals.ask(decision.question)  # however your framework asks a person


po = governed(create_po, gov, side_effects="irreversible", on_ask=ask)
try:
    await po(sku="A-1", qty=50)
except Rejected:
    ...  # the approver said no
```

* A call that **asks** runs `on_ask(decision)` (sync or async). `True` runs the call, `False`
  raises `Rejected`, a dict runs it with those (edited) arguments, and an exception
  propagates: LangGraph's `interrupt` is a natural `on_ask`, because the graph pauses there and
  its checkpointer keeps the pause.
* A call that is **announced** runs `on_announce(decision)` first, when given.
* Then it **runs** as a harness tool call does ([reliability.md](../reliability.md)): at most
  `timeout` seconds (a sync `fn` in a worker thread, which cannot be stopped); a call that only
  reads (as governance sees it) is tried again, up to twice, after an error that may pass, and
  one that does more runs once. Out of time it raises `ToolTimeout` (a `ToolError`, from
  `trellis.harness.tools.base`), whose message is what your model should read — for a call
  that does more than read (`unknown`), that it may or may not have taken effect. Hand your
  service an idempotency key of your own (your run's id and the call): Way 2 has no harness run
  to derive one from.
* The wrapped function keeps its signature, so a framework that builds tools from signatures
  (LangChain's `@tool`, OpenAI Agents' `function_tool`) builds the same tool from it.

## Where your framework asks

Governance decides; your framework pauses. Keeping the pause durable is your framework's job
(a checkpointer, a serialized state, a session) together with agent-runs ([runs.md](runs.md)):

| Framework | Where `check` goes | How it asks |
|---|---|---|
| LangGraph | `governed(fn, gov, on_ask=...)` around each tool | `on_ask` calls `interrupt(...)`; `Command(resume=...)` brings the answer back ([recipe](langgraph.md)) |
| OpenAI Agents SDK | `function_tool(fn, needs_approval=...)`, the callback returning `(await gov.check(...)).asks` | the SDK's own approval interruption; `RunState.approve` / `reject` ([recipe](openai-agents.md)) |
| Claude Agent SDK | `can_use_tool`, for every tool Claude calls (your MCP tools and Claude Code's built-ins) | deny with `interrupt=True`, resume the session after the answer ([recipe](claude-agent-sdk.md)) |
| plain code | `governed(fn, gov, on_ask=...)` or `check` | however you ask: a queue, a chat message, a paused run in agent-runs |

After the person answers, `gov.decided(...)` tells the memory service, which learns approval
suggestions from it; an accepted suggestion becomes an `approve_when` that governs your agent
and every wrapped one.

## Behaviour

* **Fails closed.** A rule that cannot be evaluated on a call asks. A catalog that cannot be
  read keeps the rules read in the last 300 s; past that every tool that does more than read
  asks, and a warning is logged once (logger `trellis.governance`). The catalog is asked again
  every 30 s.
* **Fresh.** The rules of the tools checked are read again every 30 s, conditionally (`ETag`),
  so an administrator's new rule reaches running agents within half a minute; a tool checked
  for the first time is read at once; concurrent checks share one read.
* **Tenancy.** One `Governance` reads one tenant's catalog: `tenant=`, else the key's.
* **No run access.** Governance never sees a run: pausing, journaling and recording the call
  are yours (Way 2) or the harness's bridge (Way 1).

## With Way 1

A wrapped agent's every tool call goes through the same `Governance.check` inside the harness
(`Harness.governance(tenant)`), which then pauses the run, announces the call or runs it
([governance.md](../governance.md#way-1-inside-hwrap)). One catalog serves both ways: a rule an
administrator sets on `create_po` governs your graph and every wrapped agent in the tenant
([mixing.md](mixing.md)).

Runnable: [`examples/blocks_langgraph.py`](../../examples/blocks_langgraph.py),
[`examples/blocks_openai_agents.py`](../../examples/blocks_openai_agents.py),
[`examples/blocks_claude.py`](../../examples/blocks_claude.py).
