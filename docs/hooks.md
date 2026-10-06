# Hooks

**What.** Your code around a run, its model calls and its tool calls: a guardrail that denies
or rewrites a call or asks a person about it, redaction of what reaches the model, an audit
trail, your own metrics. Subclass `Hooks` (`from trellis import Hooks`) and override what you
need; every method does nothing by default.

| Method | When | It may |
|---|---|---|
| `on_run_start(run)` | each attempt of a run, after its tools and context are ready, before the framework runs | raise: the attempt fails (`ERROR`) |
| `on_run_end(run, result)` | each attempt, once its outcome is recorded (`SUCCESS`, `PAUSED`, `ERROR`, `TIMEOUT`, `CANCELLED`) | observe (what it raises is logged) |
| `before_model(call) -> ModelCall \| None` | before each model call | return a changed call (its `messages`, its `system`): the call made instead, where the framework allows it |
| `after_model(call, reply)` | after each model call, with the framework's own reply | observe |
| `before_tool(call) -> None \| Deny \| Ask \| Rewrite` | before each harness tool call, before governance | `Deny(reason)`: not run, the model reads `"<tool> was not run: <reason>"`; `Ask(question, assignee=None)`: a person approves it first, as a governance approval (approve, edit, reject, cancel); `Rewrite(args)`: run with these arguments (governance decides on them; the next hook sees them) |
| `after_tool(call, outcome) -> outcome` | after each harness tool call ran | return another outcome: what the model reads, journaled and recorded |
| `on_error(stage, error)` | a run that failed or ran out of time (`"run"`), a model call that failed (`"model"`), a tool call that failed or timed out (`"tool"`) | observe (what it raises is logged) |

**When.** For the rules only your code knows. What the deployment already decides needs no
hook: a tool's risk and the catalog's `approve_when` (governance), redaction of the stream,
the spans and the memory records (automatic), time limits (`timeout=`).

**Where.** `Harness(hooks=[...])` for every agent of the harness; `h.wrap(..., hooks=[...])`
for one agent, after the harness's. Way 2: `governed(fn, gov, ..., hooks=[...])` for your own
tools, and each framework's own mechanism (below) outside a harness run.

| | Tool hooks | Run hooks | Model hooks |
|---|---|---|---|
| `ReAct` | the bridge | the pipeline | its own loop: every model call (a compaction too); a `before_model` call is the call sent |
| plain function | the bridge | the pipeline | none: it makes no model call |
| LangGraph, `create_agent` | the bridge | the pipeline | LangChain's own middleware, given when the graph is built: `create_agent(model, tools=..., middleware=[ModelHooks()])` (`trellis.harness.hooks.langchain`); a `before_model` call is the request made. A hand-built `StateGraph` calls its model itself: none |
| Deep Agents | the bridge | the pipeline | the same middleware: `create_deep_agent(..., middleware=[ModelHooks()])` |
| OpenAI Agents SDK | the bridge | the pipeline | the SDK's own `RunHooks`, which the harness passes to `Runner.run` (`trellis.harness.hooks.openai_agents.ModelHooks`): every call reported, none rewritten (the SDK takes nothing back) |
| Claude Agent SDK | the bridge (its harness tools); a built-in tool (`Bash`, `Write`...) that asks for permission: `before_tool` with governance, in the permission callback ([frameworks/claude-agent-sdk.md](frameworks/claude-agent-sdk.md)) | the pipeline | none: the CLI makes the model calls, and its hooks have no model-call event |

**How.**

```python
from trellis import Ask, Deny, Harness, Hooks, ModelCall, Rewrite
from trellis.contracts import ToolCall, ToolOutcome


class Refunds(Hooks):
    """A guardrail: no refund over 1000, a person for one over 100, amounts in cents."""

    async def before_tool(self, call: ToolCall) -> Deny | Ask | Rewrite | None:
        if call.tool != "refund":
            return None
        amount = call.args["amount"]
        if amount > 1000:
            return Deny("refunds over 1000 go through finance")
        if amount > 100:
            return Ask(f"Refund {amount}?", assignee="role:support-lead")
        return Rewrite({**call.args, "cents": round(amount * 100)})


class Cards(Hooks):
    """Redaction of your own: card numbers never reach the model."""

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome.model_copy(update={"output": mask(outcome.output)})

    async def before_model(self, call: ModelCall) -> ModelCall:
        return dataclasses.replace(call, messages=[mask(m) for m in call.messages])


h = Harness(hooks=[Cards()])
agent = h.wrap(target, id="support", hooks=[Refunds()])
```

`call.messages` is in the framework's own form: chat-completions dicts for `ReAct`
(`call.framework == "react"`), LangChain messages for a graph (`"langgraph"`, the system
message in `call.system`), Responses input items for the OpenAI Agents SDK (`"openai_agents"`,
the instructions in `call.system`).

**Automatic.** Hooks run in order: the harness's, then the agent's; a `Rewrite` is the next
hook's call, and the first `Deny` or `Ask` decides (the hooks after it are not asked). A
`before_tool` decision is journaled with the call: a resumed run — after a pause or a crash —
reads it instead of asking the hooks again, so an `Ask` approved once is not asked again, and a
`Rewrite` holds. A call the journal already has (its output replayed) runs no hook; nor does a
model step a resumed `ReAct` replays. The outcome `after_tool` returns is what is journaled,
recorded in memory and on the stream.

**On failure.** A hook that raises fails what it hooks: in `before_tool`/`after_tool` the call
(the framework sees the error), in `before_model`/`after_model` the model call, in
`on_run_start` the attempt (`ERROR`). `on_run_end` and `on_error` run once the outcome is
decided: what they raise is logged (`trellis.hooks`), and the run's outcome stands.

**Example.** [`examples/hooks.py`](../examples/hooks.py): a guardrail (deny, ask, rewrite)
and redaction of card numbers on a `ReAct` agent. Tests:
`tests/integration/test_hooks.py` (every adapter, Way 2), and against the real services
`tests/live/test_live_hooks.py`.
