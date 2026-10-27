# Hooks

**What.** Your code around a run and its tool calls, the same on every engine: a guardrail that
denies or rewrites a call or asks a person about it, redaction of what a tool returns and of
what your users see, an audit trail, your own metrics. Subclass `Hooks` (`from trellis import Hooks`) and override what you
need; every method does nothing by default.

| Method | When | It may |
|---|---|---|
| `on_run_start(run)` | each attempt of a run, after its tools and context are ready, before the framework runs | raise: the attempt fails (`ERROR`) |
| `on_run_end(run, result)` | each attempt, once its outcome is recorded (`SUCCESS`, `PAUSED`, `ERROR`, `TIMEOUT`, `CANCELLED`) | observe (what it raises is logged) |
| `before_tool(call) -> None \| Deny \| Ask \| Rewrite` | before each harness tool call, before governance | `Deny(reason)`: not run, the model reads `"<tool> was not run: <reason>"`; `Ask(question, assignee=None, component=None, props=None)`: a person approves it first, as a governance approval (approve, edit, reject, cancel), on your own screen when `component` names one; `Rewrite(args)`: run with these arguments (governance decides on them; the next hook sees them) |
| `after_tool(call, outcome) -> outcome` | after each harness tool call ran | return another outcome: what the model reads, journaled and recorded |
| `on_error(stage, error)` | a run that failed or ran out of time (`"run"`), a model call that failed (`"model"`), a tool call that failed or timed out (`"tool"`) | observe (what it raises is logged) |
| `on_output(item) -> item` (synchronous) | what leaves for the user: the answer `agent.run`, `agent.resume` and `RunHandle.result` return, every event of `agent.stream`, `agent.events` and AG-UI | return another item: what the user is sent ([below](#mask-what-the-ui-sees)) |

**When.** For custom logic a rule cannot express. **When a call needs a person is never a
hook's**: approval rules live in one place, the tool catalog — a tool's `risk` and its
`approve_when` (`amount > 100`), set by an administrator or a deploy script, read by every
framework and process ([governance.md](governance.md#one-place-for-approval-rules)); with
memory off, the tools' own declared risk decides. A hook is for what such a rule cannot say: a
refusal (`Deny`), a rewrite (`Rewrite`), a decision that needs outside data (an account frozen
in your database), who reviews and on which screen (`Ask(assignee=, component=)`). Credentials
in traces are masked for you ([observability.md](observability.md#redaction)) and time limits
(`run_timeout=`, `call_timeout=`) need no hook either; personal data is yours to define and mask, in these hooks.

**Where.** `h.wrap(..., hooks=[...])` for an agent's every run, in the order given. Way 2: `governed(fn, gov, ..., hooks=[...])` for your own
tools.

**Model calls are the framework's own.** `Hooks` has no model hook: each framework has its own
callbacks for a model call, and they work under `h.wrap` as they do without it. A hook class
that still defines `before_model` or `after_model` is refused where it is given
(`ConfigurationError: Cards.before_model was removed: ...`), never silently skipped.

| | Tool hooks | Run hooks | A model call: the framework's own |
|---|---|---|---|
| `ReAct` | the bridge | the pipeline | LangChain middleware: `ReAct(middleware=[...])` (yours run before the harness's `ModelHooks`, so its span shows what the model is sent) |
| plain function | the bridge | the pipeline | none: your code calls its model |
| LangGraph, `create_agent` | the bridge | the pipeline | LangChain middleware: `create_agent(model, tools=..., middleware=[yours, ModelHooks()])`; a hand-built `StateGraph` calls its model itself |
| Deep Agents | the bridge | the pipeline | the same: `create_deep_agent(..., middleware=[yours])` |
| OpenAI Agents SDK | the bridge | the pipeline | the agent's own `AgentHooks` (`Agent(hooks=...)`: `on_llm_start`, `on_llm_end`), reported, not rewritten |
| Claude Agent SDK | the bridge (its harness tools); a built-in tool (`Bash`, `Write`...) that asks for permission: `before_tool` with governance, in the permission callback ([frameworks/claude-agent-sdk.md](frameworks/claude-agent-sdk.md)) | the pipeline | the CLI makes the model calls: its own hooks (`ClaudeAgentOptions(hooks=...)`), which have no model-call event |
| Kimi Code | the bridge (its harness tools); a call of Kimi's own it asks permission for (`Bash`, `Write`...): `before_tool` with governance, in ACP's `session/request_permission` (a `Rewrite` ends the turn and Kimi calls again with the new arguments: [frameworks/kimi.md](frameworks/kimi.md#approvals-and-pauses)) | the pipeline | Kimi makes the model calls: its own hooks (`hooks` in its `config.toml`); ACP shows no model call |

`on_error("model", ...)` still sees a model call that failed, where the harness runs the call
(`ReAct`, a graph with `ModelHooks`).

**How.**

```python
from trellis import Deny, Harness, Hooks, ReAct, Rewrite
from trellis.contracts import ToolCall, ToolOutcome


class Refunds(Hooks):
    """A guardrail: no refund over 1000, amounts in cents. When a refund needs a person is the
    catalog's rule (approve_when: amount > 100), not this hook's."""

    async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
        if call.tool != "refund":
            return None
        amount = call.args["amount"]
        if amount > 1000:
            return Deny("refunds over 1000 go through finance")
        return Rewrite({**call.args, "cents": round(amount * 100)})


class Cards(Hooks):
    """Redaction of your own: no card number in what a tool returns."""

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome.model_copy(update={"output": mask(outcome.output)})


h = Harness()
# a model call is LangChain's own: MaskCards, a LangChain AgentMiddleware of yours (its
# awrap_model_call masks what the model is sent: examples/02_way1_react/hooks_guardrails.py)
target = ReAct("You handle refunds.", model, tools=[refund], middleware=[MaskCards()])
agent = h.wrap(target, id="support", hooks=[Cards(), Refunds()])
```

**Automatic.** Hooks run in the order given; a `Rewrite` is the next
hook's call, and the first `Deny` or `Ask` decides (the hooks after it are not asked). A
`before_tool` decision is journaled with the call: a resumed run — after a pause or a crash —
reads it instead of asking the hooks again, so an `Ask` approved once is not asked again, and a
`Rewrite` holds. A call the journal already has (its output replayed) runs no hook; nor does a
model step a resumed graph does not take again (its checkpoint has it). The outcome `after_tool` returns is what is journaled,
recorded in memory and on the stream.

**On failure.** A hook that raises fails what it hooks: in `before_tool`/`after_tool` the call
(the framework sees the error), in `on_run_start` the attempt (`ERROR`). `on_run_end` and `on_error` run once the outcome is
decided: what they raise is logged (`trellis.hooks`), and the run's outcome stands. An
`on_output` that raises is logged and fails closed: the user is sent the item withheld
(`trellis.harness.hooks.WITHHELD` in place of the answer, an event of the same type with its
data withheld), never as it was.

## Mask what the UI sees

The harness masks credentials in traces and nothing else: the run's events, its answer,
memory, A2A and the journal carry values as they are, because what is personal is your
domain's to say. To mask what your users see, override `on_output`. It is given each thing
leaving for the user — `Output(OutputKind.ANSWER, answer)` for the answer `agent.run`,
`agent.resume` and `RunHandle.result` return, `Output(OutputKind.EVENT, event)` for each event
of `agent.stream`, `agent.events` and `serve_chat`'s AG-UI stream (text deltas, tool calls,
custom events, the finish with its `result`) — and what it returns is what is sent:

```python
import json
from re import compile, sub

from trellis import Harness, Hooks, Output, OutputKind

CARD = compile(r"\b(?:\d[ -]?){13,16}\b")


class MaskCards(Hooks):
    def on_output(self, item: Output) -> Output:
        if item.kind is OutputKind.ANSWER:
            return item.replace(sub(CARD, "[card]", str(item.value)))
        event = item.value
        data = json.loads(sub(CARD, "[card]", json.dumps(event.data)))
        return item.replace(event.model_copy(update={"data": data}))


agent = h.wrap(target, id="support", hooks=[MaskCards()])
```

It runs in the order the hooks were given, synchronously, on every streamed
delta: keep it fast, no I/O. With no hook overriding it nothing sent is wrapped. It never
touches memory, the journal, A2A, the run's record, traces, or what the model and the tools
see — `after_tool` and the framework's own model-call callbacks are for those, as `Cards`
and `MaskCards` above.

**Example.** [`examples/02_way1_react/hooks_guardrails.py`](../examples/02_way1_react/hooks_guardrails.py): a guardrail (deny, ask, rewrite)
and redaction of card numbers on a `ReAct` agent. Tests:
`tests/integration/test_hooks.py` (every adapter, the frameworks' own model-call callbacks under
`h.wrap`, a removed model hook refused, Way 2), and against the real services
`tests/live/test_live_hooks.py`.
