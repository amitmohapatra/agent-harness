# ReAct: the harness's own loop

For a team with no agent framework: `ReAct(system, model)` is a tool-calling loop over chat
completions — native tool messages, structured output, a `chat` span per model call. Nothing
else to install. `ReAct` runs wrapped, with the blocks of the harness that wraps it — the
deployment's (`Harness()`), or your own: `Harness(runs=..., governance=..., memory=False)
.wrap(ReAct(...))`, its queued runs continued by your own scheduler with `agent.execute(job)`
([composition](../README.md#composition-a-harness-is-the-blocks-you-give-it),
[`examples/react_with_blocks.py`](../../examples/react_with_blocks.py)). A team with a loop of
its own that wants only some pieces uses the blocks instead
([docs/README.md](../README.md#way-2-pluggable-blocks-your-framework-our-pieces)).

```python
from pydantic import BaseModel
from trellis import Harness, ReAct, tool


class Forecast(BaseModel):
    city: str
    celsius: float
    advice: str


@tool(side_effects="read")
def temperature(city: str) -> float:
    """Today's temperature in a city."""
    ...


h = Harness()
target = ReAct(
    system="You give weather advice. Use the tool, then answer as JSON.",
    model="provider/model",  # a Bifrost model name (needs BIFROST_URL)
    output=Forecast,  # optional: the answer parsed into it
)
agent = h.wrap(target, id="weather", tools=[temperature])
result = await agent.run("Do I need a coat in Oslo?", user="ada")  # result.answer: a Forecast
```

`ReAct(system, model, output=None, max_steps=12, *, max_result_chars=20000, max_repeats=3,
model_timeout=None, context_window=None, prompt=None)`. `model` is a Bifrost model name, sent to
`BIFROST_URL` with the agent's virtual key, or any object with `async complete(messages, **body)
-> dict` (a chat-completions response) — your own client, or a scripted model in a test.
`model_timeout` is the most one model call may take, in seconds. `context_window` is the
model's window in tokens, only when the model object does not say (its `context_window`
attribute) and the 128k assumed is wrong for it. `prompt` names a stored prompt of the
gateway's Prompt Repository (`"triage"`, or `"triage@3"` for that version) the gateway
prepends to every model call — a model name only ([gateway.md](../gateway.md#prompts)).

## What is automatic

| | |
|---|---|
| Memory push | the context is appended to `system` |
| Memory pull | the memory tools are in every request's `tools` |
| Tool hints | **per model call**: each request carries the tools offered at that moment (the hinted ones, the memory tools, those already used, what a `tool_search` found) |
| Model steps | journaled: a resume — after a pause or a worker crash — replays the steps already taken instead of asking the model again |
| Several calls in one step | the calls whose tool only reads (`side_effects="read"`, or idempotent) run at once; the others one at a time after them, in the model's order; the tool messages follow the model's order, one per call. Steps and idempotency keys are given in the model's order before anything runs, so a resume replays each call's own output whatever order they finished in. A call that pauses lets the calls beside it finish (journaled) before the run pauses; cancelling the run cancels them ([reliability.md](../reliability.md#calls-made-at-once)) |
| Bad arguments | arguments that are not a JSON object, or do not fit the tool's schema, are an error the model reads (the tool does not run) |
| Large results | a tool result over `max_result_chars` keeps its head and its tail, with a marker in the middle naming the run artifact the whole result is kept in, and `read_result(id, offset, limit)` — offered from then on — to read the rest from what the run already has (no tool runs again) |
| The context window | the model's window: `context_window=`, else the model object's `context_window`, else 128k tokens; estimated as 4 characters a token. Past half of it, the older tool results (all but the last 3) are replaced by a placeholder naming `read_result`; past three quarters, the older turns are compacted into one summary (the task, the state, the decisions, what failed, the next steps) by one model call, keeping the system message, the first user message, the summary and the recent turns (a quarter of the window), cut between turns. Either happens only when it frees a tenth of the window, so the cached prompt breaks rarely. What was decided at each step, and the summary, are journaled: a resume reads exactly what the model read, however the memory context changed |
| A stable prompt | the tools are sorted by name and the set offered only grows within a run; earlier messages change only at a clearing or a compaction, so a provider's prompt cache keeps working |
| The step limit | after `max_steps` model calls the model is asked once more, with the same tools and `tool_choice: "none"`, for its best answer with what it has; a `warning` event (`max_steps`) and a log line say the run stopped there. No answer then fails the run |
| Stalls and failures | the same call in `max_repeats` consecutive steps stops the run; so do 3 consecutive steps in which every call failed (no such tool, arguments that do not fit, an error or a timeout: `ERROR_STREAK`) — a `ModelError` saying so |
| Sub-agents | another wrapped agent in `tools=[agent.as_tool()]` is a call like any other: a child run that answers, pauses this run with its question, or is continued after a crash ([subagents.md](../subagents.md)) |
| Slow models | a model call takes at most `model_timeout` (and what is left of the run's `timeout=`/`deadline=`), the gateway's own retries inside it; past it the run fails with a `ModelError` that may be retried (a queued run is queued again; its journaled steps are not asked again) — [reliability.md](../reliability.md#model-timeouts) |
| Stored prompt | with `prompt=`: resolved once (kept fresh), its version pinned at the run's first model call and journaled, so every call of the run — a resume included — selects the same version; each `chat` span says which (`trellis.prompt.*`) |
| Skills | with `h.wrap(..., skills=[...])`: their names and descriptions appended to `system` with the memory context, `load_skill` and `read_skill_file` in every request's `tools` ([gateway.md](../gateway.md#skills)) |
| Hooks | `before_model`/`after_model` around every model call of the loop (a `before_model` call is the call sent), the tool and run hooks as for every target ([hooks.md](../hooks.md)) |
| Records, grounding, judges, tracing | as for every target; `chat` spans carry the model, usage and finish reasons |

## Approvals, streaming, durable runs, surfaces, evaluation

A tool that asks pauses the run; `agent.resume(...)` re-runs it from the journal — model steps
included, so the model is not asked again for what it already decided, and the approved call
runs once. `agent.stream(...)` yields each step's text and the tool events. `start` + workers,
`schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` tools and `h.evaluate` work as for every
target; `llm_judge` falls back to the agent's own model when `TRELLIS_JUDGE_MODEL` is unset (and
says so once — set a stronger model).

`framework_options=` does not apply: the harness runs ReAct's loop, so they are refused
(`ConfigurationError`); its model's settings are given on `ReAct(...)`.

## Run it

* [`examples/react_agent.py`](../../examples/react_agent.py) — a tool, then a structured answer.
* [`examples/react_with_blocks.py`](../../examples/react_with_blocks.py) — your run store, your
  scheduler loop, your governance, memory off.
* [`examples/react_subagents.py`](../../examples/react_subagents.py) — two sub-agents at once,
  one asking a person.
* Tests: `tests/integration/test_react.py`, `test_react_parallel.py`, `test_react_context.py`,
  and against the real services `tests/live/test_live_matrix.py` (`react`) and
  `tests/live/test_live_subagents.py`.
