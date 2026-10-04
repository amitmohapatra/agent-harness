# ReAct: the harness's own loop

For a team with no agent framework: `ReAct(system, model)` is a tool-calling loop over chat
completions — native tool messages, structured output, a `chat` span per model call. Nothing
else to install.

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

`ReAct(system, model, output=None, max_steps=12, *, max_result_chars=20000, max_repeats=3)`.
`model` is a Bifrost model name, sent to `BIFROST_URL` with the agent's virtual key, or any
object with `async complete(messages, **body) -> dict` (a chat-completions response) — your
own client, or a scripted model in a test.

## What is automatic

| | |
|---|---|
| Memory push | the context is appended to `system` |
| Memory pull | the memory tools are in every request's `tools` |
| Tool hints | **per model call**: each request carries the tools offered at that moment (the hinted ones, the memory tools, those already used, what a `tool_search` found) |
| Model steps | journaled: a resume — after a pause or a worker crash — replays the steps already taken instead of asking the model again |
| Bad arguments | arguments that are not a JSON object, or do not fit the tool's schema, are an error the model reads (the tool does not run) |
| Large results | a tool result over `max_result_chars` is cut with a marker; the whole result is kept as a run artifact |
| Stalls | the same call in `max_repeats` consecutive steps, or `max_steps` model calls, stops the run |
| Records, grounding, judges, tracing | as for every target; `chat` spans carry the model, usage and finish reasons |

## Approvals, streaming, durable runs, surfaces, evaluation

A tool that asks pauses the run; `agent.resume(...)` re-runs it from the journal — model steps
included, so the model is not asked again for what it already decided, and the approved call
runs once. `agent.stream(...)` yields each step's text and the tool events. `start` + workers,
`schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` tools and `h.evaluate` work as for every
target; `llm_judge` falls back to the agent's own model when `TRELLIS_JUDGE_MODEL` is unset (and
says so once — set a stronger model).

## Run it

* [`examples/react_agent.py`](../../examples/react_agent.py) — a tool, then a structured answer.
* Tests: `tests/integration/test_react.py`, and against the real services
  `tests/live/test_live_matrix.py` (`react`).
