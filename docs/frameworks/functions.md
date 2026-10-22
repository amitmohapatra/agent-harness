# Plain async functions

Code that decides itself what happens — a workflow, a router, glue between agents, a step with
no model — is a target as it is: `async def fn(input, agent) -> answer`. `agent` is the run's
`Runtime` (the same object `trellis.current()` returns). Nothing else to install. This is
Way 1: the harness runs the function. Plain code that calls the blocks itself, unwrapped, is
Way 2 ([docs/README.md](../README.md#way-2-pluggable-blocks-your-framework-our-pieces);
[blocks/runs.md](../blocks/runs.md) has a `Worker` with a handler of your own).

```python
from trellis import Harness, Runtime, tool


@tool(side_effects="irreversible")
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    ...


async def billing(input: dict, agent: Runtime) -> str:
    known = agent.context  # the memory context for the question (None: memory off)
    receipt = await agent.tools.call("refund", order=input["order"], amount=input["amount"])
    size = await agent.ask("Which size?", options=["S", "L"], assignee="role:ops")
    agent.log("asked", size=size)
    return f"{receipt}; size {size}"


h = Harness()
agent = h.wrap(billing, id="billing", tools=[refund])
result = await agent.run({"order": "o-7", "amount": 40, "question": "Refund o-7"}, user="ada")
```

## What `agent` gives the function

| | |
|---|---|
| `agent.context` | the pushed memory context (also a leading system message when the input is a message list) |
| `agent.memory` | the memory SDK's verbs in the run's scope (`search`, `remember`, `history`, documents…; needs `MEMORY_URL`) |
| `await agent.tools.call(name, **args)` | any tool of the run — yours, MCP, memory — through the bridge (governance, approvals, journal, records) |
| `await agent.tools.hints(task)` | the tools that fit a task, from the memory service |
| `await agent.ask(question, ...)` | a pause for a person ([interrupts.md](../interrupts.md)); returns the answer on resume |
| `agent.log(message, **fields)` | a log line and a `log` event on the stream |
| `run_id`, `user`, `thread`, `tenant`, `attempt`, `task` | who and what the run is |
| `agent.uses(feature)` | whether the run has a part of the harness on, or is `without=` it ([what is on](../configuration.md#what-is-on-and-how-to-turn-it-off)) |

Memory is asked about the input's text: the input itself, the last user message of a list, or a
dict's `query`/`question`/`input`/`prompt`/`text`/`message` field — a dict without one gets no
context (as above: add a `question`).

## Approvals and resumes

A tool that asks, or `agent.ask(...)`, pauses the run; `agent.resume(...)` calls the function
again from its input, and the journal returns the answers already given and the tool outputs
already recorded — in order, by content. So: keep side effects in harness tools (a call through
the bridge runs once), and code outside them runs again on each attempt.

## Native or ours

A function has no framework of its own to bring skills, prompts or a sandbox: use the
harness's — `h.wrap(fn, skills=[...])`, `await h.prompt(...)`, `tools=[sandbox()]` — every call
governed and journaled ([skills.md](../skills.md#native-or-ours),
[prompts.md](../prompts.md#native-or-ours),
[sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).

## Streaming, durable runs, surfaces, evaluation

`stream` yields the tool events and `RUN_FINISHED` (a function has no text deltas). `start` +
workers (the input must be JSON), `schedule`, `serve_chat`, `serve_a2a` and `h.evaluate` work as
for every target; `llm_judge` needs `TRELLIS_JUDGE_MODEL`. An object with an `async __call__`
is a target too. Wrapping a function is also how code on its own framework gets the AG-UI and
A2A servers, which serve wrapped agents only: wrap the function that calls your graph or runner
([surfaces.md](../surfaces.md#surfaces)).

`framework_options=` does not apply: a function has no framework run call to hand them to, so
they are refused (`ConfigurationError`) — the function reads what it needs from its input.

## Run it

* [`examples/01_start/hello.py`](../../examples/01_start/hello.py) — Level 0;
  [`examples/02_way1_function/agent.py`](../../examples/02_way1_function/agent.py) — memory, local and
  MCP tools, a question, a hook, `without=`.
* [`examples/02_way1_function/queue_worker_inbox.py`](../../examples/02_way1_function/queue_worker_inbox.py), [`examples/02_way1_function/schedule.py`](../../examples/02_way1_function/schedule.py),
  [`examples/05_features/serve_agui_and_a2a.py`](../../examples/05_features/serve_agui_and_a2a.py),
  [`examples/03_way2_a2a/remote.py`](../../examples/03_way2_a2a/remote.py),
  [`examples/02_way1_function/memory_documents_feedback.py`](../../examples/02_way1_function/memory_documents_feedback.py).
* Tests: `tests/integration/test_function_agent.py`, and against the real services
  `tests/live/test_live_matrix.py` (`function`, and the worker, schedule, A2A, document and
  feedback tests).
