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
    receipt = await agent.tools.call(refund, order=input["order"], amount=input["amount"])
    size = await agent.ask("Which size?", options=["S", "L"], assignee="role:ops")
    agent.log("asked", size=size)
    return f"{receipt}; size {size}"


h = Harness()
agent = h.wrap(billing, id="billing")
result = await agent.run({"order": "o-7", "amount": 40, "question": "Refund o-7"}, user="ada")
```

## What `agent` gives the function

| | |
|---|---|
| `agent.context` | the pushed memory context (also a leading system message when the input is a message list) |
| `agent.memory` | the memory SDK's verbs in the run's scope (`search`, `remember`, `history`, documents…; needs `MEMORY_URL`) |
| `await agent.tools.call(tool, **args)` | a tool through the bridge (governance, approvals, journal, records): one of yours itself (`tool(fn)`, `mcp(...)`, `a2a(url)`, `sandbox()`...: it joins the run), or one of the run by name — the MCP tools the virtual key allows, memory's |
| `await agent.tools.use(*sources)` | your tools added to the run (the names they add), to call by name |
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
already recorded — in order, by content. So: keep side effects in tool calls (a call through
the bridge runs once), and code outside them runs again on each attempt. A function's tools
are its code's: as another agent's sub-agent its calls count as writes (they run one at a time).

## Sub-agents and child runs

A function delegates by calling a wrapped agent: `await research.run(question,
user=agent.user)` inside the run is a child run of it — a record naming the run, its tenant,
thread and deadline and what is left of its time, its journal kept in the run's (a crash
continues it), its question pausing the run (the run's resume answers it), cancelled with it
([subagents.md](../subagents.md#automatic)). `agent.start(...)` there queues
a child for a worker instead. The other way round, the wrapped function is a `ReAct`'s
sub-agent like any agent (`subagents=[agent]`): its docstring describes the tool, and its calls
are its code's, so the tool counts as a write (one at a time). Runnable:
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | `await agent.tools.use(sandbox().tools())`, then `agent.tools.call("sandbox_exec", command=...)` ([sandbox.md](../sandbox.md)) |
| Skills | none of its own: `await agent.tools.use(src.tools())`, then `list_skills` and `load_skill(name, path=None)` ([skills.md](../skills.md)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks ([hooks.md](../hooks.md)); a model call is your code's |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)) (its harness tool calls; your model's tokens are not seen); no framework turn limit (`framework_options=` is refused) |

## Native or ours

A function has no framework of its own to bring prompts or a sandbox: use the harness's —
`await h.prompt(...)`, `agent.tools.call(sandbox(), ...)` (its sandbox the run's, paused and
deleted with it) — every call governed and journaled ([prompts.md](../prompts.md#native-or-ours),
[sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).

## Streaming, durable runs, surfaces, evaluation

`stream` yields the tool events and `RUN_FINISHED` (a function has no text deltas). `start` +
workers (the input must be JSON), `schedule`, `serve_chat`, `serve_a2a` and `h.evaluate` work as
for every target; `llm_judge` asks the gateway's judge alias (or its own `model=`). An object with an `async __call__`
is a target too. Wrapping a function is also how code on its own framework gets the AG-UI and
A2A servers, which serve wrapped agents only: wrap the function that calls your graph or runner
([surfaces.md](../surfaces.md#surfaces)).

`framework_options=` does not apply: a function has no framework run call to hand them to, so
they are refused (`ConfigurationError`) — the function reads what it needs from its input.

## Conversations

With memory off, a run of a conversation (`thread=`) reads its earlier turns as
`agent.history` (each earlier run's question and answer: [runs.md](../runs.md#earlier-turns-with-memory-off)); with memory on,
memory's window is in `agent.context`. A function has no step the harness sees, so
`mode="steer"` waits as `"queue"` (said once); `"interrupt"` cancels the working run
([runs.md](../runs.md#a-message-to-a-busy-conversation-mode)).

## Limits

* A resume calls the function again from its input: code outside tool calls and `ask` runs
  again on every attempt (the journal answers the calls and questions already made, in order),
  so keep side effects in tool calls.
* The memory context reaches the function only through what it reads: `agent.context`, or an
  input that is a list of messages (the context as its leading system message); a run whose
  code reads neither is said once in the log.
* `stream` has no text deltas (a function has no model); `framework_options=` is refused.
* A queued or scheduled run's input must be JSON.

## Run it

* [`examples/01_start/hello.py`](../../examples/01_start/hello.py) — Level 0;
  [`examples/02_way1_function/agent.py`](../../examples/02_way1_function/agent.py) — memory, local and
  MCP tools, a question, a hook, `without=`.
* [`examples/02_way1_function/queue_worker_inbox.py`](../../examples/02_way1_function/queue_worker_inbox.py), [`examples/02_way1_function/schedule.py`](../../examples/02_way1_function/schedule.py),
  [`examples/06_scenarios/runs_every_framework.py`](../../examples/06_scenarios/runs_every_framework.py),
  [`examples/05_features/serve_agui_and_a2a.py`](../../examples/05_features/serve_agui_and_a2a.py),
  [`examples/03_way2_a2a/remote.py`](../../examples/03_way2_a2a/remote.py),
  [`examples/02_way1_function/memory_documents_feedback.py`](../../examples/02_way1_function/memory_documents_feedback.py).
* Tests: `tests/integration/test_function_agent.py`, and against the real services
  `tests/live/test_live_matrix.py` (`function`, and the worker, schedule, A2A, document and
  feedback tests).
