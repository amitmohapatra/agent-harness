# OpenAI Agents SDK

An `agents.Agent` is a target. The harness runs it with the SDK's own `Runner`, on a copy of
your agent that also carries the harness tools; your agent object is never changed.

**Install:** `pip install 'trellis-harness[openai-agents]'`. Your model is your own:
`OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(base_url=BIFROST_URL,
default_headers=await h.model_headers(), ...))` to go through Bifrost (the headers keep the
gateway from adding its MCP tools to the model's requests: [gateway.md](../gateway.md)). `agents.set_tracing_disabled(True)` keeps the SDK's own tracing (which goes to
OpenAI) off; the harness traces through OpenTelemetry.

This page is Way 1: the harness runs the agent. To keep calling `Runner.run` yourself and plug
in the blocks (memory, `needs_approval` from governance, the `RunState` as the run's
checkpoint in agent-runs, a judge), see the Way 2 recipe:
[blocks/openai-agents.md](../blocks/openai-agents.md).

## Using an existing OpenAI Agents project

```python
from agents import Agent
from trellis import Harness, tool

h = Harness()


@tool(side_effects="irreversible")
def create_po(supplier: str, amount: float) -> str:
    """Create a purchase order."""
    ...


buyer = Agent(name="buyer", instructions="You create purchase orders.", model=model)
agent = h.wrap(buyer, id="buyer", tools=[create_po])  # was: Runner.run(buyer, ...)

result = await agent.run("Order 12000 EUR of steel from ACME.", user="ada")
if result.interrupt:  # create_po asks a person
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
```

`tools=[...]` on `wrap` are added next to the agent's own `tools` for each run (with the MCP
tools the virtual key allows and, memory on, the memory tools). Your own `function_tool`s keep
working and are left untouched; they are not governed, journaled or recorded by the harness.

**Handoffs.** `wrap(tools=...)` reaches the agent you wrap. A specialist reached by a handoff
gets harness tools when it is built:

```python
refunds = Agent(
    name="refunds",
    instructions="You refund orders.",
    model=model,
    tools=await h.tools(refund, framework="openai_agents"),  # FunctionTools, still the harness's
)
triage = Agent(name="triage", instructions="Route requests.", model=model, handoffs=[refunds])
agent = h.wrap(triage, id="support")
```

Their calls go through the harness like any other — governed by the catalog as it is at each
call, approvals, journal, records.

**`Runner.run` itself is not intercepted**: call `agent.run`/`stream`/`resume`/`start` instead.
Outside a harness run, a harness tool refuses to run (`ToolError`: "runs inside a Harness run"),
which the SDK hands the model as the tool's error. The harness passes no
`RunConfig`: the SDK's defaults apply (`max_turns` 10); set model settings on the `Agent`.

## What is automatic

| | |
|---|---|
| Memory push | the context as a leading `system` message of the input (a string input becomes a user message after it) |
| Memory pull | the memory tools are added to the copy per run |
| Records | the transcript (the question and every assistant message), every harness tool call, the `system` outcome; approvals as `TOOL_CALL` feedback |
| Tool hints | from 5 tools, the model is offered the hinted tools per turn (`FunctionTool.is_enabled`) — the memory tools, the hinted ones, every tool already used; your own tools are never narrowed |
| Grounding, judges, tracing | as for every target ([evaluation.md](../evaluation.md), [observability.md](../observability.md)) |
| Hooks | the tool and run hooks as for every target; the model hooks through the SDK's own `RunHooks`, which the harness passes to `Runner.run`: each call reported to `before_model`/`after_model`, none rewritten (the SDK takes nothing back) ([hooks.md](../hooks.md)) |

The answer is the run's `final_output` (a pydantic `output_type` is kept as JSON in the run
record).

## Approvals and pauses

| Pause | How it continues |
|---|---|
| a harness tool that asks (`irreversible`, a catalog `approve_when`), or `trellis.current().ask` in a tool | the run re-runs from its input as the next attempt; the journal returns the calls already made and the answer given (the model is asked again) |
| the SDK's own `function_tool(needs_approval=...)` | the SDK's `RunState` is kept as the run's checkpoint and continued: `approve` runs the call; `reject` (with `answer="why"`) and `answer` are the SDK's rejection with that message, which the model reads; `edit` rejects with a message telling the model to call again with the edited arguments — that call, with exactly those arguments, then runs approved in the same attempt ([interrupts.md](../interrupts.md#framework-approvals-langchains-middleware-and-openai-agents-needs_approval)) |

The SDK's `RunState` is JSON in the run's checkpoint, so a `needs_approval` pause can be answered
from any process or worker. Several approvals in one turn are asked one at a time. Gate a tool
in one place: a `needs_approval` tool is your own (not a harness tool), so the harness does not
also ask for it.

## Streaming

`agent.stream(...)` runs `Runner.run_streamed`: text deltas (`response.output_text.delta`) as
`TEXT_MESSAGE_*` events, the harness tool calls as `TOOL_CALL_*`, then `RUN_FINISHED`.

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target: `start` + `h.worker`/`python -m trellis.harness.worker` (progress saved after
each side-effecting harness call), `schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` as a tool,
`h.evaluate` and online judges (`llm_judge` needs `TRELLIS_JUDGE_MODEL`). See
[runs.md](../runs.md), [surfaces.md](../surfaces.md), [evaluation.md](../evaluation.md).

## Limits

* A resume after a harness approval re-runs the conversation: the model is asked again for the
  steps before the pause (their tool calls replay from the journal).
* The SDK's sessions (`session=`) are not used: memory is the memory service's.
* No `RunConfig` per run (above).

## Run it

* [`examples/openai_agents_agent.py`](../../examples/openai_agents_agent.py) — harness tools
  added per run, an irreversible call approved.
* [`examples/openai_agents_handoff.py`](../../examples/openai_agents_handoff.py) — a handoff to a
  specialist built with `h.tools`, its call approved.
* Tests: `tests/integration/test_openai_agents.py` (`needs_approval`: approve, reject with a
  reason, answer, edit), and against the real services `tests/live/test_live_matrix.py`
  (`openai-agents`).
