# OpenAI Agents SDK

An `agents.Agent` is a target. The harness runs it with the SDK's own `Runner`, on a copy of
your agent that also carries the harness's tools (the gateway's MCP tools, memory's); your agent
object is never changed.

**Install:** `pip install 'trellis-harness[openai-agents]'`. A model name (or the SDK's
default) goes through Bifrost when `BIFROST_URL` is set (`RunConfig.model_provider`, unless the
run's own `run_config` names one), else to its provider with its own key
([configuration.md](../configuration.md#models)). `agents.set_tracing_disabled(True)` keeps the
SDK's own tracing (which goes to OpenAI) off; the harness traces through OpenTelemetry.

This page is Way 1: the harness runs the agent. To keep calling `Runner.run` yourself and plug
in the blocks (memory, `needs_approval` from governance, the `RunState` as the run's
checkpoint in agent-runs, a judge), see the Way 2 recipe:
[blocks/openai-agents.md](../blocks/openai-agents.md).

## Using an existing OpenAI Agents project

```python
from agents import Agent, function_tool
from trellis import Harness

h = Harness()


@function_tool
def create_po(supplier: str, amount: float) -> str:
    """Create a purchase order."""  # the catalog says it is irreversible: a person approves
    ...


buyer = Agent(
    name="buyer", instructions="You create purchase orders.", model=model, tools=[create_po]
)
agent = h.wrap(buyer, id="buyer")  # was: Runner.run(buyer, ...)

result = await agent.run("Order 12000 EUR of steel from ACME.", user="ada")
if result.interrupt:  # create_po asks a person
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="cfo")
```

Your own `function_tool`s are governed as the SDK calls them — the catalog's risk and rules
(an administrator's, in the memory service; without it, a write), approvals, the journal, the
records — and each run's copy also carries the MCP tools the virtual key allows and, memory on,
the memory tools.

**Handoffs.** Every agent reachable through its `handoffs` (recursively, cycles included) runs
as a copy whose own tools are governed like the wrapped agent's — hooks, the catalog's rules,
the journal — and that has the memory tools; your agents are not changed. A `Handoff` object
(`handoff(agent, ...)`) hides its agent until it is invoked, so that agent's tools are not
governed (said once at the first run): hand off to the `Agent` itself.

```python
refunds = Agent(name="refunds", instructions="You refund orders.", model=model, tools=[refund])
triage = Agent(name="triage", instructions="Route requests.", model=model, handoffs=[refunds])
agent = h.wrap(triage, id="support")
```

**Arguments that are not JSON.** The SDK hands a tool the arguments as the model wrote them. A
harness tool called with text that is not a JSON object does not run: the model reads "`<tool>`
was not run: its arguments are not valid JSON (...). Call it again with arguments that fit its
schema." as the tool's output and the run goes on, as with the SDK's own `function_tool`s (and
`ReAct`). An SDK `needs_approval` tool pauses before its arguments are read: the person sees
them as written (`{"arguments": "<the text>"}`).

**`Runner.run` itself is not intercepted**: call `agent.run`/`stream`/`resume`/`start` instead.
Outside a harness run, a harness tool refuses to run (`ToolError`: "runs inside a Harness run"),
which the SDK hands the model as the tool's error. `Runner.run`'s own arguments are
`framework_options=`: on `h.wrap` for every run, on `run`/`stream`/`start` for one (over the
agent's) — `max_turns`, `run_config`, `context`, `session`... The harness gives `Runner.run` the
agent, its input and its hooks itself: `starting_agent`, `input` and `hooks` are refused, and so
is a name `Runner.run` does not take ([configuration.md](../configuration.md#the-frameworks-own-run-options)).

```python
agent = h.wrap(
    triage, id="support", framework_options={"run_config": RunConfig(tracing_disabled=True)}
)
await agent.run(question, user="ada", framework_options={"max_turns": 25})
```

A queued run's own (`start`, `schedule`) are JSON (`max_turns`, a `run_config` given as a dict);
a `RunConfig` object goes on `h.wrap`, which every process builds again from code. The harness
sets no `RunConfig` field of its own (its hooks are `Runner.run`'s `hooks=`), so yours is passed
as it is.

A `SandboxAgent` is wrapped the same way: the SDK runs it only with `RunConfig(sandbox=...)`
(else `UserError: SandboxAgent execution requires RunConfig(sandbox=...)`), so give its sandbox
client there. Its own tools (`exec_command`, `apply_patch`...) are the SDK's; the harness tools
you add sit beside them ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).

```python
box = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
agent = h.wrap(
    SandboxAgent(name="analyst", instructions="..."),
    id="analyst",
    framework_options={"run_config": box},
)
```

## What is automatic

| | |
|---|---|
| Memory push | the context as a leading `system` message of the input (a string input becomes a user message after it); without a `session`, memory keeps the conversation: the run's session is the memory service's thread (`bridges.session.MemorySession`), which gives the SDK the earlier turns as its own input items — messages, `function_call`, `function_call_output`, a call never parted from its result — and each run's items are written back before it returns |
| Memory pull | the memory tools are added to the copy per run |
| Session | a run given no `session` (nor a conversation OpenAI's servers keep) reads its earlier turns from the memory service's thread through the SDK's own `Session` (`MemorySession`), so memory's window is not sent beside it; every turn is recorded there by the harness, and removing items through the session is refused (`ThreadReadOnly`). A `session` of yours is kept |
| Pauses | the SDK's saved `RunState` (a `needs_approval` tool) is kept with the run in agent-runs, so any worker resumes it |
| Agent tools | an `Agent.as_tool` call is governed by the catalog by its tool name and is a child run (its record names the run, its turns go to memory); what its run spends counts toward the run's budget. The tool runs a governed copy of its agent: that agent's own tools and MCP servers are governed as the run's (an agent tool of a shape this SDK release does not build runs its agent as it is, said once) |
| MCP servers | the agent's own `mcp_servers` (and its handed-off agents'): their tools are of the run's tool set as each server lists them, and each call goes through governance (the tool's annotations for its risk, the catalog's word over them), the journal and the record, on a copy of the server; a call that pauses the run stops the SDK's run |
| Hosted tools | file search, web search, the code interpreter, image generation, a hosted MCP server's tools: recorded as the run's tool calls with what they were asked, what they found and the step's duration; of the tool set and the catalog; run by the provider inside the model's step, so not approvable |
| Records | the transcript (the question and every assistant message), every harness tool call, the `system` outcome; approvals as `TOOL_CALL` feedback |
| Tool hints | from 5 tools, the model is offered the hinted tools per turn (`FunctionTool.is_enabled`) — the memory tools, the hinted ones, every tool already used — your own function tools as the harness's (their own `is_enabled` asked first); handoffs and agent tools are never narrowed away |
| Large results | a harness tool's result over 80,000 characters is kept in the run, the model reads a head-and-tail preview naming `/large_tool_results/<id>` and pages it with `read_file`, enabled once there is something to read ([tools.md](../tools.md#large-results)) |
| Grounding, judges, tracing | as for every target ([evaluation.md](../evaluation.md), [observability.md](../observability.md)) |
| Hooks | the tool and run hooks as for every target; a model call is the SDK's own: the agent's `AgentHooks` (`Agent(hooks=...)`: `on_llm_start`, `on_llm_end`), which run under `h.wrap` as they do without it ([hooks.md](../hooks.md)) |

The answer is the run's `final_output` (a pydantic `output_type` is kept as JSON in the run
record).

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | its own: a `SandboxAgent`, its sandbox given as `framework_options={"run_config": RunConfig(sandbox=...)}` ([above](#using-an-existing-openai-agents-project)) |
| Skills | its own: the `SandboxAgent`'s `Skills` capability ([skills.md](../skills.md#native-or-ours)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks ([hooks.md](../hooks.md)); a model call is the SDK's: the agent's `AgentHooks` (`Agent(hooks=...)`) |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)) (its tokens; the SDK reports no dollars); its native turn limit is `framework_options={"max_turns": n}` |

## Native or ours: skills, prompts, sandbox

* **Sandbox and skills:** the SDK's `SandboxAgent` (`agents.sandbox`: `Shell`, `Filesystem`,
  `Compaction`, and the `Skills` and `Memory` capabilities; Docker, Unix-local and hosted
  clients) is the native choice for an agent that works on files and folders of skills. It runs
  only with `RunConfig(sandbox=SandboxRunConfig(client=...))`, which `h.wrap` does not pass:
  run it with `Runner.run` yourself and plug the blocks in
  ([blocks/openai-agents.md](../blocks/openai-agents.md),
  [sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours),
  [skills.md](../skills.md#native-or-ours)).
* **Prompts:** `Agent(prompt=Prompt(id=..., version=...))` is a prompt stored at OpenAI, for
  OpenAI's models through the Responses API; the harness passes it through untouched. For a
  stored prompt of the gateway: `instructions=(await h.prompt("triage")).text`
  ([prompts.md](../prompts.md#native-or-ours)).

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

## Sub-agents and child runs

* **Agent tools** (`Agent.as_tool`): each call is a child run of the run — its record names the
  run (`h.runs.iterate(parent_run_id=...)`), the agent is given the memory context for its task
  before the task, its tools are governed and recorded under its own run, its turns go to memory,
  what it spends counts toward the run's budget; a pause inside it pauses the run, and the
  resume continues the same child ([subagents.md](../subagents.md#automatic)).
* **Handoffs** are not child runs: the agent handed off to goes on with the same run and the
  same conversation, its own tools and MCP servers governed as the run's and given the memory
  tools ([above](#using-an-existing-openai-agents-project)).
* **A wrapped agent** called from a tool's code (`await other.run(...)`) is a child run too, and
  the wrapped OpenAI agent is a `ReAct`'s sub-agent like any agent (`subagents=[agent]`).

Runnable: [examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target: `start` + `h.worker`/`python -m trellis.harness.worker` (progress saved after
each side-effecting harness call), `schedule`, `serve_chat`, `serve_a2a`, `a2a(url)` as a tool,
`h.evaluate` and online judges (`llm_judge` asks the gateway's judge alias (or its own `model=`)). See
[runs.md](../runs.md), [surfaces.md](../surfaces.md), [evaluation.md](../evaluation.md), and
[examples/06_scenarios/runs_every_framework.py](../../examples/06_scenarios/runs_every_framework.py)
(an agent of every framework queued, one worker for all).

## Limits

* A resume after a harness approval re-runs the conversation: the model is asked again for the
  steps before the pause (their tool calls replay from the journal).
* The SDK's sessions (`session=`) are yours to pass (`framework_options={"session": ...}`), as are `conversation_id`, `previous_response_id` and `auto_previous_response_id` (OpenAI's servers keep the thread): with a durable one, the SDK keeps the thread, so the memory context is pushed without the recent conversation (memory's window), as for a checkpointed LangGraph graph. A `SQLiteSession(":memory:")` is kept in RAM only and lost on a restart: memory's window stays on, and a warning says so once (`trellis.harness.history`). Long-term memory stays the memory service's.
* Token budgets (`Budget(max_tokens=...)`) are enforced on every run: the harness always passes its `RunHooks`, which count each reply's usage, whether or not the agent has hooks.
* A `RunConfig` object given to one in-process run (`run`/`stream`) is not kept with its record:
  a resume runs with the agent's (put it on `h.wrap`, or give it as a dict).

## Run it

* [`examples/02_way1_openai_agents/agent.py`](../../examples/02_way1_openai_agents/agent.py) — the agent's
  own tool governed, an irreversible call approved.
* [`examples/02_way1_openai_agents/handoff.py`](../../examples/02_way1_openai_agents/handoff.py) — a handoff to a
  specialist with its own tool, its call approved.
* Tests: `tests/integration/test_openai_agents.py` (`needs_approval`: approve, reject with a
  reason, answer, edit), and against the real services `tests/live/test_live_matrix.py`
  (`openai-agents`).
