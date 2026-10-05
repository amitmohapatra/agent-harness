# Sub-agents: an agent as another agent's tool

`agent.as_tool()` turns any wrapped agent into a tool of any other. The model decides when to
delegate and what to delegate at once; each call is a *child run* of the sub-agent, with its
own record, journal and answer, inside the parent's call. Nothing to configure: you name the
agents, the harness does the rest — the child's identity and time, its pauses, its crashes,
its cancellation.

```python
research = h.wrap(ReAct(system="You research.", model=model), id="research", tools=[search])
writer = h.wrap(
    ReAct(system="You write. Ask research for facts.", model=model),
    id="writer",
    tools=[research.as_tool()],
)
result = await writer.run("Write about lighthouses", user="ada")
```

| You write | Automatic |
|---|---|
| `tools=[agent.as_tool()]` (or `h.tools(agent.as_tool(), framework=...)`) | a child run per call: its id, its parent, its tenant, user, thread, time and trace; its pauses answered through the parent; its crash continued; its cancel with the parent |
| optionally `as_tool(name=, description=, side_effects=)` | the tool's name (the agent's id), description (a function's docstring, else a generic one) and side effects (from the tools the child declares) |

## What

A sub-agent is a wrapped agent called as a tool: `{"message": "<the task>"}` in, its answer
out. Each call starts a run of the sub-agent (a *child run*) that the harness records like any
other — memory pushed and recorded, tools governed, spans, grounding and judges — and that
answers the parent's model with its output.

## When

* A task splits into parts a specialist does better (research, booking, code review), each
  with its own instructions, tools and model.
* Parts that do not depend on each other should run at once: the model asks for them in one
  step, and the children that only read run together.
* A part needs a person (a choice, an approval) without the parent knowing how to ask.

Use [`a2a(url)`](surfaces.md) instead for an agent served elsewhere (another team, another
vendor); a sub-agent runs in the parent's process.

## Where

Way 1, every target on either side: a `ReAct` parent with a function, LangGraph, OpenAI Agents,
Claude or `ReAct` child, or any framework's parent calling the tool built by
`h.tools(child.as_tool(), framework=...)`; every mode the parent runs in (`run`, `stream`,
`start` and workers, a resume in another process, `serve_chat`, `serve_a2a`). Way 2 has no
sub-agent block: an agent you do not wrap calls another with `remote()`
([blocks/a2a.md](blocks/a2a.md)).

## How

```python
@tool(side_effects="read")
def flights(city: str) -> str: ...


async def booker(input: str, agent: Runtime) -> str:
    """Find a hotel in a city, once the traveller has chosen a budget."""
    budget = await agent.ask("Which budget?", options=["low", "high"])
    return await agent.tools.call("hotels", city=input, budget=budget)


scout = h.wrap(ReAct(system="You find flights.", model=model), id="scout", tools=[flights])
hotel = h.wrap(booker, id="booker", tools=[hotels])
planner = h.wrap(
    ReAct(system="You plan trips: ask scout and booker at once.", model=model),
    id="planner",
    tools=[scout.as_tool(), hotel.as_tool()],
)
result = await planner.run("Plan a weekend in Oslo", user="ada")
while result.interrupt:  # booker's question, asked through the planner
    result = await planner.resume(
        result.interrupt.interrupt_id, "answer", answer="low", reviewer="ada"
    )
```

`agent.as_tool(*, name=None, description=None, side_effects=None)`. Override only what you
know better: a `name` the model reads more easily, a `description` saying when to delegate, or
`side_effects` when the child does more (or less) than its tools say.

## Automatic

**The child run.** Its id is derived from the parent's call — the parent run, the call and its
occurrence (the call's `idempotency_key`) — so the parent's next attempt finds it again. Its
record names its parent (`RunStart.parent_run_id`; `runs.iterate(parent_run_id=...)` lists a
run's children). It inherits the parent's tenant, user and thread, the parent's `deadline`,
and what is left of the parent's time as its `timeout`: a child never works past its parent.
Its agent version is its own. Its spans are in the parent's trace, under the call
(`invoke_agent <child>` with `trellis.parent_run_id`), and so are its grounding and judge
scores.

**At once or one at a time.** The tool only reads when every tool the child declares only
reads (its own, the MCP tools its key allows) and it has no tools the harness does not run (a
framework's own: an OpenAI Agents agent's `function_tool`s or handoffs, a graph's tools not
built with `h.tools`, Claude's built-in tools); otherwise it writes. In `ReAct` the reads run at
once and the writes one at a time after them, in the model's order
([react.md](frameworks/react.md)); governance treats the tool like any other (a write is
announced; the catalog may say more).

**A child that pauses.** Its question — a person's answer, an approval of one of its tool calls
— pauses the parent with the same question: the interrupt is the parent's (its run, its inbox
entry), its fields the child's (`question`, `options`, `expects`, `tool_call`, `assignee`,
`deadline`...), and `payload["subagent"]` says which child asked (`agent_id`, `run_id`,
`interrupt_id`). `parent.resume(interrupt_id, ...)` answers it: the parent's next attempt hands
the answer to the child, which continues, and then the parent. A decision about a tool call is
fed back on the child's run, where the call was made. Two children paused at once are asked one
after the other.

**Re-runs.** When the parent runs again — after a pause, or a crash — its call finds the child by
its id:

| The child | The call |
|---|---|
| finished | returns its output, from its record (nothing runs) |
| failed, timed out, cancelled | is an error the parent's model reads |
| paused | pauses the parent with its question — or, answered, continues it |
| running (its worker died inside it) | continues it where it was |

**Crashes.** While a child works, its journal is kept in the parent's journal
(`Journal.children`) and its progress is saved with the parent's: after every side-effecting
call of the child, the parent's checkpoint holds it. When the worker dies, the parent's next
attempt continues the child from there — the calls it completed are replayed, a write that was
in flight is of [unknown effect](reliability.md#unknown-outcomes) — so no side effect of the
child is repeated. The tool itself is *resumable*: the bridge runs it again after a crash
instead of reporting it of unknown effect.

**Cancel.** Cancelling the parent cancels its children that have not ended: a running child
stops with its parent, a paused one (waiting for a person) is cancelled at once, and theirs
too. So does answering a child's question with `cancel` on the parent.

## On failure

* A child that fails (an error, its time out) is an error the parent's model reads
  (`"<tool> failed: the <child> agent's run ended ERROR: ..."`), and decides what to do; it is
  not run again.
* The parent running out of time stops its children with it.
* A child that cannot be started (the run store refuses it) is the call's error, like any tool
  call's.

## Limits

* A child runs in its parent's process, inside the parent's call — not queued to other
  workers. "Running elsewhere" can only be what a crashed attempt left: it is continued, not
  waited for.
* A child's events (its text, its own tool calls) are on its own run's stream, not the
  parent's; the parent's stream shows the call and its result.
* A checkpointed LangGraph child runs on the conversation's LangGraph thread, which its other
  runs share.
* The edited arguments of a child's tool call are checked when the child's tool runs, not when
  the parent is resumed.
* `h.feedback(child_run_id, ...)` scores the child's own trace; its spans are in the parent's.

## Run it

* [`examples/react_subagents.py`](../examples/react_subagents.py) — a planner delegating to two
  agents at once, one of which asks a person.
* Tests: `tests/integration/test_subagents.py` (every adapter as a child, two at once, a child
  that asks, a worker killed inside a child, cancel, time), and against the real services
  `tests/live/test_live_subagents.py`.
