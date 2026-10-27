# Deep Agents

`create_deep_agent(...)` returns a compiled LangGraph graph, so a Deep Agent is wrapped like any
graph — everything on the [LangGraph page](langgraph.md) applies: its own tools governed as it
runs them, the harness's added with `middleware=trellis.middleware()` (or `HarnessTools` alone),
`agent.run`/`stream`/`resume` instead of `graph.ainvoke`, pauses in place with a checkpointer.
This page is what is particular to Deep Agents. `ReAct` is a Deep Agent built this way
([react.md](react.md)): every Deep Agents feature, plus the harness's middleware.

**Install:** `pip install 'trellis-harness[deepagents]'` (brings `[langgraph]`).

This page is Way 1: the harness runs the Deep Agent. Called yourself, a Deep Agent is a
LangGraph graph, and the Way 2 recipe applies as it is: [blocks/langgraph.md](../blocks/langgraph.md).

## Using an existing Deep Agents project

```python
from deepagents import create_deep_agent
from trellis import middleware
from trellis.harness.middleware import RunCheckpointer

tools = [refund, lookup_order]  # LangChain tools: the graph's own, governed as it runs them
graph = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt="You process refund requests.",
    # the harness's part of ReAct: the MCP and memory tools per call, narrowed per turn; the
    # step limit and stall guard; the hooks, spans and call_timeout; planning on a long task
    middleware=middleware(max_steps=20, call_timeout=60),
    subagents=[
        {
            "name": "order-checker",
            "description": "Looks up an order.",
            "system_prompt": "Look the order up.",
            "tools": tools,  # the same tools
            "model": model,
        }
    ],
    interrupt_on={"send_email": True},  # Deep Agents' own approvals (optional)
    checkpointer=RunCheckpointer(),  # the checkpoint in the run: a resume continues in place
)
agent = h.wrap(graph, id="refunds")
result = await agent.run("Refund order o-7.", user="ada", thread="ticket-7")
```

Without `middleware()` the graph still runs governed and narrowed (said once at `h.wrap`):
the gateway's MCP tools and the memory tools, the step limit, the stall guard, the hooks and
`call_timeout` are what it adds ([react.md](react.md#middleware-on-your-own-graph)). Any
checkpointer of yours works too (`InMemorySaver`, Postgres); `RunCheckpointer()` keeps the
latest checkpoint in the run's journal — Deep Agents' delta-kept `messages` and `files`
included — so a resume on another worker continues it. Runnable:
[examples/02_way1_deepagents/middleware.py](../../examples/02_way1_deepagents/middleware.py).

## What the harness sees in a Deep Agent

| Part | |
|---|---|
| Your tools | the graph's own, governed at each call as the graph runs them: the catalog's risk and rules, approvals, journal, records, spans — in the main agent and in a sub-agent alike (a sub-agent's calls happen inside the run, so they are the run's) |
| Sub-agents (`subagents=[...]`, the `task` tool) | each `task` call is a child run of the run ([below](#sub-agents-and-child-runs)): its own record, its turns in memory, its spans in the run's trace. Give a sub-agent the tools it needs (`"tools": tools`); the default `general-purpose` sub-agent gets the main agent's tools |
| Planning (`TodoListMiddleware`, `write_todos`) | the plan lives in the graph's state (`state["todos"]`, read with `graph.aget_state(...)` on the thread); `write_todos` is Deep Agents' own tool, not a harness tool |
| Built-in tools | Deep Agents' own. Those that only change the graph's state — `write_todos`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` on its default backend — run as they are (the run's checkpoint keeps them); any other, `execute` among them, is governed as the graph's own tools are (the catalog's risk, else a write); `task` is a child run. Gate more with `interrupt_on` — the harness turns that pause into an approval in the run store like any other |
| `interrupt_on={tool: True \| InterruptOnConfig}` | LangChain's `HumanInTheLoopMiddleware`: an approval of the calls it holds, answered with `approve`, `edit`, `reject` (with a reason the model reads), `answer` ([interrupts.md](../interrupts.md#framework-approvals-langchains-middleware-and-openai-agents-needs_approval)); needs a checkpointer |
| Hooks | the tool and run hooks as for every target; a model call is LangChain middleware, as for `create_agent`: `create_deep_agent(..., middleware=[yours, ModelHooks()])` ([hooks.md](../hooks.md)) |
| A stored prompt of the gateway (`middleware=[ModelHooks(prompt=stored("triage"))]`) | pinned at each run's start, journaled and said (a `prompt` event, the span's `trellis.prompt.*`); each model call gets the version the run pinned ([prompts.md](../prompts.md)) |
| Tool hints | as for a `create_agent` graph ([langgraph.md](langgraph.md#what-is-automatic)): from 5 tools, each model call of the main agent is offered only the harness tools the run offers (the hinted ones, the memory tools, those already used); Deep Agents' state tools (`ls`, `read_file`...) and `task` are always offered, its other own tools as the run offers them; a sub-agent's model is offered the tools it was built with, and their calls are governed and recorded as the run's |
| Memory | the context as a leading system message of the main agent; a sub-agent, a child run, gets the context for its task before the task description (`## Your task`); the memory tools come with `HarnessTools` (or `await h.memory_tools()` for a sub-agent you give them to) |

Deep Agents' own `memory=[...]` (AGENTS.md files) and `skills=[...]` are prompts it loads from
its backend; they are independent of the memory service and can be used beside it.

## Sub-agents and child runs

A `task` call is a child run of the run (`bridges.helpers`): its record names the run
(`h.runs.iterate(parent_run_id=...)` lists it), the memory context for its task comes before
the task description, its tool calls are governed under the catalog's rules and recorded under
its own run, its task and answer go to memory under it, its model calls count toward the run's
budget, and a sub-agent cut by a crash is ended by the worker that continues the run
([subagents.md](../subagents.md#automatic)). The wrapped Deep Agent is, the other way round, a
`ReAct`'s sub-agent like any agent (`subagents=[agent]`). Runnable:
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## The store

A Deep Agent compiled without a `store` gets the memory service as its store, so a `StoreBackend` writes the user's memories
([langgraph.md](langgraph.md#what-is-automatic)).

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | `create_deep_agent(model, **sandbox().deep_agent())`: its `backend=`, the run's own sandbox; or a backend of Deep Agents' own (below) |
| Skills | `create_deep_agent(model, **src.deep_agent())`: its own `SkillsMiddleware`, the skill source's skills a `/skills/` route of its backend; with a sandbox, `**deep_agent(sandbox(), src)` — both in its one `backend=` (a `CompositeBackend`) ([skills.md](../skills.md)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks ([hooks.md](../hooks.md)); a model call is LangChain middleware: `create_deep_agent(..., middleware=[...])` |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)); its native turn limit is LangGraph's: `framework_options={"recursion_limit": n}` |

## Native or ours: skills, prompts, sandbox

* **Skills:** for `SKILL.md` folders, use Deep Agents' own: `create_deep_agent(skills=[...])`
  (its `SkillsMiddleware`, read through the backend) ([skills.md](../skills.md#native-or-ours)).
* **Prompts:** Deep Agents takes text: `system_prompt=(await h.prompt("triage")).text` for a
  stored prompt of the gateway ([prompts.md](../prompts.md#native-or-ours)).
* **Sandbox:** prefer its own backend for `execute` (`BaseSandbox`, `LangSmithSandbox`;
  `LocalShellBackend` is the host's shell, unisolated), gated with `interrupt_on`; `**sandbox().deep_agent()`
  when every command must be governed by the catalog and journaled
  ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).

## Approvals, streaming, durable runs, surfaces, evaluation

As for any graph ([langgraph.md](langgraph.md)): a harness tool that asks pauses inside the
tool node (in place with a checkpointer, by re-run without), `stream` yields the run's events,
`start`/workers/`schedule` run it durably (an `interrupt_on` pause resumed by another worker
needs a checkpointer every worker can reach; a harness approval resumed there is answered from
the journal — [which checkpointer](langgraph.md#approvals-and-pauses)), `serve_chat`/`serve_a2a`
serve it, and `h.evaluate`/judges score it (`llm_judge` asks the gateway's judge alias (or its own `model=`)).

A tool of the agent's own that asks with LangGraph's `interrupt(value)` (a checkpointer
needed) is the graph's own question: a dict `value` carries `options` (several picks with
`multiple`), `expects` with `ui_schema`, `component` with `props`, and `assignee` onto the
`Interrupt` as `ask(...)` does, the answer is checked the same way, and the tool gets it as
given ([the table](langgraph.md#approvals-and-pauses)).

A Deep Agent is a graph: `framework_options=` goes into its run's config as for any graph
(`recursion_limit`, `configurable` keys, merged key by key; a `thread_id` among them is
refused: the thread is `run(thread=)` —
[langgraph.md](langgraph.md#the-graphs-own-run-options)).

## Limits

* Tools are bound when the graph is built, as for `create_agent`: give it the harness's
  middleware (`create_deep_agent(..., middleware=[ModelHooks()])`) to hide from the model the
  tools of a part turned off later; without it they are offered, and a call is refused
  ([langgraph.md](langgraph.md#limits)).

* Deep Agents' state tools (`write_todos`, the files on its default backend) are not governed
  or journaled (above): the graph's checkpoint keeps what they did; gate them with
  `interrupt_on`.
* A sub-agent's harness call that asks pauses the whole run (the interrupt names the call), and
  the approval resumes the sub-agent where it stopped with a checkpointer, or re-runs the run
  from its journal without one — either way the call runs once.
* A sub-agent is given the context for its task, not the main agent's tools: give it the
  memory tools (`memory_search`) when it should read memory itself.

## Run it

* [`examples/02_way1_deepagents/agent.py`](../../examples/02_way1_deepagents/agent.py) — planning with
  `TodoListMiddleware`, an approval resumed in place.
* Tests: `tests/integration/test_deepagents.py`, `tests/integration/test_hitl_middleware.py`
  (`interrupt_on`), and against the real services `tests/live/test_live_matrix.py`
  (`deepagents`: a sub-agent calling a harness tool and pausing for its approval, `write_todos`,
  `interrupt_on`, a catalog rule set after the graph was built).
