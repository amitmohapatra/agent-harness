# Deep Agents

`create_deep_agent(...)` returns a compiled LangGraph graph, so a Deep Agent is wrapped like any
graph — everything on the [LangGraph page](langgraph.md) applies: tools built with `h.tools`,
`agent.run`/`stream`/`resume` instead of `graph.ainvoke`, pauses in place with a checkpointer.
This page is what is particular to Deep Agents.

**Install:** `pip install 'trellis-harness[deepagents]'` (brings `[langgraph]`).

This page is Way 1: the harness runs the Deep Agent. Called yourself, a Deep Agent is a
LangGraph graph, and the Way 2 recipe applies as it is: [blocks/langgraph.md](../blocks/langgraph.md).

## Using an existing Deep Agents project

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

tools = await h.tools(refund, lookup_order, framework="deepagents")
graph = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt="You process refund requests.",
    middleware=[TodoListMiddleware()],  # planning: write_todos
    subagents=[
        {
            "name": "order-checker",
            "description": "Looks up an order.",
            "system_prompt": "Look the order up.",
            "tools": tools,  # the same harness tools
            "model": model,
        }
    ],
    interrupt_on={"send_email": True},  # Deep Agents' own approvals (optional)
    checkpointer=InMemorySaver(),  # pauses resume in place
)
agent = h.wrap(graph, id="refunds")
result = await agent.run("Refund order o-7.", user="ada", thread="ticket-7")
```

## What the harness sees in a Deep Agent

| Part | |
|---|---|
| Your tools from `h.tools` | harness tools: governed by the catalog at each call, approvals, journal, records, spans — in the main agent and in a sub-agent alike (a sub-agent's calls happen inside the run, so they are the run's) |
| Sub-agents (`subagents=[...]`, the `task` tool) | run inside the main agent's run: one run record, one trace, one transcript (the main agent's answer). Give a sub-agent the harness tools it needs (`"tools": tools`); the default `general-purpose` sub-agent gets the main agent's tools |
| Planning (`TodoListMiddleware`, `write_todos`) | the plan lives in the graph's state (`state["todos"]`, read with `graph.aget_state(...)` on the thread); `write_todos` is Deep Agents' own tool, not a harness tool |
| Built-in file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `delete`, `execute`) and `task` | Deep Agents' own: they work on its backend (the graph state by default) and are not governed, journaled or recorded by the harness. Gate the ones that matter with `interrupt_on` — the harness turns that pause into an approval in the run store like any other |
| `interrupt_on={tool: True \| InterruptOnConfig}` | LangChain's `HumanInTheLoopMiddleware`: an approval of the calls it holds, answered with `approve`, `edit`, `reject` (with a reason the model reads), `answer` ([interrupts.md](../interrupts.md#framework-approvals-langchains-middleware-and-openai-agents-needs_approval)); needs a checkpointer |
| Hooks | the tool and run hooks as for every target; the model hooks through the same LangChain middleware as `create_agent`: `create_deep_agent(..., middleware=[ModelHooks()])` ([hooks.md](../hooks.md)) |
| Memory | the context as a leading system message of the main agent (sub-agents get the `task` description, as Deep Agents gives them); the memory tools are in `h.tools(...)` for the main agent and any sub-agent you give them to |

Deep Agents' own `memory=[...]` (AGENTS.md files) and `skills=[...]` are prompts it loads from
its backend; they are independent of the memory service and can be used beside it.

## Approvals, streaming, durable runs, surfaces, evaluation

As for any graph ([langgraph.md](langgraph.md)): a harness tool that asks pauses inside the
tool node (in place with a checkpointer, by re-run without), `stream` yields the run's events,
`start`/workers/`schedule` run it durably (an `interrupt_on` pause resumed by another worker
needs a checkpointer every worker can reach; a harness approval resumed there is answered from
the journal — [which checkpointer](langgraph.md#approvals-and-pauses)), `serve_chat`/`serve_a2a`
serve it, and `h.evaluate`/judges score it (`llm_judge` needs `TRELLIS_JUDGE_MODEL`).

A tool of the agent's own that asks with LangGraph's `interrupt(value)` (a checkpointer
needed) is the graph's own question: a dict `value` carries `options` (several picks with
`multiple`), `expects` with `ui_schema`, `component` with `props`, and `assignee` onto the
`Interrupt` as `ask(...)` does, the answer is checked the same way, and the tool gets it as
given ([the table](langgraph.md#approvals-and-pauses)).

## Limits

* Harness tools are bound when the graph is built, as for `create_agent`: build it without
  what its agent goes without (`h.tools(..., without={...})`), and give it the harness's
  middleware (`create_deep_agent(..., middleware=[ModelHooks()])`) to hide from the model the
  tools of a part turned off later; without it they are offered, and a call is refused
  ([langgraph.md](langgraph.md#limits)).

* Deep Agents' built-in tools (files, `execute`, `task`, `write_todos`) are not the harness's
  (above): gate them with `interrupt_on`.
* A sub-agent's harness call that asks pauses the whole run (the interrupt names the call), and
  the approval resumes the sub-agent where it stopped with a checkpointer, or re-runs the run
  from its journal without one — either way the call runs once.
* A sub-agent sees the `task` description, not the pushed context: give it the memory tools
  (`memory_search`) when it should read memory itself.

## Run it

* [`examples/deepagents_agent.py`](../../examples/deepagents_agent.py) — planning with
  `TodoListMiddleware`, an approval resumed in place.
* Tests: `tests/integration/test_deepagents.py`, `tests/integration/test_hitl_middleware.py`
  (`interrupt_on`), and against the real services `tests/live/test_live_matrix.py`
  (`deepagents`: a sub-agent calling a harness tool and pausing for its approval, `write_todos`,
  `interrupt_on`, a catalog rule set after the graph was built).
