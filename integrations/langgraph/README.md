# universal-agent-harness-langgraph

The LangGraph adapter for the [Universal Agent Harness](../../README.md). The harness core
never imports LangGraph; installing this package is what makes `harness.langgraph` work.

```bash
pip install "universal-agent-harness[langgraph]"
```

```python
from universal_agent_harness import AgentHarness

harness = AgentHarness(memory=memory_client, defaults={"tenant_id": "acme"})

# an existing node, unchanged
graph.add_node(
    "inventory",
    harness.langgraph.wrap_node(
        existing_node,
        agent_id="inventory-agent",
        query="question",
    ),
)


# a runtime-aware node
@harness.langgraph.agent(agent_id="answer-agent", query="question")
async def answer_node(state, agent):
    response = await agent.model.invoke(state["question"])
    return {"answer": response.text}
```

## What it does

* derives identity from the `RunnableConfig` — `thread_id` → memory thread,
  `checkpoint_ns` segments → agent-run lineage (subgraphs become parent runs),
  `langgraph_step` → turn, the executing task id → task;
* produces an agent run id that is **stable across replays of the same superstep**, so a
  checkpoint retry does not duplicate memory writes;
* builds a wrapper that declares exactly the injectables the wrapped node declared
  (`config`, `store`, `writer`, `previous`, `runtime`) and forwards them;
* maps `AgentResponse` back to a normal state update — by default, whatever the node itself
  returned.

## What it does not do

Own the graph topology, routing, reducers, the checkpointer or the state schema, or reach
into `langgraph._internal`. Application identity can be supplied per invocation:

```python
await app.ainvoke(
    state,
    {
        "configurable": {
            "thread_id": "chat-42",
            "harness": {"tenant_id": "acme", "user_id": "u1", "work_id": "wo-9"},
        }
    },
)
```

Tested against the versions listed in [COMPATIBILITY.md](../../COMPATIBILITY.md).
