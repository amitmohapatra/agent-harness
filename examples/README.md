# Examples

| File | Shows |
| --- | --- |
| [`plain_python.py`](plain_python.py) | all three integration modes, tools, artifacts, claims, child runs — no framework, no services |
| [`langgraph_agent.py`](langgraph_agent.py) | an existing node and a runtime-aware node in one graph, with a checkpointer |
| [`with_memory_and_langfuse.py`](with_memory_and_langfuse.py) | the production shape: Memory Service + OpenTelemetry + Langfuse, enabled by configuration only |

All three run as-is:

```bash
python examples/plain_python.py
python examples/langgraph_agent.py          # needs the [langgraph] extra
python examples/with_memory_and_langfuse.py # add MEMORY_SERVICE_URL / LANGFUSE_* to light up
```

They print JSON log lines (structured logging is on by default) alongside their output.
