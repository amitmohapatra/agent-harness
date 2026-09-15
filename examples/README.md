# Examples

| File | Shows |
| --- | --- |
| [`plain_python.py`](plain_python.py) | all three integration modes, tools, artifacts, claims, child runs — no framework, no services |
| [`langgraph_agent.py`](langgraph_agent.py) | the smallest useful graph: an existing node and a runtime-aware node, with a checkpointer |
| [`reorder_workflow.py`](reorder_workflow.py) | the full picture — see below |
| [`memory_tour.py`](memory_tour.py) | every memory operation the service supports, driven through the harness |

All three run as-is, with no services and no API keys:

```bash
python examples/plain_python.py
python examples/langgraph_agent.py          # needs the [langgraph] extra
python examples/reorder_workflow.py
python examples/memory_tour.py
```

They print JSON log lines (structured logging is on by default) alongside their output.

## `reorder_workflow.py`

A supply-chain reorder decision for one SKU, as a six-node LangGraph workflow:

```
reorder-workflow                      <- harness.execution(): one trace for the turn
  └── triage ──┬── inventory  (tool + model, starts a nested agent)  ┐
               ├── demand     (tool + business logic)                ├─ parallel
               └── supplier   (tool)                                 ┘
                        │
                     decide  (pure business logic, claims, artifact)
                        │
                     explain (model, claims + observations -> memory)
```

What it demonstrates:

* **parallel agents** — the three investigations run concurrently; the harness does not
  serialise them, and the whole turn is ~110 ms of mostly-simulated I/O;
* **nested agent lineage** — `inventory` calls a plain (non-node) `supplier-risk-agent`,
  which appears as a child run of the node that started it;
* **tools two ways** — registered on the harness and called through `runtime.tools`, and a
  function wrapped with `@harness.wrap_tool` and called directly;
* **model calls** through `runtime.model`, with token and cost metrics recorded;
* **real business logic** in `decide` — safety stock at a 95% service level, reorder point
  from lead time and demand volatility, pack-size rounding, supplier minimum order
  quantity, a per-order budget cap and placeability checks. No model call: this part should
  be auditable, not generated;
* **artifacts** — the purchase-order draft is stored and the state keeps only its reference;
* **claims, evidence and recommended actions** on the result, with the state update staying
  a plain dict via a `state_mapper`;
* **memory** — observations written after the result, never before;
* **Langfuse** — enabled by configuration when the keys are present, with no agent changes.

Attach the real services by setting environment variables; the code does not change:

```bash
MEMORY_SERVICE_URL=http://localhost:8080 MEMORY_API_KEY=... \
LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... LANGFUSE_HOST=https://cloud.langfuse.com \
  python examples/reorder_workflow.py
```

The workflow is also executed as a test (`tests/e2e/test_example_workflow.py`), which
asserts the arithmetic, the span hierarchy and the parallelism — so this example cannot
drift away from the code.

## `memory_tour.py`

One agent run that exercises the whole Memory Service surface through `runtime.memory`, and
prints the calls it made:

| Push | Get |
| --- | --- |
| chat turns → history | `retrieve()` → one bundle: conversation + memories + RAG + graph + summaries |
| `observe()` → episodic memory | `recall()` → ranked evidence only |
| `remember()` → typed memory (`memory_type`, `lifetime`, `visibility`) | `history()` → the conversation window |
| `add_document()` → the RAG corpus | `graph_query()` → knowledge-graph facts, with `as_of` |
| `share()` → the agent group | `memories()` → the inventory view |
| tool invocations → tool memory | `verify()` → grounding report |
| `forget()` → deletion | |

It runs against an in-process stand-in by default, so you can see the wire calls without a
service; point `MEMORY_SERVICE_URL` at a real one and nothing else changes. A test asserts
that every operation produces its own `agent.memory.*` span.
