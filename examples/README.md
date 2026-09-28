# Examples

Every file says in its docstring what it demonstrates, and every one of them runs. The last
column is the honest part: what it needs before it shows you anything real.

| File | Shows | Needs |
| --- | --- | --- |
| [`memory_quickstart.py`](memory_quickstart.py) | the README quick start end to end: onboarding, one wrapped agent, the context bundle, `remember`/`recall`, draining the queued writes | a **Memory Service** (otherwise it says so and exits 0) |
| [`memory_tour.py`](memory_tour.py) | every memory operation the service supports, driven through `runtime.memory`, and the wire call each one makes | nothing (an in-process stand-in), or a **Memory Service** |
| [`plain_python.py`](plain_python.py) | all three integration modes, tools, artifacts, claims, child runs | nothing |
| [`langgraph_agent.py`](langgraph_agent.py) | the smallest useful graph: an existing node and a runtime-aware node, with a checkpointer | the `[langgraph]` extra |
| [`reorder_workflow.py`](reorder_workflow.py) | the full picture — see below | the `[langgraph]` extra; optionally a **Memory Service** and Langfuse |
| [`langgraph_chatbot.py`](langgraph_chatbot.py) | a five-node chatbot on a Bifrost gateway, with every configuration section set — see below | a **Bifrost gateway**; optionally a Memory Service |
| [`deepagents_agent.py`](deepagents_agent.py) | Deep Agents: the six bindings, a tool held for a person, and the resumed run | the `[deepagents]` extra |
| [`openai_agents_agent.py`](openai_agents_agent.py) | the OpenAI Agents SDK: the six bindings, and its own `needs_approval` as one harness `Interrupt` | the `[openai-agents]` extra |
| [`claude_agent_sdk_agent.py`](claude_agent_sdk_agent.py) | the Claude Agent SDK: every hook driven with the payloads the CLI sends, no CLI needed | the `[claude-agent-sdk]` extra |

```bash
make examples        # every example that needs no service
make examples-live   # the two that talk to a Memory Service
```

`make examples-live` probes `/health/live` first and fails loudly when nothing answers, rather
than printing "ok" for an example that skipped.

Or one at a time, with no services and no API keys:

```bash
python examples/plain_python.py
python examples/langgraph_agent.py            # needs the [langgraph] extra
python examples/reorder_workflow.py
python examples/memory_tour.py
python examples/deepagents_agent.py           # needs the [deepagents] extra
python examples/openai_agents_agent.py        # needs the [openai-agents] extra
python examples/claude_agent_sdk_agent.py     # needs the [claude-agent-sdk] extra
```

## Against a live Memory Service

```bash
MEMORY_SERVICE_URL=http://localhost:8080 MEMORY_API_KEY=dev-key \
  python examples/memory_quickstart.py       # also memory_tour.py and reorder_workflow.py
```

Those are the defaults, so with the dev service up `python examples/memory_quickstart.py` is
enough. Three things about a live service that these examples have to handle, and that are the
usual reason a first script of your own does not work:

* **onboard first.** A WORKSPACE-visible write needs a tenant, a workspace *row* and a member.
  `memory_quickstart.py::onboard` and `memory_tour.py::onboard` do exactly what
  `tests/support.py::onboard` does; without it the service answers `Workspace not found`.
* **a workspace id cannot be reclaimed.** An id that already labels threads or documents cannot
  later become a workspace (`in use as an anchor`), so a new team never inherits an old team's
  anchors. `memory_tour.py` prints that and falls back to a fresh id rather than failing.
* **writes are asynchronous.** `await harness.drain()` (or `aclose()`) before the process exits,
  or the queued observations never leave — and a 202 means "durably queued", not "retrievable".

The three framework examples use a scripted model behind the harness's own model port, so they
show the bindings without a key and without pretending to show a real model's judgement. Point
`model=` at a `BifrostModelClient` and the same code talks to the gateway.

They print JSON log lines (structured logging is on by default) alongside their output.

## `langgraph_chatbot.py`

A support chatbot whose models are reached through a **Bifrost** gateway, as five nodes and
one conditional edge:

```
recall ──▶ plan ──▶ act ──▶ answer ──▶ verify ──▶ END
             │                 ▲
             └─────────────────┘   no lookup needed: skip `act`
```

| Node | What it does | Harness surface it shows |
| --- | --- | --- |
| `recall` | reads the context the harness already fetched | `query=` on the node drives retrieval; `runtime.memory_context` |
| `plan` | one structured model call → route | `model.structured()`, JSON-schema output, a narrowed `MemoryPolicy` |
| `act` | runs the tool the plan named | `runtime.tools.call()`, tool memory, per-tool timeout |
| `answer` | the customer-facing reply | `model.invoke()`, an `AgentResponse` carrying a `Claim` |
| `verify` | grounding gate — every number must come from a lookup | cheap, no model call; sets the failure signal |

Unlike the other examples it needs a gateway:

```bash
export BIFROST_BASE_URL=http://localhost:8090/v1
export BIFROST_API_KEY=vk-...
export MEMORY_SERVICE_URL=http://localhost:8080   # optional
python examples/langgraph_chatbot.py
```

`build_harness()` sets **every** configuration section explicitly — memory, telemetry and
its capture/sampling policy, observability, models, tools, retries, timeouts, artifacts and
evaluation events — plus every injectable provider (model, memory, tools, artifacts, policy,
redactor, evaluation sink). Most deployments set three or four of these; they are all spelled
out here so the surface is visible in one place.

The graph is executed in `tests/integration/test_langgraph_chatbot.py` against a real
gateway process and a real Memory Service, including the ungrounded-answer path — so this
example cannot drift away from the code either.

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
