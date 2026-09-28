# trellis-harness

A framework-neutral runtime layer around agents you **already have**. It is not an agent
framework, and it does not want to be one: LangGraph, CrewAI or plain Python keep owning
the agent, its state and its control flow. The harness wraps an execution and supplies the
cross-cutting concerns every production agent ends up needing.

## What is a harness? (in plain English)

A climbing harness does not climb for you. It attaches to the climber you already have and
carries the rope, the safety gear and the anchor points — so that when something goes wrong,
you are still attached to the wall.

This is that, for agents.

You write an agent. It works on your laptop. Then it goes to production, and a different set
of questions starts arriving — none of them about your agent's logic:

> *Who ran this, and for which customer?*
> *What did it remember from last week's conversation, and what did it just learn?*
> *It took 40 seconds. Where did they go?*
> *How much did that answer cost in tokens?*
> *It failed at 2am — which step, and can we safely retry it?*
> *Did any customer data end up in our logging vendor?*
> *Three agents worked on this request. What did each of them actually do?*

Answering those means writing the same plumbing into every agent: passing a tenant id
through every function, fetching memory before and saving after, opening trace spans,
counting tokens, setting timeouts, catching and classifying errors, making retries safe,
scrubbing secrets before they're logged. It is perhaps 300 lines per agent. It is boring,
it is easy to get subtly wrong, and by the fifth agent every copy has drifted.

The harness is that plumbing, written once. You hand it your agent; it hands you back the
same agent with all of the above attached:

```python
wrapped = harness.wrap(my_agent, agent_id="inventory-agent")
```

Your agent's code does not change. It does not learn a new framework. It does not even have
to know the harness exists.

**A concrete before and after.** The same agent, with production concerns handled:

```python
# before — the agent is 3 lines; the plumbing is the rest
async def inventory_agent(question, tenant_id, user_id, thread_id, trace):
    span = tracer.start_span("inventory-agent")               # tracing
    span.set_attribute("tenant.id", tenant_id)                # ...by hand, every time
    memory_ctx = memory.bind(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id)
    bundle = await memory_ctx.context(question)               # fetch memory
    started = time.time()
    try:
        async with asyncio.timeout(30):                       # deadline
            answer = await llm.complete(prompt(question, bundle))
    except Exception as exc:
        span.record_exception(exc)                            # error handling
        metrics.count("agent.errors", agent="inventory")
        raise
    finally:
        span.end()
        metrics.timing("agent.duration", time.time() - started)
    await memory_ctx.observe(answer, idempotency_key=???)     # save memory, safely
    return answer

# after — the agent is the 3 lines; the harness is the rest
@harness.agent(agent_id="inventory-agent")
async def inventory_agent(question, agent):
    response = await agent.model.invoke(prompt(question, agent.memory_context))
    return response.text
```

**What you get for that one line:** who ran it (tenant, user, thread, turn, parent agent),
what it remembered and learned, one trace covering every agent/model/tool/memory step, token
and cost numbers, deadlines that actually cancel, errors sorted into categories you can act
on, retries that don't double-charge anyone, and secrets kept out of your telemetry — by
default, without you asking.

**When you do *not* need it:** one agent, one framework, no shared memory, no multi-tenancy.
Then this is overhead and you should skip it. It earns its place at *n* agents across *m*
teams, where the alternative is the same 300 lines copy-pasted and subtly different in each.

**Contents** · [What is a harness?](#what-is-a-harness-in-plain-english) ·
[Install](#install) · [First agent](#your-first-agent) · [Quickstart](#5-minute-quickstart) ·
[Tutorial](#tutorial-six-steps) · [Integration modes](#three-integration-modes) ·
[LangGraph](#langgraph) · [Tools and models](#tools-and-models) ·
[Memory](#memory-service-integration) · [Results and errors](#results-and-errors) ·
[Timeouts, cancellation, retries](#timeouts-cancellation-retries-idempotency) ·
[Observability](#observability) · [Langfuse setup](#langfuse-setup) ·
[Extending](#extending-the-harness) · [Configuration](#configuration) ·
[Status](#status) · [Docs](#documentation) · [Performance](#performance) ·
[Events and surfaces](#events-interrupts-and-surfaces-030) ·
[Evaluation and durability](#evaluation-and-durability-030)

Every page, and the question it answers: [**docs/README.md**](docs/README.md).

---

## Install

```bash
pip install trellis-harness                          # core: plain Python
```

Every extra, what it adds, and the distribution it pulls in:

| Extra | Adds | Distribution / dependency | Page |
| --- | --- | --- | --- |
| *(none)* | the core: contracts, config, the OTel **API**, the Memory Service SDK, the gateway client | — | [docs/harness.md](docs/harness.md) |
| `[langgraph]` | the LangGraph adapter | `trellis-harness-langgraph` | [integrations/langgraph](integrations/langgraph/README.md) |
| `[deepagents]` | the Deep Agents adapter | `trellis-harness-deepagents` | [integrations/deepagents](integrations/deepagents/README.md) |
| `[openai-agents]` | the OpenAI Agents SDK adapter | `trellis-harness-openai-agents` | [integrations/openai_agents](integrations/openai_agents/README.md) |
| `[claude-agent-sdk]` | the Claude Agent SDK adapter | `trellis-harness-claude-agent-sdk` | [integrations/claude_agent_sdk](integrations/claude_agent_sdk/README.md) |
| `[agui]` | the AG-UI surface (FastAPI + SSE) | `trellis-harness-agui` | [integrations/agui](integrations/agui/README.md) |
| `[a2a]` | the A2A server and client | `trellis-harness-a2a` (`a2a-sdk`) | [docs/a2a.md](docs/a2a.md) |
| `[temporal]` | runs and schedules on Temporal, behind the same ports | `trellis-harness-temporal` (`temporalio`) | [integrations/temporal](integrations/temporal/README.md) |
| `[langfuse]` | Langfuse as an exporter on the harness's own spans | `langfuse>=3` | [docs/observability.md](docs/observability.md) |
| `[otel]` | the OTel **SDK** + OTLP exporter, when your app configures neither | `opentelemetry-sdk`, `-exporter-otlp-proto-http` | [docs/observability.md](docs/observability.md) |
| `[registry]` | validating a result against the `output_schema` an agent declared | `jsonschema` | [docs/registry.md](docs/registry.md) |
| `[logging]` | JSON structured logging | `structlog` | [docs/configuration.md](docs/configuration.md) |
| `[all]` | every row above | — | [docs/README.md](docs/README.md) |

Plain-Python users never receive a framework transitively: every adapter is a separate
distribution and the core imports none of them —
`tests/compatibility/test_matrix.py` proves it in a subprocess, and
`tests/unit/test_architecture.py` proves no trellis package imports a provider SDK at all.

## Frameworks

One core, four adapters, the same six moments (design §8): run start / context, the model
call, the tool call, the pause, the run end, compaction. A paused Deep Agents run, a paused
OpenAI Agents run and a paused Claude Agent SDK run are the same `Interrupt`, on the same
event stream, answered by the same `harness.resume` — so the AG-UI surface works over all of
them without knowing which framework ran.

| Framework | Attribute | Distribution | What it cannot express |
| --- | --- | --- | --- |
| LangGraph | `harness.langgraph` | `trellis-harness-langgraph` | — |
| Deep Agents | `harness.deepagents` | `trellis-harness-deepagents` | no post-summary compaction hook; the Memory Service names a note, not the model |
| OpenAI Agents SDK | `harness.openai_agents` | `trellis-harness-openai-agents` | SDK streaming; `Session.pop_item`; an approver cannot edit a call |
| Claude Agent SDK | `harness.claude_agent_sdk` | `trellis-harness-claude-agent-sdk` | no model client at all (it drives the `claude` CLI), so no per-call span; `PreCompact` carries no summary |

Every one of those is documented in the adapter's own README and recorded in
[COMPATIBILITY.md](COMPATIBILITY.md) and `compatibility-matrix.json`, with the reason. None of
them is faked.

Requires Python 3.12+.

### Versions actually exercised

Not a claim, a transcription: `pytest tests/compatibility` writes
[`compatibility-matrix.json`](compatibility-matrix.json) from the packages that run imported, and
this table is `make docs-compat` ([`tools/compat_table.py`](tools/compat_table.py)) pasted.

<!-- generated: make docs-compat -->
| Component | Version exercised |
| --- | --- |
| Python | 3.12.14 |
| trellis-harness | 0.3.0 |
| LangGraph | 1.2.11 |
| LangChain | 1.4.2 |
| LangChain core | 1.6.5 |
| Deep Agents | 0.7.19 |
| OpenAI Agents SDK | 0.22.3 |
| Claude Agent SDK | 0.2.160 |
| a2a-sdk (A2A protocol v1.0) | 1.1.5 |
| temporalio | 1.33.0 |
| FastAPI (AG-UI, A2A surfaces) | 0.141.1 |
| Langfuse | 4.15.2 |
| OpenTelemetry API | 1.44.0 |
| OpenTelemetry SDK | 1.44.0 |
| pydantic | 2.13.5 |
| trellis-memory (Memory Service SDK) | 0.2.1 |
| trellis-harness-langgraph | 0.2.0 |
| trellis-harness-agui | 0.1.0 |
| trellis-harness-a2a | 0.1.0 |
| trellis-harness-deepagents | 0.1.0 |
| trellis-harness-openai-agents | 0.1.0 |
| trellis-harness-claude-agent-sdk | 0.1.0 |
| trellis-harness-temporal | 0.1.0 |
<!-- /generated -->

A version outside that table is expected to work and is **not** verified here; the honest word
is "untested", not "supported". Per-capability rows, including what each framework cannot
express, are in [COMPATIBILITY.md](COMPATIBILITY.md).

## Your first agent

Ten lines, no services, and it runs as written — `pip install trellis-harness` is the only
prerequisite. You already have an identity, a span, a run record, a deadline and an event
stream; everything after this is adding providers.

```python
import asyncio

from trellis.harness import AgentHarness

harness = AgentHarness(defaults={"tenant_id": "acme"})


@harness.agent(agent_id="inventory-agent")
async def inventory(question: str, agent) -> str:
    agent.log("asked", question=question)
    return f"answering: {question}"


print(asyncio.run(inventory("how much stock of SKU-1?")).data)
```

## 5-minute quickstart

The same agent with memory attached, against a Memory Service on `localhost:8080`
(`examples/memory_quickstart.py` is this, runnable, including the tenant onboarding a
workspace-visible write needs):

```python
import asyncio

from trellis.harness import AgentExecutionContext, AgentHarness
from trellis.memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")
harness = AgentHarness(memory=memory, defaults={"tenant_id": "acme"})


async def inventory_agent(question: str) -> str:  # the agent you already have
    return f"answering: {question}"


wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent")

context = AgentExecutionContext.create(
    tenant_id="acme",
    agent_id="inventory-agent",
    user_id="u1",
    thread_id="chat-42",
)

result = asyncio.run(wrapped("how much stock of SKU-1?", context=context))
print(result.status, result.data)
```

Turning Langfuse on is configuration, not code:

```yaml
# harness.yaml
harness:
  observability:
    langfuse:
      enabled: true
```

```python
harness = AgentHarness(memory=memory, config="harness.yaml")  # business code unchanged
```

## Tutorial: six steps

Each step is small, and each one is optional — stop wherever it stops paying for itself.

### 1. Wrap what you have

```python
from trellis.harness import AgentHarness

harness = AgentHarness(defaults={"tenant_id": "acme"})


async def inventory_agent(question: str) -> str:  # your agent, untouched
    return "SKU-1 has 3 units left"


wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent")
result = await wrapped("how much stock?")

result.status  # SUCCESS
result.data  # "SKU-1 has 3 units left"  — exactly what your function returned
```

You already have: a trace span, execution metrics, a structured log line, a normalized
result, a deadline, and an error taxonomy if it throws.

### 2. Say who is asking

```python
from trellis.harness import AgentExecutionContext

context = AgentExecutionContext.create(
    tenant_id="acme",
    agent_id="inventory-agent",
    user_id="u-42",
    thread_id="chat-7",
    turn_id="turn-3",
)
result = await wrapped("how much stock?", context=context)
```

Now every span, log line, metric and memory write is attributed to that tenant, user and
conversation — and a nested agent inherits all of it automatically. This is also what makes
retries safe: ids derived from (thread, turn, agent) are stable across replays.

### 3. Take the runtime

Add a second parameter and your agent becomes "runtime-aware":

```python
@harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
async def inventory_agent(question, agent):
    agent.log("thinking", question_length=len(question))
    agent.check_cancelled()  # cooperative cancellation
    return AgentResponse.ok({"answer": "3 units"}, confidence=0.9)
```

`agent` is the [`AgentRuntime`](src/trellis/harness/runtime/agent_runtime.py):
`memory`, `memory_context`, `model`, `tools`, `artifacts`, `logger`, `tracer`,
`cancellation`, `deadline`.

### 4. Call tools through it

```python
async def inventory_db(sku: str) -> dict:
    """Stock for a SKU."""  # the docstring becomes the tool description
    return {"sku": sku, "on_hand": 3}


harness = AgentHarness(tools=[inventory_db], defaults={"tenant_id": "acme"})


@harness.agent(agent_id="inventory-agent")
async def inventory_agent(question, agent):
    stock = await agent.tools.call("inventory_db", sku="SKU-1")
    return f"{stock.output['on_hand']} units"
```

Each call now has its own span, latency, status, retry count and idempotency key — and the
argument *names* are recorded, not their values.

### 5. Give it memory

```python
harness = AgentHarness(
    memory=MemoryClient("http://memory-service:8080", api_key="..."), defaults={"tenant_id": "acme"}
)


@harness.agent(agent_id="inventory-agent")
async def inventory_agent(question, agent):
    bundle = agent.memory_context  # already fetched, before you were called
    await agent.memory.remember(
        "SKU-1 moves fast in Q4", memory_type="SEMANTIC", lifetime="LONG_TERM"
    )
    return AgentResponse.ok(
        "3 units",
        claims=[Claim(claim_id="c1", text="SKU-1 has 3 units")],
        memory_observations=[MemoryObservation(content="checked SKU-1 stock")],
    )
```

The harness fetched context before your agent ran and writes the observations after it
returns — so the answer is not waiting on the write. See
[Memory Service integration](#memory-service-integration) for the full surface.

### 6. Decide what happens when it breaks

```python
wrapped = harness.wrap(
    inventory_agent,
    agent_id="inventory-agent",
    timeout_seconds=5,  # deadline for the agent and everything it calls
    idempotent=True,  # makes it eligible for configured retries
    error_mode="result",  # return AgentResponse(status=ERROR) instead of raising
)
```

And turn on observability with configuration, not code:

```yaml
harness:
  observability:
    langfuse:
      enabled: true      # keys from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
```

That is the whole learning curve. Everything below is reference.

## Three integration modes

Runnable versions of all three: [`examples/plain_python.py`](examples/plain_python.py).

### Level 1 — wrap an existing agent

```python
wrapped = harness.wrap(existing_callable, agent_id="legacy-agent")
result = await wrapped(payload, context=context)
```

Sync callables stay sync, async stay async, callable objects work, and the agent's own
exception types still propagate (with a normalized `AgentError` attached as
`exc.agent_error`).

### Level 2 — runtime-aware agent (preferred)

```python
@harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
async def inventory_agent(state, agent):
    response = await agent.model.invoke("check stock for " + state["sku"])
    stock = await agent.tools.call("inventory_db", sku=state["sku"])
    return AgentResponse.ok({"answer": response.text, "stock": stock.output})
```

`agent` is an [`AgentRuntime`](src/trellis/harness/runtime/agent_runtime.py):
`memory`, `memory_context`, `model`, `tools`, `artifacts`, `tracer`, `logger`,
`cancellation`, `deadline`, `metadata`. Calls made through it are instrumented; calls made
around it are not (see [Limitations](docs/limitations.md)).

### Level 3 — instrument a block

```python
async with harness.execution(context, agent_id="report-agent", input=question) as runtime:
    bundle = runtime.memory_context
    answer = await existing_pipeline(question, bundle)
    runtime.state["result"] = AgentResponse.ok(answer)
```

## LangGraph

An existing node, unchanged:

```python
graph.add_node(
    "inventory",
    harness.langgraph.wrap_node(
        existing_node,
        agent_id="inventory-agent",
        query="question",
    ),
)
```

A runtime-aware node:

```python
@harness.langgraph.agent(agent_id="inventory-agent", query="question")
async def inventory_node(state, agent):
    response = await agent.model.invoke(state["question"])
    return {"answer": response.text}  # a normal LangGraph state update
```

The adapter derives identity from the `RunnableConfig` (`thread_id`, `checkpoint_ns`,
`langgraph_step`), so a replayed superstep produces the *same* agent run id and memory
writes deduplicate. It does not touch your graph topology, routing, reducers, checkpointer
or state schema. Application identity can travel in the config:

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

**One trace per turn.** Each node is its own agent run, and LangGraph runs supersteps as
separate tasks — so with nothing enclosing them, each node span starts its own trace. Wrap
the invocation to get a single trace (and a single Langfuse session) for the turn:

```python
async with harness.execution(context, agent_id="reorder-workflow", input=question):
    out = await app.ainvoke(state, config)
```

Without it the graph still works and every node is still instrumented; you just get one
trace per node instead of one per turn.

**Asking a human is not a failure.** `interrupt()` raises to hand control back to the graph
runtime, which saves the checkpoint and waits. The harness recognises that as a pause: the
span is `OK` with `status=PAUSED`, an `on_agent_pause` lifecycle event fires instead of
`on_agent_error`, and the exception is re-raised unchanged so the graph suspends exactly as
it would without the harness. `after` interceptors do not run — the turn is not over — and
they run on resume, when the node is re-entered and reaches its end.

```python
@harness.langgraph.agent(agent_id="approver")
async def approve(state, agent):
    decision = interrupt({"question": "approve the reorder?"})  # pauses here
    return {"answer": f"human said {decision}"}


await app.ainvoke(Command(resume="approved"), config)  # resumes, finishes normally
```

Small example: [`examples/langgraph_agent.py`](examples/langgraph_agent.py). A full
multi-agent workflow — fan-out/fan-in, a nested sub-agent, tools, model calls, business
logic, artifacts and memory writes — is in
[`examples/reorder_workflow.py`](examples/reorder_workflow.py).

## Tools and models

**Tools.** Register them with the harness, call them through the runtime, or wrap the
callable and keep calling it the way you already do:

```python
async def inventory_db(sku: str) -> dict:
    """Stock levels for a SKU."""  # the docstring becomes the tool description
    ...


harness = AgentHarness(memory=memory, tools=[inventory_db])


@harness.agent(agent_id="inventory-agent")
async def agent(state, runtime):
    outcome = await runtime.tools.call("inventory_db", sku=state["sku"])
    return outcome.output  # .status, .latency_ms, .cached, .artifacts


@harness.wrap_tool  # or: instrument a tool you call directly
async def pricing_api(sku: str) -> float: ...  # returns the tool's own value, instrumented
```

**Models.** Anything with `ainvoke`/`invoke` (or a plain callable) can be adapted; the
harness never imports a provider SDK:

```python
from trellis.harness import DirectModelClient

harness = AgentHarness(memory=memory, model=my_provider_client)

# or adapt one explicitly, naming the provider and default model for the spans
harness = AgentHarness(
    memory=memory,
    model=DirectModelClient(client, provider="acme-ai", model="gpt-x"),
)


@harness.agent(agent_id="answer-agent")
async def answer(state, runtime):
    response = await runtime.model.invoke(state["question"])
    response.text, response.usage.input_tokens, response.usage.cost_usd
    async for chunk in runtime.model.stream(state["question"]):  # streaming is instrumented
        ...
    return response.text
```

Instrumentation priority, highest first (§13/§58/§59 of the design):

1. `runtime.tools` / `runtime.model` — full instrumentation;
2. `harness.wrap_tool()` / `harness.wrap_model()` — full instrumentation wherever called
   inside a harness execution;
3. framework callbacks/events — whatever the framework exposes publicly;
4. OpenTelemetry auto-instrumentation — if you install it;
5. an un-instrumented call, **documented as such**. The harness never claims to intercept
   arbitrary Python calls it was not given.

Captured per model call: provider, model, profile, prompt id/version, input/output tokens,
cost, latency, tool calls, fallback flag, streaming time-to-first-token — *where the
provider reports them*. Captured per tool call: tool, version, argument **names**, start,
end, latency, status, retry, result reference, artifact reference, error, span, run.

## Memory Service integration

Before an execution the harness fetches a `ContextBundle` for the request's query and hands
it to the agent as `runtime.memory_context`. After the result is produced it writes
observations — asynchronously, so the turn never waits for consolidation, KG updates,
summaries or evaluation. Every write carries a deterministic idempotency key derived from
(tenant, thread, turn, task, agent, run, content), so retries and checkpoint replays
deduplicate.

It also labels the run success or failure when the turn ends — on both paths, including one
the harness re-raised. Tool memory learns procedures from runs it knows succeeded; without a
label the service has to wait hours before counting one as a weak positive, and never learns
anything from a failure.

```python
harness = AgentHarness(
    memory=memory,
    config={
        "memory": {
            "retrieve_before": True,
            "observe_input": True,
            "observe_output": True,
            "observe_tool_results": False,
            "observe_claims": True,
            "private_by_default": False,
            "record_outcome": True,
        }
    },
)
```

Per-agent overrides: `harness.wrap(agent, memory_policy=MemoryPolicy(observe_output=False))`.

### Driving memory yourself

That automatic path covers the common turn. Everything else the service can do is on
`runtime.memory`, instrumented the same way — each call gets its own `agent.memory.*` span,
its own timeout and a deterministic idempotency key:

| Push | | Get | |
| --- | --- | --- | --- |
| `record_input` / `record_output` | the conversation turns | `retrieve(query)` | **the 90% call**: one bounded, ranked, evidence-gated bundle — conversation window + memories + RAG knowledge + graph facts + summaries |
| `observe(MemoryObservation(...))` | something happened (episodic) | `recall(query)` | ranked evidence items, without bundle assembly |
| `remember(text, memory_type=…, lifetime=…, visibility=…)` | a typed, durable memory | `history(limit=…)` | the conversation window on its own |
| `add_document(path)` | ingest a document into the RAG corpus | `graph_query(q, hops=…, as_of=…)` | knowledge-graph entities and relationships, optionally as of a time |
| `share(text)` | publish to the agent group (declare the group once on the harness, per agent, or per call) | `memories(memory_types=…)` | the inventory view: what is held for this scope |
| `forget(memory_id)` | delete a memory | `verify(answer, bundle=…)` | grounding report: is this answer supported, claim by claim |

The vocabulary is the service's: `memory_type` (SEMANTIC, EPISODIC, PROCEDURAL, PREFERENCE,
DECISION, OUTCOME, FAILURE, SHARED…), `lifetime` (EPHEMERAL, SHORT_TERM, LONG_TERM,
ARCHIVAL), `visibility` (PRIVATE, RUN, AGENT_GROUP, THREAD, USER, WORK, WORKSPACE, TENANT).

```python
@harness.agent(agent_id="inventory-agent")
async def agent(state, runtime):
    await runtime.memory.remember(
        "SKU-1 reorders from Castor Supply below 10 days of cover",
        memory_type="SEMANTIC",
        lifetime="LONG_TERM",
        visibility="WORKSPACE",
    )
    bundle = await runtime.memory.retrieve(state["question"])  # everything, in one call
    facts = await runtime.memory.graph_query("who supplies SKU-1?", hops=2)
    report = await runtime.memory.verify(answer, bundle=bundle)  # grounded, or not
    ...
```

Anything the harness does not wrap is one attribute away — `runtime.memory.sdk` is the bound
SDK context (and `.chat`, `.files`, `.graph`, `.tools`). Those calls work; they are simply
not traced by the harness, and that is stated rather than implied.

A runnable walk through every one of these:
[`examples/memory_tour.py`](examples/memory_tour.py).

## Results and errors

Whatever your agent returns is coerced into an `AgentResponse`; returning one yourself gives
you the richer fields.

```python
result = await wrapped(payload, context=context)

result.status  # SUCCESS | PARTIAL | ERROR | TIMEOUT | CANCELLED | REJECTED
result.data  # your agent's own return value
result.claims  # Claim(claim_id, text, evidence_ids, confidence)
result.evidence  # EvidenceRef(source_type, source_id, document_id, page, citation)
result.artifacts  # ArtifactRef(artifact_id, type, uri, checksum, size_bytes)
result.recommended_actions  # RecommendedAction(action_type, description, reason_summary)
result.memory_observations  # what should be remembered (written after the turn)
result.warnings  # e.g. MEMORY_DEGRADED, MEMORY_WRITE_FAILED, RESULT_OFFLOADED
result.metrics  # model_calls, tool_calls, total_tokens, cost_usd
result.confidence
result.error  # AgentError | None
```

`AgentRequest` and `AgentResponse` are plain Pydantic models with no framework types in them,
so they serialize cleanly for queues, storage or a future A2A transport.

**Errors keep their own type.** By default the harness re-raises your exception unchanged
and attaches the normalized classification to it:

```python
try:
    await wrapped(payload, context=context)
except StaleFeedError as exc:  # your exception, not ours
    exc.agent_error.category  # VALIDATION | AUTHORIZATION | MODEL | TOOL | MEMORY |
    # TIMEOUT | CANCELLED | RATE_LIMIT | DEPENDENCY | POLICY | UNKNOWN
    exc.agent_error.retryable  # False for UNKNOWN — the harness never guesses
    exc.agent_error.trace_id
```

Prefer a result over an exception? `error_mode="result"`:

```python
wrapped = harness.wrap(agent, agent_id="inv", error_mode="result")
result = await wrapped(payload, context=context)
if not result.succeeded:
    log.warning("agent failed", code=result.error.code, category=result.error.category)
```

Cancellation is never converted: `asyncio.CancelledError` always propagates.

## Timeouts, cancellation, retries, idempotency

```python
wrapped = harness.wrap(agent, agent_id="inv", timeout_seconds=5, idempotent=True)
```

* **Deadlines are hierarchical.** The execution deadline bounds every memory, model and
  tool call inside it, and a nested agent never outlives its parent. A breach raises
  `AgentTimeoutError` (status `TIMEOUT`, `retryable=True`).
* **Cancellation propagates** into the agent's task; long agents should cooperate:

  ```python
  @harness.agent(agent_id="long-agent")
  async def long_agent(state, agent):
      for item in state["items"]:
          agent.check_cancelled()  # raises CancelledError promptly
          await process(item)
  ```
* **Retries are opt-in twice over**: `retries.enabled` in configuration *and*
  `idempotent=True` on the agent. Only `TIMEOUT`, `RATE_LIMIT` and `DEPENDENCY` are
  retried; authorization, validation, policy denials and cancellation never are.
* **Idempotency keys** are derived from (tenant, thread, turn, task, agent, run, content),
  not from a clock or a uuid — so a framework replay produces the same key and memory
  writes, artifacts and tool calls deduplicate instead of doubling.

## Observability

OpenTelemetry is the contract; Langfuse is an optional backend on the **same spans**, not a
parallel tracing model.

```
agent.run
├── agent.memory.retrieve
├── agent.model.invoke
├── agent.tool.call
└── agent.memory.observe
```

Langfuse mapping: OTel trace → Langfuse trace, thread → session, user → user (only when
capture allows), agent run → observation, model → generation, tool → tool, memory retrieval
→ retriever, skills → tags.

Nothing sensitive is exported by default: raw prompts, model inputs/outputs, tool
inputs/outputs, memory content and user ids are all **off** until switched on per
tenant/environment. See [Privacy and redaction](docs/privacy.md).

Langfuse being unavailable never fails a business execution in the default
`failure_mode: non_blocking`.

### Two backends, one request: the split

Agents and services are asked different questions, so they are answered in different places —
and the two are joined, never merged.

| | Langfuse | Datadog (OTLP) |
| --- | --- | --- |
| Scope | **per agent**: one trace per request, with its spans, cost, judge score, human score and feedback thread | **per service**: the Memory Service, `agent-runs`, `agent-schedules` — latency, saturation, errors, dependencies |
| Answers | "why did this answer come out like that?" | "is the platform healthy, and what is it costing?" |
| Fed by | the harness's own spans, with Langfuse's documented OTel attributes | the OTel collector, from the same spans plus each service's metrics |
| Never | a Langfuse project for a non-agent service | a second span tree for agents |

They are joined on two ids that travel on every hop: **`traceparent`** (W3C trace context, so a
Datadog service span and a Langfuse agent trace share a trace id) and **`X-Request-ID`** (so a
log line and a trace name the same request). `GET /v1/reads` on the Memory Service is the
separate record of *what memory was served* — by principal, kind and record ids, kept beyond a
trace's retention.

The collector configuration, the per-service environment variables and what a trace looks like
when you open it are in [docs/observability.md](docs/observability.md).

## Langfuse setup

```bash
pip install "trellis-harness[langfuse]"

export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
export UAH_LANGFUSE_ENABLED=true
```

or in the config file:

```yaml
harness:
  observability:
    langfuse:
      enabled: true
      mode: auto            # auto | sdk | otlp
      environment: staging
  telemetry:
    sampling:
      sample_rate: 0.1        # 10% of executions...
      error_sample_rate: 1.0  # ...but every failure
    capture:
      inputs: false           # opt in per tenant/environment
```

No agent code changes. Keys are validated at startup: enabling Langfuse without them fails
immediately rather than silently doing nothing.

Modes: `sdk` uses the installed Langfuse SDK; `otlp` needs no SDK at all (the harness sets
Langfuse's documented OTel attributes and you point an OTLP exporter at Langfuse); `auto`
(default) picks `sdk` when importable and falls back to `otlp`. Either way there is **one**
span tree — Langfuse is an exporter on the harness's OpenTelemetry spans, not a second
tracing system.

In a short-lived process, flush before exit:

```python
await harness.aclose()  # drains memory writeback and flushes telemetry
```

Scores (from DeepEval, an LLM judge, or a human) go back onto the trace:

```python
await harness.evaluation_provider.score(
    "groundedness", 0.93, trace_id=context.trace_id, comment="deepeval"
)
```

## Extending the harness

```python
from trellis.harness import BaseInterceptor, Order


class AuditInterceptor(BaseInterceptor):
    name = "audit"
    order = Order.USER  # runs after the core before-chain

    async def before(self, request, runtime):
        runtime.logger.info("audit.start", agent_id=runtime.agent_id)
        return request

    async def after(self, result, runtime):
        return result.add_warning("AUDITED", "reviewed by the audit interceptor")


harness = AgentHarness(
    memory=memory,
    interceptors=[AuditInterceptor()],  # or harness.add_interceptor(...)
    listeners=[lambda event, payload: metrics.count(event)],  # or harness.on(...)
    policy=CallablePolicyProvider(tool=lambda ctx, call: call.tool != "rm"),
    evaluation_sink=my_sink,
    registry=my_registry,
    redactor=MyRedactor(),
)
```

Lifecycle events: `on_agent_start`, `on_context_loaded`, `on_model_start`/`on_model_end`,
`on_tool_start`/`on_tool_end`, `on_agent_success`, `on_agent_error`, `on_agent_cancel`,
`on_agent_pause`, `on_agent_timeout`, `on_agent_finish` (the `LifecycleEvent` enum, in full). A
listener that raises is logged and swallowed — an observer can never fail a business execution.
Interceptors, which *can* change a run, are [docs/interceptors.md](docs/interceptors.md).

Agents describe themselves for a future registry (no-op by default):

```python
harness.describe("inventory-agent", skills=["inventory.analysis"], version="1.2.0")
await harness.register_agents()
```

## Status

| Area | State |
| --- | --- |
| Plain Python | supported, tested |
| LangGraph (1.2.x) | supported, tested — see [COMPATIBILITY.md](COMPATIBILITY.md) |
| Deep Agents (0.7.x) | supported, tested — [`integrations/deepagents`](integrations/deepagents/README.md) |
| OpenAI Agents SDK (0.22.x) | supported, tested (non-streaming) — [`integrations/openai_agents`](integrations/openai_agents/README.md) |
| Claude Agent SDK (0.2.x) | supported, tested — [`integrations/claude_agent_sdk`](integrations/claude_agent_sdk/README.md) |
| Memory Service integration | supported, tested against the real SDK |
| OpenTelemetry | supported, tested |
| Langfuse (3.x/4.x API) | supported, tested against 4.15.2 |
| CrewAI, Google ADK | **not implemented.** Their callables work as plain Python, but framework-level lineage, events and state mapping do not |
| Bifrost LLM gateway | supported, tested against a real gateway process: retries, circuit breaker, streaming, structured output, tool calls |
| MCP tools (Bifrost `/mcp`) | supported, tested — `MCPToolClient` lists and runs the gateway's tools under the names it gives them |
| A2A (protocol v1.0, `a2a-sdk` 1.1.5) | supported, tested — serve any agent (`trellis-harness[a2a]`), call registry agents as tools; JSON-RPC verified, gRPC and card signing not installed. See [docs/a2a.md](docs/a2a.md) |
| AI Registry | supported, tested against a scripted registry: manifest discovery, agent directory, Agent Card write-back, heartbeat and delta sync, Bifrost MCP clients configured from tool entities. No registry *service* is shipped here |

What the harness deliberately cannot do — and says so rather than implying otherwise — is
in [docs/limitations.md](docs/limitations.md). The short version: it cannot instrument an
arbitrary unwrapped library call it never sees.

## Configuration

Full reference: [docs/configuration.md](docs/configuration.md); a commented starting point:
[`harness.example.yaml`](harness.example.yaml). Environment variables (`UAH_*`,
`LANGFUSE_*`) are documented there too. Configuration is validated at construction, so a
mistake fails at startup rather than on the first execution.

## Documentation

[**docs/README.md**](docs/README.md) is the map: every page, and the question it answers. The
short version:

| Document | What is in it |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layering, ports and adapters, interceptor pipeline, execution flow |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Versions actually tested, per-feature matrix, degradation rules |
| [docs/harness.md](docs/harness.md) | `AgentHarness`: what you pass it, the three ways to attach, the runtime, the lifecycle events |
| [docs/configuration.md](docs/configuration.md) | Every setting, every environment variable |
| [docs/memory.md](docs/memory.md) | The memory runtime, visibility, the policy, and a live example |
| [docs/tools.md](docs/tools.md) | Local, MCP, memory and agent tools behind one policy surface |
| [docs/models.md](docs/models.md) | Calling a model through the gateway; which models the platform uses, with origins and licences |
| [docs/reasoning.md](docs/reasoning.md) | The bounded ReAct loop, the prompt budget, compaction |
| [docs/events.md](docs/events.md) | The `RunEvent` stream and the sinks in the box |
| [docs/interrupts.md](docs/interrupts.md) | Pausing for a person, and what the answer does |
| [docs/runs.md](docs/runs.md) | Durable run records, the paused inbox, `agent-runs` and Temporal |
| [docs/registry.md](docs/registry.md) | Discovery, heartbeat, delta sync, MCP clients from registry tools |
| [docs/artifacts.md](docs/artifacts.md) | Artifacts, claims and evidence |
| [docs/interceptors.md](docs/interceptors.md) | Extending the pipeline; listeners; policy providers; redaction |
| [docs/a2a.md](docs/a2a.md) | Serving an agent over A2A, calling registry agents as tools, Agent Cards, registry sync |
| [docs/evaluation.md](docs/evaluation.md) | The online judge, offline datasets and experiments, the CI regression gate |
| [docs/observability.md](docs/observability.md) | One trace per request, what memory was served, and the Langfuse/Datadog split with collector configuration |
| [docs/privacy.md](docs/privacy.md) | Capture policy, redaction, sampling, what never leaves |
| [docs/limitations.md](docs/limitations.md) | What the harness cannot do, stated plainly |
| [docs/performance.md](docs/performance.md) | Measured overhead and how to reproduce it |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms, causes, fixes |
| [examples/README.md](examples/README.md) | Every example, what it demonstrates, and what it needs to run |

## Performance

Harness-only overhead, measured on the development machine (see
[docs/performance.md](docs/performance.md) to reproduce; numbers are machine-specific):

| Configuration | p50 | p95 |
| --- | --- | --- |
| Telemetry enabled (spans exported in-process) | 0.66 – 0.71 ms | 1.06 – 1.91 ms |
| Telemetry disabled | 0.36 – 0.42 ms | 0.59 – 1.07 ms |
| Sampled out | 0.36 – 0.42 ms | 0.85 – 1.05 ms |

Ranges are three runs on a laptop, not a guarantee. The budget is p50 < 1 ms, p95 < 5 ms.

## Development

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --all-extras

make test          # the hermetic suite: unit, contract, integration, e2e, compatibility
make check         # ruff + pyright + tests — what a release gate runs
make bench         # overhead benchmark (writes benchmark-results.json)
make examples      # run every example end to end
make test-live     # the same paths against a *running* Memory Service
```

`make test-live` is the one suite that mocks nothing. Start the service first (in the
`agent-memory-service` checkout: `make dev-up`), then:

```bash
MEMORY_SERVICE_URL=http://localhost:8080 MEMORY_API_KEY=dev-key make test-live
```

It drives the real SDK against the real service. `make test-live-full` goes further and
checks the **database**: every memory type and visibility level written and read back out of
Postgres, a document ingested and retrieved as knowledge, graph entities and relations
created from the harness's own observations, conversation/session/turn rows, tool
invocations, agent-run lineage, a replayed write producing exactly one row, and a LangGraph
graph running against it all. Without `MEMORY_SERVICE_URL` both suites skip, so the normal
run stays hermetic.

Two service behaviours explain most surprises, and no setting changes them:

* **writes are asynchronous** — the API commits and queues; reading immediately after
  writing proves nothing (drain, or poll);
* **reads are audience-filtered** — a memory or chunk is retrievable only by a principal in
  its audience. A THREAD audience needs the thread to exist (a message creates it);
  WORKSPACE and GROUP audiences need membership. The harness refuses a write whose audience
  the context cannot express, because the service would accept it and fail the background
  job that creates the memory.

Test suite, as measured by
`pytest tests integrations/*/tests -q -p no:randomly --timeout=300`: **870 tests — 867 passed,
3 skipped** (the skips are the live-model paths, which skip when the Memory Service or the
gateway is not reachable). Categories: unit, contract (protocol conformance), integration,
end-to-end (including the real Memory Service SDK over a mocked HTTP layer), failure
injection, observability, compatibility and performance. Coverage is not quoted here because
this README does not carry a number nobody can reproduce: `make coverage` prints it.

Two tests are **load-sensitive** and will fail on a busy machine, which is worth knowing before
you read a red run as a regression: `test_parallel_nodes_run_concurrently_and_merge` asserts a
wall-clock budget (`elapsed < 0.09` for two 50 ms nodes in parallel) and measured
0.095–0.114 s while the host was running other work, then passed once it was quiet. The
threshold is right; the machine was busy.

Fallback without uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"
```

## Events, interrupts and surfaces (0.3.0)

Every run emits an ordered `RunEvent` stream to the sinks you pass
(`AgentHarness(event_sinks=[CollectingEventSink(), WebhookEventSink(url, secret=...)])`);
see [docs/events.md](docs/events.md). A run pauses the same way whatever asked, a person's
answer continues the same run, and approvals become feedback; see
[docs/interrupts.md](docs/interrupts.md). Tools: `MCPToolClient(gateway_or_model_client)`
lists and runs the gateway's MCP tools under the names Bifrost gives them,
`CompositeToolClient` puts several clients behind one port, and memory can be offered to the
model as `memory.recall` / `memory.remember` (`memory.as_tools: true`, opt-in). `react()`
runs every tool call of a step and takes a `ContextAssembler` that keeps the prompt under
budget and compacts older turns into a remembered summary. The AG-UI surface is the
`trellis-harness-agui` distribution (`pip install "trellis-harness[agui]"`):

```python
from trellis.harness_agui import agui_router

app.include_router(agui_router(harness, agent=refund_agent, agent_id="refund-agent"))
```

Agents talk to agents over A2A, through the `trellis-harness-a2a` distribution
(`pip install "trellis-harness[a2a]"`, protocol v1.0 on `a2a-sdk` 1.1.5). Serving publishes an Agent
Card generated from the Registry entity and the `AgentDescriptor`, streams the run's events as task
updates, turns a pause into `input-required` (the next message on the same task resumes the same
run), and signs push notifications with the harness's webhook rules. Calling makes every agent the
Registry lists a tool the planner can choose, next to local and MCP tools:

```python
from trellis.harness.registry import RegistryAgentDirectory, RegistrySync
from trellis.harness_a2a import A2AAgentClient, A2AServer, TrustedHeaderIdentity

server = await A2AServer.from_registry(
    harness,
    agent=refund_agent,
    url="https://agents.example.com/a2a",
    registry=harness.registry,
    identity=TrustedHeaderIdentity(allowed_tenants={"acme"}),
)
app.mount("/", server.app())  # card + JSON-RPC, at the published URL

agents = A2AAgentClient(RegistryAgentDirectory(harness.registry), credentials=team_keys)
sync = RegistrySync(harness.registry, descriptors=[...], gateway=bifrost)  # heartbeat, deltas, MCP
```

Identity on an A2A call is the platform's, never the caller's: the tenant, user and workspace travel
on a trusted header an authenticating edge sets, a foreign one is refused before a run starts, and
only the run's own caller can answer its pause. Details, including the full event mapping and the
registry sync: [docs/a2a.md](docs/a2a.md).


## Evaluation and durability (0.3.0)

**Is this agent any good?** An online judge scores sampled turns, asynchronously, and
**grounded first**: the Memory Service's `/v1/verify` (deterministic citation validation and NLI
against the bundle the run was actually given) decides everything a classifier can decide, and
only what it cannot settle reaches a model — through Bifrost, on a cheap model, with the rubric
stored and versioned in the gateway by prompt id. A verdict becomes a Langfuse score, an
`agent.judge` span and a `Feedback` record with `source="judge"`. Off by default; per-agent rate,
per-hour count and per-hour dollar ceilings.

```python
from trellis.harness import AgentHarness, BifrostModelClient, GroundedJudge

judge = GroundedJudge(model=BifrostModelClient(gateway, api_key=budgeted_key), config=cfg.judge)
harness = AgentHarness(memory=memory, judge=judge)  # the turn never waits for the verdict
```

**Is it getting better?** `DatasetBuilder` turns run records plus human corrections into a
dataset (judge feedback excluded, so the judge cannot become its own ground truth),
`ExperimentRunner` replays it against a candidate and scores it with the *same* `Judge`, and one
command gates CI on both halves:

```bash
python -m trellis.harness.evaluation.gate \
  --baseline benchmark-results.json --current build/benchmark-results.json \
  --judge build/experiment.json --max-latency-regression 20 --max-score-drop 0.05
```

See [docs/evaluation.md](docs/evaluation.md). **Where do I look when one request went wrong?**
One Langfuse trace per request holds the spans, the cost, the judge's score, the human's score
and the feedback thread; `GET /v1/reads` says what memory was served; Langfuse is per agent and
Datadog (OTLP) is per service, joined on `traceparent` + `x-request-id` — with the collector
configuration in [docs/observability.md](docs/observability.md).

**Durability.** A deployment that runs Temporal puts its runs there by configuration —
a workflow per run, signals for pause and resume, Temporal Schedules for standing intents, both
behind the same `RunStore` and `Scheduler` ports as `agent-runs`
(`pip install "trellis-harness[temporal]"`):

```yaml
runs:
  engine: temporal       # or agent_runs, the default
  temporal: { target: localhost:7233, task_queue: trellis-runs }
```

See [integrations/temporal/README.md](integrations/temporal/README.md).

## License

Apache-2.0.
