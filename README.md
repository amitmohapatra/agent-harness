# Universal Agent Harness

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
[Install](#install) · [Quickstart](#5-minute-quickstart) · [Tutorial](#tutorial-six-steps) ·
[Integration modes](#three-integration-modes) · [LangGraph](#langgraph) ·
[Tools and models](#tools-and-models) · [Memory](#memory-service-integration) ·
[Results and errors](#results-and-errors) ·
[Timeouts, cancellation, retries](#timeouts-cancellation-retries-idempotency) ·
[Observability](#observability) · [Langfuse setup](#langfuse-setup) ·
[Extending](#extending-the-harness) · [Configuration](#configuration) ·
[Status](#status) · [Docs](#documentation) · [Performance](#performance)

---

## Install

```bash
pip install universal-agent-harness                        # core: plain Python
pip install "universal-agent-harness[langgraph]"           # + LangGraph adapter
pip install "universal-agent-harness[langfuse]"            # + Langfuse observability
pip install "universal-agent-harness[otel]"                # + OTel SDK & OTLP exporter
pip install "universal-agent-harness[langgraph,langfuse]"  # combined
```

Plain-Python users never receive LangGraph transitively: the adapter is a separate
distribution (`universal-agent-harness-langgraph`) and the core imports no framework.

Requires Python 3.12+.

## 5-minute quickstart

```python
import asyncio
from universal_agent_harness import AgentExecutionContext, AgentHarness
from universal_memory import MemoryClient

memory = MemoryClient("http://memory-service:8080", api_key="...")
harness = AgentHarness(memory=memory, defaults={"tenant_id": "acme"})


async def inventory_agent(question: str) -> str:      # the agent you already have
    return f"answering: {question}"


wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent")

context = AgentExecutionContext.create(
    tenant_id="acme", agent_id="inventory-agent", user_id="u1", thread_id="chat-42",
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
harness = AgentHarness(memory=memory, config="harness.yaml")   # business code unchanged
```

## Tutorial: six steps

Each step is small, and each one is optional — stop wherever it stops paying for itself.

### 1. Wrap what you have

```python
from universal_agent_harness import AgentHarness

harness = AgentHarness(defaults={"tenant_id": "acme"})

async def inventory_agent(question: str) -> str:      # your agent, untouched
    return "SKU-1 has 3 units left"

wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent")
result = await wrapped("how much stock?")

result.status   # SUCCESS
result.data     # "SKU-1 has 3 units left"  — exactly what your function returned
```

You already have: a trace span, execution metrics, a structured log line, a normalized
result, a deadline, and an error taxonomy if it throws.

### 2. Say who is asking

```python
from universal_agent_harness import AgentExecutionContext

context = AgentExecutionContext.create(
    tenant_id="acme", agent_id="inventory-agent",
    user_id="u-42", thread_id="chat-7", turn_id="turn-3",
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
    agent.check_cancelled()                  # cooperative cancellation
    return AgentResult.ok({"answer": "3 units"}, confidence=0.9)
```

`agent` is the [`AgentRuntime`](src/universal_agent_harness/runtime/agent_runtime.py):
`memory`, `memory_context`, `model`, `tools`, `artifacts`, `logger`, `tracer`,
`cancellation`, `deadline`.

### 4. Call tools through it

```python
async def inventory_db(sku: str) -> dict:
    """Stock for a SKU."""                    # the docstring becomes the tool description
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
harness = AgentHarness(memory=MemoryClient("http://memory-service:8080", api_key="..."),
                       defaults={"tenant_id": "acme"})

@harness.agent(agent_id="inventory-agent")
async def inventory_agent(question, agent):
    bundle = agent.memory_context              # already fetched, before you were called
    await agent.memory.remember("SKU-1 moves fast in Q4",
                                memory_type="SEMANTIC", lifetime="LONG_TERM")
    return AgentResult.ok(
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
    timeout_seconds=5,        # deadline for the agent and everything it calls
    idempotent=True,          # makes it eligible for configured retries
    error_mode="result",      # return AgentResult(status=ERROR) instead of raising
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
    return AgentResult.ok({"answer": response.text, "stock": stock.output})
```

`agent` is an [`AgentRuntime`](src/universal_agent_harness/runtime/agent_runtime.py):
`memory`, `memory_context`, `model`, `tools`, `artifacts`, `tracer`, `logger`,
`cancellation`, `deadline`, `metadata`. Calls made through it are instrumented; calls made
around it are not (see [Limitations](docs/limitations.md)).

### Level 3 — instrument a block

```python
async with harness.execution(context, agent_id="report-agent", input=question) as runtime:
    bundle = runtime.memory_context
    answer = await existing_pipeline(question, bundle)
    runtime.state["result"] = AgentResult.ok(answer)
```

## LangGraph

An existing node, unchanged:

```python
graph.add_node("inventory", harness.langgraph.wrap_node(
    existing_node, agent_id="inventory-agent", query="question",
))
```

A runtime-aware node:

```python
@harness.langgraph.agent(agent_id="inventory-agent", query="question")
async def inventory_node(state, agent):
    response = await agent.model.invoke(state["question"])
    return {"answer": response.text}          # a normal LangGraph state update
```

The adapter derives identity from the `RunnableConfig` (`thread_id`, `checkpoint_ns`,
`langgraph_step`), so a replayed superstep produces the *same* agent run id and memory
writes deduplicate. It does not touch your graph topology, routing, reducers, checkpointer
or state schema. Application identity can travel in the config:

```python
await app.ainvoke(state, {"configurable": {
    "thread_id": "chat-42",
    "harness": {"tenant_id": "acme", "user_id": "u1", "work_id": "wo-9"},
}})
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

Small example: [`examples/langgraph_agent.py`](examples/langgraph_agent.py). A full
multi-agent workflow — fan-out/fan-in, a nested sub-agent, tools, model calls, business
logic, artifacts and memory writes — is in
[`examples/reorder_workflow.py`](examples/reorder_workflow.py).

## Tools and models

**Tools.** Register them with the harness, call them through the runtime, or wrap the
callable and keep calling it the way you already do:

```python
async def inventory_db(sku: str) -> dict:
    """Stock levels for a SKU."""            # the docstring becomes the tool description
    ...

harness = AgentHarness(memory=memory, tools=[inventory_db])

@harness.agent(agent_id="inventory-agent")
async def agent(state, runtime):
    outcome = await runtime.tools.call("inventory_db", sku=state["sku"])
    return outcome.output                     # .status, .latency_ms, .cached, .artifacts

@harness.wrap_tool                            # or: instrument a tool you call directly
async def pricing_api(sku: str) -> float:
    ...                                       # returns the tool's own value, instrumented
```

**Models.** Anything with `ainvoke`/`invoke` (or a plain callable) can be adapted; the
harness never imports a provider SDK:

```python
from universal_agent_harness import DirectModelClient

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
    async for chunk in runtime.model.stream(state["question"]):   # streaming is instrumented
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

```python
harness = AgentHarness(memory=memory, config={"memory": {
    "retrieve_before": True, "observe_input": True, "observe_output": True,
    "observe_tool_results": False, "observe_claims": True, "private_by_default": False,
}})
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
        memory_type="SEMANTIC", lifetime="LONG_TERM", visibility="WORKSPACE",
    )
    bundle = await runtime.memory.retrieve(state["question"])   # everything, in one call
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

Whatever your agent returns is coerced into an `AgentResult`; returning one yourself gives
you the richer fields.

```python
result = await wrapped(payload, context=context)

result.status              # SUCCESS | PARTIAL | ERROR | TIMEOUT | CANCELLED | REJECTED
result.data                # your agent's own return value
result.claims              # Claim(claim_id, text, evidence_ids, confidence)
result.evidence            # EvidenceRef(source_type, source_id, document_id, page, citation)
result.artifacts           # ArtifactRef(artifact_id, type, uri, checksum, size_bytes)
result.recommended_actions # RecommendedAction(action_type, description, reason_summary)
result.memory_observations # what should be remembered (written after the turn)
result.warnings            # e.g. MEMORY_DEGRADED, MEMORY_WRITE_FAILED, RESULT_OFFLOADED
result.metrics             # model_calls, tool_calls, total_tokens, cost_usd
result.confidence
result.error               # AgentError | None
```

`AgentRequest` and `AgentResult` are plain Pydantic models with no framework types in them,
so they serialize cleanly for queues, storage or a future A2A transport.

**Errors keep their own type.** By default the harness re-raises your exception unchanged
and attaches the normalized classification to it:

```python
try:
    await wrapped(payload, context=context)
except StaleFeedError as exc:            # your exception, not ours
    exc.agent_error.category   # VALIDATION | AUTHORIZATION | MODEL | TOOL | MEMORY |
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
          agent.check_cancelled()        # raises CancelledError promptly
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

## Langfuse setup

```bash
pip install "universal-agent-harness[langfuse]"

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
await harness.aclose()     # drains memory writeback and flushes telemetry
```

Scores (from DeepEval, an LLM judge, or a human) go back onto the trace:

```python
await harness.evaluation_provider.score("groundedness", 0.93,
                                        trace_id=context.trace_id, comment="deepeval")
```

## Extending the harness

```python
from universal_agent_harness import BaseInterceptor, Order

class AuditInterceptor(BaseInterceptor):
    name = "audit"
    order = Order.USER                      # runs after the core before-chain

    async def before(self, request, runtime):
        runtime.logger.info("audit.start", agent_id=runtime.agent_id)
        return request

    async def after(self, result, runtime):
        return result.add_warning("AUDITED", "reviewed by the audit interceptor")

harness = AgentHarness(
    memory=memory,
    interceptors=[AuditInterceptor()],                       # or harness.add_interceptor(...)
    listeners=[lambda event, payload: metrics.count(event)],  # or harness.on(...)
    policy=CallablePolicyProvider(tool=lambda ctx, call: call.tool != "rm"),
    evaluation_sink=my_sink,
    registry=my_registry,
    redactor=MyRedactor(),
)
```

Lifecycle events: `on_agent_start`, `on_context_loaded`, `on_model_start`/`on_model_end`,
`on_tool_start`/`on_tool_end`, `on_agent_success`, `on_agent_error`, `on_agent_cancel`,
`on_agent_timeout`, `on_agent_finish`. A listener that raises is logged and swallowed — an
observer can never fail a business execution.

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
| Memory Service integration | supported, tested against the real SDK |
| OpenTelemetry | supported, tested |
| Langfuse (3.x/4.x API) | supported, tested against 4.15.2 |
| CrewAI, Google ADK | **not implemented.** Their callables work as plain Python, but framework-level lineage, events and state mapping do not |
| Agent registry service, Bifrost, MCP, A2A | **not implemented.** The ports and serializable contracts exist so they can arrive without rewriting agents |

What the harness deliberately cannot do — and says so rather than implying otherwise — is
in [docs/limitations.md](docs/limitations.md). The short version: it cannot instrument an
arbitrary unwrapped library call it never sees.

## Configuration

Full reference: [docs/configuration.md](docs/configuration.md); a commented starting point:
[`harness.example.yaml`](harness.example.yaml). Environment variables (`UAH_*`,
`LANGFUSE_*`) are documented there too. Configuration is validated at construction, so a
mistake fails at startup rather than on the first execution.

## Documentation

| Document | What is in it |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layering, ports and adapters, interceptor pipeline, execution flow |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Versions actually tested, per-feature matrix, degradation rules |
| [docs/configuration.md](docs/configuration.md) | Every setting, every environment variable |
| [docs/privacy.md](docs/privacy.md) | Capture policy, redaction, sampling, what never leaves |
| [docs/limitations.md](docs/limitations.md) | What the harness cannot do, stated plainly |
| [docs/performance.md](docs/performance.md) | Measured overhead and how to reproduce it |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms, causes, fixes |
| [examples/plain_python.py](examples/plain_python.py) | All three modes, tools, artifacts, claims, child runs — runs with no services |
| [examples/langgraph_agent.py](examples/langgraph_agent.py) | An existing node and a runtime-aware node in one graph, with a checkpointer |
| [examples/reorder_workflow.py](examples/reorder_workflow.py) | The full picture: 6-node graph with parallel fan-out, a nested sub-agent, tools, model calls, real business logic, artifacts, claims, memory and Langfuse |
| [examples/memory_tour.py](examples/memory_tour.py) | Every memory operation — history, episodic, typed long/short-term, RAG ingestion, knowledge graph, inventory, grounding, deletion — and what each sends over the wire |

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

Test suite: 280 tests (276 functional + 4 benchmarks), 91% line coverage of the core and
the adapter. Categories: unit, contract (protocol conformance), integration, end-to-end
(including the real Memory Service SDK over a mocked HTTP layer), failure injection,
observability, compatibility and performance.

Fallback without uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"
```

## License

Apache-2.0.
