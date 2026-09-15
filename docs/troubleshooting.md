# Troubleshooting

### `ValueError: no tenant_id available`

No context was passed and the harness has no defaults. Either pass
`context=AgentExecutionContext.create(tenant_id=...)`, or construct the harness with
`defaults={"tenant_id": "acme"}`. Inside a LangGraph run, identity can also travel in the
config: `{"configurable": {"harness": {"tenant_id": "acme"}}}`.

### No memory context in my agent (`runtime.memory_context is None`)

One of:

* no memory client was passed to `AgentHarness(...)` (check `runtime.memory.enabled`);
* the request had **no query** — pass `objective=...`, a string input, a recognised key
  (`query`/`question`/`objective`/`prompt`/`input`/`text`), or for LangGraph
  `wrap_node(..., query="question")`;
* `memory.retrieve_before` is false;
* retrieval failed and the harness degraded — look for a `MEMORY_DEGRADED` warning on the
  result and a `memory.retrieve` span with `status=error`.

### Memory observations are missing

Writeback is asynchronous. In a short-lived process, `await harness.drain()` (or
`await harness.aclose()`) before exiting, or set `memory.writeback: false`. Check also that
`memory.observe_output` is on and that the result carried something to write: text `data`,
`claims`, or explicit `memory_observations`.

### Duplicate memory observations after a retry

Should not happen: keys are derived from (tenant, thread, turn, task, agent, run, content).
If it does, the run id is probably not stable — that requires a durable position (a thread
plus a turn or task id). For LangGraph, ensure `configurable.thread_id` is set.

### No spans anywhere

* `telemetry.enabled` is false;
* the application has not configured an OpenTelemetry SDK, so the API's no-op is in use.
  Either configure OTel yourself (recommended) or set `telemetry.configure_sdk: true` with
  `exporter: console|otlp`;
* the execution was sampled out — check `sampling.sample_rate`.

### Nothing in Langfuse

* `observability.langfuse.enabled` is true **and** `public_key`/`secret_key` are set
  (otherwise construction fails with a clear error);
* in `sdk` mode the client needs an OTel SDK provider to attach to. With no provider at
  all, Langfuse creates one; if your app configures one *after* the harness, spans created
  before that point are lost;
* nothing is flushed yet — call `harness.flush()` or `await harness.aclose()` in a
  short-lived process;
* the run was sampled out.

### Spans have no prompts/outputs

That is the default (see [privacy.md](privacy.md)). Enable the specific capture flags you
need, per environment.

### `ConfigurationError: no model client is configured`

`runtime.model` was used without `AgentHarness(model=...)`. Either pass a model client, or
call your provider directly and accept that the call is not instrumented.

### `ToolNotFoundError: no tool runtime is configured`

`runtime.tools.call` was used without `AgentHarness(tools=[...])`. Alternatively wrap the
tool with `@harness.wrap_tool` and call it directly.

### `ImportError: LangGraph support needs the adapter`

`pip install "universal-agent-harness[langgraph]"`.

### My agent's exception type changed

It should not have. The harness re-raises the original exception and attaches
`exc.agent_error`. Exceptions raised by the harness itself (`AgentTimeoutError`,
`PolicyDeniedError`, `ModelError`, `ToolError`, `MemoryUnavailableError`) are its own.
If you would rather never see an exception, wrap with `error_mode="result"`.

### `asyncio.run() cannot be called from a running event loop`

Not from the harness: `run_sync` detects a running loop and uses a worker thread. If you
see it, something else in the stack is calling `asyncio.run` inside a loop.

### Retries are not happening

Retries need **both** `retries.enabled: true` and `harness.wrap(..., idempotent=True)`, and
only apply to the configured categories (timeout, rate limit, dependency).

### A listener or evaluation sink raised

Observers are failure-isolated: the error is logged
(`lifecycle listener failed` / `evaluation sink failed`) and the execution continues.

### Everything is slower than expected

Measure before assuming (`pytest tests/performance -m performance -s`). Common causes: a
synchronous span processor in production, an interceptor doing network I/O, or payload
capture enabled with very large prompts.
