# Configuration

Three sources, applied in order: a YAML file (or a dict), then environment variables, then
explicit overrides. The result is validated immediately — a bad configuration fails at
startup, not on the first agent execution — and unknown keys are rejected rather than
silently ignored.

```python
from universal_agent_harness import AgentHarness, HarnessConfig

harness = AgentHarness(memory=memory, config="harness.yaml")
harness = AgentHarness(memory=memory, config={"timeouts": {"default_seconds": 10}})
harness = AgentHarness(memory=memory, config=HarnessConfig.load("harness.yaml",
                                                               overrides={"memory": {"enabled": False}}))
```

The YAML may be the full document (`harness:` + `frameworks:` keys, as in
[`harness.example.yaml`](../harness.example.yaml)) or just the harness section.

## Settings

Every setting, its default and a one-line meaning is in
[`harness.example.yaml`](../harness.example.yaml); the authoritative definitions are in
[`config/settings.py`](../src/universal_agent_harness/config/settings.py).

Decisions worth calling out:

| Setting | Default | Why that default |
| --- | --- | --- |
| `memory.observe_tool_results` | `false` | tool outputs frequently contain customer data |
| `memory.writeback` | `true` | the turn must not wait for consolidation |
| `memory.failure_mode` | `non_blocking` | a memory outage degrades a run, it does not fail it |
| `telemetry.configure_sdk` | `false` | applications usually configure OpenTelemetry themselves |
| `telemetry.capture.raw_*` | `false` | nothing sensitive is exported unless asked for |
| `telemetry.capture.user_id` | `false` | identity export is a policy decision |
| `retries.enabled` | `false` | retries are only safe for idempotent work |
| `evaluation_events.enabled` | `false` | evaluation is opt-in and asynchronous |
| `policy.enabled` | `false` | the harness does not invent an authorization model |
| `observability.langfuse.failure_mode` | `non_blocking` | observability is never a business dependency |

## Environment variables

| Variable | Setting |
| --- | --- |
| `UAH_MEMORY_ENABLED` | `memory.enabled` |
| `UAH_MEMORY_RETRIEVE_BEFORE` | `memory.retrieve_before` |
| `UAH_MEMORY_OBSERVE_AFTER` | `memory.observe_after` |
| `UAH_MEMORY_FAILURE_MODE` | `memory.failure_mode` |
| `UAH_OTEL_ENABLED` | `telemetry.enabled` |
| `UAH_OTEL_EXPORTER` | `telemetry.exporter` |
| `UAH_OTEL_ENDPOINT` | `telemetry.endpoint` |
| `UAH_OTEL_CONFIGURE_SDK` | `telemetry.configure_sdk` |
| `UAH_SERVICE_NAME` | `telemetry.service_name` |
| `UAH_LANGFUSE_ENABLED` | `observability.langfuse.enabled` |
| `UAH_LANGFUSE_MODE` | `observability.langfuse.mode` |
| `UAH_LANGFUSE_SAMPLE_RATE` | `observability.langfuse.sampling.sample_rate` |
| `UAH_LANGFUSE_FAILURE_MODE` | `observability.langfuse.failure_mode` |
| `LANGFUSE_HOST` / `LANGFUSE_BASE_URL` | `observability.langfuse.base_url` |
| `LANGFUSE_PUBLIC_KEY` | `observability.langfuse.public_key` |
| `LANGFUSE_SECRET_KEY` | `observability.langfuse.secret_key` |
| `LANGFUSE_TRACING_ENVIRONMENT` | `observability.langfuse.environment` |
| `UAH_DEFAULT_TIMEOUT` | `timeouts.default_seconds` |
| `UAH_MODEL_TIMEOUT` | `timeouts.model_seconds` |
| `UAH_TOOL_TIMEOUT` | `timeouts.tool_seconds` |
| `UAH_RETRIES_ENABLED` | `retries.enabled` |
| `UAH_RETRIES_MAX_ATTEMPTS` | `retries.max_attempts` |
| `UAH_EVAL_EVENTS_ENABLED` | `evaluation_events.enabled` |
| `UAH_LOG_LEVEL` | `observability.log_level` |

Booleans accept `1/true/yes/on`. An unparseable value raises at startup naming the variable.

## Per-agent overrides

Harness-wide configuration is the floor; individual agents can narrow it at wrap time:

```python
harness.wrap(
    agent,
    agent_id="inventory-agent",
    skills=["inventory.analysis"],
    memory_policy=MemoryPolicy(retrieve_before=False, observe_output=False),
    timeout_seconds=5,
    idempotent=True,               # makes the agent eligible for configured retries
    error_mode="result",           # return AgentResult(status=ERROR) instead of raising
    state_mapper=lambda r: {"inventory_result": r.data},
    interceptors=[MyInterceptor()],
)
```

## Providers

```python
AgentHarness(
    memory=MemoryClient(...),          # universal-memory SDK client (or a MemoryContext)
    model=my_model_client,             # ModelClient, or any callable/object with ainvoke
    tools=[tool_a, tool_b],            # list, {name: callable}, or a ToolClient
    artifacts="/var/lib/agent-artifacts",   # path, store instance, or None (in-process)
    policy=AllowListPolicyProvider(tools={"inventory_db"}),
    registry=my_registry,              # AgentRegistryClient; default is a no-op
    evaluation_sink=my_sink,           # EvaluationSink
    redactor=MyRedactor(),             # TelemetryRedactor
    listeners=[on_event],              # lifecycle listeners
    defaults={"tenant_id": "acme"},    # used when no context is supplied
    error_mode="raise",                # "raise" (default) | "result"
)
```
