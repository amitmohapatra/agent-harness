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

The YAML may be the full document (with a `harness:` key, as in
[`harness.example.yaml`](../harness.example.yaml)) or just the harness section.

The surface is **53 settings** and every one of them does something. Two rules keep it that
way: capture policy and sampling are defined once (under `telemetry`) and obeyed by every
backend, and a provider is enabled by *passing* it — there is no `policy.enabled` flag that
has to agree with the policy object you supplied.

## Settings

Every setting, its default and a one-line meaning is in
[`harness.example.yaml`](../harness.example.yaml); the authoritative definitions are in
[`config/settings.py`](../src/universal_agent_harness/config/settings.py).

Decisions worth calling out:

| Setting | Default | Why that default |
| --- | --- | --- |
| `memory.observe_tool_results` | `false` | tool outputs frequently contain customer data |
| `memory.record_outcome` | `true` | tool memory learns procedures from runs it knows succeeded; without a label it waits hours to guess one, and never learns from a failure |
| `memory.writeback` | `true` | the turn must not wait for consolidation |
| `memory.failure_mode` | `non_blocking` | a memory outage degrades a run, it does not fail it |
| `telemetry.configure_sdk` | `false` | applications usually configure OpenTelemetry themselves |
| `telemetry.capture.inputs` / `.outputs` | `false` | nothing sensitive is exported unless asked for |
| `telemetry.capture.user_id` | `false` | identity export is a policy decision |
| `retries.enabled` | `false` | retries are only safe for idempotent work |
| `evaluation_events.enabled` | `false` | evaluation is opt-in and asynchronous |
| `observability.failure_mode` | `non_blocking` | observability is never a business dependency |

## Environment variables

| Variable | Setting |
| --- | --- |
| `UAH_MEMORY_ENABLED` | `memory.enabled` |
| `UAH_MEMORY_RETRIEVE_BEFORE` | `memory.retrieve_before` |

| `UAH_MEMORY_FAILURE_MODE` | `memory.failure_mode` |
| `UAH_OTEL_ENABLED` | `telemetry.enabled` |
| `UAH_OTEL_EXPORTER` | `telemetry.exporter` |
| `UAH_OTEL_ENDPOINT` | `telemetry.endpoint` |
| `UAH_OTEL_CONFIGURE_SDK` | `telemetry.configure_sdk` |
| `UAH_SERVICE_NAME` | `telemetry.service_name` |
| `UAH_LANGFUSE_ENABLED` | `observability.langfuse.enabled` |
| `UAH_LANGFUSE_MODE` | `observability.langfuse.mode` |
| `UAH_SAMPLE_RATE` | `telemetry.sampling.sample_rate` |
| `UAH_CAPTURE_INPUTS` / `UAH_CAPTURE_OUTPUTS` | `telemetry.capture.inputs` / `.outputs` |
| `UAH_OBSERVABILITY_FAILURE_MODE` | `observability.failure_mode` |
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


## What is mandatory?

Almost nothing. This is the complete set of things you *must* provide:

| You must set | When | If you don't |
| --- | --- | --- |
| `tenant_id` | always — on the context or in `defaults` | `ValueError` at the first execution |
| `agent_id` | per wrapped agent | the function's name is used |
| `memory=` client | only to use memory | memory calls are no-ops; everything else works |
| `model=` client | only to use `runtime.model` | `ConfigurationError` naming the fix |
| `tools=` | only to use `runtime.tools` | `ToolNotFoundError` naming the fix |
| `LANGFUSE_*` keys | only with Langfuse enabled | startup fails with the missing key named |

Everything else has a working default. `AgentHarness(defaults={"tenant_id": "acme"})` is a
complete, valid configuration.

## What the harness checks for you

These are validated *before* anything is sent, because the service would otherwise accept
the request and fail later — in a background job, where you would never see it:

| Check | Why |
| --- | --- |
| Conversation ids hang together | `turn_id` needs a `session_id`, a session needs a thread. The harness derives one session per thread and drops ids it cannot express. |
| Visibility prerequisites | `AGENT_GROUP` needs `agent_group_id`, `WORKSPACE` needs `workspace_id`, `USER` needs `user_id`, and so on. A write to an audience the context cannot express is refused immediately, with the error naming all the ways to supply it. |
| Thread existence before ingestion | A thread-visible document is readable only by thread participants, so `add_document` creates the thread first. |
| Memory policy option names | A misspelled key (`observe_outputs`) is refused, not silently ignored. |
| Observation kinds | Only the service's vocabulary (MESSAGE, FILE, AGENT_RESULT, TOOL_RESULT, DECISION, FEEDBACK, EVENT, IMPORT) is accepted. |
| Langfuse credentials | Enabling Langfuse without keys fails at startup, not silently at runtime. |
| Unknown config keys | Rejected rather than ignored, so a typo is not a silent default. |

## What do I set, and when?

A decision table, rather than a list of knobs. Each row is a situation you will actually be
in; the setting is the answer.

### Memory

| Situation | Setting |
| --- | --- |
| "My agent should see prior context" | `memory.retrieve_before: true` (default) and give the request a query — an `objective`, a string input, or `query`/`question` in a dict. Without one, retrieval is skipped rather than guessed. |
| "Nothing should be written back automatically" | `memory.observe_input/observe_output/observe_claims: false`. Explicit `runtime.memory.*` calls still write — the policy governs only the automatic path. |
| "Tool outputs contain customer data" | Leave `memory.observe_tool_results: false` (the default). |
| "This agent's notes must not leak to the user or other agents" | `memory.private_by_default: true` — everything it writes becomes RUN-visible. |
| "Agents should share findings with each other" | Declare the group once: `defaults={"agent_group_id": "crew"}`, or `harness.wrap(..., agent_group="crew")`, or `share(..., group="crew")` for one call. Then `runtime.memory.share(...)` just works. |
| "I want the turn recorded as a conversation, not just observations" | `memory.record_messages: true`. |
| "Don't tell the service whether my runs worked" | `memory.record_outcome: false`. The harness labels each finished run success or failure so tool memory can learn from it; turning it off means the service falls back to inferring a weak positive hours later. |
| "The process exits right after the turn" | `memory.writeback: false`, or `await harness.drain()` before exit — writes are asynchronous by default. |
| "A memory outage must fail the request" | `memory.failure_mode: fail_closed`. Otherwise the run degrades with a `MEMORY_DEGRADED` warning. |
| "Context is too large / too small" | `memory.token_budget`. |
| "Memory calls are hanging" | `timeouts.memory_seconds` (one deadline for every memory call). |

### Telemetry and privacy

| Situation | Setting |
| --- | --- |
| "My app already configures OpenTelemetry" | Nothing — the default (`configure_sdk: false`) uses your provider. |
| "Nothing configures OpenTelemetry and I want traces" | `telemetry.configure_sdk: true` plus `exporter: console` or `otlp` + `endpoint`. |
| "I'm debugging and need to see prompts" | `telemetry.capture.inputs: true` (and `outputs`) — per environment, not globally. |
| "Retrieved memory text must never leave the process" | Leave `capture.memory_content: false` (the default). |
| "Our policy forbids exporting user ids" | Leave `capture.user_id: false` (the default). |
| "Too much trace volume" | `telemetry.sampling.sample_rate: 0.1`, keeping `error_sample_rate: 1.0`. |
| "This agent is business-critical, always trace it" | `sampling.critical_agents: [billing-agent]`. |
| "Losing telemetry is worse than failing the request" | `observability.failure_mode: fail_closed` (rare; it makes observability a business dependency). |
| "I need Langfuse" | `observability.langfuse.enabled: true` + `LANGFUSE_*` env vars. Capture and sampling come from `telemetry` — there is nothing else to set. |

### Execution

| Situation | Setting |
| --- | --- |
| "Agents must not run longer than N seconds" | `timeouts.default_seconds`, or `timeout_seconds=` per wrapped agent. |
| "A model provider is slow" | `timeouts.model_seconds` — bounded by the agent deadline regardless. |
| "Transient upstream failures should be retried" | `retries.enabled: true` **and** `harness.wrap(..., idempotent=True)`. Both are required: retrying a non-idempotent agent is how you double-charge someone. |
| "I want a result object instead of an exception" | `harness.wrap(..., error_mode="result")`. |
| "Results can be large" | `artifacts.inline_max_bytes` — anything bigger is stored and replaced by a reference. |
| "I need evaluation events" | `evaluation_events.enabled: true` (asynchronous by default; `synchronous: true` blocks the result). |
| "Only some agents may run / some tools may be called" | Pass `policy=` a provider. Passing it is what enables it. |

## Operational rules that are not settings

Two service behaviours that no configuration changes, and that explain most surprises:

1. **Writes are asynchronous.** The API commits your observation and queues the work that
   turns it into memories, graph edges and index entries. Reading immediately after writing
   will not show it; the live tests poll for this reason.
2. **Reads are audience-filtered.** A memory or document chunk is retrievable only by a
   principal in its audience. WORKSPACE and GROUP audiences require membership in the
   authorization service, which the harness cannot provision for you — so a
   workspace-visible document is readable only by workspace members. (The THREAD case is
   handled: `add_document` creates the thread, because a thread grants its audience only
   once it exists.)

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
    policy=AllowListPolicyProvider(tools={"inventory_db"}),   # passing it enables it
    registry=my_registry,              # AgentRegistryClient; omitted -> no registry
    evaluation_sink=my_sink,           # EvaluationSink
    redactor=MyRedactor(),             # TelemetryRedactor
    listeners=[on_event],              # lifecycle listeners
    defaults={"tenant_id": "acme"},    # used when no context is supplied
    error_mode="raise",                # "raise" (default) | "result"
)
```
