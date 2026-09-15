# Compatibility

Everything in this document is produced by a test run, not by recollection.
`tests/compatibility/test_matrix.py` writes [`compatibility-matrix.json`](compatibility-matrix.json)
with the versions that were actually exercised; the table below is that file, transcribed.

## Verified in this repository

| Component | Version tested |
| --- | --- |
| Python | 3.12.14 |
| universal-agent-harness | 0.1.0 |
| langgraph | 1.2.11 |
| langchain-core | 1.6.3 |
| langfuse | 4.15.2 |
| opentelemetry-api / -sdk | 1.44.0 |
| pydantic | 2.13.5 |
| universal-memory (Memory Service SDK) | 0.1.0 |

Declared support ranges (from `pyproject.toml`): Python `>=3.12`, `pydantic>=2.13,<3`,
`opentelemetry-api>=1.44`, and for the extras `langgraph>=1.2`, `langfuse>=3.0`.
Versions outside the "tested" row are expected to work but are not verified here; the
honest statement is "untested", not "supported".

## LangGraph feature matrix

Each row is covered by a test in `integrations/langgraph/tests/test_nodes.py`.

| Capability | 1.2.11 | Test |
| --- | --- | --- |
| Wrap an existing node | ✅ | `test_existing_node_is_wrapped_without_changing_its_contract` |
| Runtime-aware node (`state, agent`) | ✅ | `test_runtime_aware_node_receives_the_agent_runtime` |
| Async nodes | ✅ | `test_existing_node_is_wrapped_without_changing_its_contract` |
| Sync nodes | ✅ | `test_sync_node_is_supported` |
| `config: RunnableConfig` injection preserved | ✅ | `test_node_receiving_config_still_receives_it` |
| `writer: StreamWriter` injection preserved | ✅ | `test_custom_stream_writer_is_forwarded` |
| Identity from `configurable.harness` | ✅ | `test_identity_overrides_travel_in_the_config` |
| Parallel nodes stay parallel | ✅ | `test_parallel_nodes_run_concurrently_and_merge` |
| Reducers unchanged | ✅ | `test_reducers_are_untouched_by_the_wrapper` |
| Subgraph lineage → nested agent runs | ✅ | `test_subgraph_lineage_nests_agent_runs` |
| Checkpoint retry without duplicate writes | ✅ | `test_checkpoint_retry_does_not_duplicate_memory_observations` |
| Stable run id across replays | ✅ | `test_run_id_is_stable_across_replays_of_the_same_superstep` |
| Streaming (`astream`) | ✅ | `test_streaming_still_streams` |
| Cancellation propagation | ✅ | `test_cancellation_propagates_through_the_graph` |
| Exception propagation | ✅ | `test_exceptions_propagate_to_the_graph` |
| Tool events inside a graph | ✅ | `tests/integration/test_tools.py` (`wrap_tool` inside a run) |

### Which LangGraph APIs the adapter depends on

Public, documented surface only:

* `RunnableConfig` keys `configurable.thread_id`, `configurable.checkpoint_ns`,
  `configurable.checkpoint_id`, `metadata.langgraph_node`, `metadata.langgraph_step`,
  `run_id`;
* the node-argument convention — a node may declare `config`, `store`, `writer`,
  `previous` or `runtime` and LangGraph injects them by name and annotation. The adapter
  generates a wrapper that declares exactly what the wrapped node declared;
* `langgraph.types.StreamWriter`, `langgraph.store.base.BaseStore`,
  `langchain_core.runnables.RunnableConfig` as annotations.

**Trace roots.** LangGraph executes each superstep as its own task, so a node span has no
ambient parent: with nothing enclosing the invocation, every node starts its own trace.
Wrapping `app.ainvoke(...)` in `harness.execution(...)` produces one trace per turn with a
single root (asserted in `tests/e2e/test_example_workflow.py`). Both modes are supported;
only the trace shape differs.

It does **not** touch `langgraph._internal`, the checkpoint format, reducer internals or
any private attribute. Missing keys degrade to `None`; a config the adapter does not
recognise simply yields less identity, never an error.

## Langfuse

| Mode | Requires the SDK | What it does |
| --- | --- | --- |
| `sdk` | yes | creates the Langfuse client, which attaches its exporter to the TracerProvider, and enriches harness spans with Langfuse attributes |
| `otlp` | no | enriches spans with Langfuse's documented OTel attributes; you point an OTLP exporter at Langfuse |
| `auto` (default) | no | `sdk` when importable, otherwise `otlp` |

Verified against langfuse 4.15.2 using its public API only: `Langfuse(...)` (including
`host`, `environment`, `release`, `sample_rate`, `tracer_provider`, `should_export_span`),
`LangfuseOtelSpanAttributes`, `is_default_export_span`, `create_score`,
`create_dataset_item`, `get_prompt`, `flush`, `shutdown`. SDK-mode tests stub the client,
so no live project or network is needed to run the suite.

## Memory Service

The harness depends on the `universal-memory` SDK contract only:
`MemoryClient.bind(**scope)` → `MemoryContext`, then `context()`, `recall()`, `observe()`,
`chat.*`, `graph.*`, `files.*`, `tools.record()`.

Two suites cover it. `tests/e2e/test_real_memory_sdk.py` runs the real SDK against a mocked
HTTP layer and asserts the wire payloads. `tests/e2e/test_live_memory_service.py` runs
against a **live service** (`make test-live`) — the only suite that mocks nothing.

Verified live against the service at commit-time, with Postgres, Qdrant, Dragonfly and
OpenFGA behind it, and a real ONNX embedding model (`fastembed`, BAAI/bge-small-en-v1.5):

| Exercised | Result |
| --- | --- |
| Full turn: retrieve before, observe after | bundle `COMPLETE`, 12 memories, 182 tokens |
| `recall` | 5 ranked items |
| `history` | messages written by the harness read back |
| `graph_query` | 29 facts extracted from the harness's observations |
| `memories` / `forget` | inventory returned, deletion accepted |
| `add_document` | document handle returned, indexed by the worker |
| `verify` | grounding report returned |
| Replayed write | same `observation_id` — idempotency holds end to end |
| Both examples, run as a user runs them | pass |

All of it was re-run against the **containerised** service (`docker compose up -d`, the
image's own sentence-transformers models), not only a locally launched API: 357 tests pass,
31 of them live.

`make test-live-full` goes further and reads the database rather than trusting a 202.
Final state of that run — 31 live tests, 0 failed background jobs:

| Persisted | Verified |
| --- | --- |
| Memory types | SEMANTIC, EPISODIC, PROCEDURAL, PREFERENCE, DECISION, OUTCOME, FAILURE, SHARED, AGENT — each written through the harness and read back from `memories` with its lifetime |
| Visibility levels | PRIVATE, RUN, USER, THREAD, WORK, WORKSPACE, AGENT_GROUP, TENANT — each stored with the audience the caller asked for |
| Knowledge base | document -> `documents` row -> `chunks` rows -> indexed vectors -> retrieved as `knowledge` in a bundle |
| Knowledge graph | `graph_entities` and `graph_relations` created from the harness's own observations, then traversed with `graph_query` |
| Conversation | `threads`, `sessions`, `turns` and `messages` rows, with the derived session owning the turn |
| Tool memory | `tool_invocations` rows attributed to the agent and run that made the call |
| Agent lineage | a nested run's writes carry both its own run id and its parent's |
| Idempotency | a replayed write produces exactly one observation row |

Caveat on that run: the service was configured with its lightweight model stand-ins
(`embedding=hash`, `reranker=lexical`, `nli=lexical`) because the host lacked the heavy model
extras. That exercises every wire contract, scope rule and persistence path, but says nothing
about retrieval or grounding *quality* — which is the Memory Service's own concern, measured
by its own benchmarks.

### What the live run found that mocks could not

Three defects, all now fixed and covered by contract tests
(`tests/contract/test_memory_scope_rules.py`):

1. **`turn_id` was sent without `session_id`.** The service rejects the whole request
   (`session_id` is required whenever a turn is set). The LangGraph adapter derived a session;
   the plain-Python path did not. The context now derives one session per thread, and
   `scope_fields()` drops conversation ids it cannot express coherently.
2. **Observation kinds the service does not accept.** The harness wrote `AGENT_INPUT` and
   `CLAIM`; the enum is MESSAGE / FILE / AGENT_RESULT / TOOL_RESULT / DECISION / FEEDBACK /
   EVENT / IMPORT. Every automatic input and claim write was being refused. `MemoryObservation`
   now validates the kind before the wire.
3. **`record_input()` / `record_output()` silently did nothing** unless the `record_messages`
   policy was on — a config flag elsewhere quietly voiding an explicit call. The policy now
   governs only the automatic path; an explicit call always writes.

A `turn_id` is also bound to the session that created it, so turn ids must be unique per
session; the examples derive one per run.

## CrewAI and Google ADK

Not implemented. The contracts are deliberately framework-neutral so an adapter can be
added without core changes (`FrameworkAdapter`, `AgentExecutionContext`, `AgentRequest`,
`AgentResult`). Until such an adapter exists and is tested, CrewAI and ADK are **not
supported** — wrapping their callables as plain Python works, but framework-level lineage,
events and state mapping do not.

## Degradation rules

| Missing | Behaviour |
| --- | --- |
| `universal-agent-harness-langgraph` | `harness.langgraph` raises `ImportError` naming the extra |
| `langfuse` | Langfuse falls back to `otlp` attribute mode |
| `opentelemetry-sdk` | the OTel API's no-op is used; spans are created and dropped |
| `structlog` | stdlib logging with the same fields under `extra` |
| a memory client | `NoOpMemoryRuntime`: retrieval returns `None`, writes are skipped |
| a model client | `runtime.model.invoke` raises `ConfigurationError` with guidance |
| a tool runtime | `runtime.tools.call` raises `ToolNotFoundError` with guidance |

## Regenerating this matrix

```bash
pytest tests/compatibility -q       # rewrites compatibility-matrix.json
```
