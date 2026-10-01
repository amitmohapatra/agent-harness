# Trellis platform overhaul — shared spec (source of truth)

Date: 2026-09-30. Applies to: agent-contracts, bifrost-sdk, agent-memory-service, agent-runs
(absorbs agent-schedules), agent-harness. Every implementer reads this whole file first.

## 0. Rules for every change

1. **One way to do each thing.** When two APIs overlap, keep one and delete the other (no
   deprecated aliases — this is a pre-1.0 platform with no external users; update every caller,
   example, test and doc in the same change).
2. **Delete dead code completely**: unused modules, flags, deps, exports, docs, tests of deleted
   code. No commented-out code, no "kept for compat".
3. **No duplicate code.** Shared logic lives in exactly one place.
4. **Env vs code.** Env/config holds only *deployment facts*: URLs, secrets/keys, DSNs, ports,
   worker counts, sampling rates, feature *availability* that differs per deployment. Everything
   else (thresholds, limits, prompts, weights, retry counts, timeouts that are design decisions)
   is a named constant in code. Every env var is documented once in `.env.example` with a
   one-line comment; nothing undocumented is read.
5. **Complexity.** Hot paths are O(1) or O(log n) per request in stored data size: indexed
   lookups, bounded top-k, no scans over tenant data on the request path. Anything heavier
   runs as a background job.
6. **Tests.** Every behaviour change has tests (unit + integration against the real local
   services where the repo already does that). The full suite of the repo passes before commit.
   Lint (ruff) and type check (pyright/mypy, whatever the repo uses) pass.
7. **Design.** Hexagonal where the repo already is; small modules; explicit types (pydantic);
   no global mutable state; idempotent writes; async all the way.
8. **Git.** Work on branch `overhaul` in each repo (create from `main`). Commit in logical
   steps with clear messages ending with
   `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. **Do not push, do
   not merge to main.**
9. **Docs.** README + docs/ updated to describe only what exists. Remove stale claims.

Local infra already running (docker): Postgres, Qdrant, OpenFGA, Dragonfly, Bifrost gateway
(`bifrost-gateway`). Use them for integration tests; do not stop or recreate them.

## 1. Decisions (final)

- **Temporal integration is removed** (AgentRunWorkflow never executed agents; TemporalScheduler
  firings never ran). Durable execution = agent-runs queue + lease + `trellis worker`.
- **agent-schedules is merged into agent-runs** (one service: runs + schedules + inbox + worker
  queue). agent-schedules repo keeps only a README pointing to agent-runs.
- **FeedbackStore port removed**; the memory service is the feedback store.
- **`turn_run_links` and `message_versions` tables dropped** (write-only).
- **Per-tenant LLM**: the memory service keeps its BYO Bifrost virtual key design
  (agent → workspace → tenant → operator). The harness registers the agent-level memory model
  key at startup from env `TRELLIS_MEMORY_MODEL_KEY` (idempotent PUT). No other registration
  path is documented for harness users.
- **Frameworks are never modified or re-implemented.** The harness attaches to an existing
  agent object. Compaction inside a run belongs to the framework; the *durable* thread summary
  is produced by the memory service in the background from the recorded transcript.

## 2. agent-contracts (`trellis-contracts` 0.4.0)

- Remove ports: `FrameworkAdapter`, `PromptProvider`, `FeedbackStore`, `Scheduler`,
  `LifecycleListener`, `AgentRegistryClient`, `EvaluationSink`; remove
  `EvaluationProvider.submit_feedback`. Remove `LifecycleEvent` (harness keeps an internal
  enum if it needs one). Keep: ModelClient, ToolClient, ArtifactClient, MemoryPort, TelemetryProvider,
  TelemetryRedactor, EvaluationProvider (score, submit_dataset_item), AgentPolicyProvider,
  EventSink, RunStore, Judge, AgentDirectory, AgentInterceptor.
- `RunStatus` gains `QUEUED` (live, not final). State machine: QUEUED→RUNNING→(PAUSED↔RUNNING)*→final.
  `RunStore` port: add `queued(start)`.
- `InterruptReason`: QUESTION, APPROVAL, REVIEW, CHOICE, AUTH.
- `Interrupt` gains: `ui: Literal["approve","form","table","diff","choice"] = "approve"`,
  `assignee: str | None` (e.g. `user:u1`, `role:procurement`), `deadline: AwareDatetime | None`,
  `escalate_to: str | None`, `options: list[str] = []` (CHOICE), `payload_ref: ArtifactRef | None`
  (large data by reference). Validation: CHOICE requires options; REVIEW requires `expects`.
- `ScheduleSpec`/`Schedule` stay (used by agent-runs now) — align fields with the service
  (§5): add `last_run_id`, `consecutive_failures`, `last_error`, `retry_after`, `created_by`.
- `RunRecord`/`RunStart` stay; agent-runs must use them directly (§5).
- README + ADR 0002 (contracts v3) updated; A2A version text fixed (code says "1.0").
- Bump version to 0.4.0.

## 3. bifrost-sdk (0.2.0)

- Make internal (drop from `__all__`): `Breaker`, `Call`. Keep `RETRYABLE` (memory uses it).
- Delete `mcp.execute` alias; the one call is `Bifrost.execute_tool(tool_call)`.
- Delete `Bifrost.using/.prompt/.call` constructors; per-call options go through `Options`.
- Move admin resources (governance, routing, skills, prompts, vk) to `bifrost_sdk.admin` with
  an `Admin` client; `bf.mcp` keeps only read/list + client CRUD used by the harness sync.
- Export the `x-bf-*` header constants as `bifrost_sdk.headers`.
- Add typed MCP config: `MCPClientConfig` with `is_code_mode_client`, `tools_to_execute`,
  `tools_to_auto_execute` (always empty from our side — we never use Agent Mode), `connection`.
- Add `Bifrost.tools(clients: list[str] | None, only: list[str] | None) -> list[ToolDef]`
  (scoped listing honoring include-clients/include-tools semantics, names `client-tool`).
- Add `Options(mcp_clients=, mcp_tools=)` (already exists as `.mcp()` — keep one spelling).
- Add `Bifrost.mcp_logs(since, limit, parent_request_id=None)` reading the gateway MCP log API
  (if the gateway exposes one; if not, document and skip — verify against the running gateway).
- Fix tests using `memory.recall` → `memory-recall`. Fix stale `execute_tool` docstring.
- Bump 0.2.0.

## 4. agent-memory-service

### 4.1 Cleanup (verified list)
Delete: `modules/evaluation/`, `adapters/auth/`, `adapters/intelligence/` (untracked pyc dirs);
`WorkingMemory` class + wiring (keep `working_memory_ttl_seconds`); `memory.observe` no-op job
(after confirming no pending rows in job_outbox/procrastinate); LLM uses `ambiguous_extraction`,
`ambiguous_worthiness` (+ helpers, prompts, tests; switch placeholder-use tests to another use);
rerank flag + `CrossEncoderReranker` (move class under benchmark/ if benchmark scripts need it);
`source_turn_expansion` + `source_turns.py` (keep `derived_source_k`); tables `message_versions`,
`turn_run_links` (migration 0016) + repo/port methods; `sessions.end`, `turns.complete`;
deps `rapidfuzz`, `deepeval`, `ragas`, `fastembed`, `optimum`, `docling-graph`;
`hindsight-client` → optional extra `[hindsight]` (lazy import, tests skip without it).
Keep (measure, then decide in 4.8): consolidation, entity_prefetch, memory_entity_search,
query_decomposition, chunk_context. MemoryType enum stays (document 8 primary kinds:
SEMANTIC fact, PREFERENCE, EPISODIC, PROCEDURAL, TASK, USER profile, TOOL, OUTCOME).
Fix stale README/ROADMAP/ARCHITECTURE claims.

### 4.2 Per-tenant LLM
- Fix gates: `reflection.reflect_all` and `connections.connect_all` must not pre-check
  `wants()` without identity; iterate principals that have credentials and gate per identity.
  Pass `workspace_id` into `model_identity` for reflection, connections, briefs.
- Register periodic LLM jobs whenever `llm.enabled != False`.
- New table `llm_policies(tenant_id, principal_id, uses[], read_assist bool, updated_at, revision)`
  + `PUT/GET /v1/model-key/policy` (tenant admin; workspace variant) + SDK. `LLMAssist.wants(use)`
  = global allow-list ∩ policy of the resolved principal (resolution order same as keys). Default
  policy when a key exists: all uses except the removed ones; `read_assist=true`.
- Reads use `read_assist` from policy when the request does not set `use_llm` explicitly.
- Cost: tenant label on `llm_tokens_total`; per-job accounting scope; table
  `llm_usage_daily(tenant_id, use, day, tokens, calls)` upserted per call (O(1)).

### 4.3 Record path (hot, no LLM)
- `POST /v1/messages` accepts missing session/turn ids and derives them deterministically
  (session = thread-session, turn = next per thread) — SDK no longer requires them.
- New `POST /v1/memories` = literal remember: stores the content verbatim as one memory
  (kind, lifetime, visibility, entities?, valid_from?), synchronous insert returning
  `memory_id`, then enqueues index/graph jobs. Dedups on content hash within scope.
  `/v1/observations` stays = raw evidence for learning (extracted asynchronously).
  SDK `remember()` calls `POST /v1/memories` (no more observe-with-hints).
- New `POST /v1/memories/{id}/supersede {content, reason}` (bi-temporal: old gets valid_to,
  superseded_by; new row). `DELETE /v1/memories/{id}` stays (soft forget + audit).

### 4.4 Serve path: `POST /v1/context` (the push API) — target p95 < 300 ms without LLM
Request adds: `since_revision?: int`, `tools?: {available: [names] | null, k: int=8}`.
Response adds:
- `revision: int` (current scope revision for delta calls; when `since_revision` given, only
  items changed after it are returned and `delta=true`).
- `profile: [{block, text, version}]` — pinned blocks for the scope (user, agent, workspace).
- `thread_summary: {text, covers_to_sequence, version} | null` — durable (4.6).
- `procedures: [{id, title, steps, success_rate, support}]` — learned for this task (4.7).
- `tools: {candidates: [{name, score, success_rate, why}], plan: {steps, success_rate, support} | null,
  next: str | null, prefill: {arg: {tool, value, source, evidence_id}}, missing: [{tool, arg,
  entity_type, question}]}` — only when `tools` requested.
All sections rendered into `rendered` within `token_budget`. Cache fingerprint includes tools.

### 4.5 Agent tools (the pull API, ReAct mode)
`GET /v1/agent-tools` → list of `{name, description, input_schema}` (JSON schema, multilingual-
neutral descriptions). `POST /v1/agent-tools/{name}` with `{"args": {...}}` → `{"result": ...}`.
Scope from headers as every other route. Tools (final set, no others):
`memory_search(query, kinds?, time_from?, time_to?, k?)`, `memory_remember(content, kind, scope)`,
`memory_update(id, content?, invalidate?, reason)`, `memory_forget(id, reason)`,
`history_search(query?, time_from?, time_to?, k?)`, `profile_edit(block, old, new)`,
`procedures_search(task, k?)`, `tool_search(task, k?)`, `record_outcome(success, note?)`.
Every call is logged as a *pull* for prefetch learning (4.7). SDK: `ctx.agent_tools()` /
`ctx.call_agent_tool(name, args)`.

### 4.6 Summaries and profile
- Table `thread_summaries(tenant_id, thread_id, version, text, covers_to_sequence, model, created_at)`,
  unique (thread, version); latest-version index. Background job `summary.refresh` enqueued when
  a thread gains ≥ N (constant) messages since the last summary; rolling: new summary =
  f(previous summary, new messages). Uses tenant LLM (`summaries` use); without a key, an
  extractive fallback (existing rolling logic) is stored so behaviour is uniform.
  `GET /v1/threads/{id}/summary`. `/v1/context` returns summary + messages after
  `covers_to_sequence` (bounded).
- Table `profile_blocks(tenant_id, scope_key, block, text, version, updated_at)`;
  `GET /v1/profile`, `PUT /v1/profile/{block}` (full text) and `PATCH` (`old`→`new` replace,
  409 if `old` not found). Background job keeps a `user` block updated from USER/PREFERENCE
  memories (LLM when key present, deterministic template otherwise).

### 4.7 Learning (background jobs, tenant LLM when present)
- **Feedback projection** for all targets: MEMORY (exists), ANSWER (adjust confidence of the
  cited memories; label run outcome), TOOL_CALL (→ run_outcomes + tool stats + approval
  patterns), RUN (→ run_outcomes explicit).
- **Ranking uses confidence & reinforcement** (bounded multiplicative factor on fused score).
- **Tool catalog**: `PUT /v1/tools/catalog` (bulk upsert: name, description, input_schema,
  required, argument_entity_types, side_effects ∈ read|write|irreversible, source, server,
  examples; schema_hash) and `GET /v1/tools` (with stats). Tools embedded into a Qdrant
  collection `tools` (name + description + field names), per tenant/workspace.
- **Tool hints** (`POST /v1/tools/hints {task, available?, k}` = same object as
  context.tools): candidates = hybrid search ∩ available, re-scored by success rate/recency;
  plan/next from stored procedures; prefill resolution order: procedure bindings → KG entities
  of `argument_entity_types` → profile/memories → task pattern slots; unresolved → `missing`.
  Replaces `/v1/tools/plan` and `/v1/tools/procedures` (delete them + SDK methods).
- **Procedures stored**: table `procedures(tenant, scope_key, pattern, steps jsonb, bindings,
  success_rate, support, updated_at, status)`; periodic job re-mines changed patterns
  (existing prefix-tree miner) and, with LLM, distils a title/strategy text from successes AND
  failures (ReasoningBank/AWM style), admitted only when support ≥ constant and success_rate
  ≥ constant; delta updates, never wholesale rewrite.
- **Prefetch learning**: agent-tool pulls are stored (pattern of request → tool/args → whether
  the result ids were later cited/used); `/v1/context` pre-includes items with high learned
  prefetch probability for similar requests.
- **Approval patterns**: from TOOL_CALL feedback (approve/reject/edit) aggregate per
  (tool, arg-shape); `GET /v1/tools/approval-suggestions` returns rules with support ≥ constant.
  Never auto-applied.
- **Reflection/consolidation**: fixed gates; consolidation measured (4.8).

### 4.8 KG
- Fix: neighborhood LIMIT-before-dedup; `mentions` in structural set; deterministic tie-breaks.
- `/v1/graph/query` exposes `layers` and `valid_at`.
- New `GET /v1/graph/entities?q=&type=&limit=` and `GET /v1/graph/entities/{id}` (profile:
  current value per predicate, relations, history, evidence). Fill `graph_entities.summary`
  in the enrichment job (LLM when key, deterministic otherwise).
- Tool recording writes `used_entity` / `identified_by` edges (layer=procedural).

### 4.9 Multilingual + LLM in ingestion
- Language detection on write (cheap, deterministic; store `lang` on observations/memories/chunks).
- Rules stay the fast path for English; for non-English content (or when rules yield nothing)
  and a tenant key exists, LLM `contextual_extraction` (fast tier) produces facts; KG
  `relation_extraction` likewise; `chunk_context` for documents. All prompts language-neutral
  ("answer in the source language"). Retrieval: multilingual dense space must be used for
  non-English queries; the regex QueryRouter must not mis-route non-English queries (fallback
  GENERAL_SEMANTIC + query_expansion when `read_assist`).

### 4.10 Performance
Measure `/v1/context` and `/v1/recall` p50/p95 on the local stack (existing bench targets).
Target p95 < 300 ms with LLM off. Profile and fix hot spots (parallelise independent stages,
bound fan-out, cache). Record numbers in docs/MEASUREMENTS.md. Measure the kept flags
(consolidation, entity_prefetch, memory_entity_search, query_decomposition) with the existing
benches; enable if they help within the latency budget, otherwise delete them and their code.

### 4.11 SDK (`trellis-memory` 0.3.0)
Top-level verbs on `MemoryContext`: `context`, `remember`, `update`, `forget`, `search`
(= recall), `history`, `observe`, `feedback`, `record_tool`, `outcome`, `tool_hints`,
`agent_tools`, `call_agent_tool`, `profile`, `summary`. Advanced namespace `ctx.advanced`:
documents, graph, briefs, webhooks, admin/tenant APIs, model keys. Remove removed endpoints.

## 5. agent-runs (absorbs agent-schedules) 0.2.0

- Domain uses `trellis.contracts` `RunRecord`, `RunStart`, `RunStatus`, `Interrupt`,
  `InterruptResolution`, `ScheduleSpec`, `Schedule` directly (no parallel models). DB gains
  `workspace_id`, `last_resolution`, `assignee` (denormalised from awaiting), `deadline`,
  `lease_owner`, `lease_expires_at`, `queued_at`. Alembic migration.
- Queue: `POST /v1/runs` with `queue=true` creates QUEUED. `POST /v1/runs/claim
  {worker_id, agent_ids[], lease_seconds}` → one run (SELECT … FOR UPDATE SKIP LOCKED, indexed)
  or 204. `POST /v1/runs/{id}/heartbeat` extends lease. Expired leases return to QUEUED
  (attempt+1) via the ticker. O(1)/O(log n) per call.
- Inbox: `GET /v1/runs?status=PAUSED&assignee=` (index on tenant, status, assignee).
- Escalation: ticker moves PAUSED runs past `deadline` to `escalate_to` assignee (or marks
  TIMEOUT when none) and emits webhook.
- Schedules (moved from agent-schedules): `/v1/schedules` CRUD + pause/resume/fire; the ticker
  (same process as lease/escalation sweeps, separate entrypoint `agent-runs-ticker`) creates
  QUEUED runs directly in the DB (no HTTP hop, no inter-service key).
- One auth scheme: `X-Api-Key` + tenant from the key; header `X-Trellis-Tenant` only where
  a platform key acts for a tenant. One webhook signature: memory's `X-Trellis-Signature:
  t=..,v1=..` scheme (same as harness).
- One retry/backoff helper. Settings prefix `RUNS__`.
- Remove vendoring of contracts if a uv workspace/path works for Docker (build context = parent
  dir); otherwise keep `make vendor` but make the staleness test compare correctly.

## 6. agent-harness (`trellis-harness` 0.4.0) — attach layer

### 6.1 Public API (the only one)
```python
from trellis import Harness, mcp, tool, a2a, openapi, ReAct

h = Harness()  # reads env (§6.4); one constructor, no kwargs except config=
agent = h.wrap(
    target,
    id="procurement",
    tools=[mcp("erp", only=["get_stock"]), my_fn, a2a(url)],  # optional
    memory="read_write",  # "off" | "read" | "read_write"  (default "off")
    approve={"erp-create_po": "amount > 10000"},  # optional
    tool_hints=False,
)  # optional
```
`target` = compiled LangGraph graph (includes Deep Agents, which returns a compiled graph),
OpenAI Agents `Agent`, Claude Agent SDK `ClaudeAgentOptions`, `ReAct(system=..., model=...)`,
or an async callable `(input, agent) -> Any`. Framework detected by type.
`h.tools(*sources) -> list[native tools]` for teams that build their agent with our tools
before wrapping (LangChain `BaseTool` for LangGraph/Deep Agents, `FunctionTool` for OpenAI,
in-process MCP server for Claude). Tools are instrumented (policy, approval, recording).

`Agent` methods (all of them):
`run(input, *, user, thread=None, tenant=None) -> Result` · `stream(...) -> AsyncIterator[RunEvent]` ·
`start(...) -> RunHandle` (queued, durable) · `resume(interrupt_id, decision, *, answer=None, reviewer)` ·
`schedule(cron, input, *, on_behalf_of, tz="UTC") -> Schedule` · `serve_chat(app)` (AG-UI router) ·
`serve_a2a(app, url)`.
Inside tools/nodes: `trellis.current()` returns the runtime with `.memory` (the SDK verbs),
`.tools.hints(task)`, `.tools.call(name, **args)`, `.ask(question, *, ui, expects, table,
options, assignee, deadline, escalate_to)` (pauses the run; returns the answer on resume),
`.log(...)`.
`h.worker(agents).run()` — claims QUEUED runs from agent-runs for these agents and executes them
(lease + heartbeat); `python -m trellis.worker module:h` CLI entrypoint.
`h.feedback(run_id, verdict, correction=None)`.

### 6.2 What the harness does on `run` (fixed pipeline)
identity → run record (agent-runs, or in-memory when RUNS_URL unset) → if memory != off:
`/v1/context` (tools section when tool_hints) injected into the framework input as a system
message (adapter responsibility) and memory agent-tools added to the tool list (pull mode) →
execute via adapter → record transcript (`/v1/messages`), tool calls (already recorded by the
bridge), outcome → sampled judge (eval) → run finished. Writes are background-queued with
auto-drain on shutdown; failures surface as RunEvent warnings. Pause: adapter maps the
framework interrupt ⇄ contracts `Interrupt`; state persisted with the run (LangGraph
checkpointer if the graph has one; otherwise resume re-executes with idempotent tool calls).

### 6.3 Adapter contract (4 functions per framework, nothing else)
`prepare_input(input, context_text) -> native_input` · `invoke/stream(native_target, native_input,
config) -> native_output | events` · `extract(native_output) -> (answer, transcript, interrupt|None)` ·
`resume_input(interrupt, resolution) -> native_input`. Tools conversion lives in one module
`tools/convert.py` per framework format. Delete: TrellisMiddleware, TrellisHooks, TrellisRunHooks,
MemoryServiceBackend, MemoryServiceSession, BifrostChatModel, BifrostModel(+Provider), builder
`.agent()` methods, `wrap_node`, per-node decorators, `_query_of` copies, ContextAssembler
compaction, all 4 compaction implementations. Models: teams use their own framework model
objects pointed at Bifrost's OpenAI-compatible endpoint; the harness does not wrap models.

### 6.4 Env (the only knobs)
`TRELLIS_TENANT` (default tenant for single-tenant deployments), `BIFROST_URL`,
`BIFROST_VIRTUAL_KEY`, `MEMORY_URL`, `MEMORY_API_KEY`, `TRELLIS_MEMORY_MODEL_KEY`,
`RUNS_URL`, `RUNS_API_KEY`, `TRELLIS_EVAL_SAMPLE` (0..1, default 0.1),
`OTEL_EXPORTER_OTLP_ENDPOINT`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST`
(optional, exported via OTel). No YAML config, no `UAH_*`.

### 6.5 Removals (verified)
Temporal integration (whole package + extra + docs); `listeners=`/`on()` public API (internal
bus may stay); module-level `wrap_tool`, `langgraph.wrap_tool`; `AgentHarness` class and its
19 kwargs; `@agent`, `execution()`, `wrap()` old signatures; adapter registries duplication;
null objects public exports; `NoOpRedactor`; `LangfuseEvaluationSink`, `LangfusePromptProvider`,
`prompts=`; `EvaluationEventInterceptor` + sinks + `AgentEvalEvent` build path unless the judge
needs it internally; `RegistrySync` public; `pydantic-settings` dep; YAML config loader;
`react()` function replaced by the `ReAct` target (native tool messages, structured output).
Keep: GroundedJudge (skip LLM stage when the memory report says judge_consulted), JudgeBudget,
DatasetBuilder/ExperimentRunner (fixed: expected_output + evidence used in scoring),
RegressionGate (wired into CI as `python -m trellis.eval gate`), OTel telemetry, redaction,
policy (risk tiers: read auto, write notify, irreversible ask; approve rules), A2A server/client,
AG-UI router (with reconnect/replay route), webhook sink, artifacts.

### 6.6 Bugs to fix
`_WRAP_OPTIONS` agent_group; policy fail-closed unreachable; tools/__init__ exports; AG-UI tests
not in CI testpaths; summaries RUN-visibility (moot after 6.3); double answer/query writes;
silent queued-write failures; drain on shutdown.

### 6.7 Bifrost tool execution modes
`mcp(...)` sources: normal mode (harness executes each call via `execute_tool`, streaming,
per-call approval). Code Mode is chosen automatically when a source resolves to ≥ 20 tools or
≥ 3 servers AND all its tools are `side_effects=read` (from the memory catalog): the harness
then exposes Bifrost's code-mode meta-tools for that source; scripts touching non-read tools
are never produced because write tools stay in normal-mode sources. Agent Mode never used.
Code Mode nested calls are imported into memory tool records from Bifrost logs by the worker.
