# Overhaul spec — addendum 2: automate everything, trim everything (final)

Supersedes OVERHAUL-SPEC.md where they conflict. Same rules (§0 there). Driver: owner requirement
"whatever can be automated is automated; developers configure only env; one way per thing;
minimal context; APIs return precisely what callers need".

## A. Developer surface (final)

Env (harness): `BIFROST_URL`, `BIFROST_VIRTUAL_KEY`, `TRELLIS_API_KEY` (one key for memory +
runs; tenant derived from it), `MEMORY_URL` (memory on iff set), `RUNS_URL` (durable runs iff
set), optional `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` (Langfuse is just an
OTLP endpoint). Removed: TRELLIS_TENANT, TRELLIS_MEMORY_MODEL_KEY (the agent's memory model key =
BIFROST_VIRTUAL_KEY, registered automatically), TRELLIS_EVAL_SAMPLE, LANGFUSE_*, MEMORY_API_KEY,
RUNS_API_KEY.

Code: `h.wrap(target, id=)` — nothing else. `tools=` only for local functions (OpenAI/Claude/
ReAct/callable); LangGraph builds with `await h.tools(my_fn, framework="langgraph")`. MCP tools =
whatever the agent's Bifrost virtual key allows; loaded automatically; per-turn narrowed by hints.
Removed wrap params: `memory=`, `approve=`, `tool_hints=`. `ask()` loses `ui=` (inferred) and
gains `diff=(before, after)`; `assignee` defaults to the run's user. New `h.inbox(assignee=None)`.

## B. Automation rules

- Memory: on iff MEMORY_URL; read-only iff the key's role is read-only (memory side).
- Risk tier per tool: MCP annotations `readOnlyHint` → read, `destructiveHint` → irreversible,
  else write; overridden by catalog `side_effects`; `approve_when` expression in the catalog
  (admin-set, or an accepted approval suggestion) → ask when true. read = run; write = run +
  notify event; irreversible = ask. bifrost-sdk `ToolDef` must carry MCP annotations; harness
  publishes MCP tools to the catalog too.
- Tool hints: always requested when the toolbox has ≥ 5 tools; candidates also filter which
  tool schemas are sent to the LLM (always keep: memory tools + top-k + tools used this run).
- Code Mode: auto when a source has ≥ 20 tools or ≥ 3 servers and all are read-tier.
- Evals: judge runs in the memory service; sample rate + judge model in the tenant LLM policy
  (default sample 0.1, capped by budget); harness just calls verify on sampled runs.
- Outcome: derived from run status + judge + human feedback; never "no exception = success".
- Schedules: idempotent upsert on (agent_id, on_behalf_of, cron, hash(input)).
- Notifications: tenant webhook subscriptions live in agent-runs (run.paused/escalated/finished).

## C. Memory service changes

- `/v1/context`: add `format=prompt` (default for the SDK/harness) returning only
  `{rendered, bundle_id, token_estimate}`; full JSON only with `format=full`. Replace
  `[memory_id:mem_…]` tags in `rendered` with per-bundle short handles `[m1]..`, resolvable
  server-side (update/forget/verify accept them within the bundle). `window: bool` (false when the
  framework keeps its own history — LangGraph checkpointer, OpenAI session). Absolute relevance
  floor (no junk filling the budget). Procedures section only when the agent has tools.
  Tools section renders only `next` + `prefill` (keyed `tool.arg`) + `missing`. Fix the tools cost
  estimate. Summary truncated (not dropped) when over its share. Remove: `answer`,
  `since_revision`/`delta`/`removed_ids`/`revision`, `use_llm` (policy decides), diagnostics
  unless `debug=true`.
- `/v1/recall` items → `{id, kind, text, observed_on, citation, document_id?, page?}`; extras only
  with debug.
- `/v1/messages` accepts a batch `messages: [...]`; `role:"event"` replaces `/v1/observations`
  (REMOVE observations endpoint + SDK observe + hints); thread optional → default thread = run id
  when a run id is in scope.
- Threads: remove `POST /v1/threads` → `PATCH /v1/threads/{id}` (title/metadata); summary becomes
  a field of `GET /v1/threads/{id}` (remove the `/summary` route + SDK summary()).
- Profile: remove PUT; PATCH with optional `old` (empty = replace). Briefs → REMOVED (a profile
  block with `source_query`, refreshed by the background job, replaces them).
- Feedback: targets RUN, MEMORY, TOOL_CALL, PROCEDURE (ANSWER merged into RUN, BRIEF removed).
  `POST /v1/runs/{id}/outcome` REMOVED — outcome = projection of feedback + run status (harness
  sends RUN feedback with source=system from final status; judge/human override by precedence
  human > judge > system).
- Judge: `/v1/verify {bundle_id, answer, run_id}` is the one judge; it writes RUN feedback
  (source=judge) itself; model from tenant policy `models.grounding_judge`.
- Evals: table `eval_daily(tenant, agent_id, day, source, n, score_sum, rejects)` upserted per
  feedback; `GET /v1/evals?agent_id=&since=&until=` (daily series + worst recent runs);
  Prometheus `trellis_eval_score{agent,source}` on /metrics; ship `deploy/grafana/trellis.json`
  (eval scores, LLM usage, run outcomes, latency).
- Tools: catalog entries gain `approve_when` and `annotations`; `POST
  /v1/tools/approval-suggestions/{id}/accept` writes approve_when into the catalog.
- Graph: fold `/v1/graph/query` into `GET /v1/graph/entities?q=` + `/entities/{id}?depth=`.
- Model keys: levels agent + tenant only (workspace level removed); policy tenant-only with
  `uses`, `read_assist`, `models: {use: model}`, `eval_sample`.
- Remove: memory webhooks (all 7 routes, tables, jobs, SDK) — notifications live in agent-runs;
  groups (6 routes). Keep keys, workspaces, members, tenants, reads, documents, jobs (admin).
- Agent tools (5 only): `memory_search(query, kinds?, time_from?, time_to?, k?)` (kinds includes
  `message` = history; time filter before ranking; compact schemas), `memory_remember`,
  `memory_update(id, content)`, `memory_forget(id)`, `profile_edit(block, old?, new)`,
  `tool_search(task)` (harness injects the run's toolbox; returns `{next, plan:{title, steps},
  prefill, missing}`). Remove history_search, procedures_search, record_outcome, `invalidate`.
- Env → code constants: LLM MAX_TOKENS/TIMEOUT/MAX_RETRIES/FAST_USES/USES/ENABLED/MODEL/
  FAST_MODEL, HINDSIGHT timeouts/concurrency/bank, EMBEDDING_THREADS (cpu count), OTEL_EXPORTER
  (derived), AUTHENTICATION__MODE (derived), RATE_LIMIT_PER_MINUTE (tenant quota covers it).
  LLM base url env renamed `BIFROST_URL`.
- SDK verbs: context, remember, update, forget, search, history, feedback, record_tool,
  tool_hints, agent_tools, call_agent_tool, profile; `verify` sends bundle_id; keep `agent()`,
  remove `derive`; advanced: documents, graph, tools(catalog, suggestions), model_keys
  (agent/tenant), memories, job, tenant/admin/keys/workspaces/reads.
- Regenerate openapi.json (stale /v1/files).

## D. agent-runs changes

- `GET /v1/runs` returns summaries `{run_id, agent_id, status, awaiting, assignee, deadline,
  updated_at}` (no checkpoint/input/output); full record via GET /v1/runs/{id}.
- Remove `/lineage`. Schedules: POST is an upsert on (agent_id, on_behalf_of, cron, input hash);
  pause/resume → `PATCH {paused}`.
- Webhooks: tenant subscriptions (`/v1/webhooks` CRUD, per-subscription secret, events
  run.paused/run.escalated/run.finished) replace per-run/per-schedule `webhook_url` and the global
  signing secret.
- Auth: validate `TRELLIS_API_KEY` against the memory service's key registry (introspection
  endpoint `GET /v1/keys/self` on memory, cached per key with TTL) — one key system. Remove
  RUNS__SERVICE__API_KEYS.

## E. bifrost-sdk

`ToolDef.annotations` (readOnlyHint, destructiveHint, idempotentHint, openWorldHint) from the
gateway listing when present (verify live); if the gateway strips them, fall back to catalog.

## G. Observability & evals — REVISED (supersedes eval items in A–C)

- Langfuse is the eval system of record: LLM-as-judge evaluators (judge model chosen per
  evaluator in Langfuse), human annotation queues, scores, datasets, experiments, dashboards —
  per agent / per request via trace attributes. We build none of that.
- Harness emits OTel with GenAI semantic conventions + trace attributes (agent_id = trace name/
  tag, user_id, session_id = thread, run_id, tenant). One exporter → OTel Collector; the collector
  routes all spans to Datadog (end-to-end) and GenAI/agent spans to Langfuse. App env: only
  `OTEL_EXPORTER_OTLP_ENDPOINT` (+ headers). Remove GroundedJudge LLM stage, JudgeBudget,
  DatasetBuilder, ExperimentRunner, harness JUDGE_MODEL; RegressionGate reads Langfuse
  experiment scores (thin CLI) or is removed if Langfuse CI integration covers it.
- Memory keeps only what Langfuse can't do: `/v1/verify` evidence grounding (it owns the
  evidence) — the harness pushes its result as a Langfuse score on the trace; and feedback as a
  LEARNING signal (confidence, procedures, approvals). `h.feedback(...)` fans out to Langfuse
  score + memory feedback in one call. REMOVE from addendum: eval_daily, GET /v1/evals,
  Prometheus eval metric, Grafana eval dashboard, tenant policy `eval_sample` /
  `models.grounding_judge` (grounding_judge LLM model stays a normal per-use model in policy).
- Storage placement check (must verify): chat messages Postgres (truth) + Dragonfly hot cache +
  archive segments to blob (GCS when BLOB__PROVIDER=gcs); thread summaries/profile/procedures in
  Postgres; embeddings in Qdrant; documents + tool outputs > 4 KB + ask tables in blob (move ask
  tables out of the runs checkpoint into blob via payload_ref); run checkpoints stay small.

## H. Acceptance (definition of done — owner requirement)

Every response must be correct *in value*, judged from the perspective of the user query that
flowed through a harness agent — not "200 OK". Acceptance suite (live, fresh stack, repeatable):
1. Golden scenario corpus: realistic multi-session users (chat support, procurement co-work over
   days with table review + approval, nightly scheduled report, ReAct research, manual LangGraph
   node), multilingual (en, de, es, hi, ja, zh), documents + tool runs. Each scenario lists user
   queries with EXPECTED facts (ids of seeded memories/docs/tool results that must appear, facts
   that must NOT appear, expected tool candidates/prefill/missing, expected summary/profile
   content, expected interrupt/resume behaviour).
2. Every query is sent through `h.wrap(agent).run(...)` (all five target kinds). Assert on: the
   injected context (relevant items present, irrelevant absent, size ≤ budget), memory-tool pull
   results, tool hints, the agent's final answer (grounded; deterministic scripted model where the
   answer shape must be exact, real model via Bifrost for a live pass), transcript/tool/feedback
   records written, and learning effects visible on the next turn (procedure, profile, prefetch,
   approval suggestion, summary).
3. Endpoint acceptance matrix generated from each OpenAPI: per endpoint — intended value,
   isolation denials (tenant/user/agent/visibility), invalid input, idempotency, exact field set
   vs consumers. No endpoint untested.
4. Placement checks: rows in Postgres, vectors in Qdrant, hot cache in Dragonfly, blobs in (fake)
   GCS; traces in Langfuse (agent spans + scores) and the collector (all spans).
5. Report: every scenario/query/endpoint → pass/fail with evidence (actual vs expected values).
   Any failure is fixed at the root cause and the suite re-run until fully green twice.

## F. Harness bugs (being fixed in harness pass 2)
LangGraph context message growth; outcome overwrite; thread default = run id; transcript on
pause/failure; durable ask tables; idempotent schedule; h.tools memory mode; tool_search toolbox.
