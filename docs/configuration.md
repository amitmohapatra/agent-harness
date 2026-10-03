# Configuration

The harness reads the environment and nothing else — no YAML, no keyword arguments on
`Harness()` besides `config=Settings(...)` (the same fields, for tests and embedding), no
per-agent options on `wrap` beyond the agent's id and its own local tools. Unset means "not in
this deployment". Every variable, with a one-line description, is in
[`.env.example`](../.env.example); `tests/unit/test_settings.py` checks the file lists exactly
what is read.

| Variable | Unset means |
|---|---|
| `BIFROST_URL` | no MCP tools, no `ReAct` model names |
| `BIFROST_VIRTUAL_KEY` | gateway calls without a key (the gateway's own policy decides); no memory model key registered |
| `TRELLIS_API_KEY` | calls to the memory service and agent-runs without a key |
| `MEMORY_URL` | memory off: no context, no memory tools, no records, no catalog (tiers are the tools' own) |
| `RUNS_URL` | runs, the queue and schedules kept in process (needs `MEMORY_URL`: agent-runs accepts the memory service's keys, and the tenant comes from there) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no OTLP export |
| `OTEL_EXPORTER_OTLP_HEADERS` | no OTLP headers; no Langfuse scores API |

## Who the deployment is

Not configured: the memory service says it about `TRELLIS_API_KEY`
(`GET /v1/keys/self` → `{key_id, tenant_id, principal, role, may_act_as}`), asked once per
process.

* **Tenant** — the key's own. A platform key (no tenant) names one per call (`tenant=` on
  `run`/`stream`/`start`/`schedule`); a tenant key refuses any other. Without a memory
  service the tenant is `default`.
* **Memory model key** — `BIFROST_VIRTUAL_KEY` is registered as each agent's model key once
  per process (idempotently): the memory service's LLM work for the agent runs on it.

Everything else — limits, timeouts, the tool-hint threshold, the grounding sample, lease
length — is a named constant next to the code that uses it.
