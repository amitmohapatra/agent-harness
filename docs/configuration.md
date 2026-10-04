# Configuration

The harness reads the environment and nothing else — no YAML, no keyword arguments on
`Harness()` besides `config=Settings(...)` (the same fields, for tests and embedding) and the
online judges (`judges=[...]`: code that says *what* to judge; which model judges, through which
key and how often is the environment's), no per-agent options on `wrap` beyond the agent's id
and its own local tools. Unset means "not in
this deployment". Every variable, with a one-line description, is in
[`.env.example`](../.env.example); `tests/unit/test_settings.py` checks the file lists exactly
what is read.

| Variable | Unset means |
|---|---|
| `BIFROST_URL` | no MCP tools, no `ReAct` model names |
| `BIFROST_VIRTUAL_KEY` | gateway calls without a key (the gateway's own policy decides); no memory model key registered |
| `TRELLIS_API_KEY` | only allowed without `MEMORY_URL`: both services refuse a call without a key, so `Harness()` refuses `MEMORY_URL` (and `RUNS_URL`) without it (`ConfigurationError`), rather than a `401` at the first run |
| `MEMORY_URL` | memory off: no context, no memory tools, no records, no catalog (tiers are the tools' own) |
| `RUNS_URL` | runs, the queue and schedules kept in process (needs `MEMORY_URL`: agent-runs accepts the memory service's keys, and the tenant comes from there) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no OTLP export |
| `OTEL_EXPORTER_OTLP_HEADERS` | no OTLP headers; no Langfuse scores API |
| `TRELLIS_SPOOL_DIR` | memory writes this process cannot deliver are logged, counted and lost (set: kept in `<dir>/trellis-writes.jsonl` and replayed at the next start — [memory.md](memory.md#background-writes-what-is-guaranteed)) |
| `TRELLIS_WORKER_CONCURRENCY` | a worker executes as many runs at once as the machine has CPUs, from 1 to 8 (`--concurrency` on `python -m trellis.worker` and `concurrency=` on `h.worker` win over it) |
| `TRELLIS_JUDGE_MODEL` | `llm_judge` asks the judged agent's own model (a `ReAct`'s), and logs once that the judge shares it; an agent with no model the harness knows gets no judge score. Set it to a Bifrost model name — a **different, stronger model than the agent's** (a model grading itself is biased) — and the judge asks it through `BIFROST_URL` ([evaluation.md](evaluation.md#llm_judge)) |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the judge's calls go through `BIFROST_VIRTUAL_KEY`, on the agents' budget. Set it to a **separate virtual key** so evaluation spend is budgeted, limited and reported on its own |
| `TRELLIS_JUDGE_SAMPLE` | 0.1 when the harness has online judges (`Harness(judges=[...])`), nothing judged without; a number from 0 to 1 is the share of successful runs judged (by the run id) |
| `TRELLIS_GROUNDING_SAMPLE` | 0.1: a tenth of the successful runs with a text answer (and memory on) are checked against the context they were given (`/v1/verify`, a score on the trace — [observability.md](observability.md#scores)); `0` turns it off, `1` checks every run. A number from 0 to 1, else `Settings` refuses it (`ValidationError`); the run id decides, so a run is either always or never sampled |

`Harness(config=Settings(...))` takes the same deployment as fields, for tests and for
embedding (`Settings` is frozen and refuses unknown fields; `Settings.from_env(environ)` reads a
mapping instead of `os.environ`, and blank values count as unset):

| Field | Variable |
|---|---|
| `bifrost_url` | `BIFROST_URL` |
| `bifrost_virtual_key` | `BIFROST_VIRTUAL_KEY` |
| `api_key` | `TRELLIS_API_KEY` |
| `memory_url` | `MEMORY_URL` |
| `runs_url` | `RUNS_URL` |
| `otlp_endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `otlp_headers` | `OTEL_EXPORTER_OTLP_HEADERS`, parsed as the OTel spec writes it (`k1=v1,k2=v2`, values URL-decoded, keys lower-cased) |
| `spool_dir` | `TRELLIS_SPOOL_DIR` |
| `worker_concurrency` | `TRELLIS_WORKER_CONCURRENCY` (at least 1) |
| `grounding_sample` | `TRELLIS_GROUNDING_SAMPLE` (0 to 1, default 0.1) |
| `judge_model` | `TRELLIS_JUDGE_MODEL` |
| `judge_virtual_key` | `TRELLIS_JUDGE_VIRTUAL_KEY` |
| `judge_sample` | `TRELLIS_JUDGE_SAMPLE` (0 to 1; `None`: 0.1 with judges) |

`RUNS_URL` without `MEMORY_URL`, and `MEMORY_URL` without `TRELLIS_API_KEY`, are refused when the
`Harness` is built (`ConfigurationError`). The names are the platform's: agent-runs reads the
memory service at `MEMORY_URL` too, the memory SDK defaults to `MEMORY_URL` and
`TRELLIS_API_KEY`, and the gateway is `BIFROST_URL` everywhere.

## Who the deployment is

Not configured: the memory service says it about `TRELLIS_API_KEY`
(`GET /v1/keys/self` → `{key_id, tenant_id, principal, role, may_act_as}`), asked at first use
and again every 10 minutes. While the service cannot be reached the last answer stands (asked
again after 30 s), so an outage does not stop runs; a key the service refuses (`401`/`403`), or
one it could never be asked about (the first ask failed), raises `ConfigurationError` with what
went wrong.

* **Tenant** — the key's own. A platform key (no tenant) names one per call (`tenant=` on
  `run`/`stream`/`start`/`schedule`); a tenant key refuses any other. A development key
  (the memory service's `trusted_dev` mode) speaks for `default`, and without a memory
  service the tenant is `default` too — so local development needs no `tenant=`.
* **What it may write** — not the key's role: the memory service decides per scope (its
  relationship checks) and a refused write is a reported warning.
* **Memory model key** — `BIFROST_VIRTUAL_KEY` is registered as each agent's model key once
  per process (idempotently): the memory service's LLM work for the agent runs on it. A
  memory service that takes no model keys (its credential encryption is not configured) is
  logged once per process and not asked again.

Everything else — limits, timeouts, the tool-hint threshold, lease length — is a named
constant next to the code that uses it.
