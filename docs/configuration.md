# Configuration

The harness reads the environment and nothing else — no YAML, no keyword arguments on
`Harness()` besides `config=Settings(...)` (the same fields, for tests and embedding). Unset
means "not in this deployment". Every variable, with a one-line description, is in
[`.env.example`](../.env.example); `tests/unit/test_settings.py` checks the file lists exactly
what is read.

| Variable | Unset means |
|---|---|
| `TRELLIS_TENANT` | tenant `default` |
| `BIFROST_URL`, `BIFROST_VIRTUAL_KEY` | no `mcp()` tools, no `ReAct` model names, no LLM stage in the judge |
| `MEMORY_URL`, `MEMORY_API_KEY` | `memory=` must be `"off"`; no feedback, no grounded judge |
| `TRELLIS_MEMORY_MODEL_KEY` | no model key registered |
| `RUNS_URL`, `RUNS_API_KEY` | runs, the queue and schedules kept in process |
| `TRELLIS_EVAL_SAMPLE` | 0.1 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no OTLP export |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | no Langfuse export |

Everything else — limits, timeouts, the judge's model and budget, lease length — is a named
constant next to the code that uses it.
