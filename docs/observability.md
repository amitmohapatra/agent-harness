# Observability

## Traces and metrics

The harness uses the OpenTelemetry API: a `trellis.run` span per attempt (run id, attempt,
agent, framework, tenant) and a `trellis.tool` span per tool call; counters `trellis.runs`
(by outcome), `trellis.tool_calls` (by status), `trellis.writes.failed`, and the
`trellis.judge.score` histogram. With no SDK installed they cost nothing.

With `OTEL_EXPORTER_OTLP_ENDPOINT` set, `Harness()` installs a tracer provider exporting over
OTLP/HTTP (the `[otel]` extra). With `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and
`LANGFUSE_HOST`, default `https://cloud.langfuse.com`) it adds an exporter to Langfuse's OTLP
endpoint with basic auth — no Langfuse SDK. An application that already installed a provider
keeps it.

## Redaction

Every span attribute passes `trellis.harness.redaction`: names that look like secrets
(`api_key`, `password`, `token`...) and values that look like credentials (bearer tokens, JWTs,
`sk-...`) become `[redacted]`, e-mail addresses are masked, long values are cut at 2000
characters.

## Events

`agent.stream` yields contracts `RunEvent`s: `RUN_STARTED`, `CONTEXT_LOADED`,
`TEXT_MESSAGE_*`, `TOOL_CALL_*`, `CUSTOM` (`tool_notice`, `log` from `current().log(...)`,
`warning`), `INTERRUPT`, `RUN_ERROR`, `RUN_FINISHED`. A failed background write is a `warning`
event on the run's listeners, a log line and a `trellis.writes.failed` count — never silent,
never raised into the run.
