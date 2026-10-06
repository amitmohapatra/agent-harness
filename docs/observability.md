# Observability and evaluation

Langfuse is the eval system of record: scores, datasets and dataset runs, human annotation
queues and dashboards — per agent and per request, through the trace attributes below. The
harness emits OpenTelemetry with the GenAI semantic conventions, puts the memory service's
grounding score and people's feedback on the run's trace, and runs evaluations into Langfuse —
offline over a dataset, online on sampled runs ([evaluation.md](evaluation.md)). Datadog gets
every span end to end through a collector.

## Spans

The harness uses the OpenTelemetry API (`telemetry.py`); with no SDK installed the spans cost
nothing (attributes are built only for a recording span).

| Span | Attributes |
|---|---|
| `invoke_agent <agent>` — one per attempt | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.conversation.id` (the thread), `langfuse.observation.type=agent`, the trace attributes below, `langfuse.observation.input` (the question), `langfuse.observation.output` (the answer); with an agent version (`h.wrap(..., version=)`, `TRELLIS_AGENT_VERSION`), `gen_ai.agent.version` and `langfuse.version` |
| `retrieve memory` — the pushed context | `gen_ai.operation.name=retrieve`, `langfuse.observation.type=retriever`, input (the question), output (the rendered context) |
| `execute_tool <tool>` — one per call | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type` (`extension` for MCP, else `function`), `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `langfuse.observation.type=tool`, `trellis.tool.source`, `trellis.governance.action` (`run`, `announce` or `ask`: [governance.md](governance.md)) |
| `chat <model>` — a model call the harness makes (`ReAct`) | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `langfuse.observation.type=generation`, input/output |
| `score <name>` — a grounding score, an evaluator's or feedback | `langfuse.observation.type=evaluator`, `trellis.run_id` (when the score is a run's), `trellis.score.name`, `.value`, `.comment`, and a `score` event; in the run's trace, or the trace an `EvalCase` names (`trace_id`) |
| `invoke_agent <name>` — a callable's item in `evaluate(callable, ...)` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `langfuse.trace.name`, `langfuse.observation.type=agent`, `user.id`, `trellis.run_id`, input/output, and the experiment attributes ([evaluation.md](evaluation.md#each-run-is-an-item-of-a-langfuse-experiment-v3-self-hosted-and-v4)) |

A framework's own model calls are its instrumentation's (LangChain, OpenAI Agents and Claude
all have OTel GenAI instrumentations); they nest under the attempt's span. A framework's own
tools (Deep Agents' file tools, Claude Code's built-ins, an OpenAI `function_tool` of your own)
are not harness tools and get no `execute_tool` span of the harness's
([framework pages](README.md#which-target)).

### Trace attributes (Langfuse's mapping, verified against its OTel docs)

| Attribute | Langfuse field |
|---|---|
| `langfuse.trace.name` = the agent id | trace name |
| `user.id` = the run's user | user id |
| `session.id` = the thread (or the run id) | session id |
| `langfuse.trace.tags` = `[agent id, framework]` | tags |
| `langfuse.trace.metadata.run_id`, `.tenant`, `.framework` | filterable trace metadata |
| `trellis.run_id`, `trellis.tenant`, `trellis.attempt` | span attributes (Datadog facets) |

**One trace per run.** The trace id is derived from the run id (the first 128 bits of its
SHA-256, `telemetry.trace_id_of`): every attempt — a resume in another process, a worker
continuing a queued run — joins the trace the run started, and a score computed later, or a
person's feedback days later, lands on it without anything stored.

### Counters

| Counter | Attributes | Counted when |
|---|---|---|
| `trellis.runs` | `agent`, `outcome` (`success`, `error`, `timeout`, `interrupt`, `cancelled`) | an attempt ends |
| `trellis.tool_calls` | `tool`, `status` (`ok`, `error`, `timeout`) | the bridge executed a call (a replayed or rejected call is not counted) |
| `trellis.writes.failed` | `write` (the background write's label, e.g. `memory.transcript`) | a background write failed or the write queue was full |
| `trellis.writes.undelivered` | `write`, `outcome` (`spooled`, `lost`) | a background write this process gave up on |
| `trellis.runs.queue_wait` (histogram, seconds) | `agent` | a worker claimed a queued run: how long it waited (since it was queued, or since the answer that queued it again) |
| `trellis.runs.rate_limited` | `operation` (`start`, `resume`, `schedule`) | agent-runs still answered `429` after its SDK's retries ([runs.md](runs.md#admission-agent-runs-rate-limit)) |
| `trellis.run_events.undelivered` | | run events agent-runs' event log did not take |
| `trellis.notifications` | `provider`, `outcome` (`sent`, `failed`) | a pause was told to a notifier ([interrupts.md](interrupts.md#telling-people)) |

With `OTEL_EXPORTER_OTLP_ENDPOINT` set, `Harness()` also installs a meter provider exporting them
over OTLP/HTTP to `<endpoint>/v1/metrics` every 60 s (and on close) — unless the endpoint is
Langfuse's (it takes traces only) or the application installed a meter provider itself, which
it keeps. Otherwise they go wherever the application's provider sends them. agent-runs' own
Prometheus metrics (claims, rate-limited requests, the ticker) are at its `/metrics`.

## Logs

The harness logs through `logging` (`trellis.*` loggers; the run's id and the like as `extra`
fields) and leaves the configuration to the application. The worker it starts itself
(`python -m trellis.harness.worker`) logs JSON lines to stderr — `time`, `level`, `logger`,
`message`, every `extra` field, `exception` — when stderr is not a terminal (a container, a
service manager), and text when it is: nothing to set. An application installs the same
formatter with `handler.setFormatter(trellis.harness.logs.JSONFormatter())`.

## Export

`OTEL_EXPORTER_OTLP_ENDPOINT` (+ `OTEL_EXPORTER_OTLP_HEADERS`) makes `Harness()` install a
tracer provider with one OTLP/HTTP exporter (the `[otel]` extra) to `<endpoint>/v1/traces` —
unless the application installed a provider itself, which it keeps. Two deployments:

* **Straight to Langfuse** — `OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel`,
  `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 pk:sk>,x-langfuse-ingestion-version=4`.
* **Through the collector** ([deploy/otel-collector.yaml](../deploy/otel-collector.yaml)) —
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`,
  `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 pk:sk>,x-langfuse-host=https://cloud.langfuse.com`.
  The collector sends **every** span to Datadog (with the Datadog connector's APM stats), and
  only the GenAI/agent spans — those with `gen_ai.operation.name` or
  `langfuse.observation.type` — to Langfuse (a `filter` processor). It forwards the caller's
  `Authorization` header to Langfuse (`headers_setter`, batches keyed by it), so one collector
  serves several Langfuse projects. Collector env: `LANGFUSE_OTLP_ENDPOINT`, `DD_API_KEY`,
  `DD_SITE`.

Spans are exported in batches every few seconds; closing the harness (`async with Harness()`
ending, `aclose()`, a worker's shutdown) first exports what is still queued, so a short script
or a stopping worker does not leave its last runs' traces behind.

## Scores

Langfuse ingests scores through its public API (`POST /api/public/scores`), not through OTLP.
The harness uses the credentials the OTLP headers already carry — no extra variable:

* when `OTEL_EXPORTER_OTLP_HEADERS` has `Authorization=Basic …` **and** the endpoint is
  Langfuse's (`…/api/public/otel…`) or the headers name its host (`x-langfuse-host`, which the
  collector ignores), a score is posted to `<host>/api/public/scores` on the run's trace, with
  an idempotency id (`<run_id>:<name>`);
* always, the score is also a `score <name>` span in the run's trace — what Datadog and any
  other backend see, and all there is when Langfuse cannot be reached that way.

| Score | When | Value |
|---|---|---|
| `grounding` | a sampled share of successful runs with an answer — a structured one as its JSON (`TRELLIS_GROUNDING_SAMPLE`, default 0.1, chosen by the run id): the memory service's `/v1/verify {bundle_id, answer, run_id}` against the context the run was given — the one grounding judge, which owns the evidence and records the run's `judge` feedback itself | the share of the answer's claims the evidence supports, 0..1 (an answer with no checkable claim is no score) |
| an evaluator's name (`exact_match`, `llm_judge`, ...) | `h.evaluate` or `evaluate` (every item of a dataset), an online judge (`Harness(judges=[...])`, a sampled share of runs), or `judge(...)` from code you do not wrap — [evaluation.md](evaluation.md), [blocks/evaluation.md](blocks/evaluation.md) | 0..1 (`NUMERIC`), a bool (`BOOLEAN`), or a category (`CATEGORICAL`); the evaluator's comment (a judge's reasoning) |
| `feedback` | `h.feedback(run_id, verdict, correction=None)` — also the run's `human` feedback in the memory service, which outranks the judge's and the run's own | confirm/approve 1.0, edit 0.5, correct/reject 0.0; the correction as the comment |

LLM-as-judge on the traces and dataset runs against an agent are the harness's
(`Harness(judges=[...])`, `h.evaluate`, and `judge`, `evaluate` for code that is not wrapped —
[evaluation.md](evaluation.md)); annotation queues and
datasets built from traces are configured in Langfuse. Whether the *harness* got slower is `make bench`
(`tests/performance`, against the committed `benchmark-results.json`).

## Redaction

Everything a run sends out of the process passes `trellis.harness.redaction`, once, where it
leaves: every span attribute; on the run's event stream (`agent.stream`, and so `serve_chat`'s
AG-UI events, `serve_a2a`'s task updates and its push notifications), each tool call's
arguments and output and the data of `CUSTOM` events (`tool_notice`, `log`, `warning`); and the
tool calls recorded in the memory service (the arguments, the output, and the spooled copy of a
write that waits for the service). The tool and the model get the values as they are, and so
does what the run needs to continue: the journal and the interrupt kept in agent-runs. Two
things on the stream are not redacted: the answer (text deltas, the result) — what the user
asked for — and the pause (`INTERRUPT`, the interrupt in `RUN_FINISHED`) — the question, its
payload and the tool call to approve, which the person answering needs to see, as the inbox
shows them. The rules are the same everywhere and need no setting: names that look like secrets
(`api_key`, `password`, `token`...) and values that look like credentials (bearer tokens,
JWTs, `sk-...`, long key-like blobs) become `[redacted]`, e-mail addresses are masked, long
values are cut at 2000 characters. Names are matched by their words, not substrings
(`gen_ai.usage.input_tokens` and `tokenizer` stay; `refresh-token` and `X-Api-Key` go), and a
number is never treated as a credential. Mappings and lists are redacted recursively, and so
is text that is a JSON object or array (a model's tool-call arguments, a tool result as the
model read it, in a `chat` span's input and output): it stays JSON text, redacted. Bytes
become `<n bytes>`, anything else its JSON or its text. The redactor is the contracts
`TelemetryRedactor`; it never raises into a run.

Redaction of your own — a domain's identifiers, what the model itself must not read — is a
hook: `after_tool` returns the outcome masked (what the model reads, the journal and memory
keep), `before_model` the messages masked ([hooks.md](hooks.md),
[`examples/hooks.py`](../examples/hooks.py)).

## Events

`agent.stream` yields contracts `RunEvent`s: `RUN_STARTED`, `CONTEXT_LOADED`,
`TEXT_MESSAGE_*`, `TOOL_CALL_*`, `CUSTOM` (`tool_notice`, `log` from `current().log(...)`,
`warning`, `decision` — the person's decision an attempt goes on with: `decision`, `reviewer`,
`comment`, `remember`; or a call approved because its tool was approved for the run —, and
`notified`), `INTERRUPT`, `RUN_ERROR`, `RUN_FINISHED`. The decision is also on the attempt's
`invoke_agent` span (`trellis.decision.*` attributes and a `decision` span event). With
`RUNS_URL` the events are also in agent-runs' event log, readable from any process
(`agent.events`, [runs.md](runs.md#a-runs-events-from-anywhere)). A failed background write is a `warning`
event on the run's listeners, a log line and a `trellis.writes.failed` count — never silent,
never raised into the run.

Every tool call that starts on the stream (`TOOL_CALL_START`, `TOOL_CALL_ARGS`) ends there,
before the attempt's `RUN_FINISHED`: `TOOL_CALL_END`, then `TOOL_CALL_RESULT` with the call's
`status` (`ok`, `error`, `timeout`, `rejected`, `cancelled`) and `output`. A call the attempt
ends inside, on every adapter, ends the same way, with no output of its own — the result says
why:

| What cut it short | `status` | `output` |
|---|---|---|
| it asked a person (`current().ask`, an approval inside it): the run pauses | `paused` | `<tool> paused: the run waits on a person, and the call runs again when it resumes` |
| the run was cancelled (`agent.cancel`, A2A `tasks/cancel`, a person cancelling the question it asked) | `cancelled` | `<tool> was cut short: the run was cancelled` |
| the run's time limit (`timeout=`, `deadline=`) | `timeout` | `<tool> was cut short: the run ran out of time` |

A paused call runs again in the attempt that resumes the run: a new `TOOL_CALL_START` there.
A call its framework still runs when the attempt ends (one run in a task of the framework's
own, as Claude's in-process MCP server does, stopped only later) is ended then, before
`RUN_FINISHED`, the same way; its later stop adds nothing to the stream.
