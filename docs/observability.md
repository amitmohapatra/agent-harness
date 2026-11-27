# Observability

The harness emits OpenTelemetry with the GenAI semantic conventions: one trace per run, a span
per attempt, tool call and model call, with Langfuse's trace attributes, credentials in
every attribute masked; the memory service's grounding score and people's feedback go on the run's trace.
Langfuse reads the traces and is where scores, datasets and dashboards live; Datadog gets every
span end to end through a collector. Evaluation — offline over a dataset, online on sampled
runs — is [evaluation.md](evaluation.md).

## Spans

The harness uses the OpenTelemetry API (`telemetry.py`); with no SDK installed the spans cost
nothing (attributes are built only for a recording span).

| Span | Attributes |
|---|---|
| `invoke_agent <agent>` — one per attempt | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.conversation.id` (the conversation, as `session.id`), `langfuse.observation.type=agent`, the trace attributes below, `langfuse.observation.input` (the question), `langfuse.observation.output` (the answer); with an agent version (`h.wrap(..., version=)`, else `service.version`), `gen_ai.agent.version` and `langfuse.version`. How the attempt ended: `trellis.run.status`, what the run spent so far (`trellis.spent.tokens`, `.cached_tokens`, `.cache_written_tokens`, `.cost`, `.model_calls`, `.tool_calls`); a failure (`ERROR`, `TIMEOUT`) the span's error status, `error.type` (the error's code) and `langfuse.observation.level=ERROR` with `langfuse.observation.status_message`; ended short (the step limit or a budget: `PARTIAL`; `CANCELLED`, with its reason) `level=WARNING` and the message; a resumed attempt `trellis.resumed` and `trellis.decision.*` (below) |
| `retrieve memory` — the pushed context | `gen_ai.operation.name=retrieve`, `langfuse.observation.type=retriever`, input (the question), output (the rendered context) |
| `execute_tool <tool>` — one per call: a harness tool's **and a framework's own built-in's** (by adapter, below) | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type` (`extension` for MCP, else `function`), `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `langfuse.observation.type=tool`, `trellis.tool.source`, `trellis.governance.action` (`run`, `announce` or `ask` — `ask` when a person approved the call: [governance.md](governance.md)); a call that failed or timed out the error status, `error.type`, `trellis.tool.status` and level `ERROR`, what the model read its status message; one rejected (a person, a hook, a rule) level `WARNING`; one cut short by a pause `trellis.paused` (no error) |
| `invoke_agent <helper>` — a sub-agent the framework spawns itself (a Deep Agent's `task`, Claude's subagent, Kimi's `Agent`, an OpenAI agent called as a tool: a child run) | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `langfuse.observation.type=agent`, `trellis.run_id` (its child run), `trellis.parent_run_id`, `trellis.helper.framework`, `.kind`, input (its task), output (its answer), `trellis.run.status`; its own tool calls and model replies nest under it. A harness sub-agent ([subagents.md](subagents.md)) is a run of its own: its attempt span is under the parent's `execute_tool` span, with `trellis.parent_run_id` |
| `chat <model>` — a model call of a graph with the harness's `ModelHooks` middleware (a `ReAct`'s) | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens` and `gen_ai.usage.cache_creation.input_tokens` (when the prompt cache was used), `gen_ai.usage.cost` (when the gateway said what it cost), `gen_ai.response.finish_reasons`, `langfuse.observation.type=generation`, input/output |
| `chat <model>` — a model reply Claude Code, Kimi Code or the OpenAI Agents SDK made and reported (the harness does not make these calls) | `langfuse.observation.type=generation`, `gen_ai.request.model`, the `gen_ai.usage.*` above (cache reads and writes included), `gen_ai.usage.cost` where the framework reports one (Claude's `total_cost_usd`), `trellis.model_calls` (a Claude query's result counts its turn's replies); timed when reported |
| `interrupt` — the run paused for a person (`langfuse.observation.type=event`, under the attempt) | `trellis.interrupt.id`, `.reason` (`APPROVAL`, `QUESTION`...), `.question` (also its input), `.assignee` (who is asked, when a hook named them), `.tool` (the call), `.component`, `.options` |
| `decision` — the person's answer the next attempt goes on with (an event under it) | `trellis.decision.interrupt_id`, `.decision` (`APPROVE`, `EDIT`, `REJECT`...), `.reviewer`, `.comment`, `.remember`, `.choice` (`once`, `conversation`, `always`, `deny`; also its output) |
| `partial`, `timeout`, `cancelled`, `error` — how an attempt ended short (an event under it) | `error.type`, the level and status message the attempt has |
| `score <name>` — a grounding score, an evaluator's or feedback | `langfuse.observation.type=evaluator`, `trellis.run_id` (when the score is a run's), `trellis.score.name`, `.value`, `.comment`, and a `score` event; in the run's trace, or the trace an `EvalCase` names (`trace_id`) |
| `invoke_agent <name>` — a callable's item in `evaluate(callable, ...)` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `langfuse.trace.name`, `langfuse.observation.type=agent`, `user.id`, `trellis.run_id`, input/output, and the experiment attributes ([evaluation.md](evaluation.md#each-run-is-an-item-of-a-langfuse-experiment-v3-self-hosted-and-v4)) |

Every attribute name above follows the OTel GenAI semantic conventions or Langfuse's own
OTel mapping (`langfuse.observation.type`, `.level`, `.status_message`, `.input`, `.output`,
`langfuse.trace.metadata.*`; `gen_ai.usage.*` and `gen_ai.usage.cost` are a generation's
usage and cost): [Langfuse: OpenTelemetry, property mapping](https://langfuse.com/integrations/native/opentelemetry),
[observation types](https://langfuse.com/docs/observability/features/observation-types),
[log levels](https://langfuse.com/docs/observability/features/log-levels). Langfuse prices
and sums usage on generations only, so the run's spend is trace metadata, not a second count.

### What reaches Langfuse, by adapter

Every adapter: the attempt span (its outcome and spend), `retrieve memory`, an `execute_tool`
span for each harness tool (your MCP servers' tools included), the `interrupt` and `decision`
events of an approval or a question, the ending events, the scores, and a harness sub-agent's
spans under its call. Beyond those, each framework's own steps:

| Adapter | Its own tool calls (`execute_tool`) | Its sub-agents | Its model calls |
|---|---|---|---|
| `ReAct`, `create_agent` with `trellis.middleware()` | Deep Agents' file tools, `execute`, `task`; the provider's server tools (web search...) | `task` sub-agents: `invoke_agent <name>` | `chat` per call, with usage and cost (`ModelHooks`) |
| a LangChain or LangGraph graph of your own | every tool of the graph (a `ToolNode`'s, a LangChain `@tool`), its state tools too | Deep Agents' `task` | the framework's instrumentation (LangChain's OTel, Langfuse's callback) |
| Deep Agents (`create_deep_agent`) | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `write_todos`, `execute`, `task` | `task` sub-agents, their calls under them | its instrumentation |
| OpenAI Agents SDK | your `FunctionTool`s, hosted tools (file search, web search, code interpreter, image generation, hosted MCP), local shell, computer and sandbox actions | an agent called as a tool | `chat` per reply, with usage |
| Claude Agent SDK (Claude Code) | every built-in (`Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob`, `WebFetch`, `WebSearch`, `Task`/`Agent`...), its MCP servers' tools | its subagents (`agents=`, `Task`) and workflows | `chat` per query result, with usage, cache tokens and cost |
| Kimi Code (ACP) | every tool it runs (its shell, file, search and `Agent` tools) | its `Agent` sub-agents and swarm members | `chat` per request its usage log records |
| a function target | its `agent.tools.call(...)` calls | `h.wrap` sub-agents | its own instrumentation |

**Every shell command is its call's span.** A shell call's `execute_tool` span (the harness's
`sandbox_exec`, Deep Agents' `execute`, Claude Code's `Bash` and `Monitor`, Kimi Code's `Bash`,
OpenAI Agents' `shell` and `local_shell`) also says `trellis.command.text` (the command,
redacted as any tool input is), `trellis.command.where` (`sandbox`: the run's sandbox; `host`:
this machine's isolation layer; `framework`: the framework ran it on its own machine),
`trellis.command.exit_code`, `trellis.command.duration_ms`, `trellis.command.limit` when a
limit stopped it (`time`, `cpu`, `memory`, `processes`, `file`... — level `WARNING`, "stopped
by its memory limit") and `trellis.command.sandbox`; its output is the call's result, as any
tool's (redacted, cut at 2000 characters). A command the harness ran — in the run's sandbox,
or through the isolation layer — is put there from its audit record (`telemetry.command`: the
isolation layer's `sandbox_command` record, the one its audit log and the run's stream carry,
is put on the span as it is emitted), so the span and the audit log say the same; one the
framework ran is said as the framework reported it (its duration, an exit code its result
names, the time limit when it timed out).

A built-in's `trellis.governance.action` is what governance decided before the call ran
(`bridge.permitted`): `ask` when a person approved it, `announce` or `run` otherwise; a hosted
tool the provider ran inside the model's step was decided by nothing, and is `run`.

### What goes elsewhere

* **Metrics** (the counters below) go to the meter provider — a collector over OTLP — never to
  Langfuse, which takes traces only.
* **The run's audit record** — status, input, output, error, the interrupt kept for a person,
  the spend, its parent and children — is agent-runs' (`RunRecord` and its event log); the
  trace carries the run id that finds it.
* **The memory service** keeps the transcript and the tool records, values as they are; a
  background write that failed is a `write_failed <label>` span here, one that worked is only
  the memory service's.
* **Logs** go where the application's `logging` sends them ([Logs](#logs)).

### Trace attributes (Langfuse's mapping, verified against its OTel docs)

| Attribute | Langfuse field |
|---|---|
| `langfuse.trace.name` = the agent id | trace name |
| `user.id` = the run's user | user id |
| `session.id` = the conversation (`h.conversation(user=, thread=)`: the tenant's, the user's and the thread's, so two users' threads named alike are two sessions), or the run id | session id |
| `langfuse.trace.tags` = `[agent id, framework]` | tags |
| `langfuse.trace.metadata.run_id`, `.tenant`, `.framework` | filterable trace metadata |
| `langfuse.trace.metadata.status`, `.spent_tokens`, `.spent_cached_tokens`, `.spent_cache_written_tokens`, `.spent_cost`, `.spent_model_calls`, `.spent_tool_calls` (a run's own attempt sets them as it ends) | the run's outcome and spend, filterable |
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
| `trellis.writes.failed` | `write` (the background write's label, e.g. `memory.transcript`) | a background write was given up (its attempts spent, the queue full): spooled or lost |
| `trellis.writes.undelivered` | `write`, `outcome` (`spooled`, `lost`) | a background write this process gave up on |
| `trellis.runs.queue_wait` (histogram, seconds) | `agent` | a worker claimed a queued run: how long it waited (since it was queued, or since the answer that queued it again) |
| `trellis.runs.rate_limited` | `operation` (`start`, `resume`, `schedule`) | agent-runs still answered `429` after its SDK's retries ([runs.md](runs.md#admission-agent-runs-rate-limit)) |
| `trellis.run_events.undelivered` | | run events agent-runs' event log did not take |

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

An OpenTelemetry exporter variable (`OTEL_EXPORTER_OTLP_ENDPOINT`, or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`; with `OTEL_EXPORTER_OTLP_HEADERS`) makes `Harness()`
install a tracer provider with the SDK's OTLP/HTTP exporter (the `[otel]` extra), which reads
those variables itself — unless the application installed a provider itself, which it keeps
(never replaced). The resource is `Resource.create()`: `OTEL_SERVICE_NAME` and
`OTEL_RESOURCE_ATTRIBUTES` as the SDK reads them, `service.name=trellis-harness` only when they
name none. Two deployments:

* **Straight to Langfuse** — Langfuse's own names, nothing else:
  `LANGFUSE_PUBLIC_KEY=pk-lf-...`, `LANGFUSE_SECRET_KEY=sk-lf-...` and, unless the project is
  on Langfuse Cloud, `LANGFUSE_BASE_URL=https://langfuse.example.com`. With no OpenTelemetry
  exporter variable set the harness
  exports where Langfuse's SDK does: `<LANGFUSE_BASE_URL>/api/public/otel` with
  `Authorization: Basic base64(pk:sk)` and `x-langfuse-ingestion-version: 4`. The same keys
  post its scores and read its datasets.
* **Through the collector** ([deploy/otel-collector.yaml](../deploy/otel-collector.yaml)) —
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`,
  `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 pk:sk>,x-langfuse-host=https://cloud.langfuse.com`.
  The collector sends **every** span to Datadog (with the Datadog connector's APM stats), and
  only the GenAI/agent spans — those with `gen_ai.operation.name` or
  `langfuse.observation.type` — to Langfuse (a `filter` processor). It forwards the caller's
  `Authorization` header to Langfuse (`headers_setter`, batches keyed by it), so one collector
  serves several Langfuse projects. Collector env: `LANGFUSE_OTLP_ENDPOINT`, `DD_API_KEY`,
  `DD_SITE`. A set exporter variable always wins over the Langfuse keys for the traces; the
  scores, datasets and prompts still use Langfuse's own names (set them too).

Spans are exported in batches every few seconds; closing the harness (`async with Harness()`
ending, `aclose()`, a worker's shutdown) first exports what is still queued, so a short script
or a stopping worker does not leave its last runs' traces behind.

## Scores

Langfuse ingests scores through its public API (`POST /api/public/scores`), not through OTLP.
The harness uses the credentials the OTLP headers already carry (derived from the Langfuse keys
when they are all that is set) — no extra variable:

* when the OTLP headers have `Authorization=Basic …` **and** the endpoint is
  Langfuse's (`…/api/public/otel…`) or the headers name its host (`x-langfuse-host`, which the
  collector ignores), a score is posted to `<host>/api/public/scores` on the run's trace, with
  an idempotency id (`<run_id>:<name>`);
* always, the score is also a `score <name>` span in the run's trace — what Datadog and any
  other backend see, and all there is when Langfuse cannot be reached that way.

| Score | When | Value |
|---|---|---|
| `grounding` | a sampled share of successful runs with an answer — a structured one as its JSON (a fixed 0.1, chosen by the run id): the memory service's `/v1/verify {bundle_id, answer, run_id}` against the context the run was given — the one grounding judge, which owns the evidence and records the run's `judge` feedback itself | the share of the answer's claims the evidence supports, 0..1 (an answer with no checkable claim is no score) |
| an evaluator's name (`exact_match`, `llm_judge`, ...) | `h.evaluate` or `evaluate` (every item of a dataset), an online judge (`h.wrap(judges=[...])`, its sampled share of runs), or `judge(...)` from code you do not wrap — [evaluation.md](evaluation.md), [blocks/evaluation.md](blocks/evaluation.md) | 0..1 (`NUMERIC`), a bool (`BOOLEAN`), or a category (`CATEGORICAL`); the evaluator's comment (a judge's reasoning) |
| `feedback` | `h.feedback(run_id, verdict, correction=None)` — also the run's `human` feedback in the memory service, which outranks the judge's and the run's own | confirm 1.0, correct/reject 0.0; the correction as the comment |

How evaluators and judges produce the evaluator scores is [evaluation.md](evaluation.md).
Whether the *harness* got slower is `make bench` (`tests/performance`, against the committed
`benchmark-results.json`).

## Redaction

Traces mask credentials, and only traces: every span attribute passes
`trellis.harness.redaction` once, where it is set. Names that look like secrets (`api_key`,
`password`, `token`...) and values that look like credentials (bearer tokens, JWTs, `sk-...`,
long key-like blobs) become `[redacted]`, and credentials inside longer text (a command
line, an error message, a log line) are masked where they stand, the text around them kept:

| In the text | Masked | Kept |
|---|---|---|
| a token by the shape its provider documents: GitHub (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`), an AWS access key id (`AKIA`/`ASIA` and 16), Slack (`xoxb-`..., `xapp-`), a Google API key (`AIza` and 35), Stripe (`sk_live_`, `rk_test_`...) | the token | a publishable `pk_` key, a prefix on its own |
| a private key block (`-----BEGIN ... PRIVATE KEY-----`), also one cut off before its end | the whole block | a certificate, a public key |
| a URL with a password (`postgresql://ada:s3cr3t@db/app`) | the password | the user, the host |
| `Authorization: Basic ...` (and `Digest`, `Negotiate`, `Token`) | the credentials | the scheme |
| a value after a name that is a secret's (`password=hunter2`, `?api_key=...&`, `CLIENT_SECRET='...'`, `"passwd": "..."`, `aws_secret_access_key = ...`) | the value | the name; a number (`token_count=512`); `None`/`null`/`false`; an expression in code (`api_key=os.environ[...]`); after a colon, a plain word (`Password: must not be empty`) |

Long values are cut at 2000 characters. An error raised out of a span is recorded by the harness with its message
redacted (the OTel SDK's own `record_exception` keeps it as it is). Names are
matched by their words, not substrings (`gen_ai.usage.input_tokens` and `tokenizer` stay;
`refresh-token` and `X-Api-Key` go), and a number is never treated as a credential. Mappings
and lists are redacted recursively, and so is text that is a JSON object or array (a model's
tool-call arguments, a tool result as the model read it, in a `chat` span's input and
output): it stays JSON text, redacted. Bytes become `<n bytes>`, anything else its JSON or its
text. The redactor is the contracts `TelemetryRedactor`; it never raises into a run. It needs
no setting.

Personal data is not masked by the harness: what is personal is your domain's to say (an
e-mail address, a card number, an account id). Everything else carries values as they are —
the run's event stream (`agent.stream`, `serve_chat`'s AG-UI events, `serve_a2a`'s task updates
and push notifications), its answer, the memory service's transcript and tool records (and the
spooled copy of a write that waits for it), the journal, the interrupt kept in agent-runs — so
memory search and recall see the real data. Masking of your own is a hook:

* what your users see — the answer and every event of `agent.stream`, `agent.events` and
  AG-UI — `on_output` ([hooks.md](hooks.md#mask-what-the-ui-sees)); A2A, memory and the
  journal are not touched by it;
* what the model reads, and so what the journal and memory keep — `after_tool` returns the
  outcome masked, the framework's own model-call middleware the messages ([hooks.md](hooks.md),
  [`examples/02_way1_react/hooks_guardrails.py`](../examples/02_way1_react/hooks_guardrails.py)).

## Events

`agent.stream` yields contracts `RunEvent`s: `RUN_STARTED`, `CONTEXT_LOADED`,
`TEXT_MESSAGE_*`, `TOOL_CALL_*`, `CUSTOM` (`tool_notice`, `log` from `current().log(...)`,
`warning`, `decision` — the person's decision an attempt goes on with: `decision`, `reviewer`,
`comment`, `remember`; or a call approved because what a person allowed covers it — for this
conversation, always, or by the schedule that fired the run: `remembered: true` —; `custom`,
what a LangGraph graph's node wrote to its `custom` stream mode, in `value`),
`INTERRUPT`, `RUN_ERROR`, `RUN_FINISHED`. Every `RUN_FINISHED` (whatever its outcome) says
what the run spent in `data.spent`: `tokens` (input and output), `cached_tokens` and
`cache_written_tokens` (the input tokens read from and written to the provider's prompt cache,
each priced apart from plain input), `cost` (US
dollars, as the gateway reported them), `model_calls` and `tool_calls`, its sub-agents' runs
and earlier attempts included — the counts its budget reads ([runs.md](runs.md#spending-a-budget-per-run)). The decision is also on the attempt's
`invoke_agent` span (`trellis.decision.*` attributes and a `decision` span event). With
`RUNS_URL` the events are also in agent-runs' event log, readable from any process
(`agent.events`, [runs.md](runs.md#a-runs-events-from-anywhere)). A failed background write is a `warning`
event on the run's listeners, a `write_failed <label>` span with an error status in the run's
trace, a log line (an error the first time each kind fails) and a `trellis.writes.failed`
count — never silent, never raised into the run. A model call that stalled is a
`call_timeout` warning event and a `call_timeout` span in the run's trace
([reliability.md](reliability.md#stalled-model-calls)).

Every tool call that starts on the stream (`TOOL_CALL_START`, `TOOL_CALL_ARGS`) ends there,
before the attempt's `RUN_FINISHED`: `TOOL_CALL_END`, then `TOOL_CALL_RESULT` with the call's
`status` (`ok`, `error`, `timeout`, `rejected`, `cancelled`) and `output` — and, for a call its
model step repeats, which did not run, `same_as`: the id of the call it repeats
([reliability.md](reliability.md#calls-made-at-once)). A call the attempt
ends inside, on every adapter, ends the same way, with no output of its own — the result says
why:

| What cut it short | `status` | `output` |
|---|---|---|
| it asked a person (`current().ask`, an approval inside it): the run pauses | `paused` | `<tool> paused: the run waits on a person, and the call runs again when it resumes` |
| the run was cancelled (`agent.cancel`, A2A `tasks/cancel`, a person cancelling the question it asked) | `cancelled` | `<tool> was cut short: the run was cancelled` |
| the run's time limit (`timeout=`, `deadline=`) | `timeout` | `<tool> was cut short: the run ran out of time` |

A paused call runs again in the attempt that resumes the run: a new `TOOL_CALL_START` there.
A call's `tool_call_id` is its framework's id of it, else — a call a graph's node makes itself,
with no model call id — the run's id, a digest of the call (its tool and arguments) and which
time the run makes it: the same call has the same id in every attempt, and two calls never
share one, whichever attempt makes them (and so does its result's `result:<call>` message). A
call an attempt run again from its input replays from the journal (it ran before the pause:
not run again) is not announced again: its events are in the earlier attempt's part of the
run's log, where a reconnecting client, a thread's replay and `agent.events` read it once.
A call its framework still runs when the attempt ends (one run in a task of the framework's
own, as Claude's in-process MCP server does, stopped only later) is ended then, before
`RUN_FINISHED`, the same way; its later stop adds nothing to the stream.

Runnable: [examples/05_features/tracing_and_redaction.py](../examples/05_features/tracing_and_redaction.py)
— a run's spans exported to a provider of your own, a secret dropped and an e-mail masked.
