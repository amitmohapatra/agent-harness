# Privacy and redaction

The default posture is conservative: telemetry describes *what happened*, not *what was
said*. Turning payload capture on is a deliberate, per-tenant decision.

## What is never exported unless you switch it on

Raw prompts, raw model inputs and outputs, raw tool inputs and outputs, raw memory content,
agent inputs and outputs, and the end-user id. With the defaults, an agent span carries
identifiers, counts, statuses, latencies and token/cost numbers — and nothing else.

```yaml
harness:
  telemetry:
    capture:
      inputs: false          # prompts, model inputs, tool arguments, agent inputs
      outputs: false         # completions, tool results, agent results
      memory_content: false  # retrieved memory text
      user_id: false
      thread_id: true        # an opaque id, not content
```

This is the **only** capture policy. Langfuse and any other backend obey it; there is no
second block to keep in sync, which is how a backend quietly ends up with looser rules.

Two independent gates apply, in this order:

1. **capture** decides whether a payload may be attached at all;
2. **redaction** decides what an allowed payload may contain.

## Redaction rules

`DefaultRedactor`:

* drops values whose attribute **name** contains a sensitive word — matched on name
  *segments* (`api_key`, `x-api-key`, `authorization`, `access_token`, `cookie`, `secret`,
  `password`, `ssn`, `cvv`, …). Segment matching is deliberate: substring matching would
  redact `gen_ai.usage.input_tokens` because it contains "token", quietly destroying your
  own metrics. Numeric values are exempt from name-based redaction — a credential is never
  an `int`;
* drops values that *look* like credentials whatever they are called: `sk-…` keys, JWTs,
  `Bearer …`, high-entropy base64 blobs;
* masks e-mail addresses;
* truncates long values (default 2000 characters) with an explicit `...[truncated N chars]`;
* recurses into mappings and sequences, keeping the shape and dropping the secrets.

With `drop_payloads=True`, an allowed payload is replaced by a **reference**: type, byte
size and a truncated SHA-256 — enough to correlate, impossible to read.

Custom redactor:

```python
class MyRedactor:
    def redact_attributes(self, attributes): ...
    def redact_input(self, value): ...
    def redact_output(self, value): ...


AgentHarness(memory=memory, redactor=MyRedactor())
```

Or extend the default: `DefaultRedactor(extra_sensitive_keys=("patient", "iban"))`.

## Langfuse specifics

Langfuse receives the same spans the OTLP backend does, so the same capture and redaction
gates apply before anything is exported. Additionally:

* `session.id` is the thread id, and only when `capture.thread_id` is on;
* `user.id` is exported only when `capture.user_id` is on;
* trace metadata carries ids (agent run, parent run, turn, task, tenant) — never content;
* tags carry the agent id and skill ids.

Nothing sends customer documents, PII, credentials, API keys, secret headers, private
memory text or raw prompts to Langfuse by default.

## Sampling

```yaml
harness:
  telemetry:
    sampling:
      sample_rate: 0.1                 # 10% of ordinary executions
      error_sample_rate: 1.0           # every failure
      critical_agent_sample_rate: 1.0
      critical_agents: [billing-agent]
```

The decision is taken once per execution and is deterministic in the run id, so a run is
either fully traced or not traced at all, and a replay lands on the same side. Metrics and
memory writes are unaffected by sampling.

## Trace context and baggage

Cross-service propagation uses W3C Trace Context through the configured OpenTelemetry
propagator. Baggage carries ids only (`tenant_id`, `agent_id`, `agent_run_id`,
`correlation_id`) — never content, never credentials. Personal data is never placed in URL
parameters or query strings by the harness.

## Compliance mode

For environments where losing telemetry is worse than failing a request:

```yaml
harness:
  observability:
    failure_mode: fail_closed     # applies to every telemetry backend
```

This is not the default, and it means an observability outage becomes a business outage.
