# Limitations

Stated plainly, because a harness that overstates what it instruments is worse than one
that instruments less.

## Instrumentation is not magic

The harness **cannot** intercept an arbitrary, unwrapped Python function call. If your
agent imports an SDK and calls it directly, that call is invisible to the harness. What is
instrumented, in priority order:

1. calls through `runtime.tools` / `runtime.model`;
2. callables wrapped with `harness.wrap_tool()` / `harness.wrap_model()`;
3. framework callbacks and events the adapter subscribes to;
4. OpenTelemetry auto-instrumentation, if you install it for that library;
5. everything else — **not instrumented**, and the execution-level span is all you get.

The harness does not monkeypatch libraries globally, and it will not start doing so
quietly.

## Wrapping is per-execution, not per-line

`harness.execution(...)` instruments the *block*: its span, its memory context, its
deadline, its observations. It cannot see inside the code you call within the block.

## One trace per node, unless you enclose the run

LangGraph runs each superstep in its own task, so node spans have no ambient parent. Unless
you wrap the graph invocation (`async with harness.execution(context, ...)`), each node
produces a separate trace — every node is still fully instrumented, but the turn is not a
single tree. The same applies to any framework that starts tasks the harness did not create.

## What the harness does not own

Graph topology, routing, reducers, checkpoint backends, framework state schemas, prompt
content, business retries of non-idempotent work, or the Memory Service's own semantics.
If a framework changes its state model, nothing here has to change — which is the point.

## Memory

* Retrieval happens only when the request has a query: an objective, a string input, or a
  recognised key (`query`, `question`, `objective`, `prompt`, `input`, `text`). For a
  LangGraph node, pass `query="<state key>"` or a callable. Without one, retrieval is
  skipped rather than guessed at.
* The output written back is the result's text; structured `data` is *not* dumped into
  memory. Return explicit `memory_observations` for structured memory.
* Writeback is asynchronous by default: a process that exits immediately after an execution
  should `await harness.drain()` (or `await harness.aclose()`) or set
  `memory.writeback: false`.
* The writeback queue is bounded (256 in flight). Above that the harness writes inline and
  logs; it does not grow an unbounded backlog.

## Retries

Retries are off by default and only ever apply to an agent you explicitly marked
`idempotent=True`, for the configured categories (timeout, rate limit, dependency).
Authorization failures, validation failures, policy denials and cancellations are never
retried. The harness does not retry tool writes on your behalf: it propagates an
idempotency key so a tool that supports deduplication can be safe, and assumes nothing.

## Timeouts and cancellation

The deadline is enforced by cancelling the asyncio task. An agent that blocks the event
loop in synchronous CPU-bound code cannot be interrupted — that is a property of asyncio,
not of the harness. Long agents should call `runtime.check_cancelled()` between steps.

## Sampling

A sampled-out execution produces no spans at all (by design: half a trace is worse than
none). Metrics and memory writes still happen.

## Telemetry payloads

Raw prompts, model inputs/outputs, tool inputs/outputs, memory content and user ids are
**not** exported unless explicitly enabled. When you do enable them, redaction still
applies, and both are per-tenant/environment decisions with real consequences — see
[privacy.md](privacy.md).

## Not implemented (contracts exist, services do not)

Agent registry service, Bifrost model/tool gateway, MCP tool client, A2A transport, CrewAI
and Google ADK adapters. `AgentRequest`/`AgentResult` are serializable and the relevant
ports exist so these can arrive without rewriting agents — but nothing here talks to them
today.
