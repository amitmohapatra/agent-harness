# Documentation map

Every page in this repository, and the question it answers. If a page and the code disagree, the
code is right and the page is a bug — they are checked against each other on each documentation
pass, and the inconsistencies found in the last one are listed at the bottom of this file.

## Start here

| Page | The question it answers |
| --- | --- |
| [../README.md](../README.md) | What is a harness, why would I want one, and what does the smallest working agent look like? |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | How is it layered, what are the ports, and what does one execution actually do? |
| [harness.md](harness.md) | What is `AgentHarness`, what do I pass it, and what does my agent get back? |
| [configuration.md](configuration.md) | Which setting do I change for the situation I am in? Every setting and environment variable. |
| [../COMPATIBILITY.md](../COMPATIBILITY.md) | Which versions were actually tested, and which capability of which framework is unsupported and why? |
| [troubleshooting.md](troubleshooting.md) | This error message — what causes it? |

## The core APIs

| Page | The question it answers | Diagram |
| --- | --- | --- |
| [harness.md](harness.md) | the object, the ports, the runtime, the lifecycle events | sequence + class |
| [memory.md](memory.md) | how a turn gets context and what it writes back; visibility and policy | sequence |
| [tools.md](tools.md) | local, MCP and agent tools behind one policy surface | sequence + class |
| [models.md](models.md) | calling a model through the gateway — and which models the platform uses, with origins and licences | sequence |
| [reasoning.md](reasoning.md) | the bounded ReAct loop, the prompt budget, and compaction | sequence |
| [events.md](events.md) | the `RunEvent` stream every run emits, and the sinks in the box | sequence |
| [interrupts.md](interrupts.md) | how a run pauses for a person, and how the answer continues it | state |
| [runs.md](runs.md) | durable run records, the paused inbox, `agent-runs` and Temporal | sequence + state + class |
| [registry.md](registry.md) | discovery: what exists, who may see it, and the MCP clients that follow from it | sequence + class |
| [artifacts.md](artifacts.md) | where the bytes go, and how a claim carries its evidence | sequence |
| [interceptors.md](interceptors.md) | extending the pipeline; listeners; policy providers; redaction | state |
| [a2a.md](a2a.md) | serving an agent to other agents, and calling theirs | sequence |
| [evaluation.md](evaluation.md) | is this agent any good, and is it getting better? | flowchart |
| [observability.md](observability.md) | where do I look when one request went wrong? Langfuse per agent, Datadog per service | flowchart |

## Operating it

| Page | The question it answers |
| --- | --- |
| [privacy.md](privacy.md) | What leaves the process, what never does, and how to change either? |
| [performance.md](performance.md) | What does the harness itself cost, and how do I measure it here? |
| [limitations.md](limitations.md) | What can this not do? (Stated plainly, not implied.) |

## The integrations, one page each

Every adapter and surface is its own distribution; the core imports none of them
(`tests/compatibility/test_matrix.py` proves it in a subprocess).

| Distribution | Extra | Page |
| --- | --- | --- |
| `trellis-harness-langgraph` | `[langgraph]` | [../integrations/langgraph/README.md](../integrations/langgraph/README.md) |
| `trellis-harness-deepagents` | `[deepagents]` | [../integrations/deepagents/README.md](../integrations/deepagents/README.md) |
| `trellis-harness-openai-agents` | `[openai-agents]` | [../integrations/openai_agents/README.md](../integrations/openai_agents/README.md) |
| `trellis-harness-claude-agent-sdk` | `[claude-agent-sdk]` | [../integrations/claude_agent_sdk/README.md](../integrations/claude_agent_sdk/README.md) |
| `trellis-harness-agui` | `[agui]` | [../integrations/agui/README.md](../integrations/agui/README.md) |
| `trellis-harness-a2a` | `[a2a]` | [../integrations/a2a/README.md](../integrations/a2a/README.md) + [a2a.md](a2a.md) |
| `trellis-harness-temporal` | `[temporal]` | [../integrations/temporal/README.md](../integrations/temporal/README.md) |

## Runnable examples

[../examples/README.md](../examples/README.md) says what each one demonstrates, what it needs,
and which of them talk to a live Memory Service.

## The plan of record

[PLATFORM-DESIGN-2026-09-28.md](PLATFORM-DESIGN-2026-09-28.md) is the design this work follows:
§2 the shape, §5 the loop, §6 tools, §7 humans in the loop, §8 adapters, §9 A2A, §10 surfaces,
§11 evaluation, §13 developer experience, §15 the execution order, §16 the decisions taken.

It is a **plan**, and it was written before the code. Where its sketch and a shipped API differ,
the API pages above win. The known divergence is §13's constructor sketch:
`AgentHarness(surfaces=[...], evaluation=[...])`, `harness.agui.router` and `harness.a2a.serve()`
were the proposal; what shipped is `event_sinks=`/`judge=` on the constructor,
`agui_router(harness, …)` from the AG-UI distribution, and `A2AServer.from_registry(harness, …)`
from the A2A one — because a surface that is a separate distribution cannot be an attribute the
core resolves. The sibling repositories are the Memory Service (`agent-memory-service`), the
contracts (`agent-contracts`), and the two durability services (`agent-runs`,
`agent-schedules`), each with its own README.

## Where the numbers in these pages come from

No page here quotes a measurement it did not read out of an artifact in this repository:

| Number | Artifact |
| --- | --- |
| harness overhead (p50/p95) | `benchmark-results.json`, written by `make bench` |
| the tested version matrix | `compatibility-matrix.json`, written by `pytest tests/compatibility` |
| judge smoke costs | `judge-smoke-results.json` |
| retrieval and model-choice numbers | the Memory Service's `benchmark/results/` and its `docs/MEASUREMENTS.md` — never restated here |

Nothing in this documentation claims production readiness. Where a capability is partial, the
page says which part and why, and [limitations.md](limitations.md) collects them.

## What the last documentation pass found

Checked by reading every backticked identifier in these pages against the source tree, and by
running the examples against a live Memory Service. Fixed here:

| Was | Now |
| --- | --- |
| `ARCHITECTURE.md` pointed at `src/trellis/harness/contracts/ports.py` | the ports are in `trellis-contracts`; the module list also predated `events/`, `interrupts/`, `runs/` and `reasoning/` |
| `COMPATIBILITY.md` cited `tests/e2e/test_real_memory_sdk.py` | the file is `tests/e2e/test_sdk_wire_contract.py` |
| `COMPATIBILITY.md` omitted three rows that are in `compatibility-matrix.json` | temporalio, `trellis-harness-temporal` and fastapi are listed |
| `COMPATIBILITY.md` named `BAAI/bge-small-en-v1.5` as the live run's encoder | labelled as a historical run; that origin is now excluded by the provenance rule and the frozen model is Granite ([models.md](models.md)) |
| `COMPATIBILITY.md` described the SDK contract as `files.*` | `documents.*`; `files` is the deprecated spelling (ADR 0022) |
| `configuration.md` said "75 settings" | 79, and the page shows how to count them |
| `configuration.md`'s environment-variable table was split in two by a stray blank line | one table |
| `limitations.md` listed the gateway, the MCP tool client and the A2A transport as not implemented | they landed in phases 3–5; what remains is named exactly |
| `observability.md`'s `/v1/reads` example sent `Authorization: Bearer` and `X-Tenant-Id` | `X-API-Key` and `X-Trellis-Tenant` (ADR 0022) |
| the README's lifecycle list omitted `on_agent_pause`, and its install list omitted three extras | both complete |
| `examples/memory_tour.py` failed against a live service (`Workspace not found`) | it onboards the way `tests/support.py::onboard` does, and handles a workspace id already burned as an anchor |

Two are **code**, not documentation, and are recorded rather than papered over:

* `trellis.harness.memory.visibility` accepts `WORK`, `GROUP` and `GLOBAL`, which the Memory
  Service's `Visibility` enum does not define — so the pre-flight check passes a value the service
  then refuses (verified: `hints={"visibility": "WORK"}` → `ValidationError`). See
  [memory.md](memory.md).
* `trellis.contracts.a2a.A2A_PROTOCOL_VERSION` is `"0.3.0"` while a served card carries the
  installed `a2a-sdk`'s `PROTOCOL_VERSION_CURRENT` (`"1.0"`). The constant is stale; the wire is
  right. See [a2a.md](a2a.md).
