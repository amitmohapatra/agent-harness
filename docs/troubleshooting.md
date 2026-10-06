# Troubleshooting and FAQ

What an error or a symptom means, and what to do. Each feature's page also has an **On failure**
section with the details; they are indexed [at the end](#on-failure-feature-by-feature).

## Errors when the harness is built or an agent is wrapped

| You see | Why | Do |
|---|---|---|
| `ConfigurationError: RUNS_URL needs MEMORY_URL` | agent-runs accepts the keys the memory service issues, and the harness learns its tenant there | set `MEMORY_URL` too, or unset `RUNS_URL` (runs kept in process) |
| `ConfigurationError: MEMORY_URL (and RUNS_URL) need TRELLIS_API_KEY` | both services refuse a call without a key | set `TRELLIS_API_KEY` (locally: the memory service's development key) ([onboarding.md](onboarding.md)) |
| `ConfigurationError: the memory service at MEMORY_URL refused TRELLIS_API_KEY` | the key is wrong, revoked, or of another deployment | issue a key for this tenant ([onboarding.md](onboarding.md)) |
| `ConfigurationError: ... could not be reached to say who TRELLIS_API_KEY is` | the first ask of `/v1/keys/self` failed (later outages keep the last answer) | check `MEMORY_URL` and that the service is up |
| `ValidationError` from `Settings` | a sample outside 0–1, a concurrency under 1, an unknown `SANDBOX` | fix the variable ([configuration.md](configuration.md#settings-and-the-environment)) |
| `ConfigurationError: cannot wrap X` | the target is not a compiled graph, an OpenAI Agents `Agent`, `ClaudeAgentOptions` or an async function | compile the graph (`graph.compile()`), or wrap an `async def (input, agent)` |
| `ConfigurationError: a langgraph target binds its tools when it is built` | `tools=`, `mcp=` or `skills=` given to `h.wrap` for a compiled graph | build the graph with `await h.tools(..., framework="langgraph")` ([langgraph.md](frameworks/langgraph.md)) |
| `ConfigurationError: without= names no feature ...` | a name not in the `without=` table | use one of the names listed in the message ([configuration.md](configuration.md#what-is-on-and-how-to-turn-it-off)) |
| `ConfigurationError` naming a `framework_options` key | the framework's run call cannot take it, or the harness keeps it for itself | see what each framework takes ([configuration.md](configuration.md#the-frameworks-own-run-options)) |
| `ConfigurationError: ReAct with a model name needs BIFROST_URL` | `ReAct(model="name")` is a gateway model | set `BIFROST_URL`, or pass a LangChain chat model object |
| `ConfigurationError: ReAct needs the react extra` | `langchain-openai` or Deep Agents is not installed | `pip install 'trellis-harness[react]'` |
| a log line `... is installed, but this trellis-harness was tested with ...` | a framework outside the tested range | use a newer `trellis-harness`, or the framework version in the range ([versioning.md](versioning.md)) |

## Runs

| You see | Why | Do |
|---|---|---|
| `Result.status == PAUSED` and nothing happens | the run waits for a person | answer it: `agent.resume(result.interrupt.interrupt_id, ...)`; list what waits with `h.inbox(...)` ([interrupts.md](interrupts.md#an-inbox-of-your-own)) |
| `ConfigurationError: not an answer to ...` on `resume` | the answer does not fit the question (`expects`, `options`) or the edit does not fit the tool's schema; the run keeps waiting | send an answer that fits; the message says why ([interrupts.md](interrupts.md#answering)) |
| `ConfigurationError: run ... is a sub-agent's run` | the question is answered through the parent | resume the parent run's interrupt ([subagents.md](subagents.md)) |
| `ConfigurationError: only an approval is remembered for the run` | `remember="run"` on an answer, an edit or a reject | use it on an `approve` only ([interrupts.md](interrupts.md#comments-and-approving-for-the-rest-of-the-run)) |
| `Result.status == QUEUED` after a resume | the run came from the queue (`start`, a schedule): a worker continues it | run a worker (`h.worker([...]).run()` or `python -m trellis.harness.worker`) ([runs.md](runs.md#workers)) |
| a queued run never starts | no worker serves that agent id, or a run of the same `concurrency_key` is running | start a worker for it; see [runs.md](runs.md#queue-order-and-busy-conversations) |
| `TIMEOUT` with `run_timeout` / `run_deadline` | the run worked past its `timeout` or passed its `deadline` | raise the limit, or split the work ([reliability.md](reliability.md#run-time-limit-and-deadline)) |
| a schedule fires with no worker running | in process (no `RUNS_URL`) a schedule fires only when a worker of this process asks for work | run a worker, or `h.runs.schedules.fire(id)` ([runs.md](runs.md#schedules)) |
| `ConfigurationError: a queued run's input must be JSON` | `start` and `schedule` keep the input in agent-runs | send JSON, or use `run`/`stream` |
| a graph's own `interrupt()` fails the run on resume elsewhere | its checkpointer (`InMemorySaver`) holds the pause only in the process that paused | give the graph a shared, durable checkpointer ([interrupts.md](interrupts.md#how-a-run-continues)) |
| a tool ran twice after a crash | only calls through the harness are journaled | make the side effect a harness tool (`tools=` or `h.tools`); code outside a tool and the model calls run again on a re-run |
| `LeaseLostError` in a worker's log | another worker took the run (the lease lapsed) | nothing: the run is continued there; give heavy runs more heartbeat room ([runs.md](runs.md#workers)) |

## Tools, governance and the gateway

| You see | Why | Do |
|---|---|---|
| every tool that writes asks for approval | the tool catalog cannot be read (memory down): governance fails closed | bring the memory service back; rules read in the last 300 s still stand ([governance.md](governance.md#the-catalog-kept-fresh)) |
| a tool runs although it should ask | its risk is `read`/`write` and no `approve_when` holds | declare it `irreversible`, or set the catalog's `approve_when` ([governance.md](governance.md)) |
| the approval is asked twice | a framework gate (`HumanInTheLoopMiddleware`, `interrupt_on`, `needs_approval`) and the harness's both gate the tool | gate each tool in one place ([interrupts.md](interrupts.md#framework-approvals-langchains-middleware-and-openai-agents-needs_approval)) |
| no MCP tools | no `BIFROST_URL`, the key allows none, `without={"mcp"}`, or `mcp=[]` | check the key's MCP allow-list ([gateway.md](gateway.md)) |
| an MCP tool is missing | the gateway would run it itself (Agent Mode: `tools_to_auto_execute`) | take it off the client's auto-execute list ([gateway.md](gateway.md#what-the-gateway-never-does-for-a-run)) |
| the model is offered only some tools | tool hints narrow from 5 tools on | nothing, or `without={"hints"}` ([tools.md](tools.md#tool-hints-what-the-model-is-offered)) |
| `execute_tool_code` instead of the server's tools | Code Mode: 3+ read-only Code Mode servers or 20+ tools | nothing, or `without={"code_mode"}` ([tools.md](tools.md#code-mode)) |
| `"<tool> was not run: ... Call it again with arguments that fit its schema"` | the model's arguments did not fit the tool's schema | nothing: the model reads it and calls again |
| "of unknown effect" in a tool result | a write timed out, or was running when its worker died: it is never run again blind | check the target system with the call's idempotency key ([reliability.md](reliability.md#unknown-outcomes)) |

## Memory and evaluation

| You see | Why | Do |
|---|---|---|
| `agent.context` is `None` | memory is off (`MEMORY_URL` unset, `without={"memory"}`/`{"memory_push"}`) or the context read failed (a `warning` event) | check the environment and the run's events ([memory.md](memory.md)) |
| `ConfigurationError: memory is off for this run` | `trellis.current().memory` with memory off | set `MEMORY_URL`, or check `agent.uses("memory_pull")` first |
| `warning` events about writes, `trellis.writes.undelivered` | memory writes given up after retries | set `TRELLIS_SPOOL_DIR` on a volume that outlives the process ([memory.md](memory.md#background-writes-what-is-guaranteed)) |
| a person's feedback does not change memory | human feedback waits for the tenant administrator's review | approve it in the memory service; the trace score is there already |
| no judge scores | no judge model (`TRELLIS_JUDGE_MODEL`, a judge's `model=`, or a `ReAct` built with a model name) | set `TRELLIS_JUDGE_MODEL` ([evaluation.md](evaluation.md#the-judges-model)) |
| few runs judged | online judges score a sample (`TRELLIS_JUDGE_SAMPLE`, 0.1 by default) | raise the sample |
| no traces | no `OTEL_EXPORTER_OTLP_ENDPOINT` and no provider of your own | set it ([observability.md](observability.md#export)) |
| a value shows as `[redacted]` in a span | its name or value looks like a secret | expected; redact more with a hook of your own ([observability.md](observability.md#redaction)) |

## FAQ

**Do I have to change my agent's code?** No. Wrap the object you built; call `agent.run` where
you called the framework. A compiled graph takes its harness tools from `h.tools(...)` when it
is built — that is the one line inside it.

**Can I use only memory, or only the inbox?** Yes: import that block (Way 2,
[docs/README.md](README.md#way-2-pluggable-blocks-your-framework-our-pieces)), or wrap and turn
the rest off with `without=`.

**Does the harness re-implement my framework's loop?** No. The adapters call the framework's
public API; `ReAct` is LangChain's own `create_agent` with middleware.

**What happens if the memory service is down?** The run goes on without its context (a
`warning` event), writes are retried and spooled, and tools that write ask for approval until
the catalog can be read again ([architecture.md](architecture.md#the-pipeline)).

**Where do approvals show up for people?** In `h.inbox(...)` (and agent-runs' inbox), and as
agent-runs' `run.paused` webhook to a receiver of yours ([interrupts.md](interrupts.md#telling-people)).

**How do I test an agent with approvals?** `trellis.testing.Reviewer` answers them from a
script ([interrupts.md](interrupts.md#testing-trellistestingreviewer)); the examples run on a
scripted model ([examples/](../examples/README.md)).

**Which framework versions are supported?** Those in [versioning.md](versioning.md), pinned by
the extras.

## On failure, feature by feature

[Tool timeouts](reliability.md#tool-timeouts) · [Model timeouts](reliability.md#model-timeouts) ·
[Run time limit](reliability.md#run-time-limit-and-deadline) · [Retries](reliability.md#retries) ·
[Unknown outcomes](reliability.md#unknown-outcomes) · [Cancel](reliability.md#cancel) ·
[Asking](interrupts.md#asking) · [Hooks](hooks.md) · [Telling people](interrupts.md#telling-people) ·
[An inbox of your own](interrupts.md#an-inbox-of-your-own) · [Queue order](runs.md#queue-order-and-busy-conversations) ·
[Run events](runs.md#a-runs-events-from-anywhere) · [Gateway](gateway.md) ·
[Prompts](prompts.md#on-failure) · [Skills](skills.md#on-failure) ·
[Sub-agents](subagents.md#on-failure) · [Sandbox](sandbox.md#on-failure) ·
[Claude Agent SDK](frameworks/claude-agent-sdk.md#approvals-and-pauses)
