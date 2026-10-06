# Documentation

Trellis is used in one of two ways ([README](../README.md#two-ways-to-use-trellis)): **wrapped**,
where the harness runs your agent and every piece is automatic, or as **pluggable blocks**,
where your framework runs the agent and your code calls the pieces it wants. This page is the
map: every page, by way and by feature. Start with the [README](../README.md); the runnable
examples are [examples/](../examples/README.md).

## Reference

| Page | |
|---|---|
| [api.md](api.md) | every public name, its arguments and their defaults |
| [configuration.md](configuration.md) | every setting with its default and an example, what is automatic, `without=`, `framework_options=`, who the key says the deployment is |
| [architecture.md](architecture.md) | the five repositories, then the harness inside: modules, pipeline, bridge, journal, runtime, adapters, middleware |
| [flows.md](flows.md) | sequence diagrams: a run, memory, a pause and its resume, a worker crash, a schedule, MCP and Code Mode, sub-agents, AG-UI, A2A, evaluation, ReAct's middleware |
| [scenarios.md](scenarios.md) | which feature to use when |
| [onboarding.md](onboarding.md) | from running services to a configured application: the tenant, the keys, the environment, a key per person for an approvals UI |
| [troubleshooting.md](troubleshooting.md) | what an error or a symptom means, and what to do |
| [versioning.md](versioning.md) | the versions this release was tested with, the policy, how to upgrade |
| [adr/](adr/README.md) | the decisions behind the design |
| [CHANGELOG.md](../CHANGELOG.md) | what changed in each release |

## Way 1: wrapped (the harness runs your agent)

`h.wrap(agent)`, then `agent.run`. Each page says how a feature works and what the harness does
for you.

| Page | |
|---|---|
| **Frameworks** — one page each: install, the lines to add to an existing project, what is automatic, approvals, streaming, durable runs, surfaces, evaluation, limits | [LangGraph and LangChain](frameworks/langgraph.md) (`create_agent`, a hand-built `StateGraph`, checkpointers, `HumanInTheLoopMiddleware`) · [Deep Agents](frameworks/deepagents.md) · [OpenAI Agents SDK](frameworks/openai-agents.md) · [Claude Agent SDK](frameworks/claude-agent-sdk.md) · [ReAct](frameworks/react.md) · [plain functions](frameworks/functions.md) |
| [tools.md](tools.md) | the toolbox and where tools come from, their side effects, tool hints, Code Mode, `h.tools` |
| [gateway.md](gateway.md) | the Bifrost gateway: stored prompts, skills, Virtual MCPs (`mcp=`), who an MCP call is for, what the gateway never does for a run, frameworks' own MCP clients |
| [prompts.md](prompts.md) | prompts from code, `.md` files (`PROMPTS_DIR`), Langfuse and the gateway: one name, the lookup order, `ReAct(prompt=)`, `h.prompt`, pinned per run |
| [skills.md](skills.md) | Agent Skills from code, `SKILL.md` folders (`SKILLS_DIR`) and the gateway: progressive disclosure, pinned per run |
| [subagents.md](subagents.md) | `agent.as_tool()`: child runs, their pauses answered through the parent, crashes, cancel, time |
| [sandbox.md](sandbox.md) | `sandbox()`: commands and files in a sandbox of the run's own — its life, pauses, crashes, timeouts, governance |
| [governance.md](governance.md) | which calls run, are announced or ask: risks, the catalog's `approve_when`, failing closed |
| [hooks.md](hooks.md) | your code around runs, model calls and tool calls: guardrails, redaction of your own, audit |
| [memory.md](memory.md) | push, pull, what is recorded, background writes, documents, outcomes and grounding, the model key |
| [interrupts.md](interrupts.md) | `ask`, approvals (the harness's and the frameworks' own), `resume`, the journal, artifacts, an inbox of your own |
| [runs.md](runs.md) | run records, `start` and the worker, progress checkpoints, queue order, schedules, events from anywhere |
| [reliability.md](reliability.md) | time limits, retries, idempotency keys, crashes and unknown outcomes, cancel, the agent's version |
| [surfaces.md](surfaces.md) | `serve_chat` (AG-UI), `serve_a2a`, and `a2a(url)` tools |
| [observability.md](observability.md) | OTel GenAI spans, Langfuse, redaction, the collector |
| [evaluation.md](evaluation.md) | offline (`h.evaluate` over a dataset) and online (`judges=`) evaluation; the evaluators, trajectories, the judge's model and budget |

### Which target

| You have | Wrap | Page |
|---|---|---|
| a LangChain v1 agent (`create_agent`) or any compiled LangGraph graph | the graph, built with `await h.tools(..., framework="langgraph")` | [langgraph.md](frameworks/langgraph.md) |
| a Deep Agent (`create_deep_agent`) | the graph it returns, built the same way | [deepagents.md](frameworks/deepagents.md) |
| an OpenAI Agents SDK `Agent` (handoffs included) | the `Agent`, with `tools=[...]` | [openai-agents.md](frameworks/openai-agents.md) |
| a Claude Agent SDK setup | the `ClaudeAgentOptions`, with `tools=[...]` | [claude-agent-sdk.md](frameworks/claude-agent-sdk.md) |
| a model and tools, no framework | `ReAct(system=..., model=...)` | [react.md](frameworks/react.md) |
| code that decides itself (a workflow, a router, glue) | `async def fn(input, agent)` | [functions.md](frameworks/functions.md) |

Every target gets the same harness: memory push and pull, governance and approvals, records,
grounding, judges, traces, durable runs and both surfaces. What differs is how a pause resumes,
how far the tool schemas are narrowed, and what the framework does on its own — each page says,
and [architecture.md](architecture.md#the-adapters) compares them.

## Way 2: pluggable blocks (your framework, our pieces)

Your framework runs the agent, untouched; your code imports a block and calls it. Each page:
what the block is, install, setup from the environment, the API, its behaviour (errors,
retries, tenancy), and how it relates to Way 1.

| Page | Block |
|---|---|
| [blocks/memory.md](blocks/memory.md) | `trellis.memory`: the context into your prompt, the turn and each tool call recorded, feedback |
| [blocks/runs.md](blocks/runs.md) | `trellis.runs`: durable runs, a pause with your framework's checkpoint, the inbox, resume, `Worker`, schedules, webhooks |
| [blocks/governance.md](blocks/governance.md) | `trellis.harness.governance`: `Governance.check` and `governed` on your own tools, `publish`, `decided` |
| [blocks/evaluation.md](blocks/evaluation.md) | `trellis.harness.evals`: `evaluate` on any async function, `judge` on one run, `EvalServices.from_env` |
| [blocks/a2a.md](blocks/a2a.md) | `trellis.harness.a2a.remote`: call any A2A agent (serving is Way 1) |
| [sandbox.md](sandbox.md#way-2-without-a-harness) | `trellis.harness.sandbox`: a provider (`DockerSandbox`) and its sandboxes, governed with `governed` |
| [blocks/mixing.md](blocks/mixing.md) | both ways in one deployment, and the records every block shares (`trellis.contracts`) |

**Recipes**, end to end with the framework's own pause and state: an unmodified agent with
memory context and recording, governed tools, a durable pause in agent-runs answered from the
inbox, and a judge.

| Page | |
|---|---|
| [blocks/langgraph.md](blocks/langgraph.md) | a LangGraph graph: `governed` tools asking through `interrupt`, the checkpointer, `Command(resume=)`; a `Worker` continuing the graph across processes |
| [blocks/openai-agents.md](blocks/openai-agents.md) | an OpenAI Agents `Agent`: `needs_approval` from governance, the `RunState` as the run's checkpoint |
| [blocks/claude-agent-sdk.md](blocks/claude-agent-sdk.md) | a Claude Agent SDK `query()`: `can_use_tool` from governance, the session as the checkpoint |

With no harness at all — only the SDKs (`trellis.memory`, `trellis.runs`, `trellis.contracts`)
in your own LangGraph or Deep Agents code: [examples/04_no_harness](../examples/README.md#04_no_harness-only-the-sdks).

## Composition: a Harness is the blocks you give it

The two ways are one set of blocks — the run store, the memory client, the Bifrost gateway,
governance, the prompt and skill sources — composed differently.

| You want | Write | What you get |
|---|---|---|
| everything, from the deployment (Way 1) | `Harness()` | each block built from its environment variable, each off when its variable is unset |
| some blocks of your own, the rest from the deployment | `Harness(runs=RunsClient(...), governance=Governance(...))` | the blocks you pass, used as they are (and yours to close); the others built from the environment |
| a block off although the deployment names it | `Harness(memory=False)` (`gateway=False`, `runs=False`: runs kept in this process) | that block off for every agent of this harness |
| your own scheduler or worker | `await agent.execute(job)` for each run it claims | the run's next attempt with its journal, governance, memory and limits ([runs.md](runs.md#workers)) |
| `ReAct` (or any target) with your blocks | `Harness(<your blocks>).wrap(ReAct(...))` | one loop and one path: the same `ReAct` as Way 1, on your blocks |
| the blocks without the harness (Way 2) | import the block and call it | your framework runs the agent; your code calls each block where it chooses |

```python
from trellis import Harness, ReAct
from trellis.harness.governance import Governance
from trellis.runs import RunsClient

runs = RunsClient()  # RUNS_URL, TRELLIS_API_KEY
h = Harness(runs=runs, memory=False, governance=Governance())
agent = h.wrap(ReAct(system="You handle refunds.", model="provider/model"), id="refunds")
```

Every argument: [api.md](api.md#harness). Runnable:
[examples/05_features/own_scheduler.py](../examples/05_features/own_scheduler.py) (its own run
store, its own scheduler loop, governance, no memory).
