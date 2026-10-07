# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## Unreleased

### Changed

- The memory context is always asked for with the run's own tools, so an agent with any number
  of tools gets the skills it learned; the tool hints still come from five tools (now the
  memory service's threshold), and `without={"hints"}` keeps the learned skills while offering
  every tool.
- The framework extras pin the minor range they were tested with: `langgraph>=1.2,<1.3`,
  `langchain>=1.4,<1.5`, `langchain-core>=1.6,<1.7`, `langchain-openai>=1.6,<1.7`,
  `deepagents>=0.7.19,<0.8`, `openai-agents>=0.22.3,<0.23`, `claude-agent-sdk>=0.2.160,<0.3`,
  `a2a-sdk>=1.1,<1.2` ([ADR 0001](docs/adr/0001-version-policy.md)).
- `trellis-contracts>=0.6.1,<0.7` (schedules carry their run options from 0.6.1).
- The documentation: a "Start here" README, `docs/api.md`, `docs/architecture.md` (the five
  repositories, then the harness inside; was `ARCHITECTURE.md`), `docs/flows.md` (sequence
  diagrams), `docs/configuration.md` (every setting with its default and an example),
  `docs/troubleshooting.md`, `docs/versioning.md`, `docs/adr/`. `docs/blocks/contracts.md` is
  merged into `docs/blocks/mixing.md`.
- The examples are numbered, simple to complex (`examples/01_start` ... `06_scenarios`), every
  one runs offline on a scripted model, memory service and gateway, and `make examples` runs
  them in parallel. Run one with `python -m examples.<group>.<name>`.

### Added

- From the memory service, with no harness code: `memory_search` with `kinds: ["message"]`
  finds what was said in this conversation and in the same user's earlier ones
  ([memory.md](docs/memory.md#pull)), and every run is offered the skills its agent learned
  from its successful runs, across all of the agent's users, in its memory context
  ([skills.md](docs/skills.md#learned-skills)). The feature matrix's memory-pull cells send
  message searches through every adapter and way; live tests cover both end to end against the
  running service.

- A warning, once per framework, when the installed version is outside the tested range
  (`trellis.harness.compat`).
- `LocalRuns.schedules.fire(schedule_id, at=None)`: fire a schedule now, as agent-runs'
  `schedules.fire` does.
- `make docs-check` (links, anchors, snippets against the real API) and `make examples-live`.
- A weekly canary workflow: the latest framework releases, unpinned, through `make test` and
  `make matrix`; it never blocks a pull request.

## 0.4.0 — 2026-09-30 and since

One distribution with extras (`langgraph`, `deepagents`, `react`, `openai-agents`,
`claude-agent-sdk`, `agui`, `a2a`, `otel`); the earlier harness, its integrations and Temporal
are gone. Since the 0.4.0 version was set:

### Added

- `ReAct(...)` is LangChain's `create_agent` with Deep Agents' context middleware and the
  harness's (`trellis.harness.middleware`: `HarnessTools`, `ModelHooks`, `StepLimit`,
  `StallGuard`, `ReadTools`, `read_result`, `RunCheckpointer`), usable in any `create_agent` or
  Deep Agents graph.
- `framework_options=`: the framework's own run options, on `h.wrap`, `run`, `stream`, `start`
  and `schedule`; schedules take every per-run option `start` takes.
- Trajectory evaluators: `called(...)`, `tool_sequence(...)`, `EvalCase.trajectory`.
- Prompt and skill sources: code, `PROMPTS_DIR` / `SKILLS_DIR`, Langfuse and the gateway, one
  name, pinned per run.
- People in the loop: labelled options with several picks, forms (`form=`, `ui_schema=`), your
  own screens (`component=`, `props=`), comments, `remember="run"`, a graph's `interrupt()`
  with fields; `trellis.testing.Reviewer`.
- Run events in agent-runs' event log (any replica streams any run), queue order (`priority`,
  `concurrency_key`), admission (`429`), JSON logs and metrics.
- Sub-agents (`agent.as_tool()`), sandboxes (`sandbox()`), hooks (`Hooks`, `Deny`, `Ask`,
  `Rewrite`), the run time limit and deadline, cancel, the agent's version.

### Removed

- `tool(approval=fn)`: an approval rule in code is a `before_tool` hook.
- `tool(external=True)` and `resume(result=)`: a result from outside is an `ask` in the tool.
- Notifiers (Slack, SMTP): agent-runs' `run.paused` webhook tells people.
- `h.serve_inbox` and its page: an inbox is `h.inbox` plus `agent.resume`.
- The harness's own ReAct loop: `ReAct` is a `create_agent` factory.

### Fixed

- The bugs the feature matrix found (BUG-2 to BUG-10): cut-short tool calls end on the event
  stream, A2A `tasks/cancel` stops the run, JSON text is redacted as what it holds,
  `without={"mcp"}` does not ask the gateway, the LangChain middleware offers only the run's
  tools, a worker's stop out of working time is told in time, and others.
- Tool calls whose arguments the model could not produce as JSON are an error the model reads
  (OpenAI Agents, LangChain).

## 0.3.0 — 2026-09-28

The harness core: run events, one interrupt mechanism, MCP and memory tools, AG-UI.

## 0.2.0 — 2026-09-28

Renamed to `trellis-harness`: the package is `trellis.harness`.

## 0.1.0 — 2026-09-16

The first harness: memory, tools and approvals around an agent you built.
