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

- Langfuse is set up with its own three names alone: with `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` set (at `LANGFUSE_HOST`, else Langfuse Cloud) and
  `OTEL_EXPORTER_OTLP_ENDPOINT` unset, `Settings.from_env()` derives the OTLP export as
  Langfuse's SDK does — `<host>/api/public/otel`, `Authorization: Basic base64(pk:sk)`,
  `x-langfuse-ingestion-version: 4` — so traces, scores, datasets and prompts reach one project
  from one set of credentials. A set `OTEL_EXPORTER_OTLP_ENDPOINT` still wins (a collector keeps
  working), a header in `OTEL_EXPORTER_OTLP_HEADERS` wins over the derived one, and the explicit
  `OTEL_*` form for Langfuse works as before.
- `h.model_headers(prompt=)` is pinned per run, as `ReAct(prompt=)` is: every run of the
  harness pins each stored prompt handed out at its start (journaled, so a resume keeps it; a
  `prompt` event; `trellis.prompt.*` on the run's agent span), and the headers — a mapping read
  at each request — select the version the run executing pinned (outside a run, the one
  resolved when they were made). An OpenAI client (`AsyncOpenAI(default_headers=...)`, the
  OpenAI Agents SDK) reads them per request; a LangChain chat model, which copies its headers,
  selects the run's version through `ModelHooks()`; the Claude Code CLI's
  `ANTHROPIC_CUSTOM_HEADERS` get it as the CLI starts. A prompt a run cannot pin is a
  `prompt_unavailable` warning. With `prompt=` it returns that `Mapping[str, str]`, not a `dict`
  ([gateway.md](docs/gateway.md#prompts)).
- `serve_chat`: `POST {path}/runs/{run_id}/cancel` stops a chat run (the run's own user only),
  as A2A's `tasks/cancel` and `agent.cancel` do.
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
- Large results on OpenAI Agents and Claude, automatically, as `ReAct` and Deep Agents already
  had them: a harness tool's result over 80,000 characters is kept in the run's journal (across
  a pause, on any worker), the model reads Deep Agents' head-and-tail preview naming
  `/large_tool_results/<id>` and pages the rest with `read_file` — offered once there is
  something to read on OpenAI Agents, listed on the `trellis` server from the start on Claude
  ([tools.md](docs/tools.md#large-results)). Deep Agents is not needed. Closes matrix gap G8
  (F26); Way 2 is n.a. by design (your code gets the whole result).
- Tool hints narrow a `create_agent` or Deep Agents graph's model calls, automatically: its
  tools are bound when it is built, so the harness runs a copy of it whose model calls go
  through one more `awrap_model_call` handler, offering only the tools the run offers — the
  hinted ones, the memory tools, those already used, what a `tool_search` found — and leaving
  out a part turned off after the graph was built (`without=`), as `ModelHooks` does. Your
  graph object is unchanged; its own tools (Deep Agents' `ls`, `task`, ...) are always offered.
  A hand-written `StateGraph` binds its model's tools in its own code and is not narrowed
  ([tools.md](docs/tools.md#tool-hints-what-the-model-is-offered)). Closes matrix gap G12
  (F28).

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
- Claude Agent SDK: a session resumed after a person answered about a tool call is told which
  call (`mcp__trellis__<tool>`, or the built-in's name) to make again. It was told to repeat
  "your last tool call" if it had "no result yet", but the session holds that call's result as
  "Waiting for a person's approval.": the live model took it for a failure and called other
  tools until the CLI's turn limit. A pause that names no call (an `ask`) keeps the old wording.

## 0.3.0 — 2026-09-28

The harness core: run events, one interrupt mechanism, MCP and memory tools, AG-UI.

## 0.2.0 — 2026-09-28

Renamed to `trellis-harness`: the package is `trellis.harness`.

## 0.1.0 — 2026-09-16

The first harness: memory, tools and approvals around an agent you built.
