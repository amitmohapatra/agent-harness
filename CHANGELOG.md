# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## Unreleased

### Added

- Every gateway call says whose run it is and joins its trace: Bifrost dimension headers
  (`x-bf-dim-tenant`, `-agent`, `-run`, and a hash of the user, never the id) and the W3C
  `traceparent`, on `ReAct`'s calls, the judge's (`x-bf-dim-purpose: judge`) and
  `h.model_headers()` (read per request). The OpenAI Agents SDK's own trace export is left as
  set, said once.

- A framework's own tools are governed: an OpenAI Agents agent's `FunctionTool`s by `h.wrap`,
  on a copy of the agent; a `create_agent` or Deep Agents graph's tools through
  `h.governance_middleware()` (`ReAct` has it) — decided by risk (the catalog's, else the
  tool's declared `side_effects` or annotations, else `write`), asked when governance asks,
  journaled, timed and traced. The graph's state tools are let through.

- One approval per call: a tool held by the framework's own approval gate
  (`HumanInTheLoopMiddleware`, Deep Agents' `interrupt_on`, an OpenAI Agents `needs_approval`
  tool) and by governance is asked once — governance counts the framework's approval, kept in
  the run's journal; a hook's `Deny` still wins. `h.wrap` logs each tool both gate.

- Budgets, per agent and opt-in (`h.wrap(budget=Budget(max_tokens=, max_cost=, max_tool_calls=))`;
  none given, no cap): counted on the run's journal — so a resume goes on from what was spent
  and a sub-agent's spending counts toward its parent's. At a limit no new tool call starts, a
  `ReAct` gives its best answer with what it has, and the run ends `PARTIAL`, its result's error
  `BUDGET_EXCEEDED`. Dollars are what the gateway says each call cost; a tenant's or a team's
  budget is the gateway's.

- `ReAct(output=)` takes a JSON schema as well as a pydantic model; `ReAct(effort=)` sends a fixed
  `reasoning_effort` with every call of a model name; `ReAct(settings=)` takes the gateway's
  address and key in code. A read tool's retry waits what an error's `Retry-After` says when
  there is time left, else gives up. Cached input tokens are on the `chat` span.

- `ReAct` plans and keeps scratch files on a long task — LangChain's `TodoListMiddleware` and
  Deep Agents' file tools — once the model has made three calls or the request asks for steps,
  on a known window of 32,000 tokens or more; a shorter task is sent what it was before.
  `without=` turns each off: `planning`, `files` ([react.md](docs/frameworks/react.md#the-middleware)).

- `make bench-cost` (`python -m benchmark.cost`): the cost and accuracy of a `ReAct` agent over
  tool catalogues of 10, 50, 100 and 500 tools, against whatever model is configured — input,
  output and cached tokens, model calls, wall time, `tool_coverage` and task success per run,
  dollars from a price table with sources (`benchmark/cost/prices.py`), means and 95%
  intervals per variant as JSON and markdown ([evaluation.md](docs/evaluation.md#cost-benchmark)).

### Fixed

- A long-lived process without `RUNS_URL` no longer grows with every run: `LocalRuns` kept
  every finished run forever (about 4 kB each; the 30-minute load check grew 130 MB over 38,758
  runs). It now keeps the last 1,000 finished runs for an hour each, with their artifacts
  (`LocalRuns(keep_finished=, keep_finished_seconds=)`); paused, queued and running runs are
  always kept. agent-runs (`RUNS_URL`) stays the durable store for production.
- A `ReAct` run's files — a large result saved by `FilesystemMiddleware`, a summarized history
  — were lost on a resume (Deep Agents keeps them in a `DeltaChannel`, which needs every
  earlier checkpoint, and the run keeps only the latest): they are now in a channel the
  journal keeps whole.

### Changed

- The deprecated `TRELLIS_GROUNDING_SAMPLE` and `TRELLIS_JUDGE_SAMPLE` (and the matching
  `Settings` fields) are read as `TRELLIS_EVAL_SAMPLE` taking the larger of the two, so setting
  one to 0 no longer turns the other off; turn one part off with `without={"grounding"}` or
  `without={"judges"}`.
- The cost benchmark records each model call's seconds (p50 and p95 per cell) and the CPU
  seconds per run of its own process and of each `--watch name=pid` one (the model server, the
  gateway), with their peak memory. `make bench-gate` checks a report against the stored
  baseline (non-inferiority: success within 0.15, input tokens within 10% on the runs both did), run by
  `.github/workflows/bench.yml` where a gateway is configured; `make bench-load` measures many
  runs at once on one and two worker processes and checks for a memory leak.
- The retry loops (read-only tool retries, background memory writes, A2A push
  notifications) run on `tenacity`, now a direct dependency; the attempts, waits and
  `Retry-After` handling are unchanged.
- The judge asks the gateway's judge alias `trellis-judge` (routed by the Bifrost admin) unless a
  judge names its model; it never falls back to the judged agent's model. An alias the gateway
  does not route turns the judges off, said once. Bifrost's semantic cache is not used for
  judges.

- `without=` has fewer names: `memory`, `tool_search` (the tool hints and memory's `tool_search`
  tool), `mcp`, `code_mode`, `skills`, `judges`, `grounding`, `planning`, `files`, and the
  umbrella `react_auto` (`planning` and `files`). `memory` now also turns the narrowing off.
  `memory_push`, `memory_pull`, `records` and `hints` still work for a release, with a warning.

- Configuration: each setting in one layer. The environment holds endpoints and keys
  (`TRELLIS_JUDGE_VIRTUAL_KEY` kept: evaluation's own budget), one `TRELLIS_EVAL_SAMPLE` for
  grounding and judges, `TRELLIS_SPOOL_DIR` (default: the XDG state folder, except in a
  container or serverless runtime, said once) and `SANDBOX`; the tools' own variables are read
  under their own names — the OpenTelemetry SDK's exporter reads `OTEL_EXPORTER_OTLP_*` itself
  (`..._TRACES_ENDPOINT` honoured), `service.name` is only a default under `OTEL_SERVICE_NAME`
  (fixed: it overrode it), the agents' version is `OTEL_RESOURCE_ATTRIBUTES`' `service.version`,
  Langfuse is `LANGFUSE_BASE_URL` and its keys (scores no longer read from OTLP headers), and the
  Docker daemon is found as docker-py finds it (`DOCKER_HOST`, the docker context, the socket).
  Deprecated, read for one release with a warning naming the replacement: `PROMPTS_DIR`,
  `SKILLS_DIR`, `TRELLIS_AGENT_VERSION`, `TRELLIS_WORKER_CONCURRENCY`, `SANDBOX_IMAGE`,
  `TRELLIS_JUDGE_MODEL`, `TRELLIS_GROUNDING_SAMPLE`, `TRELLIS_JUDGE_SAMPLE`, `LANGFUSE_HOST`.
  A startup log line names what the harness is connected to, and `trellis doctor` prints every
  setting with its source, checks each service and key, the spool and the Docker daemon, and
  suggests what to set. Each prompt and skill source in use is logged.

- One owner per concern (the architecture review, ADR 0003): model retries, fallbacks and prompt
  caching are the gateway's — `ReAct`'s client makes one attempt, where it made three (set the
  provider's `network_config.max_retries` in Bifrost); a model's window is `context_window=`, else
  its chat model's LangChain profile — when neither says, clearing old tool results is off (it
  assumed 128k before), said once; a run's dollars are the gateway's cost.

- A function tool's description is its docstring without the sections (`Args:`, `Returns:`,
  `Raises:`...), every other paragraph kept (was: the first paragraph only), and each parameter
  is described by the docstring's `Args:` or `:param x:` entry unless its annotation says.
- The Code Mode meta-tools and `read_skill_file` describe every parameter.

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

- Tool choice. `ReAct(tool_guidance=True)` (`ToolGuidance`) adds how to choose among tools to
  every call's instructions — every part of a request addressed, the most specific tool over a
  general one, current figures from tools rather than memory, a change made by calling its
  tool and never claimed without it — naming the run's tools that change data;
  `ReAct(answer_check=True)` (`AnswerCheck`) checks the draft answer against the request once
  before it is final (one more model call; the draft is not streamed), so a skipped part or a
  change only claimed in text makes the model call the tools still needed, and a run narrowed
  by tool hints is offered every tool from the check on; when every tool result is empty or not
  found (or memory is `INSUFFICIENT`) the check asks for a plain "no record"; and the first
  change of a run is checked against the rules in the instructions and memory context before
  it runs or is sent for approval. `ReAct(capabilities={...})` tells the model what it can and
  cannot do beyond its tools, for a plain yes or no. All LangChain middleware, opt-in.
- `trellis.harness.tools.lint.lint(specs)`: what makes a tool set hard to choose among (missing
  or short descriptions, undescribed parameters, names the OpenAI API refuses, look-alike
  names and descriptions, 20+ or 128+ tools at once); the toolbox logs each finding as a
  warning when its tools change.
- `tool_coverage()`: an evaluator scoring each item by the share of its `needed_tools` (the
  item's metadata) the run called, naming the missing ones and any `avoid_tools` called; it also
  scores a run that paused for approval, the call it paused on counting as made.
- `examples/06_scenarios/tool_choice.py` and `tests/live/test_live_tool_choice.py`: a tool set
  reproducing the four typical tool-choice failures, measured with and without the switches.

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
  tools until the CLI's turn limit. A pause that names no call (an `ask`) keeps the old wording;
  after a deny the session is told the call will not run.
- Grounding (`grounding()`, the sampled check on a run) is no verdict and no score against a
  context that held no evidence, as the memory service records none: it was 0.0 on the trace,
  every claim "unsupported" against nothing, one the run's own tools returned too.

## 0.3.0 — 2026-09-28

The harness core: run events, one interrupt mechanism, MCP and memory tools, AG-UI.

## 0.2.0 — 2026-09-28

Renamed to `trellis-harness`: the package is `trellis.harness`.

## 0.1.0 — 2026-09-16

The first harness: memory, tools and approvals around an agent you built.
