# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## 0.6.0 — ReAct on Deep Agents; `middleware()` (unreleased)

`ReAct` is Deep Agents' `create_deep_agent`, every Deep Agents feature native, with the
harness's part on top — and that part is exported: `trellis.middleware()` gives a
`create_agent` or `create_deep_agent` graph of your own what `ReAct` adds.

### Breaking

- `ReAct` builds on `create_deep_agent` (was LangChain's `create_agent`): Deep Agents' file
  tools (on `backend=`), large-result eviction, summarization, sub-agents (`task`), skills,
  memory files, permissions, dangling-call repair and prompt caching are its own. The new
  signature, names following Deep Agents': `ReAct(system_prompt, model, *, tools, subagents,
  skills, memory_files, permissions, backend, store, response_format, state_schema,
  context_schema, cache, max_steps=12, max_repeats=3, call_timeout, context_window, effort,
  tool_guidance=None, answer_check=None, capabilities, middleware, checkpointer)`
  (`memory_files` is Deep Agents' `memory`; its `interrupt_on`, `name` and `debug` are not
  exposed).
- `ReAct(tools=)` refuses `sandbox()` (code runs in `backend=docker_sandbox()`) and agents
  (`a2a(url)` and wrapped agents go in `subagents=`); `max_repeats` is 2 or more, as
  `StallGuard` always required.
- `tool_guidance` and `answer_check` default to `None`, automatic: the guidance with 20 tools
  or more on a call and a known window of 32,000 tokens or more, the check when the run's
  context holds a rule (measured: see Changed). `False` turns either off.
- `ReAct(store=)`, like `checkpointer=`, is for the graph run on its own: `h.wrap` refuses it.
- `ReAct(middleware=)` with a Deep Agents middleware's name replaces Deep Agents' own (as
  `create_deep_agent` does); one named as the harness's is still refused.
- Removed, Deep Agents now owning them: `automatic.JournalFiles`, `automatic.FilesState`,
  `automatic.ScratchFiles` (`automatic.LongTaskTools` offers Deep Agents' own scratch tools and
  general-purpose helper on a long task); `ModelHooks(routed=)` (it sees a routed model itself).

### Added

- `trellis.middleware(*, max_steps, max_repeats, call_timeout, context_window, tool_guidance,
  answer_check, capabilities, system_prompt)`: the harness's part of `ReAct` — the run's tools
  per model call narrowed per turn (the memory and gateway MCP tools among them), the graph's
  own tools governed, the stall guard, the step limit then a final answer, the budget's stop,
  planning on a long task, older results cleared and read back, the tool guidance and answer
  check, and `ModelHooks` (hooks, spans, `call_timeout`, a stored prompt pinned per run).
  `h.wrap` says once when a `create_agent` or Deep Agents graph is wrapped without it.
- `trellis.docker_sandbox()`: the run's sandbox (the lifecycle of `sandbox()`: made at the first
  command, journaled, paused with the run, deleted at its end, reaped) as a Deep Agents backend
  — `ReAct(backend=docker_sandbox())` or `create_deep_agent(backend=...)`; `execute` governed.
- `ReAct(subagents=)` takes Deep Agents' `SubAgent` and `CompiledSubAgent` dicts (a `ReAct` graph
  as the `runnable`) beside wrapped agents and `a2a(url)`: each `task` call a child run; a
  declarative sub-agent gets the loop's guards and its model name routed, and a tool it names in
  `interrupt_on` asks a person (its declared risk, as the catalog says).
- On a backend outside the graph's state (a disk, a store, a sandbox) Deep Agents' file tools
  are governed and published as any tool (`middleware.OUTSIDE_STATE`).
- `RunCheckpointer` keeps a LangGraph `DeltaChannel`'s history (its last snapshot and the writes
  since, replayed in the order they were applied) beside the latest checkpoint, so a Deep
  Agents graph's `messages` and `files` come back whole on a resume anywhere.
- `LIVE_SHARD=i/n` (`make live-shard`): the live suite in shards packed by the measured
  durations (`tests/live/durations.json`, `make live-durations`), each under a job's limit on a
  CPU model.
- Python 3.13 beside 3.12: CI's gates (lint, types, tests, examples, docs, the matrix, the
  overhead benchmark) and the weekly canary run on both (`python-version: ["3.12", "3.13"]`);
  the 3.13 classifier.
- `benchmark.cost` variants `agent` (`create_agent` + `middleware()`), `main+guidance`,
  `main+check`; `TRELLIS_LIVE_VARIANTS` picks the live tool-choice test's variants.

### Changed

- Deep Agents' summarization stands after the harness's middleware that narrow the tools, so
  its budget counts what the model is sent.
- The base and the automatic rules were chosen by one Granite batch (the cost benchmark's ten
  tasks at 10 and 50 tools; the tool-choice store sets at 11 and 51 tools):
  [docs/frameworks/react.md](docs/frameworks/react.md#tool-choice).

### Fixed

- A governed framework tool call inside a sub-agent whose call id equals its parent's `task`
  call id (ids are unique per conversation only) was let through ungoverned; the marker is now
  the tool's name and the call's id.

## 0.5.0 — one breaking release (unreleased)

One API, each setting in its one place, no deprecated aliases: the environment says where the
services are; `Harness()` connects to them; the builder (`ReAct`, or your framework's own) says
what the agent is; `h.wrap` who it is and how its runs go; `agent.run` this call. Needs
`trellis-memory` 0.5 (the memory service's prompt-form `memory_ids`).

### Breaking: removed, with what replaces it

- `h.wrap(tools=, mcp=, skills=, timeout=)`: tools are the builder's — `ReAct(tools=[...],
  subagents=[...])`, or your framework's own tools, governed as it runs them (a graph adds the
  harness's with `HarnessTools()`; a function target calls `agent.tools.call(tool, ...)`);
  `h.wrap(run_timeout=)`.
- `Harness(governance=, prompts=, skills=, hooks=, judges=)`: hooks and judges are
  `h.wrap(hooks=, judges=)`; governance is the memory service's catalog; stored prompts are the
  gateway's.
- `h.tools`, `h.prompt_messages`, `h.model_headers` (the gateway's headers are sent for every
  routed model call; Claude's automatically), `h.resolve`, `h.built_for`, and `h.key`,
  `h.tenant`, `h.governance`, `h.toolbox`, `h.score` (internal now).
- `agent.as_tool()` (a wrapped agent in `ReAct(subagents=)`), `remote()` (`a2a(url)` is an
  async callable; `RemoteAgent` under it), `RunHandle.cancel` (`agent.cancel(handle.id)`),
  `RunHandle.run_id` (`handle.id`), `handle.result(timeout=)` (`wait=`),
  `agent.schedule(cron, input, on_behalf_of=)` (`schedule(cron, user=, input=)`).
- The skills API (`skills`, `skills_dir`, `Skill`, the gateway's Skills Repository source) and
  the prompt sources (`prompts_dir`, `Prompt`, Langfuse prompts): `stored("name", **values)` and
  `h.prompt(...)` read the gateway's Prompt Repository, the one prompt store.
- Virtual MCP selection (`mcp("slug")`): the key's MCP allow-list, in Bifrost.
- `ReAct(system=, prompt=, prompt_vars=, model_timeout=, output=)`: `system_prompt=` (a string
  or `stored(...)`), `call_timeout=`, `response_format=`; `ReAct(checkpointer=)` is for a graph
  run on its own (`h.wrap` refuses it).
- `tool/mcp/openapi/a2a/sandbox(timeout=)`: `call_timeout=`; `sandbox(provider, spec)`:
  `sandbox(provider, image=, spec=)`, Docker when no provider is given.
- `without=` names `memory_push`, `memory_pull`, `records`, `hints`, `skills`, `planning`,
  `files`, `react_auto`: the six names are `memory`, `tool_search`, `mcp`, `code_mode`,
  `grounding`, `judges`.
- Environment: `PROMPTS_DIR`, `SKILLS_DIR`, `TRELLIS_AGENT_VERSION`,
  `TRELLIS_WORKER_CONCURRENCY`, `SANDBOX`, `SANDBOX_IMAGE`, `TRELLIS_JUDGE_MODEL`,
  `TRELLIS_EVAL_SAMPLE`, `TRELLIS_GROUNDING_SAMPLE`, `TRELLIS_JUDGE_SAMPLE`, `LANGFUSE_HOST`;
  `Settings` has seven fields (`otlp_*` and `langfuse_*` are gone: their own SDKs read their
  variables). A fixed tenth of runs is grounded; each judge's share is `llm_judge(sample=)`.

### Added

- `stored(...)`, `llm_judge` exported from `trellis`; `h.wrap(description=)`;
  `run`/`stream`/`start`/`schedule(budget=)` (a run's own budget, kept with its record);
  `agent.tools.use(*sources)`; `a2a(url)(message, user=, tenant=)` outside a run.
- A Claude in-process MCP server's tool is governed by its annotations (`destructiveHint`:
  asks), as the run's tool set lists it.
- A function target as a sub-agent counts as a write unless its tools are declared.

### Changed

- Memory's window as the conversation itself: where memory owns a thread and the framework
  keeps no store (`ReAct`, a LangGraph graph without a checkpointer), the thread's recent turns
  arrive as LangChain's own messages — `HumanMessage`/`AIMessage` with `tool_calls`/
  `ToolMessage` — after the one memory system message, not as text inside it; an OpenAI agent's
  memory session gives the SDK the same turns as input items, its tool calls and results
  included. Each run's question, tool calls, results
  and answer are written back before the run returns (queued past 5 s or when memory is down),
  so the next turn reads them wherever it runs. `trellis.harness.history` holds the policy and
  the messages (`Said`, `from_window`). Needs the memory service's `window_format="messages"`.
- Redaction masks credentials in traces only. Span attributes still lose secret-named fields,
  bearer tokens, JWTs, `sk-` keys and key-like blobs; e-mail addresses are no longer masked
  (personal data is the domain's to define). The run's event stream (`agent.stream`, AG-UI,
  A2A task updates and pushes), its answer, the memory service's transcript, tool records and
  spool, and the journal now carry values as they are, so memory search and recall see the
  real data.

### Fixed

- OpenAI Agents: the tools of every agent reachable through `handoffs` (recursively, on copies)
  are governed — hooks, the catalog's rules, the journal; a `Handoff` object whose agent is
  hidden is said once. The harness's `RunHooks` are always passed, so a token budget is enforced
  without any hooks.
- Claude Agent SDK: built-in tools are governed by the SDK's own `PreToolUse` hook, so
  `allowed_tools`, `acceptEdits` and `bypassPermissions` no longer bypass governance; your
  `can_use_tool` is asked by that hook.
- History owner: one policy (`trellis.harness.history`) with a registry of RAM-only stores. A
  LangGraph `InMemorySaver`/`MemorySaver` or an OpenAI `SQLiteSession(":memory:")` no longer
  turns memory's window off (warned once); OpenAI `conversation_id`/`previous_response_id`/
  `auto_previous_response_id` count as the framework keeping the thread.
- An `approve_when` rule that cannot be evaluated asks a person (never allows), is logged once
  per rule, and is recorded on the run as an `APPROVAL_RULE_INVALID` warning
  (`ApprovalRuleError`).

### Added

- `Hooks.on_output(item) -> item`: mask what a user is shown. It runs, in order (the harness's
  hooks, then the agent's), on the answer `agent.run`/`agent.resume`/`RunHandle.result` return
  (`OutputKind.ANSWER`) and every event of `agent.stream`, `agent.events` and `serve_chat`'s
  AG-UI stream (`OutputKind.EVENT`); what it returns is sent. It never touches memory, the
  journal, A2A, traces or what the model and tools see. A hook that raises is logged and the
  item is withheld (`WITHHELD`), never sent as it was; with no hook overriding it nothing is
  wrapped.
- Native slots backed by our services (`trellis.harness.bridges`), with no new settings and
  each framework's defaults kept without `MEMORY_URL`: the memory service as a LangGraph/Deep
  Agents `BaseStore` (only when the graph has none; its own is kept, said once), as an OpenAI
  Agents `Session` (only when the run has none), and as a Claude `UserPromptSubmit` context
  hook. Every turn is still recorded to memory by the harness. `without={"memory"}` turns them
  off.
- Durable runs: Claude's session kept in the run's journal as the SDK's `SessionStore`
  (progress saved before each step), so a resume on another worker continues it; the OpenAI
  `RunState` of a pause resumes on any worker.
- Native helpers as child runs (lineage, turns in memory, budget, governance): a Deep Agents
  `task`, an OpenAI `Agent.as_tool` call, a Claude `agents=` subagent.
- P4: LangGraph `context=` (a `context_schema`) and `durability` pass through
  `framework_options`. P7: Claude's usage and cost are read from its `ResultMessage`.

- Learned approval suggestions are accepted automatically when the tool catalog refreshes
  (the memory service's accept endpoint), except for a tool at the highest risk, which waits for
  a person; each acceptance is logged and `trellis doctor` lists the accepted ones.

- `python -m benchmark.thread`: who keeps a conversation's history across runs, the framework's
  own thread or the memory service's window, measured on recall and input tokens per turn.
  Measured on Granite through Bifrost with the memory service: recall 1.00 vs 0.92, 202 vs 214
  tokens a turn — the owners stay as they are (ADR 0003 §6).
- Without Bifrost, model names go to any OpenAI-compatible endpoint the OpenAI SDK's own
  `OPENAI_BASE_URL` / `OPENAI_API_KEY` name (LiteLLM's proxy, Portkey, vLLM, Ollama, OpenAI):
  `ReAct(model="name")` and `llm_judge(model="name")` work there, with the client's own retries.
  Exactly one gateway serves names — Bifrost, else that endpoint, else none (a name is then an
  error naming the three) — and each declares its capabilities (`mcp`, `code_mode`, `cost`,
  `fallbacks`, `prompts`, `skills`; `trellis.harness.gateways`): what the deployment's gateway
  cannot do is off, and `trellis doctor` lists it.
- Approval rules have one place, the tool catalog (`risk`, `approve_when`), documented as such
  (governance.md "One place for approval rules"): hooks are for what a rule cannot say (a
  refusal, a rewrite, outside data, who reviews and where), and the examples that asked by an
  amount in a hook now set the catalog's rule. A framework's own approval gate still works,
  asked once, and the warning at `h.wrap` says its rule belongs in the catalog.
- One `tools=[...]` for everything an agent calls. `mcp(...)` is the entry for every MCP source:
  `mcp("https://host/mcp")` (streamable HTTP, `headers=` for its auth), `mcp(stdio("npx",
  "@mcp/files"))` (a local server the harness starts), or `mcp("finance-tools")` (a Bifrost
  Virtual MCP). A server of the agent's own is reached with the official MCP SDK (the new `mcp`
  extra), one kept connection per server, and its tools get the same `<server>-<tool>` names,
  annotations-to-risk, governance, journal, retries, paging and records as the gateway's. With a
  gateway and no `mcp(...)` entry the agent still has every MCP tool its key allows; once it
  names any, it has exactly those. A server reached both directly and through the gateway is an
  error. A wrapped agent in `tools=[...]` is a sub-agent. The same holds in `h.tools(...)` for
  every framework.
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

- `handle.result()` waits out agent-runs' request budget (429, its `Retry-After`) and polls
  again, instead of failing the wait.
- Every piece of the harness's own client-side state is bounded, so a long-lived process
  serving many tenants, tools and threads does not grow: governance and toolboxes per tenant,
  the catalog rules kept fresh and the published-once set (4,096 each), prompts and skills
  kept, an in-memory checkpointer's threads outside a run (10,000), the `h.tools` builds
  awaiting a wrap, and the names a once-only warning remembers (256) — each the most recently
  used, the rest read, sent or said again. The write spool stops at 64 MB (a write given up
  past it is lost, said once) so a long outage cannot fill the disk.
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

- Deprecated, working for one release with a warning naming the replacement:
  `h.wrap(mcp=[...])` and `h.tools(mcp=[...])` (`mcp("name")` in `tools=[...]`; `mcp=[]` is
  `without={"mcp"}`) and `agent.as_tool()` (the agent itself in `tools=[...]`; a sub-agent's
  risk, when its tools do not say it, is set in the tool catalog).
- `make bench-load` measures the production path too (`--queued MINUTES`): runs queued in
  agent-runs, claimed by worker processes, with the memory service on or as the key registry
  only, each process's resident memory slope reported (the harness workers, the memory API
  and worker, agent-runs, its ticker).
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
