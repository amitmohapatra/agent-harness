# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## 0.7.0 — any provider: one route strategy, an auth strategy, a model's profile (unreleased)

Model access is provider-neutral ([ADR 0004](docs/adr/0004-provider-neutral-models.md)): a
model name goes to any OpenAI-compatible endpoint, to Bifrost, or to the provider's own SDK,
chosen by name from one registry, and the memory service is told the agent's endpoint so its
own model work follows the harness's configuration.

### Breaking

- `BIFROST_URL` and `BIFROST_VIRTUAL_KEY` are gone, with no alias: `LLM_BASE_URL` (the
  endpoint's `/v1` root) and `LLM_API_KEY` (its bearer key; Bifrost's virtual key), with
  `LLM_GATEWAY=bifrost` for Bifrost. `Settings(bifrost_url=, bifrost_virtual_key=)` are
  `Settings(llm_base_url=, llm_api_key=, llm_gateway=, llm_auth=)`. An endpoint without
  `LLM_GATEWAY` is `openai-compatible`: model names as given, models only.
- `trellis.harness.gateways`: `ProviderKey` is `ProviderSDK`, `GatewayKind` is each route's
  `name` (its `LLM_GATEWAY` value), `bifrost_wins` is the route's `announce`; `GATEWAYS` maps
  names to routes and `model_gateway` picks from `registry()`.
- The memory service is sent the agent's whole model endpoint (`trellis.contracts.Endpoint`)
  where it took a virtual key (`MemoryScope.register_endpoint`, was `register_model_key`),
  needing trellis-memory with endpoint registration; trellis-contracts 0.6.3 and bifrost-sdk 0.4.

- A conversation is its tenant's, its user's and its thread's (`identity.conversation_key`, a
  digest of the three): the memory service's thread (its window, the OpenAI Agents
  `MemorySession`, `h.add_document(thread=)`), a graph's own checkpointer `thread_id`, the turn
  a busy conversation waits for in process, `start`'s default `concurrency_key`
  (`thread:<conversation>`) and the trace's `session.id` were keyed by the bare thread id, so
  two users (or tenants) giving the same `thread=` shared them. A conversation kept under a
  bare thread id before this release is not continued: its next run starts a new one. Read a
  graph's own state with `thread_id=await h.conversation(user=, thread=)`. AG-UI answers an
  interrupt only for the user whose run it is (`404` for another user's thread named alike).

### Changed

- The tool hints narrow what the model is offered only from 30 tools in the run
  (`agent.NARROW_FROM`). Below that, every tool is offered and the hints' tools section
  still says which tools fit. Narrowing a set of 20 hid a tool the question needed (a name
  looked up before its id) in 22-26% of turns. Those turns were wrong 2-3 times as often, on
  every adapter.

### Fixed

- A run's first scratch write (Deep Agents' `write_file` or `edit_file` on the graph's state)
  says the file is the run's own scratch copy, which the user cannot open or download
  (`middleware.SCRATCH_NOTE`). Asked for a chart, a model saved an SVG at `/tmp/…` in the
  graph's state and told the user to open it (break-and-fix round 1, Deep Agents).
- A conversation is no longer lost when memory fails mid-chat. Where memory replays the
  conversation (its window) and its context call fails (an outage, the circuit open), the
  earlier turns come from the run store, as with memory off; before, the model was sent the
  question alone and answered "this conversation has no earlier list" (break-and-fix round 1).
- OpenAI Agents: a call to a tool the model made up is an error the model reads and the run goes
  on (the SDK's `tool_not_found_behavior="return_error_to_model"`), not a `ModelBehaviorError`
  that ended the run with no answer (ADV-8).

- A table of 30 rows or more that fits the context is read whole and also kept, so `query_result`
  counts, sums, ranks and joins over it exactly (`tools.results.tabled`). Before, only a
  result over 80,000 characters was kept, so a small model added up hundreds of rows by hand
  and got them wrong.
- A Docker sandbox's anonymous volumes (an image's `VOLUME`) are removed with it: before, each
  run of such an image left one volume behind (50 runs, 50 volumes; now none).

- A tool call whose output is over 256 KiB of JSON is recorded with the memory service again:
  the service refuses a larger recorded output (422), so the call's record was lost and the
  service never learned from it. The output is recorded cut to fit, saying so
  (`clients.memory.recorded_output`).

### Changed

- The note the model gets when memory holds nothing (or only part) for the question says
  what an honest answer does: name the detail it does not have, and offer to look it up or
  ask the user for it, instead of a bare "say you do not know" (`agent.ABSTAIN_NOTES`).

### Added

- `query_result(save_as="csv" | "xlsx", file_name=)`: a query's every row kept as a file for the
  user (`Runtime.save_file`), in a run with no tool that runs code (`tools.export.runs_code`);
  each adapter offers the run's own variant (about 50 tokens more, and none where
  `query_result` is not offered). Excel needs the new `xlsx` extra (openpyxl).
- A sandbox's files for the user: the command tool says that files written in `outputs/` are
  given to the user. When the run succeeds, each file directly in `outputs/` is kept in the file
  store (`sandbox.delivered`: named in the journal, announced, served over AG-UI) before the
  sandbox is deleted.

- The sandbox's environment card: `sandbox_exec`, and Deep Agents' `execute` under
  `ReAct(backend=sandbox())`, are described with the image's Python version and installed
  distributions, and with no network, that nothing else can be installed
  (`SandboxSource.card`). The card is read once per process in a sandbox made for it alone.
  [sandbox.md](docs/sandbox.md) has an image recipe that installs from your index
  (`PIP_INDEX_URL`, `UV_INDEX_URL`, a build secret for credentials).
- A file store for the files runs make for their users (`trellis.harness.files`, on fsspec):
  `TRELLIS_FILES_URL` (`gs://`, `s3://`, `az://` with `gcsfs`, `s3fs`, `adlfs`; `file://`),
  else the XDG data folder. `Runtime.save_file(name, data, source=)` keeps a file under
  `<tenant>/<user>/<conversation>/<run>/`, names it in the run's journal and announces it (the
  `file` event); `serve_chat` serves it to the run's own user at
  `GET {path}/runs/{run_id}/files/{name}`: a 5-minute signed link where the store signs, else
  streamed. Stored only: never parsed or remembered. New dependency: `fsspec`.

- Large results in `ReAct` without a backend and in a `create_agent` graph with
  `middleware()`: kept in the run and previewed, paged with `read_file`, as on OpenAI Agents
  and Claude (`tools.results`); a result `FitWindow` cuts to fit a small window is kept too,
  its cut naming the `read_file` call that reads on from there.
- `query_result` runs on DuckDB when the new `query` extra is installed (in `all`), and on
  Python's own SQLite otherwise, chosen by what is installed ([ADR 0007](docs/adr/0007-query-engine.md)):
  one contract on both (the description names the engine; the same tables, column typing,
  `percentile_cont(x, p)` and `julianday(d)`, and limits), DuckDB locked after loading (no
  file, network, extension or setting; a 512 MB `memory_limit`). Loading a run's 4 results
  of 50,000 rows takes 0.37 s (was 1.01 s), of 500,000 rows 2.5 s (was 9.5 s); 8 calls at
  once over 50,000 rows take 1.8 s (was 7.2 s), over 500,000 rows 12 s (was 89 s)
  (`benchmark/query_engines/results/harness.md`).
  Both engines: a number is only what SQL writes as one (`1_000`, `0x10` stay text), a nested
  JSON value is its compact JSON text, a boolean answers `true`/`false`, a value a SQLite
  query builds is at most 64 MB, and a query past its time is interrupted by a watcher
  thread on either engine.
- `query_result`: read-only SQL (Python's own SQLite, in memory, no dependency) over every row
  of the run's saved results — each a table named by its tool, so one query joins them — with
  formulas, window functions and statistics (`stddev`, `median`, `percentile_cont`, `corr`,
  `regr_*`), offered with `read_file` on every adapter, bounded in time, input, sort memory and
  answer (`tools.query`). A tool declared `paged=True` is fetched to its last page in one call
  (`tools.pages`). A run with a sandbox finds every kept result as a file there too.
- Freshness: a tool's data is live by default; `tool(max_age=)`, `tool(freshness="static")`
  (or `"live"`), a LangChain tool's `metadata`, an MCP tool's `_meta` say otherwise
  (`tools.base.data_of`). Tool results written to memory's thread start with
  `[snapshot of <tool> from <UTC time>]`; replayed past the tool's age they are marked out of
  date, the context says so, and the tool stays offered (`trellis.harness.snapshots`).
- `benchmark/cost`: the `aggregate` and `freshness` classes over 10,000 SKUs.
- Tool chains (`trellis.harness.tools.links`): which tool's output is another's argument, read
  from the schemas the tools publish (a function's return annotation, now its
  `ToolSpec.output_schema`; an MCP tool's `outputSchema`), said at the end of each linked
  tool's description (`Chains: find_sku → sku_id → get_stock.`) in every framework, and kept
  whole when the tool hints narrow what is offered ([tools.md](docs/tools.md#tool-chains)).
  `benchmark/cost --suite classes --classes chain` measures two- to four-call chains.
- `docs/research/`: the improvement backlog, the evaluation protocol every change must pass to
  be kept, the scenario coverage audit and the self-explanation research.
- Conversations across runs ([ADR 0005](docs/adr/0005-conversations-across-runs.md)):
  - with memory off, a run of a conversation is given its earlier turns from the run store
    (each run's question and answer) where the framework keeps no thread of its own —
    messages for `ReAct`, graphs and OpenAI Agents, `agent.history` for a function, the
    context's text for Claude and ACP;
  - `run(mode=)` / `stream(mode=)`: `"queue"` (default, as before), `"interrupt"` (cancel the
    conversation's working run and continue), `"steer"` (give the message to the working run
    at its next step: `HarnessTools.before_model`, OpenAI Agents `call_model_input_filter`,
    Claude `PostToolUse`; a function, ACP or a graph without the harness's middleware waits as
    `"queue"`, said once — `Adapter.steers`);
  - one id per message (`conversation.message_id`): the AG-UI text message and
    `TOOL_CALL_RESULT` ids, and each message's `custom_metadata` in the memory service
    (`message_id`, `run_id`, `tool_call_id`), stable across a resume;
  - a Claude Agent SDK or ACP conversation continues its last turn's session in this process
    (`agent.sessions`).
  - `HarnessTools` (in `ReAct` and `middleware()`) has a `before_model` hook: one more
    LangGraph step per model call, so a `recursion_limit` set to the bare minimum needs one
    more.
- `h.conversation(user=, thread=)`: the key a user's conversation keeps its state under.
- `OpenAICompatible`, `Bifrost(OpenAICompatible)` and `ProviderSDK` routes, selected by
  `LLM_GATEWAY` from one registry that packages extend under the entry-point group
  `trellis.gateways`. Bifrost's features ask `gateway.has(Capability.X)`; new capabilities
  `SKILLS` and `ANTHROPIC_ROUTE` (the Claude Agent SDK's routed environment).
- `LLM_AUTH`: bifrost-sdk's auth strategies by name — `bearer`, `google` (a refreshed Google
  token), `aws` (SigV4) — on every model client, the judge and `doctor`'s probe.
- Optional extras `trellis-harness[anthropic]`, `[gemini]`, `[vertex]`, `[bedrock]` (the last
  two with bifrost-sdk's `[google]` and `[aws]` login strategies); `vertex:` and `bedrock:`
  model names for LangChain's `init_chat_model`.
- `trellis doctor` names the route, its auth and where model profiles come from, checks the
  declared route against the endpoint (a `bifrost` route must answer MCP), and refuses
  settings that fit no route with the reason.
- A conformance suite over every registered route, auth strategy and profile against stubs of
  OpenAI's chat completions, the Anthropic, Gemini, Ollama and Bifrost endpoints that speak it,
  a Google token refresh and an AWS SigV4 signature recomputed from the specification.

### Removed

- `automatic.ToolSelection`, `react.defaults_on` and the `benchmark.cost` variant
  `main+selector`: a harness fallback to LangChain's `LLMToolSelectorMiddleware` when memory
  gave no hints, which ADR 0003 §6 rules out (one owner of tool narrowing: memory's hints). A
  team adds `LLMToolSelectorMiddleware` through `ReAct(middleware=)` itself.

### Changed

- A model call's HTTP status is retryable exactly when it is in bifrost-sdk's `RETRYABLE`
  (408, 409, 425, 429, 500, 502, 503, 504), the set the gateway client and the memory service
  use; the harness's own `middleware.RETRYABLE_STATUS` (408, 409, 429 and every 5xx) is gone.
  425 is now retryable; 501, 505 and the other 5xx are not.

### Removed (one clearing pass, one read-back tool)

- `read_result`, `READ_RESULT`, `READ_LIMIT`, `FIT_CLEARED` and `FitWindow`'s clearing of older
  results: LangChain's `ClearToolUsesEdit` is the one clearing pass, and its placeholder names
  `read_file` with the cleared call's id as its path (`/large_tool_results/<call id>`, Deep
  Agents' own), paged from the graph's state with the limits every kept result uses (100 lines,
  20,000 characters). `FitWindow` cuts the longest results instead (each kept whole, its cut
  naming the `read_file` call that reads on). Before, `FitWindow` cleared results to a
  placeholder naming `read_result`, a tool that request did not offer.
- `FitWindow(turns=False)` (ReAct on Deep Agents: a summarization after it) leaves tool results
  to Deep Agents too: its overflow path offloads a result whole to its backend. Before, both cut
  the same result, so a sandbox held it twice (the harness's whole copy, written in for code,
  and Deep Agents' offload of the cut text).

## 0.6.0 — ReAct with Deep Agents on demand; `middleware()` (unreleased)

`ReAct` is LangChain's `create_agent` with the harness's part on top, and Deep Agents'
`create_deep_agent`, every Deep Agents feature native, when a Deep Agents capability is asked
for — and that part is exported: `trellis.middleware()` gives a `create_agent` or
`create_deep_agent` graph of your own what `ReAct` adds.

### Added

- `ReAct` hands wide tasks to helpers at once: the `delegate` tool, offered by itself to a
  request of several independent items of one kind on a large model (never to steps in order,
  a request to change something, a small model or a helper). Its jobs take Deep Agents' `task`
  fields (`description`, and `subagent_type` or `role`); each is a child run that only reads
  (a write is refused, never asked about), its answers merged for the model to check; a
  failed, timed-out, refused or skipped job is reported so; identical jobs run once; the
  budget is shared and stops the fan-out early; cancelling the run cancels them; with
  `RUNS_URL` the jobs queue in agent-runs for any worker. See
  [docs/frameworks/react.md](docs/frameworks/react.md).
- Child runs at once, adaptively, for every engine (`trellis.harness.admission`): a parent's
  sub-agent calls, nested `agent.run`s and `delegate` jobs are admitted under an AIMD limit (1
  to 16, from 4) that rises while the container's CPU and memory (cgroup limits), the
  provider's pace and the gateway's 429s stay healthy and halves when they do not; Kimi Code
  gets it as `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`. The live task tree: `helper_queued`,
  `helper_started`, `helper_finished` (status, seconds, tokens, cost) and `delegate` events
  on the run's stream, every framework's helpers included. See
  [docs/runs.md](docs/runs.md#child-runs-at-once-and-the-task-tree).
- A sub-agent's child run works at most its own time limit too, not only what is left of its
  parent's.
- Kimi Code as a target: `kimi(model=..., cwd=..., env=..., mcp_servers=..., mode=...,
  sandbox=..., skills=...)` drives Kimi Code's own agent (`kimi acp`, `@moonshot-ai/kimi-code`
  2.1.x) over the Agent Client Protocol, through the official SDK (`pip install
  'trellis-harness[kimi]'`). Moonshot archived the Python Kimi CLI and its Python agent SDK,
  which is why Kimi is driven through ACP. The adapter is a generic ACP client
  (`trellis.harness.adapters.acp`, `ACPAgent`); Kimi is the one agent configured, pinned and
  tested. Everything Kimi does on its own (its tools, sub-agents and swarms, plan mode,
  compaction) passes through: the harness's tools reach it as an MCP server, every call it asks
  permission for is governed (a person's approval a durable pause, resumed in Kimi's session),
  each sub-agent and swarm member is a child run, its usage log counts toward the budget
  (tokens; dollars are a follow-up), its model goes through Bifrost with the run's headers. The
  `kimi` program's tested range is checked at each run (`compat.BINARIES`); a version outside it
  fails the run. See [docs/frameworks/kimi.md](docs/frameworks/kimi.md), including what ACP
  does not show (calls Kimi does not ask about, a sub-agent's own calls) and the sandbox's
  limit (Kimi's reads stay on the host).
- `src.kimi()`: a skill set in Kimi Code's own skills folder, written per run.
- `tools.convert.mcp`: the harness's tools as an MCP server (the official SDK), served per
  attempt on the loopback interface with a bearer token.
- Requires `trellis-contracts>=0.6.2` (its `acp` error source).

### Removed

- `ReAct(tool_guidance=, answer_check=)` and `middleware(tool_guidance=, answer_check=)`, with
  the middleware they turned on, `ToolGuidance` and `AnswerCheck` (`TOOL_GUIDANCE`,
  `ANSWER_CHECK`, `WRITE_CHECK`, `NO_RECORD` among them), the benchmark's `main+guidance`
  and `main+check` variants, and the live tool-choice test's variants (it runs `ReAct` as
  built; `TRELLIS_LIVE_VARIANTS` is gone). Measured on Granite with memory off, fresh and learned, the
  answer check lowered tool coverage by about 11 points and added about two minutes a run; the
  tool guidance added about 50 seconds a run and, without memory, dropped correct answers from
  0.73 to 0.25 — neither with a significant gain. Each old argument is refused by name with a
  `TypeError` saying what to do instead: how to choose among tools goes in `system_prompt` and
  the tools' descriptions; answers are measured with `h.evaluate` (`tool_coverage`). A run its
  tool hints narrowed is no longer widened to every tool before its answer (only `tool_search`
  adds tools). `FitWindow(optional=)` and its `guidance` step, which only dropped the tool
  guidance, go too (refused the same way). See
  [docs/adr/0002-tool-choice.md](docs/adr/0002-tool-choice.md#amendment-2026-10-28-the-guards-removed).
- `ReAct`'s `delegate` tool (`trellis.harness.delegate`), with what only it used: the wide-task
  and model-tier routing that offered it (`automatic.Delegation`, `wide`, `model_tier`,
  `SMALL_PARAMS`, `SMALL_FAMILIES`), the read-only helper runs (`Runtime.delegated`, the
  `delegated` run metadata, the read-only toolbox and the bridge's `HelperOnlyReads` refusal),
  its agent-runs queueing of jobs, the `delegate` stream event and the `delegate_8` overhead
  case. Deep Agents has helpers natively: its `task` tool, offered as before (a sub-agent of
  its own, else the general-purpose helper on a long task); the model calls it several times in
  one step for helpers at once, and LangGraph runs those calls at once. Kept for every engine:
  the adaptive limit of child runs and the live task tree
  ([docs/runs.md](docs/runs.md#child-runs-at-once-and-the-task-tree)), Kimi Code's
  `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`, and a sub-agent's own time limit.

### Breaking

- `ReAct` takes Deep Agents' capabilities: given `backend=` (file tools, large-result eviction,
  summarization; `sandbox()` for `execute`), a `SubAgent`/`CompiledSubAgent` sub-agent
  (`task`), `skills=`, `memory_files=` or `permissions=`, it is `create_deep_agent` (prompt
  caching, its long-task scratch files and helper); otherwise `create_agent` with
  `middleware()` and dangling-call repair — the base measured as good or better on every class
  of task (see Changed). The new signature, names following Deep Agents': `ReAct(system_prompt,
  model, *, tools, subagents, skills, memory_files, permissions, backend, response_format,
  state_schema, context_schema, cache, max_steps=12, max_repeats=3, call_timeout,
  context_window, effort, tool_guidance=None, answer_check=None, middleware)`
  (`memory_files` is Deep Agents' `memory`; its `interrupt_on`, `name` and `debug` are not
  exposed).
- `ReAct(tools=)` refuses `sandbox()` (code runs in `backend=sandbox()`) and agents
  (`a2a(url)` and wrapped agents go in `subagents=`); `max_repeats` is 2 or more, as
  `StallGuard` always required.
- `tool_guidance` and `answer_check` default to `None`, automatic — which, measured on a small
  model, keeps both off (see Changed); `True` turns either on, `False` keeps it off.
- One name per concern (the API audit of 2026-10-27). No alias is kept: each old usage fails
  with a `TypeError` (`AttributeError` for a method) naming what to use instead.
  - `h.evaluate(agent, dataset, judges, ...)` and `evaluate(target, dataset, judges, ...)`: the
    third argument is `judges` (was `evaluators`), the word of `h.wrap(judges=)`,
    `without={"judges"}` and `judge()`. Positional calls are unchanged.
  - `h.wrap(judges=[grounding()])` is refused: every wrapped run already gets the grounding
    check on a sampled share, so it would check the same runs twice.
  - `ReAct(checkpointer=, store=)` removed: a `ReAct` checkpoints into its run and its store is
    the memory service; a graph with its own is `create_agent(..., middleware=middleware())`.
  - `ReAct(capabilities=)`, `middleware(capabilities=)` and `ToolGuidance(capabilities=)`
    removed (and `middleware.CAPABILITIES`): what the assistant can and cannot do goes in
    `system_prompt`. `ToolGuidance(text)` takes a text only.
  - `h.governance_middleware()` removed: a wrapped graph's own tools are governed already, and
    `middleware()` has the same `GovernTools`.
  - `ReAct(middleware=middleware())` is refused saying that `ReAct` already has it; the refusal
    of a middleware named as the harness's says "is the harness's" (ADR 0003 point 4 says so:
    your middleware is added, one named as the harness's — `TodoListMiddleware` among them — is
    refused).
  - `serve_chat` without `identity=` reads the trusted `X-Trellis-Identity` header, as
    `serve_a2a` does (its tenant, when named, the key's): a request without it is `401`, never
    `anonymous`. `identity=` stays the override. `IdentityRefused` and the header's parsing
    (`trusted_user`) moved to `trellis.harness.identity`, shared by both surfaces.
  - `serve_a2a` keeps the same rule: a JSON-RPC call without the trusted header, or with one
    refused, is `401` (a problem document) before the SDK reads it, never run as `anonymous`
    (`trellis.harness.a2a.identity.ANONYMOUS` removed); the card stays public, and
    `identity=` stays the override. The harness's own `a2a(url)` calls send the header.
  - `h.feedback` takes a person's verdict only — `confirm`, `reject`, `correct` — and refuses
    `approve` and `edit`, the system's record of a decision on a paused action
    (`agent.resume`). `VERDICT_SCORES` lost their entries.
- `ReAct(middleware=)` with a Deep Agents middleware's name replaces Deep Agents' own (as
  `create_deep_agent` does); one named as the harness's is still refused.
- One sandbox entry: `sandbox()`, given where each engine takes it. No alias is kept.
  - `docker_sandbox()` removed (from `trellis`, where reading the name says what to use, and
    from `trellis.harness.sandbox.backend`):
    `ReAct(backend=sandbox())` takes the sandbox itself, and a `create_deep_agent` of your own
    takes `**sandbox().deep_agent()` (`{"backend": SandboxBackend(...)}`).
  - `sandbox()` is no longer a tool source itself: its three tools are `sandbox().tools()`
    (`agent.tools.use(sandbox().tools())`, the same object every time), for an engine with no
    sandbox of its own (a function, a LangGraph graph). A bare `sandbox()` given as a tool is
    refused, naming `.tools()`. Same per-run sandbox as before: made at the first call,
    journaled, paused, deleted at the run's end, reaped.
- The model hooks are removed: a model call is each framework's own (LangChain and Deep Agents
  middleware — `ReAct(middleware=[...])` —, the OpenAI Agents SDK's `AgentHooks`, the Claude
  Agent SDK's hooks), which run under `h.wrap` as without it.
  - `Hooks.before_model` and `Hooks.after_model` removed, with `ModelCall` (`trellis.ModelCall`
    says what to use). A hook class that still defines either is refused where it is given
    (`h.wrap(hooks=)`, `governed(hooks=)`): `ConfigurationError` naming the framework's own
    slot, never silently skipped. `Hooks` keeps `on_run_start`, `on_run_end`, `on_error`,
    `before_tool` (`Deny`/`Ask`/`Rewrite`), `after_tool` and `on_output`.
  - `middleware.ModelHooks(*hooks)` and `hooks.openai_agents.ModelHooks(*hooks)` take no hooks
    of yours: what they do for the harness stays — the `chat` span, `call_timeout`, the pinned
    stored prompt and a failed call's `on_error("model", ...)`; the OpenAI run's stall watch,
    hosted tool calls and budget accounting (nothing internal relied on the public hooks).
  - The feature matrix's `F71m` row (model hooks) is gone.
- Removed, Deep Agents now owning them: `automatic.JournalFiles`, `automatic.FilesState`,
  `automatic.ScratchFiles` (`automatic.LongTaskTools` offers Deep Agents' own scratch tools and
  general-purpose helper on a long task); `ModelHooks(routed=)` (it sees a routed model itself).

### Added

- Skill sources (`trellis.skills`): skills kept in a store and edited at any time, per tenant,
  workspace, user and agent, fed to each framework's own skills slot.
  - Sources: `skills.directory(path)` (`SKILL.md` folders; an allow-list `tenants:`/`users:`),
    `skills.bifrost()` (the gateway's skills repository), `skills.langfuse()` (prompts tagged
    `skill`), a store of your own (the `SkillSource` protocol: `list`, `load`, `read_file`,
    optional `changed_since`), `skills.combine(...)`. A name kept at a more specific level wins
    (agent, user, workspace, tenant, the platform-wide sources), then the source given first; a
    disabled winner hides it.
  - Slots: `ReAct(skills=src)` and `create_deep_agent(**src.deep_agent())` (Deep Agents' own
    `SkillsMiddleware`, the skills a `/skills/` route of the backend); `trellis.deep_agent(
    sandbox(), src)` merges a sandbox and skills into its one `backend=` (a `CompositeBackend`);
    `ClaudeAgentOptions(**src.claude())` (a local plugin written per run, removed after;
    `await src.claude_dir(...)` without a harness); `src.openai()` (the OpenAI Agents SDK's
    `Skills` capability of a `SandboxAgent`); `src.tools()` (`list_skills`, `load_skill`) where
    an engine has no slot — a function, LangGraph, a plain OpenAI `Agent`.
  - Per run: the run's scope; the versions pinned in its journal (`repository.journaled`, the
    stored prompts' pin), so a resume uses them and an edit applies from the next run; a deleted
    skill and a source that is down fail soft (`skills_unavailable`; the last listing stands);
    a description, a body and a file capped, a bad name or a path out of its skill skipped.
  - `repository` has again what every versioned source shares (`Kept`, `front_matter`,
    `digest`, `inside`), and `telemetry.Langfuse` reads prompts (`prompts`, `prompt`).
- Each framework page says where the sandbox, skills and hooks go on that engine, and that
  `budget=` is the one cross-engine token, dollar and tool-call cap while a framework's own
  turn limit (`recursion_limit`, `max_turns`) goes in `framework_options`.
- Chat attachments: on `serve_chat`, the user message's file parts — AG-UI 1.0 media parts
  (`document`, `image`, `audio`, `video`) whose `source` is `{"type": "file", "value":
  "<document id>"}`, a document already uploaded to the memory service — are the run's
  `document_ids`, as `agent.run(document_ids=)` takes them. Each is read first as the caller
  (`RunMemory.document`, `GET /v1/documents/{id}`), so the memory service's own read check
  decides: one it refuses or does not know is a `404` and nothing runs. Inline (`data`) and URL
  sources are a `422`: the surface uploads and fetches nothing. A text-only message runs as
  before. A2A file parts are not attachments yet.
- `trellis.middleware(*, max_steps, max_repeats, call_timeout, context_window, tool_guidance,
  answer_check, system_prompt)`: the harness's part of `ReAct` — the run's tools
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
- `FitWindow` (in `middleware()`, so `ReAct`'s too): a run never fails because a request does
  not fit the window. Over it, the request is made to fit step by step, only as far as needed —
  the optional guidance, the tools narrowed (the used, the hinted, the relevant kept), their
  descriptions compacted, older results cleared, older turns, and last the memory context's
  lowest-ranked items (never a rule, the standing instructions or the learned plan) — each step
  said on the run (`context_fit`); a request that fits is sent unchanged. A request the
  model's server still refuses as too large (its tokenizer counts more than the estimate) is
  fitted again by the server's count and sent once more; the model's tokens per counted token
  are learned from what its replies say their requests were, so a request that fits by the
  model's own count is sent as it is, and a run's narrowed tools stay the same call after call
  while they fit. Only a user message
  that alone cannot fit fails: `ContextTooLarge` (`CONTEXT_TOO_LARGE`) naming the window and the
  size.
- With tool hints, each call is also offered the tools the memory context names (a learned
  skill's steps), beside the hinted, the used, the memory tools and what `tool_search` found.
- `automatic.ToolSelection`: LangChain's `LLMToolSelectorMiddleware` used only with no tool hints
  and over 20 tools — measured (`benchmark.cost` variant `main+selector`), not a default.
- `LIVE_SHARD=i/n` (`make live-shard`): the live suite in shards packed by the measured
  durations (`tests/live/durations.json`, `make live-durations`), each under a job's limit on a
  CPU model.
- Python 3.13 beside 3.12: CI's gates (lint, types, tests, examples, docs, the matrix, the
  overhead benchmark) and the weekly canary run on both (`python-version: ["3.12", "3.13"]`);
  the 3.13 classifier.
- `benchmark.cost` variants `deep` (`ReAct` on Deep Agents) and `deep-<piece>`, `pieces`,
  `main+guidance`, `main+check`, `main+selector`, `main@16k`; `--suite classes` (ten tasks per
  class) and `--memory`; `TRELLIS_LIVE_VARIANTS` picks the live tool-choice test's variants.

### Changed

- A run's tools are offered in the order first offered (the first call's by name, those offered
  later after them, a tool offered again in its old place), not re-sorted by name each call: the
  set only grows at its end, so a provider's prompt cache (whose prefix the tools open) is broken
  once per growth, not reordered.
- Deep Agents' summarization stands after the harness's middleware that narrow the tools, so
  its budget counts what the model is sent.
- The base was chosen by one Granite batch, each base with `middleware()` and `FitWindow`, ten
  different tasks per class (simple, multi-step, wide, a ~100k-character result, 50 and 100
  tools, 50 tools with memory on): `create_agent` equalled the best on simple tasks and was as
  good or better on every other class — on the large result it completed 10/10 runs where
  `create_deep_agent` completed 5/10 and Deep Agents' pieces on `create_agent` 6/10 —
  [docs/frameworks/react.md](docs/frameworks/react.md#the-base). The guidance and the check are
  off by default: the guidance raised coverage (0.89 to 1.00) but made the model call changes
  on 4 of 16 items that asked for none, the check added a model call per answer for no gain,
  and both broke the context's rule
  ([docs/frameworks/react.md](docs/frameworks/react.md#tool-choice)).

### Fixed
- Saving and resuming runs, for long runs and runs whose worker crashed
  ([docs/runs.md](docs/runs.md#continuing-where-a-run-stopped)):
  - A `create_agent` or `create_deep_agent` graph built with `middleware()` and compiled
    without a checkpointer kept no checkpoint: a resume or a crash ran it again from its input,
    the model asked again for every step, and a write the model planned anew (other arguments)
    ran again — seen with a worker process killed after a write. `h.wrap` now runs it as a copy
    with `RunCheckpointer()`: it continues where it stopped. A checkpointer of its own, or
    `checkpointer=False`, stays the graph's.
  - After a pause or a crash, a checkpointed graph's later call identical to an earlier step's
    (the same tool, the same arguments) replayed the earlier step's record: it was not run, and
    an approval it needed was taken from the earlier call's. The journal now keeps each call's
    record, and each answer a call read, with the step it was made in (`Journal.steps`); an
    attempt continuing from a checkpoint passes over the steps before it.
  - A long run's journal grew with every call: a `ReAct` or `middleware()` run drops a step's
    records once its checkpoint moves past it, and every run's trajectory keeps each output as
    the event stream previews it (2000 characters).
  - A graph with a durable checkpointer of its own, after a crash, started the run's turn again
    on top of the unfinished one (the question twice, a tool call left unanswered): the run's
    checkpoints are marked with its id (`trellis_run_id` metadata) and the next attempt
    continues from the latest.
  - A graph checkpoint in the journal that cannot be read back ends the run `ERROR` with
    `CHECKPOINT_UNREADABLE`, not retried and nothing run again
    (`middleware.CheckpointUnreadable`).
- A tool call a model step repeats — the same tool with the same arguments, in any key order
  (seen live: a small model on `ReAct` emitting `style_stock(style='ZZ999999')` 14 times in one
  message) — ran every time: 14 calls, 14 results back to the model, and for a write 14 side
  effects. It now runs once: each repeat runs nothing (no hooks, governance, approval, budget
  charge or journal entry) and the model reads, under the repeat's own call id, that its result
  is the first call's (`Same call as <id> above ...`); its `TOOL_CALL_RESULT` names that call
  (`same_as`). One owner, `tools.bridge` (`firsts`, `same`), fed by whoever reads the step: the
  LangGraph tool node's calls (`HarnessTools`, and `governed_call` for every tool node: `ReAct`,
  `middleware()`, `create_agent`, Deep Agents, a graph's own `ToolNode`) and an OpenAI Agents
  reply (the harness's `RunHooks`; its tools and the agent's own governed `FunctionTool`s).
  Claude and Kimi run a step's calls themselves over MCP, which carries no call id: unchanged.
  The same call in a later step runs again (`StallGuard`'s `max_repeats`); a step whose only
  run call failed still counts toward the error streak. Matrix row `F25d`.
- The tool-choice scenario's answer checks (`examples/06_scenarios/tool_choice.py`) no longer
  fail a right answer on its wording: `plain_answer` reads "no" from the same `CANNOT` words
  as `says_cannot` ("I don't have the capability to …"), and `says_no_record` also reads
  "does not have data", "doesn't exist" and "no information".

- AG-UI: a resume sent right after its pause could end with `RUN_ERROR` ("the run ended
  without a result"): the paused attempt, still finishing (its end hooks, its records) after
  announcing the pause, found the run's buffer reopened by the resume and said the run ended.
  Each attempt now answers for its own end only, and a stream closes when its attempt returned
  (after `RUN_FINISHED`, once the turn is written to memory and the end hooks ran): the next
  message's context holds the turn, and a resume finds the paused attempt done.
- A governed framework tool call inside a sub-agent whose call id equals its parent's `task`
  call id (ids are unique per conversation only) was let through ungoverned; the marker is now
  the tool's name and the call's id.
- A tenant's first runs were offered the toolbox's alphabetically first tools: the hints were
  asked before the tools just published were indexed, every hint a zero. A worker now publishes
  its agents' tools before its first claim, a toolbox's first hints wait (at most 5 s) until
  the catalog says tool search finds them, and hints whose best confidence is under 0.5 narrow
  nothing (every tool offered, with `tool_search`).
- Under Code Mode the narrowing dropped the meta-tools, cutting a run off from every tool behind
  them. The hints are asked about those tools too, and when one is hinted the meta-tools stay.
- The cost benchmark's `picks` check scored the first store an answer named, so a right
  answer that gave the other store's figures first ("S12 has 3 weeks, S40 0.2: S40 first")
  failed. It scores the store the answer concludes on. Benchmark only.
- A Code Mode script's nested calls were recorded after the run's own calls, numbered by the
  memory service in the order the log was read: a tool the run called after the script came
  before them in a learned procedure. They are now recorded in the order the gateway logged
  them, each at the step of the script that made it (the last one started by then). A script
  calling one tool twice with the same arguments is recorded once, as any call repeated at one
  step is.

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
