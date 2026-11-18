# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## 0.7.0 — any provider: one route strategy, an auth strategy, a model's profile (unreleased)

### Changed: an OpenAI `SandboxAgent`'s shell runs unasked, as every framework's shell does

A `SandboxAgent`'s `exec_command` and `write_stdin` asked a person as any write, while Claude
Code's and Kimi Code's `Bash`, Deep Agents' `execute` and the harness's `sandbox_exec` run in
the run's sandbox unasked and announced. On the skills benchmark the OpenAI agent's chats
waited on a person to `cat` a skill file (5 of 30 cases right). They now run as the others do:
announced, recorded, unasked — a command that reads and one that writes alike; on a host
`TRELLIS_HOST_EXECUTION=deny` still refuses the SDK's local sandbox. A risk the catalog declares
still asks as it says, for this shell and for every tool announced this way (Deep Agents'
`execute`, `sandbox_exec`, a framework's own write): `write` once a conversation, `irreversible`
at every call (held at the SDK's own gate, its approved call continuing the paused run), and so
does an `approve_when`. Only the SDK's own `Shell` capability's tools are the sandbox shell, and
the mark survives the harness's copies of the tool. `apply_patch` asks as a write, as `Write`
does.
### Fixed: a run a dead worker left keeps what that worker's model calls spent

A queued run whose worker died and another finished reported, in `RUN_FINISHED`'s `spent`
(and so in its record), only what its journal held at its last progress save: the model
replies the dead worker was charged for after it — those followed by calls that only read,
saved at most every 20 s — were lost (the accuracy benchmark's restart case, `long-flow-04`:
32,978 tokens reported against 40,723 metered). Each reply accounted on a worker's run now
saves what the run has spent at once, on a heartbeat: one small segment of the spending after
the journal's segments, which the next spending save replaces (`Journal.save_spending`) — a
checkpoint of segments alone, as agent-runs keeps one, and as a worker of an earlier release
reads it. One save under way at a time; a call that only reads waits, before it runs, for the
one save that carries the replies accounted before it (not for later ones, and not at all
when a save already carried them). The attempt's pause, end or hand-back is written only after
the spending save under way is answered, and none starts after it, so a late heartbeat cannot
put back an earlier attempt's checkpoint; a failed spending save is a warning once per run, and
one that finds the lease lost stops the run's next model call or read. Each save uploads one
small segment; one a later save replaced is deleted by agent-runs' payload sweep 24 h after it
was written. A
model call still under way when the worker died is not counted (its reply never came).
### Added: `h.harness_tools()`: the harness's own tools for a graph you build

A graph whose model binds its tools when it is built (a hand-built `StateGraph`, a
`create_agent` graph without `middleware()`) never had `ask_user` and had the schedule tools
only by asking for them (`h.schedule_tools()`): in the accuracy run such a graph asked in its
reply and never made a schedule. `await h.harness_tools()` gives it, as LangChain tools for its
`ToolNode` and `model.bind_tools`, what every other target is offered by itself: the MCP tools
the virtual key allows, the memory tools, the schedule tools and `ask_user` — the same tools,
not copies. Each call is the run's that makes it: governed, journaled and recorded;
`ask_user` pauses the run and its answer is the call's result; a part the run is without
answers that it is off in this run; outside a run a call is refused. A schedule tool called in
a run a schedule fired now answers that it is not for that run (every other target is offered
none there). A `create_agent` graph is offered them in the run's words at each model call
(`schedule_task` saying the hour in the user's zone, `ask_user` naming the app's own controls).
The benchmark's `graph` adapter binds them as the docs say: asking 59 to 71 of 80, schedules
20 to 57 of 61, skills 28 to 29 of 30
([frameworks/langgraph.md](docs/frameworks/langgraph.md#the-harnesss-tools-on-a-graph-you-build)).
### Fixed: a sub-agent's question is no longer kept as the person's own words

A child run's question (a `subagents=` call, a nested `run()`, a framework's own sub-agent or
async task) is what the agent above it wrote. It was written to the memory service as the
person's message, in their conversation: the service formed their memories from it (a planner's
"Research only; never reorder" became the user's standing rule, steering every later run of
theirs) and the conversation's window replayed it as a turn of theirs. It is now kept as an
agent's message (the service's `AGENT` role, `Said.delegated`): in the transcript, never the
person's words, and no window replays it. The person's own question stays theirs.

### Fixed: a harness served from another thread closes from the main one

A harness whose agents serve AG-UI or A2A from uvicorn in a thread does its background
writes on that thread's event loop. `h.aclose()` from the main loop then failed with a
`RuntimeError` ("attached to a different loop"), and when uvicorn was told to exit first,
with that loop's `CancelledError`. The writes are now drained and their workers stopped on
their own loop while it runs; a loop that has ended stopped them already. Only a close that
is itself cancelled says so (`examples/05_features/serve_agui_and_a2a.py`,
`examples/02_way1_react/subagents.py` against the real services).

### Fixed: a Kimi that never reached the harness's tools no longer answers without them

In the accuracy run, a Kimi Code session now and then opened without its connection to the
harness's MCP server: its model was offered none of the run's tools, searched the machine with
its own shell, and answered wrong as if the data did not exist. The harness's server now says
when Kimi reached it (`Served.reached`), and an agent that connects to its MCP servers as a
session opens (`Profile.connects`, Kimi's) has 10 seconds after that to have done so
(`CONNECT_SECONDS`). One that did not is started again, said on the run as an `acp_tools`
warning; when the next one does not connect either the run fails saying why
(`CONNECT_TRIES`). An ACP agent that connects only at its first call is not held to it.

### Fixed: a run's end no longer scans every run the in-process store keeps

Ending a run asks the store for its open children, once for each open status. `LocalRuns`
answered by scanning every run it keeps (up to `keep_finished`), so each run paid two full
scans at its end, and more the longer a process ran. It now keeps each parent's children in
an index of their own: `iterate(parent_run_id=)` reads only those, and a child the store
forgets leaves the index. On the overhead benchmark this branch is again level with or faster
than the release before it (a run with one tool call about 8% faster, a stream about 20%, a
run that starts a child about 15%).

### Fixed: Claude Agent SDK plan mode holds until a person approves the plan

A run with `permission_mode="plan"` (the options', or a run's `framework_options`) looked
things up, then closed an account and notified the customer without ever presenting a plan.
The harness's `PreToolUse` hook answers `allow` for every harness tool (the bridge decides
it), and the hook's word is final for the CLI: its own plan mode never saw those calls.
The hook now holds plan mode itself: until a person approves the plan Claude presents
(`ExitPlanMode`) only a call that reads runs (governance's
risk, as `PlanMode` for `ReAct`; a shell command that only reads) or Claude's write of its
plan in its plans folder. Any other — a harness tool that writes, `Write`, `Edit`, a command
that changes something, a hand-over to another harness agent — is refused, Claude reading
why, and recorded as a rejected call. The approval is kept in the run's journal
(`Journal.plan_approved`), so a resume on any worker goes on governed as ever. A run that
set plan mode and relied on its writes running sees them refused until the plan is approved.

Checked against the real CLI with Haiku, three more holes are closed. The CLI presents its
plan unasked when an SDK drives it, so the hook asks the person itself — always, whatever the
tool catalog says of `ExitPlanMode` (it only reads, so the catalog's word on it would have let
every later plan through). In plan mode the CLI asks about every harness tool after the hook
allowed it; `can_use_tool` now lets a harness tool through (the hook decided it, the bridge
governs its call), so a lookup no longer waits for a person. Resumed, the CLI presents the
plan again, worded again and under another id: the plans a run presented are kept in its
journal (`Journal.plans`), and a person's approval of any of them ends plan mode.

### Fixed: the same AG-UI `runId` sent twice at once opens one run

`POST /agui/run` claimed a caller's `runId` only after asking the run store whether it was
new, so two requests with the same `runId` arriving together both passed the check; against
agent-runs, which answers a second start of a `run_id` with the run, both were answered 200
and the run was attempted twice. The `runId` is now claimed before the store is asked: the
second request is a 422.

### Changed: the in-process run store follows the contracts' rules (`LocalRuns`)

Built on trellis-contracts' unreleased `main` (0.8.0 to be) and the agent-runs SDK beside it;
each rule below was a copy in `runs.py` and is now read from the contracts.

* **A schedule fired in process queues the run agent-runs queues** (`Schedule.run_start`).
  Three things change on such a run's record: `user_id` is `None`, where it was
  `on_behalf_of` (the run still acts for `on_behalf_of`: `Runtime.user`, its memory and its
  files are that person's, as on agent-runs); it carries the fire's `idempotency_key`
  (`<schedule_id>@<fire_time in UTC>`, the key `FireResult.idempotency_key` already gave);
  and its `metadata` gains `schedule_name`, `fire_time` (UTC) and `created_by` beside
  `schedule_id`. Code that read `record.user_id` of a scheduled run reads
  `record.on_behalf_of`.
* **A run moves as `RunRecord.moved` moves it**, the rule agent-runs follows. A status never
  becomes itself: a repeated `finish` to the status the run already ended in is a
  `ConflictError` (`a SUCCESS run cannot become SUCCESS`) and changes nothing, where
  `LocalRuns` took it and wrote the second output over the first. The pipeline already reads
  such a refusal back and goes on with the run as recorded. A checkpoint an attempt saves (a
  heartbeat's, a release's) is kept without a move. The refusal's message is the contracts'
  (`run <id>: a <STATUS> run cannot become <STATUS>`).
* `is_pause_signal` is imported from `trellis.contracts`.
* Nothing to change for the dependencies' other changes: the harness emits an `error` only
  on `RUN_ERROR` and a failed `RUN_FINISHED`, its deadlines are timezone-aware, and it reads
  no stored record itself (the SDK reads agent-runs' with `RunRecord.read` and
  `Schedule.read`). `agent.resume(..., remember="always")` on an interrupt with no tool call
  is refused (`ConfigurationError`: `only an approval of a tool call is remembered beyond its
  call`), as `remember="run"` was, by the SDK's `answer_problem`.

### Fixed: a run's children end with it, each on its own

* **Children:** every place a run ends — answered, failed, out of time, cancelled, its hold
  lost — ends its children through one cascade (`children.end_children`). A child whose
  ending fails (the run store's error, a Deep Agents server that fails its cancel) is said in
  the log and no longer leaves the run's other children working; the first such error is
  raised once every child was tried; a cancel (`agent.cancel`, a cancelled resume) and a run's
  own end only log it. A child the store ended first, or another worker holds, is left to
  whoever ended or holds it, its own children too. What this process keeps for a child (its
  background task, its live session's background subagent, its Deep Agents task) stops
  whether or not the store took the child's end, and through the keepers of the child's own
  agent; what the adapter keeps for a run stops even when the store cannot list its children.
* **A paused child of a failed run** ends: cancelled with the parent's error message (a paused
  run cannot fail), where it was left waiting for an answer that went nowhere. A child that
  fails with its parent's error is never put back on the queue for a retry.
* **A run whose caller went away** (a closed stream) leaves the children it queued for a
  worker to work; only a cancel ends those. A run's end lists only its open children.
* **Sub-agents:** an ACP agent wrapped as a tool (`tools=[h.wrap(ACPAgent(...))]`) is `write`
  unless `side_effects=` says otherwise: it runs its own tools in its own process. It was
  `read` when the harness tools it declared only read. An adapter says whether its framework
  has tools of its own (`Adapter.own_tools`); one that does not say is taken to have them.
* **Cancel:** a run stopped for a reason (`agent.cancel(reason=)`, a newer message) ends with
  that reason on its `RUN_FINISHED` event when its framework's session then reports a cancel
  of its own; the session's wording replaced it.

### Changed: one owner each for stopping, children and helpers (no behaviour change)

* Stopping an attempt and reading "the run store ended it" live in `stopping.py`; the
  adapters' six copies of opening and closing an attempt's helpers are
  `bridges.helpers.helping()`; what an adapter may leave undefined has its default in
  `adapters.base.Keeps`.
* The unit suite holds three budgets at what the source has today: the highest and mean
  function complexity, duplicate blocks, and packages that import each other
  (`tests/unit/test_complexity_budget.py`, `test_duplication_budget.py`,
  `test_import_budget.py`).

### Fixed: security audit — scope, approvals and what reaches the service's own files

* **AG-UI:** the data a paused run's question carries (`GET .../runs/{run_id}/artifacts/{id}`)
  is its run's user's only, as the run's events and files are; a client `runId` the run store
  already has (another replica's run, one this replica's buffers forgot) is refused (`422`),
  never attempted again under the caller's name.
* **Approvals:** a question a rule assigned to someone else (a `before_tool` hook's
  `Ask(assignee="role:finance")`) is theirs, answered through the inbox (`agent.resume`): the
  run's own user answering it over AG-UI is `409`, over A2A the task keeps waiting.
* **Claude Code:** a run's own `framework_options` cannot set the CLI's flags (`extra_args`) or
  its `settings` (they carry its permission mode and rules: the deployment's, as
  `permission_mode` and `allowed_tools` are); a run's own `can_use_tool` may refuse a call,
  never approve it for the user.
* **Files:** a link in a framework's `outputs/` is never followed into the file store (it could
  lead to any file the service may read); an Excel file is read into at most
  `stored.MAX_CELLS` cells (a few kilobytes could say millions).
* **Kimi Code:** a sandboxed run's Kimi is configured to ask about `FetchURL`, `WebSearch` and
  `Skill` as any run's is (it fetched from this machine's network unasked); `Glob`, `Grep` and
  `ReadMediaFile`, which read this machine's files in Kimi's own process, are asked about, and
  one outside the run's folders is refused.

### Changed: shell commands on the host — isolated where the harness runs them, one switch to turn them off (`TRELLIS_HOST_EXECUTION`)

Shell commands stay allowed by default (`TRELLIS_HOST_EXECUTION=allow`), with isolation as the
guard where the harness runs them:

* **`sandbox()` with no provider** works in the tenant's shared Docker container when the
  daemon answers, else in this machine's isolation layer (`HostSandbox`: a per-tenant base, a
  per-run uid and 0700 workspace, limits, a scrubbed environment, the network off, audit) when
  it can isolate here, else it has no shell: each command is refused with the reason, said once
  (`NoShell`). The model sees the same command tool either way. A deployment's own layer is
  installed with `trellis.harness.host.install(HostSandbox(...))`, refused at start when it
  cannot isolate.
* **Breaking: Deep Agents' `execute` on a `LocalShellBackend` is refused at `h.wrap`**
  (alone, a `CompositeBackend`'s default, a sub-agent's; `ReAct(backend=)` too), whatever the
  setting, naming `sandbox()`. Before, it ran on the host as the service's user.
* **The frameworks' own shells run as before** — Claude Code's and Kimi Code's, an OpenAI local
  shell tool or Unix-local sandbox, LangChain's host `ShellToolMiddleware` — as the service's
  user, without
  the harness's isolation; a team's own Claude `SandboxSettings` is honoured as given (on Linux
  it needs bubblewrap and socat: without them the CLI runs the command unsandboxed, warning on
  its stderr). `kimi(sandbox=sandbox())` and Claude's own sandbox are the ways to isolate them.
* **`TRELLIS_HOST_EXECUTION=deny`**, the administrator's switch, turns every shell outside a
  sandbox off: Claude Code's (`Bash`, `PowerShell`, `REPL`, `Monitor`) and Kimi's `Bash` on the
  host, an OpenAI local shell tool or Unix-local sandbox and LangChain's host shell are refused at `h.wrap` (and for a run whose own
  `framework_options` would make one), naming the setting; the isolation layer's own commands
  are refused at the call. Commands in a sandbox are not affected. `TRELLIS_REQUIRE_SANDBOX` is
  gone (`Settings(require_sandbox=)` is refused, naming `host_execution`)
  ([governance.md](docs/governance.md#commands-on-the-host)).

**Claude Code and Kimi Code, tool by tool** (breaking where noted). Reads in the workspace run
unasked and writes stay the user's under the one rule, as before. **The shell in the sandbox
runs unasked**, announced and recorded: Claude Code's and Kimi's own permission question about
a command is no longer the user's (on the host the same, unless the deployment says `deny`).
**The network out is closed by default** — Claude's `WebFetch`, `WebSearch`, Kimi's
`FetchURL`, `WebSearch`: refused, and the model told why, until the tool catalog gives the tool
a risk or an `approve_when` (before, Claude's asked the user and Kimi's ran unasked); the
harness publishes them with no risk of its own, and a run's Kimi is configured to ask about
them so the catalog decides. **MCP servers, Claude's `allowed_tools` and a permission mode
that lets calls through** (`acceptEdits`, `bypassPermissions`, `dontAsk`, `auto`) are the
deployment's: a run's own `framework_options` that change them are refused, naming
`h.wrap(framework_options=)` ([governance.md](docs/governance.md#who-decides-what)).

**Who decides what** is now said in one place: the end user only the actions taken in their
name; infrastructure — shells on the host, a tool's risk, who answers an infrastructure tool's approval
(`before_tool`'s `Ask(assignee="role:…")`, the catalog's `risk` and `approve_when`) — the
administrator's, always audited ([governance.md](docs/governance.md#who-decides-what)).

### Changed: commands run in the tenant's shared sandbox, with a workspace and a uid per run

`sandbox()` now works in **one container per tenant** (`SharedDockerSandbox`), never shared
across tenants, instead of a container per run. Inside it one isolation layer
(`trellis.harness.sandbox.isolation`) gives each run a workspace (mode 0700) and an
unprivileged uid of its own; each command runs in process and mount namespaces of its own (it
sees and can signal only its own processes), with every filesystem but its workspace read-only,
within per-command limits (`Limits`: CPU time, memory, processes, file size, open files, and
the call's timeout; the limit hit is in the result, `limit`), with an environment of its own
(no secret of the service's) and the container's network (none by default). A run's end — any
way it ends, a crash's takeover included — delivers its `outputs/`, then kills its processes,
deletes its workspace and frees its uid; the reaper does so for runs that died. Every command is
audited (`sandbox_command` on the run's stream, the `trellis.sandbox.audit` log). The same
layer runs commands on the host (allowed by default; `TRELLIS_HOST_EXECUTION=deny` switches them off) (`HostSandbox`,
`isolated_run(tenant, user, run, command, limits)`), and refuses to when the service cannot
switch uid. `sandbox(DockerSandbox())` keeps a container per run. `ExecResult` gains
`duration_ms` and `limit` (dumped only when set). See [docs/sandbox.md](docs/sandbox.md#the-shared-sandbox-one-per-tenant-isolated-per-run).

### Added: every step of every run in Langfuse

* **A framework's own tool calls are `execute_tool` spans,** with a harness tool's attributes
  and governance's action (`ask` for a call a person approved): Claude Code's built-ins and MCP
  tools, Kimi Code's tools, Deep Agents' file tools (on its state backend too), `execute` and
  `task`, OpenAI Agents' function and hosted tools, a LangChain tool in a graph of your own.
* **Every shell command is a span** — the harness's `sandbox_exec`, Deep Agents' `execute`,
  Claude Code's `Bash`/`Monitor`, Kimi's `Bash`, OpenAI Agents' shell tools: the command
  (redacted), where it ran (sandbox, host isolation, the framework's own), its exit code,
  duration and the limit that stopped it, from the command's audit record where the harness
  ran it (the isolation layer's `sandbox_command` record is the span's).
* **A sub-agent its framework spawns is an agent span** (`invoke_agent <helper>`) its calls
  and replies nest under; **a framework's own model replies are `chat` generations** with
  their usage, cache tokens and cost (Claude, Kimi, OpenAI Agents); a `ReAct` generation
  carries the cost the gateway reported (`gen_ai.usage.cost`).
* **An approval is an `interrupt` event** (who is asked, the question, the call) **and its
  answer a `decision` event** (the choice: once, conversation, always, deny); a resumed
  attempt says so (`trellis.resumed`).
* **The attempt's span says how it ended** — `trellis.run.status`, the spend (and, on the
  run's trace, `langfuse.trace.metadata.status` and `spent_*`), a failure's status and
  message, the step limit, a budget, a time-out or a cancellation with its reason (a level, a
  status message and an event) — and **a pause is no longer an error span**.
* **Redaction:** an error raised out of a span was recorded with its message as it was (the
  OTel SDK's `record_exception`), and a credential inside longer text (a command line, an
  error message) was kept; both are masked now. [observability.md](docs/observability.md#what-reaches-langfuse-by-adapter)
  lists what reaches Langfuse by adapter, and what goes elsewhere.

### Fixed: what the framework comparison of 2026-11-15 found

* **Claude Agent SDK: a model call for each reply.** A run's `model_calls` counted one per
  query result (1.07 a turn on Haiku, against the provider's 3.00): each model response the
  CLI reports — an assistant message with its id and usage, the main agent's or a
  subagent's — is now one model call, as the provider serves it.
* **Cache writes are counted apart, and the cost benchmark prices them.** A reply's input
  tokens written to the provider's prompt cache were folded into its input: the cost benchmark
  priced them as plain input (Claude's run $0.028 under the provider's on Haiku). Every adapter
  now accounts them apart (`budget.account(written=)`, `RUN_FINISHED`'s
  `spent.cache_written_tokens`, [observability.md](docs/observability.md#events)), and the
  benchmark prices them at `benchmark.cost.prices`' cache-write rate.
* **A server keeps at most 8 agent processes of a framework, not 16.** Claude's and Kimi's
  conversations each keep their CLI between turns; at 16, the comparison held 3.3 GB of Claude
  Code processes (about 200 MB each). The bound (`live.KEPT`) is now 8, about 1.6 GB: a ninth
  conversation's turn takes the place of the idle session asked for longest ago, as before
  ([claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md), [kimi.md](docs/frameworks/kimi.md)).
* **A refused memory write says why.** Claude's transcripts of 57 and 51 calls (116 and 104
  messages, one call a reply) were refused with a 422 on the comparison — more than one
  `POST /v1/messages` takes, which the transcript writes of at most 100 already fixed — and
  the log said only "Request validation failed". A failed write's warning event and log line
  now carry the memory service's details: each field it refused and why.
* **The write spool is a deployment's own.** Every process of a machine user shared one spool
  folder, so a process replayed another's undelivered memory writes — made with another key,
  refused (403: 33 transcript writes of a stopped run on the comparison). A harness now keeps
  and replays its writes in `<spool>/<deployment>/`, named by its memory service and key (a
  digest): another deployment's process leaves them, and its own deployment's next process
  replays them once ([memory.md](docs/memory.md#background-writes-what-is-guaranteed)). A spool
  file an earlier version left directly in the spool folder is not replayed (whose writes they
  are is not known) and stays there: move it into its deployment's folder (the one `trellis
  doctor`'s spool check names) to have them replayed.
* **Claude Agent SDK: a process's end inside a harness tool ends the attempt.** A
  `BaseException` that is no cancellation (`SystemExit`, `KeyboardInterrupt`; the suite's
  simulated worker death) raised by a harness tool ended the SDK's in-process tool server for
  the rest of the CLI's life, while the CLI went on: the attempt carried on with every later
  call failing, and `Harness.aclose` raised it at the end. The attempt now ends with it, as
  any framework's does, the CLI kept healthy; the worker that takes the run over starts a CLI
  of its own, which resumes the session.
* **Cost benchmark: the worker restart case is taken over on agent-runs.** `long-flow-04` timed
  out after 600 s on agent-runs for every framework: the suite's second worker asked once for
  the dead worker's run, before agent-runs' ticker had queued it again, and nothing asked
  again. It now runs a worker's claim loop (`Worker.run`, which keeps asking with a backoff, as
  a worker process does) until the run pauses or ends.
* **A background task that ends while its turn is finishing is told.** A task that ended after
  its turn last looked at it, but before the turn let go of the chat, found the chat busy and
  was told to nobody until the user's next message — a Deep Agents async task (`AsyncSubAgent`,
  read by the process holding its run: a loaded machine lost the chain of outcomes this way) and
  a task of the harness's own alike (one that ended after the turn's last look,
  `background.left`). An outcome told while a turn of the chat holds its turn in the process is
  now told again by that turn's own task once it lets go (`background.delivered`), by a turn of
  the chat's own; one that finds the outcome told already stays quiet.
* **Cost benchmark: every turn of the comparison's case lists is strictly scored.** Of the
  139 turns of a run (`benchmark/cost/results/haiku-2026-11-15/sets/`), 119 were: 18 filler
  turns carried only a scripted reply and 2 explain follow-ups were judge-only. Each filler
  turn now has an exact answer from the data — `conversation-08`'s supplier name, the Silk
  Scarf Navy's unit cost (its gold calls now read `sku_details`), the south's store count,
  August's top category by revenue (the question now says "by revenue": by units it is
  another), S12's empty audit, SKU-00042's price; `advconv-05`'s 12 counts of SKUs priced
  above £100. `explain-01`'s "Where did that number come from?" is scored on the source it
  names (the sales summary tool, outerwear, 1 July to 30 September 2026) and `explain-04`'s
  "Why do you think that's the right number?" on the figure and every input of its formula
  (units, lead time, on hand): a `tools` or `steps` check would want the tool names verbatim
  and a call count, which neither question asks for.

### Fixed: gaps the SDK full-stack pass found

* **A run cancelled in its own process keeps its end in agent-runs' log.** `agent.cancel`
  had agent-runs end the run `CANCELLED` before its attempt stopped, so the attempt's last
  events — the call cut short (`TOOL_CALL_RESULT`, `cancelled`) and `RUN_FINISHED` with the
  reason — were refused, and the log stopped at the call's arguments (every in-process
  adapter, and a run cancelled while it waited its turn). The attempt now stops first, its
  log takes its last events, and agent-runs' own cancel ends it, keeping the reason and who
  asked. A run a worker holds is asked to stop through agent-runs first, as before.
* **Kimi: a background command's result read in a later turn answers that turn.** A turn
  that waited for a task it had started (`WaitFor`) while the task ended failed `NO_ANSWER`:
  Kimi steers the task's notification into the turn, ends it after the wait, and answers in
  a turn of its own that ACP does not forward. The harness now waits for that turn and takes
  its answer from Kimi's log, as `kimi -p` does; its tokens are that run's, counted once
  ([kimi.md](docs/frameworks/kimi.md#background-work)).
* **A run's own framework objects survive its pause in its process.** An OpenAI Agents
  `RunHooks` or `context` object given to one run (`agent.run(framework_options=...)`) was lost
  when governance paused the run (a tool that declares nothing asks): the resume ran with the
  agent's options. The process the run started in now keeps them for its resume; a resume in
  another process runs with the agent's, as documented — an object is not stored
  ([configuration.md](docs/configuration.md#the-frameworks-own-run-options)).
* **Claude and Kimi: a file the user places is written where they said.** The line the
  harness adds about `outputs/` ("save a file the user asks for there") made Claude write a
  `hello.txt` the user asked for in the working folder into `outputs/` instead (1 run of 2 on
  Haiku). It now says a file the user names (a bare name: the working folder) or places goes where
  they said, and `outputs/` is for a file made to give the user with no place named
  ([sandbox.md](docs/sandbox.md)).
* **Docs: a framework's own helper's events are on its parent's stream.** subagents.md said
  every child run's events are on its own stream; a Deep Agents `task`, an `Agent.as_tool`, a
  Claude `agents=` subagent and a Kimi `Agent` are steps of the parent's framework run, their
  tool calls on the parent's stream as the library streams them, their child run without
  events of its own ([subagents.md](docs/subagents.md#limits)).
* **Docs: Kimi offers its model an MCP server over HTTP, not a stdio one.** Kimi 2.6.2 starts
  a stdio MCP server and lists its tools but leaves them out of its model's requests, given in
  ACP's `session/new` or in its own `mcp.json` alike; one over HTTP is offered at once
  ([kimi.md](docs/frameworks/kimi.md#models-swarms-and-limits)).
* **Structured output in memory.** With LangChain's tool strategy (`ToolStrategy`) the turn
  kept in memory ended with the schema's call and LangChain's `Returning structured response:
  ...` message, and no assistant text; it now ends with the structured answer (its JSON) as
  the assistant's message, as with the provider strategy.

### Changed: one approval rule on every adapter — an MCP tool with no annotations asks like any tool that says nothing

An MCP tool whose server sent **no annotations at all** is now *undeclared*, as a function
without `side_effects` is, on every way a tool arrives: it asks, and the person may allow it
once, for this conversation, always (with a memory service) or deny. Before, on the harness's
own `mcp(...)`, the gateway's MCP listing, an OpenAI agent's `MCPServer*` and Claude's
in-process servers it was `irreversible` — asked at every call, never "for this conversation"
or "always" — while the same tool through `langchain.mcp` / langchain-mcp-adapters was
undeclared. Annotations that leave a tool destructive by the MCP defaults (`destructiveHint:
true`, or `readOnlyHint: false` with no `destructiveHint`) still make it `irreversible`. The
harness publishes nothing of such a tool's risk to the tool catalog (it published
`destructiveHint: true`), so the catalog's risk for it is null and an administrator's still
wins.

A call a framework runs and asks its own user about — Claude Code's permission check (its
built-ins, its own MCP servers' tools, in a query and in a live session) and Kimi's permission
request — now offers the person the same choices as a harness tool's: always allow too, with a
memory service (Claude Code's question offered no choice at all, and asked again at every
call). What is allowed for the conversation or always holds as Claude Code's own "don't ask
again" would. The tool catalog's word on such a tool (a risk, an `approve_when`) decides before
the framework's own question, as for every tool.
Deep Agents' own `write_file`, `edit_file` and `delete` on a disk, a store or a sandbox (a
backend, or a route of a `CompositeBackend`) now **ask** as any write — allow once, for this
conversation, always — where they were only announced (a framework's own tool), while a tool
of yours that writes in the same graph asked; `ls`, `read_file`, `glob`, `grep` still run, and
the file tools on the graph's own state stay its scratch. `ReAct(backend=)` alike.
`tests/matrix/test_approval_rule.py` walks every adapter × way × risk × answer
([governance.md](docs/governance.md#one-rule-every-adapter-and-every-way-a-tool-arrives)).

### Fixed: three SDK features under the harness (real Haiku, the real stack)

* **OpenAI Agents: a governed call inside an agent tool resumes.** An `Agent.as_tool` agent
  whose own `FunctionTool` declares nothing paused the run on that call (governance asks),
  and the approval then failed the run (`AttributeError: 'Agent' object has no attribute
  'aget_state'`): the pause carried the agent tool's own SDK handle, which the run took for
  a LangGraph checkpointer's. A pause the framework gave no handle of is now answered from
  the journal: the agent tool is called again and the approved call runs once.
* **Claude Agent SDK: an in-process tool named in `allowed_tools` keeps its annotations.**
  A `create_sdk_mcp_server` tool marked `readOnlyHint` (a read) asked a person at every call
  when `allowed_tools` also named it, as the SDK's docs pre-approve an MCP tool: the name alone
  took the place of the server's listing, which carries the risk. With the memory service the
  tool was also published to the tool catalog as saying nothing — the protocol's default,
  `destructiveHint` — so the catalog's derived `irreversible` asked at every call whatever the
  tool said. The server's listing now decides it, and the catalog is told its annotations ([claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#using-an-existing-claude-agent-sdk-project)).
* **Claude Agent SDK: `include_partial_messages` streams.** The run's stream gave one text
  event per assistant message although the options asked the CLI for the model's deltas;
  it now carries each delta as it comes, as the SDK's own stream does
  ([claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#streaming)).

### Added: a provider's own tools in `ReAct`

`ReAct(tools=[{"type": "web_search_20250305", "name": "web_search"}])`: a provider's web
search, web fetch or code execution, given as LangChain's `dict` tool and bound to the model as
it is (it was refused as "not a tool"). Each server call the reply reports is the run's tool
call, and the catalog's risk decides whether the tool is offered: one that would ask a person
is held back, since no approval can stand before a call the provider makes inside its step
([react.md](docs/frameworks/react.md#a-providers-own-tools-web-search-web-fetch-code-execution)).

### Added: plan mode for `ReAct`

`ReAct(middleware=[PlanMode()])` (`trellis.harness.plan_mode`), entered on request with
`framework_options={"configurable": {"plan_mode": True}}` — never by default, as Claude's and
Kimi's is not. The run reads only, presents its plan with LangChain's `write_todos`, a person
approves it once (LangChain's `HumanInTheLoopMiddleware`), then it acts under the usual
governance ([react.md](docs/frameworks/react.md#plan-mode)).

### Changed: `RunCheckpointer` keeps every checkpoint

As LangGraph's own savers do: `get_state_history`, `update_state` and a thread continued from
a `checkpoint_id` work on a `ReAct` or `middleware()` graph. In a harness run the history is in
the run's journal (about 2 KB a checkpoint, 17 checkpoints for a one-tool turn) until the run
ends (the run store clears a run's checkpoint with its ending); a journal written by an
earlier release is read as it is, and an earlier release reads this one's (the latest
checkpoint's record is unchanged). A run's own thread forked from an earlier checkpoint by
code inside the run is refused (`RewindRefused`)
([react.md](docs/frameworks/react.md#checkpoint-history-langgraphs-time-travel)).

### Added: `agent.checkpoints` and `agent.rewind`

`agent.checkpoints(run_id)` lists a paused `ReAct` (or `middleware()`) run's checkpoints,
newest first, as LangGraph's `StateSnapshot`s. `agent.rewind(run_id, checkpoint_id, *,
values=None, reviewer, tenant=None)` continues the run's conversation from one of them in a new
run the run store records (`metadata["rewound"]`: the run, the checkpoint, who, when); the run
rewound stays as it was. A call that did more than read past the checkpoint asks a person
before it is made again, whatever was allowed before; what was spent stays counted. Refused:
a run at work or ended, a reviewer who is not the run's user, a checkpoint of another run, an
agent with no checkpoint history. Claude's file checkpoints are not driven by this API
([react.md](docs/frameworks/react.md#rewinding-a-paused-run)).

### Added: handoffs between agents, with `langgraph-swarm`

`langgraph-swarm` (`>=0.1,<0.2`, in the `react` extra) used as it is: a `create_handoff_tool`
among a `ReAct`'s tools is a governed, recorded hand-over that asks nobody; the agent handed to
answers, governed as ever; the conversation's next turn is that agent's (`active_agent` carried
from turn to turn), said on the run's stream; a pause resumes in place
([react.md](docs/frameworks/react.md#handoffs-langgraph-swarm)).

### Changed: Deep Agents' `compact_conversation` asks nobody

Deep Agents' `SummarizationToolMiddleware`, given to `ReAct(middleware=)` or a graph of your
own, compacts the conversation when the model asks: its tool is declared a read (it had
declared nothing, so every call asked a person)
([react.md](docs/frameworks/react.md#compaction-on-request)).

### Changed: `ReAct`'s general-purpose helper has the agent's skills

With `skills=`, the general-purpose helper (`task`) is given the agent's skills, as Deep
Agents gives the one it builds itself: its instructions list them and it reads them. Without
`skills=` nothing changes.

### Added: sub-agents in the background, with no server to host

`ReAct`'s own sub-agents — wrapped agents and Deep Agents' `SubAgent`/`CompiledSubAgent` dicts
— take tasks in the background in a chat a person reads: Deep Agents' five async sub-agent tools
(`start_async_task`...), the task a child run of the turn at work in the process and held in the
run store, its result a notification (langchain-core's notification metadata, Deep Agents'
shape) given to the conversation's model at its next model call or turn, or by a turn of the
chat's own on its thread stream when it is idle; its questions asked through the chat
(`trellis.harness.background`, `bridges.background_tasks`; [react.md](docs/frameworks/react.md#a-task-in-the-background)).

### Fixed: what the feature matrix found

- A conversation's turn keeps the graph state it carries to its next turn (in the file store,
  for any server) before its end is said, not after: the `carried_files_dropped` warning (files
  left out past the 256 KiB bound) is one of the run's events before its `RUN_FINISHED` — it
  came after it, where a stream had ended and AG-UI and A2A clients broke on an event past
  the end — and a next turn sent as the turn ends, on another server, starts with the state.
- A run taken over after its worker died while a harness tool ran makes that call again. The
  agent's session, resumed on the same machine (Kimi's home is its user's there), holds the
  call with the harness's "The run ended before this tool finished." as its result, and the
  resumed prompt said to repeat only a call with "no result yet": the agent took the error for
  the result and the run ended without that call's work. The prompt now names that result too
  (the journal still runs the call once).

### Fixed: the last open harness bugs

- Kimi: a conversation's kept process whose time is up (kept as long as a session is, idle, its
  place given to another conversation) or that the harness closes (`Harness.aclose`) leaves
  nothing behind: the process and what it started end, its background runs end, and the
  session's own folder leaves its user's Kimi home (its login and config stay). The
  conversation's next turn starts a new session, given its earlier turns by the harness, where
  it resumed a session whose files stayed on disk for good.
- Kimi: a session kept for a conversation's next turn whose process ended with its turn
  (nothing at work in its background) leaves its folder in its user's Kimi home once the agent
  no longer keeps it for that conversation (its place among the 128 most recent conversations'
  given away, or dropped for a later turn another worker served) and when the harness closes.
  Its folder stayed on disk for good.
- `ReAct(skills=)` and Deep Agents with the run's skills: reading a skill (`read_file`, `ls`,
  `glob`, `grep` under `/skills/`) is a call of the run — on its stream, traced, in memory's
  record of its calls — governed as a read, never asked. It was taken for the graph's own
  scratch and left no record.
- Claude: a subagent's reasoning is a `reasoning` event naming its child run, and a tool the
  API ran in a subagent's step (web search, the advisor) is recorded under that child run.
  Before, only the main agent's were.
- Deep Agents' tasks on a server of yours: the outcomes of such tasks, told by turns of the
  chat's own, may start or update a task three times in a row, as a harness task's may; the
  fourth fails saying so. Each outcome counted as the first, so a chain never ended.
- Kimi: a skill Kimi loads (`Skill`) is decided by governance — Kimi asks about it (an `ask`
  rule in its user's config), so the catalog can have a person approve it. A team config whose
  `[permission]` rules are written inline is asked over (a `[[permission.ask]]` rule after it)
  instead of refusing the run; only a `permission = { ... }` inline table is still refused.

### Fixed: what the framework comparison on Haiku found

- `middleware(context_window=)` below the model's own window (a team's `create_agent` or
  `create_deep_agent` given 16,000 tokens for a model of 200,000 that replies up to 64,000)
  reserves the reply out of the model's own window, as `ReAct(context_window=)` does. Before,
  it was taken out of the window given, which left 0 tokens for the input: every request of
  such a run was refused (`ContextTooLarge`), 10 scored turns of the comparison's small-window
  cases on LangGraph ([compare-2026-11-15](docs/research/compare-2026-11-15/README.md)).

### Fixed: what the real-model (Haiku) acceptance found

- Claude: the turn a live session runs by itself once a background task ended is charged in
  full — its tokens and a model call as well as its cost — to that task's run, or, with no run
  to charge (a background shell command's end), to the conversation's next turn. Before, only
  its cost was charged: against Haiku through the real CLI, 6,388 input and 70 output tokens
  of such a turn were on no run's budget.
- Claude: a query that pauses in a process of its own is charged its turn as the provider
  billed it. Claude Code sends the pause's placeholder to the model as the paused call's
  result, and the model's reply to it is billed; the harness left the process at the pause,
  so the run was charged its replies' tokens and no dollars, and never that reply (51-83% of
  the paused attempt's dollars against Haiku). The process is now read to its turn's result
  (30 seconds at most; a call made in that reply is refused, ending the turn), and the run is
  charged it: against Haiku, an approved and a rejected call match the provider's bill in
  tokens and dollars.

### Fixed: each library's own skills, through the harness

- `ReAct(skills=src)`: its general-purpose helper (Deep Agents' `task`) is told the run's skills
  and reads them through the same `/skills/` route, as Deep Agents gives the helper it adds
  itself. Before, the helper had no skills.
- Claude: the tool hints never narrow Claude Code's `Skill` tool away, so the run's skills (its
  plugin) and your own skill folders stay usable on a narrowed run. A `Skill` call is a call of
  the run: recorded, a read unless the catalog says more, then asked before it runs.
- Kimi: a run given skills works in a home of its own, which now links what Kimi keeps in your
  home for every session beside its `config.toml` and login: your `AGENTS.md`, your plugins, and
  your own skills (beside the run's; a name the run has a skill of is the run's).

### Fixed: LangChain, LangGraph and Deep Agents features through the harness

A library's feature works as the library does it; the harness governs, records and counts it.

- **Security.** Deep Agents' file tools on a `CompositeBackend` are governed call by call, by
  where the path is routed: a write or read on a disk, store or sandbox route is a call of the
  run (the catalog's rules, approvals, records) though another route keeps files in the graph's
  state. Before, one state route — the run's skills beside a sandbox,
  `create_deep_agent(**deep_agent(sandbox(), src))` — left every route ungoverned and unrecorded.
- **Security.** With a `PIIMiddleware` that checks input, the memory context the harness
  sends with the question passes the same rule (redacted, masked or hashed; `block` ends the
  run before a model call), and the turn recorded in memory holds the question as the graph
  kept it, not the raw input.
- **Security.** What a run keeps of its graph — the checkpoint a `RunCheckpointer` keeps in
  the run's journal, and a tool's `Command` kept for a replay — is read back in LangGraph's
  strict mode: only LangGraph's and LangChain's own types and the types the graph's state
  declares are made of stored data; anything else stays data and is never imported or called.
- A Deep Agents graph with no thread of its own (`ReAct`, a graph compiled without a
  checkpointer or with `RunCheckpointer`) keeps its state files and todos across the turns
  of a conversation, as it keeps its background tasks: a file one turn wrote is read by the
  next, one it deleted stays deleted. Bounded as before (256 KiB kept in the file store).
- A `ReAct` sub-agent's `interrupt_on` is Deep Agents' own approval gate again, with its
  `when`, `allowed_decisions` and `description` (it was flattened into an `irreversible`
  catalog risk for the tools it named); still one question a call.
- What a decision model used (`langchain_core.decisions`: the classifier of
  `AutoModeMiddleware` and `ModelRouterMiddleware`) is counted in the run's budget, and a
  call `AutoModeMiddleware` blocks is on the run's events and records as a failed call.
- A failed model call is tried again by one owner: where the graph (`ModelRetryMiddleware`,
  a retrying node) or the model's own client (`max_retries`) already retried, a failure that
  still ends the run is reported not retryable, so the run store does not run the whole run
  again on top of those tries.
- Fitting the window keeps a tool's provider fields (`extras`: `defer_loading`,
  `cache_control`) and leaves a provider's own tool (web search, code execution) as it is.
- A graph that summarizes its own history (LangChain's `SummarizationMiddleware`, Deep
  Agents' summarization) beside `middleware()`: fitting the window after it keeps the
  summary when it leaves older turns out (the summary itself was dropped with them).
- A graph's stream passes its stream transformers (`astream_events(version="v3")`):
  `PIIMiddleware(apply_to_output=True)` redacts the streamed answer; content-block text streams.
- A graph's own tool call ends with the message its tool made: artifacts and content blocks
  are kept (a replay after a pause included). Without `middleware()`, results are the
  library's as returned (no large-result or table rewriting), and a call made twice in one
  step runs twice; the harness's own tools are unchanged.
- No false `NO_ANSWER`: a `return_direct` tool's result, a non-conversation state, a Pydantic
  or dataclass state and an `@entrypoint`'s return value are the answer; a string input goes
  as it is to a graph whose input schema takes a string.
- `HumanInTheLoopMiddleware(interrupt_mode="per_call")` is an approval of its call; the gate's
  decision is governance's too (one question a call); an edit of a batch of several calls is
  refused unless it gives one decision for each. `interrupt(response_schema=...)` is the
  question's `expects`.
- A tool `langchain.mcp` (or `langchain-mcp-adapters`) made of an MCP server's is governed by
  the server's annotations (`metadata["mcp"]["tool"]["annotations"]`): a read-only one runs
  unasked.
- A model called in a graph's sync node is accounted and watched for stalls.

### Changed: Deep Agents 0.7.38, langchain 1.5, langchain-core 1.8, langgraph 1.2.22

- Deep Agents reads its async tasks' statuses itself and adds each outcome as a notification:
  the harness treats a notification as a message the user did not type (the turn's question,
  the transcript and memory record, the earlier-results and turn logic, the conversation's
  earlier turns); the records it updates are carried with the task list; the extra model turn
  after a final answer counts against `max_steps`, and past it the run keeps its answer
  instead of LangChain's "Model call limits exceeded".
- A task held by a server is read after 10 s at most, then less and less often (up to every
  two minutes, with jitter): an idle chat is told how it ended, a task whose
  run was cancelled from another server is cancelled on its server, and a follow-up to a task
  that ended is a run of its own.
- An `AsyncSubAgent`'s `webhook` is checked as an A2A push URL is (`outbound.validate_url`).
- An empty task list is not carried.

### Changed: the five task tools are governed one way, on a server or in the harness

- `start_async_task`, `update_async_task` and `cancel_async_task` are writes wherever the task
  runs, announced on the run's stream and not asked about; `check_async_task` and
  `list_async_tasks` are reads. The tool catalog may say more (`task_tools`, one table). A
  cancel of a harness task no longer pauses a chat for an approval.

### Performance: background tasks cost one run-store read per model call (PERF-4)

- A turn's pull of its tasks' statuses asks the run store at most once per model call, however
  many tasks are unfinished: a task at work in the process is known running, one that ended
  there is known ended, and the others are read one per model call, the one read longest ago
  first. `list_async_tasks` reads only the unfinished tasks.

### Performance: the carried task list keeps the unfinished tasks and the last 50 that ended (PERF-5)

- A conversation's task list no longer holds every task it ever started (it stopped being
  kept at about 700 tasks, past the 256 KiB bound): the ended tasks past the 50 updated last
  are dropped when the list is kept for the next turn. `check_async_task` of a task dropped
  answers that it is not tracked.

### Security: an outbound URL is checked by what it says and by where it leads (SEC-R-9)

- `outbound.validate_url` reads a trailing dot as the same host (`https://localhost./` is
  refused), refuses a host that is a number (`https://2130706433/`, `https://127.1/`), and
  returns the host lower-cased without the dot. `outbound.checked` adds the resolution: every
  address the host resolves to must be public. A2A push uses it (its own copy is gone), and an
  `AsyncSubAgent`'s `webhook` is checked with it each time a task is started, beside the
  check of the spec when the agent is built.

### Security: background tasks are bounded (SEC-R-2)

- A chat has at most as many unfinished background tasks as a run may have children working
  at once (the admission limit: 4 at first, 1 to 16, falling under pressure); a start past it
  is answered "Not started: this chat has N background tasks unfinished…".
- A task's outcome told by a turn of the chat's own may start or update a task three times in
  a row with no message from the user between; the fourth is refused (the depth is in the
  runs' `trellis_background` / `trellis_notified` metadata).
- A process keeps at most 1,024 background attempts at work; past it a start is refused, an
  idle chat's outcome waits for the conversation's next turn.

### Security: a background task belongs to the conversation that started it (SEC-R-1)

- The task tools, the questions asked through a chat, each pull and each recorded
  notification act on a task's run only when it is the conversation's: its tenant's and
  user's, started by its agent in its chat thread, under the task's id
  (`background.owned`; Deep Agents' tasks on a server: `bridges.async_tasks`). Any other is
  answered as a task the conversation does not track.
- A run's input names no task list: `async_tasks` in a mapping input is left out (the
  conversation's list is the one the harness carries). A message a client of a chat surface
  gives — AG-UI `state`, an A2A data part — keeps its role, content, name, id and tool calls;
  `additional_kwargs` and `response_metadata` are dropped, so it is never read as a
  notification (`history.from_client`).

### Fixed: what the user-scenario suite found (tests/scenarios)

- **A message longer than 4,000 characters got no memory context and lost its tool records.**
  The whole message went to the memory service as `ContextRequest.query`,
  `HintsRequest.task` and `ToolInvocationIn.task`, which it caps at 4,000 characters: a `422`,
  so the run had no context and each tool record was "failed and is lost". The question is
  sent as its head and its tail when it is longer (`clients.memory.fitted`).
- **AG-UI: the same `runId` posted twice at once (a client's retry racing its first try)
  started the run twice.** The id was checked before the run was opened and only marked
  taken after: it is taken before the first await, and the second request is a `422`.
- **A file that did not fit the file store left its first bytes there** (a full disk), so the
  next file failed too: a write that fails removes what it wrote (`FileStore.put`).
- **AG-UI: a user message with no text (`content: ""`) and no `state` started a run** that
  failed inside the framework (`EmptyInputError`); it is a `422`, as a blank message is.
### Changed: the `openai-agents` extra is `>=0.24,<0.25` (was `>=0.22.3,<0.23`)

Tested with openai-agents 0.24.0 (`openai` 3.20.0 still satisfies it; `urllib3` moves to
2.8.0, which it requires). What it changes through the harness:

- **A tool's failure no longer tells the model the error.** The SDK's default failure text
  is a fixed sentence from 0.23 (`An error occurred while running the tool. Please try
  again.`): a function tool, an MCP server's tool and an agent called as a tool alike, and an
  SDK tool given arguments that are not JSON (it no longer says they were not). The
  harness records the call as failed with the error, on the stream and in the tool records,
  and the child run of an agent tool keeps it in its error, as before; a tool's own
  `failure_error_function` and the run's `after_tool` hooks still word what the model reads.
- **The run's own `RunConfig.tool_error_formatter`** (0.24: the `tool_exception` kind, given
  the error) words a governed tool's failure under the SDK's default policy, as without the
  harness.
- **Still no background agents** ([openai-agents.md](docs/frameworks/openai-agents.md#limits)):
  0.24 has no background run, agent tool or handoff, and never polls a Responses `background`
  request; `tests/integration/test_background_by_adapter.py` pins what it does.

### Changed: an OpenAI agent's features work as the SDK does them

The harness governs, records and counts an OpenAI agent; it no longer changes what the SDK
does with your own tools and settings ([openai-agents.md](docs/frameworks/openai-agents.md)).

- **A session on `h.wrap` is each conversation's own.** It was one session for every user and
  thread: bob's model read alice's turns. Each run now uses a copy of it — the same class, the
  same storage — whose `session_id` is yours followed by the conversation's key (tenant, user,
  thread). A session that cannot be kept so (its id is a property, or its storage key was made
  from the id) is refused at `h.wrap`. A live `SandboxRunConfig(session=...)` or a
  `session_state` on `h.wrap`, one sandbox for every user, is refused too: give it per run.
- **`run_config` as a dict works** (it raised `TypeError` on every run), as `Runner.run` takes
  it and as a queued run keeps it.
- **Your tool's result is the SDK's.** An image or a file a function tool returns reaches the
  model as one (it was a Python repr string), and a typed result reaches `tool_use_behavior`,
  output guardrails and hooks as the object. Results of your own tools are no longer rewritten
  (tables, the 80,000-character stub): that is the harness tools' only.
- **Your tools keep their order** (they were sorted by name); the harness's follow them.
- **A call your reply repeats runs each time for your own tools**, as the SDK runs it; a harness
  tool still runs a repeated call once.
- **Your `tool_not_found_behavior` is respected.** The friendly default (`return_error_to_model`)
  applies only when your `run_config` does not say; say `"raise_error"` in a dict.
- **At `max_turns` the run ends `PARTIAL` with the SDK's outcome**, no extra model call (it was
  one more paid call for a last answer): the answer is the last message the model wrote, the
  error `BUDGET_EXCEEDED` names the limit. Your own `error_handlers` still answer it.
- **The Responses API through Bifrost, where the gateway serves it.** A model name goes to
  Bifrost on the SDK's default API (`/v1/responses`; it was forced to chat completions). A
  gateway that does not serve the route for a model answers so on the model's first call: the
  call falls back to `/v1/chat/completions`, once, and the model stays there (said once), so a
  deployment that worked before still does. On a model that fell back, an agent that uses a
  hosted tool, a stored prompt, a Responses setting, a sandbox capability needing them or a
  server-side thread is refused with each feature named (it failed mid-run or was silently
  ignored).
- **Your model routing is kept**: a `MultiProvider(provider_map=...)` entry and the `litellm/`
  and `any-llm/` prefixes keep their providers (Bifrost replaced them); a `RunConfig.model`
  object is said once as not routed.
- **A server-side thread the model ignores no longer drops the history**: with the SDK's chat
  completions model, memory keeps the conversation as with no thread.
- **A `SandboxAgent`'s capability tools are governed and recorded** (`exec_command`,
  `write_stdin`, `view_image`, `apply_patch`, `load_skill`): the catalog's risk, approvals, the
  journal, the record, the stream. They ran with no person asked. Say their risk in the
  catalog: undeclared, each call asks.
- **A sandbox `Memory`'s own model calls go through the run's model route and budget** (they
  went straight to OpenAI). Its consolidation agent's file writes in the sandbox are the SDK's
  own, not governed, which is said once at `h.wrap`.
- **An agent tool's own `run_config` no longer escapes the harness**: its child run gets the
  model route, the day, a made-up tool's error and the stall watch. Messages steered into the
  conversation are no longer given to a child's model calls.
- **An `Agent` subclass of your own is wrapped** (it was refused). A `RealtimeAgent` or a
  `VoicePipeline` is refused with what to do instead.
- **The SDK's own tracing is warned of, once, at `WARNING`** (it was `INFO`): it exports tool
  inputs and outputs to OpenAI unless turned off; the message names the switches.
- **"Allow for this conversation" sticks for the SDK's own `needs_approval` tools**: the tool's
  later calls run unasked in the run (the SDK's `always_approve`) and in the conversation's
  later runs.
- **An MCP server that retries itself owns its tries** (`max_retry_attempts`): the harness's
  read retries no longer multiply them.
- **An edited computer action runs only an action of the SDK's computer interface**, with
  that action's arguments. An approver's edit (or a hook's rewrite) that names another method
  of your computer object, or arguments that are not the action's, is refused: nothing of the
  computer is called, and the model is told the call was not run.

### Fixed: a conversation's messages one at a time, in the order sent, on every replica

Two messages of one chat that reached two server processes both ran, writing the same thread:
only queued runs (`start`) waited across processes. And a burst of messages in one process
kept its order only by luck. Now a turn of a conversation started with `run`, `stream`,
`serve_chat` or `serve_a2a`, or continued after its answer, takes the conversation's
`concurrency_key` and waits in the conversation's line (`trellis.harness.line`, the one
ordering of a conversation's turns): one turn at a time, in the order the messages were sent,
whichever replica each reaches.

- A queued turn (`start`) holds its place: direct messages sent after it wait for it, and a
  worker claims it only when no turn placed before it runs.
- `interrupt` stops the working run wherever it works, and never works beside it: sent to
  another replica, the message waits until the replica holding the turn has stopped it (at its
  next hold renewal, within 20 s). `steer` to a run in another process waits its turn, said
  once.
- A run cancelled while it waits its turn, from any process, never runs.
- A run waiting its turn that the run store ended (its deadline: `TIMEOUT`) reports what the
  store says: its result and its `RUN_FINISHED` are `TIMEOUT` with the store's error (AG-UI's
  `RUN_ERROR`), and nothing is written. Its refused hold stopped it, and it said `CANCELLED`
  and tried to write that over the store's ending.
- A run waiting behind a turn in another process follows that turn's events (one shared
  stream a run ahead, at most 32 a process, on its own client) instead of polling.
- A turn's earlier turns are read with one listing per ending, never paging past the messages
  still waiting (a burst's waiting turns used to crowd the history out).
- A process killed while it holds a turn frees it once its hold lapses; one killed after
  recording a turn and before holding it, 60 s later.
- The in-process run store (`LocalRuns`) holds a run as agent-runs does: one heartbeat holds
  it for 300 s at most, and the refusal another holder gets does not name the holder. One
  suite of checks (`tests/support/store_contract.py`) runs against `LocalRuns`, the test
  suite's `FileRuns` and agent-runs.
- A turn's place is the run store's: the line orders by `RunRecord.placed_at`, which agent-runs
  decides under the key's lock, and no longer reads every answer of every run in the line to
  recompute it. A store that sends no `placed_at` is a `ConfigurationError` naming the run,
  never a guessed order. `LocalRuns` sets `placed_at` with the same meaning.
- The line is listed by its own key (`concurrency_key=thread:<conversation>`), so another
  user's turns on a thread with the same name are never read (they were, one `get` each, and
  again whenever they moved). It lists the queued runs, then the running ones, so a queued
  turn a worker claims in between is never missed. `RunStore.iterate` takes
  `concurrency_key`, as `RunsClient.iterate` does.

Needs the agent-runs of this release (runs of a key created in their place, `placed_at` on the
run record and summary, the `concurrency_key` listing filter) and `trellis-contracts` with
`RunRecord.placed_at`. The two wire additions are optional and additive. [runs.md](docs/runs.md#queue-order-and-busy-conversations).

### Fixed: what the feature suite against the real services found

`make test-features` (`tests/features`) proves each documented feature's purpose against the
real memory service and agent-runs, models scripted ([docs/research/FEATURE_COVERAGE.md](docs/research/FEATURE_COVERAGE.md)).

- **`trellis doctor` reported every working agent-runs as failing.** Its runs check sent the
  key as a bearer token; agent-runs reads it from `X-API-Key` only, and answered `401`.
- **An A2A card ignored `h.wrap(description=)`.** The card read only the target's own
  description and docstring; it now says what `h.wrap` was told first.
- **A query over a kept CSV of one text column counted its header as a row.** The file that
  `query_result(save_as="csv")` writes for one text column (`line` over its values) was read
  back as lines, its header among them: one row too many. A kept `.csv` is now read under its
  first line whatever its columns hold; `read_file` still shows the file as kept.
- **A memory `504` (its request deadline) is proven handled, against the document that now
  lists it.** The fake memory service answers the middleware's `504 TIMEOUT` before a route
  runs or after a write is stored: a context is asked again and reaches the run (or is a
  warning when it stays `504`), and a turn stored then answered `504` is sent again under the
  same `Idempotency-Key` and kept once (`test_memory_outage.py`, `test_memory_window.py`).

### Changed: Kimi Code runs in a home of its user's own; a team's allow rules no longer stop the run

Every run of `kimi()` that is not sandboxed starts Kimi with `KIMI_CODE_HOME` set to a
folder of the run's tenant and user (`$XDG_DATA_HOME/trellis/kimi/...`, mode 0700): their
sessions, Kimi's logs and caches are theirs alone. It is an overlay of the team's Kimi home:
login and setup are linked, and `config.toml` is the team's with an `ask` rule added for
each `allow` rule, so a call a team rule allows is asked about and governance decides
(before, such a run was refused). A config whose rules are written inline is still refused.
Sessions are now kept per user on the worker, not in the team's `~/.kimi-code/sessions`.

### Added: a conversation's Kimi Code process stays between its turns while work in its background goes on

Kimi's background work (`Agent` with `run_in_background`, a `Bash` command run in the
background) died with the attempt: the `kimi acp` process ended when the turn did, and a
background sub-agent's child run ended `SUCCESS` with its launch text. A conversation's
process is now kept while a task is at work or Kimi's own turn for one that ended is under
way (`bridges/acp_live.py`, on the machinery Claude's live session uses: `bridges/live.py`).
A background sub-agent's child run stays `RUNNING` until Kimi's log says its task ended,
then ends `SUCCESS` or `ERROR` with its result; its calls are governed in its own run; what
it asks a person is asked in the conversation's next turn. A process that crashes, idles
past the limit or is held by another worker ends its runs saying so
([docs/frameworks/kimi.md](docs/frameworks/kimi.md#background-work)). A sub-agent Kimi goes
on with (`Agent(resume=...)`) is a child run named by the type it was started with.

### Fixed: Kimi Code's own configuration cannot turn the approvals off

Kimi's `config.toml` can start every session approving on its own (`default_permission_mode =
"auto"` or `"yolo"`) or in plan mode (`default_plan_mode = true`), and Kimi's ACP server still
reports the session as `default`. The harness set a session's mode only when the target named
one, so with such a config Kimi asked about no call and governance saw none: a `Bash` command
ran with nobody asked. Every session a run opens (new, resumed, loaded, forked) is now set to
the mode that asks before its first prompt, its plan mode kept as the session has it, and
Kimi's own log of the session must confirm it or the run fails without sending a prompt
([docs/frameworks/kimi.md](docs/frameworks/kimi.md#the-session-mode)).

### Fixed: a user's message that starts with "/" is never a Kimi command

Over ACP a prompt whose first block starts with `/` is a command of the agent's (Kimi's
`/compact`, `/status`, `/skill:<name>`; any other answers "Unknown ACP command"), run with
no model turn. A run's prompt usually has a block of the harness's before the user's
message, but a sandboxed run with no memory context did not: there a user typing
`/status of my order?` ran Kimi's command instead of asking the model. The user's message
no longer starts the prompt's first block in any run.

### Added: Kimi's plan is a run event

Kimi's todo list reaches an ACP client as its plan (`session/update` `plan`), which the
harness dropped. Each state of it is now a `CUSTOM` event named `plan` with Kimi's entries.

### Fixed: a turn Kimi could not finish fails the run, with Kimi's reason

Kimi answers a prompt whose turn failed (its provider's error, past Kimi's own retries)
`end_turn`, as it answers a finished one, and says why only in its log. Such a run ended
`SUCCESS` with no answer, or with the half answer that came before the failure. The run now
fails with Kimi's own error (its code, the provider's status), read from the log's
`turn.ended`.

### Fixed: a sandboxed Kimi command is recorded with its output

A sandboxed run's `Bash` runs in the client's terminal, and Kimi's last update of the call
names the terminal and carries no output, so the call was recorded (journal, memory, events)
with an empty result. It is recorded with what its terminal said.

### Changed: Kimi Code neither updates itself nor reports usage under the harness

`KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_DISABLE_TELEMETRY=1` are set for every run of
`kimi()`, whatever the agent's `env` says (it is told why, as for scheduling and tower
mode): an update would move Kimi past the tested version while it serves runs, and its
telemetry sent the tenants' usage to Moonshot.

### Changed: Kimi Code 2.6.2 (was 2.1.1)

The tested range of `kimi` is now `>=2.6.2,<2.7` (`compat.BINARIES`); 2.1.x is refused.
What changed in Kimi between the two, and what the harness does with each:

- **Swarms do not run over ACP.** Since 2.3.2 `AgentSwarm` runs only in Kimi's swarm mode,
  which only its user turns on in its own terminal; Kimi's ACP server has no mode, option or
  command for it. Kimi still offers the tool and answers a call with `Swarm mode is off...`:
  the run goes on, and the child run opened for each member ends `ERROR` with Kimi's words.
- **Kimi's `allow` rules apply.** Since 2.6 the `[permission]` rules of Kimi's `config.toml`
  (and a project's `.kimi-code/local.toml`) apply over ACP: a call an `allow` rule covers
  runs without Kimi asking, and ACP gives a client no way to be asked first. A run whose
  Kimi holds such a rule is not started: it fails before the first prompt, naming the rules.
- **A background sub-agent's launch is read from Kimi's log.** 2.6.2 sends no update of the
  result of an `Agent` call that launched its sub-agent in the background, so its child run
  stayed `RUNNING` for ever. The launch is now Kimi's log's `task.started`, with the call it
  is the task of; a sub-agent that asks permission before the turn took its launch is still
  decided in its own run.
- **Kimi's own turns** are read from its log's `turn.prompt` (2.6.2 writes no
  `agent.turn.started`), and a usage record's cache counts may be absent.
- **Unchanged, pinned by tests on 2.6.2:** plans are reviewed by a person, every question
  reaches the person, `Write`/`Edit` inside a git working tree are approved by Kimi itself,
  the schedule tools are refused when called (Kimi's `[tools] disabled` takes them off the
  model's list), tower mode stays off.

### Changed: "Allow for this conversation" on a call Kimi Code asks about is Kimi's approval for its session

A call Kimi asked about offered approve or reject only, and Kimi's own "Approve for this
session" was never chosen, so a conversation asked again about each call. The person is now
offered the choices of a tool the harness runs (allow once, allow for this conversation,
deny); "for this conversation" answers Kimi with its session approval and covers the tool's
later calls in the conversation, in this run and its later runs. `bridge.permitted` and
`bridge.asked` take `kept=` for a framework that can keep an approval for its own session.

### Fixed: a Kimi sub-agent's call is approved with its arguments, and every question Kimi asks reaches the person

- **A sub-agent's call was governed blind.** ACP shows no update of a sub-agent's own call,
  and Kimi's permission request for it carries the tool's name and one line that cuts a
  command at 50 characters. The person was asked to approve `{"request": "Requesting approval
  to Running: <the first 50 characters>"}`, a catalog rule's `approve_when` saw no command, and
  a read-only command could not be told from a write. The call is now decided with what Kimi
  shows its own user of it, read from Kimi's log of the session (`interaction.request`): the
  whole command and the folder it runs in, a written file's path and content.
- **Questions two to four of an `AskUserQuestion` call were dropped**, with the options'
  descriptions and the several-picks kind. Kimi sends them all only to a client that takes
  forms (ACP's `elicitation/create`); the harness now says it does, and the form is the run's
  question, as an MCP server's form is (`tools/elicit.py`).

### Fixed: a plan Kimi Code presents is reviewed by a person, with every option

Kimi asks its user to review every plan (`ExitPlanMode`): the plan, an option for each
approach, `Revise` and `Reject and Exit`. The harness treated the request as a read to let
through and answered with the first option that allows, so a plan was approved by nobody and
its first approach chosen. A permission request that offers a choice of the agent's own (two
options of one kind, where a call's request has one way each to allow and to refuse) is now
the agent's question to its user: the run pauses on one question that shows what the agent
shows (the plan) and every option it offers, and the person's pick is the answer. Kimi's
`AskUserQuestion` over a permission request is asked the same way, so its `Skip` is offered
too. A session Kimi's own `default_plan_mode` starts in plan mode plans as Kimi does.

### Fixed: one question a call, run once, decided once — every adapter, whatever asks

`tests/integration/test_one_question.py` checks, for every adapter and every way a call can
need a person's approval (a tool that declares nothing, a catalog rule, a write, a hook's
`Ask`, an allowance for the conversation or always, the library's own gate: LangChain's and
Deep Agents' `interrupt_on`, OpenAI's `needs_approval` and an MCP server's
`require_approval`, Claude Code's `can_use_tool`, an ACP agent's permission request), on a
run, a stream, a resume on another harness, inside a sub-agent: one question a call, its
body run once when approved and never when rejected, one record of it (journal, memory) and
one count (budget), and its decision fed back once on the run that recorded it. What did
not hold:

- **The same call twice in a turn: the second took the first's answer.** A replayed call
  did not read past the answer it had read, so the next identical call took that answer
  for its own: a function target or a graph without a checkpointer, approving the first
  and rejecting the second, ran both. A framework that goes on from its session (Claude,
  Kimi over ACP) or its saved run (OpenAI's `needs_approval`) does not make the calls before
  the paused one again, but the attempt read the journal from its start: the call made
  again got the earlier call's record (approved, it did not run) or answer (Claude Code's
  question rejected for the second call ran it). The journal keeps what each call read
  (`reads`) and how far the calls before the pause read (`cursors`); a subagent paused in its
  work, which its framework starts again, reads its own again.
- **A rejection is the call's it answered.** In a session that goes on (Claude, Kimi over
  ACP), a person's rejection of a call was taken by the next identical call: rejected, the
  same call made again later in the turn was refused without asking. A rejection now
  belongs to the call its framework named (its call id): that call — made again under its
  id, or still waiting — keeps it, and a new call, identical or not, is asked, as on every
  other adapter and as Claude Code itself does. The stall guard still stops a model that
  makes one call over and over.
- **A subagent started again ran a built-in it had run.** Claude starts a subagent paused in
  its work again: it made its earlier built-in calls again, and they ran again on their
  recorded approval. A built-in that ran is recorded in the journal; made again, it is not
  run, and the subagent reads its result.
- **An OpenAI agent tool's own approval was fed back on the calling run.** The SDK's
  `needs_approval` inside `Agent.as_tool` names the agent tool's child run, which records the
  call, as governance's approvals of a helper's call do.

### Fixed: Claude Agent SDK approvals, spending and live sessions

- **A question a run hook asks is no approval.** A call Claude Code asks about was let
  through without its question when any recorded answer was read while the call was
  decided: a `before_tool` hook that asked a person something else
  (`trellis.current().ask(...)`: a ticket number) let a shell write run unapproved. Only the
  answer to the approval of that call (governance's question, whatever made it ask) spares
  it Claude Code's own question now (`journal.approvals_read`).
- **Closing the harness ends every CLI and every part.** A live session ended its
  background runs (one that waits for a person is cancelled in the run store) before its
  CLI: a run store that failed then left the CLI process running, and `Harness.aclose`
  stopped there — the queued writes, the spans, the MCP sources and the clients were never
  closed. Each session's process now ends whatever its runs' ends meet, and `Harness.aclose`
  closes every part before it raises the first error (any other is logged).
- **A team's own hooks keep their timeout in a live session.** A live session told the CLI
  to wait 24 hours for every hook with no timeout, the team's own too (a hook that hung held
  a turn for a day): only the harness's own hooks, which may wait for a person's answer
  about a background subagent's call, are; a team's keep what they set (none: Claude Code's
  default), as in a process of their own.
- **A paused or cancelled turn is charged its own spending.** A turn that paused before its
  first result and was resumed on another worker was charged the conversation's earlier
  turns again: the session's cost was kept in the first worker's memory only (it is in the
  run's journal from the turn's start now). In a live session, the attempt a pause left lost
  its tokens and model call (its result came after the attempt stopped reading), and what
  a cancelled turn spent was charged to the conversation's next turn: a left turn's result,
  which comes at once when it is interrupted, is its own run's now.
- **The turn limit no longer ends a conversation's background work.** The one last answer
  at `max_turns` ran in a process of its own and closed the conversation's live session
  first, stopping the background subagent the same turn had started (its run `ERROR`). It
  is the session's next turn now, told the limit is reached and its tool calls refused
  (Claude Code's process answers the next prompt after its `error_max_turns` result, seen
  driving the bundled CLI); a turn in a process of its own answers as before.
- **A resume on another worker leaves a background subagent at work.** A turn that started
  a background subagent (kept going by worker 1's live session) and paused was resumed on
  worker 2, which ended the subagent's run `ERROR` ("not taken up again") while it was still
  at work, and started a second CLI process resuming the session worker 1's still held. What
  was handed on is in the run's journal now (`Journal.handed_on`, replacing a set in one
  process's memory): no later attempt ends it on any worker, and a session another worker's
  live session holds is forked (`fork_session`) — the turn goes on on worker 2, the subagent
  on worker 1.
- **A background subagent's harness tool calls are its own in two more cases.** A team's own
  `PreToolUse` hook that rewrote such a call (`updatedInput`) left the call unknown when it
  came with the rewritten arguments: refused for want of a run, or run in the turn being
  answered then. And a tool the process lists that the subagent's run was not offered (the
  hints narrow each run by its own task) failed in the relay ("not enough values to
  unpack"): the subagent is offered what the process lists, so it runs, in its run.
- **A conversation's turn is never served by a stale process.** A worker that served a
  turn kept its session (and its live process) for the conversation's next turn, even when
  another worker had served a later turn meanwhile: that turn was then answered by a process
  that never saw it. A kept session is dropped now when the run store has a turn of the
  conversation started after the last one this process served (`pipeline._served_elsewhere`):
  the stale process is ended and the turn runs as on a worker that kept none — a new session,
  the conversation's turns given as context.
- **The hour turning no longer ends a conversation's live session.** `schedule_task`'s
  description says the hour it is, and a live session served only a turn whose tools were
  listed in the same words: the first turn after the hour turned closed the process and
  stopped its background work (a background subagent's run `ERROR`). A later turn must bring
  the same tools with the same schemas now (`convert.listing`), not the same wording; the
  process keeps the wording it listed as it started. So that a session living for hours or
  days does not tell the model a stale hour (a wrong today-or-tomorrow for "at 9"), each turn
  it serves is told the day and the hour it is with its prompt (`schedules.now_said`, in the
  context the `UserPromptSubmit` hook adds): Claude Code itself says the date once a session
  and again only when the day changes (its `date_change` reminder), never the hour.

### Fixed: what an independent review of the adapters found

- **The run's `after_tool` hooks decide what the model reads of a failed call again.** A tool
  that raised had its error handed back to its framework as it was, whatever a hook made of
  the call: a LangGraph graph's wrapper or `ToolRetryMiddleware` worded it, the OpenAI SDK's
  failure function worded it, and a tool the SDK runs here returned its own result over a
  hook's. Now a failure a hook makes a result is that result, and a failure a hook rewords is
  still delivered as the framework delivers a failed call — raised into the graph (its retry,
  its wrappers and its tool node act on it), told by the SDK's failure policy (which may end
  the run), raised to the SDK as the same error saying the hook's words — the model reading
  the hook's words.
- **A call that asks waits for the calls of its model step however LangGraph runs them.**
  `create_agent` (and Deep Agents) runs each call as a task of its own: the wait for a
  `ToolNode`'s other calls never applied, LangGraph cancelled the write beside the call that
  asked, and the resumed run told the model its effect was unknown.
- **A computer call's actions are one decision.** A batch pressing the same key three times
  was read as a model looping: the third press was skipped, unsaid, and the run marked
  stalled. The same computer call step after step is still stopped, and an action that does
  not run is told to the model in its next call.
- **Two turns of a conversation ending together keep both their tasks** (the carried task
  list), on one server or two, in any order. The file store has no conditional write, so no
  turn writes over another's any more: each keeps a file of its own under the conversation's
  folder (what it read merged with its own state), readers merge every file there, and a turn
  removes the files it read only once its own holds them.
- **A LangGraph graph's retry policy is the only one for its own tools.** The harness tried a
  read whose error may pass again inside each of the graph's tries:
  `ToolRetryMiddleware(max_retries=2)` called the tool 9 times, a node's
  `RetryPolicy(max_attempts=3)` 9 times. A graph's own tool is now tried once by the bridge,
  and tried again around the graph's handling of its error only where the graph did not try
  it again itself — seen as it happens: a wrapper running the call a second time, or a retry
  policy of its node that retries the error. `ToolRetryMiddleware(max_retries=2)` and
  `RetryPolicy(max_attempts=3)` call the tool 3 times; under a wrapper that only handles the
  error (`ToolErrorMiddleware`, Deep Agents' file middleware, `ReAct`'s `GovernTools`), or none,
  the harness's 3 tries stand — one call of the run, as before, its tries its attempts: one
  start and one result on the stream, one record, one count against `max_tool_calls`, one
  question for a call that asks, and the `on_error` hooks told once if its last try fails
  (the `after_tool` hooks see each try, whose words the graph's handling of it carries).
- **An OpenAI function tool's own `timeout` is always that tool's timed-out call.** Counted
  from when its task began, a call whose task began late (another call holding the loop) was
  recorded as cut by a cancelled run, its start left open; it is counted from when the SDK
  made the call, as the SDK counts it.
- **One call, one question.** A person's answer about a call covers its framework's own tries
  of it again (a `ToolRetryMiddleware` retrying a failed irreversible call was asked about a
  second time); an edited call, or the same call of a new model step, is asked about anew.

### Fixed: a call across a run's attempts, and a turn while memory is slow or down

Four findings of the user-scenario suite (SCN-2, SCN-3, SCN-4, SCN-6), each tested in process
in `tests/integration/test_scenario_fixes.py`:

- **Writes to memory were lost when the memory service was down for more than a few seconds.**
  A background write had three tries half a second apart; once the memory client's circuit
  opened each try was refused at once, and with no spool (none in a container) transcripts,
  tool records and outcomes were "failed and is lost". While the circuit is open a write now
  spends no attempt: it waits the circuit's cool-down, queued, and is sent when the service is
  back — however long that takes, bounded by the queue (10,000 writes; past it a write is
  spooled or lost and reported, as a full queue's always was). **A write now has six attempts,
  not three**: one more than the five failed calls that open the circuit, so the first write
  of an outage is still trying when it opens. Still lost: what waits in the queue when the
  process is killed, with no spool what a clean stop cannot deliver in 10 s, and a write that
  fails six times while the service answers other calls ([memory.md](docs/memory.md)).
- **The next turn was answered without the turn before when memory took more than 5 s to keep
  it.** Where memory's window comes as messages (LangGraph's graphs, `ReAct`), the
  conversation's latest turns memory does not hold yet are added after it from the run store
  (each one's question and answer), on whichever server the next turn runs; matched by message
  id, so a turn memory holds is never given twice; with nothing in memory's window, the run
  store's newest turns (at most 10, within the conversation's budget). A chat message's
  `run_record` in memory now carries its `message_id`. One listing of the run store per turn
  of a conversation.
- **A call with no framework id had a different id in a later attempt, or another call's.**
  A call a graph's node or a function makes itself was `<run>:<step>`, the step counted per
  attempt: after a resume a checkpointed graph's payment took the id of a lookup made before
  the pause. It is `<run>:<digest of the tool and arguments>:<which time>` — the call's
  idempotency key — in every attempt. **The format of these ids changed**; a framework's own
  call ids are as they were.
- **A call replayed after a resume was announced again** (`TOOL_CALL_START`…`TOOL_CALL_RESULT`
  with `cached`), so a run's event log, a reconnecting AG-UI client and a thread's replay
  showed it twice. A replayed call emits nothing: the attempt that ran it announced it. The
  AG-UI activity line counts a run's calls across its attempts on the replica that served them.

### Fixed: a tool result shaped like the journal's record of a call is its result

A run's journal kept a call's output bare and an outcome that was none (a timeout, an unknown
effect) under the key `trellis_outcome`; a tool that returned a dict of that shape — third-party
JSON passed on — was replayed as the harness's record after a pause: every resume of the run
failed, or the tool's author chose the replayed status and metadata (security review, SEC-R-5).
Every call's record is now kept in an envelope (`trellis_record`: status, output, metadata) and
read back from those three fields only, so an output is never read as a record. A journal
written before this release is still read (a run paused before the upgrade); a server of the
release before does not read the new shape: resume a run where it paused, or upgrade every
server before resuming.

### Fixed: a run that fails leaves no child `RUNNING`

A run whose model call stalled ends `ERROR`; a helper its framework had open in the foreground
(a Claude subagent) was let go of as for a pause and never ended: `RUNNING` for ever, held by
nothing (architecture review, ARCH-1). A run's open children — helpers, sub-agents' runs — now
end with it whatever its final status, once, where the run is concluded: with its status and
error when it failed or ran out of time, cancelled with its reason when it was cancelled, and
cancelled ("the run that started it ended first") when it answered without them. A child handed
on to its conversation's session (a Claude subagent working in the background) still goes on.
One listing of the run's children at the end of every run.

### Changed: what a turn costs the run store and the journal does not grow with the conversation

- A call its framework ran itself (Claude Code's built-in tools, a provider's hosted tools, an
  ACP agent's) is journaled as the stream previews its result (2,000 characters), not whole:
  200 calls of 10 KB kept 2 MB in the journal and every pause checkpoint. Whole still while a
  subagent works in the foreground — started again after a pause, it reads its calls' results
  again (performance review, PERF-8).
- Whether another server served a turn of the conversation since this one's last is read from
  the newest runs only: two requests whatever the conversation's length, not one for each of
  up to 50 turns (PERF-6).
- The turns memory does not hold yet are read from the run store one run each; the turns it
  holds are not read.

### Fixed: two failures a busy machine showed

- **A wide table can be queried.** Typing a saved result asks every column whether a cell
  says it holds no value (`NA`, `N/A`, `-`...); asked as `IN` a list of words, DuckDB made
  each column's question a join of its own, and the statement that casts every column ran
  out of the database's 512 MiB: a CSV of 200 columns (of one row) could not be queried, and
  one of 67 failed now and then. The question is one regular expression a column now
  (`query._missing`): a table of a thousand columns loads.
- **A sandbox is deleted once.** The reaper (in the background, when a process makes a
  sandbox) could list a sandbox whose run had just ended in the same process while that
  run's end deleted it, and delete it again: two deletes of one sandbox at once, which a
  provider may refuse (stopping the reaper's round, and logging the run's end as not
  deleted). Whichever begins first deletes it; the other passes it over.
- **A Claude live session waits for its own turn for a finished task.** When a background
  task ended while no turn was served, Claude Code runs a turn of its own for it as soon as it
  gets to it; the session counted as idle from the task's end, and closed (its process with
  it) if that turn had not begun in time, or ran long: what the turn said was lost to the
  conversation. Such a turn under way is work now (`LiveSession.injected`), and one due —
  from the task's end until the turn begins — is waited for 30 seconds at least
  (`LiveSession.due`, `SETTLE_SECONDS`) before the session counts as idle.

### Fixed: a graph's own tools, as LangGraph and Deep Agents have them

The LangGraph adapter was checked feature by feature against the installed libraries
(LangGraph 1.2, LangChain 1.4's `create_agent` and its middleware, Deep Agents 0.7): each
feature run through the harness with a scripted model, 76 tests in
`tests/integration/test_langgraph_library.py`, `test_create_agent_library.py`,
`test_deepagents_library.py`, `test_tool_updates.py` and `test_tool_errors_library.py`. Most
worked as the library gives them. Five did not:

- **A tool's error reaches the graph's own handling.** The harness records a failed call and
  handed the graph a failed result in the error's place, so nothing of the graph's that acts
  on a tool's error ever saw one: LangChain's `ToolRetryMiddleware` never tried again,
  `ToolErrorMiddleware` never worded the error, the `except` of a `wrap_tool_call` wrapper
  never ran. The call is recorded as before and the error is raised again as the tool raised
  it. An error nothing handles is, as before, the failed call's result the model reads (the
  run goes on). The tries a wrapper makes of one call are not counted as a model repeating a
  call (the stall guard stopped the third).
- **A tool's `Command` is the graph's again when its call is replayed.** A call answered
  from the journal gave the graph the message the model had read and dropped the state
  update the tool had returned with it: after a resume without a checkpointer, what a Deep
  Agent's sub-agent had written was gone (`task` returns its files as a `Command`), and so
  was any state a tool of the graph's own had set. The journal keeps what the tool returned
  with the call's record.
- **A list of `Command`s and messages returned by one tool is applied.** The harness gave the
  graph one message in its place, and no update was made.
- **A call that asks waits for the calls a `ToolNode` runs beside it.** The node stops at the
  first call that raises and leaves the others running: a write beside a call waiting for an
  approval ended after the run's pause was recorded, and the resumed run told the model its
  effect was unknown ("interrupted by a crash"). (This missed `create_agent`, which runs each
  call as a task of its own: fixed above.)
- **A Deep Agent with a `mode="fork"` sub-agent is wrapped.** `h.wrap` never returned: a fork
  runs the graphs it is one of, and reading the sub-agents' file tools went round.

Four more where the harness restricted what the library does:

- **An error nothing of the graph handles fails the run**, as it fails the graph without the
  harness (`ToolNode(handle_tool_errors=False)`, `create_agent`'s default,
  `ToolRetryMiddleware(on_failure="error")`, a wrapper that raises it again): the run ends
  `ERROR` with the tool's error, as any graph's error ends a run, and the call is recorded as
  failed. It was turned into a failed result the model read. A graph with the harness's
  middleware (`ReAct`, `middleware()`: `GovernTools`) keeps the harness's documented rule:
  the model reads the failure and the run goes on.
- **The graph's own text for a failed call is what the model reads**: a `ToolNode`'s
  `handle_tool_errors` text reached the model as `"<tool> failed: <text>"`. The run's
  `after_tool` hooks still have the last word (for an error the tool raised, only since the
  fix above).
- **What a node writes to LangGraph's `custom` stream mode is on the run's events**:
  `get_stream_writer()` (and a middleware's `runtime.stream_writer`) writes are `CUSTOM` events
  named `custom`, the write in `value` (AG-UI's `CUSTOM` event), from `agent.stream` and
  `agent.run` alike. `agent.run` reads the graph with `astream` (the `values` and `custom`
  modes), as `ainvoke` does.
- **A decision the graph makes again is not a model repeating itself**: the stall guard counts
  the same call made again only when it comes from another decision — for LangGraph, another
  task or another call id (`Runtime.deciding`). A node its `RetryPolicy` ran a third time had
  its harness call refused as a stall.

### Added: a background task of Deep Agents' async sub-agents, found again in later turns

Deep Agents' async sub-agent (an `AsyncSubAgent` in `ReAct(subagents=)`, or its
`AsyncSubAgentMiddleware` in a graph of your own) runs a task on a server of yours while the
chat goes on. Deep Agents keeps its list of tasks in the graph's state, and a graph the
harness runs on its run's thread starts each turn with a new state: the task started, and
the next turn's `check_async_task` was refused its id.

- **The list goes from a conversation's turn to its next one** (`adapters.langgraph.CARRIED`:
  the one place that names the channel), per user, thread and agent, a cancelled turn's too —
  for `ReAct`, for a graph built with `middleware()`, and for one with neither `middleware()`
  nor a checkpointer, whose list is read from the state each turn ends with. A graph with a
  checkpointer of its own keeps it in its thread, as before.
- **Across servers and restarts.** The list is kept in the process that ran the turn and in
  the file store (`TRELLIS_FILES_URL`), under `.carried/<tenant>/<user>/<conversation>/<agent>/`,
  which no listing of a user's files shows: the next turn has it after a restart, and on
  another server when the servers share the store (unset, it is a folder on the server's own
  disk). Deep Agents' record of each task and nothing else, at most 256 KiB for a
  conversation; a store that cannot be read or written is said in the log and the list stays
  in the process. Nothing is added to a run's record, to agent-runs or to the memory service.
- **Turns that overlap lose no task.** What a turn ends with is merged into what is kept,
  task by task, instead of replacing it: a turn approved after the conversation's next turn
  ended, or a worker's turn still running while another ends, each keep their tasks. (Two
  turns ending at the same moment could still lose one's: fixed above.) A turn
  that waits for an approval has its list kept as it pauses: the task it started is listed
  by the next turn meanwhile, and is still found, cancelled, when the turn is cancelled at the
  approval.
- **The five task tools say what they do** where the harness says it of Deep Agents' own
  tools: a status check and the list are reads; a start, an update and a cancel are writes,
  announced and not asked about (the tool catalog still comes first). A write's result is
  declared the record of what was done, not a value that changes: with memory on, an earlier
  start in a later turn's history was marked "out of date: call start_async_task again",
  which invited a second task; it is left as it was said, and a status check's is still
  marked.

- **Each task is a child run of the turn that started it** (`bridges.async_tasks`): its
  task's id, server and graph recorded, held in the run store while it runs (no polling), and
  ended `SUCCESS`, `ERROR` or `CANCELLED` when a later check, list or cancel says how the task
  ended, on any server that shares the run store.
- **A cancelled turn cancels the tasks it started**, on their server through Deep Agents' own
  `cancel_async_task` and as runs — cancelled while it works, at its approval, or taken over
  by a newer message — where they ran on.

Not in this release: nothing tells the chat when a task ends (the model checks in a later
turn); a graph with neither `middleware()` nor a checkpointer loses the task of a turn
cancelled after its start (the task is cancelled with it). See
[docs/frameworks/react.md](docs/frameworks/react.md#a-task-in-the-background).

### Changed: Claude Code through the harness as Claude Code does it, governed and isolated

`claude-agent-sdk` 0.2.182 (Claude Code 2.1.325), from 0.2.160.

- **A user's text is text.** The run's input is sent with the SDK's `verbatim_prompts`: an
  `@/path` in a message read that file of the worker, and a `/command` ran, neither governed
  nor recorded. `framework_options={"verbatim_prompts": False}` sends a slash command on
  purpose.
- **Each tenant's user has a Claude folder of their own** (`CLAUDE_CONFIG_DIR`), and Claude
  Code's auto memory is off (`CLAUDE_CODE_DISABLE_AUTO_MEMORY`): runs on one worker shared
  `~/.claude` — its settings, sessions and memory files. The agent's own `settings`,
  `skills`, `plugins` still apply; a folder set in the options' `env` stands.
- **A call governance cannot decide does not run.** An error in the harness's `PreToolUse`
  hook left the call to Claude Code's permission mode (it ran under `bypassPermissions`).
  The hook refuses such a call, and its matcher is `on_failure="block"`.
- **Every tool of Claude Code 2.1.325 is declared by what it does** or is not offered:
  loading a deferred tool and waiting for MCP servers are reads, a shell, a worktree, a
  published page and a claude.ai project are writes, `RemoteTrigger` is off with the
  schedules and `ListAgents` with `SendMessage`; names the CLI no longer has are gone.
- **Messages from other Claude sessions are refused** (`crossSessionInbound: refuse` in the
  settings sent to the CLI): every Claude Code on a worker is one OS user's, so another
  tenant's session could send a turn into this one.
- **What Claude Code asks its user, a person is asked**: leaving plan mode, `WebFetch`,
  `WebSearch` and a `Skill` ran unasked as reads. A rule for the tool in the catalog lets it
  run.
- **At its turn limit a run ends `PARTIAL`** with what Claude said, as Claude Code ends the
  query (`error_max_turns`); the one more paid query for a last answer is gone.
- **An approval is carried for the call it answered only** (SEC-R-7): with no call id from
  the CLI the carry fell back to the tool's name, so one approved `Bash` let every later one
  through unasked. Only a real id is carried, and it is dropped when the call ends.
- **A conversation's kept process names no single run to the gateway**: it sent its first
  turn's run id and `traceparent` with every later turn's model calls (booked to the wrong
  run). It carries the dimensions that hold for all its turns.
- **No more than 16 kept Claude processes in a server process** (PERF-3): the cap was
  checked before the process started and counted per agent, so a hundred first turns at once
  kept a hundred. The place is taken first and counted over all agents; the idle session
  that gives its place is the one used longest ago of the user who holds the most.
- **A harness tool's call in a kept process is the run's its id names** (ARCH-16): two
  background runs making the same call with the same arguments were told apart by "the first
  that matches". Claude Code names the call in the MCP request (`_meta`), which the
  `trellis` server now reads (the `claude-agent-sdk` extra pins `mcp>=2.2,<3` for it).
- **The team's hooks answer before governance decides.** Claude Code asks all `PreToolUse`
  hooks at once: governance decided, and the run recorded, the arguments the model gave
  while the call ran with the ones a team's hook rewrote; a call a team's hook refused was
  still put to a person; and the record kept the result a team's `PostToolUse` hook had
  replaced. The harness's hooks now wait for the team's about the same call.
- **What Claude Code says beside its replies is on the run's stream**: its rate limit, a
  retried model call, a compacted context, a model fallback, a session-store miss and a
  failed reply were dropped; each is a warning with its own code.
- **A conversation reset is charged what the session cost since** (ARCH-13). A result
  carries its session's running cost, which Claude Code zeroes when the conversation is
  replaced: a turn in a process of its own counted the lower total as nothing spent, so the
  spend escaped the run's budget. One `SessionCost` owns the rule for a query and for a
  kept process: all of a total lower than the last seen is charged.
- **A read is not answered across a write.** A query run again from its prompt (or a
  subagent started again) made a `Read` again and was answered with what it had returned
  before a later `Write` of the run. The journal drops the reads a framework ran itself when
  a write of the run runs: such a read runs again; a write still runs once.
- **The team's hooks run between a kept session's turns.** With no turn under way every
  hook of the team's own was answered for it and never run: `Stop` and `UserPromptSubmit`
  of the turn the session runs by itself, a `SubagentStop` whose run had ended, the
  `PreToolUse` of a call nobody could decide. They run as Claude Code runs them, once.
- **Claude's reasoning and the tools the API ran are on the stream and recorded**: a
  `thinking` block is a `reasoning` event (its text; never its signature, nothing to
  memory), and a server tool's use is a call of the run, recorded as a read.
- **Options with no `system_prompt` keep none** (the SDK's default): the line about
  `outputs/` made it a custom prompt. It goes with the turn's context there, and after the
  prompt where the options have one.
- A resumed attempt is not cut at the options' `resume_session_at` again; the journal's
  session store lists a session's subagent transcripts (`list_subkeys`); the live tests'
  skip finds the bundled CLI.

### Fixed: the Claude Agent SDK's features through the harness, each tested on both paths

Driving the bundled CLI (2.1.283) against a scripted model showed what it sends, and where a
wrapped agent did not get what the SDK gives:

- **Subagents.** Claude Code runs a subagent in the background unless its call says
  otherwise. In a run with no conversation its child run ended at its launch, `SUCCESS` with
  the launch text, and its calls were recorded under the run: it now stays open until its
  task ends (`SUCCESS` with what it said; `ERROR` when it failed, was stopped, or the process
  ended first), and says so on the run's stream (`helper_finished`; a conversation's
  background subagent too, while the turn that started it is still answered). Two subagents
  of one type at work at once each record their own calls (by the task's id, `agent_id`,
  not the type's name). A subagent's forwarded text is no longer streamed as the run's.
- **Spending.** A result's cost is its session's running total and its usage the main
  agent's own. A query whose session ran a turn of its own, the last answer at the turn
  limit, and a conversation's turn in a process that resumed the session were charged the
  total again; each is charged what the cost grew by (the journal keeps the session's cost
  across attempts). A subagent's replies' tokens count toward the run's budget.
- **`max_budget_usd`** is Claude Code's cap on what one process spends: options with one run
  each turn in a process of its own again, so the cap stays a query's and a conversation
  goes on past it.
- **One call, one question.** A tool that says nothing of what it does asked a person
  twice for one call: governance's question, then on the resume Claude Code's own. The
  second is not asked — whatever made governance ask (the tool's risk, a catalog rule, a
  hook's `Ask`, a rule about the arguments a hook rewrote), the main agent's call or a
  subagent's.
- **Claude's own tools.** Its task list (`TaskCreate`, `TaskGet`, `TaskList`, `TaskUpdate`:
  `TodoWrite`'s successors), `EnterPlanMode` and `StructuredOutput` (`output_format`) asked
  a person on every call — a structured answer paused its own run. They run, as `TodoWrite`
  does.
- **Slash commands.** A turn the CLI answered itself (`/compact`: no reply of the model's
  in it) ended `ERROR` as a run with no answer; it ends `SUCCESS` with an empty answer, as
  any framework's run no model replied in. A blank reply of the model's still ends
  `ERROR` (`NO_ANSWER`).
- **Deadlines.** A native helper's child run (a Claude, Deep Agents or OpenAI Agents
  subagent) is recorded with the deadline of the run that started it, as a sub-agent's run
  is: agent-runs ends one a killed server left `RUNNING` by then.
- **Sessions.** Options with `session_id` failed when a pause resumed them (Claude Code
  takes it with `--resume` only to fork), and options with `continue_conversation` failed at
  once (the SDK asked the journal's store for the latest session): both work.
- **Dynamic workflows.** The bundled CLI offers `Workflow` with no flag: one background
  task (`local_workflow`, no `is_backgrounded`) whose agents have no task of their own, an
  agent's first call reaching the harness before any frame names it. The launch asked a
  person as an undeclared tool, and its agents' calls were taken for the main agent's. A
  workflow is now one child run, kept as a background subagent's and ended as its task
  ends; its agents' calls are its run's by their kind (`workflow-subagent`) while it is the
  one at work, and, while several are, those of a run the workflows share (`workflows`, a
  child run of the turn that made them concurrent), which decides them in and between turns
  — a person's question asked once, an approved call run once — and records them marked
  with the agent that made them.
- **Messages to other agents.** `SendMessage` resumed a finished background subagent as a
  new task no child run held, or reached another Claude session: it is disallowed, as the
  schedules are, with the agent-teams tools older CLIs have (`TeamCreate`, `TeamDelete`;
  2.1.283 has none and `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` changes nothing it sends).

Tested through the harness on the query path and in a live session: every hook event of the
options' own, `can_use_tool` with its suggestions and permission updates, every option's
flag, the system prompt and thinking forms, structured output, partial messages, the session
options and the SDK's errors. What stays as it is:
[docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#limits).

### Fixed: a run's open children end with it when it runs out of time or is cancelled

A run that ended `TIMEOUT` left a helper a framework had spawned in the foreground (a Claude
subagent, a Deep Agents `task`, an OpenAI agent tool) `RUNNING` for ever, and a sub-agent's
child run working inside one of its calls ended `CANCELLED`, as if its caller had gone away;
a run its worker was asked to stop, or whose caller went away, left its helpers `RUNNING` too.
Where the pipeline ends the run out of time or cancelled, its children still open end as it
did (`pipeline._children_ended`): `TIMEOUT` with the run's error, or `CANCELLED` with its
reason, and their children with them. A sub-agent's run its parent's clock cut off leaves its
ending to its parent. A Claude subagent working in the background of its session, handed on
past the turn that started it, is not ended with that turn (`Journal.handed_on`).

### Fixed: the runs a killed server was executing are ended, not left `RUNNING`

A run the harness executes in its own process — a turn of `run`/`stream`/`serve_chat`, a
resumed one, a sub-agent's child run, a helper a framework spawned (in the foreground or in
the background, as a Claude subagent working in its session) — is started, not claimed from
the queue, so no lease said when nothing worked on it: a server killed with it (a crash, a
restart) left it `RUNNING` for ever unless it had a deadline or agent-runs a maximum working
time. Every such run is now held in the run store while it executes (`holding.Holds`, one
place: `pipeline.attempt` for every attempt, `bridges.helpers` for the helpers open in it,
`claude_live.Background` for a background subagent's run between its attempts): agent-runs'
heartbeat naming the harness (`Harness.holds.name`), every 20 s for 60
(`holding.HOLD_SECONDS`), renewed however long the attempt takes. When the server dies the
holds stop and agent-runs' ticker ends each run `ERROR` (`holder_lost`) once its hold lapses,
never queueing it. A run that waits for a person is not held (a pause lets go of it, and of
the helpers open in it, which wait with it); whichever server continues it after the answer
holds it again. A run a worker claimed keeps its lease, and takes no hold. `LocalRuns` holds
the same way and ends a lapsed hold at its next claim, as it requeues a lapsed lease.

A background subagent's question a killed server left waiting, answered in the
conversation's next turn on another server, could continue nothing (its task went with the
server's Claude session) and its run was left `PAUSED`, the answer lost. The turn now records
the answer on that run and ends it `ERROR` (`holder_lost`, `claude_live.GONE`), and is told so
(`claude_session`). An answer reaching such a run through the run store alone is ended by
agent-runs (an answered run held when it paused that nobody holds again within 60 s).

The stand-in CLI behaves as the bundled one (2.1.283) does when its SDK's process dies (stdin
closed): with nothing in the background it ends; with a background subagent at work it goes on
to that work's end — a call it makes has nobody to decide it and does not run — then ends.

Needs the agent-runs with holds (an older one answers these heartbeats `409`: nothing is held,
as before).

### Added: a conversation's Claude Code process stays between its turns, with its background work

A Claude Agent SDK turn ran in a CLI process of its own, and the next turn started another
that resumed the session. What Claude started in the background — a subagent with
`background`, a `Bash` command with `run_in_background` — belongs to its process: a
background subagent held the turn open until it finished, a background command ended with the
turn, and an approval ended both. A conversation's turn now leaves its process running
(`bridges.claude_live`: the SDK's `ClaudeSDKClient`) and the next turn is sent into it, so
that work goes on between the turns. A turn ends at the result of its own prompt
(`ResultMessage.origin` none or `human`), never at one the session injected for a finished
task; a pause ends the turn and keeps the process, and the resumed attempt is sent into it;
cancelling a turn stops that turn only (Claude Code's own interrupt stops its background work
too, so with any under way the turn ends at its next tool call instead). Each turn's hooks,
permission check and harness tools still run for that turn's run. A background subagent
keeps the child run its `Agent` call started: open until the SDK says its task ended
(`SUCCESS` with what it said last, `ERROR` for a failed task or a session that ended first,
`CANCELLED` for one stopped in Claude Code), its calls decided, run and recorded for that
run — hooks, governance, `can_use_tool`, harness tools — with or without a turn being
answered. A call of it that asks a person pauses that run and waits in Claude Code; the
question is asked in the conversation's next turn, before its prompt, and the answer goes
on to the run (nothing answers a child run directly today). Its limits are the agent's
`run_timeout` and `budget`; past them, or cancelled (`agent.cancel(child_run_id)`), the task
is stopped with the SDK's `stop_task`, that subagent only. The process's running cost is
shared at each result by the tokens each used, so a subagent's work between two turns is
charged to its run and the next turn its own. Any other call made between two turns (the
main agent's, in a turn the session injected) has no run and is refused, not run unchecked,
and the next run says so (`claude_session`). With no live process (a restart, another server, one closed for being idle, a
run with no `thread`) a turn runs as before, resuming the session from the journal, which
holds what was mirrored between the turns too. Limits and what is not kept:
[docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#conversations).
`TaskStop` (a write) and six tools that only list or look up (reads) have a declared risk;
they asked a person on every call. Claude Code's own schedules (`CronCreate`,
`ScheduleWakeup`) stay off: a turn they fire has no run.

### Fixed: a Claude subagent is one child run when the run store takes time to answer

Claude Code says a subagent started two ways, its `Agent` call's `PreToolUse` hook and the
message that carries the call, and each started the helper by the call's id. With a run
store that takes time (agent-runs, over HTTP) the second came while the first was still
being recorded and started a second child run of the same subagent, which then ended at
once. A helper's start now takes its key one at a time (`bridges.helpers.Helpers.start`).

### Fixed: OpenAI Agents and ACP features that did not work through the harness

Each of the two libraries' own features was run through a wrapped agent with a scripted model
(`tests/integration/test_openai_agents_features.py`, `test_openai_agents_mcp.py`,
`test_acp.py`, `test_function_agent.py`); five did not work and do now:

- **An SDK approval inside an agent tool.** A `needs_approval` tool of an agent called as a
  tool (`Agent.as_tool`) paused the run, and every approval paused it again: the agent tool's
  unfinished call was journaled as done, so the resume replayed it and the run inside never
  continued. The call is now a pause to the bridge while the SDK holds its run waiting, its
  child run stays open, and the approval (or rejection) continues it, from another worker too.
- **The run's own `RunHooks`.** `framework_options={"hooks": ...}` was refused; the SDK's
  `RunHooks` are now called beside the harness's own (which still count every reply toward
  the budget). `starting_agent` and `input` stay the harness's.
- **An MCP server's resources.** The governed copy of an agent's MCP server answered
  `list_resources`, `list_resource_templates`, `read_resource` and `cached_tools` with the
  SDK base class's defaults; they are the server's own.
- **A shell tool's approval.** The SDK's approval of a call with no arguments of that name (a
  `ShellTool`'s commands) showed the person an empty call: it shows the call as the model
  made it.
- **One question for one call (ACP).** A call of an ACP agent's own that declares nothing (a
  tool of Kimi's own MCP servers, any tool of an agent with no profile) asked a person twice:
  governance's question, then the agent's. `bridge.permitted(..., framework=)` asks the
  framework's question only when no person was just asked about the call. An agent with no
  profile is named "the agent" in it, not Kimi Code.

Then, closing what was left:

- **The tools the SDK runs on this machine** (`ShellTool` with a local executor,
  `LocalShellTool`, `ApplyPatchTool`, `ComputerTool`, `CustomTool`) ran with no governance
  and no record: an `rm -rf` the model asked for ran unseen. Each call now goes through the
  bridge where the SDK hands it to the tool's code (`tools.convert.openai_local`): governed
  (one that declares nothing asks; the person sees the call), journaled, recorded, counted;
  reject and edit work; a call the tool's own `needs_approval` held is asked about once.
- **A `handoff(agent, ...)` object's agent** ran ungoverned: it now hands off to the governed
  copy of its agent (read from the SDK's `Handoff._agent_ref`, checked at `h.wrap`).
- **A tool's failure** the SDK formats is recorded as a failed call; **its own timeout** as
  that tool's call that timed out (it said "the run was cancelled"); **`failure_error_function
  =None`** (a function tool's, an MCP server's, an agent tool's) ends the run with the error, as the SDK does; an agent tool that fails under the default policy is a failed child run whose caller reads the SDK's default text (it read the harness's `check failed: ...`).
- **A call that pauses inside** (a question in a tool, an approval inside an agent tool)
  counted twice toward `max_tool_calls`: once now.
- **A `session_input_callback`** that put the earlier turns after the new input got the memory
  context marked into an earlier question: it goes into this run's.

### Fixed: a paused attempt's last words no longer stand for the run's answer in memory

A graph that paused with its own `interrupt()` after the model wrote a draft sent the draft
under the run's answer id (`conversation.answer_id`). The memory service keeps the first
message of an id it has seen, so the answer the resumed attempt gave was dropped and the
draft stood for it in the thread; the draft itself was stored a second time, named by the
resumed attempt's place. A paused attempt now names its last message by what it said
(`conversation.said_id`: who said what, and which time), like every other message that names
nothing else, so a message said again on a later attempt keeps its id and is stored once,
and the answer takes the answer id.

### Fixed: a person's approval of a helper's call counts in the memory service

A call a helper the framework spawned makes (a Deep Agents `task` sub-agent, ReAct's
helpers, an OpenAI agent called as a tool, a Claude subagent) is recorded under the helper's
run, but the decision on it was sent as feedback on the paused run, and the memory service
counts an approval only where the call it approves was recorded: those approvals reached
neither the tool's statistics nor its learned approval rule. The approval's interrupt now
names the helper in its payload (`{"native_helper": {"run_id", "name"}}`, beside a
sub-agent's), and the resume sends the decision on the helper's run
(`Feedback.agent_run_id`, the run the feedback is about).

### Fixed: update_schedule describes each of its parameters

`update_schedule`'s `task`, `condition` and `enabled` had a type and no description, so the
harness's own tool set check flagged its own tool. `task` and `condition` now carry
`schedule_task`'s descriptions (the same constants), and `enabled` says that false pauses and true
resumes; a test checks that every parameter of every tool the harness offers is described.
`doctor.SECRETS` no longer names three fields that are not settings.

### Fixed: a catalog's rules stand no more than 300 s while it cannot be read

Rules read earlier stand for `GOVERNANCE_STALE_SECONDS` (300) while the tool catalog cannot be
read; their age was judged only when a read was tried (every 30 s), so they could stand about
half a minute longer. It is judged each time a rule is asked for: at 301 s every tool that does
more than read asks.

### Fixed: an in-process run store's temporary folder goes with the store

`LocalRuns` (no `RUNS_URL`) kept a checkpoint's segments in a `trellis-runs-*` folder it made
in the temporary directory, and nothing removed it: one left behind for every harness
started without `RUNS_URL` (a run of this repository's tests left thousands). The folder is
removed when the store is closed, and when a store nobody closed is collected or the process
exits. A folder passed as `payload_root` is the caller's and stays.

### Changed: memory is asked for its whole budget unless the model's window is small

The context pushed for a question was budgeted at 5% of the model's window, between 2,000
and 8,000 tokens, and at 2,000 when the window was not known: a 128,000-token model got
6,400 and a 64,000-token one 3,200, where the memory service's accuracy is measured at
8,000. At 2,000 the service gives the profile 1,000 tokens, and a profile of ordinary size
no longer fits beside the rules that apply: it drops the whole `user` block. The budget is
8,000 now (`clients.memory.CONTEXT_TOKEN_MAX`), whether or not the window is known; only a
window known to be under 32,000 tokens gets less, a quarter of itself and never under 2,000
(`CONTEXT_TOKEN_MIN`, which replaces `CONTEXT_TOKEN_BUDGET`). The service packs what bears
on the question, so a request nothing answers stays a few hundred tokens. The front end's
context and state, which take a share of the same budget, have that share of the larger
one. A model whose window is not known and is in fact small can now be sent more than it
holds: say its window (`ReAct(context_window=)`, or a `context_window` on the target).

### Fixed: values of an earlier request are asked about when they come as memory entries

What the user asked for in an earlier chat reaches a new one as a memory entry
(`## Memories`), not only as a profile line. The note on using them (`agent.REMEMBERED`) came
only with a profile, so with entries alone the model wrote with the earlier values and said so
after the write: the approval was shown without a word of what had been filled in, and a
person who denied it never learned of it. The part of the note on the values of an earlier
request (`agent.EARLIER_VALUES`) now comes with memory entries too; the note sent with a
profile is the same text as before. In the asking benchmark ("Create a low-stock alert for
SKU-00031 too", the earlier alert an entry): asked first in 0 of 3 runs before, 3 of 3 after;
the 76 cases that were there, whose notes did not change, 218 and 221 of 228 over three runs
each. Three more cases of the same kind already passed and are kept (a costly reorder, and
the item itself left out, as a profile line and as an entry).

### Fixed: what a ReAct run is told and sent in the cases a closing run still got wrong

One change set, each part with its cause, a rule that knows no case, and generated tests
that read the words the model is sent.

- **Handing a task to a helper asks no one.** `task` was left out of the run's tool set as
  machinery when the agent has no sub-agents of its own, so its own spec, which declares
  nothing, was read, and a tool that says nothing asks a person: every task handed to the
  general-purpose helper paused for an approval of nothing, on every backend. It carries
  the helper rule's risk now (a read); each call the helper makes is governed on its own, by
  its own name. 8 of 16 generated cells (4 backends, 2 helpers, 2 risks) failed before.
- **The reply a call will send is reserved out of the model's own window.** `FitWindow` read
  `max_tokens` from the request's model; for a model given by name that is the routed
  wrapper, which has none, so nothing was reserved: 950,000 tokens could be sent where the
  provider takes 872,000. The reply is the call's own setting, else the `max_tokens` of the
  chat model that sends the call. It is taken out of the model's own window, which `ReAct`
  keeps beside a `context_window` given: given 32,000 for a model replying up to 64,000,
  the budget was zero. 51 of 60 generated models failed before.
- **A tool call cut at the output limit is said to be cut, and is made again.** The model
  was told an argument was missing, which it could not see was the limit's doing, and wrote
  the same long reply again; a call the provider could not parse at all was taken for a
  reply with no answer, and the model was made to answer with no tools. The refusal says the
  reply reached the output limit before the call's arguments were complete. The reply is a
  model call and counts as a step: a limit that did not count it would never end a run
  whose every reply is cut.
- **A file a command leaves at any depth is known to be the user's or not.** Only files
  directly in the working directory were looked at after a command, so one written a folder
  down (`reports/q3.csv`, or a folder inside `outputs/`) was neither, and an answer said it
  was delivered. Every file under the working directory is looked at (hidden folders left
  out, 5,000 at most), and a new one that is not directly in `outputs/` is named by its
  path. The command tool's description says what cannot be seen: a file written outside the
  working directory.
- **A window no one says is learned from the server's own refusal.** With no
  `context_window`, no profile in the library and a route whose model list names no window,
  nothing was fitted, and the first request over the window ended the run with the server's
  refusal. The refusal names the window in each server's wording the harness already reads
  (OpenAI's and vLLM's, Anthropic's, llama.cpp's): the request is fitted to it and sent
  again, and the window is the model's from then on. Clearing old results and the long-task
  tools stay off for such a model (they are set when the agent is built): give
  `context_window` to turn them on. No setting.
- **A page read of a result is cut as a page, never kept as a table.** A `read_file` page
  cut to fit a small window was kept as a result of its own, named as a table whose rows
  were the numbered lines ("2  SKU-00501"): a join against it matched nothing and the turn
  ran to its step limit. It is cut to its first whole lines and says where to read on in
  the file it read; nothing of it is kept.
- **After a summarization every result the run keeps is still named.** A summary is in the
  summarizer's own words and may name none of the kept results (0 of 56 in one run, whose
  model said it had lost track of them). The request after a summary names each result the
  run keeps that no remaining message names, by its table and its path, once.
- **Benchmark: a computed answer in a "cannot" case scores only against a stated figure.**
  A request written as cannot-be-done-here that a run computed all the same was scored
  right whatever it said. It is right only when the case states the figure and the answer
  holds it, and unverified where the case states none: counted neither way, in its own
  column. `cantdo-02` states what its line of Python prints, from the data.

### Fixed: a cleared result's mark names the result's own columns, and a mark is never cut as a result

Two changes of this release met badly. A cleared result's mark names the table a query
computes over (its columns), and a table of 5 rows or more read whole is kept, its message
ending with a note. The mark was built from the message's words, note and all: a JSON result
with a note after it is no JSON, so it was read as CSV and its "columns" were its data
(`columns: store_S20, category_knitwear, … planned_208, found_190`), about 2,250 characters
where the mark had 132. In a small window that long mark was then cut to fit like a result,
and the cut kept the mark's own text as the result to read on from: "the rest" of it was
the mark, and a query of that path found no rows.

- **A mark is told from the result's rows**, never from the message about it: the table the
  run already keeps, the one a query of that path reads. A cleared stub names the kept
  result's columns the same way; of a file that is Deep Agents' own, nothing is said that is
  not known.
- **A kept result's mark names the kept result's own path**, and a mark is never read as a
  result wherever the graph wrote it. Deep Agents' summarization writes the cleared request
  back as the graph's state and saves its messages as files, so the mark came to stand at
  the call's path: a query there read the mark's one line (15 failed queries in 3 live runs
  of `long-flow-02` with a backend, once the mark's columns were right and the model
  followed it). The kept path is read whatever the graph does with its messages; a result
  the run does not keep, whose only copy the graph replaced, is said to be gone.
- **A mark is no result.** A cleared result's line and a stub are shortened as marks when
  the window has no room (the path and the table's name stay, what is left out is counted);
  nothing of a mark is kept, and none is cut as rows are.
- **A result that begins with another's rows is itself.** A longer page or a wider date
  range starts with the rows of an earlier result of the same tool; it was taken for a copy
  of that one, and a count over 20 rows answered 10. A copy is the kept rows followed by
  their note, nothing else.

Seen in the closing acceptance run: 23 of 600 case runs carried such a mark (0 of 600
before), every run of `long-flow-02`, and one of them spent five of its twelve steps reading
marks. The generated tests now read the words the model is sent, not only what a query
returns: for a result of 1 to 9,000 rows as CSV, a JSON array or a JSON object, in each
state (shown, kept, cleared, cut, saved by Deep Agents, that stub cleared), the columns
named are the result's, the path and the table named give every row, and no request holds
a mark of a mark — on every backend. 25 of the 90 end-to-end situations failed before
(every cleared JSON result, and every result of a crowded step).

### Fixed: a model LangChain has no profile of replies up to its own output limit, and a cut answer says so

For a model newer than the installed LangChain package, `ChatAnthropic` falls back to 4,096
output tokens. A reply that thinks first then stops mid-call or mid-answer: 74 of 22,385
saved model calls ended at exactly 4,096 tokens, 41 of them a tool call cut before its
arguments, and one final answer went out as a success stopping at "| S20 | £".

- The route's record of the model names its output limit beside its window (Anthropic's
  `max_tokens`, a server's `max_output_tokens` or `max_completion_tokens`): asked once for
  both, with the same 5 seconds and the same kept answer, and used where LangChain knows
  none of the model and filled in its own fallback. A model LangChain knows keeps its
  profile's limit; a chat model object, or a call's own `max_tokens`, is the caller's; a
  route that names none leaves the library's fallback. No setting.
- The route's limit is used up to the share of the window the library gives the models it
  knows (its largest `max_output_tokens` of `max_input_tokens`, read from the profiles it is
  installed with: 64,000 of 200,000 today). Input and reply must fit the window together: a
  route reporting 128,000 of a 200,000 window would have left input 72,000, and one
  reporting a reply as large as its window none. The three models this applies to today
  report 128,000 of 1,000,000 and get it whole. A route that names no window to bound the
  limit by leaves the library's fallback.
- An answer whose reply reached the output limit is still the run's answer, as it is, and
  the run says so: the `output_limit` warning a reply cut before any answer already raised,
  on the stream and in the log. It is not continued.

### Fixed: a cleared result the run already keeps as a table, and a stubbed one, are queried by their call's path

A cleared result's mark names its call's path (`/large_tool_results/<call id>`) and the query
over its rows. Two results asked for by that path were still not read as their rows:

- **a table read whole** (30 rows or more: the run keeps it, and its message carries the
  note after the rows). The query was given the message as it stood, note and all: a JSON
  result became one row of thousands of columns, a CSV's figures became text
  (`sum(VARCHAR)`). It is now the table the run already keeps of it.
- **a result too large to show.** The run keeps it under a path of its own and shows a stub
  naming it; asked by the call's path, neither reader found it. Both now follow the stub to
  the kept file.

A generated test holds a result of 1 to 9,000 rows, as CSV or JSON, in each state — shown
whole, kept, cleared, saved by Deep Agents — and queries it by the path named for it: 36 of
its 144 situations failed before.

### Fixed: a held result is queryable wherever it lives; the query hands its rows to the sandbox; a file not kept is said at once

`ReAct(backend=sandbox())` answered differently from plain `ReAct` on the same question
(`long-data-03`: 105 SKUs under a 40% margin in 2 of 3 runs, where the exact query gives 104).
The cause was one cell of a wider gap, closed as four rules that hold for any task
([sandbox.md](docs/sandbox.md), [react.md](docs/frameworks/react.md)):

- **A result the run holds is queryable exactly, wherever it lives.** With a backend whose
  files are outside the graph's state (a sandbox, a disk, a store) the run now keeps a large
  result, as without a backend: Deep Agents kept the only copy where the harness cannot read
  it, so no table was named and `query_result` was not offered. A result cleared from the
  window, and one cut to fit a small window, now name their table too when they hold rows, and
  the query reads them (`middleware._paged`, `_readers`, `_kept_read`, `_rows_said`). A
  generated test crosses five backends with four ways a result is held: 13 of its 20 cells
  failed before.
- **The query hands its rows to the sandbox.** With a sandbox, a query's result of more than
  one row is kept by the run and is a CSV file there; the answer names it, and both tools'
  descriptions say which does what (the query: exact counts, sums, filters, rankings, joins;
  code: a chart, a formatted workbook, a script, starting from that file).
- **Same output with or without a sandbox.** Live, three runs: `long-data-03` went from 1 of
  3 right to 3 of 3 (104 each time, by `query_result`, no command); the other seven files and
  code cases stayed or became right every time.
- **A file for the user is kept, or the model knows at once that it was not, whichever tool
  wrote it.** A file written by its path (`write_file`, `sandbox_write`) outside the kept
  folder is told so by its path; a command that left a new file directly in the working
  directory is told that file is not the user's. The chart case kept its file in 3 of 3 runs
  (1 of 3 before). The model's command is told in Deep Agents' own entry for it
  (`SandboxBackend.aexecute_with_offload`), so a file tool's parsed output never carries the
  note. This costs one listing in the sandbox after each command of the model's.

### Fixed: at a small window a tool the question needed was dropped in silence

When a request does not fit, `FitWindow` narrows the tools. Its last step kept only the tools
the run had already used whenever the latest turn alone filled the window, so a tool the model
had not called yet was gone when it reached for it, and nothing said so: on `long-multi-02` at
8,000 tokens the model answered "none of my tools hold supplier delivery data" in 4 of 6 runs,
of a tool that had been dropped a step before to save 89 tokens of a request still 2,000 over.
The rule now, at every window size:

- a tool the narrowing does not keep is still offered by its **signature** (its name and
  arguments, the description left out) while that fits — always when the tool hints chose it,
  and for the first 8 besides the kept ones whatever the room. Only a request the model's
  server itself refused gives those up;
- a tool held back altogether is **named** after the instructions, with what finds one
  (`tool_search`, where the run has it, which is never held itself). The line's room is set
  aside, so a fitted request still fits.

### Fixed: a `ReAct` sub-agent that names no tools has the run's tools

Deep Agents gives a sub-agent that names no `tools` the main agent's. `ReAct`'s tools are not
built into its graph: they are offered per model call (`HarnessTools`), so such a sub-agent,
and Deep Agents' general-purpose helper, was offered the seven file tools only. A helper asked
for a stock level searched files for it and handed the question back, and the parent made every
call itself. A declarative sub-agent that names no `tools`, and the general-purpose helper, are
now given the run's tools, governed as the parent's are (`react._native`); one that names its
own `tools` has exactly those. Over three live runs each, the two sub-agent cases stayed right
3 of 3: `advflow-02` went from 67, 39 and 68 model calls to 42 each time (24 s to about 16 s),
`advflow-03` from 13, 7 and 10 to 9 each time (about 13 s to about 10 s).

### Changed: a handful of rows is a table too, and two figures are computed by the query

From two 8-row summaries a model divided £1,123,571.70 by 11,978 units by hand and gave £93.79
for £93.80 (`planning-03`: wrong in 4 of 9 runs, and in some run of every arm): a table under 30
rows was read as it was, with no `query_result` to compute with.

- **A table of 5 rows or more that is read whole is kept as a table** (`results.TABLE_ROWS`,
  was 30), so the model is offered `query_result` and told to compute with it.
- **Its note names dividing, and the query runs on `sql` alone** (`results.QUERY`,
  `tools.query`): `SELECT 1250.50 / 37` computes with the figures it names, no file
  needed (before, a query without `file_path` or `results` was refused).
- Measured with example figures that are no case's (`SELECT 1250.50 / 37`): `planning-03`
  right in 8 of 9 runs (5 of 9 before) and `long-flow-02` in 5 of 6 (4 of 6). Every wrong run
  is one where the model answered without calling the query; every run that called it is
  right. An earlier 9 of 9 was measured with the case's own division as the example and does
  not count. It costs a model call where the model now computes.

### Fixed: with memory off, an out-of-date value of the turn before was reused

With memory off, `ReAct` replays the conversation's earlier turns from the run store, each
result of a live tool marked out of date. Memory's replay also says so in the run's context,
with the question ("for a value the question needs, call its tool again; never answer with an
earlier one": `snapshots.note`); the run store's replay did not (`Agent.recalled` dropped the
list of out-of-date tools). The mark sits on the old result, and a model asked for a new figure
that only uses the old one as an input took it as it stood: on `freshness-03` the on-hand count
of the turn before went into the next turn's reorder formula in 3 of 3 runs, and on
`freshness-04` in 3 of 3. The note is now said with the question on that path too, and only
when a replayed result is out of date: the four freshness cases went from 20 of 27 turns right
to 26 of 27 over three runs.

### Changed: `ReAct` alone switches to Deep Agents' machinery by itself; asked for, it is Deep Agents from the first step

One `ReAct()`, two ways to run it, said by what it already takes
([react.md](docs/frameworks/react.md#two-ways-to-run-it)):

- **`ReAct` alone** is Deep Agents' graph underneath with its machinery held: an ordinary task
  is sent what `create_agent` with `middleware()` is sent (the same tools and instructions, a
  large result kept by the run, the window fitted by `FitWindow`). Once the conversation is
  past half the model's window — the run's own behaviour, where old tool results start being
  cleared — the model is offered planning (`write_todos`), Deep Agents' scratch file tools and
  its general-purpose `task` helper, for the rest of the thread.
- **`ReAct` asked for a Deep Agents capability** (`backend=`, `subagents=`, `skills=`,
  `memory_files=`, `permissions=`) offers Deep Agents' tools, planning and `read_file` from the
  first call (before: its scratch tools and helper came with the third model call), unless the
  window is known to be under 32,000 tokens.
- **The switch no longer reads the request's wording or counts model calls.** "Step by step",
  "a plan" and a list of three items matched no request of the long set (0 of 45 turns) and one
  of the benchmark's 256 cases, which asked for no steps. The switch after three model calls
  changed the tools in the middle of a run, so a provider's prompt cache was rewritten: on the
  12 pilot cases cached input fell from 456,000 to 337,000 tokens and the cost rose a third,
  and on 57 long and complex cases it fired in 26 runs and no model call used a tool it
  offered. `automatic.LONG_AFTER`, `STEP_ITEMS`, `calls_made` and `asks_for_steps` are removed;
  `automatic.long_task(request, window)` is the conversation's size against `LONG_AT`.

### Fixed: kept tables of one tool's calls made side by side

- **Each table says what its call asked** (`middleware._table`): two results of one tool
  called in one reply were numbered by which answered first, so the model took
  `sales_summary_2` for its second call and computed a profit from swapped columns (the priced
  pilot's `aggregate-04`, on three of four frameworks: a wrong result, one step more over
  80,000 tokens). They are now `sales_summary_revenue` and `sales_summary_units`: the tool, and
  the arguments that tell the reply's calls of it apart. Calls one after another are numbered
  as before.
- **Deep Agents' stub of a result it saved names its table** (`middleware._stubs_tabled`):
  `query_result` reads the saved file as a table, but Deep Agents' own stub does not say which,
  and a model took `result` for it in another file's query (`long-data-02` on `ReAct` with a
  backend: a failed query, a step more). The request's copy of the stub now says the table and
  its columns before the preview, as the harness's own stubs do.

### Fixed: a model LangChain has no profile of had no context window

A model name's window came only from the table of profiles bundled in the installed LangChain
package. A model released after that package, a local model, or any name on a provider other
than OpenAI and Anthropic had none: `ReAct` ran with clearing old tool results and the
long-task tools off, `FitWindow` stepped aside, and the memory context's and the replayed
conversation's budgets were the fixed ones (the priced pilot ran every arm so: the routed
model was newer than the table).

- **The route is asked when the table names no window** (`ModelGateway.window`,
  `gateways.routed_chat_model`): Anthropic's Models API through Anthropic's own client
  (`max_input_tokens`), an OpenAI-compatible endpoint's `/models` (the field its server names:
  `max_input_tokens`, `context_length`, `context_window`, `max_model_len`,
  `max_context_length`), with the route's auth, within 5 seconds, once per route and model.
  No table of windows is kept here.
- **A route that says none leaves the window unknown**, as before (said once): a model list
  without the model or without a window, an endpoint that does not answer (asked again by the
  next build). `context_window=` wins over all of it.

### Fixed: ReAct on Deep Agents on a small window

On a 6,000-token window (adversarial `advconv-05`, `react --variant deep`, memory off) the
summary requests Deep Agents sent were up to 23,156 tokens, 13 of 23 over the window; after a
summary the model answered the conversation's first question again in place of the run's. Five
causes, all the harness's; Deep Agents' summarization is left as it is:

- **The earlier turns replayed with memory off are sized for the request**
  (`conversation.earlier(budget=)`, `Agent.conversation_budget`), as memory's window is: each
  turn's question and answer from the newest back, then the newest turns whole while they fit.
  Before, the newest 10 turns went whole whatever the window, and Deep Agents summarized all
  of them in one request each turn (on `create_agent`, `FitWindow` left them out).
- **The task kept ahead of a summary is the run's question** (`ModelHooks`), not the first
  user message of the state, which with earlier turns replayed was the oldest turn's.
- **The readers of a kept result stay offered when `FitWindow` narrows the tools**
  (`query_result`, `read_file`): the preview says to call them.
- **A request refused after a summarization is fitted again with the latest turn's results
  cut** (Deep Agents keeps a `read_file` page whole): before, the same request was summarized
  again and refused again.
- **The model's tokens per counted token are learned where it is called**, of the request it
  was sent (`ModelHooks`), not of the request before a summarization made it smaller.

### Fixed: a kept table queried under its note's name reads its own rows

A kept table's note names the table to query (`FROM sku_catalog_2`). Inside a query, the
results a Deep Agents graph saved in its state joined the query and were named first (a
`ChainMap` iterates its last map first), so they took that name: the footwear note's table
read another turn's denim rows (ReAct on Deep Agents, adversarial `advconv-05`; the
`deepagents` adapter too). `query.tables` now names the conversation's earlier turns' results
first, then the run's own, then the graph's, which join only while a query runs. Adapters with
no graph files (OpenAI Agents, Claude, Kimi, `create_agent`) keep the names they had.

And a kept table `FitWindow` cuts to fit the window keeps its note (where it is kept, the table
to query) whole after the cut, its rest read from the same kept copy: the note was at the end
and was cut away, so the model guessed `FROM sku_catalog` and read the earlier turn's knitwear
rows for outerwear (`deepagents` adapter, 6,000 tokens).

### Added: the model is told what day it is

Memories, fetch times and earlier conversations carry dates, and nothing told the model the
day to measure them against (only `schedule_task`'s description did). The harness now says it
in the instructions, one line after them: `It is now Wed 2026-11-11 (UTC; the user's zone is
not known).`, in the user's zone when their app sends one. To the day and from the run's start,
so every model call of a run reads the same line and a prompt cache loses its prefix once a
day at most; the date only, no instruction with it.

- Where: `ReAct`, a `create_agent` or `create_deep_agent` graph (with the harness's middleware
  or without), and OpenAI Agents (after each agent's instructions). Claude Code and Kimi Code
  say the date themselves, so the harness adds none there. A hand-built `StateGraph` and a
  function call their model in their own code and get none.
- One wording of when it is (`snapshots.said`): `schedule_task` keeps its day and hour, from
  the same function. A run's start and its seconds read the one clock (`snapshots.now`).

### Fixed: a failed `query_result` names the closest tables, and stays short

A failed query's error listed every table the run keeps with its columns. With twenty saved
results it was about 1,000 characters, `FitWindow` cut its middle on a small window, and the
table the query meant went with it. The error now names the five tables closest to those the
query reads (DuckDB's own `get_table_names`, ranked with `difflib`; `result` is the queried
file's), each with its columns, then how many more there are and the query that lists them all
(`SELECT table_name FROM information_schema.tables`): about 500 characters however many
results the run keeps.

### Fixed: only a call that returned a value is marked out of date

A live tool's result from an earlier turn is marked out of date, so the model fetches the value
again. The mark went on every earlier tool message of the tool, also one of a call with no
result: in a thread a graph keeps, a call a newer message stopped read "[slow_count's result
from an earlier turn: out of date ... call slow_count again for the current value]" before
"slow_count has no result: its run stopped ...", and a failed call's error the same; in
memory's replay a failed call's error was marked too, and its tool counted among those whose
values are out of date. A call that failed or was stopped returned no value: its result is
sent as it was (`middleware._earlier`: the tool message's `status`; `snapshots.out_of_date`:
the run record's status).

### Added: "Always allow", kept in the memory service

With a memory service, a write's approval offers a fourth choice, **always allow** (`always`,
`remember="always"`; trellis-contracts 0.7.1): the tool's later calls asked of the same person
or role run without asking in every conversation of that user until the permission is revoked.

- Kept in the memory service's tool permissions (`PUT /v1/tool-permissions`, trellis-memory
  0.6.5) with the harness's service key, once, in the background (spooled when the service is
  down); read once per attempt when a call first asks, so a revoke counts from the next run.
- Never for an `irreversible` tool (neither offered nor honoured), nor where the tool's
  `approve_when` or a `before_tool` hook's `Ask` decides the call; a framework's own tools keep
  their own flow.
- Offered only where a pick is kept: an attempt that could not read the permissions, or a key
  that cannot grant (an issuer's token), offers the three choices. Permissions that cannot be
  read fail closed: the call asks, and a warning is logged once.
- **A hook's `Ask` beats every allowance:** a `before_tool` hook's question is asked whatever
  was allowed for this conversation, always, or by the run's schedule (as a PreToolUse "ask"
  in Claude Code). Before, an approval for the conversation or a schedule's covered it.
- Without a memory service nothing changes: three choices.

### Changed: a write asks, once a conversation, as in Claude Code

A tool the harness runs that says it writes now asks the person before it runs (it ran,
announced). The person picks, as in Claude Code: **allow once**, **allow for this
conversation** or **deny**. An `irreversible` tool offers allow once and deny only: it asks at
every call.

- **"For this conversation"** covers the tool's later calls in every later run of the same
  thread for the same user — a resumed run, another worker, a child run — never another
  thread, user or tool. It is read from the run store's history of answered interrupts
  (agent-runs' `resolutions`; `LocalRuns` keeps the same): no new setting, no contract change.
  `remember="run"` is that choice; before, it reached the rest of the run only, and covered an
  `irreversible` tool too.
- **The choices travel with the approval**: its interrupt's `options` (`once`, `run`,
  `reject`), and the question says them for plain chat. AG-UI takes the picked value as the
  resume's `payload`, A2A the choice in words, `agent.resume(id, "answer", answer="run")` too.
- **What does not ask:** a write the run keeps to itself — the schedule tools, the memory
  tools, the sandbox, a sub-agent's call, Deep Agents' files and `execute` — is announced as
  before. A framework's own tools (Claude Code's and Kimi's built-ins, OpenAI hosted tools)
  keep their own permission flow. The catalog's `risk` and `approve_when` are unchanged: a rule
  that does not hold still lets the call run, announced.
- **A schedule set in chat** names the writes its runs will call (`schedule_task(tools=)`);
  its confirmation allows them for its runs (kept in the schedule's metadata) or leaves them to
  ask at each run. A write it does not cover pauses in the chat's thread. A schedule made from
  code allows nothing.
- An OpenAI agent holds a write at its own gate (`needs_approval`), so the approved call
  continues the paused run.
- A call denied earlier in a run stays denied when a later approval in the run allows its tool.
- Way 2: `governed(fn, gov, side_effects="write")` and `Governance.check` ask about a write
  too (`on_ask`).

**Upgrading.** Nothing fails at startup. What changes: a write tool now asks once per
conversation.

| What you see | The fix |
|---|---|
| a `write` tool that ran now pauses (`Approve <tool>? <tool> is write. Allow once, allow for this conversation, or deny.`) | nothing: the person allows it once or for the conversation; an app renders the interrupt's `options` |
| an `irreversible` tool approved with `remember="run"` asks again | nothing: it asks at every call |
| a scheduled run's write pauses | name the tool in `schedule_task(tools=)` so the person allows it when confirming; a schedule from code: answer the pause |
| a tool that only reads is declared `write` | `@tool(side_effects="read")`, or its `risk` in the catalog |

### Changed: a tool that does not say what it does asks a person

A risky call must never run without approval. A tool that declared nothing was taken as
`write` and ran, only announced: in a live test a purge, a bulk price change and a read of
another account's data all ran. Now, as in Claude, ChatGPT or MCP, a tool that says nothing
needs the person's permission — one behaviour for every way a tool comes in. Nothing is
refused: no tool fails at startup.

- **A tool with no declared side effects asks at each call** (`<tool> has not declared its
  side effects.`): `@tool` alone, `tool(fn)`, a bare function in `tools=[...]` or given to
  `agent.tools.use`, `governed(fn, gov)`, a source of your own, a LangChain tool of your graph
  with no `metadata["side_effects"]`, an OpenAI Agents `FunctionTool`, a built-in of Claude
  Code or Kimi the harness has no word on, `Governance.check(tool, args)` given none. Its risk
  is none (`unknown`), and governance asks about a risk it does not know (it announced and
  ran). Declared tools are as before: `read` runs, `write` runs and is announced,
  `irreversible` asks.
- **An MCP tool's annotations are read with the protocol's defaults** (`readOnlyHint` false,
  `destructiveHint` true): a tool is `irreversible` unless it says it only reads (`read`) or
  says `destructiveHint: false` (`write`). A tool with no annotations is `irreversible`: each
  call waits for approval. Before, anything not marked destructive was `write`.
- **The tools that say nothing are named once**: when an agent's tools are listed, one warning
  (`trellis.tools`) lists those that declare no side effects, with what to add.
- The catalog is told the same (`destructiveHint: true`, never a `side_effects` of the
  harness's) for an MCP tool that says nothing, so an administrator's risk there stays: it is
  the place to say what a tool you cannot annotate does. Any other tool that declares nothing
  is published saying nothing of it: the catalog keeps no risk (`null`, trellis-memory 0.6.6)
  and the tool's own rule decides, so it asks with allow once, for this conversation, deny
  (always allow with a memory service). A catalog entry with no risk leaves the tool's own
  declared risk; before, it was read as one (`write`, then `irreversible`), and an undeclared
  tool asked every call with no "for this conversation".
- A built-in the harness knows only by name (the graph's state tools, Deep Agents' file tools
  and `execute`, declared a write) is the framework's own: a tool of yours that shares the name
  (`delete`, `write_file`...) says what it does itself, as any other.

**Upgrading.** Nothing fails at startup. What changes: a tool that declared nothing now pauses.

| What you see | The fix |
|---|---|
| a tool of yours that ran now pauses (`<tool> has not declared its side effects.`), and a warning names it | `@tool(side_effects="read")` (or `"write"`, `"irreversible"`) on the function; `tool(fn, side_effects=...)` for one given bare; `governed(fn, gov, side_effects=...)` |
| an MCP tool that ran now pauses (`<server>-<tool> is irreversible.`) | your server: annotate the tool (`readOnlyHint: true`, or `destructiveHint: false` for a change that can be undone); a server you do not own: its risk in the catalog, `put_catalog([{"name": "<server>-<tool>", "side_effects": "read"}])` |
| a framework's own tool that ran now pauses | a LangChain tool: `tool.metadata = {"side_effects": "read"}`; any other: its risk in the catalog (as above), or build the tool with the harness (`@tool(side_effects=...)`) |

### `query_result` over complex data

- JSON rows held at any depth, each array an object holds (`result_<key>`), JSON Lines, markdown
  tables, a one-column CSV, a quoted delimiter in a header, column names in any script.
- `NULL`, `N/A`, `-` among numbers are missing values; times with a zone are compared and
  written in UTC; money and percentages stay text, and a failed sum says how to convert them.
- A PDF, an image or binary data is refused, pointing to the sandbox; HTML or XML is queried as
  its lines, the answer saying so. A workbook is a table a sheet, under its header, merged
  cells filled.
- A text's typed tables are kept as Parquet by its content: the next query of it (this run, a
  later turn, a past result) reads them, not the text. Five queries of a 10 MB result: 3.1 s →
  0.8 s (CSV), 5.0 s → 1.3 s (JSON).

### A journal is read from its segments only

- A run's checkpoint lists its journal's segments (`trellis_segments`); that is the only
  format. A checkpoint holding the journal inline, or naming it as a run artifact
  (`journal_ref`), is refused (`HarnessError`). No migration: dev databases are reset.
  `LocalRuns` keeps the same format in its folder.

### Past results by reference

- `query_result(results=["msg_..."])` loads past tool results (the ids an earlier
  conversation's steps name in the memory context) as `t1`, `t2`… beside the run's own, read
  through the memory service's existing message read and its audience rules; a new chat is
  offered `query_result` once its context names one. "Which tool did you use last time?" is
  answered from those steps. Live history eval: `explain-08` and `hist-long-01` now pass.

### `query_result` needs DuckDB

- DuckDB is a required dependency; the SQLite fallback is gone, so the model sees one dialect
  and the query suites run once ([ADR 0007](docs/adr/0007-query-engine.md)). The `query` extra
  stays, empty. `trellis doctor` fails its "query engine" check when `duckdb` cannot be imported.

### Benchmark

- `--scripted` drives Kimi in the complex suite (`--adapter kimi`): the scripted model is served
  as a chat completions endpoint Kimi calls, keyed by the message it answers (a resumed Kimi is
  told to repeat the paused call, which a script in order answered with the next turn), and
  every request Kimi sent is kept, so `Turn.sent` is checked on Kimi as on the LangChain
  adapters. Before, a scripted Kimi run hung (Kimi was given the scripted model's text as a
  model name). Adversarial and break-and-fix: 58 and 70 cases, every strict score right, 1 of
  400 checks failed (`advconv-05`'s `fits 6000`: Kimi keeps its own context, larger than the
  case's window). Kimi is given the suite's step limit in its own slot
  (`KIMI_LOOP_MAX_STEPS_PER_TURN`), as the other adapters are.
- With a real model the Claude and Kimi CLIs call it themselves: what they send is not seen, so
  the checks of what was sent (`has`, `lacks`, `nonblank`, `fits`, `repeats`, `answered`) are
  not made for them. Before, a `has` failed and a `lacks` passed on nothing (Kimi, Haiku,
  adversarial: 48 of 52 failed checks were these).
- A history set for the complex suite (`--cases history`): earlier chats recalled, compared,
  trended, merged and explained, and how an answer was got (memory or tool, source and read
  time, after a resume); 10 new cases and 12 existing ones of those kinds.

### Credentials for tools

- `Harness(credentials=async (run, target) -> headers | None)`: the run's user's headers on each
  call of an `mcp(...)` server over HTTP, an `openapi(...)` operation and an `a2a(...)` agent,
  over their fixed `headers=`, asked once per run and target (a scheduled run asks again);
  `NeedsAuth(url)` pauses with "Connect your <name>"; a failing hook fails that call. With a
  hook, MCP connections are per server, tenant, user and headers, closed when idle. The token is
  on the request only. No hook: unchanged (`trellis.harness.credentials`, docs/tools.md).

### Schedules that run unattended

- Chat schedules pause themselves after 3 failed runs in a row (`ERROR`, `TIMEOUT`); that
  run's error says so in the thread, its record and its `run.finished` webhook. One untouched
  for 90 days asks at its next run "Still need this?" (Keep, recommended, or Cancel), paused
  until answered. No settings; `trellis_failures` and `trellis_touched` in the metadata.
  agent-runs #26 adds one run at a time and a stable per-schedule jitter. Docs: a user who left
  (`on_run_start`); `benchmark.load.threads` (1,000 thread streams, 100 fires).

### When and how to ask

- One note with every memory context, for every adapter (`agent.ASKING`): ask only when the
  model cannot find out itself, the choice is the user's and a wrong guess costs; otherwise act
  and say the assumption, or the remembered rule and its date; values filled from an earlier
  one are said, or asked about first when costly, old or in conflict; a superseded rule is asked
  about. The app's own tool when one fits, else the question tool, else the reply.
- `ask_user(..., multiple=False, recommended=None, why=None)`: several picks, one option
  recommended (`Option.recommended`, trellis-contracts 0.6.5) only with a why; a choice also
  takes another answer in words; plain chat reads the options numbered.
- Claude's `AskUserQuestion` is the run's question, as Kimi's is.
- A question's recommended option comes first wherever the model listed it: in the choice,
  the app's own control and the numbered text.
- The note asks "which one" when several things fit the user's words (it had made the model
  pick one or answer for each, against the app's own "ask which one is meant": break-and-fix
  clarify 17/21 → 21/21); how to use remembered rules, preferences and values
  (`agent.REMEMBERED`, said with their date) comes only with a context that has a profile, next
  to the asking note: no tokens on the other turns, and the dated cases 28/36 → 36/36.
- The schedule tools keep only what is theirs: the general asking words are the note's; the
  "update it" choice is recommended by the flag, not by its label.
- `schedule_task(in_minutes=)` says the minute it counted from (the tool's text says only the
  hour, for the cache): the model no longer "corrects" 03:53 for "in 2 hours" at 01:53 to 03:00.
- `ask_user` is offered by itself to a run a person is known to read: one a chat front end
  started (`serve_chat`; `Runtime.attended`), not one its schedule fired, on a framework with
  no question tool of its own (Claude Code and Kimi Code keep theirs: `Adapter.asks`), never
  on a graph with its tools bound. `without=["ask"]` turns it off (an eighth name). A run
  started from code (`run`, `stream`), queued, fired, called over A2A or by another run gets
  it only when its agent declares it (`tools=[ask_user()]`, as before): nobody is known to
  answer a pause there. A tool of the agent's named `ask_user` wins; one declared is offered
  once. `serve_chat` records a run's front end even when the app sent nothing with the
  message.
- `ask_user(fields=[{name, label, kind, options, multiple, default, why, required}])`: several
  values in one question, one answer (an object: what the person gave over what was filled
  in). No contract change: the `ask_user` component with `props.kind == "form"`, an object
  `expects` with `ui_schema`, a numbered list in plain chat. A single question is the
  one-field case; `recommended` with `why` on a number or a date shows the value filled in.
- Values a request leaves out that an earlier one had are confirmed once before they are used
  again, shown filled in (`agent.REMEMBERED`): they were filled and said. A rule, preference
  or value said more than six months ago, or one a tool shows changed, is confirmed the same
  way.
- Where the app declared tools of its own, `ask_user` says a value one of them asks for is
  asked with it (the app's date picker used in 1 of 8 runs before, 13 of 14 after).
- The asking benchmark (`benchmark/cost/asking.py`): `--ask auto|declared|none` (was
  `--no-ask-tool`), `--variant`; cases for an ambiguous target (14 products, two), a
  remembered value that is partial, old or changed elsewhere, a recommendation the data
  supports, an answer in words or outside the options, several values missing at once, a run
  nobody reads, and risky actions (cannot be undone, money, many rows, people outside, an
  instruction in a tool's result, an earlier yes to something else): each waits on its
  approval whatever the model does; a tool that declares no side effects is measured apart.

### Schedules set in chat

- Every agent has `schedule_task`, `list_schedules`, `update_schedule` and `cancel_schedule`
  (`without=["schedules"]` turns them off; `h.schedule_tools()` for a graph built with its
  tools bound): "every Monday at 8 check stock and tell me if it's under 5". A fired run is
  the same agent, user and thread, so its reply is saved into that chat; with a condition it
  answers `NO_REPORT` when the condition does not hold, and a condition that still holds is
  reported once. Limits: `TRELLIS_SCHEDULE_MIN_INTERVAL` (300 s), `TRELLIS_SCHEDULE_MAX_PER_USER`
  (20), per user `Harness(schedule_limits=)` ([docs/runs.md](docs/runs.md#schedules-users-set-in-chat)).
- Schedules are in the user's zone without asking it: AG-UI `forwardedProps.timezone` (the
  browser's), else UTC; `schedule_task` says the day and hour in it, so "tomorrow at 9" has a
  date, and takes `in_minutes` ("in 2 hours"). It is told when to schedule and when not, to
  check a condition once now and say its value, to say what its tools cannot watch and offer
  what they can; a fired run is offered no schedule tool (one made a new schedule from "remind
  me daily"). The tools ask with buttons (`Runtime.ask(options=)`) for a schedule like one the
  user has (update it, recommended, keep both, cancel; the same again is "already set"),
  `cancel_schedule(id="all")`, several that fit (`ids`) and a vague ask (`choices`). The eval
  set: `python -m benchmark.cost.schedules` (61 cases).
- AG-UI `GET {path}/threads/{thread_id}/events`: every run of a chat thread, scheduled ones
  included, live and after `Last-Event-ID`, fed by one poller per process.
- `RunStore.schedules` gains `list`, `get`, `update` and `delete` (`LocalRuns` has them); a
  `LocalRuns` fire takes the thread the schedule's metadata names. **Requires** agent-runs with
  `RunSummary.thread_id` and `schedules.list(on_behalf_of=)`.

### Docs

- Docs name the arguments the code takes: a run's time limit is `run_timeout=` (scenarios.md,
  runs.md), Langfuse's address is `LANGFUSE_BASE_URL` (README), the governance example binds a
  tenant id it names (there is no `h.tenant()`), and blocks/evaluation.md no longer points at the
  retired `eval_sample` setting.
- reliability.md: a scheduled run carries its time limit and agent version (agent-runs copies
  them into each fired run), and a `ReAct` model client makes one attempt only on the `bifrost`
  route; on the other routes the client keeps its own default retries.

### Background agents moved to their own branch

- Background sub-agents (ADR 0008), a run's inbox and a person's pause (ADR 0009) and the
  transcript of a run or sub-agent were reverted from `main` (#160, #166, #167) and continue on
  the `background-agents` branch, with the matching agent-runs and agent-contracts work, until
  they are ready to land together. `main` keeps every generic fix merged meanwhile (#162, #165,
  #168) and requires `trellis-contracts>=0.6.3,<0.8`: 0.7.0, the contracts without the run inbox,
  is admitted.

### Claude Code and Kimi Code ask as they do natively

- A `Bash` command that only reads (`ls`, `cat`, `wc`, `which`, Python that opens files to
  read) runs without a pause, as their `Read`, `Glob` and `Grep` do: the read-only check moved
  from the benchmark into `trellis.harness.governance.shell` (one implementation).
- Any other call follows the framework's own permission check, then a person: Claude Code's
  `PreToolUse` hook applies your hooks and governance, then leaves the call to Claude Code's
  mode and rules (`allowed_tools`, its settings: your allowlist, as it is); what it would ask
  its user, its `can_use_tool` asks the harness (your own callback, else a person). Kimi
  asks over ACP: after hooks and governance, a person is asked, as Kimi asked (`Write` and
  `Edit` outside a git working tree included, where they ran announced before).
- A file Claude Code or Kimi Code writes in `outputs/` of its working folder is the user's, as
  a sandbox's `outputs/` is (`sandbox.delivered`, `Adapter.workdir`): kept in the file store on
  success, announced with the `file` event, then removed from the folder. Each is told so
  (`sandbox.DELIVERS_HERE`: Claude after its system prompt, Kimi before its prompt).
- `Bash`'s risk is `write` for both (it was `irreversible`: every command asked): a team that
  wants every shell command approved gives it the risk `irreversible`, or an `approve_when`,
  in the tool catalog. `bridge.asked` asks a person about a call its framework asks about.

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

- The context's line about earlier tool values that are out of date now asks the model to
  call a tool again only for a value the question needs, and still never to answer with an
  earlier one (`snapshots.NOTE`). It said "call them again; never answer from the earlier
  ones", and on a turn that needed none of them ("Put that in a PDF for finance") ReAct
  re-fetched every input and often dropped why it could not make the file. Haiku, three
  repeats: that turn 1/3 -> 3/3 (4 of 10 across every ReAct run on main); the freshness
  cases 27/27 turns before and after.
- The tool hints narrow what the model is offered only from 30 tools in the run
  (`agent.NARROW_FROM`). Below that, every tool is offered and the hints' tools section
  still says which tools fit. Narrowing a set of 20 hid a tool the question needed (a name
  looked up before its id) in 22-26% of turns. Those turns were wrong 2-3 times as often, on
  every adapter.
- A stall no longer ends the run with no answer. A stall is the same call with the same
  arguments in `max_repeats` steps, or every call failing in 3 steps in a row. The model is
  now told so and called once more with no tool, as `StepLimit`'s last step is. The user
  gets an answer that says what went wrong, for example that the tool timed out. The run
  still stops (`ModelError`) when that call gives no answer.

### Fixed

- With a sandbox, a kept result's rows are `query_result`'s to count, sum, filter, rank and
  group, and code is for what a query cannot make. The stub's code note gave the same job to
  the sandbox ("to count, sum, filter, rank or group its rows, run code over that file"), so a
  model computed in its own Python: 105 SKUs under a 40% margin where every query said 104
  (`ReAct` with a sandbox, `long-data-03`). The note now says the file is there for a chart, a
  formatted workbook or a script, read from the file and never typed into a command; a table
  read whole says so too (its file is in the sandbox as well: 1,528 values were typed into one
  command).
- `ReAct` given a Deep Agents capability has one owner of a small window. Under 32,000
  tokens (`automatic.MIN_WINDOW`: no room for Deep Agents' tools from the first step) its
  summarization is no longer placed either, and `FitWindow` fits the request as on `ReAct`
  alone. With both, the summarization saved a second copy of each table the run kept and spent
  model calls on summaries: the small-window cases were 6 of 8 right with a backend against
  8 of 8 alone. Haiku, a backend, after: `long-data-02` at 16,000 tokens right in 12 s (out
  of time at 420 s), `long-multi-02` at 8,000 right in 24 s (wrong, 125 s), the 14-turn chat
  at 6,000 right in 69 s (113 s). From 32,000 tokens up nothing changes.
- `query_result` no longer runs a run out of its time on a small window with a backend. A
  table read whole is kept and sent with its note; Deep Agents then saved that whole message
  as a file of its own, and the query loaded the copy as one more table: a JSON result
  followed by the note read as "CSV" whose header was the JSON line, 3,902 columns, and
  typing them in one statement ran for minutes with no interrupt stopping it (`ReAct` with a
  backend, a 16,000-token window, `long-data-02`: 420 s, the run's limit). A graph's copy of
  a result the run keeps is now that result (one table; asked by the copy's path, the kept
  rows), and typing counts 64 columns a statement with the load's time looked at between two,
  so `LOAD_SECONDS` holds for any table. The same case now ends in 69 s.

- A file meant for the user that a sandbox command wrote outside the kept folder is said at
  once. Only `outputs/` of the working directory is kept; a model copied its export to
  `/outputs/` at the root, nothing was kept, and its answer said the file was saved (`ReAct`
  with a sandbox, `bf2-mixed-04`). A command that names an `outputs` folder is now told with
  its result what the kept folder holds, with its whole path, or that it holds no file yet
  (`SandboxSource.outputs_said`: Deep Agents' `execute` and `sandbox_exec`); a command that
  names none costs nothing more.
- `FitWindow` gives the tools back once it leaves the older turns out. A long chat on a small
  window was fitted in order: the tools narrowed, to none past `FIT_KEEP`, then the older
  turns left out — and the tools stayed out, though the room the turns took was free. Every
  turn after that offered the model no tool (`create_agent` with `middleware()`, a 6,000-token
  window, Haiku, `advconv-05`: from the fourth turn on, 567 tokens a request, "I only have a
  file-reading tool"). The tools the narrowing dropped are now offered again as far as they
  fit: the same case on Haiku fetched the figure again on its last turn and passed every check
  (0 of 16 failed; 1 of 16 and the answer before).
- `FitWindow` leaves the older turns out before the tools a question needs, and starts the
  latest turn at the user's question. Narrowing kept 8 tools counting the ones it must keep,
  so on Deep Agents (8 state tools) it kept none of the question's; and once the results were
  cut the request fitted with the older turns kept and the catalog tool gone (the `deepagents`
  adapter, 6,000 tokens, `advconv-05`: 0 of 12 catalog questions answered). Now 8 are kept
  besides those it must keep, and the tools are narrowed past them only after the older turns
  are left out. A note the step limit or the budget appends to the call is no longer taken for
  the question: a request over the window kept the note alone, and the model answered that it
  had looked nothing up.
- `FitWindow` leaves the older turns out in layers: first their tool calls and results, the
  oldest turn's first and only as far as it must, each such turn then its question and answer
  (what memory's window gives of an older turn); whole only when those do not fit either. They
  went all at once: with the tools kept for the question (above), a 14-turn chat on a
  6,000-token window lost the first turn's answer with every other, where all 14 questions and
  answers were a few hundred tokens (the scripted `deepagents` run of `advconv-05`, its last
  turn: 54 messages cut to 2; now 52, the oldest turn's call and result left out).
- A LangGraph graph that keeps its own thread (a checkpointer: `create_agent`, with or
  without `middleware()`, a hand-built graph) can be continued after a run stopped mid-call
  (`mode="interrupt"`, a time limit) or left paused on a call while the user wrote again. The
  thread held the model's call with no result, and Anthropic refused every later turn of it
  (400: "`tool_use` ids were found without `tool_result` blocks"); the break-it set on Haiku,
  `advexp-01` and `advfile-02`, ended two turns each ERROR. The next turn now answers such a
  call first with a failed result saying it has none (`langgraph.STOPPED_CALL`); a Deep Agent
  answers it itself (its `PatchToolCallsMiddleware`) and is left to. The complex suite's
  `answered` probe checks it.
- OpenAI Agents: a write the person allowed for the conversation no longer asks again for the
  other calls of the same model reply. Calls governance holds at the SDK's gate
  (`needs_approval`) wait there together, and a resumed `RunState` keeps each one's pending
  approval without asking its gate again: the person was asked about every one of them, in the
  SDK's bare shape ("Approve <tool>?", no choices). Governance now decides each of them when
  the run resumes, one by one, after the answer to the one before: a call allowed for the
  conversation runs, and one that still asks (an `irreversible` tool) asks with governance's
  question and choices. Live (Haiku, 46 runs each): 14 runs asked again before — every run
  with parallel writes allowed for the conversation, streamed and after a handoff too — and 0
  after.
- A question a tool asks (`ask_user`, a schedule tool's "update it?", any `Runtime.ask` in a
  tool, Claude's `AskUserQuestion`) is asked once on every adapter (#203). OpenAI Agents and a
  graph with no checkpointer run a resumed run again from its input, and Claude and Kimi tell
  their model to repeat the paused call; a real model words that call differently, the journal
  knew a call only by its content, and the person was asked again on every resume. The call a
  pause came from within is now kept as the model made it, with its place among the run's
  tool calls, and the call of that tool made there when the run resumes runs as it
  (`Replay.made`): the answer is for the arguments the person saw. The model is told which
  call that is, with its arguments (`RESUMED_ASKED`: a resumed Claude or Kimi session, which
  was told only to repeat "the call with no result"; an OpenAI agent run again from its
  input, which was told nothing). Approvals stay exact. Haiku, the four rows of #203
  (`benchmark.cost.schedules`), runs that paused on a tool's question: Claude 7 of 7 right
  (#203: three of the rows failed), OpenAI Agents 6 of 7 (4 of 7 before its model was told).
- A Kimi call that asks permission carries its arguments when the model streamed them: Kimi
  asks before it starts the call, so its arguments were only in the call's streamed updates,
  which the adapter dropped, and a pause on `Bash` showed `{}` (no command) to the person and
  to governance. On Haiku, the break-and-fix suite could approve none of 8 such pauses.
- A framework's built-in tools no longer count toward the tool hints' narrowing threshold
  (`NARROW_FROM`, 30): only the run's own tools do (`Runtime.own_tools`). Claude Code's and
  Kimi's built-ins, and Deep Agents' `task` and file tools, are tagged `builtin`. Counted with
  them, a Claude or Kimi run of 14 tools of its own narrowed, and the tool a question needed
  (`find_sku`) was never listed for the query: 4 missed turns in the break-and-fix gate.
- A run never ends `SUCCESS` with no answer. A reply that reached the model's output limit
  before it answered (no text, no tool call) ended LangChain's loop, and the run succeeded
  with an empty answer: two LangGraph turns at 4,096 tokens in the break-and-fix gate. In
  `ReAct`, `middleware()` and any `create_agent` or Deep Agents graph the model is now asked
  once more, told so (`budget.CUT_OFF`) and offered no tool (an `output_limit` warning).
  Claude Code and Kimi recover such a reply themselves. A run that still ends with no answer,
  on any adapter but a function, ends `ERROR` `NO_ANSWER` saying why.
- Kimi no longer schedules prompts that never fire. `kimi acp` runs for one attempt and ends
  with it, so a `CronCreate` schedule could not run, yet Kimi scheduled it and answered as if
  it would (found by #121; the break-and-fix gate paused on one). Kimi's own switch,
  `KIMI_DISABLE_CRON=1`, is now set in its environment (`acp.KIMI_ENV`, whatever the agent's
  own `env` says), as Claude's cron tools are disallowed: the call says scheduling is off. The
  cron tools left the run's tool set (`KIMI_OUTLIVES_THE_TURN`). ACP cannot take a built-in off
  Kimi's list, so Kimi still lists them.
- A `create_agent` or Deep Agents graph wrapped without `middleware()` computes over large
  results too: its harness tools' large results are kept, and the model is offered `read_file`
  and `query_result` once the run keeps one, as with `middleware()`. Before, it was offered
  neither, and summed rows by hand: 5 analytics turns in the break-and-fix gate, two of them
  past the 4,096-token output limit. The harness's narrowing copy offers them as schemas, and
  the graph's tool node answers them. A hand-built `StateGraph` still binds its own tools.
- A helper's own time limit (`h.wrap(run_timeout=)`) holds when it runs as a child of a run
  with a longer limit: before, the parent's remaining time replaced it, so a helper wrapped
  with 0.2 s ran its full 2 s (found by the adversarial matrix row X09).
- A Kimi run looping on one call ends with one answer (ADV-9): Kimi's own repeat breaker
  (`KIMI_CODE_REPEAT_BREAKER`: the same call 12 times in a row) ended the turn with no answer,
  the reason only in its log (`stopReason` `repeat_breaker`). When a turn ends with no answer
  after a loop (the bridge's refusal, `bridge.stalled`, or Kimi's breaker on its own tools),
  Kimi is asked once more in the same session, as at its step limit, told the turn stalled
  (`bridge.TURN_STALLED`, for this turn only); a `stalled` warning says so.
- A Claude Agent SDK run at its `max_turns`, and a Kimi run at its step limit, end with one
  answer, as `ReAct` and OpenAI Agents do (#117). Claude's query ended `error_max_turns` (and the
  SDK raised the CLI's exit after it): its session is resumed once (Claude's own `resume`), told
  the limit is reached, for one turn with no tool. Kimi ended the turn with no answer, the
  reason only in its own log (`loop.max_steps_exceeded`): it is asked once more in the same
  session. A `step_limit` warning says so on the stream. Both are told the limit is for this
  turn only (`budget.TURN_LIMIT`): the session keeps the message, and told ReAct's "This run has
  reached its limit", a model answered the conversation's next turn "I've reached the
  tool-call limit" without calling a tool.
- When memory holds nothing for a question (or only part of it) and the run offers
  `memory_search`, the note that says so now also tells the model to search the user's
  earlier conversations first (`kinds=["message"]`) before saying a detail is missing
  (`agent.EARLIER_CHATS`). Told only to offer to look it up, every adapter asked the user
  again for what they had said in an earlier chat ("the cardigan in that same colour, but in
  cotton"). Haiku, ReAct, three repeats: that turn went from 0/3 to 3/3 right, and the turns
  the note reaches went from 15/18 to 18/18, at about half a model call more each.
- `ReAct` with memory off and a known window replays each earlier turn whole — its tool calls and
  their results, not only its question and answer — so a follow-up has the figures behind an
  earlier answer (the vendor read, the 503 behind an explanation); clearing and `FitWindow`
  fit them to the window, and a result past its tool's age is marked out of date as memory's
  window marks it. Kept per conversation in this process (ADV-7).
- A large result a conversation's turn saved stays readable in its later turns: a follow-up's
  `read_file` or `query_result` on the path its history names found "File not found", so every
  what-if needed the whole fetch again. Kept per conversation in the process that saved them
  (`Agent.results`, bounded); a path no longer kept says so and how to fetch it again (ADV-6).
- A worker that dies (a `BaseException` that is not an error) in a harness tool an ACP agent
  called ends the run's attempt, as an in-process tool's does. It ended only the MCP server's
  own task group, and the run went on as if the call had failed (the `X15-kimi-…` matrix cells,
  "the worker did not die").
- An ACP agent's own usage counts toward the run's budget. The adapter said it read ACP's
  `usage` on a prompt's answer, and its test asserted nothing: it was never read, so an ACP
  agent other than Kimi (whose usage log is read as the turn runs) spent with no count and no
  cap. It is counted now when the harness keeps no usage log of the agent.
- The Claude Agent SDK loads the harness's tools from the first step. Claude Code defers every
  MCP server's tools behind its `ToolSearch` unless the server's config says `alwaysLoad`, so
  each turn's model saw only the tool names and spent a step loading them (a model that called
  one unloaded got a validation error and called it again). The harness's `trellis` server now
  says `alwaysLoad`: which tools a run offers is already the harness's choice (the tool hints,
  memory's `tool_search`). A team's own servers keep the CLI's deferral.
- A Deep Agent of yours on a backend outside its state (a disk, a store, a sandbox) had its
  file tools run ungoverned: a `write_file` changed a real file with no journal, record or
  approval. They are governed now, reads and writes, as `ReAct(backend=)` has them
  (`middleware.declare_files`, marked when the graph is wrapped).
- An ACP agent (Kimi Code) gets the memory context marked off from the question, as `ReAct`'s
  is (`history.ASKED_WITH`, moved there from `middleware` so an adapter without LangChain uses
  it). ACP's prompt has no other place for it: the context went as a bare first block, and Kimi
  on a small model read it as the user's message ("your message doesn't include a question").
- A Claude Agent SDK run no longer offers Claude Code's schedules. Its CLI runs for one attempt
  and ends with it, so a scheduled prompt (`CronCreate`) or a wake-up (`ScheduleWakeup`) never
  fires; offered, Claude answered "set this up every Monday" with a schedule nothing kept. They
  join the query's `disallowed_tools` (`adapters.claude.OUTLIVES_THE_QUERY`), after the team's.
- The memory context no longer breaks a provider's prompt cache from one turn to the next.
  A `create_agent` graph (`ReAct`'s, Deep Agents', yours) sent the context, new for every
  question, as a system message ahead of the history, so the next turn shared nothing past
  the instructions with this one. The model now gets it as the first part of the turn's
  question, after the history; the graph's state is unchanged. With a prompt-caching
  middleware that marks the request (LangChain's `AnthropicPromptCachingMiddleware`), the
  answer ending the history is marked too (`middleware.history_marked`, audit K1). The
  context is marked off from the question (`middleware.ASKED_WITH`), so a question after a
  long context does not read as one more saved note.
- Anthropic's prompt cache is on for every Anthropic chat model `ReAct` is given, on
  LangChain's loop and on Deep Agents'. A model name (routed to the provider's own SDK) is not
  a `ChatAnthropic`, so Deep Agents' `AnthropicPromptCachingMiddleware` skipped it, and plain
  `ReAct` had no caching middleware at all: nothing was cached. `middleware.PromptCache` runs
  LangChain's middleware (one given to `ReAct(middleware=)`, its settings kept) on the chat
  model the call is made with, in Deep Agents' own one's place, and marks the answer that
  ends the history. Any other provider's request is unchanged. The small hosted model by name,
  18 multi-turn cases: cached share of input 0 → 0.87, uncached input −87%, cost −62%.
- Anthropic's API given as an OpenAI-compatible endpoint (`LLM_BASE_URL=https://api.anthropic.com/v1/`)
  is called with Anthropic's own client (`ChatAnthropic`, the same endpoint and key), so its
  prompt cache is on: the OpenAI compatibility layer drops prompt caching, as Anthropic's docs
  say. `LLM_GATEWAY=openai-compatible` keeps the OpenAI-compatible client; `trellis doctor`
  says which client is used.
- A live tool's value from an earlier turn reads as out of date in every adapter, not only
  where memory's window replays the conversation. A `create_agent` or Deep Agents graph keeps
  its own thread (its checkpointer), so its earlier results are marked in the request it sends
  (`snapshots.EARLIER`; the graph's state is unchanged). The OpenAI Agents memory session
  marks them as memory's window does. In both, the tool stays offered.
- A model looping on one call is stopped on every adapter: the tool bridge does not run the
  same call a third time in a row and tells the model to answer with what it has (a `stalled`
  warning); a graph without `middleware()`, a Deep Agent and the Claude CLI ran all 30 (ADV-9).
  OpenAI Agents offers no tool on the turn after (its `FunctionTool.is_enabled`), so the
  model's reply is the run's answer: it looped on to the SDK's `max_turns` (10) and answered
  about the limit, or ended `ERROR` when that last call made no answer.

- A run at its step limit ends with one answer: `ReAct` at `max_steps` (LangChain's
  `ModelCallLimitMiddleware` counting) and OpenAI Agents at `max_turns` (the SDK's own error
  handler) call the model once more, told the limit is reached and offered no tool, and its
  answer — what it found, what is left — is the run's: not LangChain's bare "Model call limits
  exceeded" nor `MaxTurnsExceeded` with no answer. One extra model call at most, through #76's
  last-answer path.
- A run's first scratch write (Deep Agents' `write_file` or `edit_file` on the graph's state)
  says the file is the run's own scratch copy, which the user cannot open or download
  (`middleware.SCRATCH_NOTE`). Asked for a chart, a model saved an SVG at `/tmp/…` in the
  graph's state and told the user to open it (break-and-fix round 1, Deep Agents).
- A conversation is no longer lost when memory fails mid-chat. Where memory replays the
  conversation (its window) and its context call fails (an outage, the circuit open), the
  earlier turns come from the run store, as with memory off; before, the model was sent the
  question alone and answered "this conversation has no earlier list" (break-and-fix round 1).
- `query_result` adds money up exactly, on both engines: a column of decimals as written (at most
  15 digits) is `DECIMAL` — DuckDB's own `DECIMAL(p, s)`; on SQLite a `REAL` whose `sum`,
  `total` and `avg` (a window's too) add up in Python's `decimal` — and a number in an answer
  is written as SQLite writes a `REAL` (15 significant digits). 6,000 two-decimal refunds
  summed to `32266.149999999998` (ADV-2).
- OpenAI Agents: a call to a tool the model made up is an error the model reads and the run goes
  on (the SDK's `tool_not_found_behavior="return_error_to_model"`), not a `ModelBehaviorError`
  that ended the run with no answer (ADV-8).
- A blank message (empty, or whitespace only) opens no run: `run`, `stream` and `start` raise
  `ConfigurationError`, AG-UI answers `422`, an A2A task ends `rejected`. It was sent to the
  model and kept in the thread, where a provider that refuses a blank user message (Anthropic)
  would fail every later turn (ADV-1).
- `StallGuard`: three steps in a row in which every call failed stop the run only when a failure
  among them repeats (the same tool, the same error). A model correcting itself — a wrong type,
  then a missing argument, then a made-up tool name, then the right call — goes on to its
  answer, bounded by `max_steps` (ADV-5).

- A kept large result's note says its path is the run's own working copy, which the user
  cannot open or download (`results.PRIVATE`). Asked for an Excel export, a model told the
  user the rows were "saved as CSV at /large_tool_results/…, open it in Excel".
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
- `query_result(file_path="/files/<run>/<name>")` loads a file kept for the user from the store
  as the table `result` (Excel too), beside the run's own results: one query compares an earlier
  file with today's data, and its answer says where each table's rows come from and when they
  were fetched.
- Files kept for the user are read back by later runs, within the run's tenant and user: each
  file has a record beside it (`.<name>.json`: the run, what made it, the query and the results
  it read with their fetch times, the run's tool calls); a run of a chat with files is told of
  them as it starts (one line each); `read_file` reads `/files/<run>/<name>` (CSV and Excel as
  rows, with how it was made) and `/files` lists every chat's. A deleted file is said to be gone.
- A failed sandbox command says why when the sandbox knows (`sandbox.base.hint`): a command
  killed (exit 137) or out of memory names the memory limit it went past; a missing Python module
  says whether the sandbox can install it (no network: use the standard library). In
  `sandbox_exec`'s result as `hint`, at the end of Deep Agents' `execute` output.
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

- `trellis.ask_user()`: an opt-in tool the model asks the person with — free text, a choice
  (`options`), a `date`, a `date_range` or a `number` — through the run's own pause, the answer
  its result. The interrupt names the `ask_user` component (`props`: `kind`, `options`) for an
  app's own picker, carries the schema and widget hints for any other UI, and its question says
  the kind of answer for plain chat ([interrupts.md](docs/interrupts.md#asking)).
- AG-UI activity: an `ACTIVITY_SNAPSHOT` after each tool call that ends groups the run's calls
  into one line, `used 2 tools · ran 1 command` ([surfaces.md](docs/surfaces.md)).
- AG-UI parity with CopilotKit-style clients (`frontend.py`): the request's `tools` are offered
  to the model beside the agent's own (one of the agent's by the same name wins, with a
  warning); a call streams its `TOOL_CALL_*` events and pauses the run (an interrupt whose
  reason is `frontend_tool`, its `toolCallId` the call's), and the client's result — a `tool`
  message with that `toolCallId`, or a `resume` entry — is the call's output. `context` and the
  `state` reach the model in the pushed context, together at most half of its token budget;
  the run reads and sets the state (`Runtime.state`), each change streamed as `STATE_DELTA`
  (a JSON Patch; `STATE_SNAPSHOT` where there was no object to patch), and a resume reads the
  state the client sends. `forwardedProps.mode` (AG-UI) and the message's `metadata.mode`
  (A2A) are the run's `mode` (queue, interrupt, steer). A call's paused `TOOL_CALL_RESULT` is
  no longer sent on AG-UI: the call has no result until the run resumes. New dependency:
  `jsonpatch`.
- MCP elicitation: an `mcp(...)` server that asks the person in the middle of a call pauses
  the run as `ask` does — one field of a few values a choice (a dropdown), any other form a
  form with the server's schema (a `format: date` field a date picker), a link a choice — and
  the answer goes back to the server when the run resumes, on both MCP protocol eras
  (`tools/elicit.py`; [tools.md](docs/tools.md#a-server-that-asks-mcp-elicitation)).
- MCP resources: an `mcp(...)` server that has resources gets `<server>-read_resource`, its
  description the map of what there is (resources and templates, at most 20, with the server's
  `lastModified`); it reads one by its URI (a binary one described) or, given none, lists them
  all as rows. A large resource is kept, paged and queried as any large result (`tools/resources.py`;
  [tools.md](docs/tools.md#a-servers-resources-mcp-resources)).
- A file store for the files runs make for their users (`trellis.harness.files`, on fsspec):
  `TRELLIS_FILES_URL` (`gs://`, `s3://`, `az://` with `gcsfs`, `s3fs`, `adlfs`; `file://`),
  else the XDG data folder. `Runtime.save_file(name, data, source=)` keeps a file under
  `<tenant>/<user>/<conversation>/<run>/`, names it in the run's journal and announces it (the
  `file` event); `serve_chat` serves it to the run's own user at
  `GET {path}/runs/{run_id}/files/{name}`: a 5-minute signed link where the store signs, else
  streamed. Stored only: never parsed or remembered. New dependency: `fsspec`.

- Forget by chat: `memory_forget` of what the user said ("forget my address"), which the memory
  service refuses an agent, asks a person through the approval path (the run pauses as any
  call that asks does) and, approved, forgets it as the user; rejected, the model reads why.
  A memory the agent wrote is forgotten at once, as before. Needs the memory service to name
  the memory it refuses (`details.memory_id`); an older one's refusal stays the error it was.
- Where a tool's result came from: `tool(response_format="content_and_artifact")` (LangChain's)
  returns `(content, artifact)`, the artifact a `Citation(name, date, link)` or a list of them
  (also a LangChain tool's own artifact); the model reads the content alone and the call's
  outcome keeps the citations in `metadata["sources"]`, so the run's trajectory has them.
- The turn written to memory carries its run record (`Said.facts`, under
  `custom_metadata["run_record"]`, plain values): each tool result its call's status, seconds,
  sources (the trajectory's outcome of the same call), when it was fetched (`fetched`, UTC)
  and its tool's `max_age`; the turn's last message the run's status, seconds and approvals —
  what the memory service's steps list of the turn shows, with its evidence line for "why" and
  "how sure" (trellis-memory 0.6.2). The fetch time is said there only: a tool result written
  to memory is no longer stamped `[snapshot of <tool> from <UTC time>]`, and a replayed result
  past its tool's age is marked out of date from its run record.
- A `ReAct` agent with a known window sizes the conversation the memory service replays
  (`conversation_token_budget`, trellis-memory 0.6): what one request may hold (95% of the
  window, as `FitWindow` fits it) less its instructions, offered tools and question
  (LangChain's `count_tokens_approximately`), the memory context's budget and 1,024 tokens for
  the reply (`Agent.conversation_budget`). Tool-result clearing (`ThisTurnsResults`, LangChain's
  `ClearToolUsesEdit`) clears only the turn under way's older results, never the replayed
  turns' that memory already sized.
- The cost benchmark's `--suite explain`: two-turn conversations, memory on, whose second turn
  asks about the first answer (its steps, sources, tools, time, approvals...) or is an ordinary
  follow-up, scored strictly against turn 1's own journal record (exact tool names, source
  names and dates, the step count, seconds within a tolerance), with the made-up claims and
  leaked template tags counted, every model request kept, and `--render` checking what a
  llama.cpp chat template shows the model (`benchmark/cost/explain.py`).
- What a finished run spent is on its end: every `RUN_FINISHED` carries `data.spent` (tokens,
  cached tokens, dollars, model calls, tool calls — its sub-agents' runs and earlier attempts
  included), which agent-runs' event log keeps (ADV-11).

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
  (`tools.base.data_of`). Tool results replayed from memory's thread past the tool's age
  (their fetch time is in their run record) are marked out of
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
