# Every SDK's own features through the harness: real Haiku, the real stack (2026-11-15)

Does each SDK the harness wraps keep its own features working as it does on its own? Each
feature was run with the real model (`claude-haiku-5-5`) four ways:

* **alone:** the SDK as its docs show it, no harness (`graph.ainvoke`, `Runner.run`, `query`
  / `ClaudeSDKClient`, `kimi -p` or a plain ACP client);
* **off:** wrapped with `h.wrap`, harness middleware off (the default), no services;
* **stack:** wrapped, with a private agent-memory-service and agent-runs (main), and what
  they recorded read back and checked;
* **on:** wrapped with the harness middleware (`trellis.middleware()`), where it applies
  (LangChain graphs only).

Code under test: main of all five repos as cloned at the start (agent-harness `0047ecfa`,
agent-memory-service `c835844`, agent-runs `7dfe8df`, agent-contracts `27a818d`,
bifrost-sdk `ccaaef8`), then the fixes below. SDKs: LangChain 1.5 / LangGraph, Deep Agents
0.7.38, langgraph-swarm 0.1.0, OpenAI Agents 0.24, Claude Agent SDK 0.2.182 (CLI 2.1.325),
Kimi Code 2.6.2 (ACP).

**Spend: $0.212** for 628 model responses (5.53 M input, 61 k output tokens), read from the
provider's usage of every response (a recording proxy in front of `api.anthropic.com`; three
streamed OpenAI-compatible responses carried no usage and are not counted, under $0.001).
Pilot first (LangGraph, 6 cases: $0.0008; Claude, 6 cases: $0.009), then breadth. Cap $7.

## How it was run

* `scripts/` — one script per SDK (`lc_agent.py`, `deep.py`, `react_swarm.py`, `oai.py`,
  `claude.py`, `kimi_code.py`), `lifecycle.py` (restart, queue, cancel), `collisions.py`
  (part C, no model), `check.py` (the record check and the tables), `common.py`, `turns.py`,
  `run.sh` (a scrubbed environment; the key read from its file inside the command, never
  printed). `python <script> feature[:arm] ...` reruns one case.
* `data/<sdk>/<feature>.<arm>.json` — what each case saw: answers, statuses, the questions
  asked, the tool calls, its spend and, on the stack, the run record, its events (from
  agent-runs' API), the children, the memory transcript, tool records and feedback rows.
  `data/summary.json` and `data/tables.md` are built by `check.py`.
* The stack: `scripts/scenarios/stack.sh` on databases of its own (`sdkfull_*`, port 55957),
  ports 18387 (memory) and 18397 (agent-runs); Docker for the OpenAI sandbox
  (`python:3.12-slim`).
* An outcome is the feature's own result (the tool ran with these arguments, the file holds
  this text, the answer names this value, the plan has three todos done...), decided the same
  way for every arm. "Same" means the harness arm had the outcome the SDK had alone.
* The record check (`check.py`), per stack run and child: a final status matching the
  result; `RUN_STARTED` and `RUN_FINISHED` on its log; one tool record in memory per call the
  logs say was made (pauses, rejections and the memory tools, which are memories, aside); the
  question as the first message and, for a success, the answer as an assistant message; a
  feedback row for each approval.

Governance differences that are documented and expected: under the harness a tool that
declares nothing asks a person at every call (an OpenAI `FunctionTool` can declare nothing in
code), and a write asks once a conversation. The cases answered those questions `approve`;
they are listed in the data (`asked`).

## Results per SDK

`pass`/`FAIL` is the feature's outcome; "records" is the record check on the stack. Evidence:
`data/<sdk>/<feature>.<arm>.json`.

### LangGraph / LangChain `create_agent` (`data/langgraph`)

| feature | alone | off | stack (records) | middleware on |
|---|---|---|---|---|
| tools | pass | pass, same | pass (pass) | pass |
| structured output (`response_format`) | pass | pass, same | pass (answer kept only as LangChain's tool message, see O6) | pass |
| streaming (`messages`) | pass | pass, same (33 deltas) | — | pass |
| HITL (`HumanInTheLoopMiddleware`) | pass | pass, same | pass (pass; one question) | pass (one question) |
| own checkpointer (`InMemorySaver`), 2 turns | pass | pass, same | pass (pass) | pass |
| own store (`ToolRuntime.store`) | pass | pass, same (the write asked once) | pass (pass) | — |
| graph's own `interrupt()` | pass | pass, same | pass (pass) | pass |
| `ToolRetryMiddleware` | pass | pass, same (2 tries) | pass (pass) | — |
| `ModelFallbackMiddleware` | pass | pass, same | — | — |
| `PIIMiddleware` (input) | pass | pass, same | pass (no e-mail in memory) | — |
| `get_stream_writer` (custom) | pass | pass, same | — | — |
| `context_schema` | pass | pass, same | — | — |
| restart: approval answered by a new process | — | — | pass (call made once) | — |
| queued run + worker | — | — | pass (pass) | — |
| cancel while a tool works | — | — | CANCELLED (log lacks its end: B1) | — |

### Deep Agents 0.7.38 (`data/deepagents`)

| feature | alone | off | stack (records) | middleware on |
|---|---|---|---|---|
| `write_todos` (`TodoListMiddleware()` given) | pass | pass, same | pass (pass) | not offered: `Planning` holds it to a long task (C1) |
| state files across turns | pass | pass, same | pass (pass) | pass |
| sub-agent (`task`) | pass | pass, same | pass (child run; see O4) | pass |
| `interrupt_on` | pass | pass, same | pass (pass) | pass |
| `CompositeBackend` + `StoreBackend` route | pass | pass, same | pass (pass) | — |
| memory files (`AGENTS.md`) | pass | pass, same | pass (pass) | — |
| skills (`SKILL.md`) | pass | pass, same | — | pass |
| `permissions` (deny) | pass | pass, same | — | — |
| structured output | pass | pass, same | — | pass |
| restart / queued / cancel | — | — | pass / pass / CANCELLED (B1) | — |

### `ReAct` (the harness's own agent; middleware always on) (`data/react`)

No "alone" exists; each row is the library feature it builds on.

| feature | off | stack (records) |
|---|---|---|
| tools | pass | pass (pass) |
| structured output | pass | pass (O6) |
| approval (irreversible), approve | pass | pass (pass) |
| approval, reject with a reason | pass | pass (pass) |
| Deep Agents sub-agent | pass | pass (child run) |
| `backend=` files across turns | pass | pass (pass) |
| plan mode (`PlanMode`) | pass | pass (pass) |
| restart / queued / cancel | — | pass / pass / CANCELLED (B1) |

### langgraph-swarm 0.1 (`data/swarm`)

| feature | alone | off | stack (records) | middleware on |
|---|---|---|---|---|
| handoff in a turn, next turn answered by the agent handed to | pass | pass, same (the refund write asked once) | pass (pass) | pass |

### OpenAI Agents 0.24 (`data/openai_agents`)

| feature | alone | off | stack (records) |
|---|---|---|---|
| function tools | pass | pass, same | pass (pass) |
| `output_type` | pass | pass, same | pass on the re-run (first run: the model skipped the tool) |
| streaming | pass | pass, same | — |
| handoffs | pass | pass, same | pass (pass) |
| `Agent.as_tool` | pass | **failed, fixed** (F1) | pass (child run, O4) |
| input guardrail tripwire | pass | pass, same (`ERROR`, the SDK's tripwire) | pass (pass) |
| output guardrail | pass | pass, same | — |
| `SQLiteSession` | pass | pass, same | pass (pass) |
| `needs_approval` | pass | pass, same | pass (pass) |
| `RunHooks` on `h.wrap` | pass | pass, same | — |
| `RunHooks` per run | pass | lost after a pause (O1) | — |
| `context` on `h.wrap` | pass | pass, same | — |
| `context` per run | pass | lost after a pause (O1) | — |
| `max_turns` | `MaxTurnsExceeded` | `PARTIAL` (documented) | — |
| MCP server (stdio) | pass | pass, same | pass (pass) |
| `tool_use_behavior` | pass | pass, same | — |
| `ModelSettings(max_tokens)` | pass | pass, same | — |
| `SandboxAgent`, Unix-local | pass | pass, same | — |
| `SandboxAgent`, Docker | pass | pass, same | pass (pass) |
| restart / queued / cancel | — | — | pass / pass / CANCELLED (B1) |

### Claude Agent SDK 0.2.182 (`data/claude_agent_sdk`)

| feature | alone | off | stack (records) |
|---|---|---|---|
| in-process MCP tool (`readOnlyHint`, in `allowed_tools`) | pass | pass; **asked every call before F2** | pass; asked "irreversible" every call before F2 |
| built-in `Read` | pass | pass, same | pass (pass) |
| built-in `Write` (`acceptEdits`) | pass | pass in 1 of 2 runs (O2) | pass (pass) |
| `output_format` | pass | pass, same | — |
| own `PreToolUse`/`PostToolUse` hooks | pass | pass, same | — |
| own `can_use_tool` (denies `Bash`) | pass | pass, same | — |
| subagents (`agents=`) | pass | pass; one run went to the background (O3) | pass after F2 (child `SUCCESS`; before F2 the pause cut it: `HelperUnfinished`) |
| `max_turns` | `error_max_turns` | `PARTIAL` (documented) | — |
| `include_partial_messages` | 15 deltas | **1 event, fixed** (F3) | — |
| `permission_mode="plan"` | pass | pass, same | — |
| stdio MCP server | pass | pass, same | — |
| query way: session resumed | pass | pass, same | pass (pass) |
| live session (`ClaudeSDKClient`), 2 turns | pass | pass, same | pass (pass) |
| live session: background `Bash`, read 2 turns later | pass | pass, same | pass (pass) |
| restart / queued / cancel | — | — | pass / pass / CANCELLED (B1) |

### Kimi Code 2.6.2 (`data/kimi`)

| feature | alone | off | stack (records) |
|---|---|---|---|
| `Read` | pass | pass, same | pass (pass) |
| `Bash` that writes | pass (prompt mode runs it) | pass (asked, approved) | pass (pass) |
| `AGENTS.md` | pass | pass, same | — |
| MCP server (stdio, over ACP) | **fails alone too** (K1) | fails, same | fails, same |
| `Agent` sub-agent | pass | pass, same | pass (child run) |
| session across turns | pass | pass, same | pass (pass) |
| skills | pass | pass, same | — |
| `TodoList` | pass | pass, same | — |
| plan mode (ACP, plan rejected) | pass | pass, same | — |
| background `Bash`, read 2 turns later | pass | **fails, 3 of 3** (B2) | fails (B2) |
| restart / queued / cancel | — | — | pass / pass / pass |

## Part C: middleware on

`trellis.middleware()` is LangChain middleware: it applies to `create_agent`, Deep Agents and
langgraph-swarm agents (and is always on in `ReAct`). OpenAI Agents, Claude and Kimi have no
opt-in middleware; what the harness adds there is on with middleware off too (below).

| | what changes | documented |
|---|---|---|
| C1 planning | `Planning` (a `TodoListMiddleware` held until a long task) takes the place of the user's `TodoListMiddleware`: a short turn asking for a three-step plan was answered without `write_todos`; a user's own `TodoListMiddleware` beside `middleware()` fails the build (`AssertionError`) | yes (deepagents.md, langgraph.md; corrected here for 0.7.38). **Superseded:** agent-harness#277 (merged after this run) makes `Planning` step aside in a Deep Agents graph, so the team's `TodoListMiddleware` is offered at every window |
| C2 duplicates | a user's `ModelCallLimitMiddleware`, `TodoListMiddleware` or (on `create_agent`) `AnthropicPromptCachingMiddleware` beside `middleware()` fails the build; `ContextEditingMiddleware` only with a `context_window`; `SummarizationMiddleware` and `ToolRetryMiddleware` build. `ReAct(middleware=[...])` refuses the first three with a `ConfigurationError` naming the parameter (`data/collisions.json`) | yes (langgraph.md Limits) |
| C3 checkpointer | a graph with no checkpointer runs with `RunCheckpointer()` | yes |
| C4 tools and steps | memory and gateway tools added and narrowed per turn; `max_steps` (`ModelCallLimitMiddleware`), stall guard, `call_timeout`, prompt cache; a `create_agent` tools case cost 2.3x the tokens of middleware off | yes (react.md) |
| C5 HITL | the gate's decision is governance's: one question | yes |

Added with middleware off, on every SDK: governance of the SDK's own tools (above); a line
after the system prompt that files written in `outputs/` are given to the user (Claude, Kimi;
see O2); Kimi's permission mode forced to ask, its cron, tower mode, update and telemetry
off. All documented.

## Bugs found and fixed (branch `claude/cool-maxwell-w5avo8-sdkfull`, agent-harness)

| | what broke under the harness | fix | test (fails before, passes after) |
|---|---|---|---|
| F1 | OpenAI: an `Agent.as_tool` agent whose own tool declares nothing paused the run; the approval then **failed the run** (`AttributeError: 'Agent' object has no attribute 'aget_state'`): the pause carried the child's SDK handle, taken for a LangGraph checkpointer's (`pipeline._pause`) | `bcc7b5fc` | `test_openai_agents_features.py::test_a_governed_call_inside_an_agent_tool_resumes_and_runs_once` |
| F2 | Claude: an in-process tool annotated `readOnlyHint` asked a person at **every call** when `allowed_tools` named it (the SDK's own way to pre-approve it): the bare name took the place of the server's listing (`adapters/claude._declared`, `unique` keeps the first); and with the memory service it was published to the catalog as the protocol's default (`destructiveHint`), so the catalog's `irreversible` asked whatever the tool said (`governance/catalog.entry`) | `13f39622` (listing), `fb8125c0` (catalog) | `test_claude_sdk_features.py::test_an_in_process_tools_annotations_decide_it_when_allowed_tools_names_it`, `test_declared_side_effects.py::test_an_mcp_tool_whose_listing_gave_its_risk_is_published_saying_it` |
| F3 | Claude: with `include_partial_messages` the run's stream gave one text event per message (the SDK's own: one per delta) | `13f39622` | `test_claude_sdk_features.py::test_partial_messages_stream_the_runs_text_as_the_model_writes_it[query,live]`, `test_without_partial_messages_the_runs_text_is_each_message_once` |

Each was confirmed again with Haiku after the fix (the F2 stack case: no question, the
subagent's child run `SUCCESS`; the catalog row now says `readOnlyHint`).

## Bugs found, not fixed (each needs a decision)

* **B1. A run cancelled in its own process leaves agent-runs' log without its end** (every
  in-process SDK: LangGraph, Deep Agents, `ReAct`, OpenAI, Claude). The stream the caller reads
  has `TOOL_CALL_RESULT` (cancelled) and `RUN_FINISHED` with the reason; agent-runs' log of the
  run stops at `TOOL_CALL_ARGS`. Cause: `Agent.cancel` (agent.py, `runs.cancel` before
  `self.stop`) has agent-runs end the run `CANCELLED` at once (a run no worker leases), so the
  attempt's last events are refused (`append_events`: "events are appended while it runs";
  `RunLog._send` drops them on `ConflictError`). The record itself is right (`CANCELLED`, its
  reason). Fix is either the harness stopping the attempt before the store's cancel (the record
  must then still get the reason and who cancelled) or agent-runs taking a cancelled run's own
  last events; not made here.
* **B2. Kimi: a background command's result read in a later turn ends that turn `ERROR`
  (`NO_ANSWER`), 3 runs of 3**; `kimi -p` answers it. Kimi's log shows the user's turn calling
  `WaitFor`, the task already ended, and Kimi ending that turn right after the tool step to run
  its own notification turn, whose reply has the answer — a turn ACP does not forward and the
  harness by design never takes as the answer (kimi.md, Background work).
* **O1. OpenAI: a `RunHooks` or `context` object given per run is lost when governance pauses
  the run** (the resume runs with the agent's options): `on_tool_start` never fired, and the
  tool reading `ctx.context` failed. Documented ("an object is that call's only"), but the
  pause is one the harness inserted (an undeclared `FunctionTool`). On `h.wrap` both work.
* **O2. Claude: the `outputs/` line the harness adds to the system prompt** made Claude write
  `hello.txt` into `outputs/` instead of the working folder in 1 of 2 runs (Kimi gets the
  same line). Documented, but it changes where the SDK's own `Write` puts a file.
* **O3. Claude, query way:** in one run Claude ran the subagent in the background and answered
  before it ended, so the answer lacks the result (documented: in a run with no conversation a
  background task's report is not the answer).
* **O4. A library's own sub-agent (Deep Agents `task`, `Agent.as_tool`, Claude `agents=`, Kimi
  `Agent`) has a child run with no events of its own:** its calls are on the parent's log and
  its tool records under the child. subagents.md says a child's events are on its own stream;
  that holds for a wrapped agent as a child, not for these helpers.
* **O5. Deep Agents' `write_file` on a store or disk route is announced, not asked** (a
  framework's own tool under `Decision.unasked`), while a user's own write tool in the same graph
  asks. deepagents.md says those writes are "governed as writes"; governance.md says writes
  ask.
* **O6. Structured output in memory:** with LangChain's tool strategy the transcript ends with
  the tool message `Returning structured response: ...`, no assistant text (agent-runs keeps the
  output).
* **K1. Kimi (library):** an MCP server given in ACP's `session/new` that starts in about a
  second (a Python stdio server) is not offered to the model: Kimi lists its tool
  (`tool_select.initialized`) but leaves it out of the request (`llm.tools_snapshot`) as a
  runtime addition loadable only through `select_tools`, which a model without
  `dynamically_loaded_tools` is not given. The same with a plain ACP client, no harness.
* **Docs (fixed here):** Deep Agents 0.7.38 no longer adds `TodoListMiddleware` itself (only
  its Codex profile does); deepagents.md said planning is Deep Agents' own.

One test, `test_declared_side_effects.py::test_a_tool_that_declares_nothing_never_runs_unapproved_on_any_adapter[a bare function-purge_records-middleware]`,
failed once in a run of four files while the machine was short of memory, and passed in three
runs after, with and without the change; its cause was not found.

## Not tested, and why

* Bifrost: every model call went straight to Anthropic (OpenAI Agents through Anthropic's
  OpenAI-compatible chat completions), so the OpenAI Responses-only features — hosted tools,
  the sandbox's `Compaction`, server-side threads — were not reachable. It needs a Bifrost of
  this job's own (the `maximhq/bifrost:v2.5.1` image another job runs here now, with an
  Anthropic provider key) and a model it serves on `/v1/responses`; the Bifrost container on
  the machine is another job's and was not used.
* Deep Agents summarization, Kimi and Claude compaction: they start near the context window's
  end, a cost out of proportion for a breadth pass.
* OpenAI realtime/voice (refused by `h.wrap`, documented), Kimi swarms and goal mode
  (unsupported over ACP, documented), OpenAI background agents (none in 0.24).
* Kimi `AskUserQuestion`, Claude `AskUserQuestion` and MCP elicitation: not reached in this
  pass (the offline suites cover them with stand-ins).
* LangGraph time travel, `ReAct` background tasks (offered only to a chat served with
  `serve_chat`), schedules: harness features rather than an SDK's; covered offline.
