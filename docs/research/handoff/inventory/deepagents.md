# Deep Agents 0.7.38: full feature inventory against the Trellis harness

Read-only audit, 2026-11-14. No code was changed and no paid model was called.

**Sources**
- Library: `/var/tmp/cons/reactbg/.venv/lib/python3.13/site-packages/deepagents`. It matches the extracted wheel `/var/tmp/cons/research2-tmp/x/deepagents-0.7.38-py3-none-any` file for file (`diff -rq` is clean).
- Changelog: `/var/tmp/cons/research2-tmp/da_CHANGELOG.md`.
- Docs: docs.langchain.com/oss/python/deepagents (overview, backends, human-in-the-loop, event-streaming).
- Harness: `/var/tmp/cons/reactbg` (newest). `/var/tmp/cons/final-agent-harness` is older: it pins `deepagents>=0.7.19`, has no `bridges/async_tasks.py` and no `BackgroundTasks`, and differs by about 100 lines in each of `react.py`, `middleware.py` and `adapters/langgraph.py`. All file:line references below are to reactbg.

**The two ways Deep Agents is used**
- **D (direct)**: `h.wrap(create_deep_agent(...))`. Two variants:
  - without `middleware()`: the "narrowed" path, `adapters/langgraph.py:503`;
  - with `middleware=trellis.middleware()`: `react.py:130`.
- **R (ReAct)**: `ReAct(...)` builds `create_deep_agent` with the harness stack (`react.py:263-450`). It has two modes:
  - "lean": no Deep Agents capability was asked for;
  - "capability": `backend=`, a subagent, `skills=`, `memory_files=` or `permissions=` was given.

**Evidence produced in this audit.** Scratch files are in `/var/tmp/cons/inventory-tmp/deepagents/`. All of it is offline, with `ScriptedChatModel` and no network.
- `profile_check.py`: which harness profile Deep Agents resolves for ReAct's `RoutedChatModel`.
- `order_check.py`: the middleware stack as assembled, for ReAct and for `middleware()`, captured at `create_agent`.
- `probe/test_probe.py` and `probe/test_probe2.py`: whether files survive between turns, which tools are offered, and whether sandbox file writes are governed. Run with `pytest -p tests.conftest`.

**Status values**
- `works`: behaves as the library does, and the harness adds what it promises.
- `caveat`: works, but the harness changes or limits library behaviour, or two mechanisms overlap.
- `broken`: fails, or silently loses library behaviour or harness governance.
- `blocked`: blocked by design, with the reason given.
- `untested`: the code passes it through, but no test covers it.
- `n/r`: not reachable through the harness.
- `n/a`: CLI or hosting only.

## Assembled middleware order (outer → inner), captured

| Build | Main-agent stack |
|---|---|
| plain `create_deep_agent` | Filesystem > SubAgent > _ParentNotifier > Summarization > PatchToolCalls > [AsyncSubAgent] > [profile extra] > [Skills] > AnthropicCaching(lazy) > Bedrock > [Memory] > [HITL] > UnsupportedContent > ModelRefusal > [_ToolExclusion] |
| `create_deep_agent(middleware=middleware())` | Filesystem > SubAgent > _ParentNotifier > Summarization > PatchToolCalls > **TodoList(Planning) > HarnessTools > GovernTools > StallGuard > ModelCallLimit > Budgeted > FitWindow > ModelHooks** > Skills > **AnthropicCaching=PromptCache** > Bedrock > Memory > HITL > UnsupportedContent > ModelRefusal |
| ReAct, capability mode (skills+memory_files) | Filesystem > SubAgent > _ParentNotifier > **Summarization=no-op** > PatchToolCalls > Planning > HarnessTools > GovernTools > StallGuard > ModelCallLimit > Budgeted > ContextEditing > ReadTools > FitWindow > **ReActSummarization** > ModelHooks > Skills > PromptCache > Bedrock > Memory > UnsupportedContent > ModelRefusal |
| ReAct, lean | the same, plus LongTaskTools, with no summarization at all |
| ReAct general-purpose subagent | Filesystem > Summarization (DA's) > Patch > HarnessTools > GovernTools > StallGuard > ModelCallLimit > ModelHooks > PromptCache > Bedrock > ModelRefusal > UnsupportedContent. **There is no SkillsMiddleware, even when ReAct has `skills=`**; the general-purpose subagent of `create_deep_agent` has it. |
| fork subagent (ReAct or `middleware()`) | inherits the parent's whole custom stack, as Deep Agents mirrors fork middleware: BackgroundTasks, Planning, HarnessTools, Budgeted, FitWindow, ReActSummarization… |

What the order means:
- FitWindow, the Deep Agents summarization and ModelHooks' span sit **outside** SkillsMiddleware and MemoryMiddleware. The skills listing and the AGENTS.md text are added after FitWindow has measured the request, and they are not in the `chat` span.
- `create_deep_agent(middleware=[TodoListMiddleware(), *middleware()])` raises `AssertionError: Please remove duplicate middleware instances` at build time (captured).

## Inventory

Legend for the D and R columns: *pass* = passed through unchanged, *xform* = transformed, *block* = blocked, *override* = replaced by the harness.

| # | Feature (Deep Agents 0.7.38) | What it does | D: harness handling (file:line) | R: harness handling (file:line) | Governed / recorded | Tested (test::name) | Status D | Status R |
|---|---|---|---|---|---|---|---|---|
| **create_deep_agent parameters** |||||||||
| 1 | `model` (str `provider:model` / BaseChatModel) | resolved via `init_chat_model` plus a provider profile, then a harness profile | pass (Deep Agents resolves it) | xform: a model name becomes the harness `RoutedChatModel` (`react.py:344`, `gateways.py:688,823`), so Deep Agents' `resolve_model` and provider profile are skipped and the harness profile lookup misses (provider reads as `routedchatmodel`; `anthropic/claude-opus-4-7` gets no Opus 4.7 suffix; captured) | model calls accounted, spans with `middleware()` | D: every test uses a chat model object, so string specs are untested. R: test_react::test_a_model_name_is_a_gateway_model | works | caveat |
| 2 | `model=None` default (deprecated) | `claude-sonnet-4-6` with a warning | pass | block: `model` is required (ReAct's own API) | – | – | untested | blocked |
| 3 | `tools` (BaseTool / callable / provider dict) | added to the built-ins | pass. Every ToolNode call is governed (`middleware.py:1078`, installed at `adapters/langgraph.py:164`) | xform: LangChain tools stay the graph's; functions, `tool()` and MCP go to the harness toolbox (`react.py` `_split`) | yes: catalog, journal, records | test_native_tools::test_a_graphs_own_tool_is_governed_journaled_recorded_and_published; test_deepagents::test_a_deep_agent_uses_its_own_tools_and_the_harness_tools. Provider-native dict tools: none | works | works |
| 4 | `system_prompt` (str / SystemMessage with cache_control) | USER → profile BASE → SUFFIX | pass. With `middleware()`, `ModelHooks._dated` appends the date (`middleware.py:1742`) | xform: str or `stored()`; date appended | stored prompt pinned and journaled | test_react::test_the_middleware_run_in_their_order; SystemMessage form untested | caveat | caveat |
| 5 | `middleware` (name replaces default; new lands before tail) | caller extension | pass. `middleware()` follows Deep Agents' name rules, but its `Planning` is named `TodoListMiddleware` (`automatic.py:122`), so a user TodoListMiddleware crashes the build (captured) | yours goes before ModelHooks; a harness name is refused (`react.py:769`); a Deep Agents name replaces | – | test_react::test_given_middleware_is_added_before_model_hooks_and_replaces_nothing, test_a_deep_agents_middleware_given_replaces_its_own_by_name; the crash is untested | broken | works |
| 6 | `permissions` (FilesystemPermission allow/deny/interrupt, first match, subagent inheritance) | tool-level filesystem rules | pass | pass (switches to capability mode) | deny is the library's refusal; an interrupt becomes a harness approval | test_deepagents_library::test_a_denied_path_is_the_librarys_refusal_and_an_interrupt_path_asks. Subagent inheritance and ReAct: none | works | untested |
| 7 | `backend` | file store and execution | pass. `declare_files` marks file tools outside state as governed (`middleware.py:299-337`) | xform: default `JournalBackend` (`automatic.py:150`, caps 100k chars per file and 1M per run; `react.py:376`). `sandbox()` becomes SandboxBackend (`react.py` `_backend`) | outside the state: yes | see rows 30-38 | works | caveat |
| 8 | `interrupt_on` | HITL per tool | pass (HITL resumes mapped: `adapters/langgraph.py:723-768`) | block: not a parameter; approvals belong to the catalog (react.md "Not exposed") | journal key `langchain_hitl` | test_hitl_middleware::test_a_deep_agent_interrupt_on_is_answered_with_harness_decisions | works | blocked |
| 9 | `response_format` | structured output | pass | pass | answer = `structured_response` (`adapters/langgraph.py` extract) | test_deepagents_library::test_a_deep_agents_structured_answer_and_context_are_the_librarys; test_react::test_structured_output_is_the_native_response_format | works | works |
| 10 | `state_schema` (DeepAgentState subclass, forwarded to subagents) | custom state | pass | pass | – | none | untested | untested |
| 11 | `context_schema` | run context | pass (`framework_options={"context": ...}`, `adapters/langgraph.py:115,450`) | pass | – | test_deepagents_library::test_a_deep_agents_structured_answer_and_context_are_the_librarys (D only) | works | untested |
| 12 | `checkpointer` | thread persistence | your own is kept (a durable one is warned about: `adapters/langgraph.py:537-553`). With `middleware()` and none given, a `RunCheckpointer` copy is used (`:588`) | block/override: always `RunCheckpointer` (`react.py:437`), so **the thread is the run** (`adapters/langgraph.py:515`) | latest checkpoint in the run journal | test_react_deep_agents::test_middleware_on_create_deep_agent_runs_and_resumes_in_place; test_react::test_a_resume_continues_in_place | works | caveat (see row 31) |
| 13 | `store` | for StoreBackend | own store kept (warned). With none, the memory service is injected (`adapters/langgraph.py:464`) | block: always the memory service | writes become the user's memories | test_deepagents_library::test_files_on_a_store_of_the_graphs_own_are_kept_past_the_thread, test_files_on_a_store_go_to_the_memory_service_when_the_graph_has_no_store | caveat | blocked |
| 14 | `debug` | create_agent debug | pass | block (not exposed) | – | – | untested | blocked |
| 15 | `name` | graph name, `lc_agent_name` | pass | block: fixed to `"react"` (`react.py:439`); the wrap `id` names the agent | – | – | works | blocked |
| 16 | `cache` (LangGraph BaseCache) | node cache | pass | pass | – | test_prompt_cache::test_reacts_own_langgraph_cache_is_the_one_given | untested | works |
| 17 | `recursion_limit=9999` and trace metadata (`with_config`) | effectively no step cap | kept; `framework_options` may override. With `middleware()`, `max_steps=12` by default caps model calls (`react.py:130`, `middleware.py:2044`) | `max_steps=12` per run | step_limit warning on the stream | test_react::test_the_step_limit_is_12_model_calls_by_default, test_framework_options_are_the_graphs_run_config | caveat (only with `middleware()`) | works |
| **Planning** |||||||||
| 18 | `TodoListMiddleware` / `write_todos` (opt-in since 0.7.0) | plan in `state["todos"]` | user-added: works without `middleware()` (examples/02_way1_deepagents/agent.py uses `[TodoListMiddleware(), HarnessTools()]`); **crashes with `middleware()`** (captured). `middleware()` adds its own Planning, offered only on a long task and **never without `context_window=`** (`automatic.py:79-88`, `react.py:235`); Planning drops TodoListMiddleware's system prompt (`automatic.py:139-147`) | xform: Planning is held until the conversation passes 50% of a window of 32k or more, or is on from the first step in capability mode. A user TodoListMiddleware is refused (`react.py:769`). With the Codex harness profile, Planning would replace the profile's TodoList by name | state tool: not governed or journaled (`middleware.py:227,445`; the checkpoint keeps it) | test_react_long_tasks::test_a_plan_is_written_and_read_back, test_react_alone_is_offered_them_once_its_conversation_outgrows_half_the_window | broken (with `middleware()`) | caveat |
| 19 | todos across turns | persists on the thread | own checkpointer: yes | **reset every turn** (thread = run); only `async_tasks` is carried (`adapters/langgraph.py:147`) | – | probe: files lost (same mechanism) | works | broken |
| **Filesystem (FilesystemMiddleware)** |||||||||
| 20 | ls/read_file/write_file/edit_file/delete/glob/grep on StateBackend | virtual files in `state["files"]` (DeltaChannel) | state tools let through ungoverned (`middleware.py:445-447`). The first scratch write's result gets `SCRATCH_NOTE` appended (`middleware.py:375-408`): **tool result text changed** | the same; lean mode holds the scratch tools until a long task (`automatic.py:203`), and `read_file` only once there is something to read (ReadTools) | not governed (the checkpoint keeps them) | test_react_long_tasks::test_scratch_files_are_written_edited_listed_and_read, test_a_deep_agents_state_write_says_it_and_a_failed_one_does_not; test_deepagents_library::test_a_composite_backend_routes_each_path_to_its_backend | caveat | caveat |
| 21 | `execute` (sandbox backends) | shell command | declared a write (`middleware.py:333`) | sandbox(): harness SandboxBackend with an env card; on a backend that runs nothing, `execute` is machinery (`react.py` `_declared`) | yes | test_deepagents_library::test_a_local_shell_backends_execute_runs_the_command_as_a_write_of_the_run, test_execute_without_a_sandbox_backend_is_the_librarys_own_refusal; test_sandbox_backend::test_react_runs_its_commands_in_the_sandbox_governed_and_recorded | works | works |
| 22 | `FilesystemMiddleware(tools=[...])` allowlist, `FsToolName` | hide built-ins | pass (by name replacement) | pass via `middleware=` | – | none | untested | untested |
| 23 | `grep_max_count` / per-call `max_count`, truncated flag, line clipping | bounded grep | pass | pass | – | none | untested | untested |
| 24 | large tool result eviction (over 20k tokens → `/large_tool_results/<id>` with preview) | context offload | pass | lean: the run keeps the result instead (`KEPT_BY_RUN`, ReadTools/`_kept_read` `middleware.py:800`); capability: Deep Agents evicts and `query_result` reads it | kept results journaled | test_deepagents_library::test_a_large_result_of_the_agents_own_tool_is_moved_to_a_file_it_reads; test_react::test_a_huge_result_is_saved_as_a_file_the_model_reads_in_pages | works | caveat |
| 25 | HumanMessage eviction (over 50k tokens) | offload a huge user message | pass (the harness memory context is a SystemMessage, never evicted) | pass | – | none | untested | untested |
| 26 | `read_file` pagination and window header | paged reads | pass | ReadTools answers reads of cleared or kept results itself (100 lines, 20k chars; `middleware.py:747-838`) before Deep Agents' tool | – | test_react_context::test_older_results_are_cleared_past_half_the_window_and_read_back | works | caveat |
| 27 | multimodal `read_file` (images; video via `[video]`; audio/pdf blocks) | media blocks | pass. The `[video]` extra (PyAV) is not installed | pass | – | none | untested | untested |
| 28 | `offload_binary_content`, blobs, `artifacts_root` | content-addressed media offload | pass | pass | – | none | untested | untested |
| 29 | `max_execute_timeout`, parallel file-mutation guard, symlink scope (0.7.38), `virtual_mode=True` default | safety | pass. The harness sandbox bounds a command by the least of its limits | pass | – | test_sandbox_backend::test_the_command_is_bounded_by_the_least_of_its_limits; `virtual_mode=True` is used in several tests | works | works |
| **Backends** |||||||||
| 30 | StateBackend | thread-scoped files that "persist across turns via checkpoints" (Deep Agents docs) | works with your own checkpointer | `JournalBackend`, a StateBackend subclass with caps | not governed | probe: DEEP_INMEM_TURN2 reads the file | works | caveat |
| 31 | StateBackend files across turns in ReAct | – | – | **lost: turn 2 of the same thread gets `Error: File '/notes.md' not found`** (probe REACT_TURN2). AGENTS.md edits, `/large_tool_results/` and `/conversation_history/` that later replayed turns refer to are also gone | – | test_react_long_tasks::test_files_are_kept_across_a_pause (within one run only) | – | broken |
| 32 | FilesystemBackend | real disk under `root_dir` | file tools governed (reads as reads, writes as writes) | the same (`react.py` `_declared`) | yes | test_deepagents::test_its_file_tools_on_a_disk_are_governed_as_any_tool; test_react_deep_agents::test_on_a_disk_the_file_tools_are_governed_and_offered_from_the_start | works | works |
| 33 | LocalShellBackend | host shell, unisolated | governed (execute is a write) | via `backend=` | yes | test_deepagents_library::test_a_local_shell_backends_execute_runs_the_command_as_a_write_of_the_run | works | untested |
| 34 | StoreBackend (namespace factory) | cross-thread files | governed. The store becomes the memory service when the graph has none | the same | yes | row 13 tests | caveat | untested |
| 35 | CompositeBackend (routes, longest prefix) | mixed storage | **governance is decided per graph: any StateBackend route gives a `files` channel, so all file tools are treated as state tools, including routes to disk** (`middleware.py:323`). Documented in deepagents.md | ReAct decides from `files_backend` before composing (`react.py:388`) | disk route: **not governed or recorded** | test_deepagents_library::test_a_composite_backend_routes_each_path_to_its_backend (asserts `results(events) == []` while `/kept/final.md` is written to disk) | caveat | works |
| 36 | `deep_agent(sandbox(), skills)` / `src.deep_agent(backend)` (harness recipe built on Composite) | sandbox plus a skills route | **broken: the `_SkillsBackend` route subclasses StateBackend (`skills.py:543`), so the graph gets a `files` channel and the sandbox's write_file/edit_file/delete run ungoverned and unrecorded** (probe: `[('execute','ok')]` only; without skills `[('write_file','ok'),('execute','ok')]`) | ReAct governs them (probe REACT_SANDBOX_SKILLS) | D: no | test_skills::test_a_deep_agent_of_your_own_takes_a_sandbox_and_skills_in_one_backend (does not check governance) | broken | works |
| 37 | Sandbox backends (BaseSandbox, LangSmithSandbox, AgentCore/Daytona partner packages) | remote execution | governed (no `files` channel) | pass | yes | none (only the harness's own SandboxBackend: test_sandbox_backend) | untested | untested |
| 38 | ContextHubBackend (LangSmith Hub repo, `/skills/` linked repos) | hub-backed files | would be governed as outside state | pass | yes | none | untested | untested |
| 39 | Backend policy hooks (subclass or wrapper backend) and custom BackendProtocol | custom validation | pass | pass | – | none | untested | untested |
| 40 | `BackendConsumer.bind_backend` (0.7.36) | middleware receives the backend | pass | pass | – | none | untested | untested |
| **Subagents (SubAgentMiddleware, `task`)** |||||||||
| 41 | default `general-purpose` subagent | main tools, isolated context | pass. A `task` call becomes a child run (`middleware.py:1030-1062`) | xform: ReAct passes its own general-purpose spec (`react.py:429`), so Deep Agents does not auto-add one. It **ignores `HarnessProfile.general_purpose_subagent`** (enabled/description/system_prompt) and **gets no SkillsMiddleware** (captured). In lean mode it is held until a long task (machinery) | `task` is governed as a read; the child run is recorded | test_deepagents_library::test_the_general_purpose_sub_agent_works_with_the_main_agents_tools; test_react_deep_agents::test_without_a_deep_agents_capability_react_holds_deep_agents_machinery | works | caveat |
| 42 | declarative `SubAgent` (name, description, system_prompt, tools, model, middleware, skills, permissions, response_format) | isolated helper | pass; child run; memory briefing prefixed to the task description (`middleware.py:1049`: **the task text changes** when memory is on) | xform: model routed; StallGuard, step_limit, ModelHooks and PromptCache added; HarnessTools and GovernTools when it has no `tools` (`react.py:621-655`) | its calls are governed under its child run | test_native_helpers::test_a_deep_agents_sub_agent_is_a_child_run_its_turns_in_memory, test_a_deep_agents_sub_agent_that_fails_is_a_failed_child, test_a_deep_agents_sub_agents_tokens_count_toward_the_runs_budget; test_react_deep_agents::test_a_declarative_sub_agent_is_given_the_loops_guards_and_a_routed_model, test_a_sub_agent_that_names_no_tools_has_the_runs | caveat | caveat |
| 43 | `CompiledSubAgent` (runnable) | prebuilt graph | pass | pass | yes | test_deepagents_library::test_a_compiled_graph_and_a_structured_sub_agent_answer_through_task; test_react_deep_agents::test_a_compiled_graph_is_a_sub_agent_with_its_name_and_description | works | works |
| 44 | `mode="fork"` (sync, experimental) | continues the parent's conversation and system prompt | pass without `middleware()`. With `middleware()` the fork mirrors the whole harness stack (captured) | fork inherits BackgroundTasks, Planning, ReActSummarization and the no-op summarization; ReAct's own guards override by name | child run | test_deepagents_library::test_a_sub_agent_that_forks_the_conversation_reads_it_and_answers_through_task (no `middleware()`); the inherited stack is untested | works | untested |
| 45 | subagent `response_format` (static, and per call via the config key) | structured subagent answer | pass | pass | – | static: row 43 test; per call: none | works | untested |
| 46 | subagent `interrupt_on` (inherits top level, overrides) | HITL inside a helper | pass (a pause inside the helper pauses the run) | **xform: a named tool becomes `side_effects="irreversible"`; the `when` predicate, `allowed_decisions` and `description` are dropped** (`react.py:645-652`); other names stay Deep Agents' gate | catalog approval | test_react_deep_agents::test_a_function_named_in_interrupt_on_becomes_a_tool_that_asks, test_a_sub_agents_tool_named_in_its_interrupt_on_asks_a_person; the D-path subagent is untested | untested | caveat |
| 47 | `permissions` per subagent | rule override | pass | pass | – | none | untested | untested |
| 48 | state passed back (Command update, `_EXCLUDED_STATE_KEYS`, private keys) | subagent files reach the parent | journaled as a Command (`_kept_update`) | the same | journal | none specific | untested | untested |
| 49 | refusal notice (0.7.33), `ModelRefusalMiddleware` | refusals become a task result | pass (`model_error` returns ModelRefusalError unchanged, `middleware.py:1968-1995`) | pass | – | none | untested | untested |
| 50 | `checkpoint_ns` artifact (0.7.36), `lc_subagent_tool_call_id` metadata (0.7.33) | client loads helper history | pass (the harness reads only the content) | pass | – | none | untested | untested |
| 51 | identical `task` calls in one step | Deep Agents runs both | **xform: the second does not run and points to the first (`bridge.same`, `middleware.py:449-454`)**; applies to every non-state tool | the same | journal | test_react_parallel::test_a_call_repeated_in_one_step_runs_once_and_each_repeat_is_answered | caveat | caveat |
| 52 | running without subagents (profile disables general-purpose) | no `task` tool | pass | not possible: ReAct always passes a general-purpose spec | – | none | untested | caveat |
| **Async subagents (AsyncSubAgentMiddleware)** |||||||||
| 53 | `AsyncSubAgent` (name, description, graph_id, url, headers) and the start/check/update/cancel/list tools | background task on an Agent Protocol server | tool metadata declared (start, update, cancel = write; check, list = read: `middleware.py:268,334-337`); each task is a child run held in the run store (`bridges/async_tasks.py`) | the same (`react.py:656` `_remote`) | yes, child run | test_react_async_subagent (about 70 tests: t1-t3, h, a, b, c, g, e, f, i) | works | works |
| 54 | without `url` | in-process Agent Server only | n/r: no LangGraph server in the harness process | the same | – | test_react_async_subagent::test_f_today_without_a_url_the_task_cannot_start | n/r | n/r |
| 55 | `mode="fork"` (async, 0.7.24) | sends a conversation snapshot | pass | pass | – | test_react_async_subagent::test_h_a_start_that_sends_the_conversation_says_what_deep_agents_says | works | works |
| 56 | `webhook` (0.7.30) | outcome callback | pass, **not validated** | checked by `outbound.validate_url` (`react.py:664`) | – | test_react_async_subagent::test_h_a_webhook_is_checked_as_every_url_the_harness_hands_on | caveat | works |
| 57 | `update_strategy` (interrupt/enqueue) | follow-up policy | pass | pass. For harness-run SubAgent dicts it is **popped from the caller's dict** (`react.py:460`) and used by BackgroundTasks | – | test_react_background::test_an_update_sends_the_task_new_instructions_as_its_update_strategy_says, test_an_update_strategy_is_deep_agents_own | works | caveat |
| 58 | `notify_parent` (0.7.30) and `_ParentNotifierMiddleware` | wakes an idle parent on the Agent Server | pass, but inert: a `url` (required) disables the parent reference, and harness runs carry none (no `parent_thread_id` anywhere in `src/trellis`). The harness polls every 10 s instead (`bridges/async_tasks.py:64,173`) | the same | – | none | n/r | n/r |
| 59 | notifications before the model, after a final answer (jump to model), `waiting` status, notification metadata (0.7.26-0.7.33) | parent learns task outcomes | pass. The extra turn counts against `max_steps` | the same | the notification is recorded as a USER-role message with metadata | test_react_async_subagent::test_g_the_next_turn_is_told_how_the_task_went_before_its_model_call, test_g_a_task_that_ends_while_the_turn_works_gives_the_model_another_turn, test_g_the_extra_turn_counts_against_max_steps | works | works |
| 60 | task list (`async_tasks` state) across turns | thread state | own checkpointer: thread. Otherwise carried by the harness (CARRIED) through the process and the file store | carried | – | test_react_async_subagent::test_a_the_task_list_is_carried_whatever_history_the_next_turn_sees, test_c_after_a_restart_the_next_turn_has_the_task_list | works | works |
| 61 | cancelling a turn | Deep Agents leaves remote tasks running | **xform: the harness cancels the tasks the turn started on their server** (`bridges/async_tasks.py:256,265`) | the same | run CANCELLED | test_react_async_subagent::test_e_cancelling_the_turn_cancels_the_task_it_started | caveat | caveat |
| **Skills (SkillsMiddleware)** |||||||||
| 62 | sources, layering (last wins), labels, progressive disclosure (listing in the system prompt, read_file on SKILL.md) | Agent Skills | pass (sources read through the backend) | pass; a harness skill source becomes the `/skills/` route (`skills.py:436`, `react.py:379-381`) | reading SKILL.md is a state or outside-state read | test_deepagents_library::test_memory_files_and_skills_are_read_from_the_backend_into_the_prompt; test_skills::test_react_reads_a_skill_through_deep_agents_own_skills_slot, test_an_edit_applies_from_the_next_run_and_a_resume_keeps_the_pinned_version | works | works |
| 63 | `skills_metadata` cached per thread; reset with `None` (0.7.16) | reload | pass | reloaded every turn (thread = run); harmless | – | none | untested | untested |
| 64 | `pinned_skills` input (0.7.23) | force SKILL.md into the context | pass (an input dict reaches the graph) | pass | – | none | untested | untested |
| 65 | skill tools: `metadata.include_tools`, `SkillsMiddleware(tools=)`, inline disclosure (`_skill_tools`, 0.7.22/0.7.25) | tools disclosed after the skill is read | pass. Dispatched through SkillsMiddleware's `wrap_tool_call`, so still governed (`govern_tool_nodes`) | with `RoutedChatModel` the inline (`tool_addition`) path is skipped and the tool goes to `request.tools` (`_skill_tools.py:379-388`) | should be governed as a graph tool | none | untested | untested |
| 66 | skills for subagents; a fork inherits skills | per-helper skills | pass | the general-purpose helper gets none (row 41) | – | none | untested | caveat |
| **Memory (MemoryMiddleware, AGENTS.md)** |||||||||
| 67 | `memory=[...]` loaded into the system prompt (`<agent_memory>`) | always-on memory | pass | pass (`memory_files=`) | reading it is no tool call | test_deepagents_library::test_memory_files_and_skills_are_read_from_the_backend_into_the_prompt; ReAct: none | works | untested |
| 68 | agent updates memory with `edit_file` | self-improving memory | state backend: ungoverned (state tool); disk or store: a governed write | default JournalBackend: **the edit is lost after the turn** (row 31) | – | none | untested | broken |
| 69 | memory `cache_control` breakpoint (Anthropic only) | caches the memory block | pass | with a model name, `is_anthropic_chat_model(RoutedChatModel)` is False (captured), and PromptCache restores the routed model before Memory runs (`middleware.py:1886-1898`), so **no breakpoint** | – | none | untested | caveat |
| **Summarization / compaction** |||||||||
| 70 | automatic summarization (85% trigger, `/conversation_history/` offload, arg truncation, reasoning removal, cutoff by message id) | compaction | pass. With `middleware()`, ModelHooks `_with_task` re-inserts the question and the leading system messages before the summary (`middleware.py:1762`): **request changed** | xform: Deep Agents' copy is replaced by a no-op (`react.py:423`) and re-placed as `ReActSummarization` after FitWindow, only in capability mode with a window of 32k or more (`react.py:392,408`); otherwise none, and FitWindow drops turns | – | test_deepagents_library::test_deep_agents_summarizes_a_long_conversation_and_keeps_its_history_to_read; test_react_context::test_older_turns_are_summarized_and_their_history_kept_to_read, test_a_summarized_request_still_names_the_results_the_run_keeps; test_react_small_window::test_below_the_window_deep_agents_needs_the_harness_fits_the_turns | caveat | caveat |
| 71 | `ContextOverflowError` fallback (summarize and retry, `_overflow_clip`) | survives overflow | without `middleware()`: native. With it, FitWindow (inner) catches the refusal and refits first (`middleware.py:2622-2640`) | the same | – | test_fit_window (harness side only) | caveat | caveat |
| 72 | `SummarizationToolMiddleware` / `compact_conversation` | compaction on request | the tool declares nothing, so **a person is asked at each call** | the same | approval | test_deepagents_library::test_the_compaction_tool_asks_as_any_tool_that_says_nothing_then_runs | caveat | untested |
| 73 | custom summary prompt, `summarizer`, `trim_tokens_to_summarize`, `system_prompt` slot | tuning | pass (by name replacement) | via `middleware=` (replaces the no-op) | – | none | untested | untested |
| **HITL** |||||||||
| 74 | `interrupt_on` decisions approve/edit/reject/respond, batched calls | human review | approve/edit/reject/answer mapped; **edit changes only the first call of a batch and approves the rest** (`adapters/langgraph.py:752-768`) | blocked (row 8) | approval in the run store | test_hitl_middleware::test_approve_runs_the_call, test_edit_runs_the_call_with_the_edited_arguments, test_reject_tells_the_model_the_reason, test_an_answer_responds_for_the_tool, test_a_batch_of_calls_is_one_approval_and_one_decision_for_all, test_a_decision_the_tool_does_not_allow_is_refused_before_resuming | caveat | blocked |
| 75 | `when` predicate on InterruptOnConfig | argument-conditional gate | pass | dropped for subagent tools (row 46) | – | none | untested | caveat |
| 76 | `interrupt()` inside a tool | custom question | pass (FOREIGN journal key) | pass | yes | test_deepagents::test_a_tool_of_the_deep_agents_own_asking_with_interrupt_offers_what_it_says | works | works |
| **Prompt caching** |||||||||
| 77 | Anthropic caching (lazy, unconditional) | cache breakpoints | without `middleware()`: native. With it: **overridden by PromptCache** (same name, `middleware.py:1859-1910`), also in the general-purpose helper; adds a breakpoint on a replayed history answer | the same; the routed model is resolved for the decision | – | test_prompt_cache::test_an_anthropic_model_however_given_marks_what_the_next_call_repeats, test_a_graph_built_with_the_middleware_is_cached_as_react_is, test_your_own_caching_middleware_is_the_one_used, test_another_providers_request_is_sent_as_it_was | caveat | caveat |
| 78 | Bedrock / Fireworks caching | provider caching | pass (Bedrock installed; Fireworks not) | pass; a routed model is never Bedrock | – | none | untested | untested |
| **Other middleware and profiles** |||||||||
| 79 | `PatchToolCallsMiddleware` | answers dangling calls | pass; the harness skips its own STOPPED_CALL patch when present (`adapters/langgraph.py:140,640`) | pass | – | test_langgraph::test_a_call_a_newer_message_stopped_is_answered_in_the_kept_thread | works | works |
| 80 | `UnsupportedContentMiddleware` | replaces media the model can't take | pass | pass (RoutedChatModel profile) | – | none | untested | untested |
| 81 | `HarnessProfile` (excluded_tools, suffix/base prompt, tool_description_overrides, excluded_middleware, extra_middleware, general_purpose_subagent), `register_harness_profile` | per-model tuning | pass | **not applied for model names** (row 1). With a chat model object it applies, except general_purpose_subagent (row 41) | – | test_deepagents_library::test_a_harness_profile_takes_tools_away_and_adds_to_the_prompt (excluded_tools and suffix, D) | works | broken |
| 82 | built-in profiles (Opus 4.7, Sonnet 4.6, Haiku 4.5, Codex with TodoList, Nemotron 3 Ultra) | shipped tuning | pass | not applied for model names (captured: `anthropic/claude-opus-4-7` gives no suffix) | – | none | untested | broken |
| 83 | `ProviderProfile` (OpenAI Responses API default, OpenRouter/NVIDIA attribution), `register_provider_profile` | init kwargs | pass | skipped: the direct route uses `init_chat_model(name, model_provider=...)` without a profile (`gateways.py:569`); Bifrost uses ChatOpenAI against the gateway | – | none | untested | caveat |
| 84 | `RubricMiddleware` (grader loop, `rubric` input) | self-evaluation | pass | via `middleware=` | the grader's model calls are accounted | test_deepagents_library::test_a_rubric_has_the_answer_graded_and_revised | works | untested |
| 85 | `DeltaChannel` messages/files (O(N) checkpoints) | checkpoint size | RunCheckpointer keeps delta snapshots and writes | the same | journal | contract/test_langchain_internals | works | works |
| 86 | event streaming v3 (`astream_events`, `stream.subagents`) | typed subagent streams | n/r: the harness runs `astream` with values/messages/custom and emits its own events; a `custom` write becomes a CUSTOM event (`adapters/langgraph.py:433`) | the same | events | none for v3 | n/r | n/r |
| 87 | `CodeInterpreterMiddleware` (langchain-quickjs, experimental) | JS eval with programmatic tool calls | pass; not installed. Its `eval` tool would declare nothing, so it would ask a person | via `middleware=` | – | none | untested | untested |
| 88 | `BASE_AGENT_PROMPT` (deprecated, lean default prompt) | legacy prompt | pass | pass (as text) | – | – | works | works |
| 89 | ReAct "lean" holding of Deep Agents' machinery | – | – | blocked by design: task, scratch files and planning are hidden until a long task; probe REACT_LEAN_TOOLS shows no Deep Agents tool offered; documented and measured (react.md "The base") | – | test_react_long_tasks::test_react_alone_on_a_small_or_unknown_window_is_offered_none, test_react_asked_for_a_backend_is_deep_agents_from_its_first_step | – | blocked |
| **CLI / hosting** |||||||||
| 90 | Deep Agents CLI / Deep Agents Code, ACP server, Managed Deep Agents, LangSmith Agent Server deploy, frontend todo-list, OpenWiki | – | n/a | n/a | – | – | n/a | n/a |

### Counts by status

Each of the 90 rows has a D status and an R status (row 31 and row 89 are ReAct-only).

| Status | D | R |
|---|---|---|
| works | 34 | 21 |
| caveat | 15 | 23 |
| broken | 3 (rows 5, 18, 36) | 5 (rows 19, 31, 68, 81, 82) |
| blocked | 0 | 7 |
| untested | 32 | 30 |
| n/r | 3 | 3 |
| n/a | 1 | 1 |
| does not apply | 2 | 0 |

`untested` means the harness passes the feature through and nothing exercises it.

## Gap list

### P0
1. **A documented recipe bypasses governance.**
   - `create_deep_agent(**deep_agent(sandbox(), src))` and `src.deep_agent(any_backend)` run `write_file`, `edit_file` and `delete` ungoverned and unrecorded on the sandbox or disk.
   - Cause: the `/skills/` route is a StateBackend subclass (`skills.py:543`). It gives the graph a `files` channel, and `declare_files` then decides "state" for the whole graph (`middleware.py:323`).
   - Probe: with skills the run records only `[('execute','ok')]`; without skills it records `[('write_file','ok'),('execute','ok')]`.
   - ReAct is unaffected: it decides from the backend before composing (`react.py:388`).
   - The existing test, test_skills::test_a_deep_agent_of_your_own_takes_a_sandbox_and_skills_in_one_backend, does not assert governance.

### P1
2. **ReAct loses Deep Agents' thread state between turns.**
   - The thread is the run (`RunCheckpointer`, `react.py:437`; `adapters/langgraph.py:515`), and only `async_tasks` is carried (`:147`).
   - Lost each turn: StateBackend/JournalBackend files, todos, AGENTS.md edits, `skills_metadata`/`memory_contents`, `/large_tool_results/` and `/conversation_history/`. Later turns' replayed history refers to those paths.
   - Probe: turn 2 gets `Error: File '/notes.md' not found`. With Deep Agents and InMemorySaver the file is read.
   - The Deep Agents docs promise StateBackend "persists across turns via checkpoints". The harness docs do not say otherwise.
3. **ReAct model names lose Deep Agents profiles.**
   - The `RoutedChatModel` (`gateways.py:823`) makes `_harness_profile_for_model` miss: the provider reads as `routedchatmodel`. The provider profile is skipped too (`gateways.py:569`, no `apply_provider_profile`).
   - Consequences: no Opus/Sonnet/Haiku prompt suffixes, no Codex TodoList, no Nemotron profile, no profile `excluded_tools`; the OpenAI Responses-API default is not applied.
   - Captured in `profile_check.py`.
4. **`middleware()` cannot be combined with `TodoListMiddleware`**, which is Deep Agents 0.7's migration path for planning.
   - `create_deep_agent(middleware=[TodoListMiddleware(), *middleware()])` raises `AssertionError: Please remove duplicate middleware instances` (captured).
   - Cause: Planning takes the name `TodoListMiddleware` (`automatic.py:135`).
   - With the default `context_window=None`, `middleware()`'s own Planning never offers `write_todos` (`automatic.py:86-88`), although react.md says "On your own graph, `middleware()` offers planning the same way".

### P2
5. **ReAct's general-purpose helper differs from Deep Agents'.** It is given explicitly (`react.py:429`), so it ignores `HarnessProfile.general_purpose_subagent` (disable, description, prompt) and lacks SkillsMiddleware when `skills=` is set (captured stack).
6. **ReAct subagent `interrupt_on` becomes an unconditional `irreversible` risk** (`react.py:645-652`). `when`, `allowed_decisions` and `description` are dropped.
7. **HITL batch edit changes only the first call and approves the others** (`adapters/langgraph.py:762`). Deep Agents expects one decision per action.
8. **A general CompositeBackend with any StateBackend route leaves disk routes ungoverned.** This is documented in deepagents.md, and test_deepagents_library::test_a_composite_backend_routes_each_path_to_its_backend codifies it.
9. **Memory's `cache_control` breakpoint is lost for routed Anthropic names.** `is_anthropic_chat_model(RoutedChatModel)` is False, and PromptCache restores the routed model before MemoryMiddleware runs.
10. **FitWindow, ReActSummarization and ModelHooks sit outside SkillsMiddleware and MemoryMiddleware.** The skills listing and AGENTS.md are neither counted by FitWindow nor shown in the `chat` span (captured order). A large AGENTS.md can push a "fitted" request over the window.
11. **The fork subagent's inherited harness stack is untested**: BackgroundTasks, Planning, scoped step limit, FitWindow (captured).
12. **An AsyncSubAgent `webhook` is validated only in ReAct** (`react.py:664`). On the direct path it passes unchecked.

### P3
13. **`_strategies` mutates the caller's subagent dict** (`given.pop("update_strategy")`, `react.py:460`).
14. **Untested pass-throughs:**
    - state_schema; context_schema in ReAct;
    - dynamic subagent response_format; subagent permissions inheritance; D-path subagent interrupt_on;
    - `pinned_skills`; skill tools (`include_tools`); `skills_metadata` reset;
    - ContextHubBackend; LangSmith/BaseSandbox and partner sandboxes; policy-hook backends; `bind_backend`;
    - FilesystemMiddleware `tools=`, `grep_max_count`, HumanMessage eviction, `offload_binary_content`/`artifacts_root`, video/multimodal reads;
    - `excluded_middleware`, `tool_description_overrides`, running without subagents;
    - refusal notice; `checkpoint_ns` artifact; UnsupportedContentMiddleware; Bedrock/Fireworks caching;
    - Rubric, permissions and memory_files under ReAct; `compact_conversation` under ReAct;
    - CodeInterpreterMiddleware; `notify_parent`.
15. **Documentation drift:**
    - docs/matrix.md has no rows for Deep Agents' own features.
    - deepagents.md's Planning row does not mention the `middleware()` collision.
    - final-agent-harness still pins `deepagents>=0.7.19` and lacks the async-task bridge and BackgroundTasks.

## Overlaps (two mechanisms on one concern)

| Concern | Deep Agents | Harness | Interaction |
|---|---|---|---|
| Compaction | SummarizationMiddleware (85%), overflow fallback, eviction | FitWindow (95%), ContextEditing clearing (50%), kept results, `_with_task` | ReAct: one owner (no-op plus ReActSummarization after FitWindow with `turns=False`, or FitWindow alone). `middleware()`: both are active: DA summarizes at 85% (outer), FitWindow trims at 95% (inner) and pre-empts DA's overflow fallback |
| Large results | `/large_tool_results/` eviction plus `read_file` | run-kept results, ReadTools, `query_result` | ReAct lean: the harness owns them; capability: DA saves them and `query_result` reads from the state. ReAct's per-turn state makes saved paths dangling in later turns |
| Memory | AGENTS.md in the system prompt, edited with `edit_file` | memory service context (SystemMessage moved into the user turn), memory tools, StoreBackend → memory service | Independent and allowed together. AGENTS.md edits on the default ReAct backend do not persist; the memory service does |
| Skills | SkillsMiddleware (listing plus read_file, `include_tools`) | skill sources pinned per run (`/skills/` route), learned skills in the memory context, `list_skills`/`load_skill` for slot-less engines | Deep Agents' slot is used (no duplicate tools). The route triggers the P0 governance bypass on the direct path. Learned skills arrive separately in the memory context |
| Files | StateBackend/backends, permissions | governance of outside-state file tools, JournalBackend caps, SCRATCH_NOTE | The per-graph `files` channel test decides governance for all routes (P0, P2-8) |
| HITL | `interrupt_on`, permission interrupts | catalog approvals, `ask`, harness resume mapping | Direct: mapped (edit only the first call). ReAct: `interrupt_on` blocked, subagent gates converted (`when` lost) |
| Planning | opt-in TodoListMiddleware | Planning (TodoList renamed), long-task trigger | Name collision crashes the build (P1-4) |
| Caching | lazy Anthropic, Bedrock, Fireworks, memory breakpoint | PromptCache (replaces Anthropic by name, extra history breakpoint) | Memory breakpoint lost on routed names (P2-9) |
| Background tasks | AsyncSubAgentMiddleware (remote) | BackgroundTasks (local, the same tool names), AsyncTasks child runs, 10 s polling, cancel with the turn | Not mixed in one ReAct. The harness adds cancel-on-turn-cancel and polling |
| Step limits | recursion_limit 9999 | ModelCallLimit `max_steps=12`, StallGuard, budget | With `middleware()`, a Deep Agent is capped at 12 model calls by default |

## Ways the harness changes Deep Agents' behaviour

The user's rule is that library behaviour must not change.

**On every harness run** (global ToolNode patch, `middleware.py:1078`; `declare_files`):
1. The first scratch `write_file`/`edit_file` result gets `SCRATCH_NOTE` appended.
2. A repeated identical non-state call in one step (including `task`) does not run.
3. The `task` description is prefixed with a memory briefing when memory is on.
4. Deep Agents' tool `metadata` is mutated (`side_effects`, `trellis_outside_state`, `trellis_machinery`).
5. File tools outside the state, `execute`, async-task tools and `compact_conversation` go through governance. Undeclared tools such as `compact_conversation` ask a person.
6. A graph with no store gets the memory service as its store.
7. A turn's cancel cancels the remote async tasks it started; tasks are polled every 10 s.

**With `middleware()`:**
1. PromptCache replaces Deep Agents' Anthropic caching, in the main agent and the general-purpose helper.
2. The date is appended to the system prompt.
3. The question is re-inserted ahead of a summary.
4. The memory context is moved into the user message.
5. The default is 12 model calls, plus StallGuard.
6. Unhandled tool errors become model-readable results instead of failing the graph (test_tool_errors_library::test_under_the_harness_middleware_an_error_nothing_handles_is_read_by_the_model).
7. Planning is added (and collides with a user TodoListMiddleware).
8. FitWindow trims requests.

**In ReAct:**
1. Thread state is per turn.
2. Model routing bypasses Deep Agents' harness and provider profiles.
3. Lean mode hides Deep Agents' tools and removes summarization; summarization is also removed on windows under 32k.
4. JournalBackend caps file sizes.
5. The general-purpose helper is replaced (no profile settings, no skills).
6. Subagent `interrupt_on` is converted into a catalog risk.
7. Subagents get the loop guards and PromptCache.
8. `interrupt_on`, `checkpointer`, `store`, `name` and `debug` are not exposed.
9. `update_strategy` is popped from the caller's dict.
