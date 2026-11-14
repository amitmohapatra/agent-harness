# OpenAI Agents SDK 0.24.0: complete feature inventory against the Trellis harness

Read-only audit, 2026-11-14. No code was changed and no paid model calls were made. Probes ran
offline against the scripted model (`tests/support/openai_model.py`) and an unreachable chat
completions client; their scripts are in `/var/tmp/cons/inventory-tmp/openai/`.

## Sources

- **SDK source:** `/var/tmp/cons/oaiup/.venv/lib/python3.13/site-packages/agents` (320 modules,
  `__version__` 0.24.0, 253 public names in `agents.__all__`). Every package was read: core, `run_internal`,
  `models`, `memory`, `mcp`, `tracing`, `realtime`, `voice`, `sandbox` (capabilities, memory, entries,
  mounts, sandboxes, session), `extensions` (memory, models, sandbox providers, experimental), `repl`
  and `testing`.
- **Harness:** `/var/tmp/cons/oaiup` (branch `claude/cool-maxwell-w5avo8-openai-upgrade`, commit
  `cf8bc03`). Its `src/` is the same as `/var/tmp/cons/final-agent-harness` apart from
  `compat.py`, which holds the version pin, so every finding applies to both. The files the brief
  names live under `src/trellis/harness/`, not `src/trellis/`.
  - `A` = `src/trellis/harness/adapters/openai_agents.py`
  - `C` = `src/trellis/harness/tools/convert/openai_agents.py`
  - `L` = `src/trellis/harness/tools/convert/openai_local.py`
  - `K` = `src/trellis/harness/hooks/openai_agents.py`
  - `S` = `src/trellis/harness/skills_openai.py`
  - `D` = `docs/frameworks/openai-agents.md`
- **Tests:**
  - `T1` = `tests/integration/test_openai_agents.py`
  - `TF` = `test_openai_agents_features.py`
  - `TM` = `test_openai_agents_mcp.py`
  - `TL` = `test_openai_local_tools.py`
  - `TN` = `test_native_helpers.py`
  - Other files are named in full. All 469 tests in the 13 OpenAI-related files pass under 0.24.0
    (they need `TRELLIS_MEMORY_OPENAPI`/`TRELLIS_RUNS_OPENAPI` pointed at the `final-*` siblings).
- **Docs and release notes:**
  - `openai.github.io` is **blocked**: DNS fails because it is outside this session's allowed hosts.
  - The GitHub REST and GraphQL APIs are **blocked** for this repository, which is not attached to
    the session.
  - The release pages on `github.com` (v0.23.0, v0.24.0) and `raw.githubusercontent.com`
    (`docs/llms.txt`, `docs/sandbox/memory.md`) were read through WebFetch instead.
- **Release notes in brief:**
  - 0.24: `tool_error_formatter` gains the `tool_exception` kind. The RunState schema becomes 1.21,
    which older SDKs cannot read. MCP `mcp_annotations` are added. Automatic compaction is deferred
    while approvals are pending.
  - 0.23: the default tool-failure text becomes a fixed generic sentence. Encrypted session
    envelopes are required. Sandbox memory consolidation turns can be configured. Premature Chat
    Completions stream EOF is rejected.

## The Chat Completions constraint

On the Bifrost route (`LLM_GATEWAY=bifrost`, the only route with `Capability.ATTRIBUTION`), `A:1035-1056`
replaces the run's default `MultiProvider` with `BifrostProvider`. That provider is
`OpenAIProvider(..., use_responses=False)` (`A:1104`), so every agent model *named by string*
goes through `OpenAIChatCompletionsModel`. The SDK's Chat Completions path:

- raises `UserError("Hosted tools are not supported with the ChatCompletions API")` for **every
  non-`FunctionTool`** (`models/chatcmpl_converter.py:1018-1038`).
  - Probe `p_chat.py` confirms this for WebSearch, FileSearch, CodeInterpreter, ImageGeneration,
    HostedMCP, LocalShell, Shell, ApplyPatch, Custom and ToolSearch. ComputerTool takes the same
    branch (`isinstance(tool, FunctionTool)` is false).
- rejects `defer_loading`, `tool_namespace()`, `allowed_callers`, `output_json_schema`
  (`tool.py:1710-1739`), `MCPToolChoice`, `tool_choice="programmatic_tool_calling"` and
  `compaction` input items (`chatcmpl_converter.py:87, 1006`).
- **silently ignores, with one warning,** `prompt=`, `reasoning.mode/context`,
  `previous_response_id` and `conversation_id` (`openai_chatcompletions.py:83-128, 543-573`). It
  never sends `truncation`, `store`, `response_include` or `context_management`.

The constraint does **not** apply:

- to an agent whose `model` is a `Model` object (the harness warns once: `A:1059-1068`);
- to a run whose `run_config` names its own `model_provider` (`A:1049-1050`);
- on the `openai-compatible` and `provider-sdk` routes, where the SDK default (Responses, straight to
  OpenAI with `OPENAI_API_KEY`) is kept.

**Is that acceptable?** The choice itself is defensible: Bifrost fronts many providers and Chat
Completions is the common format. But the harness does not *say* so per feature. Several features
look supported in `D` and then fail mid-run with an SDK `UserError`, or silently do nothing:

- the hosted tools (`D:122`);
- the shell, patch, computer and custom tools (`D:123`), whose governance code (`L`, 386 lines) is
  unreachable on the default route;
- the stored prompt, which `D:154-157` says passes "untouched";
- `SandboxAgent`'s default capabilities;
- compaction;
- server-side threads.

That is the gap: an honest refusal or doc row per feature, plus an opt-in Responses route through
Bifrost for OpenAI models (P2-7).

## Status key

| Status | Meaning |
|---|---|
| **works** | handled and covered by a test |
| **caveat** | works, but with a limit or a change to SDK behaviour |
| **broken** | does not work, or behaves wrongly, through the harness |
| **blocked** | blocked by design; the reason is given |
| **untested** | passed through and expected to work; no test |
| **n/r** | not reachable through the harness |

## Inventory

### 1. Agent fields

| # | Feature | What it does | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|---|
| 1 | `name` | agent identity, handoff tool name | kept on the clone (`A:435`) | n/a | all `T1` | works |
| 2 | `instructions` (str or callable) | system prompt | passed through; the day is appended to the instructions of every model call (`A:994-1006`) | n/a | `test_today.py::test_an_openai_agent_is_told_the_day_after_its_instructions`, `TF::test_the_runs_context_object_reaches_tools_instructions_and_agent_hooks` | caveat (system prompt changed) |
| 3 | `prompt` (`Prompt`, `DynamicPromptFunction`) | stored OpenAI prompt (Responses) | passed through; on Bifrost the SDK drops it with a warning (`openai_chatcompletions.py:83-100`); `D:154` says "untouched" | n/a | none | blocked (Responses-only; silent drop) |
| 4 | `handoff_description` | describes the agent as a handoff target | read for the tool spec (`A:764`) | n/a | `T1` handoff tests | works |
| 5 | `tools` | the agent's tools | cloned; each tool governed (`C:89-177`) and held at the gate (`C:233-254`); **re-sorted by name** (`A:446-449`) | yes | `T1::test_the_agent_gets_the_harness_tools_next_to_its_own` | caveat (tool order changed) |
| 6 | `mcp_servers` | local MCP servers | wrapped in `GovernedServer` (`A:450, 480-628`) | yes | `TM` (10 tests) | works |
| 7 | `mcp_config.failure_error_function` | MCP failure text, or raise | honoured, with the hooks' words (`A:529-545`) | yes | `TM::test_a_call_the_server_could_not_take_is_told_as_its_failure_function_says` | works |
| 8 | `mcp_config.include_server_in_tool_names` / `convert_schemas_to_strict` | prefixed names; strict schemas | passed through on the clone; governance and records use the bare MCP name (`A:606-612`) | yes | none | untested |
| 9 | `handoffs` | delegation targets | cloned recursively, cycles kept (`A:451-457`) | target's tools yes | `T1::test_an_agent_handed_off_to_has_its_own_tools_governed_by_the_catalog` and 4 more | works |
| 10 | `model` as a string | model name | routed to Bifrost as `provider/model` (`A:1035-1107`) | attribution headers | `test_gateways.py::test_an_openai_agent_by_name_goes_through_bifrost` | works |
| 11 | `model` as a `Model` object | custom model | kept; bypasses Bifrost; warned once (`A:1051-1054`) | no gateway attribution | `test_gateways.py::test_a_run_config_with_its_own_provider_or_a_model_object_is_kept` | caveat |
| 12 | `model_settings` | sampling and tool settings | passed through | n/a | `TF::test_the_agents_model_settings_reach_the_model_as_the_sdk_sends_them` | works |
| 13 | `input_guardrails` | checks before the first turn | passed through | tripwire means a failed run | `TF::test_an_input_guardrail_that_trips_fails_the_run_before_any_tool` | works |
| 14 | `output_guardrails` | checks on the final output | passed through | yes | `TF::test_an_output_guardrail_that_trips_fails_the_run_with_no_answer` | works |
| 15 | `output_type` | structured final output | passed through; kept as JSON in the record | yes | `TF::test_a_structured_answer_is_the_agents_output_type_after_an_approval_too` | works |
| 16 | `hooks` (`AgentHooks`) | per-agent lifecycle | passed through | n/a | `test_hooks.py::test_openai_agents_model_calls_are_its_own_agent_hooks_s` | works |
| 17 | `tool_use_behavior` (`run_llm_again`, `stop_on_first_tool`, `StopAtTools`, a function) | whether a tool result ends the run | passed through, but the tool's result reaching it is the **harness's text**, not the tool's object (`C:173`) | yes | `TF::test_tool_use_behavior_ends_the_run_with_a_tools_result` (string outputs only); probe `p_custom.py`: the function receives `str` instead of a pydantic `Ship` | caveat (typed outputs flattened) |
| 18 | `reset_tool_choice` | reset `tool_choice` after a tool call | passed through | n/a | `TF` model-settings test | works |
| 19 | An `Agent` subclass defined in your own module | custom agent classes | **refused**: `detect` requires `type(target).__module__.startswith("agents.")` (`adapters/__init__.py:30-34`) | none | probe `p_sub.py`: `ConfigurationError` | broken |
| 20 | `SandboxAgent` (`default_manifest`, `base_instructions`, `capabilities`, `run_as`) | sandboxed agent | accepted (its module is `agents.sandbox`); needs `RunConfig(sandbox=)` | see §11 | `test_framework_options.py::test_an_openai_sandbox_agent_runs_wrapped_with_its_run_config_as_an_option` | works |

### 2. Runner and run arguments

| # | Feature | What it does | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|---|
| 21 | `Runner.run` | async run | called by the harness (`A:222`); `Runner.run` called directly is not intercepted (`D:73`) | yes | all `T1` | works |
| 22 | `Runner.run_streamed` | streamed run | `A:241` | yes | `T1::test_streaming_emits_text_deltas` | works |
| 23 | `Runner.run_sync` | synchronous run | harness runs are async | n/a | none | blocked (design) |
| 24 | `starting_agent` / `input` | entry agent and input | refused as options (`A:330-335`) | n/a | `test_framework_options.py::test_options_a_framework_cannot_take_are_refused_before_any_run` | blocked (design) |
| 25 | `context` | run context object | passed through | n/a | `TF::test_the_runs_context_object_reaches_tools_instructions_and_agent_hooks`, `TF::test_what_a_tool_put_in_the_context_is_there_after_an_approval` | works |
| 26 | `max_turns` | turn limit | passed through; **the harness installs a `max_turns` error handler** that makes one more tool-less model call instead of raising `MaxTurnsExceeded` (`A:855-893`) | warning event | `T1::test_at_its_turn_limit_a_run_ends_with_one_last_answer`, `T1::test_a_structured_answer_or_a_teams_handler_keeps_its_way`, `T1::test_a_last_call_that_fails_leaves_the_sdks_error` | caveat (behaviour change) |
| 27 | `hooks` (`RunHooks`) | run lifecycle | wrapped by `ModelHooks`, which calls yours (`K:32-104`) | n/a | `TF::test_the_runs_own_run_hooks_see_the_run_beside_the_harnesss` | works |
| 28 | `run_config` given as a `RunConfig` | run configuration | passed through with harness changes (§3) | n/a | many | caveat |
| 29 | `run_config` given as a dict | the SDK coerces dicts (`run_config.py:632`) | `dataclasses.replace(dict)` raises **`TypeError`** in `_told_unknown_tools` (`A:903-904`); `D:91` documents dicts for queued runs | the run ends ERROR | probe `p_dict.py`: `status ERROR code='TypeError'` | **broken** |
| 30 | `error_handlers` (`max_turns`, `model_refusal`, `invalid_final_output`) | recover from run errors | yours win over the harness's `max_turns` default (`A:891-892`) | n/a | `T1::test_a_structured_answer_or_a_teams_handler_keeps_its_way` (`max_turns` only) | works (`max_turns`); untested (other two) |
| 31 | `previous_response_id` | server-side thread | the harness treats it as a durable thread and **drops memory's window** (`A:1112-1117`); on Bifrost, Chat Completions ignores it (warning) | n/a | `T1::test_who_keeps_the_thread_decides_memorys_window` (classification only) | **broken on Bifrost** |
| 32 | `auto_previous_response_id` | server chaining inside a run | as #31; the SDK's server tracker also sends only *new* items after turn 1 (`run_internal/oai_conversation.py:518-560`), which Chat Completions cannot use | n/a | none | **broken on Bifrost** |
| 33 | `conversation_id` | OpenAI Conversations thread | as #31 | n/a | `T1::test_who_keeps_the_thread_decides_memorys_window` | **broken on Bifrost** |
| 34 | `session` | client-side history | passed through; a durable session drops memory's window; **one session object on `h.wrap` is shared by every user and thread** | n/a | `T1::test_a_session_keeps_the_thread_so_memorys_window_is_not_sent`, `T1::test_a_durable_session_keeps_the_thread`; probe `p_sess.py`: bob's model input holds alice's "my secret is 42" | **broken (cross-user exposure)** |

### 3. RunConfig fields

| # | Feature | What it does | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|---|
| 35 | `model` | run-wide model | a string goes through Bifrost; a `Model` object bypasses it **with no warning** (`_routed` checks only agents, `A:1051-1054`) | n/a | `TF` model-settings test (object) | caveat |
| 36 | `model_provider` | model resolution | your own provider is kept; the default `MultiProvider` is replaced, **including one you configured** (`provider_map`, prefix modes; `A:1049`) | n/a | `test_gateways.py::test_a_run_config_with_its_own_provider_or_a_model_object_is_kept` | caveat (override) |
| 37 | `model_settings` | run-wide settings | passed through | n/a | `TF` model-settings test | works |
| 38 | `handoff_input_filter` | global handoff filter | passed through | n/a | `TF::test_a_handoff_objects_filter_callback_and_typed_input_are_the_sdks` | works |
| 39 | `nest_handoff_history`, `handoff_history_mapper` | nested history on handoff | passed through | n/a | none | untested |
| 40 | `input_guardrails`, `output_guardrails` (run level) | run-wide guardrails | passed through | n/a | none | untested |
| 41 | `tracing_disabled`, `tracing`, `trace_include_sensitive_data`, `workflow_name`, `trace_id`, `group_id`, `trace_metadata` | SDK tracing | passed through; the SDK's export to OpenAI is left on, said once at INFO (`A:143-158`) | n/a | `T1::test_the_sdks_own_tracing_is_left_as_set_and_said_once` | caveat (#141) |
| 42 | `session_input_callback` | merge history with new input | passed through | n/a | `TF::test_the_runs_session_settings_and_callback_shape_what_memorys_thread_gives` | works |
| 43 | `call_model_input_filter` | edit each model call's input | **wrapped**: yours runs first, then the harness moves the memory context into the user turn, appends the day and inserts steered messages (`A:908-1032`) | n/a | `TF::test_the_memory_context_is_marked_in_the_question_wherever_the_callback_puts_it` | caveat (input changed) |
| 44 | `tool_error_formatter` (incl. 0.24's `tool_exception`) | model-visible tool error text | passed through | yes | `TF::test_the_run_wide_tool_error_formatter_words_a_governed_tools_failure` | works |
| 45 | `session_settings` | session read limit | passed through, applied to `MemorySession` too | n/a | `TF` session test | works |
| 46 | `reasoning_item_id_policy` | reasoning item ids (Responses) | passed through | n/a | none | untested |
| 47 | `sandbox` (`SandboxRunConfig`) | sandbox client, session, manifest, snapshot, limits | passed through | see §11 | `test_skills.py`, `test_framework_options.py` | works |
| 48 | `tool_execution` (`max_function_tool_concurrency`, `pre_approval_tool_input_guardrails`) | tool concurrency; guardrails before approval | passed through | n/a | none | untested |
| 49 | `tool_not_found_behavior` | made-up tool: raise or tell the model | **forced to `return_error_to_model`**, overriding an explicit `raise_error` (`A:896-905`) | n/a | `T1::test_a_tool_the_model_made_up_is_an_error_it_reads_and_the_run_goes_on` | caveat (override) |
| 50 | `tool_name_collision_policy` | duplicate tool names | passed through (the harness skips adding a harness tool your agent already has, `A:443-448`) | n/a | none | untested |
| 51 | `output_guardrail_blocked_message` | text when an output guardrail blocks | passed through | n/a | none | untested |

### 4. ModelSettings

| # | Feature | What it does | Harness | Tested | Status |
|---|---|---|---|---|---|
| 52 | `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`, `tool_choice`, `parallel_tool_calls` | sampling and tool settings | passed through to Chat Completions | `TF` model-settings test (`tool_choice`, `parallel_tool_calls`, `temperature`, `top_p`) | works |
| 53 | `verbosity`, `top_logprobs`, `metadata`, `extra_query`, `extra_body`, `extra_headers`, `extra_args` | extras | passed through (`openai_chatcompletions.py:707-787`); `extra_args={"background":...}` is pinned as never polled | `test_background_by_adapter.py::test_b_today_a_responses_background_request_is_answered_queued_and_never_polled` | untested (except `extra_args`) |
| 54 | `reasoning` | effort, mode, context | `effort` is sent; `mode`/`context` are dropped on Chat Completions with a warning | none | caveat |
| 55 | `truncation`, `store`, `response_include`, `context_management` | Responses-only settings | never sent on Chat Completions | none | blocked (Responses-only; silent) |
| 56 | `prompt_cache_retention`, `prompt_cache_options` | prompt caching | sent on chat (`:709-710`) | none | untested |
| 57 | `include_usage`, `preserve_raw_usage` | usage reporting | usage is counted toward the budget (`K:96-104`) | `T1::test_a_token_budget_stops_the_run_without_any_hooks` | works (usage); untested (raw) |
| 58 | `retry` (`ModelRetrySettings`), `retry_policies`, `RetryPolicy` | SDK-managed model retries | passed through; the Bifrost client's own `max_retries=0` (`A:1103`, `gateways.py:127`) because the gateway retries | none | untested (client retries overridden) |
| 59 | `timeout` | model call timeout | passed through; the harness stall watch is separate (`K:66-79`) | none | untested |

### 5. Tools

| # | Feature | What it does | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|---|
| 60 | `@function_tool` / `FunctionTool` | Python tool | governed copy (`C:89-177`) | yes: catalog, approval, journal, record | `test_govern_framework_tools.py::test_an_sdk_agents_own_tool_is_governed_on_a_copy`, `T1` | works |
| 61 | `failure_error_function` (default / custom / `None`) | failure text, or raise | the SDK's policy is kept through `_told` (`C:210-230`) | failed call recorded | `TF::test_a_failing_tool_tells_the_model_what_its_failure_function_says`, `TF::test_a_failure_the_sdk_tells_the_model_is_recorded_as_a_failed_call`, `TF::test_a_tool_that_asks_to_raise_ends_the_run_with_its_error` | works |
| 62 | `strict_mode` / `strict_json_schema` | strict schema | kept by `dataclasses.replace` | n/a | none | untested |
| 63 | `is_enabled` | dynamic availability | wrapped with the harness's tool-hint narrowing (`C:289-302`) | n/a | `test_native_tools.py::test_an_agents_own_tool_keeps_its_own_enabled_rule` | works |
| 64 | `needs_approval` (bool / callable) | SDK human approval | the SDK pause is kept as a RunState checkpoint (`A:265-277, 368-384`) | approval recorded as feedback | `T1::test_the_sdks_own_approval_pauses_and_resumes_its_run_state` and 7 more | works |
| 65 | `timeout`, `timeout_behavior` | per-tool time limit | the SDK's, recorded as a timeout (`C:180-207`) | yes | `TF::test_a_tools_own_time_limit_is_the_sdks_and_its_call_is_recorded_timed_out` and 2 more | works |
| 66 | `timeout_error_function` | timeout text | passed through | yes | none | untested |
| 67 | `tool_input_guardrails` / `tool_output_guardrails` | per-tool guardrails | passed through (the output guardrail sees the harness's text) | yes | `TF::test_tool_guardrails_decide_what_runs_and_what_the_model_reads`, `TF::test_a_tool_guardrail_that_raises_fails_the_run` | works |
| 68 | `defer_loading`, `tool_namespace()`, `allowed_callers`, `output_json_schema` / `output_type` | Responses-only tool features | the SDK rejects them on Chat Completions (`tool.py:1710-1739`) | n/a | probe `p_chat.py` (`defer_loading`) | blocked (Responses-only) |
| 69 | `custom_data_extractor` (0.24) | attach custom data to the output item | kept, but it receives the harness's text | n/a | none | untested |
| 70 | Tool outputs: `str` | text result | passed as text | yes | many | works |
| 71 | Tool outputs: `dict` / pydantic / dataclass | structured result | the harness returns JSON text (`C:173`, `results.readable`); the SDK sends `str(obj)` and keeps the object for `tool_use_behavior`, hooks and items | yes | probe `p_outputs.py`, `p_custom.py` | caveat (behaviour change) |
| 72 | Tool outputs: `ToolOutputImage`, `ToolOutputFileContent`, `ToolOutputText`, lists | multimodal tool results | **stringified**: the model gets `"type='image' image_url=... "` instead of an `input_image` part | recorded as text | probe `p_outputs.py` | **broken** |
| 73 | Arguments that are not JSON | bad arguments | the SDK's own tools: handed back to the SDK (`C:137-138`); harness tools: `not_run` text | yes | `test_tool_arguments.py::test_an_sdk_approval_of_arguments_that_are_not_json_pauses_with_them_as_written` | works |
| 74 | Identical call repeated in one reply | — | runs once; the repeat reads the first result (`C:139-141, 385-393`; `K:123-131`) | yes | `T1::test_a_call_repeated_in_one_reply_runs_once_the_agents_own_tool_too` | caveat (behaviour change) |
| 75 | Transient error in a read tool | — | **retried** by the bridge (`READ_RETRIES=2`, `tools/base.py:298-334`) | yes | probe `p_retry.py`: SDK 1 call, harness 3 | caveat (behaviour change) |
| 76 | Large or tabular results | — | over 80k chars becomes a preview stub; tables get a query note (`results.py:133-217`) | kept in the run | `T1::test_a_large_result_is_previewed_and_read_in_pages_across_a_pause` | caveat (output text changed) |
| 77 | Tool-hint narrowing / stall withdrawal | — | your own tools can be hidden per turn; all tools are withdrawn after a stall (`C:289-302, 420-424`) | n/a | `T1::test_the_hints_narrow_the_tools_each_turn_the_teams_own_too`, `TL::test_a_model_making_the_same_computer_call_step_after_step_is_stopped_and_told` | caveat (behaviour change) |
| 78 | `WebSearchTool` | hosted web search | recorded from the reply (`A:1250-1309`); a Chat Completions `UserError` on Bifrost | recorded, not approvable | none (no test names it) | blocked on Bifrost; untested off it |
| 79 | `FileSearchTool` | hosted vector search | as #78 | recorded | `T1::test_the_providers_hosted_tool_calls_are_recorded_with_the_steps_duration` (scripted Responses items) | blocked on Bifrost; works off it |
| 80 | `CodeInterpreterTool` | hosted code execution | as #78 | recorded | same `T1` test | blocked on Bifrost; works off it |
| 81 | `ImageGenerationTool` | hosted image generation | as #78; no bytes are recorded | recorded | none | blocked on Bifrost; untested off it |
| 82 | `HostedMCPTool` (+ `on_approval_request`) | OpenAI-side MCP | allowed tools listed (`A:754-759`); an approval request pauses | recorded | `TF::test_a_hosted_mcp_tools_approval_request_pauses_or_asks_its_own_callback`, `test_native_tools.py::test_an_openai_agents_own_tools_handoffs_and_mcp_tools_are_its_tool_set` | blocked on Bifrost; works off it |
| 83 | `ToolSearchTool` (0.24) | hosted or client tool search | listed by name (`A:760-761`); `UserError` on chat | none | probe `p_chat.py` | blocked |
| 84 | `ProgrammaticToolCallingTool` | Responses programmatic calling | as #83 | none | none | blocked |
| 85 | `ComputerTool` / `ComputerProvider` / safety checks | computer use | every action goes through the bridge (`L:264-337`); a refused action is skipped and explained in the next input (`A:1025`) | yes | `TF::test_a_computers_actions_are_governed_and_recorded_one_by_one` and 6 `TL` tests (scripted) | blocked on Bifrost (not a `FunctionTool`); works off it |
| 86 | `LocalShellTool` | local shell | executor wrapped (`L:180-199`) | yes | `TF::test_a_local_shell_call_asks_and_runs_once_recorded`, `TL::test_an_edited_local_shell_command_runs_as_edited` | blocked on Bifrost; works off it |
| 87 | `ShellTool` with a local executor | shell tool | executor wrapped (`L:84-85`) | yes | `TF::test_a_shell_command_the_sdk_runs_here_asks_and_runs_once_recorded` and 4 more | blocked on Bifrost; works off it |
| 88 | `ShellTool` with a hosted environment (container auto/reference, skills, network policy) | OpenAI-hosted shell | no executor, so not wrapped (`L:84`) | no | none | blocked (Responses-only) |
| 89 | `ApplyPatchTool` / `apply_diff` | file patching | editor wrapped (`L:232-261`) | yes | `TF::test_a_patch_and_a_custom_tool_ask_before_they_run`, `TL::test_a_patchs_create_and_update_operations_are_each_a_call` | blocked on Bifrost; works off it |
| 90 | `CustomTool` | free-form tool | invoke wrapped (`L:216-229`) | yes | `TL::test_an_approved_custom_call_runs_once_with_its_input` | blocked on Bifrost; works off it |
| 91 | `on_approval` callbacks (shell, patch, custom) | auto-decide an approval | kept; its decision counts as governance's | yes | `TF::test_a_shell_tools_own_approval_is_the_one_question_for_its_call` | works off Bifrost |

### 6. Agents as tools

| # | Feature | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|
| 92 | `Agent.as_tool` as a child run | the closure is rewritten to run the governed copy (`A:635-662`); a helper child run (`C:305-354`) | yes: child record, budget, memory | `TN::test_an_agent_called_as_a_tool_is_a_child_run_under_the_catalogs_rules`, `TN::test_an_agent_tool_that_fails_is_a_failed_child` | works |
| 93 | `parameters`, `custom_output_extractor`, `on_stream`, `input_builder`, `include_input_schema` | kept (`dataclasses.replace`) | yes | `TF::test_an_agent_tools_own_options_are_the_sdks` (first three) | works / untested (last two) |
| 94 | `needs_approval` on, or inside, the agent tool | the SDK's nested RunState, continued | yes | `TF::test_an_sdk_approval_inside_an_agent_tool_continues_the_agent_inside` and 4 more | works |
| 95 | `failure_error_function` | the SDK's policy | failed child | `TN::test_an_agent_tool_that_asks_to_raise_ends_the_run_as_the_sdk_does` | works |
| 96 | Its own `run_config` / `session` / `previous_response_id` / `conversation_id` / `hooks` / `max_turns` | passed through; **its own `run_config` escapes the harness's**: no Bifrost provider, no day, no steering, no `tool_not_found` override | its tools are still governed; its model calls bypass the gateway | probe `p_nested.py`: inner system prompt is `'OWN'` with no day line, while the inherited one has it | caveat (gateway bypass) |
| 97 | Agent tool of an unknown shape | runs ungoverned, warned once (`A:812-836`) | no | none | caveat |
| 98 | Model calls inside the child | no stall watch or hosted-call record (the nested run gets `as_tool`'s hooks, not `ModelHooks`); without its own `run_config` it inherits the harness's filters, steered messages included (`A:1009-1032`) | partial | none | caveat (untested) |

### 7. Handoffs

| # | Feature | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|
| 99 | Handoff to an `Agent` | cloned, governed copy | the target's tools, yes; the handoff itself is not gated (listed as a read helper, `A:762-767`) | `T1::test_a_deny_all_rule_blocks_the_tool_of_an_agent_handed_off_to` and 4 more | works |
| 100 | `handoff()` with `tool_name_override`, `input_filter`, `on_handoff`, `input_type` | `_handed` keeps them (`A:687-709`) | yes | `TF::test_a_handoff_objects_filter_callback_and_typed_input_are_the_sdks`, `T1::test_a_handoff_objects_agent_is_governed_and_recorded_as_the_runs` | works |
| 101 | `Handoff` built by hand | governed once it hands off; warned (`A:802-809`) | partial | `T1::test_a_handoff_built_by_hand_is_governed_when_it_hands_off` | caveat |
| 102 | Nested handoffs, and a cancel inside one | — | yes | `TF::test_nested_handoffs_reach_the_last_agent_whose_call_pauses_and_resumes_there`, `TF::test_a_cancel_stops_a_call_under_way_in_the_agent_handed_off_to` | works |
| 103 | `Handoff.is_enabled`, `Handoff.nest_handoff_history`, `strict_json_schema` | kept | n/a | none | untested |
| 104 | `extensions.handoff_filters` (`remove_all_tools`...), `handoff_prompt` (`RECOMMENDED_PROMPT_PREFIX`) | user code, passed through | n/a | none | untested |
| 105 | SDK refuses `Handoff._agent_ref` missing | refused at wrap (`A:671-684`) | n/a | `T1::test_an_sdk_that_no_longer_says_a_handoffs_agent_is_refused_where_it_is_wrapped` | works |

### 8. Guardrails

| # | Feature | Tested | Status |
|---|---|---|---|
| 106 | `input_guardrail` (parallel) | `TF::test_an_input_guardrail_that_trips_fails_the_run_before_any_tool` | works |
| 107 | `input_guardrail(run_in_parallel=False)` | none | untested |
| 108 | `output_guardrail` | `TF::test_an_output_guardrail_that_trips_fails_the_run_with_no_answer` | works |
| 109 | `tool_input_guardrail` / `tool_output_guardrail` | see #67 | works |

### 9. Sessions and memory

| # | Feature | What it does | Harness | Tested | Status |
|---|---|---|---|---|---|
| 110 | `SQLiteSession` on a file | durable history | durable; memory's window dropped (`history.py:101-117`) | `T1::test_a_durable_session_keeps_the_thread` | works |
| 111 | `SQLiteSession(":memory:")` | RAM history | detected as RAM, memory's window kept, warned (`history.py:72-86`) | `T1::test_who_keeps_the_thread_decides_memorys_window` | works |
| 112 | `AsyncSQLiteSession`, `AdvancedSQLiteSession`, `SQLAlchemySession`, `RedisSession`, `MongoDBSession`, `DaprSession` | durable backends | passed through; objects only, not for queued runs | none | untested |
| 113 | `EncryptedSession` | encrypted history wrapper | passed through | none | untested |
| 114 | `OpenAIConversationsSession` | history stored at OpenAI | passed through; its own OpenAI client bypasses Bifrost | none | untested (off-gateway) |
| 115 | `OpenAIResponsesCompactionSession` | `responses.compact` | calls OpenAI directly (`openai_responses_compaction_session.py:193-197, 465`); its `compaction` items then fail on Chat Completions (`chatcmpl_converter.py:1006`) | none | blocked on Bifrost (fails mid-conversation) |
| 116 | No session, memory on | memory's thread as the session | `MemorySession` injected (`A:846-847`; `bridges/session.py:47-104`); `pop_item`/`clear_session` refused | `test_bridge_session.py::test_the_thread_in_memory_is_the_agents_session`, `test_memory_window.py::test_an_openai_agent_reads_the_thread_through_its_session_as_input_items` | works (behaviour change: a session the SDK would not have) |
| 117 | No session, memory off | earlier turns as input items | `prepare_input` (`A:202-211`) | `test_bridge_session.py::test_without_memory_the_sdk_has_no_session_and_reads_the_run_stores_turns` | works |
| 118 | A session given on `h.wrap` | — | shared by every user and thread | probe `p_sess.py` | **broken** (see #34) |

### 10. RunState, interruptions and approvals

| # | Feature | Harness (file:line) | Tested | Status |
|---|---|---|---|---|
| 119 | `RunState.to_json` / `from_json` as the checkpoint | `A:273, 376` | `test_bridge_session.py::test_the_sdks_saved_run_state_resumes_on_another_worker`, `TF::test_an_sdk_approval_inside_an_agent_tool_resumes_in_another_process` | works |
| 120 | `approve` / `reject(rejection_message=)` | `A:374-384` | `T1::test_a_rejected_sdk_approval_tells_the_model_the_reason`, `T1::test_an_answered_sdk_approval_is_what_the_model_reads_instead` | works |
| 121 | `approve(always_approve=True)` / `reject(always_reject=True)` | never passed; "remember" is not mapped for the SDK's own `needs_approval` tools | none | n/r |
| 122 | Edit | emulated as a reject plus a re-call that is approved (`A:107-110, 296-302, 1170-1180`): one extra model turn | `T1::test_an_edited_sdk_approval_runs_the_edited_call_once`, `T1::test_an_edited_call_the_model_changes_again_asks_again` | caveat |
| 123 | Several interruptions in one reply | one at a time (`A:1146-1167`) | `T1::test_two_sdk_approvals_in_one_turn_are_answered_one_at_a_time`, `T1::test_two_governance_approvals_in_one_reply_are_asked_one_at_a_time` | works |
| 124 | Governance's question held at the SDK gate | `held` (`C:233-254`; `A:463-477`) | `T1::test_a_governance_approval_continues_the_sdks_run_state` and 8 more | works |
| 125 | RunState schema 1.21 (0.24) | checkpoints saved under 0.22.3 and resumed on 0.24, or saved on 0.24 and read after a rollback | none | untested (upgrade risk) |

### 11. Sandbox

| # | Feature | What it does | Harness | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|---|
| 126 | `SandboxAgent` + `SandboxRunConfig(client=)` | session per `Runner.run` | passed through | — | `test_framework_options.py::test_an_openai_sandbox_agent_runs_wrapped_with_its_run_config_as_an_option` | works |
| 127 | Capability tools `exec_command`, `write_stdin`, `view_image`, `load_skill` | made inside `Runner.run`, never in `agent.tools` | **not governed, not recorded, not on the stream, not in the tool set** | **no** | `test_background_by_adapter.py::test_b_today_the_kept_session_is_every_users_and_its_shell_is_not_governed` (pins the gap) | **broken (governance gap)** |
| 128 | `Shell` capability | shell commands, PTY | passed through | no | `test_background_by_adapter.py::test_b_an_openai_sandbox_agents_shell_runs_a_command_across_the_turns_of_a_chat` | caveat (ungoverned) |
| 129 | `Filesystem` capability | `view_image` and `apply_patch` | `apply_patch` is a `CustomTool` (`capabilities/tools/apply_patch_tool.py:175`), refused on Chat Completions | no | none | blocked on Bifrost (documented at `docs/skills.md:96-99`) |
| 130 | `Capabilities.default()` (Filesystem, Shell, Compaction) | the default `SandboxAgent` | fails on Bifrost (because of Filesystem) | — | none | blocked on Bifrost |
| 131 | `Compaction` capability | Responses server-side compaction (`context_management`, `capabilities/compaction.py:200-221`) | Chat Completions never sends it: a silent no-op | — | none | blocked (Responses-only; silent) |
| 132 | `Skills` capability (lazy) | progressive skills | `TrellisSkills` / `TrellisSkillSource` pin the run's skills (`S:31-91`) | the skill set is pinned; `load_skill` is ungoverned | `test_skills.py::test_an_openai_sandbox_agent_loads_a_skill_into_its_sandbox`, `test_skills.py::test_an_openai_sandbox_agent_reads_why_a_skill_cannot_be_loaded_and_goes_on` | works |
| 133 | `Skills` from a local directory / `Skill` objects in the manifest | SDK skill sources | passed through | no | none | untested |
| 134 | `Memory` capability, reading | `memory_summary.md` in the prompt, `MEMORY.md` search | needs `Shell`; the default `live_update=True` needs `Filesystem` (`capabilities/memory.py:43-48`), which is blocked on Bifrost | no | none | caveat (needs `live_update=False`; untested) |
| 135 | `Memory` generation, phase 1 (extraction) and phase 2 (consolidation) | after a session closes, `Runner.run` with `phase_one_model="gpt-5.4-mini"` and `phase_two_model="gpt-5.5"` | uses a **fresh `RunConfig(sandbox=...)`** (`sandbox/memory/manager.py:245-246`), so the default `MultiProvider` goes straight to OpenAI's Responses API, outside Bifrost. The harness's hooks, budget and records do not apply; the phase-2 agent writes files ungoverned; failures are only logged | **no** | none | **broken (off-gateway, ungoverned)** |
| 136 | `MemoryLayoutConfig`, `MemoryGenerateConfig`, `MemoryReadConfig` | memory options | passed through | no | none | untested |
| 137 | `Manifest`: `File`, `Dir`, `LocalFile`, `LocalDir`, `GitRepo`, env | workspace contents | passed through | no | none | untested |
| 138 | Mounts: S3, S3 Files, GCS, R2, Azure Blob, Box (+ rclone) | remote storage | passed through | no | none | untested |
| 139 | `extra_path_grants`, `run_as` `User`, concurrency/archive limits, `cwd` | host access and limits | passed through | no | none | untested |
| 140 | Snapshots / `session_state` / `session=` | sandbox persistence | passed through; a `session=` on `h.wrap` is shared by every user | no | `test_background_by_adapter.py::test_b_today_with_a_sandbox_session_per_run_the_command_ends_with_its_turn` and the test in #127 | caveat (shared; untested across a harness pause) |
| 141 | Clients: `UnixLocalSandboxClient` / Docker / E2B, Blaxel, Cloudflare, Daytona, Modal, Runloop, Vercel | sandbox providers | passed through | no | UnixLocal only | works (UnixLocal); untested (the others) |

### 12. Streaming, results, hooks, tracing, errors

| # | Feature | Harness (file:line) | Tested | Status |
|---|---|---|---|---|
| 142 | Text deltas (`response.output_text.delta`) | the stream's text (`A:1130-1134`) | `T1::test_streaming_emits_text_deltas` | works |
| 143 | `RunItemStreamEvent`, `AgentUpdatedStreamEvent`, reasoning and other raw events | not surfaced (only text plus the harness's `TOOL_CALL_*`) | none | n/r |
| 144 | `RunResultStreaming.cancel(mode="after_turn")` | not exposed; the harness cancel stops the task | matrix F05 | n/r (harness cancel works) |
| 145 | `RunResult.final_output` / `new_items` | answer and transcript (`A:265-277, 1183-1212`) | many | works |
| 146 | `RunResult` `raw_responses`, `last_response_id`, `to_input_list()`, `last_agent`, guardrail results, `context_wrapper.usage` | `Result` exposes only answer, status, interrupt and error (`harness/result.py:19-23`) | none | n/r |
| 147 | `RunHooks` (all 7 methods) | forwarded (`K:44-104`) | `TF::test_the_runs_own_run_hooks_see_the_run_beside_the_harnesss`, `TF::test_run_hooks_do_not_stand_in_for_the_harnesss_budget_or_its_last_answer` | works |
| 148 | `AgentHooks` | passed through | `test_hooks.py::test_openai_agents_model_calls_are_its_own_agent_hooks_s` | works |
| 149 | SDK tracing (default export to OpenAI, `trace_include_sensitive_data=True`) | left on, said once at INFO (`A:143-158`) | `T1::test_the_sdks_own_tracing_is_left_as_set_and_said_once` | caveat (tool inputs and outputs leave for OpenAI unless you turn it off) |
| 150 | `add_trace_processor`, `set_trace_processors`, spans, `trace()` | passed through | none | untested |
| 151 | The harness's OTel run and tool spans | its own | matrix F57 | works |
| 152 | Default tool error text (0.23 generic) | the SDK's; the error itself is recorded | `TF::test_a_failure_the_sdk_tells_the_model_is_recorded_as_a_failed_call` | works |
| 153 | `ModelRefusalError` / `ModelBehaviorError` / `UserError` | the run ends ERROR | `T1::test_a_last_call_that_fails_leaves_the_sdks_error` | works |
| 154 | Usage | counted per reply (`K:96-104`) and per agent tool (`C:335-345`) | `T1::test_a_token_budget_stops_the_run_without_any_hooks`, `TN` budget | works |

### 13. MCP

| # | Feature | Harness (file:line) | Governed / recorded | Tested | Status |
|---|---|---|---|---|---|
| 155 | `MCPServerStdio` | `GovernedServer` | yes | `TM::test_a_stdio_servers_tools_are_filtered_by_the_sdk_and_governed_by_the_harness` | works |
| 156 | `MCPServerStreamableHttp` (headers, auth) | same | yes | `TM::test_a_server_over_http_with_its_headers_is_governed_the_same` | works |
| 157 | `MCPServerSse` | same | yes | none | untested |
| 158 | `tool_filter` (static or dynamic) | the SDK's | n/a | `TM::test_a_servers_dynamic_filter_and_tool_cache_are_the_sdks` | works |
| 159 | `require_approval` | the SDK's pause; asked once | yes | `TM::test_a_servers_own_approval_policy_pauses_the_run_as_the_sdk_does`, `TM::test_a_write_the_servers_policy_asked_about_is_not_asked_about_twice` | works |
| 160 | Resources, resource templates, prompts, `cached_tools` | delegated (`A:565-584`) | n/a | `TM::test_the_governed_copy_of_a_server_reads_its_resources_and_cached_tools` | works |
| 161 | `use_structured_content`, `is_error` results | the server's result is passed through | yes | `TM::test_a_servers_error_is_what_the_model_reads_and_structured_content_is_used` | works |
| 162 | `mcp_annotations` (0.24) / annotations as risk | read for side effects (`A:560-563, 610`) | yes | `test_native_tools.py::test_an_openai_agents_mcp_server_calls_are_governed` | works |
| 163 | `max_retry_attempts` + backoff | the inner server retries, then the harness retries reads again (compounding) | yes | none | caveat (untested) |
| 164 | `tool_meta_resolver`, `custom_data_extractor`, MCP `tool_input_guardrails` / `tool_output_guardrails`, `max_list_pages`, `message_handler` | delegated via `__getattr__` (`A:547-548`) | n/a | none | untested |
| 165 | `MCPServerManager` | user-side lifecycle | n/a | none | untested |
| 166 | A failed call reworded by `after_tool` hooks | `WORDED` (`A:502-516, 617-623`) | yes | `TM::test_a_failed_call_a_hook_rewords_is_told_in_its_words_as_the_sdk_tells_it`, `TM::test_a_failed_call_a_hook_makes_a_result_is_read_as_one` | works |

### 14. Models, providers and extensions

| # | Feature | Harness | Tested | Status |
|---|---|---|---|---|
| 167 | `OpenAIResponsesModel` (the SDK default) | replaced by Chat Completions on Bifrost (`A:1104`); the default elsewhere | none | blocked on Bifrost (design) |
| 168 | `OpenAIChatCompletionsModel` | used on Bifrost | `test_gateways.py::test_an_openai_agent_by_name_goes_through_bifrost` | works |
| 169 | `OpenAIResponsesWSModel`, `responses_websocket_session`, `ResponsesWebSocketSession`, `set_default_openai_responses_transport` | overridden on Bifrost | none | blocked on Bifrost |
| 170 | `set_default_openai_key`, `set_default_openai_client`, `set_default_openai_api` | ignored on Bifrost (its own client) | none | caveat (override) |
| 171 | `OpenAIAgentRegistrationConfig`, `set_default_openai_harness` | passed through | none | untested |
| 172 | `MultiProvider` prefixes `litellm/`, `any-llm/`, a custom `provider_map` | **rewritten**: `litellm/anthropic/claude-3` goes to Bifrost as provider `litellm` (probe: `Bifrost.routed`) | none | broken (on Bifrost) |
| 173 | `LitellmModel`, `AnyLLMModel` objects | bypass Bifrost; warned | none | caveat (untested) |
| 174 | `reasoning_content_replay` (chat providers) | the SDK's | none | untested |
| 175 | `extensions.tool_output_trimmer.ToolOutputTrimmer` | runs inside your `call_model_input_filter`, before the harness's | none | untested |
| 176 | `extensions.visualization.draw_graph` | offline, on your agent | none | not applicable |
| 177 | `extensions.experimental.codex` (`codex_tool`) | one governed `FunctionTool` call; Codex's own commands and model calls are outside the harness and Bifrost | none | caveat (untested) |
| 178 | `extensions.experimental.hosted_multi_agent` | a `Model` object over a beta Responses websocket: bypasses Bifrost | none | blocked on Bifrost; n/r through the gateway |

### 15. Realtime, voice, REPL, testing

| # | Feature | Harness | Status |
|---|---|---|---|
| 179 | `RealtimeAgent`, `RealtimeRunner`, `RealtimeSession` (WebSocket, SIP) | refused by `detect` (not an `Agent`) | n/r |
| 180 | `VoicePipeline`, `SingleAgentVoiceWorkflow`, STT/TTS models | runs `Runner` itself; no harness path | n/r |
| 181 | `run_demo_loop` / REPL | developer loop | n/r (design) |
| 182 | `agents.testing` (fake model, sandbox testing) | not used; the harness has its own `ScriptedModel` | not applicable |

## Gap list

### P0: data exposure

1. **A session object given on `h.wrap` is shared by every user and thread.** This covers
   `framework_options={"session": SQLiteSession(...)}` and `SandboxRunConfig(session=...)`.
   - Probe `p_sess.py`: bob's model input contains alice's turn "my secret is 42".
   - Code: the harness passes the agent's framework options to every run (`A:845`, `agent.py:367`).
     `D:221` invites agent-level sessions, with no warning.
   - The sandbox case is pinned as "today" in
     `test_background_by_adapter.py::test_b_today_the_kept_session_is_every_users_and_its_shell_is_not_governed`.
   - Fix direction (no SDK behaviour change): refuse a session object on `h.wrap`, or require a
     per-conversation factory.

### P1: broken or ungoverned

2. **A `run_config` given as a dict crashes every run** (`TypeError` from `dataclasses.replace`,
   `A:903`; also `A:915, 1048`). `D:91` documents dicts for queued runs (`start`/`schedule`), and
   the SDK accepts them. Probe `p_dict.py`. Fix: `_coerce_run_config` first.
3. **Multimodal and structured function-tool outputs are flattened to text.** `governed` always
   returns `results.readable(...)` (`C:173`).
   - `ToolOutputImage`, `ToolOutputFileContent` and lists reach the model as a Python repr string,
     not as an image or file part.
   - Typed results reach `tool_use_behavior` functions, `stop_on_first_tool`'s `final_output`, tool
     output guardrails, `on_tool_end` and `custom_data_extractor` as `str`.
   - Probes `p_outputs.py`, `p_custom.py`.
4. **Server-side threads are lost on Bifrost.** With `conversation_id`, `previous_response_id` or
   `auto_previous_response_id`:
   - Chat Completions ignores them with a warning (`openai_chatcompletions.py:543-573`);
   - the harness treats them as a durable thread and drops memory's window (`A:1112-1117`);
   - so the conversation has no history at all, and with `auto_previous_response_id` the SDK also
     sends only new items after turn 1.
   - The only test checks the classification
     (`T1::test_who_keeps_the_thread_decides_memorys_window`).
5. **`SandboxAgent` capability tools are ungoverned.** `exec_command`, `write_stdin`, `view_image`,
   `load_skill` and `apply_patch` are made inside `Runner.run`. They get no catalog, no approval,
   no journal, no record, no stream event and no tool-set entry. The harness's own test pins this
   (#127). Shell commands therefore run with no person asked.
6. **Sandbox `Memory` generation runs off the gateway.** Phases 1 and 2 call `Runner.run` with a
   fresh `RunConfig` (`sandbox/memory/manager.py:245`), so `gpt-5.4-mini` and `gpt-5.5` go straight
   to OpenAI's Responses API, outside Bifrost, the budget, the records and governance (the phase-2
   agent writes files). Failures are only logged. Workaround: pass `phase_one_model` and
   `phase_two_model` as Model objects from `bifrost_provider(...)`. Better: the harness builds a
   `MemoryGenerateConfig` on Bifrost, or refuses `Memory(generate=...)` with a clear message.

### P2: limits and overrides that need a decision

7. **Responses-only features fail mid-run or silently on the default route.** On Bifrost:
   - the hosted tools (#78-84) and the shell, patch, computer and custom tools (#85-90) fail
     with an SDK `UserError` mid-run; `L` (the harness's governance for those tools) is unreachable;
   - the default `SandboxAgent` fails (#130);
   - `prompt=`, sandbox `Compaction`, `truncation`, `store`, `context_management` and reasoning
     mode are silently ignored.

   Fix: refuse at `h.wrap` with a per-feature message, and offer an opt-in Bifrost Responses route
   for OpenAI models. Correct `D:122-123` and `D:154-157`.
8. **An agent tool with its own `run_config` escapes the harness's run config.** Its model calls
   bypass Bifrost and attribution, and get no day, steering or `tool_not_found` override (probe
   `p_nested.py`). No warning is given, unlike `_unrouted`.
9. **`Agent` subclasses from user modules are refused** (`adapters/__init__.py:30`). Probe `p_sub.py`.
10. **Model-name routing is overridden.** `litellm/` and `any-llm/` prefixes and a configured
    `MultiProvider(provider_map=...)` are replaced by Bifrost (`A:1049-1056`). A `RunConfig.model`
    object bypasses Bifrost with no warning.
11. **No context management on the default route.** Matrix F65 says "the framework keeps its own
    context". On Bifrost both SDK mechanisms are unusable: sandbox `Compaction` is a no-op and
    `OpenAIResponsesCompactionSession` fails. A plain `Agent` has neither, so long runs can overflow
    the window.
12. **SDK tracing exports to OpenAI by default**, including tool inputs and outputs
    (`trace_include_sensitive_data=True`). The harness says so once at INFO only, while its
    redaction claim (F61) covers only its own traces.
13. **The native result and stream are not reachable.** `RunResult` details (`raw_responses`,
    `last_agent`, `to_input_list`, guardrail results) and non-text stream events are not exposed (#143, #146).

### P3: smaller gaps and test debt

14. `always_approve` / `always_reject` are never used. Approving one of the SDK's own
    `needs_approval` tools for the conversation does not stick (#121).
15. Documentation contradiction: `D:93` says "The harness sets no `RunConfig` field of its own …
    yours is passed as it is". `D:82` and the code (`A:896-1056`) set `tool_not_found_behavior`,
    wrap `call_model_input_filter`, replace `model_provider` and add `error_handlers`.
16. Child agent-tool runs inherit the harness's `call_model_input_filter`. Steered messages can be
    consumed by a child's model call and later re-inserted at child offsets in the parent
    (`A:1019-1031`; code-read, untested). Child model calls also lack the stall watch.
17. MCP `max_retry_attempts` compounds with the harness's read retries (#163).
18. RunState schema 1.21 is untested across the 0.22.3 to 0.24 upgrade: in-flight paused
    checkpoints, and rollback.
19. Pass-throughs with no test:
    - sessions: AsyncSQLite, AdvancedSQLite, SQLAlchemy, Redis, Mongo, Dapr, Encrypted, OpenAI
      Conversations;
    - MCP: `MCPServerSse`, `MCPServerManager`, `include_server_in_tool_names`, MCP guardrails;
    - run settings: `ToolExecutionConfig`, `tool_name_collision_policy`,
      `output_guardrail_blocked_message`, the `model_refusal` / `invalid_final_output` handlers,
      `nest_handoff_history`, `handoff_history_mapper`, run-level guardrails,
      `ModelSettings.retry`, `timeout_error_function`, `strict_mode`;
    - sandbox: providers other than UnixLocal, mounts, manifests, path grants, snapshots;
    - extensions: handoff filters and prompt, `ToolOutputTrimmer`, codex.
20. Realtime and voice cannot be wrapped (#179-180). A Way 2 recipe would cover them.

Not proposed, by the user's decision: background agents (`D:215-220`, pinned in
`test_background_by_adapter.py`).

## Where the harness changes SDK behaviour

The user's rule is that library behaviour must not change. Each item below departs from what
`Runner.run` does without the harness:

1. `RunConfig.tool_not_found_behavior` is forced to `"return_error_to_model"`, even over an
   explicit `"raise_error"` (`A:896-905`).
2. `max_turns` gets a default error handler: one extra tool-less model call produces an answer
   instead of `MaxTurnsExceeded` (`A:855-893`).
3. `call_model_input_filter` is wrapped (`A:908-1032`):
   - the memory context moves from a system message into the user turn;
   - the day is appended to every instruction;
   - steered and computer-refusal messages are inserted.
4. `model_provider` is replaced on Bifrost (`A:1035-1107`):
   - Chat Completions instead of Responses (`use_responses=False`);
   - client `max_retries=0`;
   - extra headers;
   - model names rewritten to `provider/model`;
   - a custom `MultiProvider` is replaced.
5. Tools are re-sorted by name on every agent copy (`A:446-449`).
6. An agent's own function-tool result becomes the harness's text (`C:173`): JSON instead of
   `str()`, multimodal outputs stringified, tables annotated, results over 80k replaced by a stub
   (`results.py:133-217`).
7. Identical calls in one reply run once (`C:139-141`).
8. Read tools are retried on transient errors: 3 calls where the SDK makes 1 (`tools/base.py:298-334`).
9. Your tools may be hidden per turn by tool hints, and all are withdrawn after a stall (`C:289-302, 420-424`).
10. Every governed `FunctionTool` gets a `needs_approval` policy (`C:233-254`). Tools with no
    declared risk ask a person, which the SDK never does.
11. Harness tools are added to every agent copy (memory tools, the gateway's MCP tools,
    `read_file`, `query_result`) (`A:355-366, 443-448`).
12. A `MemorySession` is injected when none is given, and memory or history items are prepended to
    the input (`A:202-211, 846-847`).
13. Edit decisions become a reject plus an approved re-call: one extra model turn (`A:296-302, 1170-1180`).
14. A refused computer action is skipped and explained in an extra user message (`L:340-353`, `A:1025`).
15. The harness's run time limit and cancellation can cut tool calls the SDK would let finish (`C:196-207`).

Most of these are the harness's documented value-adds. Items 1, 4 (overriding *your* provider
configuration), 5, 6 and 8 change SDK semantics in ways the documentation does not flag, or
contradicts (`D:93`).

## Counts by status

182 rows, counted by the first status in each row:

| Status | Rows |
|---|---|
| works | 74 |
| untested | 33 |
| caveat | 30 |
| blocked | 25 |
| broken | 11 |
| not reachable | 7 |
| not applicable | 2 |

Rows marked "blocked on Bifrost; works off it" count as blocked: the harness tests them only
with scripted Responses items, off the gateway.
