# Claude Agent SDK (Python) 0.2.160 + bundled Claude Code CLI 2.1.283: complete feature inventory vs Trellis

Read-only audit, 2026-11-14. Harness: `/var/tmp/cons/final-agent-harness` (branch claude/cool-maxwell-w5avo8-final @ d2bf754); paths below are under `src/trellis/harness/` and `tests/integration/` unless written otherwise. Sources: installed SDK source (`types.py`, `client.py`, `query.py`, `_internal/`), strings and `--help` of `_bundled/claude` (2.1.283), code.claude.com docs (all reachable; fetched as `.md`), SDK CHANGELOG (raw.githubusercontent, reachable; tops out at 0.2.178) and PyPI (latest 0.2.182, sdist diffed). GitHub API for the SDK repo was refused (403: repository not enabled for this session). No model was called.

**Testing legend.** `(fake)`: the real SDK drives `tests/support/fake_claude_cli.py`. `(real, gated)`: `tests/live/*`, needs a gateway and `claude` on PATH (note: the skip checks PATH while the SDK uses its bundled CLI). Apart from two gated live tests, nothing runs the real CLI.

## Status counts

| status | count |
|---|---|
| works | 89 |
| works-with-caveat | 43 |
| broken | 5 |
| blocked-by-design | 10 |
| untested | 87 |
| not reachable | 94 |
| **total rows** | **328** |


## A. ClaudeAgentOptions (all 49 fields of 0.2.160)

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `tools` | Base set of built-in tools (list, [] or preset) | pass; read by native() adapters/claude.py:1507; last answer at turn limit forces [] adapters/claude.py:622-623 | every call governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli (fake); live/test_live_claude.py::test_claude_runs_through_the_gateway_and_resumes_its_session uses tools=[] (real, gated) | **works** |  |
| `allowed_tools` | Rules that auto-allow tools without asking | pass; mcp__ names also join native() adapters/claude.py:1511-1515 | governance still runs first in PreToolUse | test_claude_sdk_features.py::test_the_options_reach_the_cli_as_the_sdk_sends_them; test_claude.py::test_a_command_claudes_own_rules_allow_runs_without_asking_a_person (fake) | **works** |  |
| `system_prompt` | str, preset(+append, exclude_dynamic_sections, snapshot), custom, file | TRANSFORMED: DELIVERS_HERE text appended to str/custom/preset.append; None becomes a custom prompt of that text; file left alone adapters/claude.py:1188, 1212-1229 | - | test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli (fake) | **works-with-caveat** | Changes library behaviour; docs/frameworks/claude-agent-sdk.md 'What is automatic' says the prompt is 'left as you wrote it' |
| `mcp_servers` | MCP servers (dict, JSON text or file path) | TRANSFORMED: text/path read into a dict adapters/claude.py:1586-1599; harness 'trellis' in-process server added with alwaysLoad adapters/claude.py:1199-1201, tools/convert/claude.py:60-63; a team server named 'trellis' refused adapters/claude.py:426-431 | team servers' calls governed (risk: annotations for in-process servers, else UNDECLARED -> person) and recorded | test_claude.py::test_mcp_servers_the_team_configured_as_a_file_or_json_stay (fake); live/test_live_gateway_features.py::test_the_claude_agent_sdks_mcp_servers_on_the_gateway (real, gated) | **works** |  |
| `strict_mcp_config` | Ignore MCP config other than mcp_servers | pass | - | test_claude_sdk_features.py::test_the_options_reach_the_cli_as_the_sdk_sends_them (fake, argv) | **works** |  |
| `permission_mode` | default/acceptEdits/plan/bypassPermissions/dontAsk/auto | pass; governance runs in every mode | governed | test_claude.py::test_built_in_tools_are_governed_in_every_permission_mode (fake) | **works** |  |
| `continue_conversation` | Continue latest session in cwd | pass on the target; refused in framework_options adapters/claude.py:240-245, 423-425; turns the journal store off bridges/claude.py:92-100 | - | test_claude_sdk_features.py::test_the_sessions_the_options_choose_are_claude_codes_to_resume_fork_or_continue (fake) | **works-with-caveat** | Session stays on the local disk: no resume on another worker |
| `resume` | Session id to resume | pass for the first query; OVERRIDDEN by the journal session on later attempts adapters/claude.py:1203-1204; refused in framework_options | - | test_claude_sdk_features.py::test_the_sessions_the_options_choose_are_claude_codes_to_resume_fork_or_continue (fake); live/test_live_claude.py (real, gated) | **works-with-caveat** |  |
| `session_id` | Fixed UUID for the new session | pass; set to None when the harness resumes adapters/claude.py:1205-1208 | - | test_claude_sdk_features.py::test_a_session_id_of_the_options_names_the_session_and_a_pause_resumes_it (fake) | **works** |  |
| `max_turns` | Turn cap; error_max_turns result | pass; TRANSFORMED at the cap: the error result is swallowed and one more query runs (resume, tools=[], mcp_servers={}, max_turns=1) adapters/claude.py:503-549, 622-623 | the extra query is charged | test_claude.py::test_at_its_turn_limit_a_query_ends_with_one_last_answer; ::test_a_last_query_at_the_limit_that_gives_no_answer_leaves_the_error (fake) | **works-with-caveat** | Behaviour change: an extra paid model call replaces the library's error result |
| `max_budget_usd` | Dollar cap per query | pass; forces a process per query (no live session) adapters/claude.py:753-763 | budget charged from total_cost_usd | test_claude.py::test_a_query_stopped_by_its_own_budget_keeps_its_error; test_claude_background_session.py::test_the_options_own_budget_stays_a_querys_each_turn_in_a_process_of_its_own (fake) | **works** |  |
| `disallowed_tools` | Remove tools from context | TRANSFORMED: CronCreate, CronDelete, CronList, ScheduleWakeup and hint-narrowed tools appended adapters/claude.py:196, 1258-1267 | - | test_claude.py::test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered; test_claude_background_session.py::test_schedules_stay_disallowed_in_a_live_session (fake) | **works** |  |
| `model` | Model id/alias | pass; also the price key of budget.account adapters/claude.py:1063-1071 | - | test_claude_sdk_features.py options test (fake, argv); live/test_live_claude.py (real, gated) | **works-with-caveat** | Tokens priced by options.model; ResultMessage.model_usage (per model, fallback, subagents on other models) ignored |
| `fallback_model` | Fallback model(s) | pass | - | test_claude_sdk_features.py::test_the_options_reach_the_cli_as_the_sdk_sends_them (fake, argv) | **works** |  |
| `betas` | Beta headers (context-1m) | pass | - | test_claude_sdk_features.py options test (fake, argv) | **works** |  |
| `permission_prompt_tool_name` | MCP tool that answers permission prompts | BLOCKED: ConfigurationError adapters/claude.py:240-245, 1174-1179 (the harness owns can_use_tool, which makes the SDK send 'stdio') | - | test_claude.py::test_a_permission_prompt_tool_of_the_teams_own_is_refused (fake) | **blocked-by-design** | Reason: the harness's can_use_tool is the permission path; SDK forbids both |
| `cwd` | Working directory | pass; also the shell-analysis root adapters/claude.py:1364 and workdir adapters/claude.py:433-436 | - | test_claude_sdk_features.py options test (fake) | **works** |  |
| `cli_path` | CLI binary (default: bundled 2.1.283) | pass | - | every integration test points it at tests/support/fake_claude_cli.py | **works** |  |
| `settings` | Extra settings file/JSON (flag layer) | pass (the SDK merges sandbox into it) | - | test_claude_sdk_features.py::test_a_sandbox_of_the_options_is_merged_into_the_settings_the_cli_gets (fake) | **works** |  |
| `add_dirs` | Extra directories the CLI may access | pass; also read roots of the shell analysis adapters/claude.py:1362 | - | test_claude_sdk_features.py options test; test_claude.py::test_a_read_outside_the_working_folder_asks_a_person (fake) | **works** |  |
| `env` | Subprocess environment | TRANSFORMED: MAX_MCP_OUTPUT_TOKENS=60000 unless set adapters/claude.py:202-203, 1182-1183; gateway ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / ANTHROPIC_CUSTOM_HEADERS adapters/claude.py:1232-1255 | - | test_claude_sdk_features.py options test; test_claude.py::test_a_large_result_is_the_harness_s_to_keep_not_the_cli_s (fake) | **works-with-caveat** | Behaviour change |
| `extra_args` | Arbitrary CLI flags | pass as-is | - | test_claude_sdk_features.py options test (fake, argv) | **works-with-caveat** | Untested interactions: --no-session-persistence defeats the journal store, --permission-prompts none silences approvals, --worktree moves the cwd the shell analysis assumes |
| `max_buffer_size` | Max stdout line buffer | OVERRIDDEN when unset: 1 MiB -> 100 MiB adapters/claude.py:210, 1191 | - | test_claude.py::test_a_tool_result_over_a_mebibyte_reaches_claude_as_a_preview (fake) | **works** | Behaviour change (benign) |
| `debug_stderr` | Deprecated; not read by the SDK | pass (no effect) | - | - | **not reachable** |  |
| `stderr` | stderr line callback | pass | - | test_claude_sdk_features.py options test (fake) | **works** |  |
| `can_use_tool` | Permission callback for 'ask' decisions | WRAPPED: harness callback always installed adapters/claude.py:1190, 1367-1421; team's own asked after governance; live relay bridges/claude_live.py:760-775 | governed (person via bridge.asked) and recorded | test_claude.py::test_the_teams_own_callback_is_asked_only_what_claude_code_asks; test_claude_sdk_features.py::test_the_teams_permission_answer_reaches_claude_code_as_given (fake); live/test_live_claude.py (real, gated) | **works** |  |
| `hooks` | Hook callbacks per event | MERGED: harness matchers appended for PreToolUse, PostToolUse, PostToolUseFailure, UserPromptSubmit adapters/claude.py:1270-1279, bridges/claude.py:62-76; live session relays every callback, harness ones with a 24 h timeout bridges/claude_live.py:679-693, and with no turn under way returns {} for every non-PreToolUse event bridges/claude_live.py:719-720 | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fake) | **works-with-caveat** | Team hooks between live turns are not run; CLI dispatches matchers concurrently while the doc says the team's run 'before the harness's own' |
| `user` | OS user for the subprocess | pass | - | test_claude_sdk_features.py options test (fake) | **works** |  |
| `include_partial_messages` | Emit StreamEvent deltas | pass; deltas not used: stream() yields whole TextBlocks adapters/claude.py:360-372 | - | test_claude_sdk_features.py::test_partial_messages_and_other_kinds_of_message_leave_the_answer_as_it_is (fake) | **works-with-caveat** | No token-level streaming |
| `include_hook_events` | Emit HookEventMessage | pass; messages ignored | - | test_claude_sdk_features.py options test (fake, argv) | **works** |  |
| `forward_subagent_text` | Forward subagent text/thinking | pass; parent stream/transcript skip parent_tool_use_id messages adapters/claude.py:368, 923 | - | test_claude_sdk_features.py options test (fake) | **works** |  |
| `verbatim_prompts` | No @path expansion / slash dispatch | pass; harness never turns it on for end-user text | - | test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli (fake; the stand-in ignores the field) | **works-with-caveat** | See gap P0-1 |
| `fork_session` | Fork on resume | pass; forced True when another worker holds the session adapters/claude.py:616, 621 | - | test_claude_sdk_features.py::test_a_forked_session_paused_inside_goes_on_in_its_fork; test_claude_background_session.py::test_a_turn_resumed_on_another_worker_leaves_the_first_its_background_subagent (fake) | **works** |  |
| `resume_session_at` | Truncate resumed transcript at a UUID | pass, and kept on every later attempt while resume is replaced adapters/claude.py:1203-1208 | - | - | **untested** | Suspected wrong: a resumed attempt truncates the run's own session at the team's UUID (gap P2) |
| `resume_drops_turn` | Validate the truncating resume | pass, same as above | - | - | **untested** |  |
| `agents` | Programmatic subagents (AgentDefinition: tools, model, skills, memory, background, effort) | pass; each Agent call is a child run adapters/claude.py:843-903; its prompt rewritten with memory briefing via updatedInput adapters/claude.py:830-840, 1338-1341 | subagent calls governed and recorded under the child run | test_claude_sdk_features.py options test; test_claude_background_session.py background tests (fake) | **works-with-caveat** | Subagent prompt changed; AgentDefinition.memory scope files are not the memory service's |
| `setting_sources` | Which filesystem settings load | pass; harness never sets it, so None loads user+project+local (host ~/.claude settings, CLAUDE.md, rules, auto memory) | auto-memory writes go through Write/Edit (governed as writes) | test_claude_sdk_features.py options test (fake, argv) | **works-with-caveat** | Multi-tenant isolation gap P1-2 |
| `skills` | Enable skills (list or 'all') | pass; ClaudeSkills sets skills='all' (skills.py:452-457) | Skill tool treated as read (auto) | test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli (fake, argv) | **works** |  |
| `sandbox` | Bash sandbox settings | pass (SDK merges into --settings) | sandboxed Bash still passes PreToolUse governance | test_claude_sdk_features.py::test_a_sandbox_of_the_options_is_merged_into_the_settings_the_cli_gets (fake) | **works** | Real sandbox never exercised; 0.2.160's ignoreViolations type is wrong (fixed 0.2.166) |
| `plugins` | Local plugins (--plugin-dir) | pass; ClaudeSkills written to a temp plugin per run and removed after adapters/claude.py:603-604, 661 (skills.py:700-729); forces a process per query | - | test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli (fake) | **works** |  |
| `max_thinking_tokens` | Deprecated thinking budget | pass | - | test_claude_sdk_features.py (fake, argv) | **works** |  |
| `thinking` | adaptive/enabled/disabled (+display) | pass; ThinkingBlocks dropped from transcript adapters/claude.py:906-940 | - | test_claude_sdk_features.py (fake, argv) | **works** |  |
| `effort` | low..max | pass | - | test_claude_sdk_features.py (fake, argv) | **works** |  |
| `output_format` | JSON-schema structured output | pass; structured_output is the answer adapters/claude.py:385-389 | StructuredOutput tool = read | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `enable_file_checkpointing` | Back up files for rewind | pass; forces the local session store bridges/claude.py:92-100; rewind_files unreachable | - | test_claude.py::test_a_session_the_cli_no_longer_holds_is_run_again_from_its_prompt (fake) | **works-with-caveat** | Checkpoints are taken but can never be rewound |
| `session_store` | Mirror transcripts to an external store | team store kept (journal off); else REPLACED by JournalSessions bridges/claude.py:79-100, adapters/claude.py:1194-1198; live Mirror bridges/claude_live.py:661-663 | - | test_claude_background_session.py::test_a_session_the_options_keep_themselves_is_served_live_too (fake) | **works-with-caveat** | JournalSessions implements only append/load: no list_subkeys, so subagent transcripts are not restored on resume |
| `session_store_flush` | batched/eager | OVERRIDDEN to eager with the journal store adapters/claude.py:227, 1198 | - | indirect (fake) | **works** |  |
| `load_timeout_ms` | Store load timeout on resume | pass | - | - | **untested** |  |
| `task_budget` | API-side task token budget | pass | - | test_claude_sdk_features.py (fake, argv) | **works** |  |

## B. SDK entry points, client methods, helpers

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `query()` | One-shot query | harness calls the private InternalClient().process_query instead adapters/claude.py:819-827 | - | all query-path tests (fake) | **works-with-caveat** | Private API; identical internals in 0.2.160 |
| `ClaudeSDKClient connect/query/receive_messages` | Streaming-input session | used for a conversation's live session bridges/claude_live.py:650-676, 806-816, 1061-1066 | - | test_claude_background_session.py (fake) | **works** |  |
| `ClaudeSDKClient.interrupt` | Interrupt current turn | used for a left turn with nothing in the background bridges/claude_live.py:1068-1071 | - | test_claude_background_session.py::test_cancelling_a_turn_with_nothing_in_the_background_interrupts_it (fake) | **works** |  |
| `ClaudeSDKClient.stop_task` | Stop one background task | used on background-run cancel/timeout/budget bridges/claude_live.py:917-920 | - | test_claude_background_session.py::test_cancelling_a_background_subagents_run_stops_that_subagent_only; ::test_a_background_subagents_run_past_its_time_is_stopped_with_stop_task (fake) | **works** |  |
| `ClaudeSDKClient.disconnect` | Close the session | used when a live session closes bridges/claude_live.py:1133-1170 | - | test_claude_background_session.py::test_closing_the_harness_closes_every_session_and_its_process (fake) | **works** |  |
| `ClaudeSDKClient.receive_response` | Messages until a result | not used (reads receive_messages) | - | - | **works** | n/a |
| `ClaudeSDKClient.set_permission_mode` | Change it mid-session | not exposed (docs Limits); set it per run via framework_options | - | - | **blocked-by-design** | Reason: harness owns the client; per-run options instead |
| `ClaudeSDKClient.set_model` | Change it mid-session | not exposed (docs Limits); set it per run via framework_options | - | - | **blocked-by-design** | Reason: harness owns the client; per-run options instead |
| `ClaudeSDKClient.rewind_files` | Restore files to a user message | not exposed by the harness | - | - | **not reachable** |  |
| `ClaudeSDKClient.reconnect_mcp_server` | Reconnect an MCP server | not exposed by the harness | - | - | **not reachable** |  |
| `ClaudeSDKClient.toggle_mcp_server` | Enable/disable an MCP server | not exposed by the harness | - | - | **not reachable** |  |
| `ClaudeSDKClient.get_mcp_status` | MCP server status | not exposed by the harness | - | - | **not reachable** |  |
| `ClaudeSDKClient.get_context_usage` | Context window breakdown | not exposed by the harness | - | - | **not reachable** |  |
| `ClaudeSDKClient.get_server_info` | Initialize result (commands, styles) | not exposed by the harness | - | - | **not reachable** |  |
| `tool() / create_sdk_mcp_server()` | In-process MCP tools | used for the trellis server tools/convert/claude.py:55-117; team's in-process servers listed for risk adapters/claude.py:1524-1544 and force a process per query adapters/claude.py:760-762 | governed and recorded | test_claude.py::test_a_query_answers_and_calls_harness_tools (fake); live/test_live_claude.py (real, gated) | **works** |  |
| `Session listing: list_sessions, get_session_info, get_session_messages, list_subagents, get_subagent_messages` | Read local transcripts | not used; usable by the team directly | - | - | **not reachable** |  |
| `Session store helpers: *_from_store, InMemorySessionStore, fold_session_summary, project_key_for_directory, import_session_to_store` | Store-backed listing/import | not used; JournalSessions lacks list_sessions/list_subkeys/delete so the helpers cannot work on it | - | - | **not reachable** |  |
| `Session mutations: rename/tag/delete/fork_session (+ _via_store)` | Edit transcripts | not used | - | - | **not reachable** |  |
| `Errors: CLINotFoundError, CLIConnectionError, ProcessError, ResultError, CLIJSONDecodeError, ClaudeSDKError` | SDK exceptions | propagate as the run's error (code = class name); ResultError swallowed only at the turn limit adapters/claude.py:526-530 | - | test_claude_sdk_features.py::test_a_cli_that_cannot_start_fails_the_run_with_the_sdks_own_error (fake) | **works** |  |
| `Transport (custom)` | Pluggable transport | not usable: the harness builds the process | - | - | **not reachable** |  |
| `CanUseToolShadowedWarning` | Advisory when allowed_tools shadow the callback | now emitted for teams whose allowed_tools allow whole tools, because the harness always installs can_use_tool | - | - | **works-with-caveat** | New warning the team did not get before |

## C. Message types and content blocks

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `UserMessage` | User/tool-result message (uuid, tool_use_result, origin) | tool results to transcript adapters/claude.py:934-939; subagent end adapters/claude.py:872-879; tool_use_result and origin unused | recorded via PostToolUse, not from the message | test_claude.py::test_the_turns_tool_calls_and_results_are_in_its_transcript (fake) | **works** |  |
| `AssistantMessage` | Model reply (content, usage, error, stop_reason) | text to stream/transcript; tool_use to transcript; usage kept by message_id adapters/claude.py:992-996; .error ignored | - | test_claude.py (fake); live/test_live_claude.py (real, gated) | **works-with-caveat** | AssistantMessage.error (rate_limit, billing_error, authentication_failed) not surfaced on its own |
| `SystemMessage init` | Session metadata | ignored | - | - | **works** | n/a |
| `SystemMessage compact_boundary` | Auto/manual compaction happened | ignored; the mirrored transcript holds the compacted session | - | - | **untested** |  |
| `Other SystemMessage subtypes (api_retry, informational incl. hook_callback_failed, status, notification, model_fallback, permission_denied, memory_saved, memory_recall, prompt_suggestion, files_persisted, local_command_output, worker_shutting_down, ...)` | Loop observability | passed through, ignored: no run event or warning | - | - | **works-with-caveat** | hook_callback_failed is how the CLI reports a harness hook that failed open (gap P1-1) |
| `TaskStartedMessage` | Background task started | ties task_id to its Agent call, detects backgrounding adapters/claude.py:892-895 | child run attribution | test_claude_background_session.py (fake) | **works** |  |
| `TaskProgressMessage` | Task progress | ignored | - | - | **works** | n/a |
| `TaskNotificationMessage` | Task ended | ends the child run adapters/claude.py:896-899 | child run outcome recorded | test_claude_background_session.py::test_a_background_subagents_run_ends_as_its_task_ended (fake) | **works** |  |
| `TaskUpdatedMessage` | Task state patch | terminal status ends the child run adapters/claude.py:900-903 | - | test_claude_background_session.py (fake) | **works** |  |
| `MirrorErrorMessage` | Session-store append failed (SDK-made) | ignored | - | - | **untested** | A team store's mirror errors are silent |
| `ResultMessage` | End of a turn (result, cost, usage, errors, origin, structured_output...) | answer/structured_output adapters/claude.py:374-390; is_error -> ClaudeRunError adapters/claude.py:383-384; cost delta adapters/claude.py:998-1023; origin filter adapters/claude.py:766-770; max_tokens adapters/claude.py:524; max_turns adapters/claude.py:519; permission_denials, deferred_tool_use, model_usage, terminal_reason, api_error_status ignored | spend charged to budget | test_claude.py, test_claude_background_session.py (fake); live/test_live_claude.py (real, gated) | **works** |  |
| `StreamEvent` | Raw API stream event | passed; not turned into deltas | - | test_claude_sdk_features.py::test_partial_messages_and_other_kinds_of_message_leave_the_answer_as_it_is (fake) | **works-with-caveat** |  |
| `RateLimitEvent` | Rate-limit status | ignored | - | - | **untested** |  |
| `ConversationResetMessage` | Session replaced (new id) | ignored; journal.session follows later messages' session_id {A}:647-649 | - | - | **untested** |  |
| `HookEventMessage` | Hook lifecycle (include_hook_events) | ignored | - | - | **works** | n/a |
| `TextBlock` | Text content | stream + transcript | - | (fake) | **works** |  |
| `ThinkingBlock` | Thinking content | dropped from transcript and memory | - | - | **works-with-caveat** |  |
| `ToolUseBlock` | Tool call | transcript; Agent start | - | (fake) | **works** |  |
| `ToolResultBlock` | Tool result | transcript; subagent end | - | (fake) | **works** |  |
| `ServerToolUseBlock / ServerToolResultBlock` | API-side tools (advisor, web_search, web_fetch, code_execution, tool_search) | not recorded as tool calls | not governable (run API-side); not recorded | - | **untested** |  |

## D. Hook events (10 typed in 0.2.160; 23 more the CLI 2.1.283 knows)

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `PreToolUse` | Before every tool call | harness governance hook adapters/claude.py:1297-1347 after the team's; pause -> deny + continue:false adapters/claude.py:1328-1330 | governed | test_claude.py::test_built_in_tools_are_governed_and_not_run_again_after_a_pause; test_claude_sdk_features.py::test_a_hook_of_the_teams_own_that_refuses_a_call_is_obeyed (fake) | **works-with-caveat** | Fails open: an exception other than Paused leaves the call to Claude Code's own check (gap P1-1) |
| `PostToolUse` | After a call succeeded | record adapters/claude.py:1462-1499 + steering additionalContext adapters/claude.py:1282-1294 | recorded | test_claude.py::test_calls_claude_runs_itself_are_recorded_with_their_result_and_duration (fake) | **works** |  |
| `PostToolUseFailure` | After a call failed | record as error adapters/claude.py:1481-1486 | recorded | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fake) | **works** |  |
| `UserPromptSubmit` | Before a prompt | memory context as additionalContext bridges/claude.py:62-76 | - | test_claude.py::test_the_context_goes_with_the_prompt_as_a_hook_adds_it (fake) | **works** |  |
| `Stop` | Team hook only | pass (team's own, run inside the run); live: not run between turns | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fired; fake) | **works** |  |
| `SubagentStart` | Team hook only | pass (team's own, run inside the run); live: not run between turns | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fired; fake) | **works** |  |
| `SubagentStop` | Team hook only | pass (team's own, run inside the run); live: not run between turns | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fired; fake) | **works** |  |
| `PermissionRequest` | Team hook only | pass (team's own, run inside the run); live: not run between turns | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fired; fake) | **works** |  |
| `PreCompact` | Team hook only | pass (team's own, run inside the run); live: not run between turns | - | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fired by /compact in ::test_a_slash_command_no_model_replied_in_is_a_turn_done_with_an_empty_answer; fake) | **works** |  |
| `Notification` | Team hook only | pass; live: not run between turns | - | registered in test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run but never fired | **untested** |  |
| `PostToolBatch` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `UserPromptExpansion` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `SessionStart` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `SessionEnd` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `StopFailure` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `PostCompact` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `PreModelSwitch` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `PostModelSwitch` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `PermissionDenied` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `Setup` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `TeammateIdle` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `TaskCreated` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `TaskCompleted` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `Elicitation` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `ElicitationResult` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `ConfigChange` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `WorktreeCreate` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `WorktreeRemove` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `InstructionsLoaded` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `CwdChanged` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `FileChanged` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `DirectoryAdded` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `MessageDisplay` | CLI event; not in the SDK's HookEvent literal | a team can still register it (the SDK passes any key; check_options does not validate keys); live: dropped between turns | - | - | **untested** |  |
| `Hook output: permissionDecision 'defer'` | Defer a call to the host (ResultMessage.deferred_tool_use) | never used by the harness; a team hook's defer leaves deferred_tool_use unread | - | - | **untested** |  |
| `Hook output: updatedToolOutput / updatedMCPToolOutput` | Replace what the model reads | pass; the harness records the original tool_response | recorded value differs from what the model saw | - | **works-with-caveat** |  |
| `Hook output: continue_/stopReason/suppressOutput/systemMessage/decision/async_` | Common controls | pass; harness uses continue_:false on pause | - | (fake, partial) | **works** |  |
| `HookMatcher.matcher / timeout` | Filter and timeout | pass; harness matchers get 24 h in live sessions bridges/claude_live.py:688; default (600 s) in query mode | - | test_claude_background_session.py::test_a_teams_own_hooks_keep_their_timeout_in_a_live_session (fake) | **works** |  |

## E. Built-in tools of CLI 2.1.283 (docs tools-reference; every name found in the binary)

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `Read` | Read a file | adapters/claude.py:147 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_every_hook_event_of_the_teams_own_runs_inside_the_run (fake) | **works** |  |
| `Write` | Create/overwrite a file | adapters/claude.py:174 (risk write) | governed (PreToolUse) and recorded (PostToolUse) | test_claude.py::test_built_in_tools_are_governed_in_every_permission_mode (fake) | **works** |  |
| `Edit` | Edit a file | adapters/claude.py:175 (risk write) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_the_teams_permission_answer_reaches_claude_code_as_given (fake) | **works** |  |
| `Bash` | Shell command (also run_in_background) | adapters/claude.py:181, 1396-1401 (risk write, read-only commands auto-allowed by shell analysis) | governed (PreToolUse) and recorded (PostToolUse) | test_claude.py::test_a_read_after_a_cd_is_placed_from_where_the_shell_may_be; ::test_a_command_a_person_rejected_is_not_run_and_claude_reads_why (fake) | **works** |  |
| `Glob` | File pattern search | adapters/claude.py:148 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `Grep` | Content search | adapters/claude.py:149 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py hook test (fake) | **works** |  |
| `NotebookEdit` | Edit notebook cells | adapters/claude.py:177 (risk write) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `WebFetch` | Fetch a URL | adapters/claude.py:151 (risk read) | governed (PreToolUse) and recorded (PostToolUse); auto-allowed where Claude Code would ask | named only in disallowed lists (fake) | **works-with-caveat** | Egress/exfiltration is not asked about (gap P2) |
| `WebSearch` | Web search | adapters/claude.py:152 (risk read) | governed (PreToolUse) and recorded (PostToolUse); auto-allowed | - | **works-with-caveat** |  |
| `Agent` | Run a subagent | adapters/claude.py:154, 1338-1341 (risk read (hands work on)) | child run; its calls governed and recorded there | test_claude_background_session.py (fake) | **works** |  |
| `AskUserQuestion` | Ask the user multiple-choice questions | adapters/claude.py:1391-1395, 1424-1451 (risk read; answered as the run's ask_user) | pauses the run; answer journaled | test_claude.py::test_claudes_question_to_its_user_is_the_runs_question (fake) | **works-with-caveat** | Untested under bypassPermissions/allowed_tools, where can_use_tool is not called |
| `TodoWrite` | Legacy checklist (off by default) | adapters/claude.py:153 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `TaskCreate` | Task-list bookkeeping | adapters/claude.py:159-162 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `TaskGet` | Task-list bookkeeping | adapters/claude.py:159-162 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `TaskList` | Task-list bookkeeping | adapters/claude.py:159-162 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `TaskUpdate` | Task-list bookkeeping | adapters/claude.py:159-162 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_claudes_own_bookkeeping_and_its_structured_answer_ask_nobody (fake) | **works** |  |
| `TaskStop` | Stop a background task/agent | adapters/claude.py:179 (risk write) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_background_session.py::test_a_background_subagent_that_ends_while_its_turn_still_runs_says_so_on_its_stream (fake) | **works** |  |
| `TaskOutput` | Removed in CLI 2.1.277 | not offered by 2.1.283 (risk -) | - | - | **not reachable** |  |
| `Skill` | Run a skill | adapters/claude.py:156 (risk read (auto)) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py argv only | **works-with-caveat** | Docs: Claude Code asks; harness auto-allows |
| `EnterPlanMode` | Enter plan mode | adapters/claude.py:158 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py bookkeeping test (fake) | **works** |  |
| `ExitPlanMode` | Present plan for approval | adapters/claude.py:157, 1396-1401 (risk read (auto-approved)) | governed (PreToolUse) and recorded (PostToolUse); plan approval answered with allow, no person | - | **works-with-caveat** | Plan-mode approval silently skipped (gap P2) |
| `ListMcpResourcesTool` | List MCP resources | adapters/claude.py:166 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `ReadMcpResourceTool` | Read an MCP resource | adapters/claude.py:167 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `ToolSearch` | Load deferred MCP tools | not in adapters/claude.py:146-182 (risk UNDECLARED -> a person approves each call) | governed (PreToolUse) and recorded (PostToolUse) | - | **broken** | Every deferred-tool load of a team MCP server pauses for a person (gap P1-4) |
| `WaitForMcpServers` | Wait for connecting MCP servers | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** |  |
| `CronCreate` | Schedule a prompt | adapters/claude.py:196, 1266 (risk disallowed) | never offered | test_claude.py::test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered (fake) | **blocked-by-design** | Reason: no run exists when the schedule fires; harness's own schedule tools instead |
| `CronDelete` | Cancel a schedule | adapters/claude.py:196, 1266 (risk disallowed) | never offered | test_claude.py::test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered (fake) | **blocked-by-design** | Reason: no run exists when the schedule fires; harness's own schedule tools instead |
| `CronList` | List schedules | adapters/claude.py:196, 1266 (risk disallowed) | never offered | test_claude.py::test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered (fake) | **blocked-by-design** | Reason: no run exists when the schedule fires; harness's own schedule tools instead |
| `ScheduleWakeup` | Self-paced /loop wakeup | adapters/claude.py:196, 1266 (risk disallowed) | never offered | test_claude.py::test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered (fake) | **blocked-by-design** | Reason: no run exists when the schedule fires; harness's own schedule tools instead |
| `Monitor` | Background command/WebSocket event stream | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_background_session.py::test_the_task_tools_the_bundled_cli_exposes_have_a_declared_risk asserts it is absent | **works-with-caveat** | Its streamed events between turns have no run |
| `PowerShell` | PowerShell command | not in SHELLS (governance/shell.py:44) (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** | No read-only analysis (matters on Windows) |
| `LSP` | Code intelligence | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** | A read-only tool asks a person |
| `EnterWorktree` | Create/enter a git worktree | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py::test_a_tool_that_says_nothing_of_what_it_does_asks_a_person_once (fake) | **works-with-caveat** | Shell analysis keeps the old cwd after the switch |
| `ExitWorktree` | Leave a worktree | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** |  |
| `Workflow` | Dynamic workflow of many subagents | final: agents' calls carry agent_type 'workflow-subagent', no task, so they fall to the parent run and are refused as UNATTENDED between live turns (risk UNDECLARED -> person (final); 'read' + child run (bg-claude-da); disallowed (react-background)) | parent run, not a child run (final) | - | **broken** | Final branch; see gap P1-5 |
| `SendMessage` | Message a subagent/teammate/other session | not listed (risk UNDECLARED -> person (final); disallowed in both open branches) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** | Resumed subagent tasks are not tied to an Agent call |
| `ListAgents` | List messageable agents/sessions | adapters/claude.py:168 (risk read (auto)) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** | Lists other local Claude sessions on the worker (other tenants' runs) without asking (gap P1-3) |
| `TeamCreate / TeamDelete` | Agent teams (behind CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS) | not listed (risk UNDECLARED (final); disallowed in both branches) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `Artifact` | Publish a page to claude.ai | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** | Needs /login auth; not available through the gateway's API key |
| `PushNotification` | Desktop/phone notification | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `RemoteTrigger` | Cloud scheduled tasks on claude.ai | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** | Outlives the run like CronCreate but is not disallowed |
| `ReportFindings` | Structured code-review findings | adapters/claude.py:171 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_background_session.py risk assertion only | **works** |  |
| `SendFeedback` | Draft feedback report | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `SendUserFile` | Send a file to the user's device | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `ShareOnboardingGuide` | Upload ONBOARDING.md | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `SubagentHandback` | Subagent final report (auto mode) | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **works-with-caveat** | In auto mode every subagent's hand-back would pause for a person |
| `EndConversation` | End the session | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `StructuredOutput` | Structured answer (output_format) | adapters/claude.py:163 (risk read) | governed (PreToolUse) and recorded (PostToolUse) | test_claude_sdk_features.py bookkeeping test (fake) | **works** |  |
| `SendUserMessage` | Agent-to-user message (--brief) | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `REPL` | Code-running tool | not listed (risk UNDECLARED -> person) | governed (PreToolUse) and recorded (PostToolUse) | - | **untested** |  |
| `Legacy names in BUILTIN_SIDE_EFFECTS: LS, MultiEdit, KillShell, BashOutput, SlashCommand, Task, ListConnectors, ReadNotifications, SearchMcpRegistry, SuggestConnectors` | Not tools of 2.1.283's documented set | adapters/claude.py:146-182; native() publishes them when tools is unset adapters/claude.py:1507 (risk listed) | - | - | **works-with-caveat** | Published tool set has ghost tools and lacks real ones; hint narrowing cannot hide the missing ones |

## F. Control protocol

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `initialize (out)` | Hooks, agents, excludeDynamicSections, systemPromptSnapshot, skills, forwardSubagentText | SDK-built from the transformed options | - | test_claude_sdk_features.py options test reads it (fake) | **works** |  |
| `can_use_tool (in)` | Permission request | wrapped adapters/claude.py:1367-1421; live relay bridges/claude_live.py:760-775 | governed | (fake); (real, gated) | **works** |  |
| `hook_callback (in)` | Hook invocation | wrapped; live relay bridges/claude_live.py:695-732 | governed/recorded | (fake) | **works-with-caveat** | Exceptions fail open |
| `mcp_message (in)` | Call into an in-process MCP server | trellis server; live relay bridges/claude_live.py:777-804 | harness tools governed by the bridge | (fake); (real, gated) | **works** |  |
| `control_cancel_request (in)` | CLI abandons a request | SDK cancels the handler task | - | - | **untested** |  |
| `interrupt (out)` | Stop the turn | used bridges/claude_live.py:1068-1071 | - | (fake) | **works** |  |
| `stop_task (out)` | Stop one task | used bridges/claude_live.py:917-920 | - | (fake) | **works** |  |
| `set_permission_mode (out)` | Mid-session change | not exposed | - | - | **blocked-by-design** | Per-run options instead |
| `set_model (out)` | Mid-session change | not exposed | - | - | **blocked-by-design** | Per-run options instead |
| `rewind_files (out)` | SDK method exists | not exposed by the harness | - | - | **not reachable** |  |
| `mcp_reconnect (out)` | SDK method exists | not exposed by the harness | - | - | **not reachable** |  |
| `mcp_toggle (out)` | SDK method exists | not exposed by the harness | - | - | **not reachable** |  |
| `mcp_status (out)` | SDK method exists | not exposed by the harness | - | - | **not reachable** |  |
| `get_context_usage (out)` | SDK method exists | not exposed by the harness | - | - | **not reachable** |  |
| `end_session (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `cancel_async_message (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_max_thinking_tokens (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `reload_plugins (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `reload_skills (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `reload_output_styles (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_mcp_permission_mode_override (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_chrome_browser_hints (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_prompt_suggestions_paused (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `apply_flag_settings (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `update_settings (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_memory_dialog (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_skills_dialog (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_status (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `export_conversation (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_chrome_dialog (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_chrome_browsers (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `select_chrome_browser (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_sandbox_dialog (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `rewind_conversation (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `fork_conversation (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `seed_read_state (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_session_cost (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `list_models (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_usage (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_binary_version (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `file_suggestions (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `read_file (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_workspace_diff (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_plan (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `background_tasks (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_settings (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `get_hooks_listing (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `list_permission_rules (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `submit_feedback (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `message_rated (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `generate_session_title (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `side_question (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_call (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `prefetch_attachments (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `rename_session (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_set_servers (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_cwd (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `claim_session (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `add_directory (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `register_repo_root (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `remote_control (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `channel_enable (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `ultrareview_launch (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `claude_authenticate (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `claude_oauth_callback (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `claude_oauth_wait_for_completion (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_authenticate (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_clear_auth (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_oauth_callback_url (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `set_color (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `mcp_read_resource (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `poll_event (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `stage_file (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `turn_handoff (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `register_device_hooks (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `upload_device_hook_template (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `remote_tools_announce (out, CLI only)` | CLI control request (meaning from its name) | no public SDK 0.2.160 method | - | - | **not reachable** |  |
| `elicitation (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** | elicitation: a team MCP server's question is declined |
| `request_user_dialog (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `oauth_token_refresh (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `host_auth_token_refresh (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `remote_tool_call (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `remote_plumbing_call (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `remote_tools_probe (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `remote_tools_reannounce (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |
| `remote_control_work_secret (in, CLI only)` | CLI asks the host | SDK 0.2.160 answers 'Unsupported control request subtype' (_internal/query.py:668) | - | - | **not reachable** |  |

## G. CLI capabilities reachable from the SDK (flags via extra_args, env, settings)

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `--agent` | Main-thread agent | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--dangerously-skip-permissions / --allow-dangerously-skip-permissions` | Bypass checks | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | PreToolUse governance still runs |
| `--autocompact` | Auto-compact window | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--bare` | Skip settings hooks, CLAUDE.md, auto memory | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Would fix the host-state leak (gap P1-2) |
| `--brief` | SendUserMessage tool | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--chrome / --no-chrome` | Claude in Chrome | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--client-data-url` | Signed config document | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--debug / --debug-file` | Debug logs | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--disable-slash-commands` | Disable all skills | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--file` | Download file resources at start | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `-n / --name` | Session display name | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--no-session-persistence` | No transcript on disk | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Defeats the journal session store |
| `--permission-prompts host\|none` | Who answers prompts | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | 'none' silently replaces person approvals with denials |
| `--plugin-url` | Plugin zip from URL | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--prompt-suggestions` | prompt_suggestion messages | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--replay-user-messages` | Echo user messages (needed for rewind UUIDs) | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--restricted` | Remove code-running tools, confine file tools | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `--safe-mode` | Disable customizations | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `-w / --worktree, --tmux` | Session in a worktree | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Shell analysis assumes options.cwd |
| `--ax-screen-reader, --bg, --cloud, --environment, --teleport, --remote-control, --from-pr, --ide` | Interactive/cloud features | reachable through extra_args/env/settings, passed as-is | - | - | **not reachable** | Not meaningful under --print stream-json |
| `CLI subcommands (agents, attach, auth, auto-mode, doctor, gateway, import, install, logs, mcp, plugin, project, respawn, rm, setup-token, stop, ultrareview, update)` | Management commands | reachable through extra_args/env/settings, passed as-is | - | - | **not reachable** | Not driven by the SDK |
| `env CLAUDE_CODE_DISABLE_AUTO_MEMORY / settings autoMemoryEnabled` | Auto memory off | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Harness never sets it |
| `env DISABLE_AUTO_COMPACT / CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | Compaction control | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env CLAUDE_CODE_ENABLE_TASKS` | TodoWrite vs Task tools | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env ENABLE_TOOL_SEARCH` | Tool search on/off | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Off would avoid gap P1-4 but surfaces WaitForMcpServers |
| `env CLAUDE_CODE_DISABLE_WORKFLOWS` | Workflows off | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` | Background tasks off | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | Agent teams | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env ENABLE_CLAUDEAI_MCP_SERVERS / settings disableClaudeAiConnectors` | claude.ai connectors | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Loaded when the host CLI is logged in with claude.ai |
| `env CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS` | query() stdin wait for background work | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env CLAUDE_CONFIG_DIR` | Relocate ~/.claude | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Only the live test sets it (cli_home fixture) |
| `env CLAUDE_AGENT_SDK_CLIENT_APP` | User-Agent app id | reachable through extra_args/env/settings, passed as-is | - | - | **untested** |  |
| `env MAX_MCP_OUTPUT_TOKENS` | MCP result cap | reachable through extra_args/env/settings, passed as-is | - | - | **works** | Harness sets 60000 adapters/claude.py:1182-1183 |
| `settings crossSessionInbound` | Refuse peer messages | reachable through extra_args/env/settings, passed as-is | - | - | **untested** | Harness never sets refuse (gap P1-3) |

## H. Claude Code features (concepts the docs list)

| feature | what it does | harness handling (file:line) | governed / recorded | tested | status | note |
|---|---|---|---|---|---|---|
| `CLAUDE.md / rules (project memory)` | Persistent instructions loaded at start | CLI-native; loaded from the worker's cwd and parents and ~/.claude when setting_sources is None | not recorded; separate from the memory service | - | **works-with-caveat** | Host files leak into every tenant's run (gap P1-2) |
| `Auto memory (~/.claude/projects/<p>/memory)` | CLI saves learnings with Write/Edit and loads them | CLI-native; never disabled | writes governed as Write/Edit; not in the memory service | - | **works-with-caveat** | Shared across users/tenants on one worker (gap P1-2) |
| `Subagent memory (AgentDefinition.memory)` | Per-subagent persistent memory scope | pass | not recorded | - | **untested** |  |
| `Compaction (auto, /compact, PreCompact/PostCompact)` | Summarise older context | CLI-native; compact_boundary ignored; /compact prompt ends SUCCESS with empty answer | - | test_claude_sdk_features.py::test_a_slash_command_no_model_replied_in_is_a_turn_done_with_an_empty_answer (fake) | **works** |  |
| `Skills` | SKILL.md folders, Skill tool | pass; harness ClaudeSkills plugin per run | Skill auto-allowed | (fake, argv) | **works** |  |
| `Slash commands / custom commands / @-mentions` | Prompt-level dispatch and file attachment | end-user prompt text is sent as-is (verbatim_prompts off) | @path reads happen without PreToolUse: not governed, not recorded | - | **broken** | Gap P0-1 |
| `Output styles` | System-prompt styles from settings | CLI-native via settings | - | - | **untested** |  |
| `Plugins` | Commands, agents, skills, hooks, MCP from a plugin | pass (local) | plugin hooks are command hooks the harness does not see | (fake, argv) | **works** |  |
| `MCP: stdio/sse/http/sdk` | External and in-process servers | pass/merged | governed and recorded | (fake); http (real, gated) | **works** |  |
| `MCP: claude.ai connectors (claudeai-proxy)` | Account connectors | loaded by the CLI if the host is logged in | UNDECLARED -> person | - | **untested** |  |
| `MCP elicitation (team servers)` | Server asks the user | SDK 0.2.160 rejects the control request | - | - | **not reachable** |  |
| `MCP resources` | List/read resources | built-in tools, read | governed/recorded | - | **untested** |  |
| `Tool search` | Deferred tool loading | trellis server alwaysLoad; team servers deferred | ToolSearch asks a person | test_claude.py::test_the_harness_tools_are_loaded_from_the_first_step (fake, config only) | **broken** | Gap P1-4 |
| `Sandboxing` | OS sandbox for Bash | pass | - | (fake, settings merge) | **works** |  |
| `Permission modes incl. auto classifier` | default/acceptEdits/plan/bypass/dontAsk/auto | pass | governance on top | (fake) | **works** |  |
| `Checkpointing / rewind` | File backups and rewind | flag passes; rewind unreachable | - | - | **not reachable** |  |
| `Sessions: resume / fork / continue` | Conversation continuity | harness-owned via journal | - | (fake); resume (real, gated) | **works** |  |
| `Session storage (SessionStore)` | External transcript mirror | JournalSessions | - | (fake) | **works-with-caveat** | No list_subkeys |
| `Cost tracking` | total_cost_usd, usage, model_usage | delta of total_cost_usd + usage | budget | test_claude.py::test_a_query_resumed_after_a_pause_or_at_its_limit_is_charged_its_cost_once (fake) | **works-with-caveat** | model_usage ignored |
| `Structured outputs` | JSON schema answer | answer = structured_output | - | (fake) | **works** |  |
| `Streaming input (multimodal)` | Streamed user messages with images | prompt is a str; non-str input JSON-dumped | - | - | **not reachable** |  |
| `Todo / task tracking` | TodoWrite/Task* tools | read, recorded as tool calls | recorded | (fake) | **works** |  |
| `Background tasks (Bash run_in_background, background subagents)` | Work past the turn | live session keeps the process; background subagent = child run held open | governed/recorded per child run | test_claude_background_session.py (fake) | **works** |  |
| `Dynamic workflows` | Workflow tool | see Workflow row | - | - | **broken** |  |
| `Agent teams` | Teammates | UNDECLARED (final); disallowed in branches | - | - | **untested** |  |
| `Cross-session messaging` | ListAgents/SendMessage, inbox socket | not configured; inbound peer turns' results ignored, their calls refused | peer text still enters the model's context | - | **works-with-caveat** | Gap P1-3 |
| `Scheduled tasks (cron, /loop)` | In-session schedules | disallowed | - | (fake) | **blocked-by-design** | No run exists when they fire |
| `Channels (MCP push)` | Server pushes messages into the session | injected turns (origin channel): result ignored, calls refused | - | - | **untested** |  |
| `OpenTelemetry` | Trace context to the CLI | SDK passes traceparent env; harness adds gateway headers | traced | - | **untested** |  |
| `Prompt suggestions` | prompt_suggestion after a turn | ignored | - | - | **untested** |  |


## Gaps, ranked

### P0

1. **End-user prompt text is dispatched by Claude Code: `@path` file reads and slash commands, ungoverned and unrecorded.** The harness sends the run's input as the prompt (`adapters/claude.py:351-355`, `:627`) and never sets `verbatim_prompts` (no reference in `src/`). The SDK's own docstring for `verbatim_prompts` (`types.py`, field docs) says that without it an `@/absolute/path` in the text makes Claude Code read a local file, and `/command` text runs commands. Neither goes through `PreToolUse`, so governance never sees the read and `tools/bridge.py` `observed` never records it. In a multi-user deployment, a chat message can read worker files (credentials, other tenants' outputs) or run `/clear` and skills. Only an argv test exists (`test_claude_sdk_features.py::test_each_form_an_option_takes_reaches_the_cli`, against the stand-in, which ignores the field). Not checked with the real CLI; the SDK's documentation is the evidence. Fix: default `verbatim_prompts=True` for text that comes from users, or stamp `client_composed` on each message.

### P1

1. **Governance fails open when the harness's hook raises.** `_governance.decide` catches only `Paused` (`adapters/claude.py:1326-1330`). Anything else, such as a run-store error in `runtime.progress(now=True)` (`:1316`) or a catalog error in `bridge.permitted`, goes back to the CLI as a hook error. Per the docs (agent-sdk/hooks "Block the action when a callback fails"), "a PreToolUse guard that crashes then lets the tool call reach the normal permission flow". Under `bypassPermissions`, `acceptEdits`, `allowed_tools` or a settings allow rule, the call then runs with no governance. The CLI reports it only as a `SystemMessage` with `tag: hook_callback_failed`, which the harness ignores. SDK 0.2.177 adds `HookMatcher(on_failure="block")`; 0.2.160 has no such option. No test covers it.
2. **The host's Claude state is shared across runs, users and tenants.** The harness never sets `setting_sources`, `CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY` or `--bare` (grep finds none in `src/`). So every run loads the worker's `~/.claude/settings.json` and rules, the CLAUDE.md files of the cwd and its parents, and auto memory from `~/.claude/projects/<p>/memory/`. Auto memory is written there too, and claude.ai connectors load when the host CLI is logged in. The docs (agent-sdk/claude-code-features, "What settingSources does not control") warn against exactly this for multi-tenant use. The memory service plays no part (`bridges/claude.py:17-19` says so), and nothing records what auto memory stores. Only the live test isolates the CLI (`tests/live/conftest.py` `cli_home`).
3. **Cross-session messaging crosses tenants on a shared worker.** Every CLI the harness starts runs as one OS user, so the CLI treats them all as "your other sessions". `ListAgents` is "read", so it runs without asking (`adapters/claude.py:168`) and lists the other runs' sessions. `crossSessionInbound` is never set to `refuse`, so a peer message arrives in a live session as an injected turn. Its result is ignored and its tool calls are refused (`bridges/claude_live.py` UNATTENDED), but its text enters the model's context. `SendMessage` is UNDECLARED on final (a person approves it) and disallowed on both open branches. Untested; the evidence is docs/cross-session-messaging "Availability".
4. **`ToolSearch` (and `WaitForMcpServers`) asks a person on every call.** Neither is in `BUILTIN_SIDE_EFFECTS` (`adapters/claude.py:146-182`), so they get UNDECLARED risk and a person approves each call. That behaviour is proven for an unknown tool in `test_claude_sdk_features.py::test_a_tool_that_says_nothing_of_what_it_does_asks_a_person_once`. The harness itself says Claude Code defers team MCP servers' tools behind ToolSearch (`tools/convert/claude.py:12-16`; only `trellis` gets `alwaysLoad`). So a team MCP server's tools cannot be used without a human approving each load step. Untested.
5. **Dynamic workflows (`Workflow`) on final.** A launch is UNDECLARED, so a person approves it. Its agents name `agent_type: workflow-subagent` and no task, so `helpers.of` finds none and their calls fall to the parent run (`adapters/claude.py:1133-1134`, `:1493-1494`). In a live session, the calls they make after the turn ends are refused as UNATTENDED (`bridges/claude_live.py:719-731`), so a workflow breaks there. The open branches disagree. **react-background** (`/var/tmp/cons/reactbg`, adapter diff vs d2bf754) disallows `Workflow`, `SendMessage`, `TeamCreate` and `TeamDelete`. **bg-claude-da** (`/var/tmp/cons/bgclaude`) makes each workflow a background child run (`Kind.WORKFLOW`, `WORKFLOW_AGENT`), with a "shared" run for agents it cannot attribute while several workflows run; that part is uncommitted work in progress in its worktree. One of the two has to win when they merge: as they stand, react-background would disallow the tool bg-claude-da supports.

### P2

1. **`BUILTIN_SIDE_EFFECTS` is stale against 2.1.283.** It lists names that are not tools of 2.1.283 (`LS`, `MultiEdit`, `KillShell`, `BashOutput`, `SlashCommand`, `Task`, `ListConnectors`, `ReadNotifications`, `SearchMcpRegistry`, `SuggestConnectors`), and `native()` publishes them as the agent's tools when `tools` is unset (`:1507`). It omits these real tools: Monitor, PowerShell, LSP, EnterWorktree, ExitWorktree, Workflow, SendMessage, ToolSearch, WaitForMcpServers, Artifact, PushNotification, RemoteTrigger, SendFeedback, SendUserFile, ShareOnboardingGuide, SubagentHandback, EndConversation, SendUserMessage, REPL. Each of those asks a person on every call, and tool-hint narrowing (`_disallowed`, `:1258-1267`) cannot hide them. `RemoteTrigger` outlives the run just as `CronCreate` does, yet it is not disallowed.
2. **`ExitPlanMode` is auto-approved.** It is "read", so `can_use_tool` answers allow (`:157`, `:1396-1401`). In plan mode, the plan approval that Claude Code asks a person for is answered with no person asked. Untested.
3. **`WebFetch` and `WebSearch` are "read"**, so they are auto-allowed where Claude Code would ask. Network egress and exfiltration go unasked unless the catalog says otherwise. `Skill` is likewise auto-allowed, although the docs say Claude Code asks.
4. **`resume_session_at` and `resume_drops_turn` stay on every resumed attempt.** `_options` replaces only `resume` (`:1203-1208`). After a pause, the CLI would truncate the run's own (forked) session at the team's UUID and throw away the paused turn's work. Untested.
5. **Team hooks in a live session.** With no turn under way, every non-PreToolUse callback returns `{}` (`bridges/claude_live.py:719-720`). So the team's `Stop`, `Notification`, `SubagentStop` and the other hooks never run between turns. Separately, the CLI dispatches matchers concurrently (types.py `hooks` docstring), while the doc says the team's hooks run "before the harness's own". Governance therefore decides on the original arguments even when a team `PreToolUse` hook returns `updatedInput`. The stand-in calls hooks in order, so no test can see this.
6. **The record differs from what the model read** when a team `PostToolUse` hook returns `updatedToolOutput`: `_recorded` stores the original `tool_response` (`:1485`).
7. **An identical built-in call in a resumed attempt is answered from the journal** ("ran before", `tools/bridge.py` `permitted`, with `journal.call` keyed by occurrence). For example, a `Read` of a file that was edited since the pause returns its pre-pause content.
8. **Real-CLI coverage.** Every integration test drives the stand-in, and its behaviour is the authors' observation of 2.1.283. Only `tests/live/test_live_claude.py::test_claude_runs_through_the_gateway_and_resumes_its_session` and `tests/live/test_live_gateway_features.py::test_the_claude_agent_sdks_mcp_servers_on_the_gateway` touch the real CLI, and both need a gateway. Real built-in tools, background subagents, sandbox, compaction, skills and the live-session path have never run against the real binary in CI. `docs/matrix.md` note 12 says the real CLI's concurrency is unverified.

### P3

1. There is no token-level streaming even with `include_partial_messages` (`:360-372`).
2. None of these becomes a run event or warning: `RateLimitEvent`, `api_retry`, `compact_boundary`, `model_fallback`, `hook_callback_failed`, `MirrorErrorMessage`, `ConversationResetMessage`, `AssistantMessage.error`.
3. `ThinkingBlock` and server-tool blocks (advisor, web_search, code_execution) are not recorded.
4. `JournalSessions` has no `list_subkeys`, `list_sessions` or `delete` (`bridges/claude.py:114-140`). So a resume never restores subagent transcripts (`_internal/session_resume.py:179`), and the `*_from_store` helpers cannot work on the journal.
5. These `ClaudeSDKClient` controls are unreachable: `set_model`, `set_permission_mode`, `rewind_files`, MCP status/toggle/reconnect, `get_context_usage`, `get_server_info`. So `enable_file_checkpointing` takes checkpoints that can never be used.
6. When a team's own MCP server elicits, the request is declined, because SDK 0.2.160 rejects the `elicitation` control request.
7. Prompts are str only: no multimodal (image) input.
8. `_queried` uses the private `InternalClient` (`:819-827`).
9. Cost is priced by `options.model`, and `model_usage` is ignored.
10. The 23 CLI hook events that the SDK's `HookEvent` lacks are untested, and the `defer` decision is untested.
11. The live-test skip checks `which claude`, but the SDK uses its bundled binary.
12. `PowerShell` is not in `SHELLS`, so it gets no read-only analysis.

## What the harness changes in how the library behaves

The user's rule is that library behaviour must not change. These are changes:

1. **System prompt:** `DELIVERS_HERE` is appended to every string, custom or preset prompt, and `None` becomes a custom prompt (`adapters/claude.py:1188, 1212-1229`). This contradicts the doc line "the options' system_prompt is left as you wrote it".
2. **Permission path:** a `can_use_tool` is always installed (`:1190`), so the SDK forces `--permission-prompt-tool stdio`. Prompts that headless Claude Code would deny now get an answer instead: reads, `WebFetch`, `WebSearch`, `Skill`, `ExitPlanMode` and read-only `Bash` are allowed automatically, and anything else asks a person. `CanUseToolShadowedWarning` now fires for teams with whole-tool `allowed_tools`.
3. **Hooks added:** a `PreToolUse` governance hook (deny/ask/updatedInput), `PostToolUse` record and steering text (`additionalContext`), and `UserPromptSubmit` memory context. The `Agent` tool's prompt is rewritten with a memory briefing (`:1338-1341`).
4. **`disallowed_tools`** gains `CronCreate`, `CronDelete`, `CronList`, `ScheduleWakeup` and the hint-narrowed tools. The branches add `Workflow`, `SendMessage`, `TeamCreate` and `TeamDelete`.
5. **`env`:** `MAX_MCP_OUTPUT_TOKENS=60000`, and the gateway's `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` and `ANTHROPIC_CUSTOM_HEADERS`.
6. **`max_buffer_size`:** 1 MiB becomes 100 MiB.
7. **`mcp_servers`:** a JSON string or file path is parsed into a dict (the SDK would pass it to `--mcp-config` as-is), and the `trellis` server is added with `alwaysLoad`.
8. **Sessions:** `session_store` is replaced by `JournalSessions` with flush `eager`. On later attempts `resume` is replaced and `session_id` cleared. `fork_session` is forced when another worker holds the session.
9. **Turn limit:** the `error_max_turns` result is swallowed (including the SDK's `ResultError`) and an extra query runs (`:503-549`).
10. **Failed resume:** a resumed session that errors is run again from the prompt, so built-ins run again (`:551-583`).
11. **Errors:** an `is_error` result becomes `ClaudeRunError`, and results of injected turns are ignored as answers.
12. **Stall watchdog:** a model step that sends nothing for 600 s ends the run (`runtime.STALL_SECONDS`).
13. **Process model:** a conversation keeps one CLI per conversation across turns (`ClaudeSDKClient`), where the library would start a process per query. Between turns, team hooks are not run and tool calls are refused. `interrupt` is avoided while background work runs.
14. **Plugins:** `ClaudeSkills` are written as a temporary plugin for each run.
15. **Replay:** an identical built-in call in a resumed attempt is answered from the journal instead of running.
16. **`AskUserQuestion`:** its answers are injected through `updated_input`.

## Features newer than 0.2.160 / 2.1.283 that would matter

The latest SDK is 0.2.182 (PyPI, 2026-11-13), which bundles CLI 2.1.325. The CHANGELOG at tag v0.2.182 lists entries only up to 0.2.178; 0.2.179-0.2.182 are diffed from the sdist.

- **`HookMatcher.on_failure="block"` (0.2.177).** Closes gap P1-1: the harness's governance matcher should use it.
- **`client_app` option (between 0.2.179 and 0.2.182).** Sets `CLAUDE_AGENT_SDK_CLIENT_APP` for the User-Agent. This is the only new `ClaudeAgentOptions` field.
- **Typed hook-specific outputs and inputs (0.2.166)** for SessionStart, UserPromptExpansion, PostToolBatch, Stop, SubagentStop, PermissionDenied, Pre/PostModelSwitch, Elicitation/ElicitationResult, CwdChanged, FileChanged, WorktreeCreate and MessageDisplay. Also added: `HookEffort`, `BackgroundTaskSummary`, `SessionCronSummary`, and Stop input with `background_tasks`/`session_crons`. Useful for the background-task tracking the live bridge does by hand. The `HookEvent` literal is still the same 10 names in 0.2.182.
- **`SystemInitData` and related types (0.2.166).** The init payload is typed.
- **`SandboxSettings.ignoreViolations` type fix (0.2.166).** The 0.2.160 shape (`file`/`network`) is misread by the CLI as command patterns.
- **Parallel tool calls kept in `get_session_messages` (0.2.170).**
- **CLI settings:** `permissions.blockWritesOutsideWorkingDirectories` (2.1.314) and `permissions.denyInsteadOfPrompting` (2.1.319). Both are relevant to confining a multi-tenant worker.

## Open branches, read without touching their worktrees

- **bg-claude-da** (`/var/tmp/cons/bgclaude`, 75e9890 plus uncommitted changes to `bridges/claude_live.py` and its tests): makes `Workflow` "read" and turns each workflow into a `Kind.WORKFLOW` child run. It disallows `SendMessage`, `TeamCreate` and `TeamDelete` (`UNHELD_AGENTS`), records `made_by` for calls by workflow agents, and the WIP adds a shared "workflows" run for concurrent workflows.
- **react-background** (`/var/tmp/cons/reactbg`, 3d892eb): its Claude change is 115348a, which disallows `Workflow`, `SendMessage`, `TeamCreate` and `TeamDelete`. The rest of the branch is ReAct/Deep Agents background work.

