# Kimi Code 2.1.1 over ACP: every feature, and what the Trellis harness does with it

Read-only audit, 2026-11-14. No code was changed and no paid model was called.

**Sources**
- The installed package `@moonshot-ai/kimi-code@2.1.1` (`/root/.kimi-code-npm/lib/node_modules/@moonshot-ai/kimi-code/dist/main.mjs`, 490,755 lines, readable `//#region` markers). `K:n` means a line in that file.
- `kimi --help`, `kimi --version` (2.1.1) and `kimi acp --help`.
- `agent-client-protocol` 0.12.1 (`/home/user/agent-harness/.venv/.../acp`).
- The kimi-code GitHub repo at `f082f04` (2026-11-14): its docs (`docs/en/**`), its changelog (`docs/en/release-notes/changelog.md`, `apps/kimi-code/CHANGELOG.md`) and `packages/acp-server/CHANGELOG.md`. I cloned it with git because the docs site is blocked (see below). The npm registry gave the version list.
- The harness at `/var/tmp/cons/final-agent-harness` (HEAD `d2bf754`). `H:n` means a line in `src/trellis/harness/adapters/acp.py`.
- The open branch `/var/tmp/cons/bgkimi` (`7971933`, two commits ahead), read only.

**Blocked hosts:** `moonshotai.github.io`, the docs site (DNS is not allowed in this session). The same docs come from the GitHub repo, which I cloned instead. The GitHub REST API (`gh api`) is not enabled for this repo, but anonymous git clone works.

**What I ran (no paid model, nothing changed in the trees):**
- The suites `tests/integration/test_kimi.py` (real Kimi 2.1.1, scripted local endpoint): **27 passed**.
- `tests/integration/test_acp.py` (scripted fake ACP agent): **59 passed**.
- My own probes against real Kimi 2.1.1 and the local scripted endpoint: `/var/tmp/cons/inventory-tmp/kimi/probes/test_probes.py`, with results quoted as "probe" below. `git status` stayed clean in both trees. I ran them with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.

**Status words:**
- *works*: passed through or mapped correctly, and tested.
- *caveat*: it works, but with a limit or a change the user should know about.
- *broken*: it does not do what Kimi does, or what the harness claims.
- *blocked-by-design*: the harness turns it off on purpose, for a stated reason.
- *untested*: the code path exists, but no test exercises this feature.
- *not reachable*: no harness path can use it.

Columns: **H** = how the harness handles it (pass / transform / block / override). **G/R** = governed / recorded. **Test** names a test function: `kimi::` is test_kimi.py with real Kimi, `acp::` is test_acp.py with the fake agent.

---

## 1. ACP methods (agent side: what the client calls)

| # | Feature (Kimi source) | What it does | H (file:line) | G/R | Test | Status |
|---|---|---|---|---|---|---|
| 1 | `initialize` (K:312150-312183) | Negotiates protocol v1. Returns agentInfo `Kimi Code CLI 2.1.1`, `loadSession`, prompt caps (image ✓, audio ✗, embeddedContext ✓), session caps (list/resume/close/delete/fork/additionalDirectories), MCP http+sse, `auth.logout`, terminal-auth method | pass. Client caps are empty unless sandboxed (H:677, H:1495-1499). The version is checked by `compat.require_binary` (H:700-702) | – | kimi::test_kimi_is_the_pinned_version, acp::test_a_kimi_outside_the_tested_range_is_refused | works |
| 2 | `authenticate` / terminal auth (`kimi acp --login`, K:309431-309466) | Device-code login for ACP clients | not used. A missing login becomes `auth_required` and the run fails | – | acp::test_an_agent_that_needs_a_login_fails_saying_so | blocked-by-design (log in out of band, or set `KIMI_MODEL_*`) |
| 3 | `logout` (K:312270) | Drops the managed provider token | not used | – | – | not reachable |
| 4 | `session/new` (K:312184-312195) | Creates a session (cwd, mcpServers, additionalDirectories) | used (H:911); its own and the harness's MCP servers are passed | – | many | works |
| 5 | `session/new.additionalDirectories` | Adds workspace roots that Kimi's path policy and system prompt include | never passed. `reads=` is a harness-only governance list (H:676) | – | – | not reachable |
| 6 | `session/resume` (K:312224-312233) | Re-attaches without history replay | used when the agent advertises it (H:890-904) | – | kimi::test_bash_asks_a_person_and_the_resume_continues_kimis_session; kimi::test_a_conversations_next_turn_continues_kimis_session_when_its_runs_have_skills | works |
| 7 | `session/load` (K:312213-312223) | Restores the session and replays its history as `session/update` | fallback path. The replay is suppressed (`loading`, H:898, H:1213) | – | acp::test_an_agent_without_session_resume_loads_the_session | works (Kimi always offers resume, so this path is for other agents) |
| 8 | `session/fork` (unstable, K:312196-312212) | Branches a session; ignores mcpServers and additionalDirectories | final branch: not used. bgkimi: forks a session another worker holds | – | bgkimi only | not reachable (final) |
| 9 | `session/list` / `session/close` / `session/delete` | Lists, tears down or deletes sessions | not used. The attempt ends by killing the process. Unsandboxed sessions build up in `~/.kimi-code/sessions` forever (they are linked from the run home, H:1442-1447) | – | – | not reachable |
| 10 | `session/set_mode` (K:312310-312315, K:311991-311999) | default / plan / auto / yolo, mapped to engine plan + permission toggles | sent only when `mode=` is given, on a new session (H:913-916). auto/yolo are refused (H:454-460). **It is never sent otherwise, so Kimi's config defaults win (rows 52-53)** | – | acp::test_the_session_mode_is_set_and_one_that_asks_nothing_refused | caveat |
| 11 | `session/set_config_option` (model / thinking / mode) and the `session/set_model` extension | Picks the model and thinking level from the catalog | not used. The model goes through env (`KIMI_MODEL_NAME`, H:271-277) | – | – | not reachable (env passthrough instead) |
| 12 | `session/prompt` text blocks | One turn | the prompt is a list of text blocks: `[DELIVERS_HERE?, <memory_context>?, prompt]` (H:965-977). Non-string input is JSON-dumped (H:532) | – | kimi::test_kimi_calls_a_harness_tool_and_answers, acp::test_the_context_goes_before_the_prompt_marked_off_from_it | works (transform) |
| 13 | `session/prompt` image / resource / resource_link blocks (K:309525-309586) | Image input (compressed, originals kept for ReadMediaFile) and embedded files | the harness only ever sends text | – | – | not reachable |
| 14 | `session/cancel` (K:312298-312309) | Interrupts the turn | used for pause, reject, edit, budget and run cancel (H:1059-1066, H:1371-1373) | – | acp::test_a_cancelled_run_stops_the_agent, kimi::test_a_call_kimi_aborted_when_its_turn_was_stopped_did_not_run | works |
| 15 | ACP slash commands over a prompt: `/compact /status /usage /mcp /tasks /help`, `/skill:<name>`, unknown → "Unknown ACP command" (K:309882-309970, K:311160-311228, K:311542-311552) | Commands run with no model turn | not used. The harness puts `DELIVERS_HERE` (unsandboxed) or `<memory_context>` first, and Kimi only parses the *first* block (`leadingText`, K:311232), so a user prompt starting with "/" goes to the model. **In a sandboxed run with no memory context, the prompt is the first block, so Kimi intercepts "/x"**: the behaviour is inconsistent | – | – | caveat (untested) |

## 2. ACP client side (what the harness serves to Kimi)

| # | Feature | What it does | H | G/R | Test | Status |
|---|---|---|---|---|---|---|
| 16 | `session/request_permission`, tool approvals (K:310498-310666, K:310905-310925) | Options approve_once / approve_always / reject; title = tool name; content = display (diff/plan) + "Requesting approval to ⟨action⟩"; **no rawInput** | harness tools are let through (H:1258-1264). Questions go to `_question`. Everything else goes to `_decided` (H:1269-1301): hooks, governance by `KIMI_BUILTINS` risk (H:171-196), and a person unless the call reads (H:1594-1607). The args come from the streamed `tool_call` text (H:1223-1228, H:1648-1655) | governed; recorded on completion | kimi::test_bash_asks_a_person…, acp::test_its_own_tools_are_governed_and_recorded | works (main agent) |
| 17 | "Approve for this session" (`allow_always`, Kimi's session approval history K:273417) | Remembers an approval for the session | never chosen; always `allow_once` (H:1610-1618) | – | – | blocked-by-design (each call is governed) |
| 18 | Rejection / edit | Kimi expects to continue its turn after a reject | **transform**: `reject_once`, then the turn is cancelled, then a new prompt says why or gives the edited args (H:143-149, H:1290-1300). Kimi's turn ends `cancelled` (an active goal pauses, row 41) | governed | kimi::test_a_rejected_bash_call_is_not_run_and_kimi_reads_why, acp::test_an_edited_call_is_made_again_with_its_arguments_and_runs | works (transform) |
| 19 | Pause on approval | – | the request is answered `cancelled`, then `session/cancel`, the process ends, and the resume tells Kimi to "call X again" (H:1286-1289, H:979-989) | journaled | kimi::test_bash_asks_a_person_and_the_resume_continues_kimis_session | works (transform) |
| 20 | Questions via `request_permission` fallback (K:310932-310980, K:310698-310720) | **Kimi itself degrades to the first question only**: `q0_opt_i` + Skip; no header, option description, multi_select or "Other" | asked as `Runtime.ask` choice; Skip removed (H:1303-1316) | the answer is the run's | acp::test_its_question_is_the_runs_question (fake). **No real-Kimi test.** Probe: 2 questions (2nd multi_select) → only "Which db?" asked; Kimi's result `{"answers":{"Which db?":"pg"}}`, the 2nd dropped | caveat |
| 21 | `elicitation/create` form mode (K:310759-310800, K:310932-310945) | All questions as one JSON-schema form (multi_select arrays, descriptions) | **never advertised**: `ClientCapabilities()` has `elicitation=None` (H:677). The ACP lib 0.12.1 has `ElicitationCapabilities` / `CreateElicitationRequest` and `Client.create_elicitation` (unstable) | – | – | not reachable (this causes row 20's data loss) |
| 22 | `fs/read_text_file`, `fs/write_text_file` (K:312625-312735) | Routes engine text I/O to the client when advertised | sandboxed runs only (H:1319-1330, H:1510-1526). Otherwise refused with method-not-found (H:1358-1365). Kimi's `stat`/`readdir`/`readBytes` and rg-based Glob/Grep stay on the host (documented: "reads are not isolated", H:1384-1388) | – | kimi::test_a_sandboxed_kimi_runs_bash_and_writes_in_the_sandbox, acp::test_a_sandboxed_run_works_in_the_sandbox_and_says_reads_are_not_isolated | caveat (disclosed) |
| 23 | `terminal/*` (K:312742-312970) | Bash (`-c`, NO_COLOR, TERM=dumb) runs in a client terminal | sandboxed runs only (H:1332-1356, H:1528-1567) | **recorded output is empty**: Kimi's final `tool_call_update` carries only `{type:"terminal"}`, no rawOutput (K:311795-311810), and `_output` reads only text (H:1680-1686). Probe: sandboxed `echo boxed-output` recorded `""`, host recorded `"boxed-output\n"` | the two above (neither asserts the record) | broken (record) |
| 24 | `session/update agent_message_chunk` | Answer text | streamed and extracted (H:538-547, H:549-593) | in the transcript and memory | kimi::test_kimis_answer_streams | works |
| 25 | `agent_thought_chunk` | Thinking | dropped from stream, record and memory | – | acp::test_every_other_update_and_an_extension_pass_through_a_turn | caveat |
| 26 | `tool_call` / `tool_call_update` (K:310189-310300) | Calls, streamed args, progress, results | recorded (H:1125-1164, H:1208-1229) | recorded | many | works |
| 27 | `tool_call.title` | **The title is the tool's execution *description* ("Launching coder agent: …", "Running: ls") when no streamed delta came first** (K:310189-310190, K:267859-267866); only the lazy delta-created call carries the tool name (K:310249-310265) | the harness takes the name from the title of the first `tool_call` (H:1218) and ignores later title changes | name ok only if args stream | real tests pass because the scripted endpoint streams | caveat (fragile: a provider that does not stream tool deltas gives wrong names, and Agent/AgentSwarm are not seen as helpers) |
| 28 | `tool_call` content diff, `kind`, `locations` | Diff of edits, file locations | ignored | – | – | caveat |
| 29 | `plan` (TodoList, K:310270-310300) | Todo list as ACP plan entries | ignored (no event, not stored) | only the TodoList call is recorded | acp::test_streaming_carries_the_agents_text (plan ignored) | caveat |
| 30 | `available_commands_update`, `current_mode_update`, `config_option_update`, `session_info_update` (title), `usage_update` (context used/size) | Session metadata | all ignored | – | acp::test_every_other_update_and_an_extension_pass_through_a_turn | caveat |
| 31 | `user_message_chunk` (load replay) | History replay | suppressed while loading (H:1213) | – | acp::test_an_agent_without_session_resume_loads_the_session | works |
| 32 | Stop reasons (K:310100-310108) | completed→end_turn; cancelled; **failed→end_turn** (refusal only if `provider.filtered`); blocked→refusal; auth failures → `auth_required` error; **never `max_tokens` or `max_turn_requests`** | `refusal` fails the run (H:588-589). `MAX_TOKENS`/`max_turn_requests` are dead code for Kimi (H:137-141, H:1018). Kimi's failed-turn error code is read only for the step limit (H:336-343). Any other failure (provider error after retries, context overflow) becomes a generic "ended with no answer" ERROR, or a SUCCESS when partial text came first | – | acp::test_a_refusal_fails_the_run, acp::test_an_error_the_agent_answers_the_prompt_with_fails_the_run | caveat |
| 33 | ACP `PromptResponse.usage` | – | Kimi sends none. The harness reads Kimi's wire log instead (row 60) | – | acp::test_acps_own_usage_counts_when_the_agent_gives_it | works |
| 34 | Extension methods/notifications | – | refused (no ext handlers) | – | acp::test_every_other_update_and_an_extension_pass_through_a_turn | works |

## 3. Built-in tools (2.1.1 registers 42 + MCP)

Kimi's default approval rule: auto for `Read Grep Glob ReadMediaFile TodoList TaskList TaskOutput WaitFor CronList WebSearch FetchURL Agent AgentSwarm AskUserQuestion NotifyUser Skill EnterPlanMode ExitPlanMode CreateGoal GetGoal SetGoalBudget UpdateGoal select_tools` (K:273248-273282). `Write`/`Edit` are auto inside a git worktree (K:273359-273402). Everything else asks (K:273285-273293), except under user deny rules, dangerous-command, sensitive-file and git-control asks (K:273590-273604).

| # | Tool (schema) | What it does | H | G/R | Test | Status |
|---|---|---|---|---|---|---|
| 35 | `Bash` {command, cwd, timeout≤300 fg / bg max, description, run_in_background, disable_timeout} (K:175702) | Shell | asks, then governed as write; read-only commands skip the person (H:1605, governance/shell) | G+R | kimi::test_bash_asks…, kimi::test_a_command_that_only_reads_runs_without_asking_a_person, kimi::test_a_team_that_wants_every_shell_command_approved…, kimi::test_a_bash_call_whose_arguments_stream_pauses_with_its_command | works |
| 36 | `Bash(run_in_background)` and auto-background on timeout (`[task] bashAutoBackgroundOnTimeout`) | Background process task; a completion notification starts a new turn | **the process is killed when the attempt ends** (H:678-709): the task dies; the notification turn is never forwarded (K:311741-311744 `driverFor` drops events of other turns) | – | – (bgkimi: test_acp_background.py) | broken (final); bgkimi keeps the process |
| 37 | `Read` {path or kimi-file://, line_offset (±), column_offset, n_lines, max_chars} (K:182345) | Text read | runs unasked; recorded; reads the host even when sandboxed | R | kimi::test_a_kept_result_stays_out_of_the_workspace_and_pages_smaller_than_kimis_read | works |
| 38 | `Write` {path, content, mode overwrite/append} | – | asks outside a git worktree, then governed. **Inside a git worktree Kimi auto-approves, so the harness never sees it before it runs.** Probe: `git init` cwd → Write ran, no pause | G only outside git; R always | kimi::test_a_file_kimi_saves_in_its_outputs_folder_is_kept_for_its_user (non-git) | caveat (in kimi.md; **matrix F32 "write asks" is not true in git repos**) |
| 39 | `Edit` {path, old_string, new_string, replace_all} | – | same as Write | same | – | caveat |
| 40 | `Glob` {pattern, head_limit, offset, path, include_ignored} / `Grep` {pattern, path, glob, type, output_mode, -i -n -A -B -C, head_limit, offset, multiline, include_ignored} | Search via rg | unasked; recorded; host | R | – | untested |
| 41 | `ReadMediaFile` {path, region, full_resolution} | Image/video into context (by model capability) | unasked; recorded | R | – | untested |
| 42 | `FetchURL` {url} | Local fetch | unasked; recorded | R | – | untested |
| 43 | `WebSearch` {query} (only when `services.moonshot_search` or Kimi OAuth exists, K:248787) | Search | unasked; recorded. Absent under Bifrost/`KIMI_MODEL_*` without a search service | R | – | untested |
| 44 | `TodoList` {todos[{title,status}]} | Todo list | unasked; recorded; plan update ignored (row 29) | R | – | untested |
| 45 | `Skill` {skill, args} | Loads a skill | unasked; recorded | R | kimi::test_kimi_reads_the_runs_skills (the listing only; the tool is never called) | untested (call) |
| 46 | `EnterPlanMode` {} (2.1.1 lets the agent enter plan mode itself; removed in 2.4.0) | Plan mode | unasked; recorded | R | – | untested |
| 47 | `ExitPlanMode` {options[≤3]} (K:205245, plan_review approval K:205775-205830) | Submits the plan; **Kimi always asks the user** (Approve / plan_opt_i, Revise, Reject and Exit) | **override**: `ExitPlanMode` is risk `read` (H:183-184), so no person is asked and `allow_once` picks the first allow option, i.e. the plan is approved and the *first* alternative chosen. Probe: mode="plan" → result "Selected approach: Fast … Approved Plan" with no interrupt | not human-governed | – | broken (behaviour change) |
| 48 | `AskUserQuestion` {questions[1-4]{question, header, options[2-4]{label, description}, multi_select}} (+`background` variant) | Structured questions | see rows 20-21 | the run's question | acp::test_its_question_is_the_runs_question | caveat |
| 49 | `Agent` {prompt, description, subagent_type, resume, run_in_background, fork, model} (K:225636) | Sub-agent (coder / explore / plan / custom) | child run from call to result (H:1166-1205) | child run; its own calls not shown | kimi::test_a_sub_agent_kimi_starts_is_a_child_run | works (foreground) |
| 50 | Sub-agent's own permission requests | Kimi asks about a sub-agent's Bash/Write… while the main turn waits | `_decided` with **no args**: no `tool_call` was seen and the request has no rawInput, so the args become `{"request": "Requesting approval to Running: <cmd, cut at 50 chars>"}` (H:1274-1275, H:1633-1645). Bash's `command` display is dropped by Kimi (K:309738-309762). The shell read-only check, `approve_when` rules and the reviewer all see no real args. Probe: args = `{"request": "Requesting approval to Running: touch from-sub.txt && echo sub-ran"}` | governed blind | acp::test_a_sub_agent_is_a_child_run_and_its_calls_are_governed (fake; asserts the tool only) | broken |
| 51 | `Agent(resume=…)`, `fork=true` (flag `subagent_fork`, off), `model` | Continue / fork a sub-agent | named "coder" (H:1176) and run as a new child; fork inherits the caller type, which the harness misnames | child run | – | caveat (bgkimi: names from `profile.bind`) |
| 52 | `Agent(run_in_background)` | Returns the task id at once | **the child run ends SUCCESS with the launch text** (H:1191-1195); the process is then killed, so the sub-agent dies | – | – | broken (final; bgkimi fixes) |
| 53 | `AgentSwarm` {description, subagent_type, prompt_template, items≤128, fork, resume_agent_ids, model} | Parallel sub-agents | one child per item (H:1178-1184, H:1196-1205). The result is parsed by a fixed-order regex `agent_id item outcome` (H:156-160), but Kimi writes `mode`, `state`, `stop_reason` and escapes items (K:208931-208946), so resumed, unstarted or escaped members are read as failed. `resume_agent_ids` gets no child. **Concurrency is capped at the harness admission limit (start 4, max 16) unless env sets it** (H:667-668); Kimi's default is unlimited plus adaptive (K:207872-207878) | children | kimi::test_a_swarm_kimi_starts_runs_unchanged_each_member_a_child_run, acp::test_a_swarm_runs_as_many_at_once_as_the_harness_admits_and_is_on_the_tree | caveat (behaviour change on concurrency) |
| 54 | `TaskList` {active_only, limit} / `TaskOutput` {task_id} / `WaitFor` {timeout≤600, task_id} (flag `wait_for` on) | Inspect / wait for background tasks | unasked; recorded; only useful inside one attempt | R | – | untested |
| 55 | `TaskStop` {task_id, reason} | Stop a task | asks, then governed (write) | G+R | – | untested |
| 56 | `CreateGoal` {objective, completionCriterion, replace} / `GetGoal` / `SetGoalBudget` {value, unit} / `UpdateGoal` {status} | Goal mode: Kimi **launches continuation turns itself** after each turn ends (K:209724-209746). CreateGoal shows a `goal_start` approval in non-auto mode (K:210121-210131) | CreateGoal is risk write, so a person is asked. **Continuation turns are not forwarded over ACP (events filtered by the prompt's turnId, K:311741), and the process is killed after the prompt's turn, so the goal is paused on interrupt** | G (CreateGoal) | – | broken (goal mode does not run under the harness) |
| 57 | `CronCreate` {cron, prompt≤8KiB, recurring} / `CronList` / `CronDelete` {id} | Scheduled prompts in a live session | **blocked**: `KIMI_DISABLE_CRON=1` forced over the agent's env (H:206-210, H:669-670). The tools are still listed | – | kimi::test_a_schedule_kimi_could_not_keep_is_refused_by_kimi_itself, acp::test_kimis_scheduling_is_turned_off_whatever_its_own_environment_says | blocked-by-design (the process does not outlive the attempt) |
| 58 | Tower tools `TowerInit/Spawn/Status/Send/Inbox/Mission/Plan/Review/Merge/Finding/Teardown` (flag `tower`, env `KIMI_CODE_EXPERIMENTAL_TOWER`, off by default) | Multi-agent git-worktree missions; spawned workers run in Kimi `auto` mode | final: **not forced off**. A user env/config that enables tower exposes the tools; they are not in `KIMI_BUILTINS` (UNDECLARED, so treated as writes), and the workers' calls are never asked about | ungoverned workers | – | caveat (final); bgkimi forces `TOWER=0` (blocked-by-design) |
| 59 | `select_tools` {names} (flag `tool-select`, off; only models with dynamic tools) | Progressive MCP tool loading | not in `KIMI_BUILTINS`; harness MCP tools would be hidden behind it | – | – | not reachable (default off) |
| 60 | `NotifyUser` {message} (flag `notify_user`, off) | Mid-turn progress for the TUI | not in `KIMI_BUILTINS` | – | – | not reachable (default off) |
| 61 | `mcp__<server>__authenticate` (K:270474-270560) | MCP OAuth: the URL arrives only as a tool progress *title* | titles ignored (H:1219-1228); the user never sees the URL | asks → governed | – | not reachable |
| 62 | Kimi's own MCP servers: ACP `mcpServers` (stdio/http/sse; `acp` transport dropped, K:309660-309700), plus user `$KIMI_CODE_HOME/mcp.json`, project `.mcp.json` / `.kimi-code/mcp.json`, plugins; `KIMI_MCP_TOOL_TIMEOUT_MS`, `KIMI_MCP_STARTUP_TIMEOUT_MS` | Extra tools `mcp__srv__tool` | target `mcp_servers` passed (H:682, H:760-769). Calls ask (fallback), are governed UNDECLARED and recorded; seen tools join the native tool set (H:711-714) | G+R | acp::test_the_teams_http_and_sse_servers_go_to_the_agent | works |
| 63 | The harness's own MCP server `trellis` | Harness tools | loopback HTTP + bearer (H:773-781); permission auto-allowed and bridged (H:1258-1264) | G+R (bridge) | kimi::test_kimi_calls_a_harness_tool_and_answers, acp::test_the_harness_mcp_server_refuses_a_request_without_its_token | works |

## 4. Permissions and modes

| # | Feature | What it does | H | G/R | Test | Status |
|---|---|---|---|---|---|---|
| 64 | Mode `default` (manual) | Kimi asks per the policy chain | assumed and governed | G | many | works |
| 65 | Mode `plan` | Read-only planning; ExitPlanMode asks for review | `mode="plan"` sets it. Plan review is auto-approved (row 47) | – | acp::test_the_session_mode_is_set… (fake) | caveat |
| 66 | Modes `auto`, `yolo` | Approve everything; Kimi never asks | refused for `mode=` (H:454-460) | – | acp::test_the_session_mode_is_set_and_one_that_asks_nothing_refused | blocked-by-design |
| 67 | Config `default_permission_mode = "auto"/"yolo"` (K:219929-219939, applied at agent bootstrap K:224295-224299) | Starts every session in that mode. ACP still reports `default` (K:311312, K:311967) | **not overridden**: no `set_session_mode("default")` unless `mode=` is given (H:913). Unsandboxed runs read the user's `config.toml` (cwd home or the linked one, H:1412). **Probe: config `auto` → `Bash touch made.txt` ran, no pause, file made: governance bypassed** | none | – | broken |
| 68 | Config `default_plan_mode = true` (K:172463-172467) | New sessions enter plan mode | not overridden. Probe: Write refused "Plan mode is active…"; ACP said `default`; the harness then auto-approves the plan (row 47) | – | – | broken |
| 69 | `[permission]` rules allow/ask/deny in config.toml (K:272858-272915) | 2.1.1 parses them but **never loads them** (`addRules` has no caller; rules state starts empty, K:273660-273714) | passed through. Kimi still asks, so the harness governs | G | kimi::test_kimi_still_asks_about_a_command_its_config_allows_and_so_a_person_is_asked | works (as Kimi 2.1.1 does). In 2.6.0 they apply, so that test fails and team allow rules skip the harness |
| 70 | Dangerous-command / sensitive-file / git-control asks; `dangerous_command_guard`, `KIMI_CODE_DANGEROUS_COMMAND_GUARD` (K:272918, K:273404, K:273319) | Extra asks | asks are governed | G | – | untested |
| 71 | Replay refusal across attempts (bridge.py:235-243, bridge.py:333-335) | – | a Kimi call (Bash/Write/Edit that asks) identical to one that ran *before a pause* is answered "not run again: this same call already ran" and the turn is cancelled. Re-running the same test command after a resume is refused | – | none (`RAN_BEFORE` has no test anywhere) | caveat (untested behaviour change) |

## 5. Sub-agent, background and session settings

| # | Feature | What it does | H | Test | Status |
|---|---|---|---|---|---|
| 72 | Built-in profiles `coder`/`explore`/`plan`; custom agent dirs `$KIMI_CODE_HOME/agents`, `~/.agents/agents`, project `.kimi-code/agents`, `.agents/agents`, `extraAgentDirs` (K:122370-122386) | Sub-agent types | passed through (unsandboxed). The run home drops `$KIMI_CODE_HOME/agents` (only `config.toml`, `credentials`, `sessions` are linked, H:1412) | – | caveat |
| 73 | `--agent`, `--agent-file` (main agent profile) | Custom main agent | CLI only; `kimi acp` takes neither | – | not reachable |
| 74 | `[subagent] timeoutMs` / `KIMI_SUBAGENT_TIMEOUT_MS` (2 h), `[swarm] timeoutMs` / `KIMI_CODE_SWARM_TIMEOUT_MS`, `[secondary_model]` pool / `KIMI_SECONDARY_MODEL` / `KIMI_SECONDARY_EFFORT`, `KIMI_CODE_SUBAGENT_SCOPE_CACHE_SIZE` / `_EVICT_TIMEOUT_MS` | Sub-agent limits and models | env/config passthrough | – | untested |
| 75 | `[task]` / `KIMI_CODE_BACKGROUND_*` (max_running_tasks, keep_alive_on_exit, bash_task_timeout_s, print_*) | Background task limits | passthrough; meaningless once the process is killed | – | untested |
| 76 | Steering a running turn | TUI steer (improved in 2.5.3) | ACP has none. `steers()` is False (H:516-520); steer waits as queue | matrix X02 | blocked-by-design |

## 6. Skills, instructions, plugins, hooks

| # | Feature | What it does | H | Test | Status |
|---|---|---|---|---|---|
| 77 | Skill discovery: `$KIMI_CODE_HOME/skills`, `~/.agents/skills`, project `.kimi-code/skills`, `.agents/skills`, `extraSkillDirs`, `mergeAllAvailableSkills`, plugins (K:122761-122847, K:123940-123985) | Skill catalog for the `Skill` tool and `/skill:` | unsandboxed: passed through. **With `skills=`**: `KIMI_CODE_HOME` becomes the run home and its `skills/` is replaced by the pinned set (H:1449-1456), so the user's Kimi skills disappear. Sandboxed: HOME is the run home and cwd is empty, so user and project skills are gone (by design) | kimi::test_kimi_reads_the_runs_skills, acp::test_the_runs_skills_are_in_kimis_skills_folder_and_its_login_is_linked | caveat |
| 78 | Run's pinned skills (harness feature) | – | written to `<runhome>/skills/<name>/SKILL.md` (H:1449-1456) | same, plus kimi::test_a_conversations_next_turn_continues_kimis_session_when_its_runs_have_skills | works |
| 79 | Built-in skills `check-kimi-code-docs`, `custom-theme`, `import-from-cc-codex`, `mcp-config`, `sub-skill` (consolidate/review), `update-config`, `write-goal` (K:123381-123660); `KIMI_CODE_BUILTIN_PRODUCT_SKILLS` | Product skills | passed through (the model may load them; `update-config` can edit the linked user `config.toml` with Write, which asks) | – | untested |
| 80 | `--skills-dir` | Replaces discovery | CLI only | – | not reachable |
| 81 | AGENTS.md: `$KIMI_CODE_HOME/AGENTS.md`, `~/.agents/AGENTS.md`, project chain `.kimi-code/AGENTS.md` / `AGENTS.md` / `agents.md` from git root to cwd, hot reload, 32 KiB warning (K:119723-119790) | Memory/instructions | unsandboxed: passed through. Run home (skills=): global Kimi AGENTS.md dropped (not linked). Sandboxed: cwd is the empty run dir, so project AGENTS.md is gone | – | caveat (untested) |
| 82 | Plugins (`$KIMI_CODE_HOME/plugins`, `KIMI_PLUGIN_ROOT`, marketplace `KIMI_CODE_PLUGIN_MARKETPLACE_URL`) | Skills, agents, MCP, commands | passed through unsandboxed; dropped in the run home | – | untested |
| 83 | Kimi hooks `[[hooks]]` (20 events: PreToolUse, PostToolUse, PostToolUseFailure, PermissionRequest, PermissionResult, UserPromptSubmit, UserPromptQueued, TurnStarted, Stop, StopFailure, Interrupt, SessionStart/End/Heartbeat, SubagentStart/Stop, TaskStarted, Pre/PostCompact, Notification; K:198269-198316) | Local commands at lifecycle points | they run inside kimi; the harness neither sees nor records them; `config.toml` is linked into the run home; none in a sandbox. A PreToolUse hook can veto a call the harness approved, and a UserPromptSubmit block ends the turn as `refusal`, which fails the run | – | untested |

## 7. Context, usage, loop control, models

| # | Feature | What it does | H | Test | Status |
|---|---|---|---|---|---|
| 84 | Auto-compaction (`[loop_control] compactionTriggerRatio`, `reservedContextSize`, `compactionMaxAttempts`, `KIMI_TOKEN_COUNTING_STRATEGY`) | Summarises the context | Kimi runs it. **Its notices ("Compacting conversation context…", "Compaction completed. - Messages compacted…") are sent as `agent_message_chunk`** (K:311847-311870), so they land in the harness answer, stream and memory when they follow the last call (H:566-567) | – (my probe could not trigger it) | caveat (untested) |
| 85 | `/compact` | Manual compaction | not reachable (row 15) | – | not reachable |
| 86 | Usage log `agents/<id>/wire.jsonl` `usage.record` and `turn.ended` | Kimi's internal durable log | read as the budget source and for step-limit / repeat-breaker detection (H:297-344, H:1068-1122) | kimi::test_kimis_usage_log_is_the_contract_the_budget_reads, kimi::test_the_usage_kimi_logs_last_is_counted, acp::test_usage_is_read_from_the_agents_log_and_the_budget_stops_the_turn | works (relies on an internal file format, pinned by tests; tokens only, no dollars) |
| 87 | `KIMI_LOOP_MAX_STEPS_PER_TURN` / `[loop_control] maxStepsPerTurn` | Step limit | asks once more (H:940-953) | kimi::test_at_its_step_limit_kimi_is_asked_once_for_a_last_answer | works |
| 88 | `KIMI_CODE_REPEAT_BREAKER` | Ends a turn after 12 identical calls | asks once more (H:958-962) | kimi::test_a_loop_kimis_repeat_breaker_ends_is_asked_once_for_an_answer | works |
| 89 | `KIMI_LOOP_MAX_RETRIES_PER_STEP`, `_MAX_ATTEMPTS_PER_STEP`, `KIMI_CODE_INFINITE_RETRY`, `maxRalphIterations` | Retries | passthrough. Final failures look like no answer (row 32) | – | untested |
| 90 | Model env `KIMI_MODEL_{PROVIDER_TYPE,BASE_URL,API_KEY,NAME,MAX_CONTEXT_SIZE,MAX_OUTPUT_SIZE,MAX_TOKENS,MAX_COMPLETION_TOKENS,CAPABILITIES,TEMPERATURE,TOP_P,REASONING_KEY,ADAPTIVE_THINKING,THINKING_EFFORT,THINKING_KEEP,OUTPUT_FORMAT,DISPLAY_NAME}`, `KIMI_API_KEY`, `KIMI_BASE_URL`, `[providers]`, `[models]`, `default_model`, `[thinking]`, `[model_catalog]` | Model and provider | passthrough. With Bifrost and `model=`, the harness sets the first four (H:271-275), and the agent's own `KIMI_MODEL_BASE_URL` wins | kimi::test_kimi_through_bifrost_carries_the_runs_headers | works |
| 91 | `KIMI_CODE_CUSTOM_HEADERS` | Extra headers | merged with gateway headers (H:281-284) | same | works |
| 92 | Thinking level (config option / env) | – | env only | – | untested |
| 93 | Image settings `KIMI_IMAGE_MAX_EDGE_PX`, `KIMI_IMAGE_READ_BYTE_BUDGET`, `[image]`; `[read] max_chars` | Media/read limits | passthrough | – | untested |
| 94 | Web settings `KIMI_WEB_SEARCH_*`, `KIMI_WEB_FETCH_*`, `[services]` | Search/fetch backends | passthrough | – | untested |

## 8. Process, environment, CLI-only features

| # | Feature | What it does | H | Test | Status |
|---|---|---|---|---|---|
| 95 | `KIMI_CODE_HOME` (sessions, config, credentials, skills, AGENTS.md, mcp.json, plugins, bin/rg, logs) | Data root | user's home normally. Run home with `skills=`/`sandbox=` (H:1428-1447) | acp::test_the_runs_skills_are_in_kimis_skills_folder_and_its_login_is_linked | caveat (rows 72, 77, 81, 82). Also, with no `rg` on PATH, Kimi downloads rg from its CDN into each fresh run home (K:181036-181080) |
| 96 | Sandboxed environment | – | only `PATH, LANG, LC_ALL, TZ` + target env, `HOME` = run home (H:1381, H:1431-1434) | acp::test_a_sandboxed_run_works_in_the_sandbox… | works (by design) |
| 97 | Telemetry (`KIMI_DISABLE_TELEMETRY`), auto-update (`KIMI_CODE_NO_AUTO_UPDATE`, `KIMI_CLI_NO_AUTO_UPDATE`) | Kimi phones home | not set by the harness (only the tests set them, tests/support/kimi.py:354-355): production runs send Kimi telemetry | – | caveat |
| 98 | Logging `KIMI_LOG_*`, stderr | Diagnostics | stderr tail kept for crash errors (H:127, H:861-864, H:874-879) | acp::test_a_process_that_ends_mid_turn_fails_the_run_with_its_stderr | works |
| 99 | Process lifetime | One process per attempt | per attempt; killed at its end (H:717-736). bgkimi keeps it while background work runs | acp::test_a_process_already_gone_when_the_attempt_stops_it_keeps_the_answer | works (final), but rows 36, 52, 56 |
| 100 | Kimi version | – | `>=2.1.1,<2.2` enforced | kimi::test_kimi_is_the_pinned_version | works |
| 101 | CLI-only surfaces: `-p` / `--output-format stream-json`, `-y/--yolo`, `--auto`, `--plan`, `-m`, `-S/-c`, `--add-dir`, `--skills-dir`, `--agent(-file)`; subcommands `export, fork, provider, session, web, server, rc/remote, login, doctor, vis, install-desktop, migrate, upgrade`; ~60 TUI slash commands (`/goal /plan /swarm /tower /yolo /auto /approvals /permission /model /effort /mcp /init /undo /fork /btw /add-dir /tasks /usage …`) | – | the harness drives `kimi acp` only | – | not reachable (by design) |
| 102 | Experimental master switch `KIMI_CODE_EXPERIMENTAL_FLAG` and `[experimental]` | Turns all flags on (tower, fork, notify, tool-select) | passthrough. One env var turns on tower (row 58) and tool-select (row 59) | – | caveat |

---

## Counts by status (102 rows)

| works | caveat | broken | blocked-by-design | untested | not reachable |
|---|---|---|---|---|---|
| 31 | 25 | 8 | 5 | 19 | 14 |

Counted from each row's status cell (its first word). "works (transform)" counts as works; "caveat (untested)" counts as caveat; "not reachable (final)" counts as not reachable. The broken rows are 23, 36, 47, 50, 52, 56, 67 and 68.

---

## Gaps, P0 to P3

### P0: governance can be bypassed
1. **Kimi's own `default_permission_mode` turns off governance.** Evidence: K:224295-224299 applies `default_permission_mode` to every new agent. ACP still reports mode `default` (K:311312, K:311967). The harness sends `session/set_mode` only when `mode=` is given (H:913). In the probe, a `config.toml` with `default_permission_mode = "auto"` let `Bash touch made.txt` run without a pause, and the file was made. Unsandboxed runs read the user's config (or a linked copy in the run home, H:1412). Fix: always `set_session_mode` to `default` or the `mode` given, on new *and* resumed sessions, and check the response's `modes.currentModeId`. Also confirm with a `config_option_update` read.

### P1
2. **Sub-agent permission requests are governed without their arguments.** Kimi's request has no rawInput, and its `command` display is dropped (K:310599-310618, K:309738-309762). The harness falls back to `{"request": "Requesting approval to Running: <first 50 chars>"}` (H:1274-1275, H:1633-1645). Probe evidence. The fake agent's test hides this: it sends a made-up JSON text and asserts only the tool name (acp::test_a_sub_agent_is_a_child_run_and_its_calls_are_governed). As a result, shell read-only checks, catalog `approve_when` rules and reviewers see no command. The wire log `interaction.request` carries the full payload (bgkimi already reads that log).
3. **Plan review is auto-approved, and the first alternative is chosen.** `ExitPlanMode` has risk `read` (H:183-184), so `_permitted` asks no person and `_chosen(…,"allow_once")` picks `plan_opt_0` (H:1610-1618). Probe: "Selected approach: Fast … Approved Plan", no interrupt. This overrides Kimi's rule that every plan needs user review (tools.md; changelog 2.4.0).
4. **`default_plan_mode = true` silently puts the session in plan mode** (K:172463-172467). The probe confirmed it. Together with gap 3, plans are then approved by nobody.
5. **Writes in a git worktree are never governed.** Kimi's `git-cwd-write-approve` (K:273359-273402) auto-approves `Write`/`Edit` there (probe: Write ran, no pause). kimi.md says so, but `docs/matrix.md` F32 claims "write asks" for Kimi, and every test runs outside git.
6. **Background work dies (final branch).** This covers `Bash/Agent(run_in_background)`, auto-backgrounded Bash and `AskUserQuestion(background)`. The process is killed at the end of the attempt, and a background `Agent` child run ends SUCCESS with the launch text (H:1191-1195). bgkimi fixes this but is not merged.
7. **Goal mode does not work.** Continuation turns (K:209724-209746) are not forwarded over ACP (K:311741), and the process ends, so the goal is paused.
8. **Tower is not forced off on the final branch.** `KIMI_CODE_EXPERIMENTAL_TOWER=1` or `KIMI_CODE_EXPERIMENTAL_FLAG=1` in env or config exposes `Tower*` tools, whose workers run in `auto` mode with no permission requests. bgkimi sets `TOWER=0`.
9. **Questions lose data.** Only the first question is asked; header, option descriptions, multi_select and "Other" are lost (K:310932-310980; probe). Fix: advertise `ClientCapabilities.elicitation.form` (ACP 0.12.1 has the types and `create_elicitation`) and map the form to `Runtime.ask`. No real-Kimi test covers questions.

### P2
10. **Sandboxed Bash results are recorded empty.** The terminal-backed `tool_call_update` has no rawOutput (K:311795-311810), and `_output` reads only text (H:1680-1686). Probe: `""` vs `"boxed-output\n"`.
11. **The tool name comes from the ACP `title`.** That title is the execution description ("Running: …", "Launching coder agent: …") unless a streamed delta created the call first (K:310189, K:310249). A non-streaming provider gives wrong names, Agent/AgentSwarm are not seen as helpers, and governance looks up the wrong risk. Use the permission title (which is the tool name) or `kind`, or read the wire log.
12. **Failed turns lose their reason.** Kimi maps `failed` to `end_turn` (K:310100-310108). The harness only reads the step-limit code (H:336-343), so provider or context errors become "no answer", or SUCCESS when partial text came first. `MAX_TOKENS` and `max_turn_requests` handling are dead code for Kimi (H:137-141, H:1018).
13. **Compaction notices enter the answer, stream and memory** (K:311847-311870). Untested.
14. **AgentSwarm parsing.** The fixed-order regex (H:156-160) misreads `mode`, `state` and `stop_reason` attributes and escaped items. `resume_agent_ids` gets no child run. `Agent(resume)` is named "coder" (H:1176). bgkimi fixes this.
15. **The run home (`skills=`) drops parts of the user's Kimi setup.** Only `config.toml`, `credentials` and `sessions` are linked (H:1412), so these are gone: user skills (replaced), global `AGENTS.md`, `mcp.json`, `agents/`, `plugins/`, and the `credentials/mcp` OAuth tokens (only if credentials is absent in the user's home). That differs from Kimi's behaviour for the same user.
16. **Replay refusal of identical calls after a pause** (`RAN_BEFORE`, bridge.py:235-243) is untested and changes behaviour. For example, re-running the same `pytest` command after an approval is refused.
17. **The ACP slash-command interception is inconsistent** (row 15). Only sandboxed runs with no memory context let Kimi intercept "/…".

### P3
18. These updates are dropped: `plan` (todos), `agent_thought_chunk`, `usage_update`, `session_info_update`, `available_commands_update`, `config_option_update`, `current_mode_update`, and the diffs and locations in tool calls.
19. MCP OAuth (`mcp__srv__authenticate`): the URL only appears in a progress title, which is ignored.
20. Production runs send Kimi telemetry and leave auto-update on. The harness sets neither `KIMI_DISABLE_TELEMETRY` nor `KIMI_CODE_NO_AUTO_UPDATE`; only the tests do.
21. Sessions are never closed or deleted, so unsandboxed sessions build up in `~/.kimi-code/sessions`.
22. Run homes re-download `rg` when it is not on PATH (network egress from a sandboxed run).
23. Not reachable at all: image or resource prompt blocks, `additionalDirectories`, `set_config_option` (thinking), `session/list/close/delete/fork` (final), and `logout`/`authenticate`.
24. Untested tools: Glob, Grep, ReadMediaFile, FetchURL, WebSearch, TodoList, Skill (call), EnterPlanMode, TaskList/Output/Stop, WaitFor, Kimi hooks, AGENTS.md.

---

## Where the harness changes Kimi's behaviour (the rule is that library behaviour must not change)

| Change | Where | Disclosed? |
|---|---|---|
| Plans approved with no person, and the first alternative auto-selected | H:183-184, H:1610-1618 | no |
| Cron forced off (`KIMI_DISABLE_CRON=1`) whatever the env says | H:206-210, H:669-670 | yes (kimi.md) |
| Tower forced off (`KIMI_CODE_EXPERIMENTAL_TOWER=0`) | bgkimi only | yes (bgkimi kimi.md) |
| Swarm concurrency capped at the harness admission limit (4 to 16) instead of Kimi's unlimited plus adaptive default | H:667-668 | yes |
| Rejections and edits cancel Kimi's turn and start a new prompt (an active goal pauses) | H:1290-1300 | yes |
| A pause cancels the turn and kills the process; the resume prompt tells Kimi to re-call | H:1286-1289, H:979-989 | yes |
| Every prompt is prefixed (`DELIVERS_HERE`, `<memory_context>`), which also disables Kimi's ACP slash commands (except sandboxed runs with no context) | H:965-977 | partly (the prefix yes, the slash effect no) |
| `session/set_mode` is sent only when `mode=` is given, so the user's config default (auto/yolo/plan) silently wins against the harness's assumption | H:913 | no |
| `allow_always` is never offered back, so Kimi's session approval memory is unused | H:1610-1618 | no |
| A call identical to one that ran before a pause is refused | bridge.py:235-243 | no |
| The run home hides the user's Kimi skills, AGENTS.md, mcp.json, agents and plugins | H:1412, H:1449-1456 | partly |
| `elicitation.form` is not advertised, so Kimi drops questions 2-4 and multi-select | H:677 | no |
| Background tasks and goal continuations are killed with the process (final branch) | H:678-709 | bgkimi documents it |

## Newer releases (2.2.0 → 2.6.2) that matter

The harness pins `>=2.1.1,<2.2`. Latest stable is 2.6.2 (2026-11-13); there are canaries for 2.7.0 and 3.0.0.

- **2.3.2: `AgentSwarm` only runs in swarm mode**, which is TUI-only (`/swarm`). The changelog says swarms "are no longer available in `kimi -p` or ACP clients such as Zed". The harness's swarm child-run mapping and the matrix's swarm claims stop applying. 2.3.2 also stops the agent delegating to sub-agents on its own.
- **2.4.0: plan mode can only be entered by the user, and "every plan requires user review"; Never Ask no longer auto-approves plans.** Gap 3 makes the harness contradict this.
- **2.6.0: `[[permission.rules]]` from `config.toml` (and trusted `.kimi-code/local.toml`) are applied** when a session starts. kimi::test_kimi_still_asks_about_a_command_its_config_allows… then fails, and team allow rules skip harness governance.
- 2.3.1: a cross-process session lock (`KIMI_CODE_SESSION_LOCK`), with takeover after a retry. This matters for resumes and for bgkimi's kept processes and forks.
- 2.5.0: sub-agent model pool semantics changed (no implicit `primary`); `[workspace] require_trust`; remote SSH/Docker environments.
- 2.4.1: Read is capped at 100k chars by default (`[read] max_chars`).
- 2.2.0: the `NotifyUser` experiment.
- 2.6.0: Tower artifact missions.
- acp-server 0.0.1-0.0.3: dependency bumps only. The technical changelog has one ACP fix: "Fix ACP initialization failing when clients send the first request during startup" (#4541).
- 2.4.0: workspace MCP reconnect with backoff (`[mcp] reconnect_*`).
- 2.6.1: long sessions no longer stall with empty replies instead of compacting.
