# Claude Agent SDK

The target is the `ClaudeAgentOptions` your team already configured. The harness runs the SDK's
`query(prompt, options)` on a copy of the options that also carries the harness's tools (the
gateway's MCP tools, memory's); your options object is never changed.

**Install:** `pip install 'trellis-harness[claude-agent-sdk]'`, and the Claude Code CLI the SDK
drives. Through Bifrost: `env={"ANTHROPIC_BASE_URL": "<gateway>/anthropic",
"ANTHROPIC_API_KEY": "<virtual key>"}` is set on the run's copy of the options by the harness
when Bifrost is the route (`LLM_GATEWAY=bifrost`, its `Capability.ANTHROPIC_ROUTE`; unless your `env` names a base URL), with Bifrost's headers — the
deny-all MCP scope, whose run it is — as `ANTHROPIC_CUSTOM_HEADERS` lines.

This page is Way 1: the harness runs `query()`. To keep calling `query()` yourself and plug in
the blocks (memory, `can_use_tool` from governance, the session as the run's checkpoint in
agent-runs, a judge), see the Way 2 recipe:
[blocks/claude-agent-sdk.md](../blocks/claude-agent-sdk.md).

## Using an existing Claude Agent SDK project

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from mcp.types import ToolAnnotations
from trellis import Harness

h = Harness()


@tool(
    "close_ticket",
    "Close a support ticket.",
    {"ticket": str},
    annotations=ToolAnnotations(destructiveHint=True),
)  # irreversible: a person approves it
async def close_ticket(args: dict) -> dict: ...


options = ClaudeAgentOptions(
    system_prompt="You triage support tickets.",
    allowed_tools=["Read"],
    mcp_servers={"desk": create_sdk_mcp_server("desk", tools=[close_ticket])},
)
agent = h.wrap(options, id="triage")  # was: query(prompt, options)

result = await agent.run("Ticket T-9 is resolved; close it.", user="ada")
if result.interrupt:
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
```

Your tools stay yours: your in-process servers (`create_sdk_mcp_server`), Claude Code's
built-ins, your MCP servers — each call governed as the CLI makes it (below), its risk the
server's annotations (`destructiveHint`: irreversible, `readOnlyHint`: read) unless the tool
catalog says otherwise. The harness's tools — the MCP tools the Bifrost virtual key allows and,
memory on, the memory tools — reach Claude as one more in-process MCP server named `trellis`
(tool names `mcp__trellis__<tool>`), added beside your own `mcp_servers` (a dict, a config file
path or its JSON text); the harness's permission callback lets them through: the bridge is
their check. That server is loaded from the first step (its config says `alwaysLoad`): Claude
Code otherwise defers every MCP server's tools behind its `ToolSearch`, so the model sees their
names only and spends a step loading them. Which tools a run offers is already the harness's
choice (the tool hints' narrowing, memory's `tool_search`); your own servers keep the CLI's
deferral.

**`query(...)` itself is not intercepted**: call `agent.run`/`stream`/`resume`/`start` instead.

## Claude Code's built-in tools

`Read`, `Write`, `Edit`, `Bash`, `WebFetch` and the rest are the CLI's own: it runs them, and
asks before them as it always does. The harness adds your governance on top and stands in for
the user Claude Code would ask:

* **Your rules first.** The SDK's own `PreToolUse` hook, which the CLI runs before every tool
  call in every permission mode (the options it runs carry the harness's hook after any of
  yours), decides each built-in call as a harness tool call is: the run's `before_tool` hooks
  ([hooks.md](../hooks.md)), then governance by the tool's risk and the tool catalog's rules.
  A call they refuse, or ask a person about, stops there.
* **Then Claude Code's own.** A call they let through goes to Claude Code's own permission
  check: its `permission_mode`, its rules (`allowed_tools`, `disallowed_tools`, its settings
  files: your allowlist, passed through as it is) and its own read-only checks. What it lets
  run, runs; nothing more is asked.
* **When Claude Code would ask** its user, it asks the harness (its `can_use_tool`): a call
  that only reads runs — a read tool, or a `Bash` command the harness reads as read-only (`ls`,
  `cat`, `wc`, `which`, Python that opens files to read: `governance/shell.py`, read with
  `shlex` and `ast`); any other asks a person. Approved (or edited), it runs once; rejected or
  denied, Claude reads why. A read-only command reads only what the run may read: the CLI's
  `cwd`, the folders `add_dirs` adds (Claude Code reads and works there itself: name a folder
  of yours there, such as the tool results the CLI saves), the run's skills, and where a kept
  result is read. Claude Code's shell keeps the folder a `cd` left it in, so after a `cd` a
  relative read is placed from every folder the shell may be in; where that cannot be followed
  (`cd $DIR`, `pushd`, `eval`), every read of a file asks until the next query. The
  environment the CLI is given (`env`: its `PATH`, `PYTHONPATH`) is the one read.
* **Claude's questions** (`AskUserQuestion`) are the run's questions, asked as `ask_user` asks
  them: each a choice (several picks with `multiSelect`, the option labelled "(Recommended)"
  recommended with its description as why, another answer in words); Claude reads the answers
  by question.
* **Risk.** `Read`, `Glob`, `Grep`, `LS`, `WebFetch`, `WebSearch`, `TodoWrite` read (they run);
  `Write`, `Edit`, `MultiEdit`, `NotebookEdit` and `Bash` write (announced; Claude Code asks
  first where it would). The tools Claude keeps its own work with run too: its task list
  (`TaskCreate`, `TaskGet`, `TaskList`, `TaskUpdate`: `TodoWrite`'s successors in the bundled
  CLI), `EnterPlanMode`, and `StructuredOutput`, which its answer in an `output_format` is
  handed to. Your in-process server's tool is what its annotations say; any other tool (an
  MCP server's of yours the CLI reaches itself, a tool of the CLI's the harness has no word
  on) says nothing of what it does: a person approves each call, until its risk is in the
  tool catalog.
  The tool catalog's `risk` and `approve_when` override these, as for any tool
  ([governance.md](../governance.md)): a team that wants every shell command approved, even
  one that only reads, gives `Bash` the risk `irreversible` (or an `approve_when` such as
  `true`) there.
* **Not offered: schedules.** A scheduled prompt (`CronCreate`) or a wake-up
  (`ScheduleWakeup`) fires only in a CLI that is running then. A query's own process ends
  with it; a conversation's process (below) outlives the turn, but only until it is idle,
  and the turn a schedule fires has no run: its tool calls are refused and its answer reaches
  nobody. So these, with `CronList` and `CronDelete`, are added to the query's
  `disallowed_tools`, after your own. Offered, Claude promised a weekly report that nothing
  would send.
* **One call, one question.** A call a person answered about before it ran (governance's
  question, a hook's `Ask`) is not asked about again when Claude Code asks too: it runs.
* **Your own `can_use_tool`** answers what Claude Code asks, as without the harness: it is
  asked when Claude Code would ask its user, after your hooks and governance, with the
  arguments as they were decided (a call a hook rewrote is asked about as rewritten) and
  what Claude Code gives with them (the call's id, the subagent's, its suggestions); what
  it answers stands, its `updated_input`, `updated_permissions` and `interrupt` included.
* **Your own hooks**, for every event the SDK has (`PreToolUse`, `PostToolUse`,
  `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStart`, `SubagentStop`,
  `PreCompact`, `Notification`, `PermissionRequest`), are registered and called as without
  the harness, before the harness's own, inside the run (`trellis.current()`). A `permission_prompt_tool_name` of your own is refused: give your check as
  `can_use_tool`.
* **Recorded, not journaled.** A built-in call that ran is on the run's stream and in its
  records, with its result or its error; it is not in the journal (the CLI keeps its result
  in its session, below).

## Where sandbox, skills and hooks go

| | Where |
|---|---|
| Sandbox | its own: `ClaudeAgentOptions(sandbox=SandboxSettings(enabled=True, ...))` ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)) |
| Skills | its own: `ClaudeAgentOptions(**src.claude())` — a local plugin the harness writes with the run's pinned skills before each query (removed after it), `skills="all"`; `ClaudeAgentOptions(skills=[...])` still filters ([skills.md](../skills.md)) |
| Hooks | `h.wrap(hooks=[...])`: the run and tool hooks, a built-in tool's in the permission callback ([hooks.md](../hooks.md)); a model call is the CLI's |
| Limits | `budget=Budget(...)` (`h.wrap` or `run`) is the one cross-engine cap: tokens, dollars and harness tool calls over the whole run, sub-agents included ([api.md](../api.md#budget)) (each query's tokens and `total_cost_usd`); its native turn limit is `framework_options={"max_turns": n}`: at it the run answers once more (its session resumed, told the limit is reached, one turn and no tool), as `ReAct` does at `max_steps` (a `step_limit` warning) |

## Native or ours: skills, prompts, sandbox

* **Skills:** for `SKILL.md` folders the CLI discovers, use Claude's own:
  `ClaudeAgentOptions(skills=["name", ...] | "all")` (the SDK then allows the `Skill` tool and,
  `setting_sources` unset, loads the user's and the project's settings)
  ([skills.md](../skills.md#native-or-ours)).
* **Sandbox:** prefer Claude's own `Bash` and file tools, confined with
  `ClaudeAgentOptions(sandbox=SandboxSettings(enabled=True, ...))`; the harness answers each
  call the CLI asks about (above), and a sandboxed `Bash` call is asked about only with
  `autoAllowBashIfSandboxed` false ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).
* **Prompts:** the options take text: `system_prompt=(await h.prompt("triage")).text` for a
  stored prompt of the gateway ([prompts.md](../prompts.md#native-or-ours)).

## What is automatic

| | |
|---|---|
| Memory push | the context goes with every prompt through Claude Code's own `UserPromptSubmit` hook (`additionalContext`), after the options' own hooks; the options' `system_prompt` is left as you wrote it |
| Long-term memory, session | the session is kept in the run's journal as the SDK's `SessionStore`, flushed at every step, so a resume on any worker continues it; options with their own `session_store` or `enable_file_checkpointing` keep the session on their machine (said once) |
| Spending | each result's usage (the main agent's tokens), its subagents' replies' usage, and what its `total_cost_usd` grew by count toward the run's budget: a result's cost is its session's running total (the subagents' included, and what the session cost before the process that resumed it), so a turn, a resumed attempt and the one last answer at the turn limit are each charged once |
| Subagents | each `agents=` subagent Claude runs is a child run (its record names the run, its turns go to memory); its tool calls go through the same `PreToolUse` governance |
| Memory pull | the memory tools are on the `trellis` server |
| Records | the transcript (the question and the assistant's text blocks), every harness tool call, the `system` outcome; approvals as `TOOL_CALL` feedback |
| Tool hints | from 30 tools of the run's own (Claude Code's built-ins are not counted), the `trellis` server lists the hinted tools (and the memory tools, and those already used) for the run: the CLI lists an MCP server's tools once per query |
| Large results | a harness tool's result over 80,000 characters is kept in the run, the model reads a head-and-tail preview naming `/large_tool_results/<id>` and pages it with the `trellis` server's `read_file` (listed from the start: the CLI lists a server's tools once) ([tools.md](../tools.md#large-results)); the CLI's own MCP output limit (`MAX_MCP_OUTPUT_TOKENS`, 25,000 tokens by default, about 50,000 characters of a CSV) is raised to 60,000 so the harness keeps such a result first: a result the CLI saves to its session's own file is one `read_file` and `query_result` cannot read (your own value, in the options' `env` or the environment, is kept) |
| Grounding, judges, tracing | as for every target |

The answer is the `ResultMessage`'s `structured_output` (an `output_format`) or its `result`; a
`ResultMessage` with `is_error` fails the run (`ClaudeRunError`).

## Approvals and pauses

A harness tool that asks (`irreversible`, a catalog `approve_when`, a hook's `Ask`),
`trellis.current().ask` inside one, or a built-in tool that asks, pauses the run: the call is
answered "Waiting for a person's approval.", the harness stops consuming the query and the CLI
process ends (a conversation's process stays, its turn ended, and the resumed attempt is
sent into it: below). The session the CLI kept (`ResultMessage.session_id`) is in the run's journal.
`agent.resume(...)` runs the next attempt in that session (`ClaudeAgentOptions(resume=...)`),
told to call the tool it was calling again: Claude goes on from where it was — the built-in
tools it ran are not run again, the steps before are not asked again — and the journal answers
the call (the approved call runs once). Every decision works: `approve`, `reject` with a reason
(the tool result Claude reads: "… was not run: the approver rejected it (why)"), `edit`,
`answer`, `cancel`.

**On failure.** The CLI keeps its sessions on its machine (or in your `session_store`). A run
resumed where the CLI does not hold the session (another worker's machine) gets a `warning`
event (`claude_session`) and runs the query again from its prompt, against the journal — the
harness calls already made are not made again; the built-ins run again.

## The run's own options

`framework_options=` sets `ClaudeAgentOptions` fields on the run's copy of the options (on
`h.wrap` for every run, on `run`/`stream`/`start` for one, over the agent's): `max_turns`,
`model`, `permission_mode`, `cwd`... before the harness's own changes, which merge with them —
`hooks` get the governance and context hooks after theirs, a `can_use_tool` is asked after governance
as the target's would be, and `mcp_servers` get the harness's `trellis` server beside them (one
of that name is refused). The session is the harness's (`resume`, `continue_conversation`) and so
is the permission prompt (`permission_prompt_tool_name`): both refused, as is a name that is not
a field ([configuration.md](../configuration.md#the-frameworks-own-run-options)).

```python
await agent.run(
    "Tidy the repo", user="ada", framework_options={"max_turns": 30, "cwd": "/srv/repo"}
)
```

## Streaming

`agent.stream(...)` yields each assistant text block as a text delta, the harness tool calls as
`TOOL_CALL_*` events, then `RUN_FINISHED`.

## Sub-agents and child runs

Each `agents=` subagent Claude runs (its `Agent` tool) is a child run of the run: its record
names the run (`h.runs.iterate(parent_run_id=...)`), its prompt carries the memory context for
its task (the `PreToolUse` hook's update), its built-in and MCP tool calls are governed and
recorded under its own run (by the subagent the CLI names; the harness's own tools it calls are
recorded under the run), its turns go to memory, its spending is in the query's
`ResultMessage`, and a subagent cut by a crash is ended by the worker that continues the run
([subagents.md](../subagents.md#automatic)). Two subagents of one type at work at once each
have their own calls (Claude Code names the subagent by its task's id, `agent_id`). A
subagent's text, which Claude Code forwards naming its `Agent` call (`forward_subagent_text`;
a background subagent's always), is its own run's: not on the run's text stream.

**In the background.** Claude Code runs a subagent in the background unless its call says
`run_in_background: false` (the bundled CLI): the `Agent` call returns at once, and the
subagent's end comes as its task's (`task_notification`). Its child run stays open until
then — `SUCCESS` with what it said, `ERROR` when its task failed or was stopped. In a run
with no conversation the SDK keeps the query's process while such a task works, so the run
ends when the task has (and the session ran its turn for it): the run's answer is its own
turn's result, and a subagent the process ended before (it died, or exited) is an `ERROR`
saying it was cut off, never a success. A subagent that works in the background of a
conversation's process keeps its child run open across the turns
([below](#conversations)). A wrapped agent called from a tool of yours
(`await other.run(...)`) is a child run too, and the wrapped options are a `ReAct`'s sub-agent
like any agent (`subagents=[agent]`). Runnable:
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target ([runs.md](../runs.md), [surfaces.md](../surfaces.md),
[evaluation.md](../evaluation.md),
[examples/06_scenarios/runs_every_framework.py](../../examples/06_scenarios/runs_every_framework.py)):
the worker's machine needs the CLI; `llm_judge` asks the gateway's judge alias (or its own `model=`).

## Conversations

A conversation's turn leaves its CLI process running, and the conversation's next turn in
this process is sent into it (the SDK's `ClaudeSDKClient`), with the turn's own prompt. With
no such process (it closed, or the server restarted) the turn starts one that resumes the
session the last turn ran in (`resume=`, through the journal's session store:
[runs.md](../runs.md#a-claude-or-acp-session-across-turns)); in another process, or the first turn, a new session — with memory off given
the run store's earlier turns in its context.

**Background work.** What Claude starts in the background — a subagent with `background`, a
`Bash` command with `run_in_background` — therefore goes on between the turns: the turn that
started it ends at its own answer, the next ones are answered while it runs, and Claude's
session holds its notification when it ends. A turn ends at the result of its own prompt: a
turn the session runs by itself (a finished task's notification) is never a user's answer.

* **A background subagent keeps its child run.** The child run its `Agent` call started
  ([below](#sub-agents-and-child-runs)) stays `RUNNING` from there until the SDK says its
  task ended (the CLI says which subagents it runs in the background: `task_started` with
  `is_backgrounded`; the bundled CLI, 2.1.283, driven against a scripted model, did so
  unless the call said `run_in_background: false`). It ends `SUCCESS` with what the subagent said last, `ERROR` when the task
  failed, `CANCELLED` when it was stopped in Claude Code (Claude's own `TaskStop`); and
  `ERROR`, saying so, when the session ends first (it closed, its process died). While it
  goes on the session holds it in the run store (a heartbeat every 20 s holding it for 60,
  `claude_live.HOLD_SECONDS`, naming the server process): a server killed with the session
  (nothing closed: a crash, a restart) holds it no more, and agent-runs ends it `ERROR`
  (`holder_lost`, saying its holder stopped) within a tick of the hold lapsing — about a
  minute — instead of leaving it `RUNNING`. A run that waits for a person is not held (a
  pause lets go of the hold) and is taken again once it runs.
* **Its calls are its run's**, whether or not a turn is being answered: the CLI names the
  subagent in each hook and permission request (`agent_id`: the task's id), and the
  harness decides, runs and records the call for that run — your hooks, governance by
  risk and the tool catalog, then Claude Code's own check and `can_use_tool`; the
  `TOOL_CALL_*` events on its stream (`agent.events(child_run_id)`); a harness tool run
  for it.
* **A call that asks a person** pauses that run as any run pauses (its `awaiting`, its
  journal), and the call waits in Claude Code: the process and the task stay. Nothing
  answers a child run's own question today (`agent.resume` refuses a run that has a
  parent, and the chat's thread stream lists a thread's turns only), so it is asked in
  the conversation's **next turn**, before that turn's prompt is sent: the turn pauses
  with the question (`payload["subagent"]` names the run, with `background: true`), and
  its answer goes on to the subagent's run — approved or edited the call runs once,
  rejected the subagent reads why — then the turn's prompt is sent. A question nobody
  answers in 12 hours ends the run `CANCELLED` and stops the task (the CLI is told to
  wait 24 hours for a hook, so it never gives the call up first: a hook it gives up on
  fails the call, it does not run it).
* **Its limits** are the agent's own `run_timeout` and `budget`, as for a run nothing
  else bounds (the turn that started it has ended): its replies' tokens count as they
  come. Past either, and when the run is cancelled (`agent.cancel(child_run_id)`, the
  chat's cancel route), the task is stopped with the SDK's `stop_task` — that subagent
  only. Cancelling the turn that started it, while that turn still runs, cancels it too;
  cancelling a later turn leaves it.
* **Spending.** A result's cost is the process's running total (its usage is the turn's
  own). What it grew by since the last result is shared by the tokens each used: the turn
  its own, a background run what its subagent's replies used since — so a subagent's
  work between two turns is charged to its run, and the next turn its own. The turn the
  session runs for a finished task is charged to that task's run, which waits for it (30
  seconds at most) before it ends. The shares are exact when every reply costs the same
  per token, an estimate otherwise (another model, cached tokens).
* Any other call made between two turns has no run, so nothing can check it (your hooks,
  governance, an approval, the budget) or record it: the main agent's own, in the turn
  the session runs for a finished task, is refused, and the conversation's next run says
  how many were (`claude_session` warning). A background shell command needs no call to
  go on (its `Bash` call was decided when it was made) and has no run; what it and
  injected turns cost with no background run to charge is on the next turn.
* Not done: a background subagent's own subagents have no run of their own kept open; a
  child run's end is on the stream of the turn that started it only while that turn is still
  being answered (`helper_finished`), not delivered to an open chat after it (Claude's next
  answer has it). A subagent's child run has the deadline of the run that started it, so
  agent-runs ends one still unended by then. A foreground subagent's run is not held (it may
  wait long for its parent's person), so one a killed server left `RUNNING` is ended only by
  its deadline or the service's `RUNS__LIMITS__MAX_RUN_SECONDS`, as is a turn's own run; nor
  is a background subagent's run that a killed server left `PAUSED` on a question (a person
  may still answer it, and nothing runs it then).
* A turn that is left — its run paused, or cancelled — is stopped with the SDK's `interrupt`
  only when nothing runs in the background: Claude Code's interrupt stops the background work
  too. With work under way the run ends at once and the CLI's turn at its next tool call
  (refused with the hook's `continue: false`, which leaves the work). The next message waits
  for that turn's result, 30 seconds at most; past that the process is replaced.
* A message is sent into the process once no turn is under way in it: sent during one, Claude
  Code may hand it to that turn at its next tool call, and it gets no answer of its own.
* The process is one per conversation, in one server process, at most 16 per agent. It closes
  5 minutes after its last activity when nothing works in it, 6 hours after its last turn
  whatever does, when a turn's options are not the ones it was started with
  (`framework_options`, another set of harness tools), and with the harness
  (`Harness.aclose`). Background work that ends with it is said in the conversation's next
  run (`claude_session`), and a background subagent's run ends saying so; a restart loses
  it.
* Not kept: a run with no `thread`, a sub-agent's run, the last answer at the turn limit, and
  options with a `ClaudeSkills` plugin, an in-process MCP server of your own
  (`create_sdk_mcp_server`: its tools would run in the context of the process's first turn)
  or a budget of their own (`max_budget_usd`: Claude Code counts it for a process — in one
  that serves several turns the turn that passes it ends `error_max_budget_usd` and so does
  every turn after, and the SDK cannot give a running process its cap back — so each turn
  gets a process, and the cap you set for a query stays a query's).
  These run in a process of their own, as before.
* The model calls of a kept process carry the gateway headers (the run's dimensions, its
  `traceparent`) of the turn that started it: the CLI reads them once, as it starts.
 `mode="steer"` gives the message to Claude after
its next tool call (a `PostToolUse` hook's `additionalContext`); a query that calls no tool
takes none, and the message then runs as its own turn ([runs.md](../runs.md#a-message-to-a-busy-conversation-mode)).

## The session your options choose

`resume`, `fork_session` and `continue_conversation` on your options choose the session a
run's first query starts from, as without the harness; `session_id` names the one it starts
(the attempt that resumes it after a pause names it once, as the one to resume: Claude Code
takes both only to fork). The session `continue_conversation` continues is the CLI's latest in
its folder, on its machine: such options keep their session there, not in the run's journal
(said once), as options with `enable_file_checkpointing` do.

## Limits

* A built-in that ran is not journaled (above).
* `ClaudeSDKClient`'s own controls (`set_permission_mode`, `set_model`, `rewind_files`,
  `get_mcp_status`, `reconnect_mcp_server`, `toggle_mcp_server`, `get_context_usage`,
  `get_server_info`) are not reachable: the harness holds a conversation's client and uses
  its `query`, `interrupt` and `stop_task`. Set the model and the permission mode on the
  options, or per run with `framework_options`.
* A prompt that is a slash command runs as Claude Code runs it. One the CLI answers itself,
  with no reply of the model's in its turn (`/compact`), ends `SUCCESS` with an empty answer,
  as a run of any framework in which no model replied does; a turn whose model did reply,
  its last reply blank, ends `ERROR` (`NO_ANSWER`), as on every framework.
* In a run with no conversation, what Claude says in the turn its session runs after a
  background task ended is on the stream and in the transcript, not the run's answer (its
  own turn's result is).
* A model call is the CLI's: its own hooks (`ClaudeAgentOptions(hooks=...)`) have no model-call
  event ([hooks.md](../hooks.md)).
* A session resumes on the machine whose CLI holds it (or a shared `session_store`);
  elsewhere the query runs again from its prompt (above).
* Large results: a tool's result comes back from the CLI in one message (as the model reads it),
  which the SDK reads into a buffer of `max_buffer_size` bytes (1 MiB by default: a message
  over it fails the run with `CLIJSONDecodeError`). A harness tool's large result is cut before
  (above); a team's own MCP server's, or a built-in's, is not — what the model reads of it is
  Claude Code's to decide — so the harness runs the options with 100 MiB unless they name their
  own (an agent-runs artifact holds at most 50 MiB), escaped as JSON.

## Run it

* [`examples/02_way1_claude/agent.py`](../../examples/02_way1_claude/agent.py) — offline it
  drives a scripted stand-in for the CLI (`tests/support/fake_claude_cli.py`); with
  `LLM_GATEWAY=bifrost` the real `claude` CLI through Bifrost.
* Tests: `tests/integration/test_claude.py`, and against the real services
  `tests/live/test_live_matrix.py` (`claude`).
