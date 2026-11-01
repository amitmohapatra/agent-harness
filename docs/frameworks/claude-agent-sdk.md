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

`Read`, `Write`, `Edit`, `Bash`, `WebFetch` and the rest are the CLI's own: it runs them. The
harness decides whether it may, through the SDK's own `PreToolUse` hook, which the CLI runs
before every tool call in every permission mode — the options it runs carry the harness's hook
after any of yours:

* **What.** Every built-in call is decided as a harness tool call is, under `allowed_tools`,
  `acceptEdits` and `bypassPermissions` too:
  the run's `before_tool` hooks ([hooks.md](../hooks.md)), then governance by the tool's risk —
  and a person when it asks. Approved (or edited), it runs once; rejected or denied, Claude reads
  why.
* **Risk.** `Read`, `Glob`, `Grep`, `LS`, `WebFetch`, `WebSearch`, `TodoWrite` read (they run);
  `Write`, `Edit`, `MultiEdit`, `NotebookEdit` write (they run, announced); `Bash` is
  irreversible (it asks). Your in-process server's tool is what its annotations say; any other
  tool the CLI asks about (an MCP server's of yours) is a write.
  The tool catalog's `risk` and `approve_when` override these, as for any tool
  ([governance.md](../governance.md)).
* **Not offered: schedules.** The CLI runs for one attempt and ends with it, so a scheduled
  prompt (`CronCreate`) or a wake-up (`ScheduleWakeup`) would never fire: these, with
  `CronList` and `CronDelete`, are added to the query's `disallowed_tools`, after your own.
  Offered, Claude promised a weekly report that nothing would send.
* **Your own `can_use_tool`** is asked by the harness's hook after its decision, with the
  arguments as they were decided (a hook's rewrite, a reviewer's edit): what it answers stands.
  A `permission_prompt_tool_name` of your own is refused: give your check as `can_use_tool`.
* **Not recorded.** A built-in call is not journaled or recorded in memory (the CLI keeps its
  result in its session, below).

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
  `ClaudeAgentOptions(sandbox=SandboxSettings(enabled=True, ...))`; the harness decides each
  call the CLI asks about (above), and a sandboxed `Bash` call is asked about only with
  `autoAllowBashIfSandboxed` false ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).
* **Prompts:** the options take text: `system_prompt=(await h.prompt("triage")).text` for a
  stored prompt of the gateway ([prompts.md](../prompts.md#native-or-ours)).

## What is automatic

| | |
|---|---|
| Memory push | the context goes with every prompt through Claude Code's own `UserPromptSubmit` hook (`additionalContext`), after the options' own hooks; the options' `system_prompt` is left as you wrote it |
| Long-term memory, session | the session is kept in the run's journal as the SDK's `SessionStore`, flushed at every step, so a resume on any worker continues it; options with their own `session_store` or `enable_file_checkpointing` keep the session on their machine (said once) |
| Spending | each query's usage and `total_cost_usd`, read from its `ResultMessage` (subagents included), count toward the run's budget |
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
process ends. The session the CLI kept (`ResultMessage.session_id`) is in the run's journal.
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
([subagents.md](../subagents.md#automatic)). A wrapped agent called from a tool of yours
(`await other.run(...)`) is a child run too, and the wrapped options are a `ReAct`'s sub-agent
like any agent (`subagents=[agent]`). Runnable:
[examples/06_scenarios/child_runs_every_framework.py](../../examples/06_scenarios/child_runs_every_framework.py).

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target ([runs.md](../runs.md), [surfaces.md](../surfaces.md),
[evaluation.md](../evaluation.md),
[examples/06_scenarios/runs_every_framework.py](../../examples/06_scenarios/runs_every_framework.py)):
the worker's machine needs the CLI; `llm_judge` asks the gateway's judge alias (or its own `model=`).

## Conversations

A conversation's turn continues the session the conversation's last turn ran in, in this
process (`resume=`, through the journal's session store, with the turn's own prompt:
[runs.md](../runs.md#a-claude-or-acp-session-across-turns)); in another process, or the first turn, a new session — with memory off given
the run store's earlier turns in its context. `mode="steer"` gives the message to Claude after
its next tool call (a `PostToolUse` hook's `additionalContext`); a query that calls no tool
takes none, and the message then runs as its own turn ([runs.md](../runs.md#a-message-to-a-busy-conversation-mode)).

## Limits

* A built-in that ran is not journaled or recorded (above).
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
