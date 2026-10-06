# Claude Agent SDK

The target is the `ClaudeAgentOptions` your team already configured. The harness runs the SDK's
`query(prompt, options)` on a copy of the options that also carries the harness tools; your
options object is never changed.

**Install:** `pip install 'trellis-harness[claude-agent-sdk]'`, and the Claude Code CLI the SDK
drives. Through Bifrost: `env={"ANTHROPIC_BASE_URL": "<gateway>/anthropic",
"ANTHROPIC_API_KEY": "<virtual key>"}` in the options.

This page is Way 1: the harness runs `query()`. To keep calling `query()` yourself and plug in
the blocks (memory, `can_use_tool` from governance, the session as the run's checkpoint in
agent-runs, a judge), see the Way 2 recipe:
[blocks/claude-agent-sdk.md](../blocks/claude-agent-sdk.md).

## Using an existing Claude Agent SDK project

```python
from claude_agent_sdk import ClaudeAgentOptions
from trellis import Harness, tool

h = Harness()


@tool(side_effects="irreversible")
def close_ticket(ticket: str) -> str:
    """Close a support ticket."""
    ...


options = ClaudeAgentOptions(system_prompt="You triage support tickets.", allowed_tools=["Read"])
agent = h.wrap(options, id="triage", tools=[close_ticket])  # was: query(prompt, options)

result = await agent.run("Ticket T-9 is resolved; close it.", user="ada")
if result.interrupt:
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
```

The harness tools — `tools=[...]`, the MCP tools the Bifrost virtual key allows and, memory on,
the memory tools — reach Claude as one in-process MCP server named `trellis` (tool names
`mcp__trellis__<tool>`), added beside your own `mcp_servers` (a dict, a config file path or its
JSON text); the harness's permission callback lets them through: the bridge is their check.
To build options yourself instead, `await h.tools(..., framework="claude_agent_sdk")` returns
that server's config (add it to `mcp_servers` as `"trellis"`).

**`query(...)` itself is not intercepted**: call `agent.run`/`stream`/`resume`/`start` instead.

## Claude Code's built-in tools

`Read`, `Write`, `Edit`, `Bash`, `WebFetch` and the rest are the CLI's own: it runs them. The
harness decides whether it may, through the SDK's own permission callback — the options it runs
carry a `can_use_tool` of the harness's:

* **What.** A built-in call the CLI asks permission for is decided as a harness tool call is:
  the run's `before_tool` hooks ([hooks.md](../hooks.md)), then governance by the tool's risk —
  and a person when it asks. Approved (or edited), it runs once; rejected or denied, Claude reads
  why.
* **Risk.** `Read`, `Glob`, `Grep`, `LS`, `WebFetch`, `WebSearch`, `TodoWrite` read (they run);
  `Write`, `Edit`, `MultiEdit`, `NotebookEdit` write (they run, announced); `Bash` is
  irreversible (it asks). Any other tool the CLI asks about (your own MCP server's) is a write.
  The tool catalog's `risk` and `approve_when` override these, as for any tool
  ([governance.md](../governance.md)).
* **Your own `can_use_tool`** is asked after the harness's decision, with the arguments as they
  were decided (a hook's rewrite, a reviewer's edit): what it answers stands.
* **Where not.** The CLI asks only about calls it does not already allow: a built-in your
  `allowed_tools` names whole (`"Read"`), or every call under `permission_mode=
  "bypassPermissions"`, never reaches the callback (the SDK warns of it). A
  `permission_prompt_tool_name` of your own is refused: give your check as `can_use_tool`.
* **Not recorded.** A built-in call is not journaled or recorded in memory (the CLI keeps its
  result in its session, below).

## Native or ours: skills, prompts, sandbox

* **Skills:** for `SKILL.md` folders the CLI discovers, use Claude's own:
  `ClaudeAgentOptions(skills=["name", ...] | "all")` (the SDK then allows the `Skill` tool and,
  `setting_sources` unset, loads the user's and the project's settings). Use the
  harness's (`h.wrap(..., skills=[...])`, on the `trellis` server) for skills from Bifrost's
  registry, or a version pinned per run and replayed ([skills.md](../skills.md#native-or-ours)).
* **Sandbox:** prefer Claude's own `Bash` and file tools, confined with
  `ClaudeAgentOptions(sandbox=SandboxSettings(enabled=True, ...))`; the harness decides each
  call the CLI asks about (above), and a sandboxed `Bash` call is asked about only with
  `autoAllowBashIfSandboxed` false ([sandbox.md](../sandbox.md#native-sandboxes-theirs-or-ours)).
* **Prompts:** the options take text: `system_prompt=await h.prompt(...)` for a prompt from a
  folder, Langfuse or Bifrost ([prompts.md](../prompts.md#native-or-ours)).

## What is automatic

| | |
|---|---|
| Memory push | the context is appended to the options' `system_prompt` (a string, a `preset`'s `append`, a `custom` prompt); with no system prompt it is the system prompt |
| Memory pull | the memory tools are on the `trellis` server |
| Records | the transcript (the question and the assistant's text blocks), every harness tool call, the `system` outcome; approvals as `TOOL_CALL` feedback |
| Tool hints | from 5 tools, the `trellis` server lists the hinted tools (and the memory tools, and those already used) for the run: the CLI lists an MCP server's tools once per query |
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

## Streaming

`agent.stream(...)` yields each assistant text block as a text delta, the harness tool calls as
`TOOL_CALL_*` events, then `RUN_FINISHED`.

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target ([runs.md](../runs.md), [surfaces.md](../surfaces.md),
[evaluation.md](../evaluation.md)): the worker's machine needs the CLI; `llm_judge` needs
`TRELLIS_JUDGE_MODEL`.

## Limits

* A built-in tool the CLI does not ask about (`allowed_tools`, `bypassPermissions`) is not the
  harness's; a built-in that ran is not journaled or recorded (above).
* No model hooks: the CLI makes the model calls ([hooks.md](../hooks.md)).
* A session resumes on the machine whose CLI holds it (or a shared `session_store`);
  elsewhere the query runs again from its prompt (above).
* A `system_prompt` given as a prompt file is read by the CLI; the context has no place in it
  and is not added (use a string or a preset with `append`).

## Run it

* [`examples/claude_agent_sdk_agent.py`](../../examples/claude_agent_sdk_agent.py) — offline it
  drives a scripted stand-in for the CLI (`tests/support/fake_claude_cli.py`); with
  `BIFROST_URL` the real `claude` CLI through Bifrost.
* Tests: `tests/integration/test_claude.py`, and against the real services
  `tests/live/test_live_matrix.py` (`claude`).
