# Claude Agent SDK

The target is the `ClaudeAgentOptions` your team already configured. The harness runs the SDK's
`query(prompt, options)` on a copy of the options that also carries the harness tools; your
options object is never changed.

**Install:** `pip install 'trellis-harness[claude-agent-sdk]'`, and the Claude Code CLI the SDK
drives. Through Bifrost: `env={"ANTHROPIC_BASE_URL": "<gateway>/anthropic",
"ANTHROPIC_API_KEY": "<virtual key>"}` in the options.

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
JSON text) and pre-allowed in `allowed_tools`: the harness's bridge is their permission check.
To build options yourself instead, `await h.tools(..., framework="claude-agent-sdk")` returns
that server's config (add it to `mcp_servers` as `"trellis"` and its tools to `allowed_tools`).

**`query(...)` itself is not intercepted**: call `agent.run`/`stream`/`resume`/`start` instead.

## Claude Code's built-in tools

`Read`, `Write`, `Edit`, `Bash`, `WebFetch` and the rest are the CLI's own. The harness does not
see them: they are not tiered, approved, journaled or recorded, and on a re-run after a pause
they run again. Their permissions are the SDK's — `allowed_tools`, `disallowed_tools`,
`permission_mode`, `can_use_tool`. Put anything with side effects that must happen once, or be
approved, behind a harness tool, and keep the built-ins to what is safe to repeat (or turn them
off with `disallowed_tools`).

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

A harness tool that asks (`irreversible`, a catalog `approve_when`), or `trellis.current().ask`
inside one, pauses the run: the harness stops consuming the query and the CLI process ends.
`agent.resume(...)` starts the query again from the prompt as the run's next attempt; the
journal returns the harness calls already made and the answer given, so the approved call runs
once (Claude is asked again for the steps before it). Every decision works: `approve`, `reject`
with a reason (the tool result Claude reads: "… was not run: the approver rejected it (why)"),
`edit`, `answer`, `cancel`.

## Streaming

`agent.stream(...)` yields each assistant text block as a text delta, the harness tool calls as
`TOOL_CALL_*` events, then `RUN_FINISHED`.

## Durable runs, workers, schedules, AG-UI, A2A, evaluation

The same as every target ([runs.md](../runs.md), [surfaces.md](../surfaces.md),
[evaluation.md](../evaluation.md)): the worker's machine needs the CLI; `llm_judge` needs
`TRELLIS_JUDGE_MODEL`.

## Limits

* Built-in tools are not the harness's (above).
* A pause ends the CLI process; the resume is a new query, not the SDK session resumed.
* A `system_prompt` given as a prompt file is read by the CLI; the context has no place in it
  and is not added (use a string or a preset with `append`).

## Run it

* [`examples/claude_agent_sdk_agent.py`](../../examples/claude_agent_sdk_agent.py) — offline it
  drives a scripted stand-in for the CLI (`tests/support/fake_claude_cli.py`); with
  `BIFROST_URL` the real `claude` CLI through Bifrost.
* Tests: `tests/integration/test_claude.py`, and against the real services
  `tests/live/test_live_matrix.py` (`claude`).
