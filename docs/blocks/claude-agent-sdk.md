# Recipe: Claude Agent SDK with the blocks (Way 2)

A Claude Agent SDK `query()` with your `ClaudeAgentOptions`, not wrapped, with the blocks
plugged in by your own code:

| Block | What it does here |
|---|---|
| memory ([memory.md](memory.md)) | the context for the question is appended to the system prompt; your tool records its calls; the turn and the outcome are recorded |
| governance ([governance.md](governance.md)) | the SDK's own permission callback, `can_use_tool`, asks `Governance.check` about every tool Claude calls: your in-process MCP tools and Claude Code's built-ins alike. An irreversible call is denied with `interrupt=True`: the query stops there |
| runs ([runs.md](runs.md)) | the pause waits in `role:support`'s inbox with the session id as its checkpoint; after the reviewer's answer a new query resumes the session, and the permission callback lets exactly the approved call through |
| evaluation ([evaluation.md](evaluation.md)) | the answer is judged on the run's trace |

The runnable version is [`examples/blocks_claude.py`](../../examples/blocks_claude.py): offline
it runs a scripted stand-in for the `claude` CLI that asks the permission callback as the real
CLI does; with `BIFROST_URL` it runs the real `claude` through Bifrost's Anthropic route. The
snippets below are from it. Wrapping the same options instead (Way 1):
[frameworks/claude-agent-sdk.md](../frameworks/claude-agent-sdk.md).

## Install

```bash
pip install -e '../agent-harness[claude-agent-sdk]'   # governance, evals, the Claude Agent SDK
pip install -e ../agent-runs/sdk/python                # trellis.runs (the harness brings it too)
```

The SDK drives the Claude Code CLI, which must be on the machine. Set `MEMORY_URL`, `RUNS_URL`
and `TRELLIS_API_KEY` for the services, and `BIFROST_URL` with `TRELLIS_JUDGE_MODEL` for the
judge; through Bifrost, `env={"ANTHROPIC_BASE_URL": "<gateway>/anthropic", ...}` in the options.

## The tools, and the permission callback

Your tools are an in-process MCP server as before; a tool records its own call in the run's
memory scope:

```python
from claude_agent_sdk import tool
from trellis.memory import current_context


@tool("close_ticket", "Close a support ticket.", {"ticket": str})
async def close_ticket(args: dict) -> dict:
    closed = await helpdesk.close(args["ticket"])
    if (scope := current_context()) is not None:  # the run's memory scope
        await scope.record_tool("close_ticket", args, output=closed)
    return {"content": [{"type": "text", "text": closed}]}
```

`can_use_tool` is where governance goes: Claude asks it before every tool call, so it covers
Claude Code's built-in tools (`Read`, `Bash`...) as well as yours. Say what each tool does; a
tool not named counts as a write:

```python
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext
from trellis.harness.governance import Decision, Governance

SERVER = "ops"  # your MCP server: its tools are mcp__ops__<tool>
SIDE_EFFECTS = {"ticket": "read", "close_ticket": "irreversible", "Read": "read", "Grep": "read"}
gov = Governance.from_env(agent_id="triage", tenant="acme")


def permission(asked: list[Decision], approved: list[ToolCall]):
    """Governance decides every call; a call that asks is denied and stops the query (`asked`
    keeps it), unless a reviewer `approved` it: that runs, once."""

    async def can_use_tool(name: str, args: dict, context: ToolPermissionContext):
        tool = name.removeprefix(f"mcp__{SERVER}__")
        decision = await gov.check(tool, args, side_effects=SIDE_EFFECTS.get(tool, "write"))
        if not decision.asks:
            return PermissionResultAllow()
        if approved and approved[0].tool == tool:
            return PermissionResultAllow(updated_input=approved.pop().args)
        asked.append(decision)
        return PermissionResultDeny(
            message=f"Waiting for approval: {decision.reason}", interrupt=True
        )

    return can_use_tool


def options(system: str, can_use_tool, session: str | None = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=system,
        mcp_servers={SERVER: create_sdk_mcp_server(SERVER, tools=[close_ticket])},
        can_use_tool=can_use_tool,
        resume=session,  # a session id: continue that conversation
    )
```

## One run

```python
async def claude(prompt: str, options: ClaudeAgentOptions) -> ResultMessage:
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    return result


run = await runs.start(RunStart(tenant_id="acme", agent_id="triage", user_id=user, input=question))

scope = memory.bind(tenant_id="acme", user_id=user).agent("triage", agent_run_id=run.run_id)
pushed = await scope.context(question)
system = f"You triage support tickets.\n\n{pushed.rendered}"

async with scope:  # what close_ticket records in
    asked: list[Decision] = []
    result = await claude(question, options(system, permission(asked, [])))
    if asked:  # governance asked: the run waits in agent-runs, the session id its checkpoint
        decision = asked[0]
        pending = ToolCall(tool=decision.tool, args=dict(decision.args))
        await runs.pause(
            Interrupt(
                tenant_id="acme",
                run_id=run.run_id,
                reason=InterruptReason.APPROVAL,
                question=decision.question,
                tool_call=pending,
                assignee="role:support",
            ),
            checkpoint={"session_id": result.session_id},
        )

        ...  # a reviewer answers from the inbox: runs.resume(InterruptResolution(...))

        record = await runs.get(run.run_id, tenant="acme")
        resolution = record.last_resolution
        approved = resolution.decision is InterruptDecision.APPROVE
        await gov.decided(
            decision,
            "approve" if approved else "reject",
            reviewer=resolution.reviewer or "unknown",
            run_id=run.run_id,
            user=user,
        )
        verdict = "approved" if approved else f"rejected ({resolution.answer})"
        again = permission(asked, [pending] if approved else [])
        session = record.checkpoint["session_id"]
        result = await claude(
            f"The reviewer {verdict} {decision.tool}. Go on.", options(system, again, session)
        )

answer = result.result or ""
await runs.finish(run.run_id, RunStatus.SUCCESS, output=answer, tenant="acme")
await scope.history.add([("USER", question), ("ASSISTANT", answer)])
await scope.feedback("run", run.run_id, "confirm", source="system")
```

* The denial with `interrupt=True` ends the query there: Claude is not asked to go on without
  the call. The session (`result.session_id`) is the checkpoint: the CLI keeps the conversation,
  and `resume=session` continues it in any process on the same machine (or one that shares the
  CLI's session store).
* After an approval the callback lets that one call through, with the arguments the reviewer
  approved (`updated_input`); after a reject the call stays denied, and the follow-up prompt
  tells Claude why.
* The reviewer reads the inbox and answers exactly as in the
  [LangGraph recipe](langgraph.md#one-run).

Then the answer is judged:

```python
case = EvalCase(
    input=question, output=answer, run_id=run.run_id, bundle_id=pushed.bundle_id, memory=scope
)
scores, failed = await judge(
    case, [grounding(), llm_judge("Says which ticket was closed.")], services=services
)
```

## What you own, and what Way 1 does for you

| | This recipe (Way 2) | Wrapped (Way 1) |
|---|---|---|
| where the pause lives | the session id in the checkpoint; a new query resumes it | the harness stops the CLI and re-runs against its journal: no call is made twice |
| governance | `can_use_tool` asking `check`, for every tool Claude calls | the harness's own tools (an in-process MCP server, `mcp__trellis__*`); Claude Code's built-ins stay under the SDK's permissions |
| recording, memory push | yours | automatic; the context appended to `system_prompt` |
| evaluation, serving | `judge(...)` where you choose; serving is yours | judges in the background; `serve_chat`, `serve_a2a` |

To run both ways in one deployment, see [mixing.md](mixing.md).
