"""Way 2, pluggable blocks: a plain Claude Agent SDK ``query()`` — not wrapped — with the
blocks plugged in by the team's own code.

* Memory (``trellis.memory``): the context for the question is appended to the system prompt;
  the turn, each tool call and the outcome are recorded.
* Governance (``trellis.harness.governance``): the SDK's own permission callback,
  ``can_use_tool``, asks ``Governance.check`` about every tool Claude calls — your in-process
  MCP tools and Claude Code's built-ins alike. ``close_ticket`` is irreversible, so its call is
  denied with ``interrupt=True``: the query stops there.
* Runs (``trellis.runs``): the pause waits in ``role:support``'s inbox with the session id as
  its checkpoint; after the reviewer's answer a new query resumes the session, and the
  permission callback lets exactly the approved call through.
* Evaluation (``trellis.harness.evals``): the answer is judged on the run's trace.

Offline (no ``BIFROST_URL``) the CLI is a scripted stand-in (``tests/support/fake_claude_cli.py``)
that asks the permission callback as the real CLI does; with ``BIFROST_URL`` it is the real
``claude`` through Bifrost. Without ``RUNS_URL`` / ``MEMORY_URL`` the run store is the in-process
one and the memory service a scripted one in this process.

    python -m examples.03_way2_governance.claude_recipe
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    ToolPermissionContext,
    create_sdk_mcp_server,
    query,
    tool,
)
from examples._support.offline import Runs, claude_cli, judge_services, memory_client, runs_store

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ToolCall,
)
from trellis.harness.evals import EvalCase, EvalServices, grounding, judge, llm_judge
from trellis.harness.governance import Decision, Governance
from trellis.memory import MemoryClient, MemoryContext, PromptContext, current_context

TENANT = "default"  # the tenant your key speaks for (a development key's: default)
AGENT = "triage"
SERVER = "ops"  # your in-process MCP server: its tools are mcp__ops__<tool>
#: what each tool does: governance decides from it (and the catalog, memory on); a tool not
#: named here — a built-in Claude Code tool — counts as a write
SIDE_EFFECTS = {"ticket": "read", "close_ticket": "irreversible", "Read": "read", "Grep": "read"}
#: the run's memory scope and the context pushed into it (both None with memory off)
Recall = tuple[MemoryContext | None, PromptContext | None]


# --------------------------------------------------------------------------- your tools
@tool("close_ticket", "Close a support ticket.", {"ticket": str})
async def close_ticket(args: dict[str, Any]) -> dict[str, Any]:
    closed = f"closed {args['ticket']}"
    if (scope := current_context()) is not None:  # the run's memory scope (memory on)
        await scope.record_tool("close_ticket", args, output=closed)
    return {"content": [{"type": "text", "text": closed}]}


def permission(gov: Governance, asked: list[Decision], approved: list[ToolCall]) -> Any:
    """``can_use_tool``: governance decides every call; a call that asks is denied and stops
    the query (``asked`` keeps it) — unless a reviewer ``approved`` it: that runs, once."""

    async def can_use_tool(
        name: str, args: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        tool_name = name.removeprefix(f"mcp__{SERVER}__")
        decision = await gov.check(
            tool_name, args, side_effects=SIDE_EFFECTS.get(tool_name, "write")
        )
        if not decision.asks:
            return PermissionResultAllow()
        if approved and approved[0].tool == tool_name:
            return PermissionResultAllow(updated_input=approved.pop().args)
        asked.append(decision)
        return PermissionResultDeny(
            message=f"Waiting for approval: {decision.reason}", interrupt=True
        )

    return can_use_tool


async def claude(prompt: str, options: ClaudeAgentOptions) -> ResultMessage:
    """One query, to its result."""
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    assert result is not None
    return result


def options(
    system: str, can_use_tool: Any, script: Sequence[dict[str, Any]], session: str | None = None
) -> ClaudeAgentOptions:
    """Your options; ``script`` is what the offline stand-in for the CLI does."""
    return ClaudeAgentOptions(
        system_prompt=system,
        mcp_servers={SERVER: create_sdk_mcp_server(SERVER, tools=[close_ticket])},
        can_use_tool=can_use_tool,
        resume=session,
        **claude_cli(script, SERVER),
    )


# --------------------------------------------------------------------------- one run
async def recall(memory: MemoryClient | None, run_id: str, user: str, question: str) -> Recall:
    """The run's memory scope and the context for the question (memory on)."""
    if memory is None:
        return None, None
    scope = memory.bind(tenant_id=TENANT, user_id=user).agent(AGENT, agent_run_id=run_id)
    return scope, await scope.context(question)


async def review(runs: Runs, run_id: str) -> None:
    """A reviewer, any time later, from any process: the inbox, and an answer (here, to the
    run this example started)."""
    async for waiting in runs.iterate(
        status=RunStatus.PAUSED, assignee="role:support", tenant=TENANT
    ):
        assert waiting.awaiting is not None
        print("inbox:", waiting.run_id, waiting.awaiting.question)
        if waiting.run_id != run_id:
            continue
        answer = InterruptResolution(
            interrupt_id=waiting.awaiting.interrupt_id,
            run_id=waiting.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="user:lead",
        )
        await runs.resume(answer, tenant=TENANT)  # RUNNING again, attempt 2


async def finish(
    runs: Runs,
    services: EvalServices,
    memory: Recall,
    *,
    run_id: str,
    question: str,
    answer: str,
) -> None:
    """End the run, record the turn and its outcome (memory on), and judge the answer."""
    await runs.finish(run_id, RunStatus.SUCCESS, output=answer, tenant=TENANT)
    print(RunStatus.SUCCESS.value, answer)
    scope, pushed = memory
    if scope is not None:
        await scope.history.add([("USER", question), ("ASSISTANT", answer)])
        await scope.feedback("run", run_id, "confirm", source="system")
    case = EvalCase(
        input=question,
        output=answer,
        run_id=run_id,
        bundle_id=pushed.bundle_id if pushed is not None else None,
        memory=scope,
    )
    scores, failed = await judge(
        case, [grounding(), llm_judge("Says which ticket was closed.")], services=services
    )
    print("judged:", [(s.name, s.value) for s in scores], failed)


async def main() -> None:
    gov = Governance.from_env(agent_id=AGENT, tenant=TENANT)
    runs, memory, services = runs_store(), memory_client(), judge_services()
    question, user = "Ticket T-9 is resolved; close it.", "ada"
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id=AGENT, user_id=user, input=question))
    scope, pushed = await recall(memory, run.run_id, user, question)
    system = "You triage support tickets."
    if pushed is not None:
        system = f"{system}\n\n{pushed.rendered}"
    script = [{"tool": "close_ticket", "args": {"ticket": "T-9"}}, {"text": "Closed T-9."}]

    async with scope or contextlib.nullcontext():  # what close_ticket records in
        asked: list[Decision] = []
        result = await claude(question, options(system, permission(gov, asked, []), script))
        if asked:  # governance asked: the run waits in agent-runs
            decision = asked[0]
            pending = ToolCall(tool=decision.tool, args=dict(decision.args))
            await runs.pause(
                Interrupt(
                    tenant_id=TENANT,
                    run_id=run.run_id,
                    reason=InterruptReason.APPROVAL,
                    question=decision.question,
                    tool_call=pending,
                    assignee="role:support",
                ),
                checkpoint={"session_id": result.session_id},
            )

            await review(runs, run.run_id)

            record = await runs.get(run.run_id, tenant=TENANT)
            assert record is not None and record.checkpoint and record.last_resolution
            resolution = record.last_resolution
            approved = resolution.decision is InterruptDecision.APPROVE
            await gov.decided(  # the memory service learns approval rules from it (memory on)
                decision,
                "approve" if approved else "reject",
                reviewer=resolution.reviewer or "unknown",
                run_id=run.run_id,
                user=user,
            )
            verdict = "approved" if approved else f"rejected ({resolution.answer})"
            follow_up = f"The reviewer {verdict} {decision.tool}. Go on."
            again = permission(gov, asked, [pending] if approved else [])
            session = record.checkpoint["session_id"]
            result = await claude(follow_up, options(system, again, script, session))

    answer = result.result or ""
    await finish(
        runs, services, (scope, pushed), run_id=run.run_id, question=question, answer=answer
    )
    for client in (gov, runs, services, memory):
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
