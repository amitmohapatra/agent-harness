"""Claude Agent SDK, wrapped, with every piece that applies: memory, local and MCP tools,
governance and a person, a hook, ``without=``, a time limit and ``ClaudeAgentOptions`` fields.

The target is your ``ClaudeAgentOptions``, with your tools on your own in-process MCP server
(``desk``); the harness's tools (the gateway's MCP tools, memory's) reach Claude as one more,
``trellis`` (``mcp__trellis__*``). Every call — yours, the harness's, Claude Code's built-in
tools — is governed through Claude's hooks and ``can_use_tool``: ``close_ticket`` says it is
destructive, so it waits for a person. The memory context is
appended to the system prompt. A pause stops the CLI; the resume continues its session, the
journal answering the paused call.

Offline this drives a scripted stand-in for the Claude Code CLI (``cli_path``). With Bifrost
set (``LLM_BASE_URL``, ``LLM_GATEWAY=bifrost``) it runs the real ``claude`` CLI (install Claude
Code), pointed at Bifrost's Anthropic route: ``ANTHROPIC_BASE_URL=<gateway>/anthropic`` and
``ANTHROPIC_API_KEY=<virtual key>`` in the options' ``env``.

    python -m examples.02_way1_claude.agent
"""

from __future__ import annotations

import asyncio
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from examples._support.gateway import McpTool
from examples._support.offline import claude_cli, offline_blocks
from mcp.types import ToolAnnotations

from trellis import Harness, Hooks, Rewrite
from trellis.contracts import ToolCall

#: How Claude names your server's tool.
CLOSE: str = "mcp__desk__close_ticket"


def ticket(number: str) -> str:
    """A support ticket's state."""
    return f"{number}: resolved by the customer"


@tool(
    "close_ticket",
    "Close a support ticket.",
    {"ticket": str, "note": str},
    annotations=ToolAnnotations(destructive_hint=True),  # irreversible: a person approves it
)
async def close_ticket(args: dict[str, Any]) -> dict[str, Any]:
    text = f"closed {args['ticket']} ({args.get('note', '')})"
    return {"content": [{"type": "text", "text": text}]}


class Signed(Hooks):
    """Your rule: every closing note says who closed it."""

    async def before_tool(self, call: ToolCall) -> Rewrite | None:
        if call.tool != CLOSE:
            return None
        return Rewrite({**call.args, "note": f"{call.args.get('note', '')} [triage agent]"})


async def main() -> None:
    script = [
        {"tool": "helpdesk-ticket", "args": {"number": "T-9"}},
        {"tool": "close_ticket", "server": "desk", "args": {"ticket": "T-9", "note": "fixed"}},
        {"text": "Closed T-9: {last}"},
    ]
    helpdesk = {"helpdesk-ticket": McpTool(ticket, read_only=True)}
    async with Harness(**offline_blocks(mcp=helpdesk)) as h:
        options = ClaudeAgentOptions(
            system_prompt="You triage support tickets.",
            mcp_servers={"desk": create_sdk_mcp_server("desk", tools=[close_ticket])},
            **claude_cli(script, "trellis"),
        )
        agent = h.wrap(
            options,
            id="triage",
            hooks=[Signed()],
            run_timeout=300,
            framework_options={"max_turns": 6},  # a ClaudeAgentOptions field, for every run
            without={"grounding"},
        )
        result = await agent.run("Ticket T-9 is resolved; close it.", user="ada")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
