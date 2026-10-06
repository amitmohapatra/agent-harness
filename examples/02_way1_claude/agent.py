"""Claude Agent SDK, wrapped, with every piece that applies: memory, local and MCP tools,
governance and a person, a hook, ``without=``, a time limit and ``ClaudeAgentOptions`` fields.

The target is your ``ClaudeAgentOptions``; the harness's tools (yours, the gateway's MCP tools,
memory's) reach Claude as one in-process MCP server, ``trellis`` (``mcp__trellis__*``), and
Claude Code's built-in tools are governed through ``can_use_tool``. The memory context is
appended to the system prompt. A pause stops the CLI; the resume continues its session, the
journal answering the paused call.

Offline this drives a scripted stand-in for the Claude Code CLI (``cli_path``). With
``BIFROST_URL`` set it runs the real ``claude`` CLI (install Claude Code), pointed at Bifrost's
Anthropic route: ``ANTHROPIC_BASE_URL=<gateway>/anthropic`` and
``ANTHROPIC_API_KEY=<virtual key>`` in the options' ``env``.

    python -m examples.02_way1_claude.agent
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import ClaudeAgentOptions
from examples._support.gateway import McpTool
from examples._support.offline import claude_cli, offline_blocks

from trellis import Harness, Hooks, Rewrite, tool
from trellis.contracts import ToolCall


def ticket(number: str) -> str:
    """A support ticket's state."""
    return f"{number}: resolved by the customer"


@tool(side_effects="irreversible")
def close_ticket(ticket: str, note: str = "") -> str:
    """Close a support ticket."""
    return f"closed {ticket} ({note})"


class Signed(Hooks):
    """Your rule: every closing note says who closed it."""

    async def before_tool(self, call: ToolCall) -> Rewrite | None:
        if call.tool != "close_ticket":
            return None
        return Rewrite({**call.args, "note": f"{call.args.get('note', '')} [triage agent]"})


async def main() -> None:
    script = [
        {"tool": "helpdesk-ticket", "args": {"number": "T-9"}},
        {"tool": "close_ticket", "args": {"ticket": "T-9", "note": "fixed"}},
        {"text": "Closed T-9: {last}"},
    ]
    helpdesk = {"helpdesk-ticket": McpTool(ticket, read_only=True)}
    async with Harness(**offline_blocks(mcp=helpdesk)) as h:
        options = ClaudeAgentOptions(
            system_prompt="You triage support tickets.", **claude_cli(script, "trellis")
        )
        agent = h.wrap(
            options,
            id="triage",
            tools=[close_ticket],
            hooks=[Signed()],
            timeout=300,
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
