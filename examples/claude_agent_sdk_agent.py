"""Claude Agent SDK: the target is your ``ClaudeAgentOptions``; harness tools reach Claude as
an in-process MCP server (``mcp__trellis__*``).

Offline this drives a scripted stand-in for the Claude Code CLI (``cli_path``). With
``BIFROST_URL`` set it runs the real ``claude`` CLI (install Claude Code), pointed at
Bifrost's Anthropic route: ``ANTHROPIC_BASE_URL=<gateway>/anthropic`` and
``ANTHROPIC_API_KEY=<virtual key>`` in the options' ``env``.

    .venv/bin/python examples/claude_agent_sdk_agent.py
"""

from __future__ import annotations

import asyncio
import json

from _offline import FAKE_CLAUDE_CLI, gateway, online
from claude_agent_sdk import ClaudeAgentOptions

from trellis import Harness, tool


@tool(side_effects="irreversible")
def close_ticket(ticket: str) -> str:
    """Close a support ticket."""
    return f"closed {ticket}"


def options() -> ClaudeAgentOptions:
    prompt = "You triage support tickets."
    if online():
        url, key = gateway()
        origin = url.removesuffix("/v1")
        return ClaudeAgentOptions(
            system_prompt=prompt,
            env={"ANTHROPIC_BASE_URL": f"{origin}/anthropic", "ANTHROPIC_API_KEY": key},
        )
    script = [{"tool": "close_ticket", "args": {"ticket": "T-9"}}, {"text": "Closed T-9."}]
    return ClaudeAgentOptions(
        system_prompt=prompt,
        cli_path=FAKE_CLAUDE_CLI,
        env={"FAKE_CLAUDE_SCRIPT": json.dumps(script)},
    )


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(options(), id="triage", tools=[close_ticket])
        result = await agent.run("Ticket T-9 is resolved; close it.", user="ada")
        while result.interrupt is not None:
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
