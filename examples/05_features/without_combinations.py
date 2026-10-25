"""``without=``: one switch turns parts of the harness off — on ``h.wrap`` for every run of the
agent, on ``agent.run``/``stream``/``start``/``schedule`` for one run, added to the agent's.

Six names, each one part: ``memory`` (the context, the memory tools, the records, the tool
narrowing), ``tool_search`` (the narrowing alone), ``mcp`` (the gateway's MCP tools and Code
Mode), ``code_mode``, ``grounding``, ``judges``. Any other name is refused before the run starts.

The agent here reports, from inside each run, whether the memory context reached it and which
parts are on (``agent.uses(part)``; ``trellis.current().uses`` in the code a run calls) — and
calls the gateway's MCP tool when ``mcp`` is on.

    python -m examples.05_features.without_combinations
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection

from examples._support.gateway import McpTool
from examples._support.offline import offline_blocks

from trellis import Harness, Runtime
from trellis.contracts import ConfigurationError
from trellis.harness.features import Feature

#: the parts a run asks about
PARTS: tuple[Feature, ...] = ("memory_push", "records", "tool_search", "mcp", "judges")


def ticket(number: str) -> str:
    """A support ticket's state: open, waiting on the customer, or closed.

    Args:
        number: the ticket's number, like "T-1"
    """
    return f"{number}: open"


async def report(question: str, agent: Runtime) -> str:
    on = [part for part in PARTS if agent.uses(part)]
    looked = await agent.tools.call("helpdesk-ticket", number="T-1") if agent.uses("mcp") else "-"
    context = "yes" if agent.context else "no"
    return f"context {context}; on: {', '.join(on) or 'nothing'}; ticket: {looked}"


async def main() -> None:
    helpdesk = {"helpdesk-ticket": McpTool(ticket, read_only=True)}
    async with Harness(**offline_blocks(mcp=helpdesk)) as h:
        # every run of this agent: no online judges
        agent = h.wrap(report, id="report", without={"judges"})
        runs: list[Collection[str]] = [
            (),  # only the agent's own: judges off
            {"memory"},  # this run: no memory at all
            {"tool_search"},  # memory on, its tool narrowing off
            {"mcp"},  # no MCP tools
            {"memory", "mcp"},  # both
        ]
        for without in runs:
            result = await agent.run("What is open?", user="ada", without=without)
            named = ", ".join(sorted(without)) or "(none)"
            print(f"without {named:>12}: {result.answer}")

        try:
            await agent.run("What is open?", user="ada", without={"tracing"})
        except ConfigurationError as refused:
            print("without tracing: refused —", str(refused).split(":")[0])


if __name__ == "__main__":
    asyncio.run(main())
