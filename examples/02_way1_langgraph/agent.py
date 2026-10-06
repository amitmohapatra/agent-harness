"""LangGraph (LangChain v1 ``create_agent``), wrapped, with every piece that applies: memory,
local and MCP tools, governance and a person, a hook, ``without=``, a time limit and the graph's
own run config.

A compiled graph binds its tools when it is built, so it is built with ``h.tools(...)``: your
functions and the gateway's MCP tools (offline: a scripted gateway) as LangChain tools whose every
call goes through the harness — governed, journaled, recorded. No checkpointer here: the
approval re-runs the graph against the run's journal (the model plans again; the stock check is
replayed, not run again; the approved reorder runs once). ``stategraph.py`` resumes in place.

    python -m examples.02_way1_langgraph.agent
"""

from __future__ import annotations

import asyncio

from examples._support.gateway import McpTool
from examples._support.offline import langchain_model, offline_blocks
from langchain.agents import create_agent

from trellis import Harness, Hooks, Rewrite, tool
from trellis.contracts import ToolCall


def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


@tool(side_effects="irreversible")
def reorder(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"ordered {qty} x {sku}"


class Twenty(Hooks):
    """Your rule: reorders are always 20 units."""

    async def before_tool(self, call: ToolCall) -> Rewrite | None:
        return Rewrite({**call.args, "qty": 20}) if call.tool == "reorder" else None


async def main() -> None:
    erp = {"erp-stock": McpTool(stock, read_only=True)}
    async with Harness(**offline_blocks(mcp=erp)) as h:
        plan = [("erp-stock", {"sku": "SKU-1"}), ("reorder", {"sku": "SKU-1", "qty": 5})]
        model = langchain_model([*plan, *plan, "Ordered 20 x SKU-1.", "SKU-1: 3 units."])
        graph = create_agent(
            model,
            tools=await h.tools(reorder, framework="langgraph"),  # + the MCP and memory tools
            system_prompt="You keep stock above 10 units. Check stock, then reorder if low.",
        )
        agent = h.wrap(
            graph,
            id="stock-keeper",
            hooks=[Twenty()],
            timeout=300,
            framework_options={"recursion_limit": 20},  # LangGraph's own run config
        )

        result = await agent.run("Is SKU-1 low? Top it up if so.", user="ada")
        while result.interrupt is not None:  # the reorder is irreversible: a person approves it
            call = result.interrupt.tool_call
            print("asks:", result.interrupt.question, call.args if call else "")
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
        print(result.status.value, result.answer)

        quiet = await agent.run("How many SKU-1?", user="ada", without={"memory_push"})
        print("without the pushed context:", quiet.status.value, quiet.answer)


if __name__ == "__main__":
    asyncio.run(main())
