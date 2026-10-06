"""``ReAct``, wrapped, with every piece on: memory, local and MCP tools, governance and a person
in the loop, a hook, ``without=``, a time limit and the framework's own options.

* memory — the context about the user is pushed in before the model is asked, the model searches
  memory itself (``memory_search``), and the run is recorded;
* tools — ``lead_time`` is a local function; ``erp-stock`` and ``erp-reorder`` are MCP tools of
  the gateway (offline: a scripted gateway serving two functions);
* governance — ``erp-stock`` only reads (its server says ``readOnlyHint``), so it runs;
  ``erp-reorder`` is destructive, so the run pauses for a person and the approved call runs once,
  the graph continuing from its checkpoint;
* a hook — ``CapOrders`` rewrites any order above 50 units to 50 before governance sees it;
* ``timeout=`` bounds every run of the agent; ``framework_options=`` is LangGraph's own run config;
* ``without={"memory"}`` — the second run has no memory at all.

    python -m examples.02_way1_react.agent
"""

from __future__ import annotations

import asyncio

from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, react_model

from trellis import Harness, Hooks, ReAct, Rewrite, tool
from trellis.contracts import ToolCall


def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3, "SKU-2": 40}.get(sku, 0)


def reorder(sku: str, qty: int) -> str:
    """Order units of a SKU from the supplier."""
    return f"PO-{sku}-{qty}"


@tool(side_effects="read")
def lead_time(sku: str) -> str:
    """How long the supplier takes to deliver a SKU."""
    return f"{sku}: 4 days"


class CapOrders(Hooks):
    """Your rule, in code: no order over 50 units."""

    async def before_tool(self, call: ToolCall) -> Rewrite | None:
        if call.tool == "erp-reorder" and call.args.get("qty", 0) > 50:
            return Rewrite({**call.args, "qty": 50})
        return None


ERP = {
    "erp-stock": McpTool(stock, read_only=True),
    "erp-reorder": McpTool(reorder, destructive=True),
}


async def main() -> None:
    memory = ScriptedMemory(context="Ada buys from ACME; she wants a PO number in every answer.")
    model = react_model(
        [
            ("memory_search", {"query": "preferred supplier"}),
            [("erp-stock", {"sku": "SKU-1"}), ("lead_time", {"sku": "SKU-1"})],
            ("erp-reorder", {"sku": "SKU-1", "qty": 80}),
            "SKU-1 was at 3 units; ordered 50 from ACME (PO-SKU-1-50), here in 4 days.",
            ("erp-stock", {"sku": "SKU-2"}),
            "SKU-2 has 40 units: no order needed.",
        ]
    )
    async with Harness(**offline_blocks(memory=memory, mcp=ERP)) as h:
        target = ReAct(system="You keep stock above 10 units. Check, then reorder.", model=model)
        agent = h.wrap(
            target,
            id="buyer",
            tools=[lead_time],
            hooks=[CapOrders()],
            timeout=300,  # the most working time of each run, however it starts
            framework_options={"recursion_limit": 40},  # LangGraph's own run config
        )

        result = await agent.run("Top up SKU-1 if it is low.", user="ada")
        while result.interrupt is not None:
            asked = result.interrupt
            print("approve?", asked.question, asked.tool_call.args if asked.tool_call else "")
            result = await agent.resume(asked.interrupt_id, "approve", reviewer="lee")
        print(result.status.value, result.answer)

        bare = await agent.run("How many SKU-2?", user="ada", without={"memory"})
        print(bare.status.value, bare.answer)
        await h.writes.drain()
    print("memory writes:", dict(memory.written))


if __name__ == "__main__":
    asyncio.run(main())
