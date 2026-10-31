"""``trellis.middleware()`` on graphs of your own: Deep Agents' ``create_deep_agent`` and
LangChain's ``create_agent``, each given the harness's part of ``ReAct`` and the run's
checkpointer, then wrapped.

What the middleware gives them: the run's tools per model call (here the gateway's MCP tool
``shop-order`` and the memory tools, beside the graph's own ``refund``), narrowed per turn; the
graph's own tools governed (the catalog says ``refund`` is irreversible: a person approves it,
and the resume continues the graph in place, the model not asked again); ``max_steps``
(LangChain's model call limit); the stall guard; the hooks and spans; ``call_timeout``. A graph
wrapped without it still runs governed and narrowed, said once in the log.

    python -m examples.02_way1_deepagents.middleware
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent
from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import langchain_model, offline_blocks
from langchain.agents import create_agent
from langchain_core.tools import tool

from trellis import Harness, middleware
from trellis.harness.middleware import RunCheckpointer


def order(number: str) -> str:
    """An order's amount and state."""
    return f"order {number}: 40 EUR, delivered damaged"


@tool
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


SYSTEM = "You process refund requests. Look the order up, then refund it."
SCRIPT = [("shop-order", {"number": "o-7"}), ("refund", {"order": "o-7", "amount": 40.0})]


async def main() -> None:
    shop = {"shop-order": McpTool(order, read_only=True)}
    catalog = {"refund": {"risk": "irreversible"}}  # an administrator's, in the tool catalog
    async with Harness(**offline_blocks(memory=ScriptedMemory(catalog=catalog), mcp=shop)) as h:
        graphs = {
            "deep": create_deep_agent(
                model=langchain_model([*SCRIPT, "Refunded 40.0 on o-7."]),
                tools=[refund],
                system_prompt=SYSTEM,
                middleware=middleware(max_steps=8, call_timeout=60),
                checkpointer=RunCheckpointer(),
            ),
            "plain": create_agent(
                langchain_model([*SCRIPT, "Refunded 40.0 on o-7."]),
                tools=[refund],
                system_prompt=SYSTEM,
                middleware=middleware(max_steps=8, call_timeout=60),
                checkpointer=RunCheckpointer(),
            ),
        }
        for name, graph in graphs.items():
            agent = h.wrap(graph, id=f"refunds-{name}")
            result = await agent.run("Refund order o-7 (40 EUR).", user="ada")
            while result.interrupt is not None:
                print(f"{name} asks:", result.interrupt.question)
                result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="bo")
            print(f"{name}:", result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
