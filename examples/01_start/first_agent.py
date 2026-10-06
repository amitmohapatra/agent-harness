"""Level 1: a model with tools, and a person who approves the risky one.

``ReAct`` is a tool-calling agent for teams with no framework (LangChain's ``create_agent`` with
the harness's middleware). ``stock`` only reads, so it runs; ``reorder`` is irreversible, so the
run pauses until a person approves it, and the approved call runs once.

    python -m examples.01_start.first_agent       # a scripted model; BIFROST_URL: a real one
"""

from __future__ import annotations

import asyncio

from examples._support.offline import react_model

from trellis import Harness, ReAct, tool


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


@tool(side_effects="irreversible")
def reorder(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"ordered {qty} x {sku}"


async def main() -> None:
    model = react_model(
        [
            ("stock", {"sku": "SKU-1"}),
            ("reorder", {"sku": "SKU-1", "qty": 20}),
            "SKU-1 was at 3 units: I ordered 20 more.",
        ]
    )
    async with Harness() as h:
        target = ReAct(system="You keep stock above 10 units.", model=model)
        agent = h.wrap(target, id="stock-keeper", tools=[stock, reorder])

        result = await agent.run("Is SKU-1 low? Top it up if so.", user="ada")
        while result.interrupt is not None:  # reorder is irreversible: a person decides
            print("waiting for a person:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lee")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
