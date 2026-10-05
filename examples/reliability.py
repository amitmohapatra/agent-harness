"""Time limits, retries, unknown outcomes and cancel — what you write is only the timeouts.

A read that fails twice is retried; a write that hangs past its ``timeout`` is reported to the
code (or the model) as of unknown effect, with the idempotency key its service would have
deduplicated on; a run past its ``timeout`` ends ``TIMEOUT``; a run is cancelled with a reason.

    .venv/bin/python examples/reliability.py
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime, current, tool
from trellis.contracts import ToolError

asked: list[str] = []


@tool(side_effects="read")
async def quote(sku: str) -> str:
    """The price of a SKU (its service is busy twice before it answers)."""
    asked.append(sku)
    if len(asked) < 3:
        raise ToolError("the price service is busy", retryable=True)
    return f"{sku} costs 7"


@tool(side_effects="write", timeout=0.2)
async def order(sku: str) -> str:
    """Order a SKU (the supplier never answers in time)."""
    runtime = current()
    assert runtime is not None
    print("ordering with idempotency key", runtime.idempotency_key)
    await asyncio.sleep(10)
    return "ordered"


async def buy(sku: str, agent: Runtime) -> list[str]:
    return [await agent.tools.call("quote", sku=sku), await agent.tools.call("order", sku=sku)]


pondering: list[str] = []


async def ponder(question: str, agent: Runtime) -> str:
    pondering.append(agent.run_id)
    await asyncio.sleep(10)
    return "an answer, too late"


async def main() -> None:
    async with Harness() as h:
        buyer = h.wrap(buy, id="buyer", tools=[quote, order], version="2026.10")
        result = await buyer.run("A-1", user="ada")
        print(result.status.value, result.answer)  # the price, then: may or may not have ordered

        thinker = h.wrap(ponder, id="thinker")
        timed = await thinker.run("What is the meaning of it all?", user="ada", timeout=0.2)
        assert timed.error is not None
        print(timed.status.value, timed.error.code, timed.error.message)

        running = asyncio.create_task(thinker.run("And now?", user="ada"))
        await asyncio.sleep(0.1)
        record = await thinker.cancel(pondering[-1], reason="nobody is waiting any more")
        print(record.status.value, (await running).status.value)


if __name__ == "__main__":
    asyncio.run(main())
