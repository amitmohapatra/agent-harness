"""``ReAct``'s sub-agents, each call a child run of the run:

* a **Deep Agents sub-agent** — a ``SubAgent`` dict, called through Deep Agents' ``task`` tool;
  its ``interrupt_on`` is the catalog's: the ``wire`` it names asks a person, and the resume
  continues the sub-agent where it stopped;
* a **wrapped agent** — any framework's (here a function), a tool named after its id;
* an **A2A agent** — ``a2a(url)``, served here on a real port by a harness of its own.

    python -m examples.02_way1_react.subagents
"""

from __future__ import annotations

import asyncio
import socket
import threading

import uvicorn
from examples._support.offline import langchain_model, react_model
from fastapi import FastAPI
from langchain_core.tools import tool as langchain_tool

from trellis import Harness, ReAct, Runtime, a2a


@langchain_tool
def wire(supplier: str, amount: float) -> str:
    """Wire money to a supplier."""
    return f"wired {amount} EUR to {supplier}"


async def stock(input: str, agent: Runtime) -> str:
    """How many units of an item are in stock."""
    return f"{input}: 12 units in stock"


async def pricing(input: str, agent: Runtime) -> str:
    """The supplier's price for an item."""
    return f"{input}: 4.20 EUR a unit"


def serve_pricing() -> tuple[str, uvicorn.Server, Harness]:
    """The pricing agent, served over A2A by its own harness on a free port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/a2a"
    remote, app = Harness(), FastAPI()
    remote.wrap(pricing, id="pricing").serve_a2a(app, url)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return url, server, remote


async def main() -> None:
    url, server, remote = serve_pricing()
    while not server.started:  # noqa: ASYNC110 - uvicorn says so with a flag, not an event
        await asyncio.sleep(0.05)
    payer = {
        "name": "payer",
        "description": "Pays a supplier.",
        "system_prompt": "You pay suppliers: wire the amount asked, then say so.",
        "model": langchain_model([("wire", {"supplier": "Acme", "amount": 50.4}), "Paid."]),
        "tools": [wire],
        "interrupt_on": {"wire": True},  # the catalog's: the wire asks a person
    }
    try:
        async with Harness() as h:
            buyer = ReAct(
                "You restock items: check stock, get the price, then have the payer pay.",
                react_model(
                    [
                        [("stock", {"message": "bolts"}), ("pricing", {"message": "bolts"})],
                        ("task", {"description": "Pay Acme 50.40 EUR.", "subagent_type": "payer"}),
                        "Bolts: 12 in stock at 4.20 EUR; Acme paid 50.40 EUR.",
                    ]
                ),
                subagents=[payer, h.wrap(stock, id="stock"), a2a(url, name="pricing")],
            )
            agent = h.wrap(buyer, id="buyer")
            result = await agent.run("Restock 12 bolts.", user="ada")
            while result.interrupt is not None:
                print("asks:", result.interrupt.question)
                result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="bo")
            print(result.status.value, result.answer)
    finally:
        server.should_exit = True
        await remote.aclose()


if __name__ == "__main__":
    asyncio.run(main())
