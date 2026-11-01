"""An MCP server that asks the person in the middle of a call (MCP elicitation), on the official
SDK, over stdio: ``order`` asks which size (one field of a few values: a choice), ``book`` asks
for a day and a note (a form; the day a date). ``LEGACY`` set in the environment: the server
speaks only the protocol before 2026-07-28, where the question comes as a request of the
server's own during the call; otherwise it comes back in the call's result (2026-07-28)."""

from __future__ import annotations

import os
from datetime import date
from typing import Annotated, Literal

import anyio
from mcp.server.elicitation import ElicitationResult
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve
from mcp.server.runner import serve_loop
from mcp.server.stdio import stdio_server
from pydantic import BaseModel

server = MCPServer("Shop")
served: list[str] = []


class Size(BaseModel):
    size: Literal["S", "L"]


class Booking(BaseModel):
    day: date
    note: str


def which_size(item: str) -> Elicit[Size]:
    return Elicit(f"Which size of {item}?", Size)


def when(item: str) -> Elicit[Booking]:
    return Elicit(f"When should {item} be delivered?", Booking)


@server.tool()
def order(item: str, size: Annotated[ElicitationResult[Size], Resolve(which_size)]) -> str:
    """Order an item; the person picks its size.

    Args:
        item: The item to order.
    """
    served.append("order")
    if size.action != "accept":
        return f"not ordered ({size.action})"
    return f"ordered {item} in size {size.data.size}"


@server.tool()
def book(item: str, booking: Annotated[Booking, Resolve(when)]) -> str:
    """Book a delivery; the person gives the day.

    Args:
        item: The item to deliver.
    """
    served.append("book")
    return f"{item} on {booking.day.isoformat()}: {booking.note}"


@server.tool()
def served_calls() -> list[str]:
    """The calls this process served to the end."""
    return list(served)


async def handshake_only() -> None:
    """Serve over stdio the protocol before 2026-07-28 only (the handshake), as an older
    server does: the client's ``server/discover`` is not known, so it falls back."""
    low = server._lowlevel_server
    async with stdio_server() as (read, write), low.lifespan(low) as state:
        options = low.create_initialization_options()
        await serve_loop(low, read, write, lifespan_state=state, init_options=options)


if __name__ == "__main__":
    if os.environ.get("LEGACY"):
        anyio.run(handshake_only)
    else:
        server.run("stdio")
