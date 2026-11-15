"""An MCP server with one tool of each risk its annotations can say, over stdio:
``python -m tests.support.mcp_risks_server``. ``look`` says it only reads (``readOnlyHint``),
``note`` that it is not destructive (``destructiveHint: false``), ``tidy`` says nothing (no
annotations at all), ``purge`` that it is destructive (``destructiveHint: true``) and ``guess``
only that it does not just read (``readOnlyHint: false``: destructive by the protocol's
default). Each call that runs adds a line — the tool's name and its ``n`` — to the file ``RAN``
names, so a test sees which calls' bodies ran."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("Ledger")


def _ran(tool: str, n: int) -> str:
    with Path(os.environ["RAN"]).open("a") as log:
        log.write(f"{tool} {n}\n")
    return f"{tool} {n} done"


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def look(n: int) -> str:
    """Look at an entry of the ledger."""
    return _ran("look", n)


@server.tool(annotations=ToolAnnotations(destructive_hint=False))
def note(n: int) -> str:
    """Add a note to an entry of the ledger."""
    return _ran("note", n)


@server.tool()
def tidy(n: int) -> str:
    """Tidy an entry of the ledger."""
    return _ran("tidy", n)


@server.tool(annotations=ToolAnnotations(destructive_hint=True))
def purge(n: int) -> str:
    """Purge an entry of the ledger."""
    return _ran("purge", n)


@server.tool(annotations=ToolAnnotations(read_only_hint=False))
def guess(n: int) -> str:
    """Guess an entry of the ledger."""
    return _ran("guess", n)


if __name__ == "__main__":
    server.run("stdio")
