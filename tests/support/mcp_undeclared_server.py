"""An MCP server whose tools say nothing of what they do (no annotations), over stdio:
``python -m tests.support.mcp_undeclared_server``. Each call that runs adds a line — the tool's
name — to the file ``RAN`` names, so a test sees whether a tool's body ran at all."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

server = MCPServer("Back Office")


def _ran(tool: str) -> None:
    with Path(os.environ["RAN"]).open("a") as log:
        log.write(f"{tool}\n")


@server.tool()
def purge_records(table: str) -> str:
    """Purge every record of a table."""
    _ran("purge_records")
    return f"purged {table}"


@server.tool()
def set_prices(percent: int) -> str:
    """Change every price by a percentage."""
    _ran("set_prices")
    return f"prices changed by {percent}%"


@server.tool()
def read_account(account: str) -> str:
    """Read an account's data."""
    _ran("read_account")
    return f"the data of {account}"


if __name__ == "__main__":
    server.run("stdio")
