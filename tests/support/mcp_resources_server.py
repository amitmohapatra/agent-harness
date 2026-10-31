"""An MCP server with resources (MCP resources), on the official SDK, over stdio: a handbook
(markdown), the stock as CSV (``ROWS`` rows, 3 by default: a large one is kept and queried as
any large result), a page by its number (a template) and a logo (binary). ``NONE`` set in the
environment: the server has no resource (its capability declared all the same)."""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("Docs")


@server.tool()
def ping() -> str:
    """Answer pong: whether the server is up."""
    return "pong"


if not os.environ.get("NONE"):

    @server.resource(
        "docs://handbook",
        name="handbook",
        description="How the team works",
        mime_type="text/markdown",
    )
    def handbook() -> str:
        return "# Handbook\nShip on Tuesdays."

    @server.resource("data://stock.csv", name="stock", mime_type="text/csv")
    def stock() -> str:
        rows = int(os.environ.get("ROWS", "3"))
        lines = [f"SKU-{i:05d},{i % 7}" for i in range(rows)]
        return "\n".join(["sku,on_hand", *lines])

    @server.resource("docs://pages/{page}", name="page", description="A page by its number")
    def page(page: str) -> str:
        return f"page {page}"

    @server.resource("docs://logo.png", name="logo", mime_type="image/png")
    def logo() -> bytes:
        return b"\x89PNG\r\n"


if __name__ == "__main__":
    server.run("stdio")
