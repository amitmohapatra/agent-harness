"""A tiny stdio MCP server the SDKs' own MCP features connect to: one read tool."""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("warehouse")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def warehouse_stock(sku: str) -> str:
    """Units of a SKU in the central warehouse."""
    return f"{sku}: 12 units at warehouse W2"


if __name__ == "__main__":
    server.run()
