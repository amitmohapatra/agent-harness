"""Way 2, the gateway: Bifrost's MCP tools and your own model client, through ``bifrost-sdk`` —
no harness.

* ``bifrost.tools()`` — the MCP tools the virtual key allows, with their servers' annotations
  (``readOnlyHint``, ``destructiveHint``: what governance decides from);
* ``bifrost.execute_tool(call, options=Options(mcp_clients=[...]))`` — one call, scoped to a
  server, as the gateway runs it;
* ``NO_GATEWAY_TOOLS`` — the headers for a model client of your own pointed at the gateway
  (``ChatOpenAI(default_headers=...)``): the gateway then adds none of its MCP tools to your
  requests and runs none itself, so your code decides every call. (A wrapped agent's
  ``h.model_headers()`` is the same, plus a stored prompt's selection.)

Offline the gateway is a scripted one serving two functions; with ``BIFROST_URL`` (and
``BIFROST_VIRTUAL_KEY``) it is the real one and the tools are whatever the key allows.

    python -m examples.03_way2_gateway.mcp_and_model_headers
"""

from __future__ import annotations

import asyncio
import json
import os

from bifrost_sdk import NO_GATEWAY_TOOLS, Bifrost, Options
from examples._support.gateway import McpTool, ScriptedGateway


def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


def reorder(sku: str, qty: int) -> str:
    """Order units of a SKU."""
    return f"PO-{sku}-{qty}"


def gateway() -> Bifrost:
    if os.environ.get("BIFROST_URL"):
        return Bifrost(os.environ["BIFROST_URL"], api_key=os.environ.get("BIFROST_VIRTUAL_KEY"))
    erp = {
        "erp-stock": McpTool(stock, read_only=True),
        "erp-reorder": McpTool(reorder, destructive=True),
    }
    return ScriptedGateway(erp).gateway().client


async def main() -> None:
    bifrost = gateway()
    tools = await bifrost.tools()
    for found in tools:
        hints = found.annotations.model_dump(exclude_none=True) if found.annotations else {}
        print(f"{found.name} ({found.client}): {hints}")

    if any(t.name == "erp-stock" for t in tools):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "erp-stock", "arguments": json.dumps({"sku": "SKU-1"})},
        }
        turn = await bifrost.execute_tool(call, options=Options(mcp_clients=["erp"]))
        print("erp-stock ->", turn["content"])

    print("your model client's headers:", NO_GATEWAY_TOOLS)
    await bifrost.aclose()


if __name__ == "__main__":
    asyncio.run(main())
