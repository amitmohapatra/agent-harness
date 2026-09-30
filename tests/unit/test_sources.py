from __future__ import annotations

import httpx
import pytest
import respx

from trellis import openapi, tool
from trellis.harness.tools.sources import as_source


async def test_a_function_becomes_a_validated_tool_and_stays_callable() -> None:
    @tool(side_effects="irreversible")
    async def refund(order: str, amount: float = 1.0) -> str:
        """Refund an order.

        More detail nobody needs in a tool list."""
        return f"{order}:{amount}"

    [resolved] = await refund.resolve()
    assert resolved.spec.description == "Refund an order."
    assert resolved.spec.side_effects == "irreversible"
    assert resolved.spec.input_schema is not None
    assert resolved.spec.input_schema["required"] == ["order"]
    assert await resolved.run({"order": "o1", "amount": "2"}) == "o1:2.0"
    assert await refund("o2") == "o2:1.0"
    with pytest.raises(ValueError):
        await resolved.run({"order": "o1", "surprise": 1})


def test_a_bare_function_is_a_source_and_anything_else_is_refused() -> None:
    def lookup(sku: str) -> int:
        return 1

    assert as_source(lookup).spec.name == "lookup"  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        as_source(42)  # type: ignore[arg-type]


DOCUMENT = {
    "openapi": "3.1.0",
    "servers": [{"url": "http://erp.test"}],
    "paths": {
        "/orders/{order_id}": {
            "get": {
                "operationId": "get_order",
                "summary": "Read an order.",
                "parameters": [
                    {"name": "order_id", "in": "path", "schema": {"type": "string"}},
                    {"name": "expand", "in": "query", "schema": {"type": "boolean"}},
                ],
            },
            "delete": {
                "operationId": "cancel_order",
                "parameters": [{"name": "order_id", "in": "path"}],
            },
        },
        "/orders": {
            "post": {
                "operationId": "create_order",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
            }
        },
    },
}


@respx.mock
async def test_openapi_operations_are_tools_judged_by_their_method() -> None:
    respx.get("http://erp.test/orders/o1", params={"expand": "true"}).mock(
        return_value=httpx.Response(200, json={"id": "o1"})
    )
    respx.post("http://erp.test/orders").mock(return_value=httpx.Response(201, json={"id": "o2"}))
    tools = {t.name: t for t in await openapi(DOCUMENT).resolve()}
    assert {n: t.side_effects for n, t in tools.items()} == {
        "get_order": "read",
        "cancel_order": "irreversible",
        "create_order": "write",
    }
    schema = tools["get_order"].spec.input_schema
    assert schema is not None and schema["required"] == ["order_id"]
    assert await tools["get_order"].run({"order_id": "o1", "expand": True}) == {"id": "o1"}
    assert await tools["create_order"].run({"body": {"sku": "a"}}) == {"id": "o2"}


@respx.mock
async def test_openapi_reads_a_document_by_url_and_keeps_only_what_was_asked_for() -> None:
    respx.get("http://erp.test/openapi.json").mock(return_value=httpx.Response(200, json=DOCUMENT))
    tools = await openapi("http://erp.test/openapi.json", only=["get_order"]).resolve()
    assert [t.name for t in tools] == ["get_order"]
