from __future__ import annotations

import httpx
import pytest
import respx

from trellis import a2a, openapi, tool
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS
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
    # one default for every remote tool: an OpenAPI operation as an A2A exchange
    assert {t.timeout for t in tools.values()} == {REMOTE_TIMEOUT_SECONDS} == {a2a("x").timeout}
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


async def test_variadic_parameters_are_not_tool_arguments() -> None:
    def flexible(sku: str, *rest: str, **options: str) -> str:
        return sku

    [resolved] = await tool(flexible).resolve()
    assert resolved.spec.input_schema is not None
    assert set(resolved.spec.input_schema["properties"]) == {"sku"}


async def test_an_openapi_document_must_say_where_its_server_is() -> None:
    with pytest.raises(ValueError, match="names no server; pass base_url="):
        await openapi({"openapi": "3.1.0", "paths": {}}).resolve()


@respx.mock
async def test_only_operations_are_tools_and_an_optional_body_is_optional() -> None:
    document = {
        "openapi": "3.1.0",
        "paths": {
            "/notes": {
                "summary": "not an operation",
                "parameters": [{"name": "shared", "in": "query"}],
                "post": {
                    "operationId": "add_note",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    },
                },
                "get": {"summary": "no operationId: not a tool"},
            }
        },
    }
    respx.post("http://notes.test/notes").mock(return_value=httpx.Response(204))
    source = openapi(document, base_url="http://notes.test", headers={"X-Key": "k"})
    [add] = await source.resolve()
    assert add.name == "add_note" and add.spec.description == ""
    assert add.spec.input_schema is not None and add.spec.input_schema["required"] == []
    assert await add.run({}) is None  # no content, no result
    again = await source.resolve()  # one HTTP client per source, however often it resolves
    assert [t.name for t in again] == ["add_note"]
