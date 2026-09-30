from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from trellis import mcp, openapi, tool
from trellis.contracts import ConfigurationError
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.tools.sources import as_source


class Services:
    def __init__(self, gateway: Any = None, effects: dict[str, str] | None = None) -> None:
        self._gateway = gateway
        self.effects = effects or {}

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            raise ConfigurationError("no gateway")
        return self._gateway

    async def side_effects(self, names: list[str]) -> dict[str, str]:
        return {n: self.effects[n] for n in names if n in self.effects}


class Def:
    def __init__(self, name: str, client: str, *, code_mode: bool = True) -> None:
        self.name, self.client, self.description, self.parameters = name, client, f"{name} tool", {}
        self.code_mode = code_mode


class Gateway:
    def __init__(self, defs: list[Def]) -> None:
        self.defs = defs
        self.asked: list[tuple[Any, Any]] = []
        self.executed: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []

    async def tools(self, servers: Any, only: Any) -> list[Def]:
        self.asked.append((tuple(servers), only))
        return self.defs

    async def execute(
        self, name: str, args: dict[str, Any], *, clients: Any, parent_request_id: Any = None
    ) -> Any:
        self.executed.append((name, args, tuple(clients)))
        return {"ok": name}


async def test_a_function_becomes_a_validated_tool_and_stays_callable() -> None:
    @tool(side_effects="irreversible")
    async def refund(order: str, amount: float = 1.0) -> str:
        """Refund an order.

        More detail nobody needs in a tool list."""
        return f"{order}:{amount}"

    [resolved] = await refund.resolve(Services())
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


async def test_mcp_tools_come_from_the_gateway_with_their_catalogued_side_effects() -> None:
    gateway = Gateway([Def("erp-get_stock", "erp"), Def("erp-create_po", "erp")])
    source = mcp("erp", only=["get_stock", "erp-create_po"])
    tools = await source.resolve(Services(gateway, {"erp-get_stock": "read"}))
    assert gateway.asked == [(("erp",), ("erp-create_po", "erp-get_stock"))]
    assert {t.name: t.side_effects for t in tools} == {
        "erp-get_stock": "read",
        "erp-create_po": "write",
    }
    assert await tools[0].run({"sku": "a"}) == {"ok": "erp-get_stock"}
    assert gateway.executed == [("erp-get_stock", {"sku": "a"}, ("erp",))]


async def test_a_large_read_only_source_goes_to_code_mode() -> None:
    defs = [Def(f"wiki-t{i}", "wiki") for i in range(20)]
    effects = {d.name: "read" for d in defs}
    tools = await mcp("wiki").resolve(Services(Gateway(defs), effects))
    assert [t.name for t in tools] == [s.name for s in CODE_MODE_TOOLS]
    assert all(t.code_mode for t in tools)


async def test_a_server_that_is_no_code_mode_client_keeps_its_source_in_normal_mode() -> None:
    # a script only sees the gateway's Code Mode clients
    defs = [Def(f"wiki-t{i}", "wiki", code_mode=False) for i in range(20)]
    tools = await mcp("wiki").resolve(Services(Gateway(defs), {d.name: "read" for d in defs}))
    assert not any(t.code_mode for t in tools) and len(tools) == 20


async def test_one_write_tool_keeps_a_large_source_in_normal_mode() -> None:
    defs = [Def(f"a-t{i}", "a") for i in range(3)]
    effects = {d.name: "read" for d in defs} | {"a-t0": "write"}
    tools = await mcp("a", "b", "c").resolve(Services(Gateway(defs), effects))
    assert not any(t.code_mode for t in tools)
    assert len(tools) == 3


async def test_three_read_only_servers_go_to_code_mode_and_scripts_are_scoped_to_them() -> None:
    defs = [Def("a-x", "a"), Def("b-y", "b"), Def("c-z", "c")]
    gateway = Gateway(defs)
    tools = await mcp("a", "b", "c").resolve(Services(gateway, {d.name: "read" for d in defs}))
    execute = next(t for t in tools if t.name == "executeToolCode")
    await execute.run({"code": "print(1)"})
    assert gateway.executed == [("executeToolCode", {"code": "print(1)"}, ("a", "b", "c"))]


async def test_mcp_needs_a_gateway() -> None:
    with pytest.raises(ConfigurationError):
        await mcp("erp").resolve(Services())


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
    tools = {t.name: t for t in await openapi(DOCUMENT).resolve(Services())}
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
    tools = await openapi("http://erp.test/openapi.json", only=["get_order"]).resolve(Services())
    assert [t.name for t in tools] == ["get_order"]
