"""The gateway and memory client modules: the only places the harness calls those services."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from tests.support.memory import AGENT_TOOLS, FakeMemoryService
from trellis.contracts import ToolCall, ToolError, ToolOutcome, ToolSpec
from trellis.harness.clients import bifrost
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import READ_ONLY_TOOLS, Memory
from trellis.harness.identity import Identity

GATEWAY = "http://gw.test"

CLIENTS = {
    "clients": [
        {
            "config": {
                "client_id": "c1",
                "name": "erp",
                "connection_type": "http",
                "connection_string": {"value": "http://erp", "type": "plain_text"},
                "tools_to_execute": ["*"],
                "is_code_mode_client": False,
            },
            "tools": [
                {
                    "name": "get_stock",
                    "description": "Stock of a SKU.",
                    "parameters": {"type": "object"},
                },
                {
                    "name": "create_po",
                    "description": "Create a PO.",
                    "parameters": {"type": "object"},
                },
            ],
            "state": "healthy",
        }
    ]
}


@respx.mock
async def test_the_gateway_lists_scoped_tools_by_their_execution_name() -> None:
    respx.get(f"{GATEWAY}/api/mcp/clients").mock(return_value=httpx.Response(200, json=CLIENTS))
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    tools = await gateway.tools(["erp"], ["erp-get_stock"])
    assert [t.name for t in tools] == ["erp-get_stock"]
    await gateway.aclose()


@respx.mock
async def test_a_call_is_scoped_to_its_server_and_a_failed_tool_raises() -> None:
    route = respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        side_effect=[
            httpx.Response(
                200, json={"role": "tool", "content": '{"units": 7}', "tool_call_id": "c"}
            ),
            httpx.Response(200, json={"role": "tool", "content": "no such sku", "is_error": True}),
        ]
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    assert await gateway.execute(
        "erp-get_stock", {"sku": "a"}, clients=["erp"], parent_request_id="run_1"
    ) == {"units": 7}
    request = route.calls[0].request
    assert request.headers["x-bf-mcp-include-clients"] == "erp"
    assert request.headers["x-bf-parent-request-id"] == "run_1"
    assert json.loads(request.content)["function"] == {
        "name": "erp-get_stock",
        "arguments": '{"sku": "a"}',
    }
    with pytest.raises(ToolError, match="no such sku"):
        await gateway.execute("erp-get_stock", {"sku": "?"}, clients=["erp"])
    await gateway.aclose()


@respx.mock
async def test_code_mode_calls_are_read_back_from_the_log_by_the_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.get(f"{GATEWAY}/api/mcp-logs").mock(
        return_value=httpx.Response(
            200,
            json={
                "logs": [
                    {
                        "id": "l1",
                        "timestamp": "2026-09-30T00:00:00Z",
                        "server_label": "wiki",
                        "tool_name": "search",
                        "status": "success",
                        "llm_request_id": "run_1",
                        "arguments": '{"q": "x"}',
                        "result": "found",
                        "latency": 12,
                    }
                ]
            },
        )
    )
    monkeypatch.setattr(bifrost, "LOG_POLL_SECONDS", 0.0)
    gateway = Gateway(f"{GATEWAY}/v1", None)
    [entry] = await gateway.code_mode_calls("run_1", datetime(2026, 9, 30, tzinfo=UTC))
    assert entry.name == "wiki-search" and entry.arguments == {"q": "x"}
    assert route.calls[0].request.url.params["llm_request_ids"] == "run_1"
    # the log is read again until two reads agree: it is written behind the calls
    assert route.call_count == 2
    await gateway.aclose()


# --------------------------------------------------------------------------- memory
def identity() -> Identity:
    return Identity(tenant="t", user="u", agent_id="a", run_id="run_1", thread="th")


def memory(service: FakeMemoryService) -> Memory:
    return Memory("http://mem", None, client=service.client())


async def test_a_run_memory_is_bound_to_the_run_scope() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    bundle = await run.context("q", tools=["erp-get_stock"])
    assert bundle.rendered == service.context_text
    [call] = service.named("context")
    assert call.scope == {
        "tenant_id": "t",
        "user_id": "u",
        "agent_id": "a",
        "agent_run_id": "run_1",
        "thread_id": "th",
        "custom_metadata": {},
    }
    assert call.body["tools"] == {"available": ["erp-get_stock"], "k": 8}
    assert call.body["token_budget"] == 2000


async def test_agent_tools_are_listed_once_and_a_reader_gets_only_the_read_ones() -> None:
    service = FakeMemoryService()
    mem = memory(service)
    everything = await mem.bind(identity()).agent_tools(read_only=False)
    reads = await mem.bind(identity()).agent_tools(read_only=True)
    assert [t.name for t in everything] == [t["name"] for t in AGENT_TOOLS]
    assert {t.name for t in reads} <= READ_ONLY_TOOLS
    assert len(service.named("agent_tools")) == 1
    # the SDK returns the tool's result itself
    assert await mem.bind(identity()).call_agent_tool("memory_search", {"query": "x"}) == [
        "memory_search ok"
    ]


async def test_records_carry_idempotency_and_the_catalog_says_what_it_knows() -> None:
    service = FakeMemoryService(catalog={"erp-get_stock": "read"})
    run = memory(service).bind(identity())
    await run.record_messages([("user", "hi"), ("assistant", "hello")], "run_1", 2)
    assert [(c.body["role"], c.idempotency_key) for c in service.named("message")] == [
        ("USER", "run_1:user:0"),
        ("ASSISTANT", "run_1:2:msg:1"),
    ]
    await run.record_tool(
        ToolCall(tool="t", args={"a": 1}, task="q", step=1), ToolOutcome(tool="t", output=2)
    )
    assert service.named("record_tool")[0].body["status"] == "ok"
    assert await run.side_effects(["erp-get_stock", "missing"]) == {"erp-get_stock": "read"}
    await run.publish_catalog(
        [
            ToolSpec(name="refund", side_effects="irreversible", source="local"),
            ToolSpec(name="remote", source="a2a"),  # side effects unknown: left to the catalog
        ]
    )
    refund, remote = service.named("put_catalog")[0].body["tools"]
    assert refund["side_effects"] == "irreversible" and "side_effects" not in remote
    await memory(service).scoped("t", "a").register_model_key("sk")
    [key] = service.named("model_key")
    assert key.body["virtual_key"] == "sk"
    assert key.idempotency_key is not None and key.idempotency_key.startswith("model-key:a:")
    assert key.scope == {"tenant_id": "t", "agent_id": "a", "custom_metadata": {}}
