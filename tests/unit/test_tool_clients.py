"""MCP tools through the gateway, several clients behind one port, and memory as tools."""

from __future__ import annotations

import json
from typing import Any

import pytest
from trellis.contracts import ToolCall, ToolStatus
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.errors import ToolError

from trellis.harness.tools import LocalToolClient
from trellis.harness.tools.composite import CompositeToolClient
from trellis.harness.tools.mcp import MCPToolClient
from trellis.harness.tools.memory_tools import RECALL, REMEMBER, MemoryToolClient


class _MCP:
    """``bf.mcp`` as Bifrost answers it: one entry per server with its ``config`` and its
    tools already named ``<server>-<tool>`` in the chat-completions function shape."""

    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    async def clients(self, **filters: Any) -> list[dict[str, Any]]:
        return [
            {
                "config": {"name": "billing", "client_id": "c1"},
                "state": "connected",
                "tools": [
                    {
                        "name": "billing-refund",
                        "description": "Refund an order",
                        "parameters": {"type": "object"},
                    },
                    {"name": "billing-lookup", "description": "Find an order"},
                ],
            },
            {
                "config": {"name": "crm"},
                "state": "connected",
                "tools": [{"name": "crm-contact", "description": "Look up a contact"}],
            },
        ]

    async def execute(self, tool_call: dict[str, Any], **options: Any) -> dict[str, Any]:
        self.executed.append(tool_call)
        name = tool_call["function"]["name"]
        if name == "billing-refund":
            return {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps({"refunded": True}),
            }
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps({"ok": False}),
            "is_error": True,
        }


class _Gateway:
    def __init__(self) -> None:
        self.mcp = _MCP()


async def test_mcp_tools_are_listed_as_the_gateway_names_them_and_executed_through_it() -> None:
    gateway = _Gateway()
    client = MCPToolClient(gateway)
    specs = await client.list_tools()
    assert [s.name for s in specs] == ["billing-refund", "billing-lookup", "crm-contact"]
    assert [s.server for s in specs] == ["billing", "billing", "crm"]
    assert all(s.source == "mcp" for s in specs)
    assert specs[0].input_schema == {"type": "object"} and client.spec("crm-contact").description
    outcome = await client.call("billing-refund", order_id=91)
    assert outcome.status is ToolStatus.OK and outcome.output == {"refunded": True}
    sent = gateway.mcp.executed[-1]
    assert sent["function"] == {"name": "billing-refund", "arguments": '{"order_id": 91}'}
    failed = await client.call(ToolCall(tool="crm-contact", idempotency_key="k9"))
    # ``is_error`` is the verdict, whatever the content says
    assert failed.status is ToolStatus.ERROR and failed.error_class == "MCPToolError"
    assert failed.output == {"ok": False} and gateway.mcp.executed[-1]["id"] == "k9"
    only_crm = MCPToolClient(gateway, clients=["crm"])
    assert [s.name for s in await only_crm.list_tools()] == ["crm-contact"]


async def test_the_mcp_client_reuses_a_model_clients_gateway() -> None:
    """Inference and tool execution share one virtual key, one retry policy, one breaker."""

    class _ModelClient:
        def __init__(self, gateway: Any) -> None:
            self.gateway = gateway

    gateway = _Gateway()
    names = [s.name for s in await MCPToolClient(_ModelClient(gateway)).list_tools()]
    assert names == ["billing-refund", "billing-lookup", "crm-contact"]
    with pytest.raises(TypeError, match="Bifrost client"):
        MCPToolClient(object())


async def test_a_gateway_failure_is_a_tool_error() -> None:
    class Down(_MCP):
        async def execute(self, tool_call, **options):
            raise RuntimeError("gateway unreachable")

    gateway = _Gateway()
    gateway.mcp = Down()
    with pytest.raises(ToolError, match="unreachable"):
        await MCPToolClient(gateway).call("billing__refund")


async def test_the_composite_routes_to_the_client_that_listed_the_tool() -> None:
    local = LocalToolClient()
    local.register(lambda x: x * 2, name="double")
    local.register(lambda: "local wins", name="billing-refund")
    composite = CompositeToolClient([local, MCPToolClient(_Gateway())])
    names = [s.name for s in await composite.list_tools()]
    assert names == ["double", "billing-refund", "billing-lookup", "crm-contact"]
    assert (await composite.call("double", x=4)).output == 8
    assert (await composite.call("billing-refund")).output == "local wins"  # first client wins
    assert composite.spec("double") is not None
    with pytest.raises(ToolError, match="unknown tool"):
        await composite.call("nope")
    local.register(lambda: "late", name="late")  # registered after the first listing
    assert (await composite.call("late")).output == "late"  # found on the re-listing
    with pytest.raises(ToolError, match="unknown tool"):
        await composite.call("never")


class _Memory:
    enabled = True

    def __init__(self) -> None:
        self.observed: list[MemoryObservation] = []

    async def recall(self, query: str, /, **options: Any) -> list[Any]:
        return [{"memory_id": "mem_1", "content": f"about {query}", "score": 0.9, "extra": 1}]

    async def observe(self, observation: MemoryObservation, /) -> Any:
        self.observed.append(observation)
        return {"observation_id": "obs_1"}


class _Runtime:
    def __init__(self, memory: Any) -> None:
        self.memory = memory


async def test_memory_tools_read_and_write_through_the_runtime() -> None:
    client = MemoryToolClient()
    assert await client.list_tools() == []  # nothing until a runtime with memory is attached
    memory = _Memory()
    client.attach(_Runtime(memory))
    assert [s.name for s in await client.list_tools()] == [RECALL, REMEMBER]
    hits = await client.call(RECALL, query="timezone", limit=3)
    assert hits.status is ToolStatus.OK
    assert hits.output == [{"memory_id": "mem_1", "content": "about timezone", "score": 0.9}]
    stored = await client.call(REMEMBER, content="Prefers metric units", kind="preference")
    assert stored.status is ToolStatus.OK and stored.output == {"remembered": True}
    assert memory.observed[0].kind == "AGENT_RESULT"
    assert memory.observed[0].hints == {"memory_type": "PREFERENCE"}
    assert (await client.call("memory.nope")).status is ToolStatus.ERROR
    memory.enabled = False
    assert await client.list_tools() == []
    assert (await client.call(RECALL, query="x")).error_class == "MemoryDisabled"
