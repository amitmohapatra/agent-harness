"""The gateway and memory client modules: the only places the harness calls those services."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx

from tests.support.memory import AGENT_TOOLS, FakeMemoryService
from trellis.contracts import ToolCall, ToolError, ToolOutcome
from trellis.harness import runtime as runtime_module
from trellis.harness.clients import bifrost
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import READ_ONLY_TOOLS, Memory
from trellis.harness.identity import Identity

GATEWAY = "http://gw.test"


@respx.mock
async def test_the_gateway_lists_what_the_virtual_key_allows_by_execution_name() -> None:
    """With admin auth on, ``/api`` refuses a virtual key: the toolbox is the gateway's ``/mcp``
    ``tools/list`` asked with the key — Code Mode clients through their meta-tools."""
    admin = respx.get(f"{GATEWAY}/api/mcp/clients").mock(return_value=httpx.Response(401))

    def rpc(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "tools/list":
            tools = [
                {"name": "erp-get_stock", "annotations": {"readOnlyHint": True}},
                {"name": "listToolFiles", "annotations": {}},
                {"name": "executeToolCode", "annotations": {}},
            ]
            result: dict[str, object] = {"tools": tools}
        elif body["params"]["name"] == "listToolFiles":
            result = {"content": [{"type": "text", "text": "servers/\n  wiki.pyi"}]}
        else:
            text = "def read(repo: str) -> dict:  # Read a wiki.\n"
            result = {"content": [{"type": "text", "text": text}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

    listing = respx.post(f"{GATEWAY}/mcp").mock(side_effect=rpc)
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    stock, read = await gateway.tools()
    assert stock.name == "erp-get_stock" and not stock.code_mode
    assert stock.annotations is not None and stock.annotations.read_only_hint is True
    assert (read.name, read.client, read.code_mode) == ("wiki-read", "wiki", True)
    assert all(c.request.headers["authorization"] == "Bearer vk" for c in listing.calls)
    assert not admin.called
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
async def test_in_a_run_a_request_waits_what_is_left_of_the_call_and_carries_its_key() -> None:
    executed = respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        return_value=httpx.Response(200, json={"role": "tool", "content": "ok"})
    )
    message = {"role": "assistant", "content": "hi"}
    completed = respx.post(f"{GATEWAY}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": message}]})
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    run = SimpleNamespace(
        run_id="run_1", idempotency_key="run_1:k:0", remaining=lambda: 12.5, tenant="t", user="u"
    )
    token = runtime_module._current.set(run)  # type: ignore[arg-type]
    try:
        assert await gateway.execute("erp-ship", {}, clients=["erp"]) == "ok"
        await gateway.complete([{"role": "user", "content": "hi"}], model="m")
    finally:
        runtime_module._current.reset(token)
    request = executed.calls[0].request
    assert json.loads(request.content)["id"] == "run_1:k:0"
    assert request.extensions["timeout"]["read"] == 12.5
    assert completed.calls[0].request.extensions["timeout"]["read"] == 12.5
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


async def test_the_key_says_who_the_deployment_is() -> None:
    service = FakeMemoryService(tenant="acme")
    key = await memory(service).key()
    assert (key.tenant_id, key.principal, key.role) == ("acme", "svc:harness", "service")


async def test_a_run_memory_is_bound_to_the_run_scope() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    pushed = await run.context("q", tools=["erp-get_stock"], window=False)
    assert pushed.rendered.startswith(service.context_text)
    assert pushed.bundle_id is not None and pushed.bundle_id.startswith("bnd_")
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
    assert call.body["token_budget"] == 2000 and call.body["window"] is False


async def test_agent_tools_are_listed_once_and_tiered_read_or_write() -> None:
    service = FakeMemoryService()
    mem = memory(service)
    listed = await mem.bind(identity()).agent_tools()
    again = await mem.bind(identity()).agent_tools()
    assert [t.name for t in listed] == [t["name"] for t in AGENT_TOOLS] == [t.name for t in again]
    assert {t.name for t in listed if t.side_effects == "read"} <= READ_ONLY_TOOLS
    writes = {"memory_remember", "memory_update", "memory_forget", "profile_edit"}
    assert {t.name for t in listed if t.side_effects == "write"} == writes
    search = next(t for t in listed if t.name == "memory_search")
    assert (search.input_schema or {})["required"] == ["query"]  # the service's own schema
    assert len(service.named("agent_tools")) == 1
    # the SDK returns the tool's result itself
    found: Any = await mem.bind(identity()).call_agent_tool("memory_search", {"query": "x"})
    assert [item["kind"] for item in found] == ["memory", "chunk"] and found[0]["id"] == "mem_1"


async def test_records_carry_idempotency() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    await run.record_messages([("user", "hi"), ("assistant", "hello")], "run_1", 2)
    [batch] = service.named("messages")
    # each message names itself, so a re-recorded attempt stores it once
    assert [(m["role"], m["source_message_id"]) for m in batch.body["messages"]] == [
        ("USER", "run_1:user:0"),
        ("ASSISTANT", "run_1:2:msg:1"),
    ]
    assert {m["source_system"] for m in batch.body["messages"]} == {"trellis-harness"}
    await run.record_tool(
        ToolCall(tool="t", args={"a": 1}, task="q", step=1), ToolOutcome(tool="t", output=2)
    )
    assert service.named("record_tool")[0].body["status"] == "ok"
    await run.run_feedback("confirm", source="system", key="run_1:outcome")
    [feedback] = service.named("feedback")
    assert feedback.body["target_kind"] == "run" and feedback.body["target_id"] == "run_1"
    assert feedback.body["source"] == "system" and feedback.idempotency_key == "run_1:outcome"


async def test_the_model_key_is_registered_once_in_the_agents_scope() -> None:
    service = FakeMemoryService()
    await memory(service).scoped("t", "a").register_model_key("sk")
    [key] = service.named("model_key")
    assert key.body["virtual_key"] == "sk"
    assert key.idempotency_key is not None and key.idempotency_key.startswith("model-key:a:")
    assert key.scope == {"tenant_id": "t", "agent_id": "a", "custom_metadata": {}}


@respx.mock
async def test_a_tool_result_that_is_not_json_is_returned_as_it_came() -> None:
    respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        side_effect=[
            httpx.Response(200, json={"role": "tool", "content": "plain words"}),
            httpx.Response(200, json={"role": "tool", "content": [{"type": "text", "text": "x"}]}),
        ]
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    assert await gateway.execute("wiki-read", {}, clients=["wiki"]) == "plain words"
    assert await gateway.execute("wiki-read", {}, clients=["wiki"]) == [
        {"type": "text", "text": "x"}
    ]
    await gateway.aclose()


async def test_a_document_can_be_added_without_waiting_for_it() -> None:
    service = FakeMemoryService()
    info = await memory(service).for_user("t", "u").add_document(b"text", wait=None)
    assert info.status == "STAGED"  # its parse job has not run yet
    assert [c.name for c in service.named("document")] == ["document"]  # read once, not polled


async def test_a_spooled_memory_write_replays_in_its_own_scope() -> None:
    """Each write the harness spools (its ``record``) is the same request again, in the scope
    it was made in, when an empty process replays it."""
    from trellis.contracts import (
        AgentExecutionContext,
        Interrupt,
        InterruptDecision,
        InterruptReason,
        InterruptResolution,
    )

    service = FakeMemoryService()
    memory = Memory("http://memory.test", "key", client=service.client())
    scope = Identity(tenant="acme", user="ada", agent_id="a", run_id="run_1", thread="thr")
    run = memory.bind(scope)
    call, outcome = ToolCall(tool="erp-get", args={"sku": "1"}), ToolOutcome(tool="erp-get")
    asked = Interrupt(
        interrupt_id="run_1.1.1",
        tenant_id="acme",
        run_id="run_1",
        question="Refund?",
        reason=InterruptReason.APPROVAL,
        tool_call=ToolCall(tool="refund", args={"amount": 5}),
    )
    decision = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.APPROVE, reviewer="cfo"
    ).to_feedback(
        asked, AgentExecutionContext.create(tenant_id="acme", agent_id="a", agent_run_id="run_1")
    )
    assert decision is not None
    records = [
        run.record("record_messages", messages=[["user", "hi"]], run_id="run_1", attempt=1),
        run.record(
            "record_tool",
            call=call.model_dump(mode="json"),
            outcome=outcome.model_dump(mode="json"),
        ),
        run.record(
            "run_feedback", verdict="confirm", source="system", comment=None, key="run_1:outcome"
        ),
        run.record("feedback", record=decision.model_dump(mode="json")),
        run.record(
            "publish_catalog", entries=[{"name": "erp-get", "description": "", "source": "local"}]
        ),
    ]
    for record in json.loads(json.dumps(records)):  # through the spool's JSON and back
        work = memory.replay(record)
        assert work is not None
        await work()
    assert [c.name for c in service.calls] == [
        "messages",
        "record_tool",
        "feedback",
        "feedback",
        "put_catalog",
    ]
    assert all(c.scope.get("tenant_id") == "acme" for c in service.calls)
    assert service.named("messages")[0].body["messages"][0]["source_message_id"] == "run_1:user:0"
    assert memory.replay({"op": "unknown", "scope": {}, "args": {}}) is None
    await memory.aclose()
