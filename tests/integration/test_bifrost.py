"""The Bifrost gateway client, against a real HTTP server.

Every test here drives actual sockets: the retry budget, the circuit breaker and the SSE
stream are the three things a mocked transport would let pass while broken.
"""

from __future__ import annotations

import pytest
from tests.support_gateway import FakeGateway, completion
from universal_agent_contracts.errors import ModelError
from universal_agent_contracts.tool import ToolSpec

from universal_agent_harness import AgentHarness, BifrostModelClient, ModelRequest, tool_schemas


async def test_a_plain_completion_is_normalized() -> None:
    with FakeGateway([completion("reorder 47 units")]) as gateway:
        client = BifrostModelClient(gateway.url, model="gpt-4o-mini", api_key="vk-test")
        response = await client.invoke("how much stock?")
        await client.aclose()

    assert response.text == "reorder 47 units"
    assert response.provider == "bifrost"
    assert response.usage is not None and response.usage.total_tokens == 18
    assert response.finish_reason == "stop"
    assert response.latency_ms is not None
    sent = gateway.requests[0]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"] == [{"role": "user", "content": "how much stock?"}]


async def test_tool_calls_come_back_intact() -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "inventory_db", "arguments": '{"sku": "SKU-1"}'},
    }
    with FakeGateway([completion("", tool_calls=[call])]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        spec = ToolSpec(
            name="inventory_db",
            description="Stock levels for a SKU.",
            input_schema={"type": "object", "properties": {"sku": {"type": "string"}}},
        )
        response = await client.invoke(
            ModelRequest(
                messages=[{"role": "user", "content": "stock?"}], tools=tool_schemas([spec])
            )
        )
        await client.aclose()

    assert response.tool_calls == [call]
    sent_tool = gateway.requests[0]["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["function"]["name"] == "inventory_db"
    assert sent_tool["function"]["parameters"]["properties"]["sku"]["type"] == "string"


async def test_a_retryable_status_is_retried_and_a_client_error_is_not() -> None:
    with FakeGateway([(503, {"error": "upstream"}), completion("second time")]) as gateway:
        client = BifrostModelClient(gateway.url, model="m", max_retries=2, backoff_seconds=0.01)
        assert (await client.invoke("q")).text == "second time"
        await client.aclose()
    assert gateway.call_count == 2, "the 503 must be retried"

    with FakeGateway([(400, {"error": "bad request"})]) as gateway:
        client = BifrostModelClient(gateway.url, model="m", max_retries=3, backoff_seconds=0.01)
        with pytest.raises(ModelError, match="400"):
            await client.invoke("q")
        await client.aclose()
    assert gateway.call_count == 1, "a 400 is a bug in the request; retrying only burns budget"


async def test_the_circuit_opens_after_repeated_failures() -> None:
    """Without this, every request during an outage pays a full timeout."""
    with FakeGateway([(500, {"error": "down"})]) as gateway:
        client = BifrostModelClient(
            gateway.url,
            model="m",
            max_retries=0,
            backoff_seconds=0.01,
            circuit_failure_threshold=3,
            circuit_open_seconds=30.0,
        )
        for _ in range(3):
            with pytest.raises(ModelError):
                await client.invoke("q")
        assert gateway.call_count == 3

        with pytest.raises(ModelError, match="circuit open"):
            await client.invoke("q")
        await client.aclose()

    assert gateway.call_count == 3, "the fourth call must not reach the gateway at all"


async def test_streaming_yields_deltas_as_they_arrive() -> None:
    with FakeGateway([completion("reorder 47 units now")]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        chunks = [chunk async for chunk in client.stream("q")]
        await client.aclose()

    assert chunks == ["reorder", "47", "units", "now"]
    assert gateway.requests[0]["stream"] is True


async def test_structured_output_is_parsed_and_repaired_once() -> None:
    schema = {
        "type": "object",
        "properties": {"sku": {"type": "string"}, "shortfall": {"type": "integer"}},
        "required": ["sku", "shortfall"],
    }
    with FakeGateway(
        [completion("not json at all"), completion('{"sku":"SKU-1","shortfall":47}')]
    ) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        response = await client.structured("q", schema=schema)
        await client.aclose()

    assert response.data == {"sku": "SKU-1", "shortfall": 47}
    assert gateway.call_count == 2, "one repair round, not an unbounded loop"
    repair = gateway.requests[1]
    assert repair["response_format"]["json_schema"]["schema"] == schema
    assert repair["messages"][-1]["role"] == "user", "the failure is fed back as a turn"


async def test_the_harness_instruments_a_gateway_call(context) -> None:
    """Wired into the harness, the call is a model call like any other: traced and counted."""
    with FakeGateway([completion("answered")]) as gateway:
        client = BifrostModelClient(gateway.url, model="gpt-4o-mini")
        harness = AgentHarness(model=client, defaults={"tenant_id": "acme"})

        async def agent(payload, runtime):
            reply = await runtime.model.invoke("say something")
            return reply.text

        result = await harness.wrap(agent, agent_id="chat")("hi", context=context)
        assert result.data == "answered"
        # the harness recorded the call for the evaluation event and the result metrics
        assert result.metrics["model_calls"] == 1
        await harness.aclose()
        await client.aclose()
