"""What the harness actually puts on the wire, checked against the running service.

The other suites assert on the SDK calls the harness makes. This one goes a layer lower: it
wraps the SDK's HTTP transport in a tap that forwards every request to the **real** Memory
Service and keeps the bytes and headers that crossed the socket. So the payload shapes,
scope fields and idempotency headers asserted here are the ones the service received — and
the responses are the ones it really sent.

Nothing is simulated. The failure tests point the same SDK at a port nothing listens on.
"""

from __future__ import annotations

import json

import httpx
import pytest
from universal_memory import MemoryClient

from universal_agent_harness import AgentHarness, AgentResult, MemoryObservation
from tests.support import DEAD_SERVICE_URL, MEMORY_API_KEY, MEMORY_SERVICE_URL, span_by_name


class WireTap(httpx.AsyncBaseTransport):
    """A real HTTP transport that keeps a copy of each exchange as it passes through."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.exchanges: list[tuple[httpx.Request, bytes, httpx.Response]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        response = await self._inner.handle_async_request(request)
        await response.aread()  # buffer it here; httpx serves the same bytes downstream
        self.exchanges.append((request, body, response))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()

    def to(self, path: str) -> list[tuple[httpx.Request, bytes, httpx.Response]]:
        return [e for e in self.exchanges if e[0].url.path == path]

    def sent_to(self, path: str) -> list[dict]:
        return [json.loads(body) for _, body, _ in self.to(path)]


@pytest.fixture
def wire():
    return WireTap()


@pytest.fixture
async def tapped_client(wire):
    client = MemoryClient(
        MEMORY_SERVICE_URL,
        api_key=MEMORY_API_KEY,
        timeout=120.0,
        http_client=httpx.AsyncClient(
            transport=wire, timeout=120.0, base_url=MEMORY_SERVICE_URL
        ),
    )
    yield client
    await client.aclose()


async def test_full_turn_puts_the_documented_requests_on_the_wire(tapped_client, wire, context, spans):
    harness = AgentHarness(
        memory=tapped_client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 120}},
    )

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def inventory_agent(state, agent):
        bundle = agent.memory_context
        assert bundle is not None, "the service answered /v1/context"
        assert bundle.evidence.status in ("COMPLETE", "INCOMPLETE", "INSUFFICIENT")
        return AgentResult.ok(
            "reorder 50 units",
            memory_observations=[MemoryObservation(content="SKU-1 was reordered")],
        )

    result = await inventory_agent({"question": "how much stock?"}, context=context)
    assert result.data == "reorder 50 units"

    # -- /v1/context: one request, carrying the scope the service indexes on
    contexts = wire.sent_to("/v1/context")
    assert len(contexts) == 1
    payload = contexts[0]
    assert payload["query"] == "how much stock?"
    scope = payload["scope"]
    assert scope["tenant_id"] == "acme"
    assert scope["agent_id"] == "inventory-agent"
    assert scope["thread_id"] == context.thread_id
    # The fixture context belongs to another agent, so this run is a *child* of it: the
    # lineage is recorded and the trace identity is inherited.
    assert scope["parent_agent_run_id"] == context.agent_run_id
    assert scope["agent_run_id"] != context.agent_run_id
    assert scope["trace_id"] == context.trace_id
    assert all(e[2].status_code == 200 for e in wire.to("/v1/context"))

    # -- /v1/observations: input, output, explicit observation; each separately replayable
    observations = wire.to("/v1/observations")
    assert len(observations) == 3
    keys = {request.headers.get("idempotency-key") for request, _, _ in observations}
    assert None not in keys and len(keys) == 3
    assert all(response.status_code < 300 for _, _, response in observations)
    contents = {json.loads(body)["content"] for _, body, _ in observations}
    assert "SKU-1 was reordered" in contents

    # -- and the span carries what the service actually returned
    retrieval = span_by_name(spans, "agent.memory.retrieve")
    assert retrieval.attributes["memory.evidence.status"] == payload_status(wire)
    assert isinstance(retrieval.attributes["memory.token_estimate"], int)


def payload_status(wire: WireTap) -> str:
    return json.loads(wire.to("/v1/context")[0][2].text)["evidence"]["status"]


async def test_a_real_connection_failure_is_classified_as_a_memory_error(context):
    client = MemoryClient(DEAD_SERVICE_URL, api_key=MEMORY_API_KEY, timeout=3.0, max_retries=0)
    harness = AgentHarness(
        memory=client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False, "failure_mode": "fail_closed"},
                "timeouts": {"memory_seconds": 5}},
    )

    @harness.agent(agent_id="inv", error_mode="result")
    async def agent(state, runtime):
        return "unreachable"

    result = await agent({"question": "q"}, context=context)
    assert result.status == "ERROR"
    assert result.error.category == "MEMORY"
    await harness.aclose()


async def test_a_real_outage_still_produces_a_result(context):
    client = MemoryClient(DEAD_SERVICE_URL, api_key=MEMORY_API_KEY, timeout=3.0, max_retries=0)
    harness = AgentHarness(
        memory=client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 5}},
    )

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        assert runtime.memory_context is None
        return "answered without memory"

    result = await agent({"question": "q"}, context=context)
    assert result.data == "answered without memory"
    assert any(w.code == "MEMORY_DEGRADED" for w in result.warnings)
    await harness.aclose()


async def test_agent_scope_is_a_real_sdk_scope(context):
    """The harness's context must map onto ``universal_memory.Scope`` without loss."""
    from universal_memory.models import Scope

    scope = Scope(**context.for_agent("child-agent").scope_fields())
    assert scope.tenant_id == "acme"
    assert scope.agent_id == "child-agent"
    assert scope.parent_agent_run_id == context.agent_run_id
    assert scope.thread_id == context.thread_id
