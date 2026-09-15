"""End-to-end against the real ``universal-memory`` SDK with the HTTP layer mocked.

The fake in ``tests/support.py`` keeps the unit tests fast, but the contract that actually
ships is the SDK's: real ``MemoryClient.bind``, real ``Scope`` validation, real request
payloads. This test asserts the harness produces requests the Memory Service would accept —
scope fields, idempotency headers, and the ``/v1/context`` + ``/v1/observations`` calls of
the documented turn flow.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from universal_memory import MemoryClient

from universal_agent_harness import AgentHarness, AgentResult, MemoryObservation

BASE_URL = "http://memory-service.test"

BUNDLE = {
    "query": "how much stock?",
    "query_type": "FACTUAL",
    "bundle_id": "bundle-1",
    "conversation": {"thread_id": "chat-1", "message_ids": [], "rendered": ""},
    "memories": [
        {
            "item_id": "m1",
            "representation": "MEMORY",
            "text": "SKU-1 reorder point is 50",
            "score": 0.8,
            "citation": "[1]",
            "evidence": [],
            "attributes": {},
        }
    ],
    "knowledge": [],
    "graph_facts": [],
    "summaries": [],
    "evidence": {"status": "COMPLETE", "required_groups": [], "satisfied_groups": [],
                 "missing_groups": [], "escalations": [], "notes": [], "unused": []},
    "token_budget": 2000,
    "token_estimate": 120,
    "rendered": "SKU-1 reorder point is 50",
    "cache_hit": False,
}


@pytest.fixture
def sdk_client():
    return MemoryClient(BASE_URL, api_key="test-key")


@respx.mock
async def test_full_turn_against_the_real_sdk(sdk_client, context, spans):
    context_route = respx.post(f"{BASE_URL}/v1/context").mock(
        return_value=httpx.Response(200, json=BUNDLE)
    )
    observations_route = respx.post(f"{BASE_URL}/v1/observations").mock(
        return_value=httpx.Response(200, json={"observation_id": "obs-1", "job_ids": ["j1"]})
    )

    harness = AgentHarness(
        memory=sdk_client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}},
    )

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def inventory_agent(state, agent):
        bundle = agent.memory_context
        assert bundle is not None
        assert bundle.memories[0].text == "SKU-1 reorder point is 50"
        assert bundle.evidence.status == "COMPLETE"
        return AgentResult.ok(
            "reorder 50 units",
            memory_observations=[MemoryObservation(content="SKU-1 was reordered")],
        )

    result = await inventory_agent({"question": "how much stock?"}, context=context)
    assert result.data == "reorder 50 units"

    assert context_route.called
    payload = json.loads(context_route.calls[0].request.content)
    assert payload["query"] == "how much stock?"
    assert payload["scope"]["tenant_id"] == "acme"
    assert payload["scope"]["agent_id"] == "inventory-agent"
    assert payload["scope"]["thread_id"] == "chat-1"
    # The fixture context belongs to another agent, so this run is a *child* of it: the
    # lineage is recorded and the trace identity is inherited.
    assert payload["scope"]["parent_agent_run_id"] == context.agent_run_id
    assert payload["scope"]["agent_run_id"] != context.agent_run_id
    assert payload["scope"]["trace_id"] == context.trace_id

    assert observations_route.call_count == 3  # input, output, explicit observation
    keys = {call.request.headers.get("idempotency-key") for call in observations_route.calls}
    assert None not in keys and len(keys) == 3

    from tests.support import span_by_name

    retrieval = span_by_name(spans, "agent.memory.retrieve")
    assert retrieval.attributes["memory.evidence.status"] == "COMPLETE"
    assert retrieval.attributes["memory.token_estimate"] == 120
    assert retrieval.attributes["memory.item_count"] == 1


@respx.mock
async def test_sdk_errors_are_classified_by_the_harness(sdk_client, context):
    respx.post(f"{BASE_URL}/v1/context").mock(return_value=httpx.Response(503, json={"detail": "down"}))

    harness = AgentHarness(
        memory=sdk_client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False, "failure_mode": "fail_closed"}},
    )

    @harness.agent(agent_id="inv", error_mode="result")
    async def agent(state, runtime):
        return "unreachable"

    result = await agent({"question": "q"}, context=context)
    assert result.status == "ERROR"
    assert result.error.category == "MEMORY"


@respx.mock
async def test_degraded_memory_still_produces_a_result(sdk_client, context):
    respx.post(f"{BASE_URL}/v1/context").mock(side_effect=httpx.ConnectError("no route"))
    respx.post(f"{BASE_URL}/v1/observations").mock(
        return_value=httpx.Response(200, json={"observation_id": "o", "job_ids": []})
    )

    harness = AgentHarness(
        memory=sdk_client, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        assert runtime.memory_context is None
        return "answered without memory"

    result = await agent({"question": "q"}, context=context)
    assert result.data == "answered without memory"
    assert any(w.code == "MEMORY_DEGRADED" for w in result.warnings)


@respx.mock
async def test_agent_scope_is_a_real_sdk_scope(sdk_client, context):
    """The harness's context must map onto ``universal_memory.Scope`` without loss."""
    from universal_memory.models import Scope

    scope = Scope(**context.for_agent("child-agent").scope_fields())
    assert scope.tenant_id == "acme"
    assert scope.agent_id == "child-agent"
    assert scope.parent_agent_run_id == context.agent_run_id
    assert scope.thread_id == "chat-1"
