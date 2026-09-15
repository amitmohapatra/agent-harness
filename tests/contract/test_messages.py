"""``AgentRequest``/``AgentResult``: coercion, serializability, A2A readiness (§7, §90)."""

from __future__ import annotations

import json

from universal_agent_harness import (
    AgentExecutionContext,
    AgentRequest,
    AgentResult,
    AgentStatus,
    ArtifactRef,
    Claim,
)


def test_coerce_passes_results_through_and_wraps_everything_else():
    original = AgentResult.ok("x")
    assert AgentResult.coerce(original) is original
    assert AgentResult.coerce({"a": 1}).data == {"a": 1}
    assert AgentResult.coerce(None).status is AgentStatus.SUCCESS


def test_result_is_json_serializable_for_a2a_and_queues():
    result = AgentResult.ok(
        {"total": 3},
        claims=[Claim(claim_id="c1", text="stock is low", evidence_ids=["e1"])],
        artifacts=[ArtifactRef(artifact_id="a1", type="report")],
    )
    payload = json.loads(result.model_dump_json())
    assert payload["status"] == "SUCCESS"
    assert payload["claims"][0]["claim_id"] == "c1"
    assert AgentResult.model_validate(payload).data == {"total": 3}


def test_request_round_trips_with_its_context():
    ctx = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", thread_id="t")
    request = AgentRequest.create(ctx, {"query": "how much stock?"}, objective="check stock")
    payload = json.loads(request.model_dump_json())
    restored = AgentRequest.model_validate(payload)
    assert restored.context == ctx
    assert restored.query == "check stock"


def test_query_extraction_prefers_objective_then_common_input_keys():
    ctx = AgentExecutionContext.create(tenant_id="acme")
    assert AgentRequest.create(ctx, "plain text").query == "plain text"
    assert AgentRequest.create(ctx, {"question": "q?"}).query == "q?"
    assert AgentRequest.create(ctx, {"other": 1}).query is None
    assert AgentRequest.create(ctx, {"question": "q?"}, objective="o").query == "o"


def test_status_ok_semantics():
    assert AgentStatus.SUCCESS.ok and AgentStatus.PARTIAL.ok
    assert not AgentStatus.ERROR.ok and not AgentStatus.TIMEOUT.ok


def test_add_warning_returns_a_new_result():
    base = AgentResult.ok("x")
    warned = base.add_warning("W", "careful")
    assert not base.warnings and warned.warnings[0].code == "W"
