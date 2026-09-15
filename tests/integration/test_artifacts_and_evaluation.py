"""Artifacts (§43), claims/evidence (§44), evaluation events (§49), lifecycle hooks (§18)."""

from __future__ import annotations

import pytest
from tests.support import span_names

from universal_agent_harness import (
    AgentHarness,
    AgentResult,
    Claim,
    CollectingEvaluationSink,
    EvidenceRef,
    LifecycleEvent,
    RecommendedAction,
)


async def test_artifacts_created_during_a_run_are_registered_on_the_result(harness, context):
    async def agent(payload, runtime):
        ref = await runtime.artifacts.put("a long report", type="report", mime_type="text/plain")
        assert ref.checksum.startswith("sha256:")
        return AgentResult.ok({"report": ref.artifact_id})

    result = await harness.wrap(agent, agent_id="reporter")(None, context=context)
    assert len(result.artifacts) == 1
    assert result.artifacts[0].type == "report"
    assert result.artifacts[0].size_bytes == len("a long report")


async def test_artifact_ids_are_stable_for_identical_content(harness, context):
    ids: list[str] = []

    async def agent(payload, runtime):
        ref = await runtime.artifacts.put("same bytes", type="report")
        ids.append(ref.artifact_id)
        return "ok"

    wrapped = harness.wrap(agent, agent_id="reporter")
    await wrapped(None, context=context)
    await wrapped(None, context=context)
    assert ids[0] == ids[1]


async def test_artifact_roundtrip_through_the_file_store(memory, context, tmp_path):
    harness = AgentHarness(
        memory=memory, artifacts=str(tmp_path), defaults={"tenant_id": "acme"}
    )

    async def agent(payload, runtime):
        ref = await runtime.artifacts.put(b"binary payload", type="blob")
        assert await runtime.artifacts.get(ref.artifact_id) == b"binary payload"
        assert ref.uri and ref.uri.startswith("file://")
        return "stored"

    await harness.wrap(agent, agent_id="writer")(None, context=context)
    assert list(tmp_path.iterdir())


async def test_oversized_results_are_offloaded_to_an_artifact(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"artifacts": {"inline_max_bytes": 100}, "memory": {"writeback": False}},
    )

    async def agent(payload):
        return "x" * 500

    result = await harness.wrap(agent, agent_id="verbose")(None, context=context)
    assert result.data is None
    assert result.artifacts and result.artifacts[0].size_bytes == 500
    assert any(w.code == "RESULT_OFFLOADED" for w in result.warnings)


async def test_artifact_creation_is_traced(harness, context, spans):
    async def agent(payload, runtime):
        await runtime.artifacts.put("content", type="chart")
        return "ok"

    await harness.wrap(agent, agent_id="charts")(None, context=context)
    assert "agent.artifact.create" in span_names(spans)


async def test_claims_evidence_and_recommendations_survive_the_pipeline(harness, context):
    async def agent(payload):
        return AgentResult.ok(
            "stock is low",
            claims=[Claim(claim_id="c1", text="SKU-1 has 3 units", evidence_ids=["e1"], confidence=0.9)],
            evidence=[EvidenceRef(source_type="document", source_id="e1", page=2)],
            recommended_actions=[
                RecommendedAction(
                    action_type="reorder",
                    description="reorder 50 units of SKU-1",
                    reason_summary="below safety stock",
                    evidence_ids=["e1"],
                )
            ],
            confidence=0.82,
        )

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.claims[0].claim_id == "c1"
    assert result.evidence[0].page == 2
    assert result.recommended_actions[0].action_type == "reorder"
    assert result.confidence == 0.82


async def test_evaluation_events_carry_references_not_payloads(memory, context):
    sink = CollectingEvaluationSink()
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        evaluation_sink=sink,
        config={"evaluation_events": {"enabled": True, "synchronous": True}, "memory": {"writeback": False}},
    )

    async def agent(payload, runtime):
        return AgentResult.ok(
            "secret business answer",
            evidence=[EvidenceRef(source_type="memory", source_id="m1")],
        )

    await harness.wrap(agent, agent_id="inv", skills=["inventory.analysis"])(
        "the question", context=context
    )

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.agent_id == "inv"
    assert event.skills == ["inventory.analysis"]
    assert event.status == "SUCCESS"
    assert event.evidence_refs[0].source_id == "m1"
    assert event.request_ref == context.request_id
    assert "secret business answer" not in event.model_dump_json()


async def test_evaluation_events_are_off_by_default(memory, context):
    sink = CollectingEvaluationSink()
    harness = AgentHarness(memory=memory, defaults={"tenant_id": "acme"}, evaluation_sink=sink)

    async def agent(payload):
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    await harness.drain()
    assert sink.events == []


async def test_lifecycle_events_fire_in_order(harness, context):
    seen: list[str] = []
    harness.on(lambda event, payload: seen.append(event))

    async def agent(payload, runtime):
        return "ok"

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert seen[0] == LifecycleEvent.AGENT_START
    assert LifecycleEvent.CONTEXT_LOADED in seen
    assert seen[-1] == LifecycleEvent.AGENT_FINISH
    assert LifecycleEvent.AGENT_SUCCESS in seen


async def test_model_and_tool_lifecycle_events(memory, context):
    async def echo(x: int) -> int:
        return x

    async def model(prompt):
        return "answer"

    seen: list[str] = []
    harness = AgentHarness(
        memory=memory, model=model, tools=[echo], defaults={"tenant_id": "acme"},
        listeners=[lambda event, payload: seen.append(event)],
    )

    async def agent(payload, runtime):
        await runtime.model.invoke("q")
        await runtime.tools.call("echo", x=1)
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    for expected in (
        LifecycleEvent.MODEL_START,
        LifecycleEvent.MODEL_END,
        LifecycleEvent.TOOL_START,
        LifecycleEvent.TOOL_END,
    ):
        assert expected in seen


async def test_a_failing_listener_never_breaks_an_execution(harness, context):
    def broken(event, payload):
        raise RuntimeError("listener exploded")

    harness.on(broken)

    async def agent(payload):
        return "still fine"

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == "still fine"


async def test_error_and_timeout_lifecycle_events(harness, context):
    seen: list[str] = []
    harness.on(lambda event, payload: seen.append(event))

    async def agent(payload):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert LifecycleEvent.AGENT_ERROR in seen
    assert LifecycleEvent.AGENT_SUCCESS not in seen


async def test_registry_hook_is_a_noop_by_default(harness):
    harness.describe("inventory-agent", skills=["inventory.analysis"], version="1.2.0")
    await harness.register_agents()  # must not raise
    assert harness.descriptors["inventory-agent"].version == "1.2.0"


async def test_in_memory_registry_receives_descriptors(memory):
    from universal_agent_harness.registry.client import InMemoryAgentRegistry

    registry = InMemoryAgentRegistry()
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        registry=registry,
    )
    harness.describe("inventory-agent", skills=["inventory.analysis"])
    await harness.register_agents()
    assert registry.get("inventory-agent").skill_ids == ["inventory.analysis"]
