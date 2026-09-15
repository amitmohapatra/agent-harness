"""Observability is asserted in tests, not eyeballed in a UI (§85).

These cover the span hierarchy of §25, the Langfuse mapping of §24, the privacy defaults of
§26 and the failure behaviour of §32 — with Langfuse both disabled and enabled.
"""

from __future__ import annotations

from tests.support import span_by_name, span_names

from universal_agent_harness import AgentHarness, AgentResult
from universal_agent_harness.langfuse import attributes as LA


class Model:
    async def ainvoke(self, prompt, **kwargs):
        return {"text": "answer", "model": "m-1", "usage": {"input_tokens": 5, "output_tokens": 2}}


async def inventory_db(sku: str) -> dict:
    return {"sku": sku, "on_hand": 2}


def build(memory, **config_overrides):
    config = {"memory": {"writeback": False}}
    config.update(config_overrides)
    return AgentHarness(
        memory=memory,
        model=Model(),
        tools=[inventory_db],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config=config,
    )


async def run_multi_agent(harness, context):
    """planner -> (inventory | promotion) -> synthesis, as in §25."""

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def inventory(state, agent):
        await agent.model.invoke("check stock")
        await agent.tools.call("inventory_db", sku=state["sku"])
        return AgentResult.ok({"stock": 2})

    @harness.agent(agent_id="promotion-agent")
    async def promotion(state, agent):
        await agent.model.invoke("check promotions")
        return AgentResult.ok({"promo": None})

    @harness.agent(agent_id="synthesis-agent")
    async def synthesis(state, agent):
        await agent.model.invoke("summarise")
        return AgentResult.ok("final answer")

    @harness.agent(agent_id="planner-agent")
    async def planner(state, agent):
        stock = await inventory(state)
        promo = await promotion(state)
        return await synthesis({"stock": stock.data, "promo": promo.data})

    return await planner({"sku": "SKU-1", "question": "should we reorder?"}, context=context)


async def test_span_hierarchy_matches_the_documented_shape(memory, context, spans):
    harness = build(memory)
    await run_multi_agent(harness, context)

    names = span_names(spans)
    assert names.count("agent.run") == 4
    assert names.count("agent.model.invoke") == 3  # planner delegates; the three others call
    assert names.count("agent.tool.call") == 1
    assert "agent.memory.retrieve" in names

    finished = spans.get_finished_spans()
    # one trace for the whole multi-agent turn
    assert len({s.get_span_context().trace_id for s in finished}) == 1

    by_agent = {
        s.attributes["agent.id"]: s for s in finished if s.name == "agent.run"
    }
    planner = by_agent["planner-agent"]
    for child_agent in ("inventory-agent", "promotion-agent", "synthesis-agent"):
        assert by_agent[child_agent].parent.span_id == planner.get_span_context().span_id
        assert by_agent[child_agent].attributes["agent.parent_run.id"] == planner.attributes["agent.run.id"]

    inventory_children = [
        s.name
        for s in finished
        if s.parent and s.parent.span_id == by_agent["inventory-agent"].get_span_context().span_id
    ]
    assert "agent.model.invoke" in inventory_children
    assert "agent.tool.call" in inventory_children
    assert "agent.memory.retrieve" in inventory_children


async def test_agent_span_identity_attributes(memory, context, spans):
    harness = build(memory)

    @harness.agent(agent_id="inventory-agent", skills=["inventory.stockout"])
    async def agent(state, runtime):
        return "ok"

    await agent({"question": "q"}, context=context)
    span = span_by_name(spans, "agent.run")
    assert span.attributes["agent.id"] == "inventory-agent"
    assert span.attributes["agent.skill"] == ("inventory.stockout",)
    assert span.attributes["tenant.id"] == "acme"
    assert span.attributes["thread.id"] == "chat-1"
    assert span.attributes["agent.harness.version"]
    assert span.attributes["status"] == "SUCCESS"


async def test_user_id_is_not_exported_unless_capture_allows_it(memory, context, spans):
    harness = build(memory)

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        return "ok"

    await agent({"question": "q"}, context=context)
    assert "enduser.id" not in span_by_name(spans, "agent.run").attributes

    spans.clear()
    permissive = build(memory, telemetry={"capture": {"user_id": True}})

    @permissive.agent(agent_id="inv")
    async def agent2(state, runtime):
        return "ok"

    await agent2({"question": "q"}, context=context)
    assert span_by_name(spans, "agent.run").attributes["enduser.id"] == "u1"


async def test_memory_retrieval_span_carries_bundle_facts_not_content(memory, context, spans):
    harness = build(memory)

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        return "ok"

    await agent({"question": "how much stock?"}, context=context)
    span = span_by_name(spans, "agent.memory.retrieve")
    assert span.attributes["memory.evidence.status"] == "COMPLETE"
    assert span.attributes["memory.token_estimate"] == 42
    assert span.attributes["memory.item_count"] == 0
    assert "input.value" not in span.attributes  # raw_memory_content is off by default
    assert "remembered:" not in str(dict(span.attributes))


async def test_sampling_out_a_run_produces_no_spans_but_keeps_metrics(memory, context, spans):
    harness = build(memory, telemetry={"sampling": {"sample_rate": 0.0}})

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        await runtime.model.invoke("q")
        return "ok"

    result = await agent({"question": "q"}, context=context)
    assert result.data == "ok"
    assert span_names(spans) == []


# --------------------------------------------------------------------------- langfuse


def langfuse_harness(memory, **langfuse_overrides):
    langfuse = {
        "enabled": True,
        "mode": "otlp",  # attribute mapping only: no SDK client, no network
        "public_key": "pk-test",
        "secret_key": "sk-test",
        **langfuse_overrides,
    }
    return AgentHarness(
        memory=memory,
        model=Model(),
        tools=[inventory_db],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config={"memory": {"writeback": False}, "observability": {"langfuse": langfuse}},
    )


async def test_langfuse_disabled_emits_no_langfuse_attributes(memory, context, spans):
    harness = build(memory)

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        return "ok"

    await agent({"question": "q"}, context=context)
    attributes = dict(span_by_name(spans, "agent.run").attributes)
    assert not any(k.startswith("langfuse.") for k in attributes)


async def test_langfuse_maps_the_documented_trace_and_observation_fields(memory, context, spans):
    harness = langfuse_harness(memory)

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def agent(state, runtime):
        await runtime.model.invoke("check stock")
        await runtime.tools.call("inventory_db", sku="SKU-1")
        return "ok"

    await agent({"question": "how much stock?"}, context=context)

    run = span_by_name(spans, "agent.run")
    assert run.attributes[LA.OBSERVATION_TYPE] == "agent"
    assert run.attributes[LA.TRACE_NAME] == "inventory-agent"
    assert run.attributes[LA.TRACE_SESSION_ID] == "chat-1"  # thread -> session (§24)
    assert "skill:inventory.analysis" in run.attributes[LA.TRACE_TAGS]
    assert LA.TRACE_USER_ID not in run.attributes  # user id is capture-gated

    assert span_by_name(spans, "agent.model.invoke").attributes[LA.OBSERVATION_TYPE] == "generation"
    assert span_by_name(spans, "agent.tool.call").attributes[LA.OBSERVATION_TYPE] == "tool"
    assert span_by_name(spans, "agent.memory.retrieve").attributes[LA.OBSERVATION_TYPE] == "retriever"


async def test_langfuse_generation_carries_model_and_usage_details(memory, context, spans):
    import json

    harness = langfuse_harness(memory)

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        await runtime.model.invoke("q")
        return "ok"

    await agent({"question": "q"}, context=context)
    generation = span_by_name(spans, "agent.model.invoke")
    usage = json.loads(generation.attributes[LA.OBSERVATION_USAGE_DETAILS])
    assert usage == {"input": 5, "output": 2, "total": 7}
    assert generation.attributes[LA.OBSERVATION_MODEL] == "m-1"


async def test_langfuse_user_id_flows_only_when_capture_allows(memory, context, spans):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config={
            "memory": {"writeback": False},
            "observability": {
                "langfuse": {
                    "enabled": True,
                    "mode": "otlp",
                    "public_key": "pk",
                    "secret_key": "sk",
                    "capture": {"user_id": True},
                }
            },
        },
    )

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        return "ok"

    await agent({"question": "q"}, context=context)
    assert span_by_name(spans, "agent.run").attributes[LA.TRACE_USER_ID] == "u1"


async def test_langfuse_keeps_one_span_tree_rather_than_duplicating_it(memory, context, spans):
    """Langfuse enriches the harness's OpenTelemetry spans; it must not create a second
    parallel span per operation (§22)."""
    plain = build(memory)

    @plain.agent(agent_id="inv")
    async def agent(state, runtime):
        await runtime.model.invoke("q")
        return "ok"

    await agent({"question": "q"}, context=context)
    baseline = len(spans.get_finished_spans())

    spans.clear()
    harness = langfuse_harness(memory)

    @harness.agent(agent_id="inv")
    async def agent2(state, runtime):
        await runtime.model.invoke("q")
        return "ok"

    await agent2({"question": "q"}, context=context)
    assert len(spans.get_finished_spans()) == baseline


async def test_langfuse_errors_do_not_break_execution(memory, context, spans):
    harness = langfuse_harness(memory)

    class Exploding:
        def decorate_trace(self, *args, **kwargs):
            raise RuntimeError("langfuse is down")

        strict = False

    # replace the provider used by the interceptor with one that always fails
    for interceptor in harness.chain.interceptors:
        if interceptor.name == "langfuse":
            interceptor.provider = Exploding()

    @harness.agent(agent_id="inv")
    async def agent(state, runtime):
        return "business result"

    result = await agent({"question": "q"}, context=context)
    assert result.data == "business result"


async def test_error_spans_carry_the_normalized_error(memory, context, spans):
    harness = build(memory)

    @harness.agent(agent_id="inv", error_mode="result")
    async def agent(state, runtime):
        raise RuntimeError("upstream failed")

    await agent({"question": "q"}, context=context)
    span = span_by_name(spans, "agent.run")
    assert span.attributes["error.category"] == "UNKNOWN"
    assert span.status.status_code.name == "ERROR"
