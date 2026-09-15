"""The whole memory surface through the harness: what is pushed, what is read, what is traced.

`test_memory.py` covers the automatic path (retrieve before, observe after). This file
covers the operations an agent drives itself — typed memories, the KG, RAG ingestion,
conversation history, the inventory view and grounding — and asserts each one is
instrumented rather than passed through untraced.
"""

from __future__ import annotations

import pytest
from tests.support import span_by_name, span_names

#: The automatic read/write path is off in this file so each assertion sees only the
#: operation under test; `test_memory.py` covers the automatic path.
#: A dict override changes only the named fields, so the harness's other memory settings
#: (``private_by_default``...) still apply — a whole MemoryPolicy would replace them.
EXPLICIT_ONLY = {
    "retrieve_before": False,
    "observe_input": False,
    "observe_output": False,
    "observe_claims": False,
}


async def run(harness, context, body):
    """Run ``body(runtime)`` as an agent so it has a real runtime."""
    captured = {}

    async def agent(payload, runtime):
        captured["value"] = await body(runtime)
        return "done"

    await harness.wrap(agent, agent_id="memory-agent", memory_policy=EXPLICIT_ONLY)(
        None, context=context
    )
    return captured["value"]


# --------------------------------------------------------------------------- writes


async def test_remember_writes_a_typed_long_term_memory(harness, memory, context, spans):
    await run(harness, context, lambda rt: rt.memory.remember(
        "the EU-1 warehouse ships on Tuesdays",
        memory_type="SEMANTIC", lifetime="LONG_TERM", visibility="WORKSPACE",
        source="ops-handbook",
    ))

    call = memory.observations[-1]
    assert call["content"] == "the EU-1 warehouse ships on Tuesdays"
    assert call["hints"] == {"memory_type": "SEMANTIC", "lifetime": "LONG_TERM",
                             "visibility": "WORKSPACE"}
    assert call["source"] == "ops-handbook"           # arbitrary metadata rides along
    assert call["idempotency_key"]                     # stable across replays

    span = span_by_name(spans, "agent.memory.remember")
    assert span.attributes["memory.type"] == "SEMANTIC"
    assert span.attributes["memory.lifetime"] == "LONG_TERM"
    assert span.attributes["memory.visibility"] == "WORKSPACE"
    assert "the EU-1 warehouse" not in str(dict(span.attributes))   # content is not exported


async def test_short_term_and_episodic_memories_are_just_different_hints(harness, memory, context):
    await run(harness, context, lambda rt: rt.memory.remember(
        "user asked about SKU-1 twice today", memory_type="EPISODIC", lifetime="SHORT_TERM",
    ))
    assert memory.observations[-1]["hints"] == {
        "memory_type": "EPISODIC", "lifetime": "SHORT_TERM"
    }


async def test_private_by_default_policy_applies_to_remember(memory, context):
    from universal_agent_harness import AgentHarness

    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"},
        config={"memory": {"private_by_default": True, "writeback": False}},
    )
    await run(harness, context, lambda rt: rt.memory.remember("a working note"))
    assert memory.observations[-1]["hints"]["visibility"] == "RUN"


async def test_share_publishes_to_the_agent_group(harness, memory, context):
    await run(harness, context, lambda rt: rt.memory.share("SKU-1 was reordered"))
    hints = memory.observations[-1]["hints"]
    assert hints["memory_type"] == "SHARED" and hints["visibility"] == "AGENT_GROUP"


async def test_forget_deletes_and_is_traced(harness, memory, context, spans):
    await run(harness, context, lambda rt: rt.memory.forget("mem_42"))
    assert memory.of("forget")[0]["memory_id"] == "mem_42"
    assert span_by_name(spans, "agent.memory.forget").attributes["memory.id"] == "mem_42"


async def test_document_ingestion_feeds_the_rag_corpus(harness, memory, context, spans, tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Reorder policy: never exceed a 30-day cover.")

    handle = await run(harness, context, lambda rt: rt.memory.add_document(
        doc, title="Reorder policy", visibility="WORKSPACE",
    ))
    assert handle.document_id == "doc_1"
    call = memory.of("files.add")[0]
    assert call["title"] == "Reorder policy" and call["visibility"] == "WORKSPACE"
    assert span_by_name(spans, "agent.memory.ingest").attributes["memory.document.id"] == "doc_1"


# --------------------------------------------------------------------------- reads


async def test_context_bundle_is_the_one_call_that_returns_everything(harness, context, spans):
    bundle = await run(harness, context, lambda rt: rt.memory.retrieve("how much stock?"))
    # conversation window, memories, RAG knowledge, graph facts and summaries in one object
    for group in ("conversation", "memories", "knowledge", "graph_facts", "summaries"):
        assert hasattr(bundle, group)
    assert bundle.evidence.status == "COMPLETE"
    assert span_by_name(spans, "agent.memory.retrieve").attributes["memory.evidence.status"]


async def test_recall_returns_ranked_evidence_without_bundle_assembly(harness, memory, context, spans):
    await run(harness, context, lambda rt: rt.memory.recall("stock levels", limit=5))
    assert memory.of("recall")[0]["limit"] == 5
    assert "agent.memory.recall" in span_names(spans)


async def test_graph_query_traverses_the_knowledge_graph(harness, memory, context, spans):
    answer = await run(harness, context, lambda rt: rt.memory.graph_query(
        "who supplies SKU-1?", hops=2,
    ))
    assert answer.facts[0].predicate == "supplied_by"
    call = memory.of("graph.query")[0]
    assert call["hops"] == 2 and call["query"] == "who supplies SKU-1?"
    span = span_by_name(spans, "agent.memory.graph")
    assert span.attributes["memory.graph.hops"] == 2
    assert span.attributes["memory.result_count"] == 1


async def test_graph_query_accepts_entities_and_a_temporal_view(harness, memory, context):
    from datetime import UTC, datetime

    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    await run(harness, context, lambda rt: rt.memory.graph_query(
        entities=["SKU-1"], hops=1, as_of=as_of,
    ))
    call = memory.of("graph.query")[0]
    assert call["entities"] == ["SKU-1"] and call["as_of"] == as_of


async def test_history_reads_the_conversation(harness, memory, context, spans):
    messages = await run(harness, context, lambda rt: rt.memory.history(limit=10))
    assert messages[0].content == "how much stock?"
    assert memory.of("chat.history")[0]["limit"] == 10
    assert span_by_name(spans, "agent.memory.history").attributes["memory.result_count"] == 1


async def test_inventory_view_lists_what_is_held(harness, memory, context, spans):
    items = await run(harness, context, lambda rt: rt.memory.memories(
        memory_types=["SEMANTIC"], limit=20,
    ))
    assert items[0].memory_type == "SEMANTIC"
    assert memory.of("memories")[0]["limit"] == 20
    assert span_by_name(spans, "agent.memory.list").attributes["memory.result_count"] == 1


async def test_get_one_memory_by_id(harness, memory, context):
    item = await run(harness, context, lambda rt: rt.memory.get("mem_7"))
    assert item.memory_id == "mem_7"


async def test_verify_grounds_an_answer_against_the_evidence(harness, memory, context, spans):
    report = await run(harness, context, lambda rt: rt.memory.verify(
        "SKU-1 has 3 units left", query="stock for SKU-1",
    ))
    assert report.grounded is True
    span = span_by_name(spans, "agent.memory.verify")
    assert span.attributes["memory.grounding.hallucination_rate"] == 0.0
    assert span.attributes["memory.grounding.grounded"] is True


# --------------------------------------------------------------------------- degradation


async def test_reads_degrade_and_writes_propagate(memory, context):
    from universal_agent_harness import AgentHarness

    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    memory.fail_retrieval = True
    memory.fail_observation = True

    async def agent(payload, runtime):
        # a failing read degrades to an empty result...
        assert await runtime.memory.graph_query("who supplies SKU-1?") is None
        assert await runtime.memory.memories() == []
        # ...a failing write is never silently dropped
        with pytest.raises(ConnectionError):
            await runtime.memory.remember("this will not land")
        return "handled"

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.data == "handled"


async def test_every_operation_is_a_noop_without_a_memory_client(context):
    from universal_agent_harness import AgentHarness

    harness = AgentHarness(defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        m = runtime.memory
        assert m.enabled is False
        assert await m.retrieve("q") is None
        assert await m.recall("q") == []
        assert await m.remember("x") is None
        assert await m.memories() == []
        assert await m.history() == []
        assert await m.graph_query("q") is None
        assert await m.add_document("f") is None
        assert await m.verify("a") is None
        assert m.graph is None and m.files is None
        return "fine"

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == "fine"
