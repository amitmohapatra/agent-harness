"""The whole memory surface through the harness, against a running Memory Service.

`test_memory.py` covers the automatic path (retrieve before, observe after). This file
covers the operations an agent drives itself — typed memories, the KG, RAG ingestion,
conversation history, the inventory view and grounding — asserting both that the request
carried what it should and that the service accepted and answered it.

Assertions describe *shape and contract*, not specific content: what the retriever ranks
first or which predicate the extractor chooses is the service's business, and pinning it
here would make this a change-detector rather than a test.
"""

from __future__ import annotations

import asyncio

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


#: How long a write may take to become listable. Writes are asynchronous — the API commits
#: an observation and queues the work that turns it into a memory — so a read straight after
#: a write proves nothing and polling is the honest way to wait.
#:
#: A timeout here is a **failure**, not a skip. Both of these tests used to skip themselves
#: when the memory never arrived, which meant they never ran at all: the service had an
#: unscheduled outbox sweep, so a write whose fast-path dispatch was missed sat undispatched
#: forever, and the suite reported that as a clean run. Measured after the sweep was
#: scheduled, the write lands in about a second.
MATERIALISE_SECONDS = 30


async def _until_listed(rt, marker: str):
    """The memories this run wrote, once the service has them. Fails if they never arrive."""
    for _ in range(MATERIALISE_SECONDS):
        held = await rt.memory.memories(limit=50)
        mine = [m for m in held if marker in m.content]
        if mine:
            return mine
        await asyncio.sleep(1)
    raise AssertionError(
        f"no memory containing {marker!r} was listable after {MATERIALISE_SECONDS}s: the "
        f"write was accepted and never became a memory"
    )


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
    await run(
        harness,
        context,
        lambda rt: rt.memory.remember(
            "the EU-1 warehouse ships on Tuesdays",
            memory_type="SEMANTIC",
            lifetime="LONG_TERM",
            visibility="WORKSPACE",
            source="ops-handbook",
        ),
    )

    call = memory.observations[-1]
    assert call["content"] == "the EU-1 warehouse ships on Tuesdays"
    assert call["hints"] == {
        "memory_type": "SEMANTIC",
        "lifetime": "LONG_TERM",
        "visibility": "WORKSPACE",
    }
    assert call["source"] == "ops-handbook"  # arbitrary metadata rides along
    assert call["idempotency_key"]  # stable across replays

    span = span_by_name(spans, "agent.memory.remember")
    assert span.attributes["memory.type"] == "SEMANTIC"
    assert span.attributes["memory.lifetime"] == "LONG_TERM"
    assert span.attributes["memory.visibility"] == "WORKSPACE"
    assert "the EU-1 warehouse" not in str(dict(span.attributes))  # content is not exported


async def test_short_term_and_episodic_memories_are_just_different_hints(harness, memory, context):
    await run(
        harness,
        context,
        lambda rt: rt.memory.remember(
            "user asked about SKU-1 twice today",
            memory_type="EPISODIC",
            lifetime="SHORT_TERM",
        ),
    )
    assert memory.observations[-1]["hints"] == {"memory_type": "EPISODIC", "lifetime": "SHORT_TERM"}


async def test_private_by_default_policy_applies_to_remember(memory, context):
    from universal_agent_harness import AgentHarness

    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"private_by_default": True, "writeback": False}},
    )
    await run(harness, context, lambda rt: rt.memory.remember("a working note"))
    assert memory.observations[-1]["hints"]["visibility"] == "RUN"


async def test_share_publishes_to_the_agent_group(harness, memory, context):
    await run(harness, context, lambda rt: rt.memory.share("SKU-1 was reordered"))
    hints = memory.observations[-1]["hints"]
    assert hints["memory_type"] == "SHARED" and hints["visibility"] == "AGENT_GROUP"


async def test_forget_deletes_and_is_traced(harness, memory, context, spans):
    """Delete a memory this test created, rather than a made-up id the service rejects."""

    async def body(rt):
        await rt.memory.remember(
            f"Depot {context.turn_id} stores the reserve stock for SKU-1.", visibility="USER"
        )
        mine = await _until_listed(rt, context.turn_id)
        await rt.memory.forget(mine[0].memory_id)
        return mine[0].memory_id

    memory_id = await run(harness, context, body)
    assert memory.of("forget")[0]["memory_id"] == memory_id
    assert span_by_name(spans, "agent.memory.forget").attributes["memory.id"] == memory_id


async def test_document_ingestion_feeds_the_rag_corpus(harness, memory, context, spans, tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Reorder policy: never exceed a 30-day cover.")

    handle = await run(
        harness,
        context,
        lambda rt: rt.memory.add_document(
            doc,
            title="Reorder policy",
            visibility="WORKSPACE",
        ),
    )
    assert handle.document_id.startswith("doc_"), "the service returns a document handle"
    call = memory.of("files.add")[0]
    assert call["title"] == "Reorder policy" and call["visibility"] == "WORKSPACE"
    span = span_by_name(spans, "agent.memory.ingest")
    assert span.attributes["memory.document.id"] == handle.document_id


# --------------------------------------------------------------------------- reads


async def test_context_bundle_is_the_one_call_that_returns_everything(harness, context, spans):
    bundle = await run(harness, context, lambda rt: rt.memory.retrieve("how much stock?"))
    # conversation window, memories, RAG knowledge, graph facts and summaries in one object
    for group in ("conversation", "memories", "knowledge", "graph_facts", "summaries"):
        assert hasattr(bundle, group)
    # the verdict is the service's to make; what matters is that it made one
    assert bundle.evidence.status in ("COMPLETE", "INCOMPLETE", "INSUFFICIENT")
    assert span_by_name(spans, "agent.memory.retrieve").attributes["memory.evidence.status"]


async def test_recall_returns_ranked_evidence_without_bundle_assembly(
    harness, memory, context, spans
):
    await run(harness, context, lambda rt: rt.memory.recall("stock levels", limit=5))
    assert memory.of("recall")[0]["limit"] == 5
    assert "agent.memory.recall" in span_names(spans)


async def test_graph_query_traverses_the_knowledge_graph(harness, memory, context, spans):
    answer = await run(
        harness,
        context,
        lambda rt: rt.memory.graph_query(
            "who supplies SKU-1?",
            hops=2,
        ),
    )
    assert answer is not None
    for fact in getattr(answer, "facts", []):
        assert fact.subject and fact.predicate, "a fact needs a subject and a predicate"
    call = memory.of("graph.query")[0]
    assert call["hops"] == 2 and call["query"] == "who supplies SKU-1?"
    span = span_by_name(spans, "agent.memory.graph")
    assert span.attributes["memory.graph.hops"] == 2
    assert "memory.result_count" in span.attributes


async def test_graph_query_accepts_entities_and_a_temporal_view(harness, memory, context):
    from datetime import UTC, datetime

    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    await run(
        harness,
        context,
        lambda rt: rt.memory.graph_query(
            entities=["SKU-1"],
            hops=1,
            as_of=as_of,
        ),
    )
    call = memory.of("graph.query")[0]
    assert call["entities"] == ["SKU-1"] and call["as_of"] == as_of


async def test_history_reads_the_conversation(harness, memory, context, spans):
    async def body(rt):
        await rt.memory.record_input("how much stock?")
        await rt.memory.record_output("Three units.")
        return await rt.memory.history(limit=10)

    messages = await run(harness, context, body)
    assert [m.content for m in messages] == ["how much stock?", "Three units."]
    assert memory.of("chat.history")[0]["limit"] == 10
    assert span_by_name(spans, "agent.memory.history").attributes["memory.result_count"] == 2


async def test_inventory_view_lists_what_is_held(harness, memory, context, spans):
    """The inventory is scoped to this execution, and a fresh context holds nothing yet —
    so what is asserted is the request and the shape of the answer."""
    items = await run(
        harness,
        context,
        lambda rt: rt.memory.memories(
            memory_types=["SEMANTIC"],
            limit=20,
        ),
    )
    assert isinstance(items, list)
    assert all(m.memory_type for m in items)
    call = memory.of("memories")[0]
    assert call["limit"] == 20 and call["memory_types"] == ["SEMANTIC"]
    span = span_by_name(spans, "agent.memory.list")
    assert span.attributes["memory.result_count"] == len(items)


async def test_get_one_memory_by_id(harness, memory, context):
    async def body(rt):
        await rt.memory.remember(
            f"Depot {context.turn_id} is the overflow site for SKU-1.", visibility="USER"
        )
        mine = await _until_listed(rt, context.turn_id)
        return await rt.memory.get(mine[0].memory_id), mine[0].memory_id

    fetched, memory_id = await run(harness, context, body)
    assert fetched.memory_id == memory_id
    assert fetched.content


async def test_verify_grounds_an_answer_against_the_evidence(harness, memory, context, spans):
    """A verdict is the point, not a particular verdict: an unverifiable claim *should* come
    back unsupported."""
    report = await run(
        harness,
        context,
        lambda rt: rt.memory.verify(
            "SKU-1 has 3 units left",
            query="stock for SKU-1",
        ),
    )
    assert report is not None
    assert report.claims, "the report should carry a per-claim verdict"
    assert report.claims[0].verdict in ("supported", "unsupported", "contradicted", "borderline")
    span = span_by_name(spans, "agent.memory.verify")
    assert 0.0 <= span.attributes["memory.grounding.hallucination_rate"] <= 1.0
    assert span.attributes["memory.grounding.grounded"] is report.grounded


# --------------------------------------------------------------------------- degradation


async def test_reads_degrade_and_writes_propagate(dead_memory, context):
    """Against a service that is genuinely down: reads degrade, writes propagate."""
    from universal_agent_harness import AgentHarness

    harness = AgentHarness(
        memory=dead_memory,
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"writeback": False, "retrieve_before": False},
            "timeouts": {"memory_seconds": 3.0},
        },
    )

    async def agent(payload, runtime):
        # a failing read degrades to an empty result...
        assert await runtime.memory.graph_query("who supplies SKU-1?") is None
        assert await runtime.memory.memories() == []
        # ...a failing write is never silently dropped
        with pytest.raises(Exception, match=r"(?i)connect|unavailable|timeout"):
            await runtime.memory.remember("this will not land")
        return "handled"

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.data == "handled"
    await harness.aclose()


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
