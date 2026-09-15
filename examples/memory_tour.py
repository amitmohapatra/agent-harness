"""Everything the Memory Service does, driven through the harness.

    python examples/memory_tour.py                                    # in-process demo service
    MEMORY_SERVICE_URL=http://localhost:8080 python examples/memory_tour.py   # the real one

The Memory Service is one service with several kinds of memory behind it. This script walks
the whole surface — what you *push* into it, what you *get* back, and which call to reach
for — with every operation going through ``runtime.memory`` so it is traced, timed,
scope-correct and idempotent.

    PUSH                                   GET
    ----------------------------------     -------------------------------------------
    chat messages      -> history          retrieve()   -> one bundle with everything
    observations       -> episodic memory  recall()     -> ranked evidence only
    remember()         -> typed memory     history()    -> the conversation window
    documents          -> RAG corpus       graph_query()-> knowledge-graph facts
    share()            -> agent group      memories()   -> the inventory view
    tool invocations   -> tool memory      verify()     -> grounding report
    forget()           -> deletion

The one call that matters most is ``retrieve()``: it returns the conversation window, the
relevant memories, the RAG knowledge, the graph facts and the summaries as a single bounded,
ranked, evidence-gated bundle — which is what an agent actually wants to put in a prompt.
The rest of this file exists so you know what is underneath it.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentRuntime,
    MemoryObservation,
)

TENANT, USER, THREAD = "acme", "planner-7", "chat-memory-tour"


async def tour(agent: AgentRuntime) -> dict[str, Any]:
    """One agent run that exercises every memory operation the harness instruments."""
    memory = agent.memory
    out: dict[str, Any] = {}

    # ----------------------------------------------------------------- PUSH
    # 1. Conversation history. The turn itself: what the user said, what the agent replied.
    #    These are the raw record; everything else is derived from them by the service.
    await memory.record_input("How much stock of SKU-1 do we have?")

    # 2. An episodic observation: something that happened. The service classifies it,
    #    extracts entities into the knowledge graph and decides what is worth keeping.
    await memory.observe(
        MemoryObservation(
            content="Stock check for SKU-1 returned 95 units on hand at EU-1.",
            kind="EVENT",
            metadata={"sku": "SKU-1", "warehouse": "EU-1"},
        )
    )

    # 3. A typed memory: you say what kind of knowledge this is and how long it should live.
    #    memory_type: SEMANTIC | EPISODIC | PROCEDURAL | PREFERENCE | DECISION | OUTCOME ...
    #    lifetime:    EPHEMERAL | SHORT_TERM | LONG_TERM | ARCHIVAL
    #    visibility:  PRIVATE | RUN | AGENT_GROUP | THREAD | USER | WORK | WORKSPACE | TENANT
    await memory.remember(
        "SKU-1 is reordered from Castor Supply when cover falls below 10 days.",
        memory_type="SEMANTIC", lifetime="LONG_TERM", visibility="WORKSPACE",
        source="reorder-policy",
    )
    await memory.remember(
        "The planner prefers weekly digests over per-event alerts.",
        memory_type="PREFERENCE", lifetime="LONG_TERM", visibility="USER",
    )
    await memory.remember(
        "Checking SKU-1 twice in one session usually means a customer escalation.",
        memory_type="EPISODIC", lifetime="SHORT_TERM",
    )

    # 4. A document: parsed, chunked and indexed, becoming retrievable knowledge (RAG).
    #    Ingestion is asynchronous — the handle returns immediately.
    policy_doc = Path(__file__).with_name("_reorder_policy.txt")
    policy_doc.write_text(
        "Reorder policy v3\n"
        "Safety stock is held at a 95% service level.\n"
        "Never exceed 30 days of cover for class-A parts.\n"
    )
    try:
        handle = await memory.add_document(
            policy_doc, title="Reorder policy v3", visibility="WORKSPACE"
        )
        out["document_id"] = getattr(handle, "document_id", None)
    finally:
        policy_doc.unlink(missing_ok=True)

    # 5. Cross-agent knowledge. An agent's working memory is private to its run by default;
    #    this is the explicit way to publish something to the other agents in the group.
    await memory.share("SKU-1 reorder was raised with Castor Supply on 2026-09-15.")

    # 6. The agent's own answer, recorded as the assistant turn.
    await memory.record_output("You have 95 units of SKU-1, about 4 days of cover.")

    # ----------------------------------------------------------------- GET
    # 7. The 90% call: one bounded, ranked, evidence-gated bundle for this turn.
    bundle = await memory.retrieve("Should we reorder SKU-1?", token_budget=1500)
    if bundle is not None:
        facts = memory.describe(bundle)
        out["bundle"] = {
            # evidence status is COMPLETE | INCOMPLETE | INSUFFICIENT
            "evidence_status": facts["evidence_status"],
            "tokens": facts["token_estimate"],
            # counts: memories / knowledge (RAG) / graph facts / summaries
            "counts": facts["counts"],
            "cache_hit": facts["cache_hit"],
        }
        # bundle.rendered is prompt-ready text; bundle.conversation is the recent window.
        out["prompt_preview"] = (getattr(bundle, "rendered", "") or "")[:80]

    # 8. Ranked evidence without bundle assembly — when you want the items, not a prompt.
    out["recall"] = len(await memory.recall("reorder policy for class-A parts", limit=5))

    # 9. The conversation window on its own.
    out["history"] = len(await memory.history(limit=20))

    # 10. Knowledge graph: entities and their relationships, optionally as of a point in time.
    answer = await memory.graph_query("who supplies SKU-1?", hops=2)
    out["graph_facts"] = [
        f"{f.subject} -{f.predicate}-> {f.object}" for f in getattr(answer, "facts", [])
    ] if answer else []
    await memory.graph_query(entities=["SKU-1"], hops=1, as_of=datetime.now(UTC))

    # 11. The inventory view: what do we actually hold for this user/thread/run?
    held = await memory.memories(memory_types=["SEMANTIC", "PREFERENCE"], limit=50)
    out["held"] = [getattr(m, "memory_id", "?") for m in held][:5]

    # 12. Grounding: is the answer we are about to give supported by the evidence?
    report = await memory.verify(
        "SKU-1 has 95 units on hand and is reordered from Castor Supply.",
        bundle=bundle,
    ) if bundle is not None else None
    if report is not None:
        out["grounding"] = {
            "grounded": getattr(report, "grounded", None),
            "hallucination_rate": getattr(report, "per_claim_hallucination_rate", None),
        }

    # 13. Deletion, when a memory should not have been kept.
    if held:
        await memory.forget(getattr(held[0], "memory_id", "unknown"))
        out["forgot"] = getattr(held[0], "memory_id", None)

    # 14. Anything the harness does not wrap is still one attribute away — the bound SDK
    #     context. These calls work, they are simply not traced by the harness.
    out["sdk_escape_hatch"] = type(memory.sdk).__name__ if memory.sdk else None
    return out


async def main() -> None:
    url = os.environ.get("MEMORY_SERVICE_URL")
    if url:
        from universal_memory import MemoryClient  # noqa: PLC0415 - optional in this example

        client: Any = MemoryClient(url, api_key=os.environ.get("MEMORY_API_KEY"))
    else:
        client = DemoMemoryClient()   # prints the calls it would have made

    harness = AgentHarness(
        memory=client,
        defaults={"tenant_id": TENANT, "user_id": USER, "agent_group_id": "supply-chain"},
        config={
            "memory": {
                "record_messages": True,     # let the harness write the chat turns
                "retrieve_before": False,    # this example drives retrieval explicitly
                "observe_input": False,
                "observe_output": False,
                "observe_claims": False,
                "writeback": False,          # await the writes so the output is ordered
            }
        },
    )
    async def memory_tour(_payload: Any, runtime: AgentRuntime) -> dict[str, Any]:
        return await tour(runtime)

    wrapped = harness.wrap(memory_tour, agent_id="memory-tour", skills=["memory.tour"])

    context = AgentExecutionContext.create(
        tenant_id=TENANT, agent_id="memory-tour", user_id=USER,
        thread_id=THREAD, turn_id="turn-1", work_id="wo-2291",
    )
    result = await wrapped(None, context=context)

    print("\n=== memory tour ===")
    print(f"service      : {url or 'in-process demo (set MEMORY_SERVICE_URL for the real one)'}")
    for key, value in result.data.items():
        print(f"{key:<13}: {value}")
    if not url:
        print("\ncalls the harness made:")
        for name, payload in client.calls:
            detail = {k: v for k, v in payload.items() if k not in ("scope", "custom_metadata")}
            print(f"  {name:<16} {str(detail)[:96]}")

    await harness.aclose()


# --------------------------------------------------------------------------- demo service
# A tiny stand-in so this file runs with nothing installed. It mirrors the SDK's shape; the
# real client is a drop-in replacement (that is the point of depending on the SDK contract).


class DemoMemoryClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def bind(self, **scope: Any) -> DemoContext:
        return DemoContext(self, scope)


class DemoContext:
    def __init__(self, client: DemoMemoryClient, scope: dict[str, Any]) -> None:
        self._client, self.scope = client, _Scope(scope)
        self.chat, self.graph, self.files = _Chat(client), _Graph(client), _Files(client)

    def derive(self, **changes: Any) -> DemoContext:
        return DemoContext(self._client, {**self.scope.fields, **changes})

    def _record(self, name: str, **payload: Any) -> None:
        self._client.calls.append((name, payload))

    async def context(self, query: str, **options: Any) -> Any:
        self._record("context", query=query, **options)
        return _Bundle(query)

    async def recall(self, query: str, **options: Any) -> list[Any]:
        self._record("recall", query=query, **options)
        return [_Item("k1"), _Item("k2")]

    async def observe(self, content: str, **kwargs: Any) -> Any:
        self._record("observe", content=content[:60], hints=kwargs.get("hints"))
        return _Ack()

    async def memories(self, **options: Any) -> list[Any]:
        self._record("memories", **options)
        return [_Memory("mem_policy"), _Memory("mem_preference")]

    async def get_memory(self, memory_id: str) -> Any:
        self._record("get_memory", memory_id=memory_id)
        return _Memory(memory_id)

    async def forget(self, memory_id: str) -> None:
        self._record("forget", memory_id=memory_id)

    async def verify(self, answer: str, **options: Any) -> Any:
        self._record("verify", answer=answer[:40])
        return _Grounding()


class _Scope:
    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields

    def __getattr__(self, item: str) -> Any:
        return self.fields.get(item)


class _Chat:
    def __init__(self, client: DemoMemoryClient) -> None:
        self._client = client

    async def user(self, content: str, **kwargs: Any) -> Any:
        self._client.calls.append(("chat.user", {"content": content[:50]}))
        return _Ack()

    async def assistant(self, content: str, **kwargs: Any) -> Any:
        self._client.calls.append(("chat.assistant", {"content": content[:50]}))
        return _Ack()

    async def history(self, *, limit: int = 50, include_internal: bool = False) -> list[Any]:
        self._client.calls.append(("chat.history", {"limit": limit}))
        return [_Message("How much stock of SKU-1 do we have?")]


class _Graph:
    def __init__(self, client: DemoMemoryClient) -> None:
        self._client = client

    async def query(self, query=None, *, entities=None, hops=1, as_of=None) -> Any:
        self._client.calls.append(
            ("graph.query", {"query": query, "entities": entities, "hops": hops})
        )
        return _GraphAnswer()


class _Files:
    def __init__(self, client: DemoMemoryClient) -> None:
        self._client = client

    async def add(self, file: Any, **kwargs: Any) -> Any:
        self._client.calls.append(("files.add", {"file": Path(str(file)).name, **kwargs}))
        return _Document()


class _Ack:
    observation_id, job_ids = "obs_demo", ()


class _Item:
    def __init__(self, item_id: str) -> None:
        self.item_id, self.text, self.score = item_id, "...", 0.8


class _Memory:
    def __init__(self, memory_id: str) -> None:
        self.memory_id, self.memory_type, self.lifetime = memory_id, "SEMANTIC", "LONG_TERM"


class _Message:
    def __init__(self, content: str) -> None:
        self.content, self.role = content, "USER"


class _Document:
    document_id, filename, job_ids = "doc_demo", "_reorder_policy.txt", ()


class _Fact:
    subject, predicate, object = "SKU-1", "supplied_by", "Castor Supply"


class _GraphAnswer:
    facts, entities, matched, visited = (_Fact(),), (), (), 1


class _Grounding:
    per_claim_hallucination_rate, supported, unsupported = 0.0, 2, 0
    grounded = True


class _Conversation:
    thread_id, message_ids, rendered = THREAD, (), "user: How much stock of SKU-1?"


class _Evidence:
    status, notes, unused = "COMPLETE", (), ()


class _Bundle:
    def __init__(self, query: str) -> None:
        self.query, self.query_type, self.bundle_id = query, "FACTUAL", "bundle_demo"
        self.conversation, self.evidence = _Conversation(), _Evidence()
        self.memories = (_Item("m1"), _Item("m2"))
        self.knowledge = (_Item("k1"),)          # RAG chunks from the ingested document
        self.graph_facts = (_Item("g1"),)
        self.summaries = ()
        self.token_budget, self.token_estimate, self.cache_hit = 1500, 240, False
        self.rendered = "SKU-1: 95 on hand. Policy: reorder below 10 days of cover."


if __name__ == "__main__":
    asyncio.run(main())
