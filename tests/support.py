"""Shared test doubles and span helpers.

The Memory Service itself is out of scope here (it has its own test suite); what these
tests must prove is that the *harness* calls it correctly — right scope, right idempotency
keys, right ordering, right degradation — so the fake records calls and is deliberately
faithful to the SDK's public shape.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# --------------------------------------------------------------------------- memory fake


@dataclass
class FakeAck:
    observation_id: str = "obs_1"
    job_ids: list[str] = field(default_factory=list)
    deduplicated: bool = False


@dataclass
class FakeEvidence:
    status: str = "COMPLETE"
    notes: list[str] = field(default_factory=list)
    unused: list[Any] = field(default_factory=list)


@dataclass
class FakeConversation:
    thread_id: str = "chat-1"
    message_ids: list = field(default_factory=list)
    rendered: str = ""
    summary: str | None = None


@dataclass
class FakeBundle:
    query: str = ""
    query_type: str = "FACTUAL"
    bundle_id: str = "bundle_1"
    rendered: str = "remembered: SKU-1 stock is low"
    conversation: FakeConversation = field(default_factory=FakeConversation)
    memories: list[Any] = field(default_factory=list)
    knowledge: list[Any] = field(default_factory=list)
    graph_facts: list[Any] = field(default_factory=list)
    summaries: list[Any] = field(default_factory=list)
    token_budget: int = 2000
    token_estimate: int = 42
    cache_hit: bool = False
    evidence: FakeEvidence = field(default_factory=FakeEvidence)


class FakeScope:
    def __init__(self, fields: dict[str, Any]) -> None:
        self.__dict__.update(fields)
        self._fields = fields

    def __getattr__(self, item: str) -> Any:
        return self._fields.get(item)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self._fields)


class FakeChat:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = calls

    async def user(self, content: str, **kwargs: Any) -> FakeAck:
        self._calls.append(("chat.user", {"content": content, **kwargs}))
        return FakeAck()

    async def assistant(self, content: str, **kwargs: Any) -> FakeAck:
        self._calls.append(("chat.assistant", {"content": content, **kwargs}))
        return FakeAck()

    async def internal(self, content: str, **kwargs: Any) -> FakeAck:
        self._calls.append(("chat.internal", {"content": content, **kwargs}))
        return FakeAck()

    async def history(self, *, limit: int = 50, include_internal: bool = False) -> list[Any]:
        self._calls.append(
            ("chat.history", {"limit": limit, "include_internal": include_internal})
        )
        return [FakeMessage()]


@dataclass
class FakeMemoryRecord:
    memory_id: str = "mem_1"
    content: str = "SKU-1 reorder point is 50"
    memory_type: str = "SEMANTIC"
    lifetime: str = "LONG_TERM"
    visibility: str = "USER"


@dataclass
class FakeGraphFact:
    relation_id: str = "rel_1"
    subject: str = "SKU-1"
    predicate: str = "supplied_by"
    object: str = "Castor Supply"
    fact_text: str = "SKU-1 is supplied by Castor Supply"


@dataclass
class FakeGraphAnswer:
    matched: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    facts: list = field(default_factory=lambda: [FakeGraphFact()])
    visited: int = 1


@dataclass
class FakeGrounding:
    per_claim_hallucination_rate: float = 0.0
    supported: int = 2
    unsupported: int = 0
    contradicted: int = 0

    @property
    def grounded(self) -> bool:
        return self.per_claim_hallucination_rate == 0.0


@dataclass
class FakeFileHandle:
    document_id: str = "doc_1"
    filename: str = "policy.txt"
    checksum: str = "abc"
    size_bytes: int = 12
    job_ids: list = field(default_factory=lambda: ["job_1"])


@dataclass
class FakeMessage:
    message_id: str = "msg_1"
    role: str = "USER"
    kind: str = "VISIBLE"
    sequence: int = 1
    content: str = "how much stock?"


class FakeGraph:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]], client: Any = None) -> None:
        self._calls = calls
        self._client = client

    async def query(self, query=None, *, entities=None, hops=1, as_of=None) -> FakeGraphAnswer:
        self._calls.append(
            ("graph.query", {"query": query, "entities": entities, "hops": hops, "as_of": as_of})
        )
        if self._client is not None and self._client.fail_retrieval:
            raise ConnectionError("memory service unavailable")
        return FakeGraphAnswer()


class FakeFiles:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = calls

    async def add(self, file: Any, **kwargs: Any) -> FakeFileHandle:
        self._calls.append(("files.add", {"file": str(file)[:40], **kwargs}))
        return FakeFileHandle()


class FakeTools:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = calls

    async def record(self, tool: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._calls.append(("tools.record", {"tool": tool, "args": args, **kwargs}))
        return {"recorded": True}

    async def lookup(self, tool: str, args: dict[str, Any]) -> Any:
        self._calls.append(("tools.lookup", {"tool": tool, "args": args}))
        return None


class FakeMemoryContext:
    def __init__(self, client: FakeMemoryClient, scope: dict[str, Any]) -> None:
        self._client = client
        self.scope = FakeScope(scope)
        self.chat = FakeChat(client.calls)
        self.tools = FakeTools(client.calls)
        self.graph = FakeGraph(client.calls, client)
        self.files = FakeFiles(client.calls)

    def derive(self, **changes: Any) -> FakeMemoryContext:
        return FakeMemoryContext(self._client, {**self.scope.model_dump(), **changes})

    async def context(self, query: str, **options: Any) -> FakeBundle:
        self._client.calls.append(("context", {"query": query, "scope": self.scope.model_dump(), **options}))
        if self._client.fail_retrieval:
            raise ConnectionError("memory service unavailable")
        if self._client.retrieval_delay:
            await asyncio.sleep(self._client.retrieval_delay)
        return FakeBundle(query=query)

    async def recall(self, query: str, **options: Any) -> list[Any]:
        self._client.calls.append(("recall", {"query": query, **options}))
        if self._client.fail_retrieval:
            raise ConnectionError("memory service unavailable")
        return []

    async def memories(self, **options: Any) -> list[FakeMemoryRecord]:
        self._client.calls.append(("memories", dict(options)))
        if self._client.fail_retrieval:
            raise ConnectionError("memory service unavailable")
        return [FakeMemoryRecord()]

    async def get_memory(self, memory_id: str) -> FakeMemoryRecord:
        self._client.calls.append(("get_memory", {"memory_id": memory_id}))
        return FakeMemoryRecord(memory_id=memory_id)

    async def forget(self, memory_id: str) -> None:
        self._client.calls.append(("forget", {"memory_id": memory_id}))

    async def verify(self, answer: str, **options: Any) -> FakeGrounding:
        self._client.calls.append(("verify", {"answer": answer, **options}))
        return FakeGrounding()

    async def observe(self, content: str, **kwargs: Any) -> FakeAck:
        self._client.calls.append(
            ("observe", {"content": content, "scope": self.scope.model_dump(), **kwargs})
        )
        if self._client.fail_observation:
            raise ConnectionError("memory service unavailable")
        return FakeAck(observation_id=f"obs_{len(self._client.calls)}")


class FakeMemoryClient:
    """Mimics ``universal_memory.MemoryClient`` closely enough to bind and record."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_retrieval = False
        self.fail_observation = False
        self.retrieval_delay = 0.0

    def bind(self, **scope: Any) -> FakeMemoryContext:
        return FakeMemoryContext(self, scope)

    # -- assertions helpers ------------------------------------------------
    def of(self, kind: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == kind]

    @property
    def observations(self) -> list[dict[str, Any]]:
        return self.of("observe")

    @property
    def retrievals(self) -> list[dict[str, Any]]:
        return self.of("context")



# --------------------------------------------------------------------------- span helpers


def span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def span_by_name(exporter: InMemorySpanExporter, name: str) -> Any:
    for span in exporter.get_finished_spans():
        if span.name == name:
            return span
    raise AssertionError(f"span {name!r} not found in {span_names(exporter)}")


def children_of(exporter: InMemorySpanExporter, parent_name: str) -> list[Any]:
    parent = span_by_name(exporter, parent_name)
    parent_id = parent.get_span_context().span_id
    return [s for s in exporter.get_finished_spans() if s.parent and s.parent.span_id == parent_id]
