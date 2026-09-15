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
class FakeBundle:
    query: str = ""
    query_type: str = "FACTUAL"
    bundle_id: str = "bundle_1"
    rendered: str = "remembered: SKU-1 stock is low"
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
        return []

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
