"""Test doubles: there are none for the Memory Service.

Every test in this suite talks to a **running** Memory Service. What lives here is a
recording proxy — it forwards each call to the real client, returns the service's real
response, and keeps a note of what was sent so a test can assert on the request as well as
the outcome. Nothing is stubbed: validation, persistence, extraction, indexing and failure
all come from the service.

Why no fakes: every integration defect found in this project — a ``turn_id`` sent without a
``session_id``, observation kinds the service does not accept, a visibility whose audience
nobody was in — was accepted happily by a fake and rejected by the service. A fake tests our
idea of the service, which is the thing that was wrong.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

MEMORY_SERVICE_URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")
MEMORY_API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
#: A port nothing listens on: the honest way to test a dependency being down.
DEAD_SERVICE_URL = "http://127.0.0.1:9"


# ------------------------------------------------------------------------- fault injection


class FaultInjectingTransport(httpx.AsyncBaseTransport):
    """Real HTTP to the real service, with faults applied at the socket.

    The service will not fail on request, so an outage has to be produced somewhere. This
    produces it in the only place that is honest: the network between us and it. A dropped
    connection raises the ``httpx.ConnectError`` a real refusal raises; a stalled path is
    genuinely slow. Everything that does get through is a real request to the real service.
    """

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        #: paths whose connection is refused, e.g. ``{"/v1/context"}``
        self.drop: set[str] = set()
        #: paths held open for this many seconds before proceeding
        self.stall: dict[str, float] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if (delay := self.stall.get(path)) is not None:
            await asyncio.sleep(delay)
        if path in self.drop:
            raise httpx.ConnectError("connection refused", request=request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# --------------------------------------------------------------------------- recording tap


class RecordingMemoryClient:
    """The real ``MemoryClient``, with the calls it makes recorded."""

    def __init__(self, client: Any, faults: FaultInjectingTransport | None = None) -> None:
        self._client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: present only on the fault-injecting fixture; ``None`` on a healthy client
        self.faults = faults

    def bind(self, **scope: Any) -> RecordingContext:
        return RecordingContext(self._client.bind(**scope), self.calls, scope)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- assertion helpers -------------------------------------------------
    def of(self, kind: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == kind]

    @property
    def observations(self) -> list[dict[str, Any]]:
        return self.of("observe")

    @property
    def retrievals(self) -> list[dict[str, Any]]:
        return self.of("context")

    def clear(self) -> None:
        self.calls.clear()


class _RecordingAPI:
    """Forwards attribute calls to a real sub-API, recording each one."""

    def __init__(self, target: Any, calls: list, prefix: str, scope: dict[str, Any]) -> None:
        self._target, self._calls, self._prefix, self._scope = target, calls, prefix, scope

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        async def recorded(*args: Any, **kwargs: Any) -> Any:
            # Record arguments under their parameter names, so a test asserting on
            # ``tools.record(tool=...)`` reads the same whether the caller passed it
            # positionally or by keyword.
            payload = _bind(attr, args, kwargs)
            payload["scope"] = dict(self._scope)
            self._calls.append((f"{self._prefix}.{name}", payload))
            return await attr(*args, **kwargs)

        return recorded


def _bind(fn: Any, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Name every argument of a call, falling back to positions if the signature is opaque."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return {**kwargs, **{f"arg{i}": a for i, a in enumerate(args)}}
    bound.apply_defaults()
    payload = dict(bound.arguments)
    # ``**kwargs`` in the target's signature arrives as a nested dict; flatten it.
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            payload.update(payload.pop(name, {}))
    return payload


class RecordingContext:
    """A real bound ``MemoryContext`` whose calls are recorded before being forwarded."""

    def __init__(self, context: Any, calls: list, scope: dict[str, Any]) -> None:
        self._ctx = context
        self._calls = calls
        self._scope = scope
        self.scope = context.scope
        self.chat = _RecordingAPI(context.chat, calls, "chat", scope)
        self.files = _RecordingAPI(context.files, calls, "files", scope)
        self.graph = _RecordingAPI(context.graph, calls, "graph", scope)
        self.tools = _RecordingAPI(context.tools, calls, "tools", scope)
        self.runs = _RecordingAPI(context.runs, calls, "runs", scope)

    def derive(self, **changes: Any) -> RecordingContext:
        return RecordingContext(
            self._ctx.derive(**changes), self._calls, {**self._scope, **changes}
        )

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self._calls.append((name, {**payload, "scope": self.scope.model_dump(exclude_none=True)}))

    async def context(self, query: str, **options: Any) -> Any:
        self._record("context", {"query": query, **options})
        return await self._ctx.context(query, **options)

    async def recall(self, query: str, **options: Any) -> Any:
        self._record("recall", {"query": query, **options})
        return await self._ctx.recall(query, **options)

    async def observe(self, content: str, **kwargs: Any) -> Any:
        self._record("observe", {"content": content, **kwargs})
        return await self._ctx.observe(content, **kwargs)

    async def memories(self, **options: Any) -> Any:
        self._record("memories", dict(options))
        return await self._ctx.memories(**options)

    async def get_memory(self, memory_id: str) -> Any:
        self._record("get_memory", {"memory_id": memory_id})
        return await self._ctx.get_memory(memory_id)

    async def forget(self, memory_id: str) -> Any:
        self._record("forget", {"memory_id": memory_id})
        return await self._ctx.forget(memory_id)

    async def verify(self, answer: str, **options: Any) -> Any:
        self._record("verify", {"answer": answer, **options})
        return await self._ctx.verify(answer, **options)


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
