"""``MemoryRuntime``: the Memory Service, as an agent sees it.

The harness depends on the Memory Service's **SDK contract only** (§10) — a
``MemoryContext`` bound to this execution's scope. Everything else (bundle shape, evidence
report, tool memory) stays the service's business; the harness reads a handful of documented
facts off a bundle for telemetry and passes the bundle itself through to the agent untouched.

Reads are bounded by the memory deadline and degrade per :class:`MemoryConfig.failure_mode`:
``non_blocking`` runs the agent without context and records a warning; ``fail_closed``
turns the failure into a ``MemoryUnavailableError``. Writes always propagate their failure
to the writeback handler rather than being silently dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from universal_agent_harness.contracts.artifacts import MemoryObservation
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.errors import AgentError, MemoryUnavailableError
from universal_agent_harness.memory.policy import MemoryPolicy
from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import (
    MEMORY_CONTEXT_TOKENS,
    MEMORY_LATENCY,
    MEMORY_OPERATIONS,
)
from universal_agent_harness.telemetry.tracer import HarnessTracer, Stopwatch


class MemoryRuntime:
    """Memory operations for one agent execution."""

    enabled = True

    def __init__(
        self,
        memory_context: Any,
        *,
        context: AgentExecutionContext,
        policy: MemoryPolicy,
        tracer: HarnessTracer,
        retrieval_timeout: float | None = 10.0,
        observation_timeout: float | None = 15.0,
        fail_closed: bool = False,
    ) -> None:
        self._ctx = memory_context
        self.context = context
        self.policy = policy
        self.tracer = tracer
        self.retrieval_timeout = retrieval_timeout
        self.observation_timeout = observation_timeout
        self.fail_closed = fail_closed
        self.last_error: AgentError | None = None

    # -- underlying SDK handle (escape hatch for advanced callers) ------------------
    @property
    def sdk(self) -> Any:
        """The bound ``universal_memory.MemoryContext``. Use it for anything the harness
        does not wrap (graph queries, files, explicit ``remember``...)."""
        return self._ctx

    @property
    def chat(self) -> Any:
        return self._ctx.chat

    @property
    def files(self) -> Any:
        return self._ctx.files

    @property
    def graph(self) -> Any:
        return self._ctx.graph

    # -- retrieval -------------------------------------------------------------------
    async def retrieve(self, query: str, /, **options: Any) -> Any | None:
        """The bounded context bundle for ``query``, or ``None`` when memory is degraded."""
        if not query or not query.strip():
            return None
        budget = options.pop("token_budget", self.policy.token_budget)
        require_evidence = options.pop("require_evidence", self.policy.require_evidence)
        watch = Stopwatch()
        with self.tracer.memory_span("retrieve", **{N.MEMORY_KIND: "context"}) as span:
            span.set_input(query, category="memory")
            try:
                bundle = await _with_timeout(
                    self._ctx.context(
                        query,
                        token_budget=budget,
                        require_evidence=require_evidence,
                        **options,
                    ),
                    self.retrieval_timeout,
                )
            except Exception as exc:
                self.last_error = AgentError.of(exc, source="memory.retrieve")
                span.error(exc, **{N.STATUS: "error"})
                self._metric("retrieve", "error", watch.ms)
                if self.fail_closed or require_evidence:
                    raise MemoryUnavailableError(
                        f"memory retrieval failed: {exc}", source="memory.retrieve"
                    ) from exc
                return None
            facts = self.describe(bundle)
            span.set_attributes(_span_attributes(facts))
            span.set_output(getattr(bundle, "rendered", None), category="memory")
            span.ok()
        self._metric("retrieve", "ok", watch.ms)
        if facts.get("token_estimate") is not None:
            self.tracer.metrics.value(
                MEMORY_CONTEXT_TOKENS, float(facts["token_estimate"]), operation="retrieve"
            )
        return bundle

    async def recall(self, query: str, /, **options: Any) -> list[Any]:
        """Ranked evidence without bundle assembly (the service's ``recall``)."""
        with self.tracer.memory_span("retrieve", **{N.MEMORY_KIND: "recall"}) as span:
            span.set_input(query, category="memory")
            try:
                return await _with_timeout(
                    self._ctx.recall(query, **options), self.retrieval_timeout
                )
            except Exception as exc:
                self.last_error = AgentError.of(exc, source="memory.recall")
                span.error(exc)
                if self.fail_closed:
                    raise MemoryUnavailableError(str(exc), source="memory.recall") from exc
                return []

    # -- writes ----------------------------------------------------------------------
    async def observe(self, observation: MemoryObservation, /) -> Any | None:
        """Submit one observation. Idempotent: the key is derived from this execution."""
        key = observation.idempotency_key or self.context.idempotency_key(
            "obs", observation.kind, observation.content
        )
        watch = Stopwatch()
        hints = {**self.policy.visibility_hints(), **observation.hints}
        with self.tracer.memory_span("observe", **{N.MEMORY_KIND: observation.kind}) as span:
            span.set_input(observation.content, category="memory")
            try:
                ack = await _with_timeout(
                    self._ctx.observe(
                        observation.content,
                        kind=observation.kind,
                        idempotency_key=key,
                        hints=hints,
                        **observation.metadata,
                    ),
                    self.observation_timeout,
                )
            except Exception as exc:
                self.last_error = AgentError.of(exc, source="memory.observe")
                span.error(exc, **{N.STATUS: "error"})
                self._metric("observe", "error", watch.ms)
                raise
            span.set(**{N.MEMORY_OBSERVATION_ID: getattr(ack, "observation_id", None)})
            span.ok()
        self._metric("observe", "ok", watch.ms)
        return ack

    async def record_input(self, text: str, /, **metadata: Any) -> Any | None:
        """The turn's user message, when the application asked the harness to record it."""
        if not self.policy.record_messages or not self._ctx.scope.thread_id:
            return None
        key = self.context.idempotency_key("msg", "user", text)
        return await self._ctx.chat.user(text, idempotency_key=key, **metadata)

    async def record_output(self, text: str, /, **metadata: Any) -> Any | None:
        if not self.policy.record_messages or not self._ctx.scope.thread_id:
            return None
        key = self.context.idempotency_key("msg", "assistant", text)
        return await self._ctx.chat.assistant(text, idempotency_key=key, **metadata)

    async def share(self, content: str, /, **metadata: Any) -> Any:
        """Publish to the agent group explicitly — the only way memory crosses agents."""
        return await self.observe(
            MemoryObservation(
                content=content,
                kind="AGENT_RESULT",
                hints={"memory_type": "SHARED", "visibility": "AGENT_GROUP"},
                metadata=metadata,
                idempotency_key=self.context.idempotency_key("share", content),
            )
        )

    # -- tool memory (used by the tool client when enabled) ---------------------------
    @property
    def tools(self) -> Any:
        return self._ctx.tools

    async def record_tool_call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        output: Any = None,
        status: str = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        task: str = "",
        step: int | None = None,
    ) -> Any | None:
        return await self._ctx.tools.record(
            tool,
            dict(args),
            output=output,
            status=status,
            error_class=error_class,
            latency_ms=latency_ms,
            task=task,
            step=step,
        )

    # -- introspection ----------------------------------------------------------------
    def describe(self, bundle: Any, /) -> dict[str, Any]:
        """The documented facts the harness reads off a bundle. Never its content."""
        if bundle is None:
            return {}
        evidence = getattr(bundle, "evidence", None)
        counts = {
            group: len(getattr(bundle, group, []) or [])
            for group in ("memories", "knowledge", "graph_facts", "summaries")
        }
        return {
            "bundle_id": getattr(bundle, "bundle_id", None) or None,
            "query_type": getattr(bundle, "query_type", None),
            "token_estimate": getattr(bundle, "token_estimate", None),
            "token_budget": getattr(bundle, "token_budget", None),
            "cache_hit": getattr(bundle, "cache_hit", None),
            "evidence_status": getattr(evidence, "status", None),
            "item_count": sum(counts.values()),
            "counts": counts,
        }

    def _metric(self, operation: str, status: str, ms: float) -> None:
        self.tracer.metrics.count(MEMORY_OPERATIONS, operation=operation, status=status)
        self.tracer.metrics.duration(MEMORY_LATENCY, ms, operation=operation, status=status)


class NoOpMemoryRuntime:
    """Used when memory is disabled or no client was provided. Every call is a no-op."""

    enabled = False
    policy = MemoryPolicy(retrieve_before=False, observe_input=False, observe_output=False)
    last_error = None

    def __init__(self, context: AgentExecutionContext | None = None) -> None:
        self.context = context

    @property
    def sdk(self) -> Any:
        return None

    async def retrieve(self, query: str, /, **options: Any) -> None:
        return None

    async def recall(self, query: str, /, **options: Any) -> list[Any]:
        return []

    async def observe(self, observation: MemoryObservation, /) -> None:
        return None

    async def record_input(self, text: str, /, **metadata: Any) -> None:
        return None

    async def record_output(self, text: str, /, **metadata: Any) -> None:
        return None

    async def share(self, content: str, /, **metadata: Any) -> None:
        return None

    async def record_tool_call(self, *args: Any, **kwargs: Any) -> None:
        return None

    def describe(self, bundle: Any, /) -> dict[str, Any]:
        return {}


def _span_attributes(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        N.MEMORY_BUNDLE_ID: facts.get("bundle_id"),
        N.MEMORY_QUERY_TYPE: facts.get("query_type"),
        N.MEMORY_EVIDENCE_STATUS: facts.get("evidence_status"),
        N.MEMORY_TOKEN_ESTIMATE: facts.get("token_estimate"),
        N.MEMORY_TOKEN_BUDGET: facts.get("token_budget"),
        N.MEMORY_CACHE_HIT: facts.get("cache_hit"),
        N.MEMORY_ITEM_COUNT: facts.get("item_count"),
    }


async def _with_timeout(awaitable: Any, timeout: float | None) -> Any:  # noqa: ASYNC109
    if timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout)
