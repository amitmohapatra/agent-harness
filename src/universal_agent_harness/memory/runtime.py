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
import logging
from collections.abc import Mapping
from typing import Any

from universal_agent_harness.contracts.artifacts import MemoryObservation
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.errors import AgentError, MemoryUnavailableError
from universal_agent_harness.memory import visibility as vis
from universal_agent_harness.memory.policy import MemoryPolicy
from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import (
    MEMORY_CONTEXT_TOKENS,
    MEMORY_LATENCY,
    MEMORY_OPERATIONS,
)
from universal_agent_harness.telemetry.tracer import HarnessTracer, Stopwatch

log = logging.getLogger("universal_agent_harness.memory")


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
        self._thread_ready = False

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
        require_evidence = options.pop("require_evidence", False)
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
        with self.tracer.memory_span("recall") as span:
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
        vis.check(hints.get("visibility"), self.context)
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
        """Record the turn's user message.

        The ``record_messages`` policy governs whether the harness does this *automatically*;
        calling it yourself always writes. A call that silently did nothing because of a
        config flag elsewhere is worse than no method at all.

        Returns ``None`` only when there is no thread to record into — a message needs a
        conversation.
        """
        if not self._ctx.scope.thread_id:
            return None
        with self.tracer.memory_span("observe", **{N.MEMORY_KIND: "message.user"}) as span:
            span.set_input(text, category="memory")
            return await self._write(
                self._ctx.chat.user(
                    text, idempotency_key=self.context.idempotency_key("msg", "user", text),
                    **metadata,
                ),
                span,
                "record_input",
            )

    async def record_output(self, text: str, /, **metadata: Any) -> Any | None:
        """Record the agent's answer as the assistant turn. Always writes when called."""
        if not self._ctx.scope.thread_id:
            return None
        with self.tracer.memory_span("observe", **{N.MEMORY_KIND: "message.assistant"}) as span:
            span.set_input(text, category="memory")
            return await self._write(
                self._ctx.chat.assistant(
                    text,
                    idempotency_key=self.context.idempotency_key("msg", "assistant", text),
                    **metadata,
                ),
                span,
                "record_output",
            )

    async def share(self, content: str, /, *, group: str | None = None, **metadata: Any) -> Any:
        """Publish to the agent group — the only way memory crosses agents.

        A group is required because it *is* the audience: without one there is nobody to
        share with. Declare it once (``AgentHarness(defaults={"agent_group_id": ...})`` or
        ``harness.wrap(..., agent_group="crew")``) and it flows automatically; ``group=``
        overrides it for a single call.
        """
        if group:
            # Rebind the SDK context too: the scope that travels with the write is built
            # from it, so changing only the harness context would send the old group.
            self.context = self.context.with_fields(agent_group_id=group)
            if hasattr(self._ctx, "derive"):
                self._ctx = self._ctx.derive(agent_group_id=group)
        if not self.context.agent_group_id:
            from universal_agent_harness.contracts.errors import ConfigurationError  # noqa: PLC0415

            raise ConfigurationError(
                "share() needs an agent group: the group is the audience. Set it once with "
                'AgentHarness(defaults={"agent_group_id": "..."}), per agent with '
                'harness.wrap(..., agent_group="..."), or per call with share(..., group="...").',
                source="memory.share",
            )
        return await self.observe(
            MemoryObservation(
                content=content,
                kind="AGENT_RESULT",
                hints={"memory_type": "SHARED", "visibility": "AGENT_GROUP"},
                metadata=metadata,
                idempotency_key=self.context.idempotency_key("share", content),
            )
        )

    async def remember(
        self,
        content: str,
        /,
        *,
        memory_type: str = "SEMANTIC",
        lifetime: str = "LONG_TERM",
        visibility: str | None = None,
        **metadata: Any,
    ) -> Any | None:
        """Write a *typed* memory: what kind of knowledge it is, how long it should live and
        who may see it. ``observe`` lets the service classify; this states it explicitly.

        ``memory_type``: SEMANTIC, EPISODIC, PROCEDURAL, PREFERENCE, DECISION, OUTCOME,
        FAILURE, SHARED... ``lifetime``: EPHEMERAL, SHORT_TERM, LONG_TERM, ARCHIVAL.
        ``visibility``: PRIVATE, RUN, AGENT_GROUP, THREAD, USER, WORK, WORKSPACE, TENANT.
        """
        hints: dict[str, Any] = {"memory_type": memory_type, "lifetime": lifetime}
        if visibility:
            vis.check(visibility, self.context)
            hints["visibility"] = visibility
        with self.tracer.memory_span(
            "remember",
            **{
                N.MEMORY_TYPE: memory_type,
                N.MEMORY_LIFETIME: lifetime,
                N.MEMORY_VISIBILITY: visibility
                or ("RUN" if self.policy.private_by_default else None),
            },
        ) as span:
            span.set_input(content, category="memory")
            ack = await self._write(
                self._ctx.observe(
                    content,
                    kind="EVENT",
                    idempotency_key=self.context.idempotency_key("remember", memory_type, content),
                    hints={**self.policy.visibility_hints(), **hints},
                    **metadata,
                ),
                span,
                "remember",
            )
        return ack

    async def forget(self, memory_id: str, /) -> None:
        """Delete a memory. Idempotent in the service; traced here because it is a write."""
        with self.tracer.memory_span("forget", **{N.MEMORY_ID: memory_id}) as span:
            await self._write(self._ctx.forget(memory_id), span, "forget")

    async def memories(self, **options: Any) -> list[Any]:
        """The inventory view: current memories anchored to this execution's scopes.

        ``recall`` is the ranked, query-driven view; this is "what do we hold about this
        user / thread / run", for audit screens and debugging.
        """
        with self.tracer.memory_span("list") as span:
            items = await self._read(self._ctx.memories(**options), span, "list", default=[])
            span.set(**{N.MEMORY_RESULT_COUNT: len(items)})
        return items

    async def get(self, memory_id: str, /) -> Any | None:
        with self.tracer.memory_span("list", **{N.MEMORY_ID: memory_id}) as span:
            return await self._read(self._ctx.get_memory(memory_id), span, "get")

    async def history(self, *, limit: int = 50, include_internal: bool = False) -> list[Any]:
        """The conversation window: what was actually said in this thread."""
        with self.tracer.memory_span("history") as span:
            messages = await self._read(
                self._ctx.chat.history(limit=limit, include_internal=include_internal),
                span,
                "history",
                default=[],
            )
            span.set(**{N.MEMORY_RESULT_COUNT: len(messages)})
        return messages

    async def graph_query(
        self,
        query: str | None = None,
        /,
        *,
        entities: list[str] | None = None,
        hops: int = 1,
        as_of: Any = None,
    ) -> Any | None:
        """Knowledge-graph traversal: resolve entities and walk a bounded neighbourhood.
        ``as_of`` gives the temporal view — what the graph believed at that time."""
        with self.tracer.memory_span("graph", **{N.MEMORY_HOPS: hops}) as span:
            span.set_input(query or entities, category="memory")
            answer = await self._read(
                self._ctx.graph.query(query, entities=entities, hops=hops, as_of=as_of),
                span,
                "graph",
            )
            if answer is not None:
                span.set(**{N.MEMORY_RESULT_COUNT: len(getattr(answer, "facts", []) or [])})
        return answer

    async def add_document(self, file: Any, /, **options: Any) -> Any | None:
        """Ingest a document so its chunks become retrievable knowledge (the RAG corpus).

        Ingestion is asynchronous in the service: the handle comes back immediately and the
        document becomes retrievable once parsing and indexing finish.
        """
        visibility = options.get("visibility")
        vis.check(visibility, self.context)
        # A document inherits the thread's audience unless told otherwise, and a thread only
        # grants that audience once it exists. Ingesting into a thread that was never written
        # to produces chunks nobody — including the caller — can retrieve, so create it.
        if self.context.thread_id and visibility in (None, "THREAD"):
            await self._ensure_thread()
        with self.tracer.memory_span("ingest") as span:
            handle = await self._write(self._ctx.files.add(file, **options), span, "ingest")
            if handle is not None:
                span.set(**{N.MEMORY_DOCUMENT_ID: getattr(handle, "document_id", None)})
        return handle

    async def verify(self, answer: str, /, **options: Any) -> Any | None:
        """Grounding check: claim by claim, is this answer supported by the evidence?

        Returns the service's grounding report. Use it before returning an answer that must
        be defensible; it is a *read* and does not write memory.
        """
        with self.tracer.memory_span("verify") as span:
            report = await self._read(self._ctx.verify(answer, **options), span, "verify")
            if report is not None:
                span.set(
                    **{
                        N.MEMORY_HALLUCINATION_RATE: getattr(
                            report, "per_claim_hallucination_rate", None
                        ),
                        N.MEMORY_GROUNDED: getattr(report, "grounded", None),
                    }
                )
        return report

    async def record_outcome(self, *, success: bool, note: str | None = None) -> Any | None:
        """Tell the service whether this run worked.

        Tool memory learns procedures from *successful* trajectories, and without a label it
        has to guess: a run with no failing call counts as a weak positive, but only once it
        is older than the service's window — hours later — and a run that failed for any
        reason other than a failing tool call is never counted as a negative at all. The
        harness knows the answer the moment the turn ends, so it says so.

        Recorded once per run, keyed by ``agent_run_id``; the service upserts, so a retry of
        the same turn overwrites rather than duplicates.
        """
        run_id = self.context.agent_run_id
        if not run_id:
            return None
        with self.tracer.memory_span("record_outcome") as span:
            span.set(**{N.STATUS: "ok" if success else "error"})
            return await self._write(
                self._ctx.runs.outcome(run_id, success=success, note=note),
                span,
                "record_outcome",
            )

    async def _ensure_thread(self) -> None:
        """Create the thread if it does not exist yet. Idempotent in the service."""
        if self._thread_ready:
            return
        try:
            await _with_timeout(self._ctx.chat.create(), self.observation_timeout)
        except Exception as exc:  # a pre-existing thread, or a service that does not need it
            log.debug("thread creation skipped: %s", exc)
        self._thread_ready = True

    # -- shared plumbing for the operations above --------------------------------------
    async def _read(self, awaitable: Any, span: Any, operation: str, default: Any = None) -> Any:
        """A read follows the degradation policy: ``fail_closed`` raises, otherwise the
        caller gets ``default`` and the failure is recorded."""
        watch = Stopwatch()
        try:
            value = await _with_timeout(awaitable, self.retrieval_timeout)
        except Exception as exc:
            self.last_error = AgentError.of(exc, source=f"memory.{operation}")
            span.error(exc, **{N.STATUS: "error"})
            self._metric(operation, "error", watch.ms)
            if self.fail_closed:
                raise MemoryUnavailableError(
                    f"memory {operation} failed: {exc}", source=f"memory.{operation}"
                ) from exc
            return default
        span.ok()
        self._metric(operation, "ok", watch.ms)
        return value

    async def _write(self, awaitable: Any, span: Any, operation: str) -> Any:
        """A write never silently disappears: the failure propagates to the caller (or to
        the writeback handler, when it was scheduled)."""
        watch = Stopwatch()
        try:
            value = await _with_timeout(awaitable, self.observation_timeout)
        except Exception as exc:
            self.last_error = AgentError.of(exc, source=f"memory.{operation}")
            span.error(exc, **{N.STATUS: "error"})
            self._metric(operation, "error", watch.ms)
            raise
        span.ok()
        self._metric(operation, "ok", watch.ms)
        return value

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

    #: The SDK sub-APIs are unavailable without a client; attribute access returns ``None``
    #: so ``if runtime.memory.graph:`` reads naturally.
    chat = None
    files = None
    graph = None
    tools = None

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

    async def remember(self, content: str, /, **kwargs: Any) -> None:
        return None

    async def forget(self, memory_id: str, /) -> None:
        return None

    async def memories(self, **options: Any) -> list[Any]:
        return []

    async def get(self, memory_id: str, /) -> None:
        return None

    async def history(self, **options: Any) -> list[Any]:
        return []

    async def graph_query(self, query: str | None = None, /, **options: Any) -> None:
        return None

    async def add_document(self, file: Any, /, **options: Any) -> None:
        return None

    async def verify(self, answer: str, /, **options: Any) -> None:
        return None

    async def record_outcome(self, **kwargs: Any) -> None:
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
