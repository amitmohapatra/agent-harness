"""Memory around the execution (§10).

``before``: fetch the bundle and hand it to the agent on the runtime.
``after``: schedule observations — *after* the result has been produced, so the turn never
waits for consolidation, KG updates, summaries or eval processing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from universal_agent_harness.contracts.artifacts import MemoryObservation
from universal_agent_harness.contracts.errors import AgentError, ErrorCategory
from universal_agent_harness.contracts.events import LifecycleEvent
from universal_agent_harness.contracts.messages import AgentRequest, AgentResult
from universal_agent_harness.interceptors.base import BaseInterceptor, Order
from universal_agent_harness.memory.writeback import WritebackQueue

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.evaluation.events import LifecycleDispatcher
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class MemoryContextInterceptor(BaseInterceptor):
    """Pre-execution retrieval."""

    name = "memory_context"
    order = Order.MEMORY_CONTEXT

    def __init__(self, events: LifecycleDispatcher | None = None) -> None:
        self.events = events

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        memory = runtime.memory
        if not memory.enabled or not memory.policy.retrieve_before:
            return request
        query = request.query
        if not query:
            return request
        bundle = await memory.retrieve(query)
        runtime.memory_context = bundle
        facts = memory.describe(bundle)
        if bundle is None and memory.last_error is not None:
            runtime.state.setdefault("warnings", []).append(
                ("MEMORY_DEGRADED", "memory retrieval failed; running without context")
            )
        if self.events is not None:
            self.events.emit(
                LifecycleEvent.CONTEXT_LOADED,
                {"context": runtime.context, "bundle": facts, "has_context": bundle is not None},
            )
        runtime.state["memory_facts"] = facts
        return request


class MemoryObservationInterceptor(BaseInterceptor):
    """Post-execution writes, off the critical path."""

    name = "memory_observation"
    order = Order.MEMORY_OBSERVATION

    def __init__(
        self, writeback: WritebackQueue | None = None, *, fail_closed: bool = False
    ) -> None:
        self.writeback = writeback or WritebackQueue()
        #: With ``fail_closed`` a failed write fails the execution; otherwise the result is
        #: returned with an explicit warning. Either way the failure is never silent (§77).
        self.fail_closed = fail_closed

    async def after(self, result: AgentResult, runtime: AgentRuntime) -> AgentResult:
        memory = runtime.memory
        if not memory.enabled:
            return result
        policy = memory.policy
        if not result.status.ok:
            # Nothing is observed from a failed turn — but the failure itself is worth saying
            # out loud, because tool memory only ever infers negatives from a failing tool
            # call and would otherwise treat this run as neutral. Inline on purpose: the turn
            # has already failed, and a label that races the caller's shutdown is worse than
            # one that costs a failing turn a few milliseconds.
            await self._label(runtime, success=False, note=_failure_note(result))
            return result
        observations = list(result.memory_observations)
        request: AgentRequest | None = runtime.state.get("request")

        if policy.observe_input and request is not None and request.query:
            observations.append(
                MemoryObservation(
                    # "what the agent was asked" is an EVENT; AGENT_RESULT is its answer.
                    content=request.query,
                    kind="EVENT",
                    metadata={"agent_id": runtime.agent_id},
                    idempotency_key=runtime.idempotency_key("obs", "input", request.query),
                )
            )
        if policy.observe_output:
            summary = _summarize(result)
            if summary:
                observations.append(
                    MemoryObservation(
                        content=summary,
                        kind="AGENT_RESULT",
                        metadata={"agent_id": runtime.agent_id, "status": str(result.status)},
                        idempotency_key=runtime.idempotency_key("obs", "output", summary),
                    )
                )
        if policy.observe_claims:
            for claim in result.claims:
                observations.append(
                    MemoryObservation(
                        content=claim.text,
                        kind="AGENT_RESULT",
                        # the *kind* says where it came from; the hint says what it is
                        hints={"memory_type": "SEMANTIC"},
                        metadata={"claim_id": claim.claim_id, "confidence": claim.confidence},
                        idempotency_key=runtime.idempotency_key("obs", "claim", claim.claim_id),
                    )
                )
        if not observations and not policy.record_outcome:
            return result

        # A turn with nothing to observe still has an outcome to report — and that report
        # goes through the same writeback path as everything else. Doing it inline put an
        # HTTP round trip back on the critical path for exactly the agents that write
        # nothing, which is the opposite of what writeback is for.
        if policy.writeback:
            scheduled = self.writeback.submit(
                self._write(runtime, observations, result),
                name=f"memory-writeback:{runtime.run_id}",
            )
            if scheduled is not None:
                return result
            runtime.logger.warning("memory writeback saturated; writing inline")
        try:
            await self._write(runtime, observations, result)
        except Exception as exc:
            if self.fail_closed:
                raise
            return result.add_warning(
                "MEMORY_WRITE_FAILED",
                "memory observations could not be written",
                error=str(exc),
            )
        return result

    async def on_error(self, error: AgentError, runtime: AgentRuntime) -> None:
        """``after`` does not run when the harness re-raises, so the label is recorded here.

        Both paths are idempotent: the service upserts the outcome on (tenant, run), so a
        turn that reaches both records one row, not two.
        """
        await self._label(runtime, success=False, note=str(error.category))
        return None

    async def _label(self, runtime: AgentRuntime, *, success: bool, note: str) -> None:
        """Record the run outcome, never letting it disturb the turn it describes."""
        memory = runtime.memory
        if not memory.enabled or not memory.policy.record_outcome:
            return
        try:
            await memory.record_outcome(success=success, note=note)
        except Exception as exc:  # a label is not worth failing or re-failing a turn over
            runtime.logger.warning("memory.outcome.failed", error_message=str(exc))

    async def _write(
        self, runtime: AgentRuntime, observations: list[MemoryObservation], result: AgentResult
    ) -> None:
        """Write everything this turn produced. Each write stands on its own.

        These used to be one sequence of awaits inside a single ``try``, so the first failure
        discarded every write after it: a turn with ``record_messages`` on lost its question,
        its answer *and* every claim because ``/v1/messages`` rejected the scope. They are
        independent facts about the turn; one being unwritable is not a reason to lose the
        rest. Failures are collected and reported together, so the caller still learns that
        something was lost and what.
        """
        memory = runtime.memory
        policy = memory.policy
        request: AgentRequest | None = runtime.state.get("request")
        summary = _summarize(result)

        # (name, thunk) so nothing is turned into a coroutine until it is about to be awaited
        writes: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        if policy.record_outcome:
            writes.append(
                ("outcome", lambda: memory.record_outcome(
                    success=result.status.ok, note=str(result.status)
                ))
            )
        if policy.record_messages:
            if request is not None and request.query:
                writes.append(("message.user", lambda q=request.query: memory.record_input(q)))
            if summary:
                writes.append(("message.assistant", lambda t=summary: memory.record_output(t)))
        writes.extend(
            (f"observe[{o.kind}]", lambda o=o: memory.observe(o)) for o in observations
        )

        failures: list[tuple[str, Exception]] = []
        for name, thunk in writes:
            try:
                await thunk()
            except Exception as exc:  # noqa: PERF203 - one failure must not stop the others
                failures.append((name, exc))

        if failures:
            first = failures[0][1]
            error = AgentError.of(first, category=ErrorCategory.MEMORY, source="memory.observe")
            runtime.logger.warning(
                "memory.writeback.failed",
                error_code=error.code,
                error_message=error.message,
                failed=[name for name, _ in failures],
                written=len(writes) - len(failures),
            )
            raise MemoryWriteError(failures) from first


class MemoryWriteError(Exception):
    """One or more writes for a turn failed. Names which, so the warning is actionable."""

    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{name}: {exc}" for name, exc in failures)
        super().__init__(f"{len(failures)} memory write(s) failed - {detail}")


def _failure_note(result: AgentResult) -> str:
    error = result.error
    return f"{result.status}: {error.category}" if error else str(result.status)


def _summarize(result: AgentResult) -> str | None:
    """What the harness writes back as the agent's output.

    Text data is written verbatim; structured data is *not* dumped wholesale into memory —
    an agent that wants structured memory should return explicit ``memory_observations``.
    """
    if isinstance(result.data, str) and result.data.strip():
        return result.data.strip()
    if result.claims:
        return "\n".join(c.text for c in result.claims)
    return None
