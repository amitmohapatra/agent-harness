"""The interceptor pipeline (§19/§20).

Ordering is deterministic and explicit: ``before`` runs in ascending ``order``, ``after``
and ``on_error`` in descending order, so the pipeline nests like an onion around the
developer's agent. Interceptors are pure functions of (request|result, runtime) — they
return the (possibly modified) value rather than mutating a shared object.

Cost is O(I) with a small, bounded I (§64): the chain is materialised once at harness
construction, not rebuilt per execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from universal_agent_contracts.errors import AgentError
from universal_agent_contracts.messages import AgentRequest, AgentResponse

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class Order:
    """Canonical positions.

    ``before`` runs in ascending order and ``after`` in descending order, so these numbers
    produce exactly the pipeline of §19 in both directions::

        before:  identity -> policy -> memory context -> telemetry -> langfuse -> timeout
                 -> user -> [agent]
        after:   [agent] -> result validation -> memory observation -> evaluation -> user
                 -> timeout -> langfuse -> telemetry -> memory context -> policy -> identity

    The post-execution trio is numbered so that validation happens first (the result is
    finalised), then the memory write (which uses the finalised result), then the evaluation
    event (which reports what was finally produced).
    """

    IDENTITY = 10
    POLICY = 20
    MEMORY_CONTEXT = 30
    TELEMETRY = 40
    OBSERVABILITY = 45
    TIMEOUT = 50
    USER = 60
    EVALUATION = 70
    MEMORY_OBSERVATION = 80
    RESULT_VALIDATION = 90


class BaseInterceptor:
    """Convenience base: override only what you need."""

    name = "interceptor"
    order = Order.USER

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        return request

    async def after(self, result: AgentResponse, runtime: AgentRuntime) -> AgentResponse:
        return result

    async def on_error(self, error: AgentError, runtime: AgentRuntime) -> AgentResponse | None:
        return None


class InterceptorChain:
    """An ordered, immutable pipeline."""

    __slots__ = ("_after", "_before")

    def __init__(self, interceptors: Iterable[BaseInterceptor]) -> None:
        ordered = sorted(interceptors, key=lambda i: (i.order, i.name))
        self._before: tuple[BaseInterceptor, ...] = tuple(ordered)
        self._after: tuple[BaseInterceptor, ...] = tuple(reversed(ordered))

    def __len__(self) -> int:
        return len(self._before)

    @property
    def names(self) -> list[str]:
        return [i.name for i in self._before]

    @property
    def interceptors(self) -> Sequence[BaseInterceptor]:
        return self._before

    def with_extra(self, extra: Iterable[BaseInterceptor]) -> InterceptorChain:
        return InterceptorChain([*self._before, *extra])

    async def before(self, request: AgentRequest, runtime: AgentRuntime) -> AgentRequest:
        for interceptor in self._before:
            request = await interceptor.before(request, runtime)
        return request

    async def after(self, result: AgentResponse, runtime: AgentRuntime) -> AgentResponse:
        for interceptor in self._after:
            result = await interceptor.after(result, runtime)
        return result

    async def on_error(self, error: AgentError, runtime: AgentRuntime) -> AgentResponse | None:
        """First interceptor that produces a result wins; the rest still see the error."""
        recovered: AgentResponse | None = None
        for interceptor in self._after:
            produced = await interceptor.on_error(error, runtime)
            if produced is not None and recovered is None:
                recovered = produced
        return recovered
