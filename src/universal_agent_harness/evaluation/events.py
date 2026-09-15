"""Evaluation event emission (§49) and lifecycle dispatch (§18).

Both are observer-shaped and both are failure-isolated: an evaluator or a listener that
raises is logged, never propagated. Evaluation is asynchronous unless an application
explicitly asks for synchronous scoring (§29).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from universal_agent_harness.contracts.events import AgentEvalEvent
from universal_agent_harness.runtime.logging import get_logger

log = get_logger("universal_agent_harness.evaluation")


class LifecycleDispatcher:
    """Fan lifecycle events out to listeners. O(L) in the number of listeners (§64)."""

    __slots__ = ("_async_listeners", "_listeners")

    def __init__(self, listeners: Sequence[Any] = ()) -> None:
        self._listeners: list[Any] = []
        self._async_listeners: list[Any] = []
        for listener in listeners:
            self.add(listener)

    def add(self, listener: Any) -> None:
        handler = getattr(listener, "on_event", listener)
        if inspect.iscoroutinefunction(handler):
            self._async_listeners.append(handler)
        else:
            self._listeners.append(handler)

    @property
    def count(self) -> int:
        return len(self._listeners) + len(self._async_listeners)

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        """Synchronous fan-out; async listeners are scheduled if a loop is running."""
        for handler in self._listeners:
            try:
                handler(str(event), payload)
            except Exception:
                log.warning("lifecycle listener failed", lifecycle_event=str(event))
        if not self._async_listeners:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - sync context without a loop
            return
        for handler in self._async_listeners:
            task = loop.create_task(_safe(handler, str(event), payload))
            task.add_done_callback(lambda t: t.exception())


async def _safe(handler: Any, event: str, payload: Mapping[str, Any]) -> None:
    try:
        await handler(event, payload)
    except Exception:
        log.warning("async lifecycle listener failed", lifecycle_event=event)


class LoggingEvaluationSink:
    """The default sink: one structured log line per evaluation event."""

    name = "logging"

    async def emit(self, event: AgentEvalEvent) -> None:
        log.info(
            "agent.eval",
            agent_id=event.agent_id,
            agent_run_id=event.agent_run_id,
            status=event.status,
            latency_ms=event.latency_ms,
            skills=",".join(event.skills) or None,
            trace_id=event.trace_id,
        )


class CollectingEvaluationSink:
    """Keeps the last ``max_items`` events in memory. For tests and local inspection only —
    bounded on purpose (§65)."""

    name = "collecting"

    def __init__(self, max_items: int = 500) -> None:
        self.events: list[AgentEvalEvent] = []
        self.max_items = max_items

    async def emit(self, event: AgentEvalEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_items:
            del self.events[: len(self.events) - self.max_items]


class CompositeEvaluationSink:
    """Several sinks behind one port, each isolated from the others' failures."""

    name = "composite"

    def __init__(self, sinks: Sequence[Any]) -> None:
        self.sinks = [s for s in sinks if s is not None]

    async def emit(self, event: AgentEvalEvent) -> None:
        for sink in self.sinks:
            try:
                await sink.emit(event)
            except Exception:
                log.warning("evaluation sink failed", sink=type(sink).__name__)


class NoOpEvaluationProvider:
    """Scores go nowhere. Keeps agent code identical whether or not scoring is configured."""

    name = "noop"

    async def score(self, name: str, value: float | str, /, **kwargs: Any) -> None:
        return None

    async def submit_dataset_item(self, dataset: str, item: Mapping[str, Any]) -> None:
        return None

    async def submit_feedback(self, feedback: Mapping[str, Any]) -> None:
        return None
