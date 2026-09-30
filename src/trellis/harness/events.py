"""A run's event stream: numbered ``RunEvent``s, delivered to whoever is listening.

Nothing is built when nobody listens (``run()`` without a stream costs a length check per
event). Delivery is synchronous and non-blocking: a listener is a callable that enqueues.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from trellis.contracts import (
    AgentError,
    AgentExecutionContext,
    Interrupt,
    RunEvent,
    RunEventType,
    RunOutcome,
)

Listener = Callable[[RunEvent], None]

#: The ``CUSTOM`` event name a degraded side effect (a failed memory write...) is reported as.
WARNING: Final = "warning"
#: The ``CUSTOM`` event name of a write-tier tool call, announced to the people watching.
NOTICE: Final = "tool_notice"
#: The ``CUSTOM`` event name of ``trellis.current().log(...)``.
LOG: Final = "log"


class RunEvents:
    """One attempt's emitter. ``sequence`` is per attempt, as the contract orders it."""

    __slots__ = ("_listeners", "attempt", "context", "sequence")

    def __init__(self, context: AgentExecutionContext, attempt: int = 1) -> None:
        self.context = context
        self.attempt = attempt
        self.sequence = 0
        self._listeners: list[Listener] = []

    def listen(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, type: RunEventType, **fields: Any) -> None:
        if not self._listeners:
            return
        event = RunEvent.of(self.context, type, self.sequence, attempt=self.attempt, **fields)
        self.sequence += 1
        for listener in self._listeners:
            listener(event)

    # ------------------------------------------------------------------ the vocabulary
    def text(self, type: RunEventType, message_id: str, delta: str | None = None) -> None:
        data: dict[str, Any] = {"role": "assistant"}
        if delta is not None:
            data["delta"] = delta
        self.emit(type, message_id=message_id, data=data)

    def tool(self, type: RunEventType, tool_call_id: str, **data: Any) -> None:
        self.emit(type, tool_call_id=tool_call_id, data=data)

    def custom(self, name: str, **data: Any) -> None:
        self.emit(RunEventType.CUSTOM, data={"name": name, **data})

    def warning(self, code: str, message: str) -> None:
        self.custom(WARNING, code=code, message=message)

    def finished(
        self,
        outcome: RunOutcome,
        *,
        error: AgentError | None = None,
        interrupt: Interrupt | None = None,
        **data: Any,
    ) -> None:
        if interrupt is not None:
            data["interrupt"] = interrupt.awaiting()
        self.emit(RunEventType.RUN_FINISHED, outcome=outcome, error=error, data=data)
