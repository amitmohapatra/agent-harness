"""A run's event stream: numbered ``RunEvent``s, delivered to whoever is listening.

Nothing is built when nobody listens (``run()`` without a stream costs a length check per
event). Delivery is synchronous and non-blocking: a listener is a callable that enqueues.

The stream leaves the process (AG-UI, A2A task updates and push notifications), so what tools
were called with and returned, and the data of custom events, pass the redactor here, once,
for every listener (``redaction.py``); a tool's output is then cut to a preview. The model and
the tool itself get the values as they are. The answer and the pause are not redacted: they
are what the user asked for and what the person answering needs.
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
from trellis.harness.redaction import DEFAULT as REDACTOR

Listener = Callable[[RunEvent], None]

#: The ``CUSTOM`` event name a degraded side effect (a failed memory write...) is reported as.
WARNING: Final = "warning"
#: The ``CUSTOM`` event name of a write-tier tool call, announced to the people watching.
NOTICE: Final = "tool_notice"
#: The ``CUSTOM`` event name of ``trellis.current().log(...)``.
LOG: Final = "log"
#: How much of a tool result rides on the event stream.
PREVIEW_CHARS: Final = 2000


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
        """A step of a tool call: its ``args`` and ``output`` redacted, the output a preview."""
        if not self._listeners:
            return
        redacted = REDACTOR.redact_input(data)
        if "output" in redacted:
            redacted["output"] = _preview(redacted["output"])
        self.emit(type, tool_call_id=tool_call_id, data=redacted)

    def custom(self, name: str, **data: Any) -> None:
        if self._listeners:
            self.emit(RunEventType.CUSTOM, data={"name": name, **REDACTOR.redact_input(data)})

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


def _preview(output: Any) -> Any:
    if output is None or isinstance(output, bool | int | float):
        return output
    text = output if isinstance(output, str) else repr(output)
    if isinstance(output, dict | list) and len(text) <= PREVIEW_CHARS:
        return output
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
