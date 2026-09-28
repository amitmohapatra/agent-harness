"""``RunEventStream``: what ``runtime.events`` is.

One stream per execution. It numbers events within an attempt, stamps the run, thread and
attempt, redacts each payload the way the tracer redacts a span, and hands the event to
every sink. A sink that raises is logged and skipped: the stream exists to let a UI watch a
run, and a UI that cannot be reached must never fail the run it is watching. An event the
contracts refuse (a text event with no message id) is a harness bug, and raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.errors import AgentError
from trellis.contracts.runs import Interrupt, RunEvent, RunEventType, RunOutcome

from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)

#: The ``webhook_url`` a run carries on its notified events, set from the request metadata.
WEBHOOK_URL_FIELD: Final = "webhook_url"


class RunEventStream:
    def __init__(
        self,
        context: AgentExecutionContext,
        sinks: Sequence[Any] = (),
        *,
        redactor: Any = None,
    ) -> None:
        """``redactor`` (the harness's telemetry redactor) is applied to every payload: what
        leaves the process on a stream or a webhook is redacted exactly like a span."""
        self.context = context
        self._sinks = list(sinks)
        self._redactor = redactor
        self._sequence = 0
        self.attempt = 1

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    @property
    def sequence(self) -> int:
        """The sequence the next event gets."""
        return self._sequence

    def next_attempt(self) -> int:
        """A retry is a new attempt: the sequence restarts and the attempt number moves."""
        self.attempt += 1
        self._sequence = 0
        return self.attempt

    async def emit(
        self,
        event_type: RunEventType | str,
        *,
        message_id: str | None = None,
        tool_call_id: str | None = None,
        step: str | None = None,
        outcome: RunOutcome | str | None = None,
        error: AgentError | None = None,
        interrupt: Interrupt | None = None,
        **data: Any,
    ) -> RunEvent:
        """Build and publish one event; ``ValueError`` when the contracts refuse the shape."""
        kind = RunEventType(event_type)
        if interrupt is not None:
            # the contracts spell it: INTERRUPT *is* the interrupt, RUN_FINISHED carries it
            if kind is RunEventType.INTERRUPT:
                data = {**interrupt.awaiting(), **data}
            else:
                data["interrupt"] = interrupt.awaiting()
        if self._redactor is not None and data:
            data = self._redacted(data)
        event = RunEvent.of(
            self.context,
            kind,
            self._sequence,
            attempt=self.attempt,
            message_id=message_id,
            tool_call_id=tool_call_id,
            step=step,
            outcome=RunOutcome(outcome) if outcome is not None else None,
            error=error,
            data=data,
        )
        self._sequence += 1
        for sink in self._sinks:
            try:
                await sink.publish(event)
            except Exception as exc:  # a sink must never fail the run it watches
                log.warning("run_event.sink_failed", sink=type(sink).__name__, error=str(exc))
        return event

    def _redacted(self, data: dict[str, Any]) -> dict[str, Any]:
        redacted = self._redactor.redact_output(data)
        return redacted if isinstance(redacted, dict) else {"payload": redacted}
