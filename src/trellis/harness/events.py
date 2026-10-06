"""A run's event stream: numbered ``RunEvent``s, delivered to whoever is listening.

Nothing is built when nobody listens (``run()`` without a stream costs a length check per
event). Delivery is synchronous and non-blocking: a listener is a callable that enqueues.

Two kinds of listeners: the people watching (``listen``: a stream, a surface) and the run's
log (``record``: agent-runs' event log, ``runlog.py``), which must have an attempt's last events
before the run store records the pause or the ending — it takes nothing after. So the pipeline
holds the watchers' copies of those last events (:meth:`~RunEvents.settle`), has the log
write them, records the ending, and only then releases them: whoever reads
``RUN_FINISHED`` finds the run already paused or ended.

The stream leaves the process (AG-UI, A2A task updates and push notifications), so what tools
were called with and returned, and the data of custom events, pass the redactor here, once,
for every listener (``redaction.py``); a tool's output is then cut to a preview. The model and
the tool itself get the values as they are. The answer and the pause are not redacted: they
are what the user asked for and what the person answering needs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Final, Protocol, TypeVar

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
_Written = TypeVar("_Written")


class Sink(Protocol):
    """The run's log (``runlog.RunLog``): every event at once, closed once the attempt's last
    events are in."""

    def __call__(self, event: RunEvent) -> None: ...

    async def close(self) -> None: ...


#: The ``CUSTOM`` event name a degraded side effect (a failed memory write...) is reported as.
WARNING: Final = "warning"
#: The ``CUSTOM`` event name of a write-tier tool call, announced to the people watching.
NOTICE: Final = "tool_notice"
#: The ``CUSTOM`` event name of ``trellis.current().log(...)``.
LOG: Final = "log"
#: The ``CUSTOM`` event name of a person's decision the run goes on with: at the start of the
#: attempt that continues after it (``decision``, ``reviewer``, ``comment``, ``remember``), and
#: for a call approved without asking because a reviewer approved its tool for the rest of the
#: run (``tool``, ``remembered: true``).
DECISION: Final = "decision"
#: How much of a tool result rides on the event stream.
PREVIEW_CHARS: Final = 2000


class RunEvents:
    """One attempt's emitter. ``sequence`` is per attempt, as the contract orders it."""

    __slots__ = ("_held", "_listeners", "_sinks", "attempt", "context", "sequence")

    def __init__(self, context: AgentExecutionContext, attempt: int = 1) -> None:
        self.context = context
        self.attempt = attempt
        self.sequence = 0
        self._listeners: list[Listener] = []
        self._sinks: list[Sink] = []
        self._held: list[RunEvent] | None = None

    def listen(self, listener: Listener) -> None:
        """Someone watching: every event from now on."""
        self._listeners.append(listener)

    def record(self, sink: Sink) -> None:
        """The run's log: every event from now on, at once, even while the watchers' copies
        are held."""
        self._sinks.append(sink)

    @property
    def quiet(self) -> bool:
        """Nobody listens: no event is built."""
        return not self._listeners and not self._sinks

    def hold(self) -> None:
        """Keep the watchers' copies of the events from now on until :meth:`release`."""
        self._held = []

    def release(self, *, deliver: bool = True) -> None:
        """Hand the watchers what was held (``deliver=False``: drop it — the ending it
        announced was not recorded), and deliver at once again."""
        held, self._held = self._held or [], None
        for event in held if deliver else ():
            for listener in self._listeners:
                listener(event)

    async def settle(
        self, last: Callable[[], None], write: Callable[[], Awaitable[_Written]]
    ) -> _Written:
        """An attempt's end: emit its ``last`` events, have the run's log take them, then
        ``write`` the pause or the ending to the run store; the watchers get those events once
        it is written (none, when the write raises)."""
        self.hold()
        try:
            last()
            for sink in self._sinks:
                await sink.close()
            written = await write()
        except BaseException:
            self.release(deliver=False)
            raise
        self.release()
        return written

    def emit(self, type: RunEventType, **fields: Any) -> None:
        if self.quiet:
            return
        event = RunEvent.of(self.context, type, self.sequence, attempt=self.attempt, **fields)
        self.sequence += 1
        for sink in self._sinks:
            sink(event)
        if self._held is not None:
            self._held.append(event)
            return
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
        if self.quiet:
            return
        redacted = REDACTOR.redact_input(data)
        if "output" in redacted:
            redacted["output"] = _preview(redacted["output"])
        self.emit(type, tool_call_id=tool_call_id, data=redacted)

    def custom(self, name: str, **data: Any) -> None:
        if not self.quiet:
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
