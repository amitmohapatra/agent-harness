"""From the harness's ``RunEvent`` stream to A2A task updates, one to one.

A2A has a narrower vocabulary than the event stream: a task has a *state*, a status *message*
and *artifacts*. So the mapping is a funnel, and the interesting decisions are which events
deserve to be seen by a remote agent and how a run's ending becomes a task state.

* `RUN_STARTED` → `WORKING`.
* `TEXT_MESSAGE_CONTENT` → `WORKING`, the delta as agent text.
* Progress events (tool calls, steps, context, state, custom, raw) → `WORKING`, the event as a
  data part, so a caller can follow the work without parsing prose.
* Text start/end, `INTERRUPT`, `MESSAGES_SNAPSHOT` → nothing: the deltas carried the text and the
  finish event carries the pause.
* `RUN_FINISHED(success/partial)` → the result as an artifact, then `COMPLETED`.
* `RUN_FINISHED(interrupt)` → `INPUT_REQUIRED` with the question and its schema. Not terminal,
  which is what lets the next message resume the same task.
* `RUN_FINISHED(error/timeout)` → `FAILED` with the error's code and message.
* `RUN_FINISHED(rejected)` → `REJECTED`: a policy refusal is a refusal, not a crash.
* `RUN_FINISHED(cancelled)` → `CANCELED`.

Every payload here has already been through the harness's redactor: a sink never sees more than
a span would, so neither does another agent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from a2a.helpers import new_data_part, new_text_part
from a2a.types import Part, TaskState
from trellis.contracts.runs import RunEvent, RunEventType, RunOutcome

#: A run's ending, as a task state.
OUTCOME_STATES: Final[dict[RunOutcome, TaskState]] = {
    RunOutcome.SUCCESS: TaskState.TASK_STATE_COMPLETED,
    RunOutcome.PARTIAL: TaskState.TASK_STATE_COMPLETED,
    RunOutcome.INTERRUPT: TaskState.TASK_STATE_INPUT_REQUIRED,
    RunOutcome.ERROR: TaskState.TASK_STATE_FAILED,
    RunOutcome.TIMEOUT: TaskState.TASK_STATE_FAILED,
    RunOutcome.REJECTED: TaskState.TASK_STATE_REJECTED,
    RunOutcome.CANCELLED: TaskState.TASK_STATE_CANCELED,
}
#: Task states a task cannot leave (the SDK latches them; a second terminal update raises).
TERMINAL_STATES: Final = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
        TaskState.TASK_STATE_CANCELED,
    }
)
#: Progress events a caller sees as a data part, under this name.
PROGRESS_EVENTS: Final = frozenset(
    {
        RunEventType.STEP_STARTED,
        RunEventType.STEP_FINISHED,
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
        RunEventType.CONTEXT_LOADED,
        RunEventType.STATE_SNAPSHOT,
        RunEventType.STATE_DELTA,
        RunEventType.CUSTOM,
        RunEventType.RAW,
    }
)
#: Events the transport does not carry: the text deltas already said it, or the finish event will.
IGNORED_EVENTS: Final = frozenset(
    {
        RunEventType.TEXT_MESSAGE_START,
        RunEventType.TEXT_MESSAGE_END,
        RunEventType.MESSAGES_SNAPSHOT,
        RunEventType.INTERRUPT,
    }
)
#: The artifact a finished run's result is published as.
RESULT_ARTIFACT: Final = "result"


@dataclass(frozen=True, slots=True)
class Update:
    """What one run event means for the task: a state, something to say, something to publish."""

    state: TaskState
    parts: tuple[Part, ...] = ()
    artifact: tuple[str, Any] | None = None
    #: The interrupt the run paused on, as JSON, when this update is a pause.
    interrupt: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def update_for(event: RunEvent) -> Update | None:
    """The task update for a run event, or ``None`` for an event A2A does not carry."""
    kind = event.type
    if kind is RunEventType.RUN_FINISHED:
        return _finished(event)
    if kind is RunEventType.RUN_ERROR:
        return Update(TaskState.TASK_STATE_FAILED, parts=(text_part(_error_text(event)),))
    if kind is RunEventType.RUN_STARTED:
        return Update(TaskState.TASK_STATE_WORKING)
    if kind is RunEventType.TEXT_MESSAGE_CONTENT:
        delta = event.data.get("delta")
        parts = (text_part(str(delta)),) if delta else ()
        return Update(TaskState.TASK_STATE_WORKING, parts=parts) if parts else None
    if kind in PROGRESS_EVENTS:
        return Update(TaskState.TASK_STATE_WORKING, parts=(data_part(_progress(event)),))
    return None  # IGNORED_EVENTS, and anything a future contracts version adds


def text_part(text: str) -> Part:
    """One text part. Thin, but it keeps the SDK's helper names in one place."""
    return new_text_part(text)


def data_part(data: Any) -> Part:
    """One structured part. ``data`` must be JSON-serialisable (proto ``Value``)."""
    return new_data_part(data)


def value_part(value: Any) -> Part:
    """A run's input, output or answer as one part: text when it is text, data when it is not.

    One implementation, because the stream and the task rebuilt from the run store must publish the
    same value the same way.
    """
    if isinstance(value, str):
        return text_part(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return text_part(str(value))
    return data_part(value)


def asked(interrupt: Mapping[str, Any]) -> dict[str, Any]:
    """What a caller is told about a pause, and nothing else.

    ``Interrupt.awaiting()`` is the *unredacted* record — its own docstring says it carries the tool
    call's arguments — so it never leaves the process whole. A caller needs the question (sent as
    text), which interrupt it is answering, why, and the shape of the answer.
    """
    return {
        key: interrupt[key]
        for key in ("interrupt_id", "reason", "expects", "payload")
        if interrupt.get(key) is not None
    }


def _finished(event: RunEvent) -> Update:
    outcome = event.outcome or RunOutcome.ERROR
    state = OUTCOME_STATES.get(outcome, TaskState.TASK_STATE_FAILED)
    if outcome is RunOutcome.INTERRUPT:
        interrupt = dict(event.data.get("interrupt") or {})
        question = str(interrupt.get("question") or "the agent is waiting for an answer")
        return Update(
            state,
            parts=(text_part(question), data_part(asked(interrupt))),
            interrupt=interrupt,
        )
    if state is TaskState.TASK_STATE_COMPLETED:
        result = event.data.get("result")
        return Update(state, artifact=(RESULT_ARTIFACT, result) if result is not None else None)
    return Update(state, parts=(text_part(_error_text(event)), data_part(_failure(event))))


def _error_text(event: RunEvent) -> str:
    if event.error is not None and event.error.message:
        return event.error.message
    outcome = event.outcome.value if event.outcome else "error"
    return f"the run ended {outcome}"


def _failure(event: RunEvent) -> dict[str, Any]:
    error = event.error
    return _pruned(
        {
            "outcome": event.outcome.value if event.outcome else None,
            "code": error.code if error else None,
            "category": str(error.category) if error and error.category else None,
            "retryable": error.retryable if error else None,
        }
    )


def _progress(event: RunEvent) -> dict[str, Any]:
    """A progress event as data: what happened, on which step or call, and its payload."""
    return _pruned(
        {
            "event": event.type.value,
            "step": event.step,
            "tool_call_id": event.tool_call_id,
            "sequence": event.sequence,
            "attempt": event.attempt,
            **event.data,
        }
    )


def _pruned(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


__all__ = [
    "IGNORED_EVENTS",
    "OUTCOME_STATES",
    "PROGRESS_EVENTS",
    "RESULT_ARTIFACT",
    "TERMINAL_STATES",
    "Update",
    "asked",
    "data_part",
    "text_part",
    "update_for",
    "value_part",
]
