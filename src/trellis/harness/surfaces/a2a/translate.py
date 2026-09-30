"""From a run's ``RunEvent`` stream to A2A task updates, and between A2A parts and values.

* ``RUN_STARTED`` → ``WORKING``; text deltas → ``WORKING`` with the text;
* progress (tool calls, steps, context, custom) → ``WORKING`` with the event as a data part;
* ``RUN_FINISHED`` success → the result as an artifact, then ``COMPLETED``; interrupt →
  ``INPUT_REQUIRED`` with the question (not terminal: the next message resumes the task);
  error/timeout → ``FAILED``; rejected → ``REJECTED``; cancelled → ``CANCELED``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from a2a.helpers import new_data_part, new_text_part
from a2a.types import Message, Part, TaskState
from google.protobuf.json_format import MessageToDict

from trellis.contracts import RunEvent, RunEventType, RunOutcome

OUTCOME_STATES: Final[dict[RunOutcome, TaskState]] = {
    RunOutcome.SUCCESS: TaskState.TASK_STATE_COMPLETED,
    RunOutcome.PARTIAL: TaskState.TASK_STATE_COMPLETED,
    RunOutcome.INTERRUPT: TaskState.TASK_STATE_INPUT_REQUIRED,
    RunOutcome.ERROR: TaskState.TASK_STATE_FAILED,
    RunOutcome.TIMEOUT: TaskState.TASK_STATE_FAILED,
    RunOutcome.REJECTED: TaskState.TASK_STATE_REJECTED,
    RunOutcome.CANCELLED: TaskState.TASK_STATE_CANCELED,
}
#: States a task cannot leave (the SDK refuses a second terminal update).
TERMINAL_STATES: Final = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
        TaskState.TASK_STATE_CANCELED,
    }
)
PROGRESS_EVENTS: Final = frozenset(
    {
        RunEventType.STEP_STARTED,
        RunEventType.STEP_FINISHED,
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
        RunEventType.CONTEXT_LOADED,
        RunEventType.CUSTOM,
    }
)
RESULT_ARTIFACT: Final = "result"
#: What a caller is told about a pause: never the whole (unredacted) interrupt.
ASKED_FIELDS: Final = ("interrupt_id", "reason", "ui", "options", "expects", "payload")


@dataclass(frozen=True, slots=True)
class Update:
    """What one run event means for the task."""

    state: TaskState
    parts: tuple[Part, ...] = ()
    result: Any = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def update_for(event: RunEvent) -> Update | None:
    """The task update for a run event, or ``None`` for one A2A does not carry."""
    kind = event.type
    if kind is RunEventType.RUN_FINISHED:
        return _finished(event)
    if kind is RunEventType.RUN_STARTED:
        return Update(TaskState.TASK_STATE_WORKING)
    if kind is RunEventType.TEXT_MESSAGE_CONTENT and event.data.get("delta"):
        return Update(TaskState.TASK_STATE_WORKING, (new_text_part(str(event.data["delta"])),))
    if kind in PROGRESS_EVENTS:
        progress = {"event": kind.value, "tool_call_id": event.tool_call_id, **event.data}
        clean = {k: v for k, v in progress.items() if v is not None}
        return Update(TaskState.TASK_STATE_WORKING, (value_part(clean),))
    return None


def value_part(value: Any) -> Part:
    """A value as one part: text when it is text, data when it is JSON, else its text."""
    if isinstance(value, str):
        return new_text_part(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return new_text_part(str(value))
    return new_data_part(value)


def asked(interrupt: Mapping[str, Any]) -> dict[str, Any]:
    return {k: interrupt[k] for k in ASKED_FIELDS if interrupt.get(k) not in (None, [])}


def text_and_data(message: Message | None) -> tuple[str, dict[str, Any]]:
    """A message's text and its merged data parts."""
    if message is None:
        return "", {}
    texts: list[str] = []
    data: dict[str, Any] = {}
    for part in message.parts:
        which = part.WhichOneof("content")
        if which == "text":
            texts.append(part.text)
        elif which == "data":
            value = MessageToDict(part.data)
            if isinstance(value, Mapping):
                data.update(value)
    return "\n".join(texts), data


def values(parts: Sequence[Part]) -> list[Any]:
    """Parts as plain values: text as strings, data as objects, URLs as strings."""
    found: list[Any] = []
    for part in parts:
        which = part.WhichOneof("content")
        if which == "text":
            found.append(part.text)
        elif which == "data":
            found.append(MessageToDict(part.data))
        elif which == "url":
            found.append(part.url)
    return found


def _finished(event: RunEvent) -> Update:
    outcome = event.outcome or RunOutcome.ERROR
    state = OUTCOME_STATES.get(outcome, TaskState.TASK_STATE_FAILED)
    if outcome is RunOutcome.INTERRUPT:
        interrupt = dict(event.data.get("interrupt") or {})
        question = str(interrupt.get("question") or "the agent is waiting for an answer")
        return Update(state, (new_text_part(question), new_data_part(asked(interrupt))))
    if state is TaskState.TASK_STATE_COMPLETED:
        return Update(state, result=event.data.get("result"))
    message = event.error.message if event.error is not None else f"the run ended {outcome.value}"
    return Update(state, (new_text_part(message),))
