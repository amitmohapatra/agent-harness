"""From the harness's ``RunEvent`` stream to AG-UI events, one to one.

The contracts already use the AG-UI spellings; what the transport adds is the shape
(camelCase members, ``outcome`` as the protocol's discriminated object), the two events
AG-UI does not have (``CONTEXT_LOADED`` becomes ``CUSTOM``; ``INTERRUPT`` is folded into
``RUN_FINISHED{outcome: {type: "interrupt", interrupts}}``), and the failure outcomes, which
AG-UI reports as ``RUN_ERROR`` (a cancellation is its own outcome, not an error).
"""

from __future__ import annotations

import json
from typing import Any, Final

from trellis.contracts.runs import RunEvent, RunEventType, RunOutcome

from trellis.harness.surfaces.agui.events import (
    AGUIEvent,
    AGUIEventType,
    InterruptEntry,
    Outcome,
    OutcomeType,
)

#: Outcomes an AG-UI client hears as ``RUN_ERROR`` rather than ``RUN_FINISHED``.
FAILURE_OUTCOMES: Final = frozenset({RunOutcome.ERROR, RunOutcome.REJECTED, RunOutcome.TIMEOUT})
#: Harness events ``RUN_FINISHED`` already carries, so a client does not hear them twice.
FOLDED: Final = frozenset({RunEventType.INTERRUPT, RunEventType.RUN_ERROR})
#: Harness events an AG-UI client sees as ``CUSTOM`` events of that name.
CUSTOM_EVENTS: Final = {RunEventType.CONTEXT_LOADED: "context_loaded"}
_STEP_EVENTS: Final = frozenset({RunEventType.STEP_STARTED, RunEventType.STEP_FINISHED})
_TEXT_EVENTS: Final = frozenset(
    {
        RunEventType.TEXT_MESSAGE_START,
        RunEventType.TEXT_MESSAGE_CONTENT,
        RunEventType.TEXT_MESSAGE_END,
    }
)
_TOOL_EVENTS: Final = frozenset(
    {
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    }
)


def translate(event: RunEvent) -> AGUIEvent | None:
    """The AG-UI event for a harness event; None for events the transport does not carry
    (the harness's own ``INTERRUPT``, which the finish event folds in)."""
    base: dict[str, Any] = {
        "timestamp": int(event.timestamp.timestamp() * 1000),
        "thread_id": event.thread_id,
        "run_id": event.run_id,
    }
    kind = event.type
    if kind in FOLDED:
        return None
    if kind is RunEventType.RUN_FINISHED:
        return _finished(event, base)
    if kind in CUSTOM_EVENTS:
        return AGUIEvent(
            type=AGUIEventType.CUSTOM, name=CUSTOM_EVENTS[kind], value=event.data, **base
        )
    if kind.value not in AGUIEventType.__members__:
        return None
    fields: dict[str, Any] = {"type": AGUIEventType(kind.value), **base, **_members(event)}
    return AGUIEvent(**fields)


def _members(event: RunEvent) -> dict[str, Any]:
    """The members a type carries besides the run and thread."""
    kind, data = event.type, dict(event.data)
    if kind in _STEP_EVENTS:
        return {"step_name": event.step}
    if kind in _TEXT_EVENTS:
        return _text_members(event, data)
    if kind in _TOOL_EVENTS:
        return _tool_members(event, data)
    return _state_members(kind, data)


def _text_members(event: RunEvent, data: dict[str, Any]) -> dict[str, Any]:
    members: dict[str, Any] = {"message_id": event.message_id}
    if event.type is RunEventType.TEXT_MESSAGE_START:
        members["role"] = data.get("role", "assistant")
    if event.type is RunEventType.TEXT_MESSAGE_CONTENT:
        members["delta"] = data.get("delta", "")
    return members


def _tool_members(event: RunEvent, data: dict[str, Any]) -> dict[str, Any]:
    members: dict[str, Any] = {"tool_call_id": event.tool_call_id}
    if event.type is RunEventType.TOOL_CALL_START:
        members["tool_call_name"] = data.get("tool")
        members["parent_message_id"] = event.message_id
    elif event.type is RunEventType.TOOL_CALL_ARGS:
        members["delta"] = _json(data.get("args", {}))
    elif event.type is RunEventType.TOOL_CALL_RESULT:
        members["message_id"] = event.message_id or f"result_{event.tool_call_id}"
        members["content"] = _json(data.get("output"))
        members["role"] = "tool"
    return members


def _state_members(kind: RunEventType, data: dict[str, Any]) -> dict[str, Any]:
    if kind is RunEventType.STATE_SNAPSHOT:
        return {"snapshot": data.get("snapshot", data)}
    if kind is RunEventType.STATE_DELTA:
        return {"delta": data.get("delta", [])}
    if kind is RunEventType.MESSAGES_SNAPSHOT:
        return {"messages": data.get("messages", [])}
    if kind is RunEventType.RAW:
        return {"event": data.get("event", data), "source": data.get("source")}
    if kind is RunEventType.CUSTOM:
        return {
            "name": data.get("name", "custom"),
            "value": {k: v for k, v in data.items() if k != "name"},
        }
    return {}


def _finished(event: RunEvent, base: dict[str, Any]) -> AGUIEvent:
    outcome = event.outcome or RunOutcome.SUCCESS
    if outcome in FAILURE_OUTCOMES:
        error = event.error
        return AGUIEvent(
            type=AGUIEventType.RUN_ERROR,
            message=error.message if error else outcome.value,
            code=str(error.code) if error else outcome.value.upper(),
            **base,
        )
    if outcome is RunOutcome.CANCELLED:
        return AGUIEvent(
            type=AGUIEventType.RUN_FINISHED, outcome=Outcome(type=OutcomeType.CANCELLED), **base
        )
    if outcome is RunOutcome.INTERRUPT:
        return AGUIEvent(
            type=AGUIEventType.RUN_FINISHED,
            outcome=Outcome(
                type=OutcomeType.INTERRUPT,
                interrupts=[interrupt_entry(event.data.get("interrupt") or {})],
            ),
            **base,
        )
    return AGUIEvent(
        type=AGUIEventType.RUN_FINISHED,
        outcome=Outcome(type=OutcomeType.SUCCESS),
        result=event.data.get("result"),
        **base,
    )


def interrupt_entry(raw: dict[str, Any]) -> InterruptEntry:
    """A contracts ``Interrupt`` (as ``awaiting()`` spells it) as the protocol's entry: the
    reason lower-cased, the question as the message, ``expects`` as the response schema,
    the tool call's id, and the rest under ``metadata``."""
    call = raw.get("tool_call") or {}
    metadata = {
        k: raw[k] for k in ("payload", "payload_ref", "tool_call") if raw.get(k) is not None
    }
    return InterruptEntry(
        id=str(raw.get("interrupt_id")),
        reason=str(raw.get("reason", "QUESTION")).lower(),
        message=raw.get("question"),
        tool_call_id=call.get("idempotency_key") if isinstance(call, dict) else None,
        response_schema=raw.get("expects"),
        metadata=metadata or None,
    )


def _json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
