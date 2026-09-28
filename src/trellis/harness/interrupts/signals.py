"""From a pause signal to a contracts ``Interrupt``.

``AgentPaused`` carries the question directly. LangGraph's ``interrupt(value)`` raises a
``GraphInterrupt`` whose args hold ``Interrupt`` objects with a ``.value``; those are read
structurally, since the core never imports langgraph. Anything else becomes a question made
of the signal's class name, so a paused run always has something to show a person.
"""

from __future__ import annotations

from typing import Any

from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.errors import AgentPaused
from trellis.contracts.runs import Interrupt, InterruptReason
from trellis.contracts.tool import ToolCall


class ApprovalRequired(AgentPaused):
    """A tool call the policy wants a person to approve before it runs (design §6)."""

    def __init__(self, call: ToolCall, *, reason: str | None = None) -> None:
        question = f"Approve the tool call {call.tool}?" + (f" ({reason})" if reason else "")
        # the call itself travels as ``Interrupt.tool_call``; the payload only says why
        super().__init__(question, expects={"type": "boolean"}, payload={"reason": reason})
        self.tool_call = call


def _asked(signal: Any) -> dict[str, Any] | None:
    """What the person is being asked, if the signal says; None otherwise."""
    if signal is None:
        return None
    if callable(getattr(signal, "awaiting", None)):
        try:
            value = signal.awaiting()
        except Exception:  # a broken signal must not turn a pause into a crash
            return None
        return value if isinstance(value, dict) else None
    values = [
        getattr(arg, "value", arg)
        for arg in getattr(signal, "args", ())
        if arg is not None and not isinstance(arg, str | bytes)
    ]
    if not values:
        return None
    return {"question": values[0]} if len(values) == 1 else {"question": values}


def interrupt_from_signal(signal: BaseException, context: AgentExecutionContext) -> Interrupt:
    if isinstance(signal, ApprovalRequired):
        return Interrupt.from_paused(
            signal, context=context, reason=InterruptReason.APPROVAL, tool_call=signal.tool_call
        )
    if isinstance(signal, AgentPaused):
        return Interrupt.from_paused(signal, context=context)
    question = _asked(signal) or {}
    text = question.get("question")
    return Interrupt(
        tenant_id=context.tenant_id,
        run_id=context.agent_run_id,
        question=text if isinstance(text, str) and text.strip() else type(signal).__name__,
        payload=question if question and not isinstance(text, str) else None,
    )
