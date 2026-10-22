"""A conversation's messages as the harness carries them between the memory service and a
framework: who said it, what, and the tool calls made or answered — the shape a framework's
own checkpoint replays.

* :class:`Said` — one message; :class:`Called` — one tool call an assistant message made;
* :func:`from_window` — the memory service's window (``window_format="messages"``) as
  messages, which each adapter turns into its framework's own (``prepare_input(history=)``);
* :func:`recorded` — a message as the memory service stores it, and :func:`said_of` back
  from that stored form (the write spool keeps it, a restart replays it).

When memory owns a conversation's history (the framework keeps none durably), the run's
window is the thread's recent messages as these, and the run's own messages — its question,
its tool calls and their results, its answer — are written back before the run returns, so
the next turn's window holds them wherever it runs (:meth:`Agent.recorded_run`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from trellis.memory.models import WindowMessage


class Speaker(StrEnum):
    """Who said a message, as the frameworks name them."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Called:
    """A tool call an assistant message made: its id (the result names it), the tool and its
    arguments."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Said:
    """One message: an assistant's carries the tool calls it made (``tool_calls``), a tool's
    the call it answers (``tool_call_id``)."""

    role: Speaker
    content: str
    tool_calls: tuple[Called, ...] = ()
    tool_call_id: str | None = None


#: The memory service's role of each speaker, and the reverse.
MEMORY_ROLES: Final[Mapping[Speaker, str]] = {
    Speaker.USER: "USER",
    Speaker.ASSISTANT: "ASSISTANT",
    Speaker.TOOL: "TOOL",
    Speaker.SYSTEM: "SYSTEM",
}
SPEAKERS: Final[Mapping[str, Speaker]] = {role: speaker for speaker, role in MEMORY_ROLES.items()}
#: The kind the memory service keeps a tool result as: not chat, but replayed with its call.
INTERNAL: Final = "INTERNAL"


def from_window(messages: Sequence[WindowMessage]) -> list[Said]:
    """The memory service's window as messages, in its order (a role the window does not
    replay is left out)."""
    return [
        Said(
            role=SPEAKERS[m.role],
            content=m.text,
            tool_calls=tuple(Called(c.id, c.name, dict(c.arguments)) for c in m.tool_calls),
            tool_call_id=m.tool_call_id,
        )
        for m in messages
        if m.role in SPEAKERS
    ]


def recorded(said: Said) -> dict[str, Any]:
    """``said`` as the memory service's message: a tool result is INTERNAL, as the service
    keeps every tool step; the rest is the visible chat."""
    message: dict[str, Any] = {"role": MEMORY_ROLES[said.role], "content": said.content}
    if said.role is Speaker.TOOL:
        message["kind"] = INTERNAL
    if said.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": dict(c.arguments)} for c in said.tool_calls
        ]
    if said.tool_call_id is not None:
        message["tool_call_id"] = said.tool_call_id
    return message


def said_of(stored: Mapping[str, Any] | Sequence[Any]) -> Said:
    """A message back from its :func:`recorded` form - or from the ``[role, content]`` pair a
    spool written before tool calls were kept holds."""
    if not isinstance(stored, Mapping):
        role, content = stored
        return Said(Speaker(role), content)
    return Said(
        role=SPEAKERS[stored["role"]],
        content=stored["content"],
        tool_calls=tuple(
            Called(c["id"], c["name"], c.get("arguments") or {})
            for c in stored.get("tool_calls") or ()
        ),
        tool_call_id=stored.get("tool_call_id"),
    )
