"""Who owns a conversation's history across runs: one policy for every adapter.

A framework that keeps a conversation keeps it in a *thread store* — a LangGraph checkpointer,
an OpenAI Agents session, a conversation on OpenAI's servers. The memory service keeps it too:
its thread, given back as the *window* of the pushed context. One of the two owns it, never
both, and :func:`owner` says which:

* ``Owner.FRAMEWORK`` — the framework was given a durable store (it survives a worker restart,
  and a replica elsewhere reads it): memory's window is left out;
* ``Owner.MEMORY`` — the framework keeps no store, or one kept in this process's memory only
  (:data:`IN_RAM`), lost on a restart, or its store is memory's own thread
  (:data:`MEMORY_THREAD`: an OpenAI Agents run's session): memory owns the history, and each
  run's turns are written before it returns. A store kept in RAM is said once per agent and
  store type.

A store of a type :data:`IN_RAM` does not name is taken as durable: the harness cannot see
where an unknown store keeps its data, and a team that brings one (``PostgresSaver``, a Redis
session) chose it for that.

When memory owns it, the history travels as messages:

* :class:`Said` — one message; :class:`Called` — one tool call an assistant message made;
* :func:`from_window` — the memory service's window (``window_format="messages"``) as
  messages, which each adapter turns into its framework's own (``prepare_input(history=)``);
* :func:`recorded` — a message as the memory service stores it, and :func:`said_of` back
  from that stored form (the write spool keeps it, a restart replays it); the run-record
  facts a message carries (``Said.facts``: a tool result's status, seconds and sources, the
  turn's last message the run's) go with it under ``custom_metadata`` (:data:`RUN_RECORD`),
  which the service's steps list of the turn shows.

When memory owns a conversation's history and the framework keeps no store at all, the run's
window is the thread's recent messages as these (a store kept in RAM replays the thread itself
while the process lives, so there the window comes as the context's text, never twice), and
the run's own messages — its question,
its tool calls and their results, its answer — are written back before the run returns, so
the next turn's window holds them wherever it runs (:meth:`Agent.recorded_run`).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from trellis.harness.bounded import SAID_ONCE
from trellis.memory.models import WindowMessage

log = logging.getLogger("trellis.harness")


#: The path of SQLite's in-memory database.
SQLITE_IN_MEMORY: Final = ":memory:"


class Owner(StrEnum):
    """Who keeps a conversation's history across runs."""

    FRAMEWORK = "framework"
    MEMORY = "memory"


def _always(store: Any) -> bool:
    return True


def _sqlite_in_memory(store: Any) -> bool:
    return str(getattr(store, "db_path", "")) == SQLITE_IN_MEMORY


#: The stores known to keep a thread in the process's memory only, by their type's qualified
#: name (a subclass matches too), each with what makes an instance one.
IN_RAM: Final[Mapping[str, Callable[[Any], bool]]] = {
    "langgraph.checkpoint.memory.InMemorySaver": _always,
    "agents.memory.sqlite_session.SQLiteSession": _sqlite_in_memory,
    "claude_agent_sdk._internal.session_store.InMemorySessionStore": _always,
}


class MemoryThread:
    """The thread store that is the memory service's own thread, which the framework reads
    through a slot of its own (an OpenAI Agents run's ``bridges.session.MemorySession``):
    memory owns the history, and its window is not pushed again."""


#: The one :class:`MemoryThread`: what an adapter's ``thread_store`` names.
MEMORY_THREAD: Final = MemoryThread()


def in_ram(store: Any) -> bool:
    """Whether ``store`` keeps what it holds in the process's memory only (:data:`IN_RAM`)."""
    for kind in type(store).__mro__:
        check = IN_RAM.get(f"{kind.__module__}.{kind.__qualname__}")
        if check is not None:
            return check(store)
    return False


def owner(store: Any | None, *, agent_id: str) -> Owner:
    """Who owns the history, given the thread ``store`` the framework keeps (``None``: none)."""
    if store is None or store is MEMORY_THREAD:
        return Owner.MEMORY
    if in_ram(store):
        _said_in_ram(agent_id, type(store).__name__)
        return Owner.MEMORY
    return Owner.FRAMEWORK


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and store type, of the most recent
def _said_in_ram(agent_id: str, store: str) -> None:
    log.warning(
        "the framework's thread store keeps the conversation in this process's memory only, "
        "lost when the worker restarts: memory's window keeps the history instead; give it a "
        "durable store (a LangGraph PostgresSaver, a SQLiteSession on a file) to keep it there",
        extra={"agent_id": agent_id, "store": store},
    )


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
    #: what the run's record says of this message (:data:`RUN_RECORD`), when it says anything
    facts: Mapping[str, Any] | None = None
    #: the harness's id of the message (``conversation.message_id``): the same wherever the
    #: message is seen — the AG-UI stream, the memory service's thread, an attempt after a
    #: resume — ``None`` until the run's transcript names it (``conversation.identified``)
    id: str | None = None


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
#: The ``custom_metadata`` key the memory service reads a message's run-record facts under.
RUN_RECORD: Final = "run_record"
#: The ``custom_metadata`` key a stored message keeps its harness id under (``Said.id``).
MESSAGE_ID: Final = "message_id"


def from_window(messages: Sequence[WindowMessage]) -> list[Said]:
    """The memory service's window as messages, in its order, each with the run record the
    service keeps of it (``Said.facts``; a role the window does not replay is left out)."""
    return [
        Said(
            role=SPEAKERS[m.role],
            content=m.text,
            tool_calls=tuple(Called(c.id, c.name, dict(c.arguments)) for c in m.tool_calls),
            tool_call_id=m.tool_call_id,
            facts=getattr(m, RUN_RECORD, None) or None,
        )
        for m in messages
        if m.role in SPEAKERS
    ]


def answered(said: Sequence[Said]) -> list[Said]:
    """``said`` (oldest first) as a model API takes it: an assistant message whose calls are
    not all answered in it (a run stopped between a call and its result, a read that cut one
    off) is kept without its calls, and not at all when it said nothing; a result whose call
    is not kept is left out."""
    results = {s.tool_call_id for s in said if s.role is Speaker.TOOL}
    kept: list[Said] = []
    for s in said:
        whole = not s.tool_calls or all(c.id in results for c in s.tool_calls)
        message = s if whole else Said(s.role, s.content, id=s.id)
        if message.role is Speaker.TOOL or message.content or message.tool_calls:
            kept.append(message)
    calls = {c.id for s in kept for c in s.tool_calls}
    return [s for s in kept if s.role is not Speaker.TOOL or s.tool_call_id in calls]


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
    metadata: dict[str, Any] = {}
    if said.facts:
        metadata[RUN_RECORD] = dict(said.facts)
    if said.id is not None:
        metadata[MESSAGE_ID] = said.id
    if metadata:
        message["custom_metadata"] = metadata
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
        facts=(stored.get("custom_metadata") or {}).get(RUN_RECORD),
        id=(stored.get("custom_metadata") or {}).get(MESSAGE_ID),
    )


__all__ = [
    "INTERNAL",
    "IN_RAM",
    "MEMORY_THREAD",
    "MESSAGE_ID",
    "RUN_RECORD",
    "Called",
    "MemoryThread",
    "Owner",
    "Said",
    "Speaker",
    "answered",
    "from_window",
    "in_ram",
    "owner",
    "recorded",
    "said_of",
]
