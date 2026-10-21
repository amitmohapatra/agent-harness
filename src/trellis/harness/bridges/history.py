"""Who owns a conversation's history across runs: one policy, used by every adapter.

A framework that keeps a conversation keeps it in a *thread store* — a LangGraph checkpointer,
an OpenAI Agents session. The memory service keeps it too: its thread, given back as the
*window* of the pushed context. One of the two owns it, never both, and :func:`thread_for` says
which, as a :class:`Thread` for the framework:

* ``OWN`` — the store the framework was given is durable (it survives a worker restart, and a
  replica elsewhere reads it): it owns the history, and memory's window is left out;
* ``BRIDGED`` — the framework's slot holds the memory service's thread (an OpenAI Agents
  ``MemorySession``): the framework reads it, and memory's window is left out;
* ``RUN`` — the framework works on the run's own thread, kept in the run's journal (as
  ``ReAct``'s ``RunCheckpointer``): memory's window owns the history;
* ``KEPT`` — the framework keeps its store as it was given — one kept in RAM, or none — and
  memory's window owns the history.

What the per-step run state needs (a checkpoint, a resume) is the journal's and agent-runs',
whoever owns the history.

:data:`POLICY` is the rule in force, the one line that changes it:

* ``Policy.DURABLE_THREAD`` (now): a durable framework thread owns the history; else the
  memory service's thread through the framework's slot, where it has one (``BRIDGED``); else
  memory's window, the framework's store left as given;
* ``Policy.MEMORY_WINDOW``: memory's window owns the history; the framework works on the run's
  own thread (``RUN``), and only a durable thread the team gave it explicitly is kept (``OWN``,
  the window off).

What is not durable is named once (:data:`IN_RAM`): LangGraph's ``InMemorySaver``
(``MemorySaver``), an OpenAI Agents ``SQLiteSession`` on ``":memory:"``, the Claude Agent SDK's
``InMemorySessionStore``. A store of any other type is taken as durable: the harness cannot see
where an unknown store keeps its data, and a team that brings one (``PostgresSaver``, a Redis
session) chose it for that — the assumption is the team's, and documented. A store kept in RAM
is said once per agent and store type, with what keeps the history instead and the fix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Final

from trellis.harness.bridges import said


class Thread(StrEnum):
    """The thread a framework works on in a run (the module says what each means)."""

    OWN = "own"
    BRIDGED = "bridged"
    RUN = "run"
    KEPT = "kept"


class Policy(StrEnum):
    """Who owns a conversation's history across runs (the module says what each means)."""

    DURABLE_THREAD = "durable_thread"
    MEMORY_WINDOW = "memory_window"


#: The policy in force.
POLICY: Final = Policy.DURABLE_THREAD
#: The threads on which the framework owns the history: memory's window is left out.
FRAMEWORK_OWNS: Final = frozenset({Thread.OWN, Thread.BRIDGED})


def _always(store: Any) -> bool:
    return True


def _sqlite_in_memory(store: Any) -> bool:
    return str(getattr(store, "db_path", "")) == ":memory:"


#: The stores known to keep a thread in the process's memory only, by their type's qualified
#: name (a subclass matches too), each with what makes an instance one.
IN_RAM: Final[Mapping[str, Callable[[Any], bool]]] = {
    "langgraph.checkpoint.memory.InMemorySaver": _always,
    "agents.memory.sqlite_session.SQLiteSession": _sqlite_in_memory,
    "claude_agent_sdk._internal.session_store.InMemorySessionStore": _always,
}
#: What the warning tells the reader to give instead, by the store's framework (its module's
#: first name).
DURABLE_INSTEAD: Final[Mapping[str, str]] = {
    "langgraph": "a durable checkpointer such as LangGraph's PostgresSaver "
    "(langgraph-checkpoint-postgres)",
    "agents": "a durable session (a SQLiteSession on a file, a Redis or SQLAlchemy session), or "
    "none: with memory on, the memory service keeps the conversation",
    "claude_agent_sdk": "a durable session store, or none: the harness keeps the session in the "
    "run's journal",
}


def in_ram(store: Any) -> bool:
    """Whether ``store`` keeps what it holds in the process's memory only (:data:`IN_RAM`)."""
    for kind in type(store).__mro__:
        check = IN_RAM.get(f"{kind.__module__}.{kind.__qualname__}")
        if check is not None:
            return check(store)
    return False


def thread_for(
    store: Any | None,
    *,
    bridged: bool,
    agent_id: str,
    memory: bool,
) -> Thread:
    """The thread a framework works on under :data:`POLICY`, given the thread ``store`` it was
    given (``None``: none): ``bridged`` — the harness can give its slot the memory service's
    thread in this run; ``memory`` — memory is on for it (what replaces a store kept in RAM, as
    the warning says)."""
    if store is not None and not in_ram(store):
        return Thread.OWN
    if POLICY is Policy.MEMORY_WINDOW:
        chosen = Thread.RUN
    else:
        chosen = Thread.BRIDGED if store is None and bridged else Thread.KEPT
    if store is not None:
        _said_in_ram(store, agent_id=agent_id, memory=memory, replaced=chosen is Thread.RUN)
    return chosen


def owns(thread: Thread) -> bool:
    """Whether the framework owns the history on ``thread`` (memory's window left out)."""
    return thread in FRAMEWORK_OWNS


def _said_in_ram(store: Any, *, agent_id: str, memory: bool, replaced: bool) -> None:
    kind = type(store)
    instead = DURABLE_INSTEAD.get(kind.__module__.split(".", 1)[0], "a durable store")
    works = "the run's own thread (its journal) is used instead; " if replaced else ""
    owner = (
        "the memory service's window keeps the conversation"
        if memory
        else "memory is off, so the history will not survive a restart"
    )
    said(
        f"in_ram:{agent_id}:{kind.__module__}.{kind.__qualname__}",
        "%s: its %s keeps the thread in this process's memory only, lost when the worker "
        "restarts: %s%s; give it %s",
        agent_id,
        kind.__name__,
        works,
        owner,
        instead,
        extra={"agent_id": agent_id, "store": kind.__name__, "memory": memory},
    )


__all__ = [
    "DURABLE_INSTEAD",
    "FRAMEWORK_OWNS",
    "IN_RAM",
    "POLICY",
    "Policy",
    "Thread",
    "in_ram",
    "owns",
    "thread_for",
]
