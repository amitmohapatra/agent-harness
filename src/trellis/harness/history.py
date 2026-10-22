"""Who owns a conversation's history across runs: one policy for every adapter.

A framework that keeps a conversation keeps it in a *thread store* — a LangGraph checkpointer,
an OpenAI Agents session, a conversation on OpenAI's servers. The memory service keeps it too:
its thread, given back as the *window* of the pushed context. One of the two owns it, never
both, and :func:`owner` says which:

* ``Owner.FRAMEWORK`` — the framework was given a durable store (it survives a worker restart,
  and a replica elsewhere reads it): memory's window is left out;
* ``Owner.MEMORY`` — the framework keeps no store, or one kept in this process's memory only
  (:data:`IN_RAM`), lost on a restart: memory's window owns the history. A store kept in RAM is
  said once per agent and store type.

A store of a type :data:`IN_RAM` does not name is taken as durable: the harness cannot see
where an unknown store keeps its data, and a team that brings one (``PostgresSaver``, a Redis
session) chose it for that.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Final

from trellis.harness.bounded import SAID_ONCE

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
}


def in_ram(store: Any) -> bool:
    """Whether ``store`` keeps what it holds in the process's memory only (:data:`IN_RAM`)."""
    for kind in type(store).__mro__:
        check = IN_RAM.get(f"{kind.__module__}.{kind.__qualname__}")
        if check is not None:
            return check(store)
    return False


def owner(store: Any | None, *, agent_id: str) -> Owner:
    """Who owns the history, given the thread ``store`` the framework keeps (``None``: none)."""
    if store is None:
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


__all__ = ["IN_RAM", "Owner", "in_ram", "owner"]
