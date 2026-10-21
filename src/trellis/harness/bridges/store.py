"""LangGraph's long-term store (``BaseStore``), backed by the memory service.

A graph's store is what its nodes and tools read and write across threads
(``runtime.store``, ``get_store()``, Deep Agents' ``StoreBackend``). A graph wrapped without a
store of its own is run with a :class:`MemoryStore` while memory is on for the run: each item
is one memory of the run's user (``USER`` visibility: every conversation of theirs, every agent
acting for them, nobody else), so what a graph stores is in the memory service like everything
else the user's agents learned — retrieved into their context, searchable, governed by its
retention. A graph compiled with its own store keeps it (said once).

Each item is a memory whose text is the item as JSON — its namespace, key and value — so the
memory service's search reads it and the store reads it back exactly. ``put`` of a key already
held is the service's ``supersede`` (the old version stays readable in a temporal view);
``put(..., None)``/``delete`` its ``forget``. ``get``, ``list_namespaces`` and a search without
a query read the user's memory inventory; a search with a ``query`` is the service's ranked
recall, in its order (its items' times are when they were read: a recall carries none). The
store holds what the service holds: a value of at most :data:`MAX_CHARS` characters as JSON,
no TTL, no index of the store's own (``index=``/``ttl=`` are not taken).

The calls are the run's: made in its memory scope, they fail as a memory call fails (the
memory SDK's error, which is the graph's), and outside a run there is no store to call.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from trellis.contracts import ConfigurationError
from trellis.memory import MemoryContext

#: The key that marks a memory as a store item (its text is the item as JSON).
MARKER: Final = "langgraph_store"
#: The most characters a memory holds (the service's ``content`` limit).
MAX_CHARS: Final = 8000
#: The inventory page the store reads (the service's largest).
PAGE: Final = 500
#: How many ranked memories a search with a query asks the service for, at most.
RECALL_LIMIT: Final = 100
#: Who may read an item: the run's user, in every conversation and agent of theirs.
VISIBILITY: Final = "USER"


@dataclass(frozen=True, slots=True)
class _Stored:
    """A store item, as the memory holding it says."""

    memory_id: str
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def item(self) -> Item:
        return Item(
            value=self.value,
            key=self.key,
            namespace=self.namespace,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def found(self, score: float | None) -> SearchItem:
        return SearchItem(
            self.namespace, self.key, self.value, self.created_at, self.updated_at, score
        )


class MemoryStore(BaseStore):
    """The graph's store for one run: ``memory`` is the run's memory scope, ``loop`` the
    event loop the run works on (a synchronous call from another thread — a sync node or
    tool — is made there)."""

    __slots__ = ("loop", "memory")

    def __init__(self, memory: MemoryContext, loop: asyncio.AbstractEventLoop) -> None:
        self.memory = memory
        self.loop = loop

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        if _on_loop(self.loop):
            raise ConfigurationError(
                "the run's store is the memory service: on the event loop, call it with its "
                "async methods (aget, aput, asearch...)"
            )
        return asyncio.run_coroutine_threadsafe(self.abatch(list(ops)), self.loop).result()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        given = list(ops)
        # a search with a query is the service's ranking: only the other operations read the
        # inventory (once for the batch)
        ranked_only = all(isinstance(op, SearchOp) and op.query for op in given)
        inventory = None if ranked_only else await self._inventory()
        results: list[Result] = []
        for op in given:
            if isinstance(op, GetOp):
                assert inventory is not None
                found = _held(inventory, op.namespace, op.key)
                results.append(None if found is None else found.item())
            elif isinstance(op, PutOp):
                assert inventory is not None
                inventory = await self._put(inventory, op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(await self._search(inventory, op))
            elif isinstance(op, ListNamespacesOp):
                assert inventory is not None
                results.append(_namespaces(inventory, op))
            else:
                raise ConfigurationError(f"the run's store takes no {type(op).__name__}")
        return results

    # ------------------------------------------------------------------ the operations
    async def _inventory(self) -> list[_Stored]:
        """Every item the user's memory holds now, newest first."""
        found: list[_Stored] = []
        async for memory in self.memory.advanced.memories.iter(page_size=PAGE):
            stored = _parsed(memory)
            if stored is not None:
                found.append(stored)
        return found

    async def _put(self, inventory: list[_Stored], op: PutOp) -> list[_Stored]:
        if op.index not in (None, False) or op.ttl is not None:
            raise ConfigurationError(
                "the run's store is the memory service, which indexes every memory itself and "
                "keeps it for its retention: put() takes no index= or ttl="
            )
        held = _held(inventory, op.namespace, op.key)
        rest = [s for s in inventory if s is not held]
        if op.value is None:
            if held is not None:
                await self.memory.forget(held.memory_id)
            return rest
        content = _content(op.namespace, op.key, op.value)
        now = datetime.now(UTC)
        if held is None:
            ack = await self.memory.remember(content, visibility=VISIBILITY)
            created = now
        elif _content(held.namespace, held.key, held.value) == content:
            return inventory
        else:
            ack = await self.memory.update(held.memory_id, content, reason="the store's put")
            created = held.created_at
        written = _Stored(ack.memory_id, op.namespace, op.key, dict(op.value), created, now)
        return [written, *rest]

    async def _search(self, inventory: list[_Stored] | None, op: SearchOp) -> list[SearchItem]:
        if not op.query:
            assert inventory is not None
            under = [s for s in inventory if _under(s, op) and _fits(s.value, op.filter)]
            return [s.found(None) for s in under[op.offset : op.offset + op.limit]]
        wanted = min(RECALL_LIMIT, max(op.limit + op.offset, 10) * 4)
        ranked = await self.memory.search(op.query, limit=wanted, kinds=["memory"])
        matched: list[_Stored] = []
        for hit in ranked:
            # a ranked hit carries no times of its own: it says when it was read
            now = datetime.now(UTC)
            stored = _parsed_text(hit.id, hit.text, now, now)
            if stored is not None and _under(stored, op) and _fits(stored.value, op.filter):
                matched.append(stored)
        chosen = matched[op.offset : op.offset + op.limit]
        count = len(matched) or 1
        return [s.found(1 - (op.offset + n) / count) for n, s in enumerate(chosen)]


# ---------------------------------------------------------------------- items as memories
def _content(namespace: Sequence[str], key: str, value: dict[str, Any]) -> str:
    """The memory text of an item: its namespace, key and value as JSON."""
    text = json.dumps(
        {MARKER: 1, "namespace": list(namespace), "key": key, "value": value},
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(text) > MAX_CHARS:
        raise ConfigurationError(
            f"store item {'/'.join(namespace)}/{key} is {len(text)} characters as JSON: a "
            f"memory holds at most {MAX_CHARS}"
        )
    return text


def _parsed(memory: Any) -> _Stored | None:
    """The item an inventory memory is (``None``: another memory), with its times."""
    times = memory.model_extra
    created = datetime.fromisoformat(times["created_at"])
    updated = datetime.fromisoformat(times["updated_at"])
    return _parsed_text(memory.memory_id, memory.content, created, updated)


def _parsed_text(memory_id: str, text: str, created: datetime, updated: datetime) -> _Stored | None:
    """The item a memory's text is, or ``None`` for any other memory."""
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get(MARKER) != 1:
        return None
    namespace, key, value = data.get("namespace"), data.get("key"), data.get("value")
    if not isinstance(namespace, list) or not isinstance(key, str) or not isinstance(value, dict):
        return None
    return _Stored(memory_id, tuple(str(n) for n in namespace), key, value, created, updated)


def _held(inventory: list[_Stored], namespace: tuple[str, ...], key: str) -> _Stored | None:
    return next((s for s in inventory if s.namespace == namespace and s.key == key), None)


def _under(stored: _Stored, op: SearchOp) -> bool:
    prefix = tuple(op.namespace_prefix)
    return stored.namespace[: len(prefix)] == prefix


#: The comparisons a search filter takes (``{"field": {"$gt": 3}}``), as LangGraph's own stores.
COMPARISONS: Final = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
}


def _fits(value: dict[str, Any], given: dict[str, Any] | None) -> bool:
    """Whether ``value`` matches a search filter: each field equal, or each comparison of a
    ``{"$op": operand}`` true."""
    for name, wanted in (given or {}).items():
        actual = value.get(name)
        if isinstance(wanted, dict) and wanted and all(k in COMPARISONS for k in wanted):
            if not all(COMPARISONS[k](actual, v) for k, v in wanted.items()):
                return False
        elif actual != wanted:
            return False
    return True


def _namespaces(inventory: list[_Stored], op: ListNamespacesOp) -> list[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for stored in inventory:
        namespace = stored.namespace
        if all(_matches(c, namespace) for c in op.match_conditions or ()):
            found.add(namespace[: op.max_depth] if op.max_depth is not None else namespace)
    return sorted(found)[op.offset : op.offset + op.limit]


def _matches(condition: MatchCondition, namespace: tuple[str, ...]) -> bool:
    path = tuple(condition.path)
    if not path:
        return True
    if len(path) > len(namespace):
        return False
    part = namespace[: len(path)] if condition.match_type == "prefix" else namespace[-len(path) :]
    return all(p in ("*", n) for p, n in zip(path, part, strict=True))


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


__all__ = ["MAX_CHARS", "MemoryStore"]
