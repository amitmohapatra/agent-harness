"""Bounded containers for the harness's own client-side state: a long-lived process keeps a
little per tenant, tool, prompt or thread, and none of it may grow without bound (invariant
I9). Each holds at most ``maxsize`` entries and forgets the least recently used one past it —
which is always safe for what it holds: a cache read again, a set of what was already sent
(sent again), a log-once name (said again)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from typing import Final

#: The default bound: generous for any one deployment's tenants, tools or prompts.
MAXSIZE: Final = 1024


class LRU[K, V]:
    """A mapping of at most ``maxsize`` entries: reading or writing one makes it the most
    recently used; past the bound the least recently used goes."""

    def __init__(self, maxsize: int = MAXSIZE) -> None:
        if maxsize < 1:
            raise ValueError("an LRU holds at least one entry")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __getitem__(self, key: K) -> V:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def setdefault(self, key: K, make: Callable[[], V]) -> V:
        """The value of ``key``, made (``make()``) and kept when it is not there."""
        if key in self._data:
            return self[key]
        value = make()
        self[key] = value
        return value

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._data.pop(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(list(self._data))

    def values(self) -> list[V]:
        return list(self._data.values())


class LRUSet[K]:
    """A set of at most ``maxsize`` members, the least recently added or touched forgotten
    past it."""

    def __init__(self, members: Iterable[K] = (), maxsize: int = MAXSIZE) -> None:
        self._data: LRU[K, None] = LRU(maxsize)
        for member in members:
            self.add(member)

    def add(self, member: K) -> None:
        self._data[member] = None

    def discard(self, member: K) -> None:
        self._data.pop(member)

    def __contains__(self, member: object) -> bool:
        return member in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)


#: How many names a "said once" warning remembers (a model name, a model type): past it the
#: least recently said may be said again.
SAID_ONCE: Final = 256
