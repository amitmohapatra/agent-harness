"""A value read from a service and kept fresh: what the key is, the memory tools listed, an
agent's toolbox.

Read once, kept for ``ttl`` seconds, read again after that — one read at a time, however many
runs ask at once. While the service cannot be reached the last value read is kept (and asked
again after ``retry`` seconds, so a service that is down is not asked by every run): an outage
degrades to what was known. Only a value that was never read raises, and so does an error the
caller says no stale value may hide (``fatal``: a key the service now refuses).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

log = logging.getLogger("trellis.fresh")


class Fresh[T]:
    def __init__(
        self,
        read: Callable[[], Awaitable[T]],
        *,
        what: str,
        ttl: float,
        retry: float,
        fatal: tuple[type[Exception], ...] = (),
    ) -> None:
        self._read = read
        self._what = what
        self._ttl = ttl
        self._retry = retry
        self._fatal = fatal
        self._until = 0.0
        self._lock = asyncio.Lock()
        #: the last value read (``None`` before the first)
        self.value: T | None = None

    async def get(self) -> T:
        async with self._lock:
            now = _now()
            if self.value is not None and now < self._until:
                return self.value
            try:
                self.value = await self._read()
            except self._fatal:
                raise
            except Exception as exc:
                if self.value is None:
                    raise
                log.warning(
                    "%s could not be read again (%s: %s): the last one read is used",
                    self._what,
                    type(exc).__name__,
                    exc,
                )
                self._until = now + self._retry
                return self.value
            self._until = now + self._ttl
            return self.value


def _now() -> float:
    return time.monotonic()
