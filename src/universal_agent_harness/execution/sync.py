"""Sync/async bridge.

``asyncio.run`` must never be called when a loop is already running in this thread, so
synchronous callers need a loop of their own. The subtlety is *which* loop: a throwaway loop
per call leaves any connection opened during that call — an HTTP client's pool, a database
handle — bound to a loop that no longer exists. The next call on that connection fails with
"Event loop is closed", and closing the client raises.

So the bridge keeps **one** background loop for the process and runs every synchronous call
on it. Async resources then have a single, stable loop affinity, and a sync agent using the
Memory Service behaves exactly like an async one.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Awaitable, Coroutine
from typing import Any


def in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _BridgeLoop:
    """A single background event loop, started on first use."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="uah-sync-bridge", daemon=True
            )
            thread.start()
            self._loop, self._thread = loop, thread
            return loop

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        loop = self._ensure()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def shutdown(self) -> None:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()


_BRIDGE = _BridgeLoop()
atexit.register(_BRIDGE.shutdown)


def run_sync[T](awaitable: Awaitable[T] | Coroutine[Any, Any, T]) -> T:
    """Run ``awaitable`` to completion from synchronous code, on the shared bridge loop."""
    return _BRIDGE.run(_as_coroutine(awaitable))


def shutdown_bridge() -> None:
    """Stop the bridge loop. Called at interpreter exit; exposed for tests."""
    _BRIDGE.shutdown()


async def _as_coroutine[T](awaitable: Awaitable[T]) -> T:
    return await awaitable
