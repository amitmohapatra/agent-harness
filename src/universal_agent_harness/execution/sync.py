"""Sync/async bridge (§72).

``asyncio.run`` must never be called when a loop is already running in this thread, so when
one is, the coroutine is run on a fresh loop in a worker thread. That keeps the harness
usable from synchronous framework code (a sync LangGraph node, a CrewAI tool) without ever
blocking or re-entering the caller's loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Coroutine
from typing import Any


def in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_sync[T](awaitable: Awaitable[T] | Coroutine[Any, Any, T]) -> T:
    """Run ``awaitable`` to completion from synchronous code."""
    if not in_event_loop():
        return asyncio.run(_as_coroutine(awaitable))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _as_coroutine(awaitable)).result()


async def _as_coroutine[T](awaitable: Awaitable[T]) -> T:
    return await awaitable
