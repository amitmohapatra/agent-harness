"""Helpers of the live suite."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Final

from trellis import Harness
from trellis.memory import MemoryContext

#: How long a background effect of the services (a projection, a learning job) may take.
SETTLE_SECONDS: Final = 60.0


async def eventually(
    check: Callable[[], Awaitable[bool]], *, within: float = SETTLE_SECONDS, every: float = 1.0
) -> bool:
    """Whether ``check`` holds within ``within`` seconds."""
    deadline = time.monotonic() + within
    while True:
        if await check():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(every)


def memory_scope(
    h: Harness, *, user: str, agent_id: str, thread: str | None = None
) -> MemoryContext:
    """The memory service in a user's (and thread's) scope, read as the agent reads it."""
    assert h.memory is not None
    scope = {"tenant_id": h.settings.tenant, "user_id": user, "agent_id": agent_id}
    if thread is not None:
        scope["thread_id"] = thread
    return h.memory.client.bind(**scope)
