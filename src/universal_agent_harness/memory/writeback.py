"""Post-execution memory writes, off the critical path (§10, §74).

The current turn's result is returned as soon as the agent finishes; observations are
submitted afterwards. The queue is **bounded**: when more work is pending than
``max_pending``, the harness stops scheduling and writes inline instead of growing an
unbounded backlog. The Memory Service itself provides the durable semantics — this is only
about not blocking the caller.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("universal_agent_harness.memory")


class WritebackQueue:
    """A bounded set of background tasks. Not a durable queue — the service is that."""

    __slots__ = ("_pending", "max_pending", "on_error")

    def __init__(
        self, max_pending: int = 256, on_error: Callable[[BaseException], None] | None = None
    ) -> None:
        self.max_pending = max_pending
        self._pending: set[asyncio.Task[Any]] = set()
        self.on_error = on_error

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def saturated(self) -> bool:
        return len(self._pending) >= self.max_pending

    def submit(self, coro: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any] | None:
        """Schedule ``coro``. Returns ``None`` (and does not schedule) when saturated —
        the caller then decides to await it inline or drop it."""
        if self.saturated:
            return None
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:  # pragma: no cover - no running loop
            return None
        if name:
            task.set_name(name)
        self._pending.add(task)
        task.add_done_callback(self._finish)
        return task

    def _finish(self, task: asyncio.Task[Any]) -> None:
        self._pending.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:  # pragma: no cover
                log.debug("writeback error handler failed", exc_info=True)
        else:
            log.warning("memory writeback failed: %s", error, exc_info=error)

    async def drain(self, timeout: float | None = 30.0) -> int:  # noqa: ASYNC109
        """Await outstanding writes. Used at shutdown and in tests; never on the hot path."""
        if not self._pending:
            return 0
        pending = list(self._pending)
        done, not_done = await asyncio.wait(pending, timeout=timeout)
        for task in not_done:
            task.cancel()
        return len(done)
