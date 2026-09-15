"""Cooperative cancellation (§39).

Two things must both work: an agent that *checks* the token stops promptly, and an agent
that does not still gets cancelled because the harness cancels the underlying asyncio task.
Cancellation is never swallowed — :class:`asyncio.CancelledError` propagates out of the
harness — and a token cancelled by a deadline reports ``reason="timeout"`` so the caller
can tell a timeout from an operator cancel.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime


class CancellationToken:
    """A token an agent can poll, await, or hang a callback on."""

    __slots__ = ("_callbacks", "_event", "_reason")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()
        for callback in self._callbacks:
            # A listener that raises must not stop the other listeners, or cancellation.
            with contextlib.suppress(Exception):
                callback(reason)

    def on_cancel(self, callback: Callable[[str], None]) -> None:
        if self._event.is_set():
            callback(self._reason or "cancelled")
        else:
            self._callbacks.append(callback)

    async def wait(self) -> str:
        await self._event.wait()
        return self._reason or "cancelled"

    def raise_if_cancelled(self) -> None:
        """Call this between steps in a long agent. Raises :class:`asyncio.CancelledError`."""
        if self._event.is_set():
            raise asyncio.CancelledError(self._reason or "cancelled")

    def child(self) -> CancellationToken:
        """A token cancelled whenever this one is (but cancellable on its own too)."""
        token = CancellationToken()
        self.on_cancel(token.cancel)
        return token


def remaining_seconds(deadline: datetime | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


def tightest(*deadlines: datetime | None) -> datetime | None:
    """The earliest non-null deadline. A nested call never outlives its parent (§38)."""
    present = [d for d in deadlines if d is not None]
    return min(present) if present else None
