"""Cancellation tokens and deadline arithmetic (§38, §39)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from universal_agent_harness import CancellationToken
from universal_agent_harness.runtime.cancellation import remaining_seconds, tightest


async def test_cancel_sets_reason_and_wakes_waiters():
    token = CancellationToken()
    waiter = asyncio.ensure_future(token.wait())
    await asyncio.sleep(0)
    token.cancel("timeout")
    assert await waiter == "timeout"
    assert token.cancelled and token.reason == "timeout"


async def test_raise_if_cancelled_propagates_cancellation():
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()


async def test_callbacks_fire_once_and_late_registration_fires_immediately():
    token = CancellationToken()
    seen: list[str] = []
    token.on_cancel(seen.append)
    token.cancel("stop")
    token.cancel("again")
    token.on_cancel(seen.append)
    assert seen == ["stop", "stop"]


async def test_child_tokens_follow_the_parent():
    parent = CancellationToken()
    child = parent.child()
    parent.cancel("parent gone")
    assert child.cancelled and child.reason == "parent gone"


async def test_a_listener_that_raises_does_not_break_cancellation():
    token = CancellationToken()
    token.on_cancel(lambda reason: (_ for _ in ()).throw(RuntimeError("bad listener")))
    token.cancel()
    assert token.cancelled


def test_deadline_helpers():
    now = datetime.now(UTC)
    assert tightest(None, None) is None
    assert tightest(now + timedelta(seconds=5), now + timedelta(seconds=1)) == now + timedelta(seconds=1)
    assert remaining_seconds(None) is None
    assert remaining_seconds(now - timedelta(seconds=5)) == 0.0
