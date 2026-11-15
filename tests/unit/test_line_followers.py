"""Following the run ahead in a conversation's line (``line.Followers``): one event stream a
run, shared by every run waiting behind it here; a stream that says nothing for
:data:`~trellis.harness.line.STILL_SECONDS` lets its waiters read the line again; one that
cannot be read says it cannot be followed now."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from trellis.contracts import RunEventType
from trellis.harness import line
from trellis.harness.line import Followers
from trellis.runs import RunsError

if TYPE_CHECKING:
    from trellis.harness.runlog import EventLog


class Log:
    """An event log whose stream of a run yields what :attr:`said` gets, until it ends."""

    def __init__(self, *, fails: bool = False) -> None:
        self.streams: list[str] = []
        self.said: asyncio.Queue[Any] = asyncio.Queue()
        self.fails = fails

    async def stream_events(self, run_id: str, *, after: int, tenant: str) -> AsyncIterator[Any]:
        self.streams.append(run_id)
        if self.fails:
            raise RunsError("the event log is down")
        while True:
            yield await self.said.get()


def finished(position: int, attempt: int = 1) -> Any:
    event = SimpleNamespace(type=RunEventType.RUN_FINISHED, attempt=attempt)
    return SimpleNamespace(position=position, event=event)


async def test_runs_waiting_behind_one_run_share_its_one_stream() -> None:
    log = Log()
    followers = Followers(lambda: cast("EventLog", log))
    first = asyncio.ensure_future(followers.left("run_ahead", 1, "t"))
    second = asyncio.ensure_future(followers.left("run_ahead", 1, "t"))
    await asyncio.sleep(0)
    log.said.put_nowait(finished(7))
    assert await first is True and await second is True
    assert log.streams == ["run_ahead"]  # one stream, both waiters on it


async def test_a_run_ahead_that_says_nothing_lets_its_waiters_read_the_line_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(line, "STILL_SECONDS", 0.05)
    quiet = cast("EventLog", Log())
    assert await Followers(lambda: quiet).left("run_quiet", 1, "t") is True


async def test_an_event_log_that_cannot_be_read_cannot_be_followed_now() -> None:
    down = cast("EventLog", Log(fails=True))
    assert await Followers(lambda: down).left("run_ahead", 1, "t") is False
