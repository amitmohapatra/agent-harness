"""``holding.Holds``: the runs a harness holds in its run store while it executes them — one
hold a run however many parts hold it, a heartbeat at once and every third of
``HOLD_SECONDS``, a refused one tried again in silence, a store that cannot be reached said in
the log, and a release that lets go in the store too."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from trellis.harness import holding
from trellis.harness.holding import Holds, name_of
from trellis.runs import DependencyUnavailableError, LeaseLostError


class Store:
    """What the holds call of a run store, recorded; each call answers what ``answers`` says
    (an exception to raise), else nothing."""

    def __init__(self) -> None:
        self.beats: list[tuple[str, str, int, str | None]] = []
        self.released: list[tuple[str, str, str | None]] = []
        self.answers: list[Exception | None] = []

    def _answer(self) -> None:
        answer = self.answers.pop(0) if self.answers else None
        if answer is not None:
            raise answer

    async def heartbeat(self, run_id: str, worker_id: str, **given: Any) -> None:
        self.beats.append((run_id, worker_id, given["lease_seconds"], given["tenant"]))
        self._answer()

    async def release(self, run_id: str, worker_id: str, **given: Any) -> None:
        self.released.append((run_id, worker_id, given["tenant"]))
        self._answer()


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(holding, "HOLD_SECONDS", 0.3)  # a heartbeat each 0.1 s


async def beats(store: Store, n: int) -> None:
    async with asyncio.timeout(5):
        while len(store.beats) < n:  # noqa: ASYNC110 - the holds' own task
            await asyncio.sleep(0.01)


def test_each_harness_holds_under_a_name_of_its_own() -> None:
    one, two = name_of(), name_of()
    assert one != two and one.startswith("trellis@") and len(one) <= 200


async def test_a_run_is_held_once_while_anything_holds_it() -> None:
    store = Store()
    holds = Holds(lambda: store, name="h1")
    holds.hold("run_1", "t")
    holds.hold("run_1", "t")
    await beats(store, 1)
    assert store.beats[0] == ("run_1", "h1", 0.3, "t") and holds.holds("run_1")
    holds.let_go("run_1")
    await beats(store, 3)  # still held: renewed
    holds.let_go("run_1")
    holds.let_go("run_1")  # nothing held: no change
    assert not holds.holds("run_1")
    held = len(store.beats)
    await asyncio.sleep(0.25)
    assert len(store.beats) == held


async def test_a_refused_hold_is_tried_again_in_silence_and_an_unreachable_store_is_said(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Store()
    store.answers = [
        LeaseLostError("run_1 is PAUSED", status=409),
        DependencyUnavailableError("agent-runs is restarting", status=503),
    ]
    holds = Holds(lambda: store, name="h1")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        async with holds.holding("run_1", "t"):
            await beats(store, 3)
    assert "run run_1 not held: agent-runs is restarting" in caplog.text
    assert "PAUSED" not in caplog.text
    assert not holds.holds("run_1")


async def test_a_release_lets_go_here_and_in_the_store(caplog: pytest.LogCaptureFixture) -> None:
    store = Store()
    holds = Holds(lambda: store, name="h1")
    holds.hold("run_1", "t")
    await beats(store, 1)
    await holds.release("run_1", "t")
    assert store.released == [("run_1", "h1", "t")] and not holds.holds("run_1")
    store.answers = [LeaseLostError("run_2 ended", status=409)]
    await holds.release("run_2", "t")  # never held here, and ended there: nothing to do
    store.answers = [DependencyUnavailableError("down", status=503)]
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        await holds.release("run_3", "t")
    assert "run run_3 not let go of: down" in caplog.text


async def test_closing_ends_every_hold() -> None:
    store = Store()
    holds = Holds(lambda: store)
    holds.hold("run_1", "t")
    holds.hold("run_2", "t")
    await beats(store, 2)
    holds.aclose()
    assert not holds.holds("run_1") and not holds.holds("run_2")
    held = len(store.beats)
    await asyncio.sleep(0.25)
    assert len(store.beats) == held
