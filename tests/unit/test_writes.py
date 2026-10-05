"""Background writes: drained in the background, retried while a failure may pass, holding the
run when the queue is full, and kept in the spool (then replayed) or counted lost when this
process cannot deliver them."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from trellis.contracts import AgentExecutionContext, RunEvent
from trellis.harness import writes as module
from trellis.harness.events import RunEvents
from trellis.harness.writes import SPOOL_FILE, Writes
from trellis.memory.errors import DependencyUnavailableError, ValidationError


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real backoff between attempts, and short shutdown bounds."""
    monkeypatch.setattr(module, "WRITE_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(module, "SUBMIT_WAIT_SECONDS", 0.05)


def run_events() -> tuple[RunEvents, list[RunEvent]]:
    seen: list[RunEvent] = []
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    events.listen(seen.append)
    return events, seen


async def test_writes_run_in_the_background_and_drain_waits_for_them() -> None:
    writes, done = Writes(), []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    for _ in range(10):
        await writes.submit("slow", slow)
    assert done == []
    await writes.drain()
    assert len(done) == 10
    await writes.aclose()


async def test_a_write_that_may_pass_is_tried_again() -> None:
    writes, tries = Writes(), []

    async def flaky() -> None:
        tries.append(1)
        if len(tries) < module.WRITE_ATTEMPTS:
            raise DependencyUnavailableError("memory is restarting", retryable=True)

    await writes.submit("memory.transcript", flaky)
    await writes.drain()
    assert len(tries) == module.WRITE_ATTEMPTS and writes.failed == 0
    await writes.aclose()


async def test_a_failed_write_is_counted_and_reported_to_the_run() -> None:
    writes = Writes()
    events, seen = run_events()
    tries: list[int] = []

    async def boom() -> None:
        tries.append(1)
        raise ConnectionError("memory is down")

    await writes.submit("memory.transcript", boom, events=events)
    await writes.drain()
    assert writes.failed == 1 and len(tries) == module.WRITE_ATTEMPTS
    event: RunEvent = seen[0]
    assert event.data["name"] == "warning"
    assert "memory is down" in event.data["message"]
    await writes.aclose()


async def test_a_refusal_is_not_tried_again_nor_spooled(tmp_path: Path) -> None:
    writes, tries = Writes(spool=tmp_path), []

    async def refused() -> None:
        tries.append(1)
        raise ValidationError("bad request", retryable=False)

    await writes.submit("memory.outcome", refused, record={"op": "x", "scope": {}, "args": {}})
    await writes.drain()
    assert tries == [1] and writes.failed == 1 and writes.spooled == 0
    assert not (tmp_path / SPOOL_FILE).exists()
    await writes.aclose()


def test_the_loop_shutting_down_drains_the_queue() -> None:
    done: list[int] = []
    writes = Writes()

    async def write() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    async def main() -> None:
        for _ in range(20):
            await writes.submit("w", write)
        # returning without drain: asyncio.run cancels the workers, which finish first

    asyncio.run(main())
    assert len(done) == 20


async def test_a_full_queue_holds_the_writer_until_there_is_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "MAX_PENDING", 1)
    monkeypatch.setattr(module, "WRITERS", 1)
    monkeypatch.setattr(module, "SUBMIT_WAIT_SECONDS", 5.0)
    writes, done = Writes(), []
    gate = asyncio.Event()

    async def held() -> None:
        await gate.wait()
        done.append("held")

    async def write(name: str) -> Any:
        async def work() -> None:
            done.append(name)

        return work

    await writes.submit("first", held)  # taken by the one writer, which waits on the gate
    await asyncio.sleep(0)
    await writes.submit("second", await write("second"))  # fills the queue
    third = asyncio.create_task(writes.submit("third", await write("third")))
    await asyncio.sleep(0.01)
    assert not third.done()  # the writer waits: no room, but nothing dropped
    gate.set()
    await third
    await writes.drain()
    assert done == ["held", "second", "third"] and writes.failed == 0
    await writes.aclose()


async def test_a_queue_that_stays_full_gives_the_write_up_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(module, "MAX_PENDING", 1)
    writes = Writes(spool=tmp_path)
    events, seen = run_events()
    gate = asyncio.Event()

    async def held() -> None:
        await gate.wait()

    for _ in range(module.WRITERS + 1):  # every writer busy, and the queue full
        await writes.submit("held", held)
    await asyncio.sleep(0)
    await writes.submit("second", held, events=events)  # no record: cannot be kept
    assert writes.failed == 1
    assert seen[0].data["message"] == "second: the write queue stayed full"
    await writes.submit("kept", held, record={"op": "x", "scope": {}, "args": {}})
    assert writes.spooled == 1  # with a record, kept for the next start instead
    gate.set()
    await writes.aclose()


async def test_undelivered_writes_are_spooled_and_replayed_at_the_next_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def down() -> None:
        raise ConnectionError("memory is down")

    first = Writes(spool=tmp_path / "spool")
    record = {"op": "record_messages", "scope": {"tenant_id": "t"}, "args": {"n": 1}}
    await first.submit("memory.transcript", down, record=record)
    await first.drain()
    assert (first.spooled, first.failed) == (1, 0)
    [kept] = (tmp_path / "spool" / SPOOL_FILE).read_text().splitlines()
    assert json.loads(kept) == {"label": "memory.transcript", **record}
    await first.aclose()

    replayed: list[dict[str, Any]] = []

    def replay(found: dict[str, Any]) -> Any:
        async def work() -> None:
            replayed.append(found)

        return work

    with caplog.at_level(logging.INFO, logger="trellis.writes"):
        second = Writes(spool=tmp_path / "spool", replay=replay)
        second.start()
        await second.drain()
    assert replayed == [{"label": "memory.transcript", **record}]
    assert "replaying 1 spooled write(s)" in caplog.text
    assert list((tmp_path / "spool").iterdir()) == []  # claimed and removed
    await second.aclose()


async def test_a_spool_line_nothing_can_replay_is_dropped_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / SPOOL_FILE).write_text(
        "not json\n" + json.dumps({"op": "unknown", "scope": {}, "args": {}}) + "\n"
    )
    writes = Writes(spool=tmp_path, replay=lambda record: None)
    with caplog.at_level(logging.ERROR, logger="trellis.writes"):
        writes.start()
    assert "cannot be read, dropped" in caplog.text
    assert "nothing here replays the spooled write" in caplog.text
    await writes.aclose()
    Writes(spool=tmp_path, replay=lambda record: None).start()  # nothing left: nothing to do
    Writes(spool=tmp_path).start()  # no replay: the spool is left alone


async def test_a_replay_larger_than_the_queue_keeps_the_rest_for_later(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(module, "MAX_PENDING", 1)
    monkeypatch.setattr(module, "WRITERS", 0)  # nobody drains: the queue stays full
    lines = [json.dumps({"op": "x", "scope": {}, "args": {"n": n}}) for n in range(3)]
    (tmp_path / SPOOL_FILE).write_text("\n".join(lines) + "\n")

    async def nothing() -> None:
        return None

    writes = Writes(spool=tmp_path, replay=lambda record: nothing)
    writes.start()
    kept = (tmp_path / SPOOL_FILE).read_text().splitlines()
    assert [json.loads(line)["args"]["n"] for line in kept] == [1, 2]


async def test_an_unwritable_spool_loses_the_write_and_says_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("")  # a file where the spool directory should be

    async def down() -> None:
        raise ConnectionError("memory is down")

    writes = Writes(spool=blocked / "spool")
    with caplog.at_level(logging.ERROR, logger="trellis.writes"):
        await writes.submit("w", down, record={"op": "x", "scope": {}, "args": {}})
        await writes.drain()
    assert (writes.failed, writes.spooled) == (1, 0)
    assert "cannot be written" in caplog.text
    await writes.aclose()


async def test_a_shutdown_past_the_drain_bound_keeps_or_counts_what_is_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(module, "DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(module, "WRITERS", 1)
    writes = Writes(spool=tmp_path)
    started = asyncio.Event()

    async def stuck() -> None:
        started.set()
        await asyncio.Event().wait()

    record = {"op": "x", "scope": {}, "args": {}}
    await writes.submit("stuck", stuck, record=record)
    await writes.submit("queued", stuck)  # no record: lost, and counted
    await started.wait()
    await writes.aclose()
    assert (writes.spooled, writes.failed) == (1, 1)
    assert json.loads((tmp_path / SPOOL_FILE).read_text())["label"] == "stuck"


def test_a_loop_stopped_mid_write_keeps_the_cut_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """asyncio.run returning cancels the writers; the drain bound passes with the write
    still hanging, so it is kept."""
    monkeypatch.setattr(module, "DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(module, "WRITERS", 1)
    writes = Writes(spool=tmp_path)

    async def main() -> None:
        started = asyncio.Event()

        async def stuck() -> None:
            started.set()
            await asyncio.Event().wait()

        await writes.submit("stuck", stuck, record={"op": "x", "scope": {}, "args": {}})
        await started.wait()

    asyncio.run(main())
    assert writes.spooled == 1


async def test_a_harness_replays_memory_writes_only_with_memory_on(tmp_path: Path) -> None:
    from trellis import Harness, Settings

    async with Harness(config=Settings(spool_dir=str(tmp_path))) as h:
        assert h._replay({"op": "record_messages", "scope": {}, "args": {}}) is None


async def test_background_writes_belong_to_no_run() -> None:
    from trellis.harness import runtime as runtime_module

    writes, seen = Writes(), []

    async def look() -> None:
        seen.append(runtime_module.current())

    token = runtime_module._current.set(object())  # type: ignore[arg-type]
    try:
        await writes.submit("look", look)  # the first write starts the workers, in a run
    finally:
        runtime_module._current.reset(token)
    await writes.drain()
    assert seen == [None]
    await writes.aclose()
