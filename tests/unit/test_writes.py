"""Background writes: drained in the background, retried while a failure may pass, holding the
run when the queue is full, and kept in the spool (then replayed) or counted lost when this
process cannot deliver them."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from trellis.contracts import AgentExecutionContext, RunEvent
from trellis.harness import writes as module
from trellis.harness.events import RunEvents
from trellis.harness.writes import SPOOL_FILE, Writes
from trellis.memory.breaker import CircuitBreaker
from trellis.memory.errors import CircuitOpenError, DependencyUnavailableError, ValidationError
from trellis.memory.errors import TimeoutError as MemoryTimeoutError


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


def test_the_queue_holds_10_000_writes_for_4_writers_and_a_full_one_waits_5_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """docs/architecture.md: ``MAX_PENDING`` 10 000, ``WRITERS`` 4, ``WRITE_ATTEMPTS`` 6 (one
    more than the five failed calls that open the memory client's circuit: SCN-6),
    ``SUBMIT_WAIT_SECONDS`` 5."""
    monkeypatch.undo()  # the values themselves, not the fast ones
    assert (module.MAX_PENDING, module.WRITERS, module.WRITE_ATTEMPTS) == (10_000, 4, 6)
    assert CircuitBreaker().threshold == 5
    assert module.SUBMIT_WAIT_SECONDS == 5.0


async def test_a_write_that_may_pass_is_tried_again() -> None:
    writes, tries = Writes(), []

    async def flaky() -> None:
        tries.append(1)
        if len(tries) < 3:
            raise DependencyUnavailableError("memory is restarting", retryable=True)

    await writes.submit("memory.transcript", flaky)
    await writes.drain()
    assert len(tries) == 3 and writes.failed == 0  # the third of its attempts
    await writes.aclose()


def circuit_open(retry_after: float) -> CircuitOpenError:
    """What the memory client raises, sending nothing, while its circuit is open."""
    return CircuitOpenError(
        "memory service circuit open after 5 failed calls",
        code="CIRCUIT_OPEN",
        status=0,
        retryable=True,
        retry_after=retry_after,
    )


async def test_a_write_waits_out_an_open_circuit_spending_no_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCN-6: the memory service down for longer than a write's attempts (its client's circuit
    open, then open again after a probe that failed): the write waits each cool-down, queued,
    and is delivered once the service is back — every wait is the circuit's, none an attempt."""
    waits: list[float] = []
    sleep = asyncio.sleep

    async def waited(seconds: float) -> None:
        waits.append(seconds)
        await sleep(0)

    monkeypatch.setattr(module.asyncio, "sleep", waited)
    monkeypatch.setattr(module, "WRITE_BACKOFF_SECONDS", 0.5)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: 0.0)
    writes, tries = Writes(), []
    down = [circuit_open(30.0), circuit_open(0.0)] * 3 + [
        DependencyUnavailableError("the probe failed", retryable=True),
        DependencyUnavailableError("the probe failed", retryable=True),
        circuit_open(12.5),
    ]

    async def write() -> None:
        tries.append(1)
        if down:
            raise down.pop(0)

    await writes.submit("memory.transcript", write)
    await writes.drain()
    assert len(tries) == 10 and writes.failed == writes.spooled == 0
    # the circuit's cool-down (none left: half a second, not at once), and the backoff after
    # the two failed probes, its attempts whole again after each cool-down
    assert waits == [30.0, 0.5] * 3 + [0.0, 0.0, 12.5], waits
    await writes.aclose()


async def test_the_first_write_of_an_outage_outlasts_the_failures_that_open_the_circuit() -> None:
    """SCN-6, the first seconds of an outage: the only write in flight fails five times for
    want of the service — the failed calls that open its client's circuit — and has a try
    left: refused by the open circuit, it waits and is delivered when the service is back."""
    writes, tries = Writes(), []
    down: list[Exception] = [
        *(DependencyUnavailableError("unavailable", retryable=True) for _ in range(5)),
        circuit_open(0.0),
    ]

    async def write() -> None:
        tries.append(1)
        if down:
            raise down.pop(0)

    await writes.submit("memory.transcript", write)
    await writes.drain()
    assert len(tries) == 7 and writes.failed == 0
    await writes.aclose()


async def test_a_write_whose_own_calls_time_out_gives_up_though_the_circuit_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service is up, and answers this write's every call with its own timeout (a 504:
    ``/v1/verify`` past its 15 s budget on a busy machine). Those timeouts open the client's
    circuit, and its cool-downs spend no attempt — but the timed-out calls do, across them: the
    write is given up after its attempts, not tried again forever. It was: its attempts whole
    after each cool-down, every probe timing out and opening the circuit again, a drain never
    ended (the scenario suite's 55-turn Claude chat, out of time at 1800 s, its grounding
    checks still queued). An outage — the service not answering at all — still spends none."""
    sleep = asyncio.sleep

    async def at_once(seconds: float) -> None:  # the cool-downs and backoffs, not waited
        await sleep(0)

    monkeypatch.setattr(module.asyncio, "sleep", at_once)
    writes, tries = Writes(), []

    async def verify() -> None:
        tries.append(1)
        if len(tries) % 2:
            raise MemoryTimeoutError("past its time", code="TIMEOUT", status=504, retryable=True)
        raise circuit_open(30.0)

    await writes.submit("memory.verify", verify)
    async with asyncio.timeout(5):
        await writes.drain()
    assert writes.failed == 1 and len(tries) == 2 * module.WRITE_ATTEMPTS - 1, len(tries)
    await writes.aclose()


async def test_writes_held_by_an_open_circuit_fill_the_queue_and_overflow_as_ever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While the circuit is open the writes wait in the queue, bounded by its size: one more
    is given up as a full queue's always is (lost here, with no spool), and said."""
    monkeypatch.setattr(module, "MAX_PENDING", 2)
    monkeypatch.setattr(module, "WRITERS", 1)
    writes, delivered = Writes(), []
    events, seen = run_events()
    back = asyncio.Event()

    def write(name: str) -> Any:
        async def work() -> None:
            if not back.is_set():
                raise circuit_open(0.0)
            delivered.append(name)

        return work

    for name in ("a", "b", "c"):  # one waiting on the circuit, two queued
        await writes.submit("memory.transcript", write(name))
    await writes.submit("memory.transcript", write("d"), events=events)
    assert writes.failed == 1
    assert seen[0].data["message"] == (
        "memory.transcript failed and is lost: the write queue stayed full"
    )
    back.set()
    await writes.drain()
    assert delivered == ["a", "b", "c"]
    await writes.aclose()


async def test_a_writes_retries_back_off_at_random_from_half_a_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the module's own backoff, not this file's
    ranges: list[tuple[float, float]] = []

    def uniform(low: float, high: float) -> float:
        ranges.append((low, high))
        return 0.0

    monkeypatch.setattr(module.random, "uniform", uniform)
    writes, tries = Writes(), []

    async def flaky() -> None:
        tries.append(1)
        if len(tries) < 3:
            raise DependencyUnavailableError("memory is restarting", retryable=True)

    await writes.submit("memory.transcript", flaky)
    await writes.drain()
    assert len(tries) == 3 and ranges[:2] == [(0, 0.5), (0, 1.0)]  # full jitter, doubled
    await writes.aclose()


async def test_a_shutdown_waits_at_most_10_s_for_the_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the module's own bound, not this file's
    bounds: list[float] = []
    timeout = asyncio.timeout

    def bounded(delay: float | None) -> Any:
        bounds.append(delay or 0.0)
        return timeout(delay)

    monkeypatch.setattr(module.asyncio, "timeout", bounded)
    writes = Writes()

    async def done() -> None:
        return None

    await writes.submit("memory.transcript", done)
    await writes.aclose()
    assert 10.0 in bounds  # the drain bound


async def test_a_failed_write_is_counted_and_reported_to_the_run() -> None:
    writes = Writes()
    events, seen = run_events()
    tries: list[int] = []

    async def boom() -> None:
        tries.append(1)
        raise ConnectionError("memory is down")

    await writes.submit("memory.transcript", boom, events=events)
    await writes.drain()
    assert writes.failed == 1 and len(tries) == 6  # 6 attempts in all
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


async def test_a_refusal_says_what_the_service_refused(caplog: pytest.LogCaptureFixture) -> None:
    """MEM-: four Claude transcripts were refused with a 422 on the comparison of 2026-11-15,
    and the log said only "Request validation failed". The memory service's answer names each
    field it refused and why (its problem's ``details``): the warning event and the log say so,
    so the next refusal is diagnosed from the log alone."""
    from trellis.memory.errors import error_from_problem

    field = {"loc": ["body", "messages"], "msg": "List should have at most 100 items after "
             "validation, not 116", "type": "too_long"}  # fmt: skip
    answer = {"code": "VALIDATION", "detail": "Request validation failed", "retryable": False}
    refusal = error_from_problem(422, answer | {"details": {"errors": [field]}})
    writes = Writes()
    events, seen = run_events()

    async def refused() -> None:
        raise refusal

    with caplog.at_level(logging.WARNING, logger="trellis.writes"):
        await writes.submit("memory.transcript", refused, events=events)
        await writes.drain()
    for said in (seen[0].data["message"], caplog.text):
        assert "Request validation failed" in said
        assert "List should have at most 100 items after validation, not 116" in said
        assert '"loc": ["body", "messages"]' in said
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


def test_a_close_from_another_loop_drains_the_writes_on_their_own() -> None:
    """A harness served from another thread (uvicorn's loop) does its writes there; closed from
    the main loop, its writes are drained and its workers stopped on their own loop."""
    done: list[int] = []
    writes = Writes()
    owner = asyncio.new_event_loop()
    thread = threading.Thread(target=owner.run_forever, daemon=True)
    thread.start()

    async def write() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    async def submit() -> None:
        for _ in range(5):
            await writes.submit("w", write)

    try:
        asyncio.run_coroutine_threadsafe(submit(), owner).result(timeout=5)
        asyncio.run(writes.aclose())
        assert len(done) == 5
        assert writes._workers == []
    finally:
        owner.call_soon_threadsafe(owner.stop)
        thread.join(timeout=5)
        owner.close()


def test_a_close_from_another_loop_that_ends_meanwhile_is_not_cancelled_itself() -> None:
    """The writes' loop ends while a close from another loop waits on it (uvicorn told to exit
    first): its shutdown stops the workers, and the close returns rather than raise the
    cancellation of the loop that went away."""
    writes = Writes()
    started, stop = threading.Event(), threading.Event()

    async def write() -> None:
        started.set()
        await asyncio.sleep(30)

    async def serve() -> None:
        await writes.submit("w", write)
        while not stop.is_set():  # noqa: ASYNC110 - the other thread says when, with an Event
            await asyncio.sleep(0.01)

    thread = threading.Thread(target=asyncio.run, args=(serve(),), daemon=True)
    thread.start()
    assert started.wait(5)

    async def close() -> None:
        closing = asyncio.create_task(writes.aclose())
        await asyncio.sleep(0.05)
        stop.set()  # the serving loop ends: asyncio.run cancels what is left on it
        await closing

    asyncio.run(close())
    thread.join(timeout=5)
    assert writes._workers == []


async def test_a_close_from_another_loop_that_is_cancelled_itself_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "DRAIN_SECONDS", 0.1)
    writes = Writes()
    owner = asyncio.new_event_loop()
    thread = threading.Thread(target=owner.run_forever, daemon=True)
    thread.start()
    started = threading.Event()

    async def write() -> None:
        started.set()
        await asyncio.sleep(30)

    async def submit() -> None:
        await writes.submit("w", write)

    try:
        asyncio.run_coroutine_threadsafe(submit(), owner).result(timeout=5)
        assert started.wait(5)
        closing = asyncio.create_task(writes.aclose())
        await asyncio.sleep(0.05)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
    finally:
        # the workers the cancelled close left: stopped on their loop, as its shutdown would
        asyncio.run_coroutine_threadsafe(Writes._close(writes), owner).result(timeout=15)
        owner.call_soon_threadsafe(owner.stop)
        thread.join(timeout=5)
        owner.close()


def test_a_close_after_its_loop_ended_forgets_the_workers_that_loop_stopped() -> None:
    """The loop the writes ran on has ended (its shutdown stopped the workers, which finished
    the queue first): a close from a new loop has nothing left to wait for."""
    done: list[int] = []
    writes = Writes()

    async def write() -> None:
        done.append(1)

    async def submit() -> None:
        await writes.submit("w", write)

    asyncio.run(submit())
    asyncio.run(writes.aclose())
    assert done == [1]
    assert writes._workers == []


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
    assert seen[0].data["message"] == "second failed and is lost: the write queue stayed full"
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


async def test_a_process_replays_only_its_own_deployments_spooled_writes(tmp_path: Path) -> None:
    """MEM-: the spool was one folder per machine user, so a process replayed every other
    process's undelivered writes — on the comparison of 2026-11-15, 33 transcript writes of a
    stopped run were replayed by another arm, made with another key, and refused (403). Each
    deployment (the memory service and the key its writes are made with) keeps its own: another
    deployment's process sharing the directory leaves them, and the writes a stopped process
    left are replayed once by its deployment's next processes, however many start."""
    from tests.support.memory import FakeMemoryService
    from trellis import Harness, Settings

    service = FakeMemoryService()

    def deployment(key: str) -> Harness:
        settings = Settings(memory_url="http://memory", api_key=key, spool_dir=str(tmp_path))
        return Harness(config=settings, memory=service.client())

    async def answer(input: str, agent: Any) -> str:
        return "ok"

    service.fail.add("messages")  # the transcript cannot be written: it is spooled
    stopped = deployment("key-a")
    await stopped.wrap(answer, id="echo").run("hi", user="u1", thread="t1")
    await stopped.writes.drain()
    assert stopped.writes.spooled == 1  # left in the spool (the process is never closed)
    service.fail.clear()
    async with deployment("key-b") as other:  # another deployment on the same machine
        other.writes.start()
        await other.writes.drain()
    assert service.named("messages") == []
    async with deployment("key-a") as one, deployment("key-a") as two:
        for later in (one, two):
            later.writes.start()
        for later in (one, two):
            await later.writes.drain()
    [replayed] = service.named("messages")
    assert [m["content"] for m in replayed.body["messages"]] == ["hi", "ok"]
    await stopped.aclose()


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


# --------------------------------------------------------------------------- never only a warning


@pytest.mark.parametrize("outcome", ["spooled", "lost"])
async def test_a_failed_write_is_counted_traced_reported_and_logged_as_an_error_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    spans: Any,
    outcome: str,
) -> None:
    spooled = outcome == "spooled"
    counted: list[tuple[str, ...]] = []
    monkeypatch.setattr(module.metrics, "write_failed", lambda label: counted.append((label,)))
    monkeypatch.setattr(
        module.metrics, "write_undelivered", lambda label, out: counted.append((label, out))
    )
    monkeypatch.setattr(module, "_ERRORS_SAID", set())
    writes = Writes(spool=tmp_path if spooled else None)
    events, seen = run_events()
    events.trace_run = "run_parent"  # a sub-agent's run: its spans are in its parent's trace

    async def down() -> None:
        raise ConnectionError("memory is down")

    record = {"op": "record_tool", "scope": {}, "args": {}}
    with caplog.at_level(logging.WARNING, logger="trellis.writes"):
        for _ in range(2):
            await writes.submit("memory.record_tool", down, events=events, record=record)
            await writes.drain()
    await writes.aclose()
    assert counted == [("memory.record_tool",), ("memory.record_tool", outcome)] * 2
    said = [r for r in caplog.records if "memory.record_tool" in r.getMessage()]
    assert [r.levelno for r in said] == [logging.ERROR, logging.WARNING]  # an error once
    assert [e.data["code"] for e in seen] == ["write_failed", "write_failed"]
    assert all("memory is down" in e.data["message"] for e in seen)
    found = [s for s in spans.get_finished_spans() if s.name == "write_failed memory.record_tool"]
    assert len(found) == 2
    span = found[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["trellis.write.outcome"] == outcome
    assert span.context.trace_id == module.telemetry.trace_id_of("run_parent")
    assert span.events[0].name == "write_failed"


async def test_a_failed_write_of_no_run_is_still_counted_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, spans: Any
) -> None:
    monkeypatch.setattr(module, "_ERRORS_SAID", set())
    writes = Writes()

    async def down() -> None:
        raise ConnectionError("memory is down")

    with caplog.at_level(logging.ERROR, logger="trellis.writes"):
        await writes.submit("memory.model_key", down)
        await writes.drain()
    await writes.aclose()
    assert writes.failed == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert spans.get_finished_spans() == ()  # no run to trace it in


async def test_a_write_that_swallows_its_cancellation_does_not_hold_the_close(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A benchmark process sat at ``Harness.aclose`` for fifteen minutes, every case done:
    the close awaited its cancelled workers with no bound, and one never ended. A write whose
    code swallows the cancellation (or waits on what never comes) is given the drain bound
    once more, then left behind: said in the log with where it waits, and counted lost."""
    monkeypatch.setattr(module, "DRAIN_SECONDS", 0.05)
    writes, entered = Writes(), asyncio.Event()

    over = False

    async def deaf() -> None:
        entered.set()
        while not over:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # the cancellation is swallowed: the write goes on waiting

    await writes.submit("memory.deaf", deaf)
    await entered.wait()
    left = list(writes._workers)
    with caplog.at_level(logging.ERROR, logger="trellis.writes"):
        async with asyncio.timeout(2):
            await writes.aclose()
    over = True  # the test's own: the loop's shutdown ends the write
    while not all(worker.done() for worker in left):
        for worker in left:
            worker.cancel()
        await asyncio.sleep(0.01)
    assert writes.failed == 1 and writes._queue is None and not writes._workers
    assert any(
        "memory.deaf" in r.getMessage() and "did not stop" in r.getMessage() for r in caplog.records
    )


async def test_a_worker_that_failed_fails_the_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker ended by an error of the queue's own code (never a write's: those are given
    up) is not passed over in silence: the close raises it."""
    writes = Writes()

    async def broken(item: Any) -> None:
        raise RuntimeError("the worker's own")

    monkeypatch.setattr(writes, "_attempt", broken)

    async def done() -> None:
        return None

    await writes.submit("memory.transcript", done)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="the worker's own"):
        await writes.aclose()
    assert writes._queue is None and not writes._workers
