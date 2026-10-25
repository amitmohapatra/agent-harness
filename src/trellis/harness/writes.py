"""Side effects that must not hold a run: transcript and tool records, outcomes, verdicts.

A write is queued and the run moves on; a few workers drain the queue. A write that fails is
tried again — :data:`WRITE_ATTEMPTS` attempts in all, with full-jitter backoff — when the
failure may pass (the service unavailable, a timeout; not a refusal the same request would get
again). A write that still fails is never raised into the run, and never only a warning:

* counted (``trellis.writes.failed``, and ``trellis.writes.undelivered`` by outcome: kept in the
  spool or lost);
* traced: a ``write_failed`` span in the run's trace, its status an error;
* reported to the run's event listeners as a ``write_failed`` warning event;
* logged: an error, the first time a kind of write fails in this process (each later one a
  warning, so a long outage does not flood the log);
* listed by ``trellis doctor`` while it waits in the spool (its "records failing" check).

A full queue holds the run instead of dropping the write: ``await submit(...)`` waits up to
:data:`SUBMIT_WAIT_SECONDS` for room, and only then gives the write up.

**What is guaranteed.** Every write is attempted at least once — in this process, or, with a
spool directory (``TRELLIS_SPOOL_DIR``), after the next start. A write given up on (its attempts
spent on failures that may pass, a queue that stayed full, the process stopping with it still
queued) is appended to ``<spool>/trellis-writes.jsonl`` when it can be described as data (a
transcript, a tool record, an outcome, a decision, catalog entries: ``record=``), and the next
harness that starts writing with the same directory replays the file and removes it. Writes
are idempotent, so a replayed write that had in fact landed is stored once. Without a spool
directory — or for a write that is not data (a grounding check, the model key) — it is lost:
logged, counted (``trellis.writes.undelivered``, ``outcome=lost``) and reported.

Draining is automatic: when the event loop shuts down (``asyncio.run`` returning, a server
stopping) it cancels the workers, and a cancelled worker finishes the queue first, bounded by
:data:`DRAIN_SECONDS`; what is left then is spooled or counted lost. ``await harness.aclose()``
does the same explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
from collections.abc import Awaitable, Callable
from contextvars import Context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from trellis.harness import telemetry
from trellis.harness.events import RunEvents
from trellis.harness.telemetry import metrics

log = logging.getLogger("trellis.writes")

#: Workers draining the queue concurrently.
WRITERS: Final = 4
#: Pending writes held at most; past it a write waits for room (:data:`SUBMIT_WAIT_SECONDS`).
MAX_PENDING: Final = 10_000
#: How long a write waits for room in a full queue before it is given up (spooled or lost).
SUBMIT_WAIT_SECONDS: Final = 5.0
#: How long a shutdown waits for the queue to empty.
DRAIN_SECONDS: Final = 10.0
#: Attempts per write, and the backoff ceiling of the first retry (doubled after each).
WRITE_ATTEMPTS: Final = 3
WRITE_BACKOFF_SECONDS: Final = 0.5
#: The spool file, in the spool directory.
SPOOL_FILE: Final = "trellis-writes.jsonl"
#: The most the spool file may hold, in bytes: past it a write given up is lost, not kept (a
#: long outage must not fill the disk), said once until the spool is replayed.
MAX_SPOOL_BYTES: Final = 64 * 1024 * 1024

Work = Callable[[], Awaitable[object]]
#: Turns a spooled record back into its write (``None``: nothing replays it here).
Replay = Callable[[dict[str, Any]], Work | None]


@dataclass(frozen=True, slots=True)
class _Item:
    label: str
    work: Work
    events: RunEvents | None
    #: the write as data, for the spool: what replays it after a restart
    record: dict[str, Any] | None = None


class Writes:
    """The queue and its workers, bound to the event loop that first used them. ``spool`` is
    the directory undelivered writes are kept in; ``replay`` turns a kept record back into its
    write."""

    def __init__(
        self, *, spool: str | os.PathLike[str] | None = None, replay: Replay | None = None
    ) -> None:
        self._queue: asyncio.Queue[_Item] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._spool = Path(spool) if spool is not None else None
        self._replay = replay
        self._hurry = False
        #: writes given up and lost (not spooled)
        self.failed = 0
        #: writes given up and kept in the spool for the next start
        self.spooled = 0
        #: the spool reached :data:`MAX_SPOOL_BYTES` (said once)
        self._full = False

    def start(self) -> None:
        """Bind to the running loop now (and replay the spool), rather than at the first
        write."""
        self._ensure()

    async def submit(
        self,
        label: str,
        work: Work,
        *,
        events: RunEvents | None = None,
        record: dict[str, Any] | None = None,
    ) -> None:
        """Queue ``work``; with the queue full, wait for room (at most
        :data:`SUBMIT_WAIT_SECONDS`). ``record`` is the write as data, which the spool keeps
        when it cannot be delivered."""
        item = _Item(label, work, events, record)
        queue = self._ensure()
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        try:
            async with asyncio.timeout(SUBMIT_WAIT_SECONDS):
                await queue.put(item)
        except TimeoutError:
            self._give_up(item, "the write queue stayed full")

    async def drain(self) -> None:
        """Wait until every queued write has been attempted."""
        if self._queue is not None and self._loop is asyncio.get_running_loop():
            await self._queue.join()

    async def aclose(self) -> None:
        """Drain (at most :data:`DRAIN_SECONDS`), then stop the workers: what is still queued
        is spooled or counted lost."""
        try:
            async with asyncio.timeout(DRAIN_SECONDS):
                await self.drain()
        except TimeoutError:
            self._hurry = True  # the bound is spent: the workers give the rest up at once
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()
        self._queue = None
        self._hurry = False

    # ------------------------------------------------------------------ internals
    def _ensure(self) -> asyncio.Queue[_Item]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._loop = loop
            self._queue = asyncio.Queue(MAX_PENDING)
            # in a context of their own: the workers outlive the run that happened to start
            # them, and belong to none (``trellis.current()`` is ``None`` there)
            self._workers = [
                loop.create_task(
                    self._work(self._queue), name=f"trellis-writes-{i}", context=Context()
                )
                for i in range(WRITERS)
            ]
            self._replayed(self._queue)
        return self._queue

    async def _work(self, queue: asyncio.Queue[_Item]) -> None:
        item: _Item | None = None
        try:
            while True:
                item = await queue.get()
                await self._attempt(item)
                queue.task_done()
                item = None
        except asyncio.CancelledError:
            # The loop is going away: finish what was promised (the write that was cut off
            # included, writes are idempotent), within bounds; keep or count the rest.
            await self._finish(queue, item)
            raise

    async def _finish(self, queue: asyncio.Queue[_Item], cut: _Item | None) -> None:
        pending = [cut] if cut is not None else []
        current: _Item | None = None
        try:
            async with asyncio.timeout(0 if self._hurry else DRAIN_SECONDS):
                while pending or not queue.empty():
                    current = pending.pop() if pending else queue.get_nowait()
                    try:
                        await self._attempt(current)
                    finally:
                        queue.task_done()
                    current = None
        except (TimeoutError, asyncio.CancelledError):
            left = [current] if current is not None else []
            while not queue.empty():
                left.append(queue.get_nowait())
                queue.task_done()
            for undelivered in left:
                self._give_up(undelivered, "still undelivered when the process stopped")

    async def _attempt(self, item: _Item) -> None:
        def wait(state: RetryCallState) -> float:  # full jitter, not a secret
            return random.uniform(0, WRITE_BACKOFF_SECONDS * 2 ** (state.attempt_number - 1))

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(WRITE_ATTEMPTS),
                retry=retry_if_exception(_passing),
                wait=wait,
                sleep=asyncio.sleep,
                reraise=True,
            ):
                with attempt:
                    await item.work()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._give_up(item, f"{type(exc).__name__}: {exc}", keep=_passing(exc))

    def _give_up(self, item: _Item, reason: str, *, keep: bool = True) -> None:
        """The write will not be delivered by this process: kept in the spool when it can be
        (``keep``: its failure may pass, so the same write may land later), else lost —
        either way counted, traced, reported to the run and logged (see the module)."""
        if keep and self._kept(item):
            self.spooled += 1
            outcome, said = SPOOLED, f"{item.label} failed and is kept for the next start"
        else:
            self.failed += 1
            outcome, said = LOST, f"{item.label} failed and is lost"
        metrics.write_failed(item.label)
        metrics.write_undelivered(item.label, outcome)
        _logged(item.label, f"trellis write: {said}: {reason}")
        events = item.events
        if events is not None:
            events.warning("write_failed", f"{said}: {reason}")
            run = events.context.agent_run_id
            telemetry.failure(
                events.trace_run or run,
                run,
                f"write_failed {item.label}",
                reason,
                {"trellis.write": item.label, "trellis.write.outcome": outcome},
            )

    def _kept(self, item: _Item) -> bool:
        if self._spool is None or item.record is None:
            return False
        line = json.dumps({"label": item.label, **item.record}, default=str)
        path = self._spool / SPOOL_FILE
        try:
            self._spool.mkdir(parents=True, exist_ok=True)
            held = path.stat().st_size if path.exists() else 0
            if held + len(line) + 1 > MAX_SPOOL_BYTES:
                if not self._full:
                    log.error(
                        "the write spool %s is full (%d bytes): writes given up are lost "
                        "until it is replayed at the next start",
                        path,
                        held,
                    )
                    self._full = True
                return False
            with path.open("a", encoding="utf-8") as spool:
                spool.write(line + "\n")
        except OSError as exc:
            log.error("the write spool %s cannot be written: %s", self._spool, exc)
            return False
        return True

    def _replayed(self, queue: asyncio.Queue[_Item]) -> None:
        """Queue what an earlier process kept. The file is claimed first (renamed), so two
        processes sharing the directory replay it once."""
        if self._spool is None or self._replay is None:
            return
        claimed = self._spool / f"{SPOOL_FILE}.{os.getpid()}.replaying"
        try:
            (self._spool / SPOOL_FILE).replace(claimed)
        except OSError:  # nothing kept (or another process took it)
            return
        lines = claimed.read_text(encoding="utf-8").splitlines()
        claimed.unlink()
        replayed = 0
        for line in lines:
            try:
                record = json.loads(line)
                work = self._replay(record)
            except (ValueError, KeyError, TypeError) as exc:
                log.error("a spooled write cannot be read, dropped: %s: %.200s", exc, line)
                continue
            if work is None:
                log.error("nothing here replays the spooled write %.200s", line)
                continue
            item = _Item(str(record.get("label", "replay")), work, None, record)
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                self._kept(item)  # back to the spool, for the start after this one
                continue
            replayed += 1
        log.info("replaying %d spooled write(s) from %s", replayed, self._spool)


#: What became of a write given up (``trellis.writes.undelivered``'s ``outcome``).
SPOOLED: Final = "spooled"
LOST: Final = "lost"
#: The kinds of writes whose first failure in this process was logged as an error.
_ERRORS_SAID: set[str] = set()


def _logged(label: str, message: str) -> None:
    """An error the first time ``label`` fails in this process, a warning after (bounded: one
    entry per kind of write)."""
    if label in _ERRORS_SAID:
        log.warning("%s", message)
        return
    _ERRORS_SAID.add(label)
    log.error("%s (said once as an error; later failures of %s are warnings)", message, label)


def _passing(exc: BaseException) -> bool:
    """Whether a write's failure may pass (tried again, spooled when it does not): its own
    ``retryable``, else yes."""
    return isinstance(exc, Exception) and bool(getattr(exc, "retryable", True))
