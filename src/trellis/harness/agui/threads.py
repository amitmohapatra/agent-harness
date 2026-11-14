"""``GET {path}/threads/{thread_id}/events``: every run of one user's chat thread on one SSE
stream — the runs the chat starts and the runs its schedules fire (``schedules.py``), live and
on reconnect — so a reply that arrives while the chat is open shows at once.

On connect the stream replays the thread's latest :data:`REPLAY_RUNS` runs (after the run and
event its ``Last-Event-ID`` names, ``<run_id>:<number>``), then follows the runs that start in
the thread. One poller per surface serves every open stream: each tick it lists the newest runs
of the agent once per tenant with an open thread (``RunSummary.thread_id`` says whose) and hands
a new run to the streams of its thread; nothing polls while no stream is open.

A run's events come from agent-runs' event log (``RUNS_URL``), numbered by position, so a
reconnect anywhere resumes where it left off. In process (``LocalRuns``) they are the events of
its attempts from when the stream follows it, its reply said in full from its record when it
ends. A scheduled run's events are held until it ends: one that answered ``NO_REPORT`` (its
condition did not hold, or it was reported already) is not delivered; a pause (a question, an
approval) is delivered at once, so the user answers it in the chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Final, cast

from trellis.contracts import RunEvent, RunEventType, RunRecord, RunStatus
from trellis.harness.agui.events import AGUIEvent, AGUIEventType, Outcome, OutcomeType
from trellis.harness.agui.translate import translate
from trellis.harness.background import NOTIFIED
from trellis.harness.bounded import LRU
from trellis.harness.identity import Identity
from trellis.harness.runlog import event_log
from trellis.harness.schedules import NO_REPORT, Fired

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.runs import RunEventEntry

log = logging.getLogger("trellis.agui")

#: Seconds between two looks for new runs in the open threads.
POLL_SECONDS: Final = 1.0
#: The runs of a thread a stream replays on connect, the newest.
REPLAY_RUNS: Final = 20
#: The runs one look lists per tenant, newest first (pages of agent-runs' default).
LOOK_PAGES: Final = 4
#: The run ids already handed to the streams, the most recent.
SEEN: Final = 10_000


class Threads:
    """The open thread streams of one surface, and the one poller that feeds them."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        #: (tenant, thread) -> the queues of its open streams (``None``: a run gone meanwhile)
        self._open: dict[tuple[str, str], list[asyncio.Queue[RunRecord | None]]] = {}
        self._seen: LRU[str, None] = LRU(SEEN)
        #: the poller's task while a stream is open
        self.polling: asyncio.Task[None] | None = None

    async def stream(
        self, tenant: str, user: str, thread: str, last: str | None
    ) -> AsyncGenerator[str]:
        """The SSE chunks of ``user``'s thread ``thread``, after the event ``last`` names."""
        queue: asyncio.Queue[RunRecord | None] = asyncio.Queue()
        key = (tenant, thread)
        self._open.setdefault(key, []).append(queue)  # first: no run falls between
        if self.polling is None:
            self.polling = asyncio.create_task(self._poll(), name="agui:threads")
        try:
            after_run, after = _position(last)
            done: set[str] = set()
            replayed = await self._replayed(tenant, user, thread)
            ids = [r.run_id for r in replayed]
            if after_run in ids:
                replayed = replayed[ids.index(after_run) :]
            for record in replayed:
                done.add(record.run_id)
                skip = after if record.run_id == after_run else -1
                async with contextlib.aclosing(self._run(record, skip)) as chunks:
                    async for chunk in chunks:
                        yield chunk
            while True:
                record = await queue.get()
                if record is None or record.run_id in done or Identity.of(record).user != user:
                    continue
                done.add(record.run_id)
                async with contextlib.aclosing(self._run(record, -1)) as chunks:
                    async for chunk in chunks:
                        yield chunk
        finally:
            self._open[key].remove(queue)
            if not self._open[key]:
                del self._open[key]
            if not self._open and self.polling is not None:
                self.polling.cancel()
                self.polling = None

    async def _replayed(self, tenant: str, user: str, thread: str) -> list[RunRecord]:
        """The thread's latest runs of ``user``, oldest first."""
        runs = self.agent.harness.runs
        found: list[RunRecord] = []
        async for summary in runs.iterate(
            agent_id=self.agent.id, thread_id=thread, tenant=tenant, limit=REPLAY_RUNS, max_pages=1
        ):
            self._seen[summary.run_id] = None
            record = await runs.get(summary.run_id, tenant=tenant)
            if record is not None and Identity.of(record).user == user:
                found.append(record)
        return found[::-1]

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(POLL_SECONDS)
            for tenant in {t for t, _ in self._open}:
                try:
                    await self._look(tenant)
                except Exception:
                    log.warning("thread streams: no look at %s's runs", tenant, exc_info=True)

    async def _look(self, tenant: str) -> None:
        """One listing of the tenant's newest runs of the agent: each new run in an open
        thread goes to that thread's streams."""
        runs = self.agent.harness.runs
        async for summary in runs.iterate(
            agent_id=self.agent.id, tenant=tenant, max_pages=LOOK_PAGES
        ):
            if summary.run_id in self._seen:
                break  # the older ones were looked at
            self._seen[summary.run_id] = None
            streams = self._open.get((tenant, summary.thread_id or ""))
            if not streams:
                continue
            record = await runs.get(summary.run_id, tenant=tenant)
            for queue in streams:
                queue.put_nowait(record)

    async def _run(self, record: RunRecord, after: int) -> AsyncGenerator[str]:
        """One run's events numbered above ``after``; a scheduled run's held until it ends."""
        # a turn of the chat's own for a background task's outcome is delivered as one
        fired = Fired.of(record.metadata) is not None or NOTIFIED in record.metadata
        async with (
            contextlib.aclosing(self._events(record)) as events,
            contextlib.aclosing(_delivered(record, events, after, fired=fired)) as chunks,
        ):
            async for chunk in chunks:
                yield chunk

    async def _events(self, record: RunRecord) -> AsyncGenerator[tuple[int, AGUIEvent]]:
        agent, tenant = self.agent, record.tenant_id
        store = event_log(agent.harness.runs)
        if store is not None:
            logged = store.stream_events(record.run_id, after=0, tenant=tenant)
            async with contextlib.aclosing(
                cast("AsyncGenerator[RunEventEntry]", logged)
            ) as entries:
                async for entry in entries:
                    translated = translate(agent.hooks.event(entry.event))
                    if translated is not None:
                        yield entry.position, translated
            return
        # in process: what its attempts emit from now, its reply said from its record
        number, finished = 0, None
        if not record.final:
            followed = cast("AsyncGenerator[RunEvent]", agent.events(record.run_id, tenant=tenant))
            async with contextlib.aclosing(followed) as live:
                async for event in live:
                    if event.type in _TEXT:  # said in full from the record at the end
                        continue
                    translated = translate(agent.hooks.event(event))
                    if translated is None:
                        continue
                    if _ends(translated):
                        finished = translated
                        continue
                    yield number, translated
                    number += 1
        ended = await agent.harness.runs.get(record.run_id, tenant=tenant)
        replied = _reply(ended, finished) if ended is not None and ended.final else []
        for said in replied:
            yield number, said
            number += 1


async def _delivered(
    record: RunRecord,
    events: AsyncIterator[tuple[int, AGUIEvent]],
    after: int,
    *,
    fired: bool,
) -> AsyncGenerator[str]:
    """The run's events numbered above ``after`` as SSE chunks; a scheduled run's held until
    it ends or pauses, and left out when it ends with nothing to report."""
    held: list[str] = []
    async for number, event in events:
        if number <= after:
            continue
        chunk = _encoded(event, f"{record.run_id}:{number}")
        if not fired:
            yield chunk
            continue
        held.append(chunk)
        if event.type is AGUIEventType.RUN_FINISHED or event.type is AGUIEventType.RUN_ERROR:
            # its end, or a pause the user answers: delivered, but a quiet end
            if not (_ends(event) and event.result == NO_REPORT):
                for kept in held:
                    yield kept
            held = []


_TEXT: Final = frozenset(
    {
        RunEventType.TEXT_MESSAGE_START,
        RunEventType.TEXT_MESSAGE_CONTENT,
        RunEventType.TEXT_MESSAGE_END,
    }
)


def _reply(record: RunRecord, finished: AGUIEvent | None) -> list[AGUIEvent]:
    """A finished run's reply from its record: its text, then its end."""
    ids = {"thread_id": record.thread_id, "run_id": record.run_id}
    said: list[AGUIEvent] = []
    output = record.output
    if output is not None and output != NO_REPORT:
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        message = {"message_id": f"{record.run_id}:reply", **ids}
        said = [
            AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_START, role="assistant", **message),
            AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_CONTENT, delta=text, **message),
            AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_END, **message),
        ]
    if finished is not None:
        return [*said, finished]
    if record.status is RunStatus.CANCELLED:
        outcome = Outcome(type=OutcomeType.CANCELLED)
    elif record.status in (RunStatus.SUCCESS, RunStatus.PARTIAL):
        outcome = Outcome(type=OutcomeType.SUCCESS)
    else:
        why = record.error.message if record.error is not None else str(record.status)
        return [*said, AGUIEvent(type=AGUIEventType.RUN_ERROR, message=why, **ids)]
    return [
        *said,
        AGUIEvent(type=AGUIEventType.RUN_FINISHED, outcome=outcome, result=output, **ids),
    ]


def _ends(event: AGUIEvent) -> bool:
    """Whether ``event`` ends its run (a pause does not)."""
    if event.type is AGUIEventType.RUN_ERROR:
        return True
    return event.type is AGUIEventType.RUN_FINISHED and (
        event.outcome is None or event.outcome.type is not OutcomeType.INTERRUPT
    )


def _encoded(event: AGUIEvent, ident: str) -> str:
    return f"id: {ident}\ndata: {json.dumps(event.wire(), default=str)}\n\n"


def _position(last: str | None) -> tuple[str | None, int]:
    """The run and the event number of a ``Last-Event-ID`` (``<run_id>:<number>``)."""
    if not last or ":" not in last:
        return None, -1
    run_id, _, number = last.rpartition(":")
    with contextlib.suppress(ValueError):
        return run_id, int(number)
    return None, -1
