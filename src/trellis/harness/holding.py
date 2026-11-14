"""Holding the runs this process executes in the run store, so a process that dies with them
leaves runs the store ends instead of runs ``RUNNING`` for ever.

A run a worker claimed from the queue has a lease its worker keeps with heartbeats
(``trellis.runs.Worker``): when the worker dies, agent-runs queues the run again. Every other
run the harness executes runs in this process — a turn of ``agent.run`` / ``stream`` /
``serve_chat``, a resumed one, a sub-agent's child run (``subagents``), a helper a framework
spawned (``bridges.helpers``), a Claude subagent working in the background
(``bridges.claude_live``) — and nothing else could run it. Such a run is *held*: the same
heartbeat, naming this harness (:attr:`Holds.name`), every third of :data:`HOLD_SECONDS`,
while the run executes here. When this process dies (a crash, a ``SIGKILL``, a restart) the
holds stop, and once the last lapses agent-runs' ticker ends the run ``ERROR``
(``holder_lost``); an in-process store (``LocalRuns``) does the same at its next sweep.

A run held by several parts at once (a background subagent's attempt and the session that
keeps it) is one hold, kept while any of them holds it (:meth:`Holds.hold`,
:meth:`Holds.let_go`). A run that waits for a person (``PAUSED``) has no hold: the store lets go
of it with the pause, refuses one while it waits, and the attempt that continues it after the
answer — on this server or another — takes it again. A helper left open while its run waits for
a person is let go of in the store (:meth:`Holds.release`), not left to lapse: it is not dead,
only waiting with its run.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import secrets
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from trellis.runs import LeaseLostError, RunsError

log = logging.getLogger("trellis.harness")

#: How long one heartbeat holds a run (renewed every third of it): a process killed while it
#: executes runs leaves them ended this long after, within one tick of agent-runs' ticker.
#: agent-runs' default lease: long enough that a slow call or two never lets a hold lapse
#: while the process lives.
HOLD_SECONDS: Final = 60


def name_of() -> str:
    """A name for one harness's holds: its host and process, and a token of its own (two
    harnesses in one process, or a process id the host gave again, are not the same holder)."""
    return f"trellis@{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(4)}"[:200]


class Holding(Protocol):
    """What holding needs of a run store (``RunStore``'s ``heartbeat`` and ``release``)."""

    async def heartbeat(
        self, run_id: str, worker_id: str, *, lease_seconds: int = ..., tenant: str | None = ...
    ) -> Any: ...

    async def release(self, run_id: str, worker_id: str, *, tenant: str | None = ...) -> Any: ...


@dataclass
class _Held:
    #: its heartbeats
    task: asyncio.Task[None] = field(repr=False)
    #: how many parts hold the run now
    count: int = 1


class Holds:
    """The runs this harness holds in its run store, each renewed by a task of its own while
    anything holds it. ``runs`` is read at each heartbeat (the harness's store)."""

    def __init__(self, runs: Callable[[], Holding], name: str | None = None) -> None:
        self._runs = runs
        self.name = name or name_of()
        self._held: dict[str, _Held] = {}

    def holds(self, run_id: str) -> bool:
        """Whether this harness holds ``run_id`` now."""
        return run_id in self._held

    def hold(self, run_id: str, tenant: str) -> None:
        """Hold ``run_id`` until as many :meth:`let_go` as holds: its first hold starts its
        heartbeats (the first at once)."""
        held = self._held.get(run_id)
        if held is not None:
            held.count += 1
            return
        beating = self._beating(run_id, tenant)
        task = asyncio.get_running_loop().create_task(beating, context=contextvars.Context())
        self._held[run_id] = _Held(task)

    def let_go(self, run_id: str) -> None:
        """One hold of ``run_id`` less: with none left its heartbeats stop (what the store
        holds lapses unless the run ended or paused meanwhile, as it does when a hold ends)."""
        held = self._held.get(run_id)
        if held is None:
            return
        held.count -= 1
        if held.count <= 0:
            del self._held[run_id]
            held.task.cancel()

    @contextlib.asynccontextmanager
    async def holding(self, run_id: str, tenant: str) -> AsyncIterator[None]:
        """``run_id`` held while the block runs."""
        self.hold(run_id, tenant)
        try:
            yield
        finally:
            self.let_go(run_id)

    async def release(self, run_id: str, tenant: str) -> None:
        """Let go of ``run_id`` here and in the store, which keeps it ``RUNNING``, unheld (a
        helper whose run waits for a person): every hold of it ends."""
        held = self._held.pop(run_id, None)
        if held is not None:
            held.task.cancel()
        with contextlib.suppress(LeaseLostError):  # it ended or paused meanwhile
            try:
                await self._runs().release(run_id, self.name, tenant=tenant)
            except RunsError as exc:  # the hold lapses instead
                log.warning("run %s not let go of: %s", run_id, exc)

    def aclose(self) -> None:
        """Every hold ends here (the harness is closing; what it ran ended or was left)."""
        for held in self._held.values():
            held.task.cancel()
        self._held.clear()

    async def _beating(self, run_id: str, tenant: str) -> None:
        """Heartbeats holding ``run_id``, every third of :data:`HOLD_SECONDS`. A refused one
        (the run waits for a person, or ended) is tried again at the next; a store that cannot
        be reached too, said in the log."""
        while True:
            try:
                await self._runs().heartbeat(
                    run_id, self.name, lease_seconds=HOLD_SECONDS, tenant=tenant
                )
            except LeaseLostError:
                pass
            except RunsError as exc:
                log.warning("run %s not held: %s", run_id, exc)
            await asyncio.sleep(HOLD_SECONDS / 3)
