"""What stopping a run's work in this process means: one owner for every place that stops an
attempt, and for deciding that the run store ended a run while this process worked on it.

* **Stopping an attempt** (:func:`stop_attempt`) says why on its runtime and cancels the task
  it runs in; the pipeline reads the reason and ends the run ``CANCELLED`` with it. Every stop
  of a run's work goes through ``Agent.stop``, which calls it: a person's cancel, a newer
  message's interrupt, a refused hold, a background subagent's or task's stop.
* **Ended elsewhere** (:func:`ended_elsewhere`) is what a final record read back from the run
  store means for the process still working on the run: it stops, and writes nothing.
* **A refused ending write** (:func:`unless_ended`) is the same knowledge reached by writing:
  the store ended the run first, or another holder has it, so the write is nobody's to make.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import RunRecord, RunStatus
from trellis.runs import ConflictError, LeaseLostError

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The reason a run cancelled from another process ends with here (the store keeps the one
#: it was cancelled with).
CANCELLED_ELSEWHERE: Final = "cancelled from another process"
#: Why a run the run store ended meanwhile (its hold lapsed: ``holder_lost``) stops here.
ENDED_ELSEWHERE: Final = "ended in the run store while it worked here"


def stop_attempt(runtime: Runtime, reason: str) -> asyncio.Task[Any]:
    """Stop the attempt ``runtime`` runs, for ``reason``: the first reason given stands and its
    task is cancelled once; later stops change nothing. The task, to wait on."""
    assert runtime.running_in is not None
    if runtime.cancelled is None:
        runtime.cancelled = reason
        runtime.running_in.cancel()
    return runtime.running_in


def ended_elsewhere(record: RunRecord | None) -> str | None:
    """Why what this process does for a run stops because the run store ended it
    (:data:`CANCELLED_ELSEWHERE`, :data:`ENDED_ELSEWHERE`); ``None`` while the store has not
    ended it, or has no such run."""
    if record is None or not record.status.final:
        return None
    return CANCELLED_ELSEWHERE if record.status is RunStatus.CANCELLED else ENDED_ELSEWHERE


async def unless_ended(write: Awaitable[object]) -> bool:
    """Make an ending or cancel write; ``False`` when the store had ended the run first, or
    another holder has it (``ConflictError``, ``LeaseLostError``) — the write is then nobody's
    to make."""
    try:
        await write
    except (ConflictError, LeaseLostError):
        return False
    return True
