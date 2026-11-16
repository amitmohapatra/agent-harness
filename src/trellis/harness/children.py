"""What a run's ending means for its children: one cascade for every place a run ends.

A child is a run started for another's — a sub-agent's run inside one of its calls, a helper a
framework spawned, a background task, a live session's background subagent, a Deep Agents
async task. **Every child that has not ended ends with its parent** (:func:`end_children`),
apart from those its caller names as kept (the helpers a run handed on past its attempt and
its conversation's background tasks, when it was not cancelled: ``pipeline._children_kept``)
and, unless it was cancelled, those it queued for a worker (``agent.start`` inside the run):

* **in the run store** it ends as its parent did — failed with the parent's error when that
  failed or ran out of time, else cancelled with the parent's reason — unless the store ended
  it first (``stopping.unless_ended``); a child paused for a person waits on nobody once its
  parent ended, so it ends too;
* **in this process** what is kept at work for it outside any attempt of its caller stops
  (:func:`stop_kept`): a harness background task's attempt, and the adapter's keepers — a live
  session's background subagent, a Deep Agents async task;
* **its own children** end with it, none kept.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Final

from trellis.contracts import AgentError, RunStatus
from trellis.harness import stopping

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

#: Why a child's keeper stops when its parent's ending names no reason.
PARENT_ENDED: Final = "its parent ended"


async def end_children(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
    kept: Collection[str] = (),
) -> None:
    """The run ``run_id`` ended ``status``: each of its children but those in ``kept`` ends with
    it (:func:`end_child`) — failed with ``error`` when the run failed or ran out of time, else
    cancelled with ``reason``. A child that ended already is left as it is; so is one queued
    for a worker (``agent.start`` inside the run: it works in its own time, whatever the run
    that queued it answered), unless the run was cancelled — a cancel ends them all."""
    runs = agent.harness.runs
    for_a_worker = () if status is RunStatus.CANCELLED else (RunStatus.QUEUED,)
    async for child in runs.iterate(parent_run_id=run_id, tenant=tenant):
        if child.status.final or child.status in for_a_worker or child.run_id in kept:
            continue
        await end_child(agent, child.run_id, tenant, status=status, reason=reason, error=error)


async def end_child(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
) -> None:
    """One child run ends — unless the store ended it first (``stopping.unless_ended``) — what
    this process keeps at work for it stops (:func:`stop_kept`, with ``reason`` or
    :data:`PARENT_ENDED`), and its own children end with it (:func:`end_children`, none
    kept)."""
    runs = agent.harness.runs
    await stopping.unless_ended(
        runs.finish(run_id, status, error=error, tenant=tenant)
        if error is not None
        else runs.cancel(run_id, reason=reason, tenant=tenant)
    )
    await stop_kept(agent, run_id, reason or PARENT_ENDED)
    await end_children(agent, run_id, tenant, status=status, reason=reason, error=error)


async def stop_kept(agent: Agent, run_id: str, reason: str) -> None:
    """What this process keeps at work for run ``run_id`` outside any attempt of its caller
    stops: a harness background task's attempt (``background.Tasks.stop``) and the adapter's
    keepers (its ``child_cancelled``: a live session's background subagent, a Deep Agents async
    task). A run nothing keeps here is no change."""
    await agent.adapter.child_cancelled(run_id, reason)
    agent.harness.background.stop(run_id, reason)
