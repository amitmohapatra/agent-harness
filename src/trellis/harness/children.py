"""What a run's ending means for its children: one cascade for every place a run ends.

A child is a run started for another's — a sub-agent's run inside one of its calls, a helper a
framework spawned, a background task, a live session's background subagent, a Deep Agents
async task. **Every child that has not ended ends with its parent** (:func:`end_children`),
apart from those its caller names as kept (the helpers a run handed on past its attempt and
its conversation's background tasks, when it was not cancelled: ``pipeline._children_kept``)
and, unless someone cancelled it, those it queued for a worker (``agent.start`` inside the
run: they work in their own time):

* **in this process** what is kept at work for it outside any attempt of its caller stops
  first (:func:`stop_kept`), from what this process knows of it — the adapter's keepers (a
  live session's background subagent, a Deep Agents async task: ``Adapter.kept_children``)
  stop even when the run store cannot be read — and each child the store lists stops too: a
  harness background task's attempt, the keepers of the agent whose run it is;
* **in the run store** it ends as its parent did — failed with the parent's error when that
  failed or ran out of time (never retried: its parent's end is its own), else cancelled with
  the parent's reason — unless the store ended it first (``stopping.unless_ended``). A child
  paused for a person waits on nobody once its parent ended, so it ends too: cancelled, with
  the error's message, when it cannot fail from a pause;
* **its own children** end with it, none kept — when it ended here: a child the store ended
  first, or another holder has, is whoever ended or holds it's to end the children of.

Each child ends on its own: one whose ending fails (the run store's error, a keeper's — a
remote server's cancel) is said in the log and the others end all the same; the first such
error is then its caller's (:func:`end_children` raises it once every child was tried, and
:func:`children_ended` only logs it, for a caller whose own ending stands whatever the
children's).
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import AgentError, RunStatus
from trellis.harness import stopping

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.harness")

#: Why a child's keeper stops when its parent's ending names no reason.
PARENT_ENDED: Final = "its parent ended"
#: The statuses of a child that has not ended and works or waits on its parent.
OPEN: Final = (RunStatus.RUNNING, RunStatus.PAUSED)


async def end_children(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
    kept: Collection[str] = (),
    cancelled: bool | None = None,
) -> None:
    """The run ``run_id`` ended ``status``: each of its children but those in ``kept`` ends with
    it (:func:`end_child`) — failed with ``error`` when the run failed or ran out of time, else
    cancelled with ``reason``. A child that ended already is left as it is; so is one queued
    for a worker (``agent.start`` inside the run: it works in its own time, whatever the run
    that queued it answered), unless someone cancelled the run (``cancelled``: by default,
    whether it ended ``CANCELLED``; a run whose caller only went away was not) — a cancel
    ends them all."""
    if cancelled is None:
        cancelled = status is RunStatus.CANCELLED
    failed: Exception | None = None
    stopped: set[str] = set()
    for child in agent.adapter.kept_children(run_id):  # from this process, not the store
        if child not in kept:
            stopped.add(child)
            try:
                await stop_kept(agent, child, reason or PARENT_ENDED)
            except Exception as exc:
                log.warning("run %s not stopped with its parent %s: %s", child, run_id, exc)
                failed = failed or exc
    runs = agent.harness.runs
    for listed in (*OPEN, RunStatus.QUEUED) if cancelled else OPEN:
        async for found in runs.iterate(parent_run_id=run_id, status=listed, tenant=tenant):
            if found.run_id in kept:
                continue
            keeper = agent.harness.agents.get(found.agent_id, agent)
            try:
                await end_child(
                    keeper,
                    found.run_id,
                    tenant,
                    status=status,
                    reason=reason,
                    error=error,
                    current=found.status,
                    stop=found.run_id not in stopped,
                )
            except Exception as exc:
                log.warning("run %s not ended with its parent %s: %s", found.run_id, run_id, exc)
                failed = failed or exc
    if failed is not None:
        raise failed


async def children_ended(agent: Agent, run_id: str, tenant: str, **ending: Any) -> None:
    """:func:`end_children` for a caller whose own ending stands whatever its children's: a
    child that cannot be ended is said in the log, never raised."""
    try:
        await end_children(agent, run_id, tenant, **ending)
    except Exception as exc:
        log.warning("run %s: its open children could not be ended: %s", run_id, exc)


async def end_child(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
    current: RunStatus | None = None,
    stop: bool = True,
) -> None:
    """One child run ends (``current``: its status, when known), what this process keeps at
    work for it stops (:func:`stop_kept`, with ``reason`` or :data:`PARENT_ENDED`) whether or
    not its ending could be written (``stop`` false: its caller stopped it already), and its
    own children end with it (:func:`end_children`,
    none kept). A child that cannot take ``status`` (a paused one cannot fail) is cancelled
    with the error's message. One the store ended first, or another holder has
    (``stopping.unless_ended``), is left as it is there, and so are its children — whoever
    ended or holds it ends them."""
    runs = agent.harness.runs
    fails = error is not None and (current is None or current.can_become(status))
    why = reason or (error.message if error is not None else None)
    failed: Exception | None = None
    ended = False
    try:
        ended = await stopping.unless_ended(
            runs.finish(run_id, status, error=_final(error), tenant=tenant)
            if fails
            else runs.cancel(run_id, reason=why, tenant=tenant)
        )
    except Exception as exc:
        failed = exc
    if stop:
        try:
            await stop_kept(agent, run_id, reason or PARENT_ENDED)
        except Exception as exc:
            failed = failed or exc
    if ended:
        try:
            await end_children(agent, run_id, tenant, status=status, reason=reason, error=error)
        except Exception as exc:
            failed = failed or exc
    if failed is not None:  # what this process keeps for it, and its children, were tried
        raise failed


def _final(error: AgentError | None) -> AgentError | None:
    """``error`` as a child's ending carries it: never retryable (the store would put a
    worker's run back on the queue): its parent ended."""
    if error is None or not error.retryable:
        return error
    return error.model_copy(update={"retryable": False})


async def stop_kept(agent: Agent, run_id: str, reason: str) -> None:
    """What this process keeps at work for run ``run_id`` outside any attempt of its caller
    stops: a harness background task's attempt (``background.Tasks.stop``) and the adapter's
    keepers (its ``child_cancelled``: a live session's background subagent, a Deep Agents async
    task). A run nothing keeps here is no change; a second stop of one is none either. A
    keeper whose stop fails (a remote server's cancel) fails its caller once the background
    task was stopped too."""
    try:
        await agent.adapter.child_cancelled(run_id, reason)
    finally:
        agent.harness.background.stop(run_id, reason)
