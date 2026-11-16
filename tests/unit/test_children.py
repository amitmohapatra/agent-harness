"""``children``: the one cascade that ends a run's children when the run ends — in the run
store, and what this process keeps at work for them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from trellis.contracts import AgentError, Interrupt, RunRecord, RunStart, RunStatus
from trellis.harness import children
from trellis.harness.adapters.base import Keeps
from trellis.harness.bridges.live import LiveSessions
from trellis.harness.runs import LocalRuns

TENANT = "t"


class Keeping(Keeps):
    """An adapter whose framework keeps work going for a child: what it was told to stop."""

    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str | None]] = []

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        self.cancelled.append((run_id, reason))


def agent_of(runs: LocalRuns, adapter: Keeps | None = None) -> tuple[Any, list[tuple[str, str]]]:
    """What the cascade reads of an agent — its run store, its harness's background tasks, its
    adapter (one that keeps the default: nothing going for a child) — and the background
    stops it made."""
    stopped: list[tuple[str, str]] = []
    tasks = SimpleNamespace(stop=lambda run_id, reason: stopped.append((run_id, reason)))
    harness = SimpleNamespace(runs=runs, background=tasks)
    return SimpleNamespace(harness=harness, adapter=adapter or Keeps()), stopped


async def started(
    runs: LocalRuns, run_id: str, parent: str | None = None, *, queue: bool = False
) -> None:
    start = RunStart(
        run_id=run_id, tenant_id=TENANT, agent_id="chat", user_id="u", parent_run_id=parent
    )
    await runs.start(start, queue=queue)


async def found(runs: LocalRuns, run_id: str) -> RunRecord:
    record = await runs.get(run_id, tenant=TENANT)
    assert record is not None
    return record


async def test_a_cancelled_runs_children_are_cancelled_but_the_kept_and_the_ended() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    for child in ("working", "paused", "done", "kept"):
        await started(runs, child, "parent")
    await runs.pause(Interrupt(tenant_id=TENANT, run_id="paused", question="Which?"))
    await runs.finish("done", RunStatus.SUCCESS, output="found", tenant=TENANT)
    adapter = Keeping()
    agent, stopped = agent_of(runs, adapter)
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.CANCELLED, reason="stopped", kept=["kept"]
    )
    statuses = {r: (await found(runs, r)).status for r in ("working", "paused", "done", "kept")}
    assert statuses == {
        "working": RunStatus.CANCELLED,
        "paused": RunStatus.CANCELLED,  # it waits on nobody once its parent ended
        "done": RunStatus.SUCCESS,
        "kept": RunStatus.RUNNING,
    }
    # what this process keeps for each child that ended stops: the adapter's, a background task
    assert sorted(stopped) == [("paused", "stopped"), ("working", "stopped")]
    assert sorted(adapter.cancelled) == sorted(stopped)


async def test_a_failed_runs_children_end_as_it_did_with_its_error() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    error = AgentError(code="RUN_TIMEOUT", message="out of time")
    agent, stopped = agent_of(runs)  # an adapter that keeps the default ``child_cancelled``
    await children.end_children(agent, "parent", TENANT, status=RunStatus.TIMEOUT, error=error)
    ended = await found(runs, "child")
    assert ended.status is RunStatus.TIMEOUT and ended.error == error
    assert stopped == [("child", children.PARENT_ENDED)]  # the ending named no reason


async def test_a_child_queued_for_a_worker_ends_only_with_a_cancelled_run() -> None:
    """``agent.start`` inside a run queues a child that works on a worker in its own time: the
    run that queued it answering, failing or running out of time is not its end; a cancel of
    that run is."""
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "queued", "parent", queue=True)
    agent, stopped = agent_of(runs)
    error = AgentError(code="RUN_TIMEOUT", message="out of time")
    await children.end_children(agent, "parent", TENANT, status=RunStatus.SUCCESS)
    await children.end_children(agent, "parent", TENANT, status=RunStatus.TIMEOUT, error=error)
    assert (await found(runs, "queued")).status is RunStatus.QUEUED and stopped == []
    await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert (await found(runs, "queued")).status is RunStatus.CANCELLED


async def test_a_childs_own_children_end_with_it_none_kept() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    await started(runs, "grandchild", "child")
    agent, stopped = agent_of(runs)
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.CANCELLED, kept=["grandchild"]
    )
    assert (await found(runs, "child")).status is RunStatus.CANCELLED
    assert (await found(runs, "grandchild")).status is RunStatus.CANCELLED  # kept by nobody
    assert stopped == [("child", children.PARENT_ENDED), ("grandchild", children.PARENT_ENDED)]


async def test_a_child_the_store_ended_first_is_left_as_it_ended_and_its_keeper_stops() -> None:
    runs = LocalRuns()
    await started(runs, "child")
    await runs.finish("child", RunStatus.SUCCESS, output="found", tenant=TENANT)
    agent, stopped = agent_of(runs)
    await children.end_child(agent, "child", TENANT, status=RunStatus.CANCELLED, reason="late")
    assert (await found(runs, "child")).status is RunStatus.SUCCESS
    assert stopped == [("child", "late")]


def test_a_live_sessions_background_run_is_cancelled_and_any_other_run_is_no_change() -> None:
    """The adapters that keep live sessions (Claude, ACP) stop a child through
    ``LiveSessions.cancel``: the background subagent's run a session attends is cancelled
    with the reason, a run no session attends is nobody's here."""
    sessions = LiveSessions()
    cancelled: list[str] = []
    kept: Any = SimpleNamespace(cancel=cancelled.append)
    sessions.attended["child"] = kept
    sessions.cancel("child", "the turn was cancelled")
    sessions.cancel("another", "the turn was cancelled")
    assert cancelled == ["the turn was cancelled"]
