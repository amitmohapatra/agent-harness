"""``children``: the one cascade that ends a run's children when the run ends — in the run
store, and what this process keeps at work for them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from trellis.contracts import AgentError, Interrupt, RunRecord, RunStart, RunStatus
from trellis.harness import children
from trellis.harness.adapters.base import Keeps
from trellis.harness.bridges.live import Kept, LiveSessions
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
    harness = SimpleNamespace(runs=runs, background=tasks, agents={})
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
    """Whoever ended it there ends its children: a second cascade over a child that ended
    (``Agent.cancel`` after the attempt's own) cancels nothing of it again."""
    runs = LocalRuns()
    await started(runs, "child")
    await started(runs, "grandchild", "child")
    await runs.finish("child", RunStatus.SUCCESS, output="found", tenant=TENANT)
    agent, stopped = agent_of(runs)
    await children.end_child(agent, "child", TENANT, status=RunStatus.CANCELLED, reason="late")
    assert (await found(runs, "child")).status is RunStatus.SUCCESS
    assert (await found(runs, "grandchild")).status is RunStatus.RUNNING
    assert stopped == [("child", "late")]


class Refusing(Keeps):
    """An adapter whose framework fails to stop what it keeps for one child (a remote
    server's cancel that raises)."""

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        if run_id == "first":
            raise ConnectionError("the task's server is down")


async def test_a_child_whose_keeper_fails_to_stop_leaves_no_other_child_working() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    for child in ("first", "second"):
        await started(runs, child, "parent")
    await started(runs, "grandchild", "first")
    agent, stopped = agent_of(runs, Refusing())
    with pytest.raises(ConnectionError, match="server is down"):  # its caller's, at the end
        await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    for run_id in ("first", "second", "grandchild"):
        assert (await found(runs, run_id)).status is RunStatus.CANCELLED
    # the background task of the child whose keeper failed was stopped all the same
    assert sorted(stopped) == [
        (r, children.PARENT_ENDED) for r in ("first", "grandchild", "second")
    ]


async def test_runs_that_name_each_other_as_parent_end_once_each() -> None:
    """A loop in the parent links (bad data) ends: a run the cascade ended is final, and the
    cascade passes over a final run."""
    runs = LocalRuns()
    await started(runs, "a", "b")
    await started(runs, "b", "a")
    agent, stopped = agent_of(runs)
    await children.end_children(agent, "a", TENANT, status=RunStatus.CANCELLED)
    assert (await found(runs, "a")).status is RunStatus.CANCELLED
    assert (await found(runs, "b")).status is RunStatus.CANCELLED
    assert sorted(stopped) == [("a", children.PARENT_ENDED), ("b", children.PARENT_ENDED)]


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


async def test_a_paused_child_of_a_failed_run_is_cancelled_and_its_children_end() -> None:
    """A paused run cannot fail: a child waiting for a person when its parent failed is
    cancelled with the parent's error, and its own children end with it."""
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    await started(runs, "grandchild", "child")
    await runs.pause(Interrupt(tenant_id=TENANT, run_id="child", question="Approve?"))
    agent, _ = agent_of(runs)
    error = AgentError(code="BOOM", message="the parent failed")
    await children.end_children(agent, "parent", TENANT, status=RunStatus.ERROR, error=error)
    assert (await found(runs, "child")).status is RunStatus.CANCELLED
    assert (await found(runs, "grandchild")).status is RunStatus.ERROR


class Failing(LocalRuns):
    """A run store whose ending writes fail (it is unreachable), and that says what it was
    asked to end with."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False
        self.finished: list[AgentError | None] = []

    async def cancel(self, run_id: str, **given: Any) -> RunRecord:
        if self.down:
            raise ConnectionError("the run store is down")
        return await super().cancel(run_id, **given)

    async def finish(self, run_id: str, status: RunStatus, **given: Any) -> RunRecord:
        self.finished.append(given.get("error"))
        return await super().finish(run_id, status, **given)


async def test_a_child_whose_ending_cannot_be_written_still_stops_here() -> None:
    runs = Failing()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    runs.down = True
    agent, stopped = agent_of(runs)
    with pytest.raises(ConnectionError, match="store is down"):
        await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert stopped == [("child", children.PARENT_ENDED)]


async def test_a_child_ends_with_its_parents_error_never_to_be_retried() -> None:
    """A worker's run that fails with a retryable error goes back on the queue: a child's
    failure is its parent's end, so it carries the error as not retryable."""
    runs = Failing()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    agent, _ = agent_of(runs)
    error = AgentError(code="RATE", message="rate limited", retryable=True)
    await children.end_children(agent, "parent", TENANT, status=RunStatus.ERROR, error=error)
    [carried] = runs.finished
    assert carried is not None and not carried.retryable and carried.message == "rate limited"


class Unlisted(LocalRuns):
    """A run store that cannot list a run's children (it is unreachable)."""

    def iterate(self, **given: Any) -> Any:
        raise ConnectionError("the run store is down")


class Holding(Keeping):
    """An adapter whose framework keeps a child of ``parent`` at work in this process."""

    def kept_children(self, run_id: str) -> list[str]:
        return ["kept-here"] if run_id == "parent" else []


async def test_what_this_process_keeps_for_a_run_stops_when_the_store_cannot_list() -> None:
    adapter = Holding()
    agent, stopped = agent_of(Unlisted(), adapter)
    with pytest.raises(ConnectionError, match="store is down"):
        await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert adapter.cancelled == [("kept-here", children.PARENT_ENDED)]
    assert stopped == [("kept-here", children.PARENT_ENDED)]


async def test_a_childs_keepers_are_its_own_agents() -> None:
    """A sub-agent's run's own children are kept by the sub-agent's adapter (its live
    session), not by the adapter of the run that called it."""
    runs = LocalRuns()
    await started(runs, "parent")
    start = RunStart(run_id="child", tenant_id=TENANT, agent_id="sub", user_id="u")
    await runs.start(start.model_copy(update={"parent_run_id": "parent"}))
    agent, _ = agent_of(runs, Keeping())
    sub, _ = agent_of(runs, Keeping())
    agent.harness.agents["sub"] = sub
    await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert sub.adapter.cancelled == [("child", children.PARENT_ENDED)]
    assert agent.adapter.cancelled == []


async def test_a_run_whose_caller_went_away_leaves_what_it_queued_for_a_worker() -> None:
    """Only a cancel ends the children a run queued for a worker: a run that ends cancelled
    because its caller went away (``cancelled=False``) leaves them to work."""
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "queued", "parent", queue=True)
    agent, _ = agent_of(runs)
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.CANCELLED, cancelled=False
    )
    assert (await found(runs, "queued")).status is RunStatus.QUEUED


@pytest.mark.parametrize("paused", [False, True])
async def test_an_answered_runs_open_children_are_cancelled(paused: bool) -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    if paused:
        await runs.pause(Interrupt(tenant_id=TENANT, run_id="child", question="Which?"))
    agent, _ = agent_of(runs)
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.SUCCESS, reason="answered first"
    )
    record = await found(runs, "child")
    assert record.status is RunStatus.CANCELLED


async def test_a_failed_runs_queued_and_kept_children_are_left_as_they_are() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "queued", "parent", queue=True)
    await started(runs, "kept", "parent")
    agent, stopped = agent_of(runs)
    error = AgentError(code="BOOM", message="the parent failed")
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.ERROR, error=error, kept=["kept"]
    )
    assert (await found(runs, "queued")).status is RunStatus.QUEUED
    assert (await found(runs, "kept")).status is RunStatus.RUNNING
    assert stopped == []


async def test_a_caller_whose_ending_stands_logs_a_child_that_cannot_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs = Failing()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    runs.down = True
    agent, _ = agent_of(runs)
    await children.children_ended(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert "could not be ended" in caplog.text


async def test_a_run_ended_elsewhere_stops_every_helper_it_handed_on() -> None:
    """A run the store ended first stops here what it handed on: a helper whose keeper fails
    to stop leaves no other working, and the run's own end goes on."""
    from trellis.harness import pipeline
    from trellis.harness.journal import Journal

    class FailsFirst(Keeping):
        async def child_cancelled(self, run_id: str, reason: str | None) -> None:
            await super().child_cancelled(run_id, reason)
            if run_id == "first":
                raise ConnectionError("the task's server is down")

    adapter = FailsFirst()
    agent, _ = agent_of(LocalRuns(), adapter)
    runtime: Any = SimpleNamespace(cancelled="ended elsewhere", run_id="parent", tenant=TENANT)
    journal = Journal(handed_on=["first", "second"])
    await pipeline._ended_with_children(agent, runtime, journal, RunStatus.CANCELLED, ended=False)
    assert [run for run, _ in adapter.cancelled] == ["first", "second"]


class Listing(LocalRuns):
    """A run store that says which statuses each listing of a run's children asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[RunStatus | None] = []

    def iterate(self, **given: Any) -> Any:
        self.asked.append(given.get("status"))
        return super().iterate(**given)


async def test_only_a_runs_open_children_are_listed() -> None:
    """A run's end reads its children that have not ended, never every one it ever had."""
    runs = Listing()
    await started(runs, "parent")
    agent, _ = agent_of(runs)
    await children.end_children(agent, "parent", TENANT, status=RunStatus.SUCCESS)
    assert runs.asked == [RunStatus.RUNNING, RunStatus.PAUSED]
    runs.asked.clear()
    await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert runs.asked == [RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.QUEUED]


def test_each_keeper_names_the_runs_it_keeps_for_a_parent() -> None:
    """What stops with a run when the store cannot list its children: a live session's
    background subagents (Claude, ACP) and the Deep Agents async tasks (LangGraph) that run
    started, and nothing for a framework that keeps none."""
    from trellis.harness.adapters.acp import ACPAdapter
    from trellis.harness.adapters.claude import ClaudeAdapter
    from trellis.harness.adapters.langgraph import LangGraphAdapter

    for live in (ClaudeAdapter(), ACPAdapter()):
        live.live.attended["mine"] = cast("Any", SimpleNamespace(parent_run_id="parent"))
        live.live.attended["theirs"] = cast("Any", SimpleNamespace(parent_run_id="other"))
        assert live.kept_children("parent") == ["mine"]
    graph = LangGraphAdapter()
    helper = SimpleNamespace(parent_run_id="parent")
    graph.tasks._kept["task"] = cast("Any", SimpleNamespace(helper=helper))
    assert graph.kept_children("parent") == ["task"] and graph.kept_children("other") == []
    assert Keeps().kept_children("parent") == []


def test_a_background_subagent_cancelled_twice_keeps_the_first_reason() -> None:
    stopped: list[tuple[str, str]] = []
    kept: Any = object.__new__(Kept)
    kept.stopped, kept.run_id = None, "child"
    kept.agent = SimpleNamespace(stop=lambda run_id, reason: stopped.append((run_id, reason)))
    kept.cancel("the turn was cancelled")
    kept.cancel("its parent ended")
    assert kept.stopped == "the turn was cancelled"


async def test_a_run_kept_here_whose_stop_fails_is_said_and_the_stores_children_still_end() -> None:
    class FailsHere(Holding):
        async def child_cancelled(self, run_id: str, reason: str | None) -> None:
            await super().child_cancelled(run_id, reason)
            if run_id == "kept-here":
                raise ConnectionError("the task's server is down")

    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    agent, _ = agent_of(runs, FailsHere())
    with pytest.raises(ConnectionError, match="server is down"):
        await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert (await found(runs, "child")).status is RunStatus.CANCELLED


async def test_a_child_whose_own_children_cannot_be_listed_ended_and_says_so() -> None:
    class Grandchildless(LocalRuns):
        def iterate(self, **given: Any) -> Any:
            if given.get("parent_run_id") == "child":
                raise ConnectionError("the run store is down")
            return super().iterate(**given)

    runs = Grandchildless()
    await started(runs, "parent")
    await started(runs, "child", "parent")
    agent, stopped = agent_of(runs)
    with pytest.raises(ConnectionError, match="store is down"):
        await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert (await found(runs, "child")).status is RunStatus.CANCELLED
    assert stopped == [("child", children.PARENT_ENDED)]


async def test_a_child_kept_here_that_the_store_lists_too_is_stopped_once() -> None:
    runs = LocalRuns()
    await started(runs, "parent")
    await started(runs, "kept-here", "parent")
    adapter = Holding()
    agent, stopped = agent_of(runs, adapter)
    await children.end_children(agent, "parent", TENANT, status=RunStatus.CANCELLED)
    assert (await found(runs, "kept-here")).status is RunStatus.CANCELLED
    assert adapter.cancelled == stopped == [("kept-here", children.PARENT_ENDED)]


async def test_a_child_kept_here_that_its_caller_keeps_goes_on() -> None:
    adapter = Holding()
    agent, stopped = agent_of(LocalRuns(), adapter)
    await children.end_children(
        agent, "parent", TENANT, status=RunStatus.SUCCESS, kept=["kept-here"]
    )
    assert adapter.cancelled == stopped == []
