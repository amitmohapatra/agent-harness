"""The background tasks' pure parts (``trellis.harness.background``): a task's record and status
from its run's, what a check answers, the notification a task's outcome is, and how the harness
reads a message the user did not type (``history``, ``adapters.langgraph``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, get_notification_metadata

from trellis.contracts import (
    AgentError,
    Interrupt,
    RunRecord,
    RunStart,
    RunStatus,
)
from trellis.harness import background
from trellis.harness.adapters.langgraph import ASKED, added, carried_of, langchain_message, notified
from trellis.harness.history import (
    NOTIFICATION,
    Said,
    Speaker,
    from_window,
    notification_in,
    recorded,
    said_of,
)
from trellis.harness.runs import LocalRuns
from trellis.memory.models import WindowMessage

TASK = {
    "task_id": "task_1",
    "agent_name": "research",
    "thread_id": "task_1",
    "run_id": "run_1",
    "status": "running",
    "created_at": "2026-11-14T00:00:00Z",
    "last_checked_at": "2026-11-14T00:00:00Z",
    "last_updated_at": "2026-11-14T00:00:00Z",
}


def record(status: RunStatus, **more: Any) -> RunRecord:
    start = RunStart(
        run_id="run_1",
        tenant_id="t",
        agent_id="research",
        thread_id="task_1",
        user_id="u",
        input="x",
    )
    return RunRecord.from_start(start).model_copy(update={"status": status, **more})


def test_a_tasks_status_is_its_runs_in_deep_agents_words() -> None:
    assert {s: background.status_of(record(s)) for s in RunStatus} == dict(background.STATUS)
    made = background.task_of(record(RunStatus.RUNNING), "research")
    assert {k: made[k] for k in ("task_id", "thread_id", "run_id", "status")} == {
        "task_id": "task_1",
        "thread_id": "task_1",
        "run_id": "run_1",
        "status": "running",
    }


def test_a_record_observed_changes_its_times_only_when_its_run_moved_on() -> None:
    same = background.observed(TASK, record(RunStatus.RUNNING))
    assert same["last_updated_at"] == TASK["last_updated_at"]
    moved = background.observed(TASK, record(RunStatus.SUCCESS))
    assert moved["status"] == "success" and moved["last_updated_at"] != TASK["last_updated_at"]


def test_what_a_check_answers_by_status() -> None:
    assert background.result_of(TASK, record(RunStatus.RUNNING)) == {
        "status": "running",
        "thread_id": "task_1",
    }
    done = background.result_of(TASK, record(RunStatus.SUCCESS, output={"a": 1}))
    assert done["result"] == '{"a": 1}'
    assert background.result_of(TASK, record(RunStatus.SUCCESS))["result"] == (
        "(completed with no output)"
    )
    failed = record(RunStatus.ERROR, error=AgentError(code="x", message="it broke"))
    assert background.result_of(TASK, failed)["error"] == "it broke"
    assert background.result_of(TASK, record(RunStatus.TIMEOUT))["error"] == background.FAILED
    asked = Interrupt(interrupt_id="i1", tenant_id="t", run_id="run_1", question="Which?")
    waiting = background.result_of(TASK, record(RunStatus.PAUSED, awaiting=asked))
    assert waiting["interrupts"] == [{"id": "i1", "value": "Which?"}]
    assert background.result_of(TASK, record(RunStatus.PAUSED))["interrupts"] == []


def test_an_outcome_is_told_in_deep_agents_shape_escaped_and_cut() -> None:
    assert background.update_of(TASK, record(RunStatus.QUEUED)) is None
    failed = background.update_of(TASK, record(RunStatus.ERROR))
    assert failed is not None and failed["error"] == background.FAILED
    long = "<b>" + "x" * background.RESULT_CHARS
    update = background.update_of(TASK, record(RunStatus.SUCCESS, output=long))
    assert update is not None
    blocks = background.content(update)
    text = "".join(b["text"] for b in blocks)
    assert text.startswith(background.NOTE) and "&lt;b&gt;" in text and "[cut off;" in text
    assert text.endswith(background.NEXT["success"])
    asked = Interrupt(interrupt_id="i1", tenant_id="t", run_id="run_1", question="Which?")
    waiting = background.update_of(TASK, record(RunStatus.PAUSED, awaiting=asked))
    assert waiting is not None and "<details>" in "".join(
        b["text"] for b in background.content(waiting)
    )
    message = background.notification(update)
    assert get_notification_metadata(message) == {
        "source": "async_subagent",
        "data": {
            "task_id": "task_1",
            "agent_name": "research",
            "status": "success",
            "run_id": "run_1",
        },
    }
    [given] = background.input_of(update)
    assert given["response_metadata"] == {ASKED: True}
    assert given["additional_kwargs"] == message.additional_kwargs


def test_a_notification_input_is_a_notification_in_the_conversation() -> None:
    update = background.update_of(TASK, record(RunStatus.SUCCESS, output="done"))
    assert update is not None
    told = notification_in(background.input_of(update))
    assert told is not None and told.role is Speaker.USER and "done" in told.content
    assert told.notified == {
        "source": "async_subagent",
        "data": {
            "task_id": "task_1",
            "agent_name": "research",
            "status": "success",
            "run_id": "run_1",
        },
    }
    plain = notification_in(
        [{"role": "user", "content": "hi", "additional_kwargs": {"lc_source": "x"}}]
    )
    assert plain is not None and plain.content == "hi" and plain.notified == {"source": "x"}
    assert notification_in("hi") is None
    assert notification_in([{"role": "user", "content": "hi"}]) is None
    assert notification_in([{"role": "user", "content": "hi", "additional_kwargs": "no"}]) is None


def test_a_notification_is_recorded_as_one_and_replayed_as_one() -> None:
    noted = {"source": "async_subagent", "data": {"task_id": "task_1"}}
    said = Said(Speaker.USER, "update", notified=noted, facts={"status": "SUCCESS"})
    stored = recorded(said)
    assert stored["custom_metadata"]["run_record"] == {"status": "SUCCESS", NOTIFICATION: noted}
    back = said_of(stored)
    assert (back.notified, back.facts) == (noted, {"status": "SUCCESS"})
    window = WindowMessage(id="m", role="USER", text="update", run_record={NOTIFICATION: noted})
    [replayed] = from_window([window])
    assert replayed.notified == noted and replayed.facts is None
    message = langchain_message(replayed)
    assert notified(message) == noted
    bare = langchain_message(Said(Speaker.USER, "u", notified={"source": "s"}))
    assert notified(bare) == {"source": "s"}
    assert notified("not a message") is None


def test_the_turns_question_is_the_users_never_a_notification_after_it() -> None:
    told = background.notification(
        {"task_id": "t", "agent_name": "a", "status": "success", "run_id": "r", "result": "R"}
    )
    messages = [
        HumanMessage("earlier", id="h0"),
        AIMessage("before"),
        HumanMessage("now", id="h1"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "look", "args": {}}]),
        ToolMessage("seen", tool_call_id="c1"),
        told,
        AIMessage("done"),
    ]
    said = added(messages)
    assert [s.role for s in said] == [
        Speaker.ASSISTANT,
        Speaker.TOOL,
        Speaker.USER,
        Speaker.ASSISTANT,
    ]
    assert said[2].notified is not None and said[3].content == "done"
    asked = HumanMessage(
        "x", response_metadata={ASKED: True}, additional_kwargs=told.additional_kwargs
    )
    assert [s.content for s in added([HumanMessage("old"), asked, AIMessage("reply")])] == ["reply"]
    assert added([told]) == []


def test_a_conversation_carries_only_what_its_graph_holds_anything_in() -> None:
    assert carried_of({"async_tasks": {}, "messages": []}) == {}
    assert carried_of({"async_tasks": {"t": TASK}}) == {"async_tasks": {"t": TASK}}


# --------------------------------------------------------------------------- with a run store
def runtime(runs: LocalRuns, **more: Any) -> Any:
    """What the background parts read of a run: its agent's run store, its tenant, its
    conversation's carried state."""
    harness = SimpleNamespace(runs=runs, background=background.Tasks())
    agent = SimpleNamespace(harness=harness, id="chat")
    return SimpleNamespace(
        **{"agent": agent, "tenant": "t", "carried": {}, "notification": None, **more}
    )


async def stored(runs: LocalRuns, status: RunStatus = RunStatus.RUNNING, **more: Any) -> RunRecord:
    start = RunStart(
        run_id=more.pop("run_id", "run_1"),
        tenant_id="t",
        agent_id="research",
        thread_id=more.pop("thread_id", "task_1"),
        user_id="u",
        input="x",
        **more,
    )
    made = await runs.start(start)
    if status is RunStatus.SUCCESS:
        await runs.finish(made.run_id, status, output="done", tenant="t")
    found = await runs.get(made.run_id, tenant="t")
    assert found is not None
    return found


async def test_the_five_tools_answer_what_they_cannot_do() -> None:
    runs = LocalRuns()
    offered = background.Offered({"research": None}, described={"research": "Research."})
    assert await offered.call("check_async_task", None, {}, {"task_id": "x"}) == (
        background.NO_RUN,
        {},
    )
    run = runtime(runs)
    said, _ = await offered.call(
        "start_async_task", run, {}, {"subagent_type": "other", "description": "d"}
    )
    assert said == "Unknown async subagent type `other`. Available types: `research`"
    assert await offered.call("check_async_task", run, {}, {"task_id": "task_9"}) == (
        "No tracked task found for task_id: 'task_9'",
        {},
    )
    gone = {"task_1": {**TASK, "run_id": "run_gone"}}
    assert (await offered.call("check_async_task", run, gone, {"task_id": "task_1"}))[0] == (
        "No tracked task found for task_id: 'task_1'"
    )
    listed = await offered.call(
        "list_async_tasks", run, {"task_1": TASK}, {"status_filter": "error"}
    )
    assert listed == (background.NONE, {})
    assert await background.pulled(run, gone) == ({}, [])
    await offered.asked(run, {"task_1": {**TASK, "status": "waiting", "run_id": "run_gone"}})


async def test_a_question_is_asked_only_while_the_task_waits_on_it() -> None:
    runs = LocalRuns()
    working = await stored(runs)
    assert await background.ask(runtime(runs), working) is None


def test_a_turn_of_the_chats_own_is_quiet_only_for_what_was_told() -> None:
    runs = LocalRuns()
    told = notification_in(
        background.input_of(
            {"task_id": "task_1", "agent_name": "a", "status": "success", "run_id": "run_1"}
        )
    )
    run = runtime(runs, notification=told, attempt=1)
    assert background.quiet(run, "NO_REPORT") and not background.quiet(runtime(runs), "NO_REPORT")
    run.carried = {"async_tasks": {"task_1": {**TASK, "status": "success"}}}
    assert background.of_turn(run) == ["run_1"]
    queued: dict[str, Any] = {**TASK, "superseded_run_ids": ["run_0"]}
    run.carried = {"async_tasks": {"task_1": queued}}
    assert background.of_turn(run) == ["run_1", "run_0"]


async def test_a_turn_of_the_chats_own_is_stale_once_told_or_once_its_question_went() -> None:
    runs = LocalRuns()
    event = {"task_id": "task_1", "agent_name": "a", "status": "success", "run_id": "run_1"}
    run = runtime(runs, notification=notification_in(background.input_of(event)), attempt=1)
    assert not await background.stale(run)  # its task is not carried: told nowhere yet
    run.carried = {"async_tasks": {"task_1": {**TASK, "status": "success"}}}
    assert await background.stale(run)
    waiting = {**event, "status": "waiting"}
    run.notification = notification_in(background.input_of(waiting))
    assert await background.stale(run)  # its run waits on nothing (none recorded)
    await stored(runs)
    assert await background.stale(run)  # it works: answered already
    run.attempt = 2
    assert not await background.stale(run)  # a resumed one goes on


def test_a_record_an_input_tells_is_recorded_once() -> None:
    event = {"task_id": "task_1", "agent_name": "a", "status": "success", "run_id": "run_0"}
    done = {"task_1": {**TASK, "status": "success"}}
    assert background.recorded(event, done) == {}  # an earlier run's, of a task that ended
    assert background.recorded(None, done) == {}
    assert background.recorded(event, {"task_1": TASK})["task_1"]["run_id"] == "run_0"


async def test_a_child_run_of_a_turn_with_no_chat_is_told_nowhere() -> None:
    runs = LocalRuns()
    parent = await stored(runs, run_id="run_p", thread_id="run_p")  # no thread of its own
    child = await stored(runs, run_id="run_c", thread_id="run_c", parent_run_id=parent.run_id)
    agent = runtime(runs).agent
    assert await background.origin_of(agent, child) is None
    await background.delivered(agent, child)  # nothing to tell, nowhere: no error
    orphan = await stored(runs, run_id="run_o", thread_id="run_o")
    assert await background.origin_of(agent, orphan) is None
    chat = await stored(
        runs, run_id="run_q", thread_id="chat-1", metadata={"frontend": {"tools": []}}
    )
    helped = await stored(
        runs,
        run_id="run_h",
        thread_id="run_h",
        parent_run_id=chat.run_id,
        metadata={"native_helper": {"name": "optimizer", "task_id": "t_1"}},
    )
    found = await background.origin_of(agent, helped)
    assert found == {
        "thread": "chat-1",
        "name": "optimizer",
        "agent": "research",
        "task_id": "t_1",
        "frontend": {"tools": []},
    }


async def test_a_chat_is_idle_whatever_another_users_thread_of_its_name_does() -> None:
    from trellis.harness.conversation import Turns

    runs = LocalRuns()
    agent = runtime(runs).agent
    agent.turns = Turns()
    start = RunStart(
        run_id="run_b", tenant_id="t", agent_id="chat", thread_id="chat-1", user_id="bob", input="x"
    )
    await runs.start(start)
    assert await background.idle(agent, "t", "u", "chat-1")  # bob's turn is not u's chat
    mine = start.model_copy(update={"run_id": "run_u", "user_id": "u"})
    await runs.start(mine)
    assert not await background.idle(agent, "t", "u", "chat-1")
    assert await background.idle(agent, "t", "u", "chat-1", ending="run_u")


async def test_a_question_answered_elsewhere_is_not_asked_again() -> None:
    runs = LocalRuns()
    await stored(runs)  # its run works again: answered on another server
    offered = background.Offered({"research": None}, described={"research": "Research."})
    await offered.asked(runtime(runs), {"task_1": {**TASK, "status": "waiting"}})


def test_two_overlapping_turns_changing_one_task_lose_neither_update() -> None:
    """The conversation's task list is merged task by task: of two records of one task, the
    one updated last stands (``last_updated_at``; at the same second, the further status) —
    whichever turn ends last — and a task only one of them has is kept."""
    from trellis.harness.tools.results import merged

    running = {**TASK, "status": "running", "last_updated_at": "2026-11-14T00:00:01Z"}
    ended = {**TASK, "status": "success", "last_updated_at": "2026-11-14T00:00:05Z"}
    other = {**TASK, "task_id": "task_2"}
    newer = merged({"async_tasks": {"task_1": ended}}, {"async_tasks": {"task_1": running}})
    assert newer["async_tasks"]["task_1"]["status"] == "success"  # the older turn ended last
    both = merged(
        {"async_tasks": {"task_1": running, "task_2": other}}, {"async_tasks": {"task_1": ended}}
    )
    assert both["async_tasks"] == {"task_1": ended, "task_2": other}
    same = {**running, "last_updated_at": ended["last_updated_at"]}
    assert merged({"async_tasks": {"task_1": ended}}, {"async_tasks": {"task_1": same}}) == {
        "async_tasks": {"task_1": ended}
    }
    again = {**TASK, "run_id": "run_2", "status": "running", "last_updated_at": "2026-11-14T00:00:09Z"}
    assert merged({"async_tasks": {"task_1": ended}}, {"async_tasks": {"task_1": again}}) == {
        "async_tasks": {"task_1": again}  # a follow-up after its end: updated later
    }
