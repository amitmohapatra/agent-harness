"""The background tasks' pure parts (``trellis.harness.background``): a task's record and status
from its run's, what a check answers, the notification a task's outcome is, and how the harness
reads a message the user did not type (``history``, ``adapters.langgraph``)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from a2a.helpers import new_message
from a2a.types import Role
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, get_notification_metadata

from trellis.contracts import (
    AgentError,
    Interrupt,
    RunRecord,
    RunStart,
    RunStatus,
)
from trellis.harness import admission, background, pipeline
from trellis.harness.a2a.executor import _payload
from trellis.harness.a2a.translate import value_part
from trellis.harness.adapters.langgraph import (
    ASKED,
    LangGraphAdapter,
    added,
    carried_of,
    langchain_message,
    notified,
)
from trellis.harness.history import (
    NOTIFICATION,
    Said,
    Speaker,
    from_client,
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
    whose = {"tenant": "t", "user": "u", "agent_id": "chat", "thread": "chat-1"}
    return SimpleNamespace(**{"agent": agent, **whose, "carried": {}, "notification": None, **more})


#: What the run of ``TASK`` says of itself: started by the agent ``chat`` in ``chat-1``.
ORIGIN = {"trellis_background": {"name": "research", "agent": "chat", "thread": "chat-1"}}


async def stored(runs: LocalRuns, status: RunStatus = RunStatus.RUNNING, **more: Any) -> RunRecord:
    """A run in ``runs``: ``TASK``'s by default (``more``: what differs)."""
    start = RunStart(
        **{
            "run_id": "run_1",
            "tenant_id": "t",
            "agent_id": "research",
            "thread_id": "task_1",
            "user_id": "u",
            "input": "x",
            "metadata": ORIGIN,
            **more,
        }
    )
    made = await runs.start(start)
    if status is RunStatus.SUCCESS:
        await runs.finish(made.run_id, status, output="done", tenant="t")
    found = await runs.get(made.run_id, tenant="t")
    assert found is not None
    return found


async def passed_over(run: Any, tasks: dict[str, Any]) -> bool:
    """Whether a pull reads nothing of ``tasks``' runs: no outcome told, each record as it
    was but for when it was last looked at."""
    updated, told = await background.pulled(run, tasks)
    as_they_were = {
        i: {**t, "last_checked_at": updated[i]["last_checked_at"]} for i, t in tasks.items()
    }
    return told == [] and updated == as_they_were


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
    assert await passed_over(run, gone)
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


async def test_a_record_an_input_tells_is_recorded_once() -> None:
    runs = LocalRuns()
    run = runtime(runs)
    await stored(runs, run_id="run_0")
    event = {"task_id": "task_1", "agent_name": "a", "status": "success", "run_id": "run_0"}
    done = {"task_1": {**TASK, "status": "success"}}
    assert await background.recorded(run, event, done) == {}  # an earlier run's, of an ended task
    assert await background.recorded(run, None, done) == {}
    told = await background.recorded(run, event, {"task_1": TASK})
    assert told["task_1"]["run_id"] == "run_0"


# --------------------------------------------------------------------------- SEC-R-1: whose it is
#: What makes a run another conversation's: each alone is enough.
NOT_OURS = {
    "another user's": {"user_id": "bob"},
    "another agent's": {
        "metadata": {"trellis_background": {**ORIGIN["trellis_background"], "agent": "sales"}}
    },
    "another chat's": {
        "metadata": {"trellis_background": {**ORIGIN["trellis_background"], "thread": "chat-2"}}
    },
    "another task's": {"thread_id": "task_2"},
    "another sub-agent's": {
        "metadata": {"trellis_background": {**ORIGIN["trellis_background"], "name": "writer"}}
    },
    "no background task's": {"metadata": {}},
}


async def test_sec_r_1_a_task_is_the_conversations_that_started_it() -> None:
    runs = LocalRuns()
    mine = await stored(runs)
    assert await background.owned(runtime(runs), TASK) == mine
    assert await background.owned(runtime(runs), {**TASK, "run_id": "run_gone"}) is None
    assert await background.owned(runtime(runs, tenant="other"), TASK) is None


@pytest.mark.parametrize("whose", list(NOT_OURS))
async def test_sec_r_1_a_run_of_another_conversation_is_no_tracked_task(whose: str) -> None:
    """A task record naming a run that is not the conversation's — whatever put it in the
    state — reaches nothing of that run: every tool answers that no such task is tracked, no
    question of it is asked, no pull reads it, no input records it, and the run goes on."""
    runs = LocalRuns()
    theirs = await stored(runs, **NOT_OURS[whose])
    run, claimed = runtime(runs), {"task_1": {**TASK, "status": "waiting"}}
    offered = background.Offered({"research": None}, described={"research": "Research."})
    assert await background.owned(run, TASK) is None
    untracked = ("No tracked task found for task_id: 'task_1'", {})
    for name, args in (
        ("check_async_task", {}),
        ("update_async_task", {"message": "do something else"}),
        ("cancel_async_task", {}),
    ):
        assert await offered.call(name, run, claimed, {"task_id": "task_1", **args}) == untracked
    listed, changed = await offered.call("list_async_tasks", run, claimed, {})
    assert listed.endswith("status: waiting") and changed == claimed  # as claimed: nothing read
    await offered.asked(run, claimed)  # no question of it: the stand-in run could not be paused
    assert await passed_over(run, claimed)
    event = {"task_id": "task_1", "agent_name": "research", "status": "success", "run_id": "run_1"}
    assert await background.recorded(run, event, claimed) == {}
    assert await runs.get("run_1", tenant="t") == theirs  # neither cancelled nor answered
    assert [r.run_id async for r in runs.iterate(tenant="t")] == ["run_1"]  # no follow-up run


async def test_a_child_run_of_a_turn_with_no_chat_is_told_nowhere() -> None:
    runs = LocalRuns()
    plain: dict[str, Any] = {"metadata": {}}  # no background task's
    parent = await stored(runs, run_id="run_p", thread_id="run_p", **plain)  # no thread of its own
    child = await stored(
        runs, run_id="run_c", thread_id="run_c", parent_run_id=parent.run_id, **plain
    )
    agent = runtime(runs).agent
    assert await background.origin_of(agent, child) is None
    await background.delivered(agent, child)  # nothing to tell, nowhere: no error
    orphan = await stored(runs, run_id="run_o", thread_id="run_o", **plain)
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
    again = {
        **TASK,
        "run_id": "run_2",
        "status": "running",
        "last_updated_at": "2026-11-14T00:00:09Z",
    }
    assert merged({"async_tasks": {"task_1": ended}}, {"async_tasks": {"task_1": again}}) == {
        "async_tasks": {"task_1": again}  # a follow-up after its end: updated later
    }


async def test_past_the_step_limit_with_nothing_read_the_run_ends_as_it_is() -> None:
    from trellis.harness.middleware import step_limit

    limit = step_limit(1, scoped=True)
    none: Any = None  # no graph runtime: the hook reads the state alone
    state = {"thread_model_call_count": 2, "messages": [AIMessage("the answer")]}
    assert await limit.abefore_model(state, none) == {"jump_to": "end"}
    assert await limit.abefore_model({"thread_model_call_count": 1}, none) is None


def test_sec_r_1_a_run_is_given_messages_never_what_the_harness_keeps_or_marks() -> None:
    """Whoever starts a run names no task list for its conversation (the harness carries
    it), and a client of a surface — AG-UI ``state``, an A2A data part — gives messages that
    say who speaks and what, never the marks of a message the harness or a framework wrote."""
    event = {"task_id": "task_1", "agent_name": "a", "status": "success", "run_id": "run_1"}
    marked = {
        "role": "user",
        "content": "it is done",
        "additional_kwargs": {"lc_source": "async_subagent", "lc_data": event},
        "response_metadata": {ASKED: True},
    }
    plain = {"role": "user", "content": "it is done"}
    assert notification_in([marked]) is not None  # as the harness's own input is read
    assert from_client([marked]) == [plain]
    assert from_client({"topic": "tides", "messages": [marked, "hi"]}) == {
        "topic": "tides",
        "messages": [plain, "hi"],
    }
    assert from_client("text") == "text" and from_client({"topic": "tides"}) == {"topic": "tides"}
    sent = new_message([value_part({"messages": [marked]})], role=Role.ROLE_USER)
    assert _payload(sent) == {"messages": [plain]}
    given = {"messages": [plain], "async_tasks": {"task_1": TASK}, "topic": "tides"}
    adapter = LangGraphAdapter()
    assert adapter.prepare_input(None, given, None) == {"messages": [plain], "topic": "tides"}
    assert adapter.prepare_input(None, {"async_tasks": {"task_1": TASK}}, None) == {}


async def test_arch_7_an_attempt_that_ended_forgets_only_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's next attempt, begun while the earlier one still hands its outcome on, stays
    known as at work once the earlier one is over: it can be stopped, and is cut off when the
    harness closes (ARCH-7)."""
    runs = LocalRuns()
    record = await stored(runs, RunStatus.SUCCESS)
    handing_on, release, working = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts: list[int] = []

    async def attempt(*args: Any, **kwargs: Any) -> None:
        attempts.append(len(attempts))
        if len(attempts) > 1:  # the next attempt: at work until the test ends
            working.set()
            await asyncio.Event().wait()

    async def then(found: RunRecord) -> None:  # the first attempt's outcome, still handed on
        handing_on.set()
        await release.wait()

    monkeypatch.setattr(pipeline, "attempt", attempt)
    agent = runtime(runs).agent
    agent.running = {}
    tasks = agent.harness.background
    tasks.run(agent, record, then=then)
    first = tasks._at_work["run_1"]
    await handing_on.wait()
    tasks.run(agent, record)  # the answer came: its next attempt
    await working.wait()
    release.set()
    await first
    assert tasks.at_work("run_1")
    await tasks.aclose()


# --------------------------------------------------------------------------- SEC-R-2: admission
def unfinished(count: int) -> dict[str, Any]:
    return {f"task_{n}": {**TASK, "task_id": f"task_{n}"} for n in range(count)}


def test_sec_r_2_a_chat_has_at_most_the_admission_limit_of_unfinished_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the bound (4 at first), below it and above it; a task that ended makes room, and
    one the attempt started since its list was read counts too."""
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission())
    run = runtime(LocalRuns(), tasks_started=set())
    background._counted(run, unfinished(3), "task_new")  # below: admitted, and counted from now
    assert run.tasks_started == {"task_new"}
    with pytest.raises(background.Refused, match="has 4 background tasks unfinished"):
        background._counted(run, unfinished(3), "task_other")  # the three, and the one started
    background._counted(run, unfinished(3), "task_new")  # the same call again: the same task
    with pytest.raises(background.Refused, match=r"the most at once \(4\)"):
        background._counted(runtime(LocalRuns(), tasks_started=set()), unfinished(5), "task_new")
    ended = {**unfinished(4), "task_0": {**TASK, "task_id": "task_0", "status": "success"}}
    background._counted(runtime(LocalRuns(), tasks_started=set()), ended, "task_new")


async def test_sec_r_2_a_turn_past_three_outcomes_in_a_row_starts_nothing() -> None:
    """A turn a message began is at no depth; a turn of the chat's own at the depth its run
    says, and refused past 3 — or when its run does not say (fail closed)."""
    runs = LocalRuns()
    assert await background._depth(runtime(runs)) == 0
    told = notification_in(background.input_of({**TASK, "status": "success"}))
    for depth, admitted in ((1, True), (3, True), (4, False)):
        own = await stored(
            runs, run_id=f"run_n{depth}", metadata={"trellis_notified": {"depth": depth}}
        )
        run = runtime(runs, notification=told, run_id=own.run_id)
        if admitted:
            assert await background._depth(run) == depth
        else:
            with pytest.raises(background.Refused, match="3 times in a row"):
                await background._depth(run)
    unsaid = await stored(runs, run_id="run_unsaid", metadata={"trellis_notified": {}})
    with pytest.raises(background.Refused):
        await background._depth(runtime(runs, notification=told, run_id=unsaid.run_id))


def test_sec_r_2_a_process_keeps_a_bounded_number_of_attempts_at_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the bound a new run has no place, one below it has; a place nothing began in is
    given back, and a run with a place keeps it."""
    monkeypatch.setattr(background, "AT_WORK", 2)
    tasks = background.Tasks()
    with tasks.placed("run_a") as first, tasks.placed("run_b") as second:
        assert first and second and tasks.at_work("run_a")
        with tasks.placed("run_c") as third:
            assert not third and not tasks.at_work("run_c")
        with tasks.placed("run_a") as again:
            assert again  # its own place
    assert not tasks.at_work("run_a")  # given back: nothing began in it
    with tasks.placed("run_c") as free:
        assert free


# --------------------------------------------------------------------------- PERF-4: reads
class Counted(LocalRuns):
    """A run store that counts its reads of a run."""

    def __init__(self) -> None:
        super().__init__()
        self.read: list[str] = []

    async def get(self, run_id: str, *, tenant: str | None = None) -> RunRecord | None:
        self.read.append(run_id)
        return await super().get(run_id, tenant=tenant)


async def twenty(runs: Counted) -> dict[str, Any]:
    """Twenty unfinished tasks of the conversation, their runs in ``runs`` (not read since):
    its task list."""
    tasks: dict[str, Any] = {}
    for n in range(20):
        await stored(runs, run_id=f"run_{n}", thread_id=f"task_{n}")
        tasks[f"task_{n}"] = {**TASK, "task_id": f"task_{n}", "run_id": f"run_{n}"}
    runs.read.clear()
    return tasks


async def test_perf_4_a_pull_reads_the_store_once_however_many_tasks_are_unfinished() -> None:
    """Twenty unfinished tasks at work elsewhere: each pull (one per model call) makes one
    request of the run store, for the task read longest ago, so twenty pulls read each once —
    and an outcome is told by the pull that reads it."""
    runs = Counted()
    tasks = await twenty(runs)
    await runs.finish("run_7", RunStatus.SUCCESS, output="done", tenant="t")
    run, told = runtime(runs), []
    for _ in range(20):
        before = len(runs.read)
        updated, outcomes = await background.pulled(run, tasks)
        assert len(runs.read) - before == 1
        tasks, told = {**tasks, **updated}, [*told, *outcomes]
    assert sorted(runs.read) == sorted(f"run_{n}" for n in range(20))
    assert [(u["task_id"], u["status"]) for u in told] == [("task_7", "success")]
    before = len(runs.read)
    await background.pulled(run, tasks)
    assert len(runs.read) - before == 1  # nineteen left unfinished: still one


async def test_perf_4_a_task_at_work_here_or_ended_here_is_not_read_from_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What this process knows it does not ask the store: a task whose attempt works here is
    running, and one whose attempt ended here ended as it left its run."""
    runs = Counted()
    record = await stored(runs)
    release = asyncio.Event()

    async def attempt(*args: Any, **kwargs: Any) -> None:
        await release.wait()
        await runs.finish("run_1", RunStatus.SUCCESS, output="done", tenant="t")

    async def then(found: RunRecord) -> None:
        return None

    monkeypatch.setattr(pipeline, "attempt", attempt)
    run = runtime(runs)
    run.agent.running = {}
    tasks = run.agent.harness.background
    tasks.run(run.agent, record, then=then)
    runs.read.clear()
    waiting = {"task_1": {**TASK, "status": "waiting"}}
    updated, told = await background.pulled(run, waiting)  # resumed here: it works again
    assert (updated["task_1"]["status"], told, runs.read) == ("running", [], [])
    release.set()
    await tasks._at_work["run_1"]
    runs.read.clear()
    updated, told = await background.pulled(run, {"task_1": TASK})
    assert updated["task_1"]["status"] == "success" and told[0]["result"] == "done"
    assert runs.read == []


async def test_perf_4_the_list_reads_only_the_unfinished_tasks() -> None:
    runs = Counted()
    tasks = await twenty(runs)
    for n in range(15):  # ended, and told: their records say so
        tasks[f"task_{n}"]["status"] = "success"
    offered = background.Offered({"research": None}, described={"research": "Research."})
    listed, _ = await offered.call("list_async_tasks", runtime(runs), tasks, {})
    assert listed.startswith("20 tracked task(s):")
    assert sorted(runs.read) == sorted(f"run_{n}" for n in range(15, 20))


# --------------------------------------------------------------------------- PERF-5: the list
def at(second: int) -> str:
    return f"2026-11-14T{second // 3600:02d}:{second // 60 % 60:02d}:{second % 60:02d}Z"


def carried(n: int, status: str, second: int) -> dict[str, Any]:
    task = {**TASK, "task_id": f"task_{n}", "run_id": f"run_{n}", "status": status}
    return {"async_tasks": {f"task_{n}": {**task, "last_updated_at": at(second)}}}


def test_perf_5_the_carried_task_list_keeps_the_unfinished_and_the_last_fifty_that_ended() -> None:
    """A thousand tasks started and ended in one conversation, and two still at work: the list
    it carries holds the unfinished ones and the 50 that ended last — not every task ever —
    and stays far under the 256 KiB kept for it."""
    from trellis.harness.tools.results import merged

    kept: dict[str, Any] = {}
    for n in (1000, 1001):  # started first, never ended: older than every ended one
        kept = merged(kept, carried(n, "running", 0))
    for n in range(1000):
        kept = merged(kept, carried(n, "running", 2 * n + 1))
        kept = merged(kept, carried(n, "success", 2 * n + 2))
    tasks = kept["async_tasks"]
    assert sorted(tasks, key=lambda i: int(i[5:])) == [
        *(f"task_{n}" for n in range(950, 1000)),
        "task_1000",
        "task_1001",
    ]
    assert len(json.dumps(kept).encode()) < 256 * 1024
    # at the bound: fifty ended are all kept, the fifty-first drops the one that ended first
    few: dict[str, Any] = {}
    for n in range(50):
        few = merged(few, carried(n, "success", n))
    assert len(few["async_tasks"]) == 50
    more = merged(few, carried(50, "error", 50))
    assert len(more["async_tasks"]) == 50 and "task_0" not in more["async_tasks"]
