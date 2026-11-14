"""``ReAct``'s own sub-agents in the background, with no server to host
(``trellis.harness.background``, ``bridges.background_tasks.BackgroundTasks``): a turn starts a task
and replies at once, the chat takes new messages while it works, and its outcome comes back to
the chat — at the conversation's next turn, or by a turn of the chat's own when it is idle.
The tools are Deep Agents' async sub-agent tools, by name and arguments; the task is a child run
of the turn, at work in this process and held in the run store.

A chat turn here is what ``serve_chat`` starts (:func:`chatted`: a front end's run, which a
person reads); no model is called (each is scripted), and the sub-agents are functions or
scripted graphs whose work waits on a gate the test opens.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, Final

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from tests.integration.test_react_async_subagent import Model
from tests.support.arm import arm
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    RunStatus,
)
from trellis.harness import background, pipeline, schedules
from trellis.harness.adapters.langgraph import notified
from trellis.harness.agent import Agent
from trellis.harness.agui import threads as threads_module
from trellis.harness.agui.threads import Threads
from trellis.harness.conversation import Mode
from trellis.harness.frontend import Frontend
from trellis.harness.harness import LOCAL_TENANT
from trellis.harness.result import Result
from trellis.harness.runs import LocalRuns

USER: Final = "ada"
THREAD: Final = "chat-1"
START: Final = (
    "start_async_task",
    {"description": "lighthouses", "subagent_type": "research"},
)
TOOLS: Final = (
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
)


async def chatted(agent: Agent, text: str, *, mode: Mode = Mode.QUEUE) -> Result:
    """A chat turn as ``serve_chat`` starts one: a front end's run, which a person reads."""
    record = await agent._opened(
        text, user=USER, thread=THREAD, tenant=None, mode=mode, app=Frontend()
    )
    return await pipeline.attempt(agent, record, text, mode=mode)


async def until(check: Callable[[], Any], seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not await _value(check()):  # noqa: ASYNC110 - another task's progress
            await asyncio.sleep(0.02)


async def _value(found: Any) -> Any:
    return await found if asyncio.iscoroutine(found) else found


def told(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """The task updates among ``messages``: each notification's data (its ``TaskEvent``)."""
    return [
        (notified(m) or {}).get("data") or {}
        for m in messages
        if (notified(m) or {}).get("source") == background.SOURCE
    ]


def results(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, ToolMessage)]


class Gate:
    """The sub-agent's work, held until the test opens it."""

    def __init__(self) -> None:
        self.opened = asyncio.Event()
        self.started: list[str] = []

    async def researcher(self, input: str, agent: Runtime) -> str:
        """Research a topic in depth."""
        self.started.append(input)
        await self.opened.wait()
        return f"FOUND: {input}"


@pytest.fixture
def gate() -> Gate:
    return Gate()


def chat(harness: Harness, gate: Gate, model: Model, **more: Any) -> Agent:
    research = harness.wrap(gate.researcher, id="research")
    return harness.wrap(ReAct("You help.", model, subagents=[research], **more), id="chat")


async def task_run(harness: Harness, parent: str) -> Any:
    """The background task's run the turn ``parent`` started."""
    [child] = [c async for c in harness.runs.iterate(parent_run_id=parent)]
    found = await harness.runs.get(child.run_id)
    assert found is not None
    return found


# --------------------------------------------------------------------------- start, chat on
async def test_a_turn_starts_a_task_and_replies_at_once(harness: Harness, gate: Gate) -> None:
    model = Model(turns=[START, "I started the research."])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    assert first.status is RunStatus.SUCCESS and first.answer == "I started the research.", (
        first.error,
        first.interrupt,
    )
    assert set(TOOLS) <= set(model.requests[0]["tools"])  # Deep Agents' five, offered
    [said] = results(model.seen[1])
    task = await task_run(harness, first.run_id)
    assert said == f"Launched async subagent. task_id: {task.thread_id}"
    # the turn ended while its task could not (held): a child run of the turn, still at work
    assert task.status is RunStatus.RUNNING and task.parent_run_id == first.run_id
    assert task.metadata[background.BACKGROUND]["name"] == "research"
    await until(lambda: gate.started == ["lighthouses"])
    assert harness.holds.holds(task.run_id)  # held in the run store while it works
    gate.opened.set()
    await until(lambda: _final(harness, task.run_id))


async def _final(harness: Harness, run_id: str) -> bool:
    found = await harness.runs.get(run_id)
    return found is not None and found.final


async def test_a_second_message_is_answered_while_the_task_runs(
    harness: Harness, gate: Gate
) -> None:
    model = Model(turns=[START, "Started.", "4"])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    second = await chatted(agent, "what is 2+2?")
    assert second.status is RunStatus.SUCCESS and second.answer == "4", second.error
    task = await task_run(harness, first.run_id)
    assert task.status is RunStatus.RUNNING  # answered while the task could not end
    assert not told(model.seen[2])  # it works: nothing to tell
    gate.opened.set()


async def test_the_result_arrives_at_the_next_turn_before_its_model_call(
    memory_harness: Harness,
    memory_service: Any,
    gate: Gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task ended while the chat was busy (a turn at work on another server: no turn of
    the chat's own starts), so the next turn's model is given its result first: a message after
    the user's question that the user did not type (langchain-core's notification metadata,
    Deep Agents' shape). The user's question is still the turn's; the notification is recorded
    as one; the turn after is not told again."""
    monkeypatch.setattr(background, "idle", _busy)

    def answers(messages: list[BaseMessage]) -> str:
        [update] = told(messages)
        return f"Your research is back: {update['status']}"

    model = Model(turns=[START, "Started.", answers, "Nothing new."])
    agent = chat(memory_harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(memory_harness, first.run_id)
    gate.opened.set()
    await until(lambda: _final(memory_harness, task.run_id))
    second = await chatted(agent, "anything new?")
    assert second.answer == "Your research is back: success", second.error
    shown = model.seen[2]
    # the user's question (with memory's context, as its question carries it), then the update
    assert isinstance(shown[-2], HumanMessage) and shown[-2].text.endswith("anything new?")
    assert notified(shown[-1]) is not None and "FOUND: lighthouses" in str(shown[-1].content)
    assert told(shown) == [
        {
            "task_id": task.thread_id,
            "agent_name": "research",
            "status": "success",
            "run_id": task.run_id,
        }
    ]
    kept = [m for msgs in memory_service.threads.values() for m in msgs]
    noted = [m for m in kept if "notification" in (m["custom_metadata"].get("run_record") or {})]
    assert len(noted) == 1 and noted[0]["role"] == "USER"
    third = await chatted(agent, "and now?")
    assert third.answer == "Nothing new."
    # the earlier notification is in its history, as one; nothing new is told
    assert len(told(model.seen[3])) == 1


async def _busy(*args: Any, **kwargs: Any) -> bool:
    return False


# --------------------------------------------------------------------------- an idle chat
async def test_an_idle_chat_gets_the_result_by_a_turn_of_its_own_on_its_thread_stream(
    harness: Harness, gate: Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task ends while no turn works in the chat: a turn of the chat's own starts in its
    thread — a run of the agent whose input is the notification, which its model answers — and
    the thread's stream (``GET /agui/threads/{thread}/events``) delivers its reply as it
    delivers a scheduled run's. The next turn is not told again."""
    monkeypatch.setattr(threads_module, "POLL_SECONDS", 0.01)

    def reports(messages: list[BaseMessage]) -> str:
        [update] = told(messages)
        return f"The research on lighthouses is done ({update['status']})."

    model = Model(turns=[START, "Started.", reports, "Nothing else."])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    chunks: list[str] = []

    async def follow() -> None:
        async for chunk in Threads(agent).stream(LOCAL_TENANT, USER, THREAD, None):
            chunks.append(chunk)

    following = asyncio.create_task(follow())
    try:
        await until(lambda: chunks)  # the first turn, replayed
        gate.opened.set()
        await until(lambda: any("is done (success)" in c for c in chunks))
    finally:
        following.cancel()
        await asyncio.gather(following, return_exceptions=True)
    runs = [r async for r in harness.runs.iterate(agent_id="chat", thread_id=THREAD)]
    assert len(runs) == 2
    own = await harness.runs.get(runs[0].run_id)
    assert own is not None and own.status is RunStatus.SUCCESS
    assert own.metadata[background.NOTIFIED]["task_id"] == task.thread_id
    assert notified(model.seen[2][-1]) is not None  # its question: the notification
    later = await chatted(agent, "anything else?")
    assert later.answer == "Nothing else." and len(told(model.seen[3])) == 1  # in its history


async def test_an_outcome_a_turn_already_told_is_quiet_when_a_turn_of_its_own_tells_it(
    harness: Harness, gate: Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn of the chat's own for an outcome the conversation's model was told already (the
    task list it carries says so: a server that found the chat idle a moment late) calls no
    model and is not delivered — it answers as a scheduled run with nothing to report does;
    and one outcome starts one such turn, however often it is told."""
    monkeypatch.setattr(background, "idle", _busy)
    model = Model(turns=[START, "Started.", "Back: success."])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    gate.opened.set()
    await until(lambda: _final(harness, task.run_id))
    assert (await chatted(agent, "anything new?")).answer == "Back: success."  # told here
    ended = await harness.runs.get(task.run_id)
    assert ended is not None
    update = background.update_of({"task_id": task.thread_id, "agent_name": "research"}, ended)
    assert update is not None
    calls, origin = len(model.seen), ended.metadata[background.BACKGROUND]
    told = [
        background.notify(agent, ended, update, user=USER, thread=THREAD, origin=origin)
        for _ in range(2)
    ]
    read = harness.runs.get

    async def slow(*args: Any, **kwargs: Any) -> Any:  # a store across the network: both read
        found = await read(*args, **kwargs)
        await asyncio.sleep(0.01)  # before either starts it
        return found

    monkeypatch.setattr(harness.runs, "get", slow)
    await asyncio.gather(*told)  # at once: one starts it, the other finds it at work
    monkeypatch.setattr(harness.runs, "get", read)
    await background.notify(agent, ended, update, user=USER, thread=THREAD, origin=origin)
    await until(lambda: _all_final(harness))
    assert await _own_turns(harness) == 1
    [own] = [
        r
        async for r in harness.runs.iterate(agent_id="chat", thread_id=THREAD)
        if r.run_id not in (first.run_id,)
    ][:1]
    found = await harness.runs.get(own.run_id)
    assert found is not None and background.NOTIFIED in found.metadata
    assert (found.status, found.output) == (RunStatus.SUCCESS, schedules.NO_REPORT)
    assert len(model.seen) == calls  # no model call


async def _own_turns(harness: Harness) -> int:
    runs = [r async for r in harness.runs.iterate(agent_id="chat", thread_id=THREAD)]
    found = [await harness.runs.get(r.run_id) for r in runs]
    return sum(1 for r in found if r is not None and background.NOTIFIED in r.metadata)


async def _told_once(harness: Harness) -> bool:
    """Whether the chat's own turn for the outcome ran, and every turn of the chat ended."""
    return await _own_turns(harness) == 1 and await _all_final(harness)


async def _all_final(harness: Harness) -> bool:
    runs = [r async for r in harness.runs.iterate(agent_id="chat", thread_id=THREAD)]
    return all(r.status.final for r in runs)


# --------------------------------------------------------------------------- the four other tools
_TASK_ID: Final = re.compile(r"task_[0-9a-f]{16,32}")


def with_task(name: str, **args: Any) -> Callable[[list[BaseMessage]], Any]:
    """A call of ``name`` with the last task id the model can read in what it is shown."""

    def turn(messages: list[BaseMessage]) -> Any:
        found = [i for m in messages for i in _TASK_ID.findall(str(m.content))]
        return (name, {"task_id": found[-1], **args}) if found else "No task id."

    return turn


def says_what_the_tool_said(messages: list[BaseMessage]) -> str:
    return str(messages[-1].content)


def reports_the_task(messages: list[BaseMessage]) -> str:
    """A start's answer as a model gives it: what it started, naming the task."""
    return f"Started the research: {str(messages[-1].content).rsplit(' ', 1)[-1]}"


async def test_check_list_and_cancel_reach_the_task(harness: Harness, gate: Gate) -> None:
    model = Model(
        turns=[
            START,
            reports_the_task,
            with_task("check_async_task"),
            says_what_the_tool_said,
            ("list_async_tasks", {}),
            says_what_the_tool_said,
            with_task("cancel_async_task"),
            says_what_the_tool_said,
            ("list_async_tasks", {"status_filter": "running"}),
            says_what_the_tool_said,
        ]
    )
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    checked = await chatted(agent, "how is it going?")
    assert checked.status is RunStatus.SUCCESS, (checked.error, checked.answer)
    assert json.loads(checked.answer) == {"status": "running", "thread_id": task.thread_id}
    listed = await chatted(agent, "which tasks are there?")
    assert listed.answer == (
        f"1 tracked task(s):\n- task_id: {task.thread_id}  agent: research  status: running"
    )
    asked = await chatted(agent, "stop it")  # a cancel stops work: a write, asked in a chat
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.tool == "cancel_async_task"
    cancelled = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer=USER)
    assert cancelled.status is RunStatus.SUCCESS, cancelled.error
    assert cancelled.answer == f"Cancelled async subagent task: {task.thread_id}"
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED
    assert not harness.background.at_work(task.run_id)  # its attempt stopped here too
    none = await chatted(agent, "anything running?")
    assert none.answer == "No async subagent tasks tracked."


async def test_a_check_of_a_finished_task_has_its_result(harness: Harness, gate: Gate) -> None:
    model = Model(turns=[START, reports_the_task])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    gate.opened.set()
    await until(lambda: _final(harness, task.run_id))
    await until(lambda: _all_final(harness))  # the idle chat was told
    model.turns += [with_task("check_async_task"), says_what_the_tool_said]
    checked = await chatted(agent, "what did it find?")
    assert json.loads(checked.answer) == {
        "status": "success",
        "thread_id": task.thread_id,
        "result": "FOUND: lighthouses",
    }


@pytest.mark.parametrize("strategy", ["interrupt", "enqueue"])
async def test_an_update_sends_the_task_new_instructions_as_its_update_strategy_says(
    harness: Harness, strategy: str
) -> None:
    """``update_async_task`` follows Deep Agents' ``update_strategy``: a follow-up is a new run
    of the task's own conversation, under the same task id — ``interrupt`` (the default)
    cancels the run at work first (``conversation.Mode.INTERRUPT``), ``enqueue`` waits for it
    (``Mode.QUEUE``)."""
    release = asyncio.Event()
    heard: list[str] = []

    async def writer(input: str, agent: Runtime) -> str:
        """Write on a topic."""
        heard.append(input)
        if input == "lighthouses":
            await release.wait()
        return f"WROTE: {input}"

    model = Model(
        turns=[
            ("start_async_task", {"description": "lighthouses", "subagent_type": "writer"}),
            reports_the_task,
            with_task("update_async_task", message="shorter, please"),
            says_what_the_tool_said,
        ]
    )
    wrapped = harness.wrap(writer, id="writer")
    agent = harness.wrap(ReAct("You help.", model, subagents=[wrapped]), id="chat")
    if strategy == "enqueue":  # a wrapped agent's tasks: the strategy its ReAct gives them
        _found_in(agent.target).offered.strategies["writer"] = "enqueue"
    first = await chatted(agent, "write about lighthouses")
    task = await task_run(harness, first.run_id)
    await until(lambda: heard == ["lighthouses"])
    updated = await chatted(agent, "make it shorter")
    assert updated.status is RunStatus.SUCCESS, updated.error
    [said] = results(model.seen[3])
    if strategy == "interrupt":
        assert said == f"Updated async subagent. task_id: {task.thread_id}"
        await until(lambda: heard == ["lighthouses", "shorter, please"])
        stopped = await harness.runs.get(task.run_id)
        assert stopped is not None and stopped.status is RunStatus.CANCELLED
    else:
        assert said == (
            "Queued follow-up to async subagent without interrupting its current run. "
            f"task_id: {task.thread_id}"
        )
        await asyncio.sleep(0.2)
        assert heard == ["lighthouses"]  # it waits for the run at work
        release.set()
        await until(lambda: heard == ["lighthouses", "shorter, please"])
        done = await harness.runs.get(task.run_id)
        assert done is not None and done.status is RunStatus.SUCCESS
    follow = [r async for r in harness.runs.iterate(agent_id="writer") if r.run_id != task.run_id]
    [later] = [await harness.runs.get(r.run_id) for r in follow]
    assert later is not None and later.thread_id == task.thread_id  # the task's conversation
    release.set()


def _found_in(graph: Any) -> Any:
    from trellis.harness.bridges.background_tasks import BackgroundTasks

    for node in graph.builder.nodes.values():
        runnable = getattr(node, "runnable", None)
        func = getattr(runnable, "afunc", None) or getattr(runnable, "func", None)
        owner = getattr(func, "__self__", None)
        if isinstance(owner, BackgroundTasks):
            return owner
    raise AssertionError("no BackgroundTasks in the graph")


# --------------------------------------------------------------------------- with the turn
async def test_cancelling_the_turn_cancels_its_tasks(harness: Harness, gate: Gate) -> None:
    model = Model(turns=[START, "never said"], hold_from=1)
    agent = chat(harness, gate, model)
    turn = asyncio.create_task(chatted(agent, "research lighthouses"))
    await until(lambda: gate.started)  # the task works; the turn's next model call hangs
    [run_id] = agent.running
    task = await task_run(harness, run_id)
    await agent.cancel(run_id, reason="the user stopped the turn")
    assert (await turn).status is RunStatus.CANCELLED
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED
    await until(lambda: not harness.background.at_work(task.run_id))  # stopped here too


async def test_two_tasks_at_once(harness: Harness, gate: Gate) -> None:
    both = [
        ("start_async_task", {"description": "lighthouses", "subagent_type": "research"}),
        ("start_async_task", {"description": "tides", "subagent_type": "research"}),
    ]
    model = Model(turns=[both, "Started both.", ("list_async_tasks", {}), says_what_the_tool_said])
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses and tides")
    assert first.answer == "Started both.", first.error
    await until(lambda: sorted(gate.started) == ["lighthouses", "tides"])  # at once
    tasks = [c async for c in harness.runs.iterate(parent_run_id=first.run_id)]
    assert len(tasks) == 2 and all(t.status is RunStatus.RUNNING for t in tasks)
    listed = await chatted(agent, "which tasks?")
    assert listed.answer.startswith("2 tracked task(s):"), listed.answer
    gate.opened.set()
    await until(lambda: _all_ended(harness, [t.run_id for t in tasks]))


async def _all_ended(harness: Harness, run_ids: list[str]) -> bool:
    found = [await harness.runs.get(r) for r in run_ids]
    return all(r is not None and r.final for r in found)


async def test_a_run_nobody_reads_hands_its_sub_agents_work_in_the_foreground(
    harness: Harness, gate: Gate
) -> None:
    """The tools are offered where ``ask_user`` is: a chat a person reads. A run started from
    code (``agent.run``) has nobody to deliver an outcome to: its sub-agents are called in its
    turn (a wrapped agent's own tool), and no task is started."""
    model = Model(turns=[("research", {"message": "lighthouses"}), says_what_the_tool_said])
    agent = chat(harness, gate, model)
    gate.opened.set()
    done = await agent.run("research lighthouses", user=USER, thread=THREAD)
    assert done.answer == "FOUND: lighthouses", done.error
    assert not set(TOOLS) & set(model.requests[0]["tools"])


# --------------------------------------------------------------------------- limits
class Waits(Model):
    """A scripted model whose second call first waits for ``before_second`` (the task to end
    while the turn works: its outcome comes after the turn's final answer)."""

    before_second: Any = None

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        if len(self.seen) == 1 and self.before_second is not None:
            await self.before_second()
        return await super()._agenerate(messages, *args, **kwargs)


async def test_the_extra_model_turn_counts_against_max_steps(harness: Harness, gate: Gate) -> None:
    """An outcome after the final answer gives the model another turn — a model call as any:
    past ``max_steps`` it is not made, and the run ends with the answer it had."""

    async def ended() -> None:
        gate.opened.set()
        await until(lambda: not harness.background._at_work)

    model = Waits(turns=[START, "Started; I will tell you.", "never said"], before_second=ended)
    agent = chat(harness, gate, model, max_steps=1)
    first = await chatted(agent, "research lighthouses")
    assert first.status is RunStatus.SUCCESS, first.error
    assert first.answer == "Started; I will tell you."
    assert not told(model.seen[1])  # its last read before the answer: the task still worked
    assert model.turns == ["never said"]  # no turn was left for it
    # its outcome was not read into a turn the model never got: a turn of the chat's own
    # tells it once the turn ended
    model.turns = ["The research is done."]
    await until(lambda: _told_once(harness))
    assert len(told(model.seen[2])) == 1


async def test_the_extra_model_turn_tells_the_outcome_within_max_steps(
    harness: Harness, gate: Gate
) -> None:
    """Within ``max_steps``, an outcome after the final answer gives the model another turn,
    which tells it."""

    async def ended() -> None:
        gate.opened.set()
        await until(lambda: not harness.background._at_work)

    model = Waits(turns=[START, "Started.", says_what_the_tool_said], before_second=ended)
    agent = chat(harness, gate, model)
    first = await chatted(agent, "research lighthouses")
    assert first.status is RunStatus.SUCCESS, first.error
    assert "FOUND: lighthouses" in first.answer and len(told(model.seen[2])) == 1


async def test_a_task_keeps_to_its_own_sub_agents_max_steps(harness: Harness) -> None:
    """The task is a run of its sub-agent, with its sub-agent's limits: a ``ReAct`` sub-agent
    with ``max_steps=1`` answers after one step, and that answer is the task's result."""

    @tool(side_effects="read")
    def look(topic: str) -> str:
        """Look a topic up."""
        return f"seen {topic}"

    inner = Model(turns=[("look", {"topic": "a"}), "AT THE LIMIT", ("look", {"topic": "b"})])
    research = harness.wrap(ReAct("You research.", inner, tools=[look], max_steps=1), id="research")
    model = Model(turns=[START, "Started.", "told"])
    agent = harness.wrap(ReAct("You help.", model, subagents=[research]), id="chat")
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    await until(lambda: _final(harness, task.run_id))
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and (ended.status, ended.output) == (RunStatus.SUCCESS, "AT THE LIMIT")
    assert inner.turns == [("look", {"topic": "b"})]  # never a second step


# --------------------------------------------------------------------------- Deep Agents' own
async def test_a_deep_agents_sub_agent_runs_in_the_background_too(harness: Harness) -> None:
    """A ``SubAgent`` dict runs as its compiled graph — the graph Deep Agents' ``task`` tool
    runs — in a run of its own, a child of the turn (its agent: the parent's id and its
    name)."""
    notes = {
        "name": "notes",
        "description": "Takes notes on a topic.",
        "system_prompt": "You take notes.",
        "model": Model(turns=["NOTES: lighthouses"]),
        "tools": [],
    }
    model = Model(
        turns=[
            ("start_async_task", {"description": "lighthouses", "subagent_type": "notes"}),
            "Started.",
            "told",
        ]
    )
    agent = harness.wrap(ReAct("You help.", model, subagents=[notes]), id="chat")
    first = await chatted(agent, "take notes on lighthouses")
    assert first.status is RunStatus.SUCCESS, first.error
    task = await task_run(harness, first.run_id)
    assert task.agent_id == "chat.notes"
    await until(lambda: _final(harness, task.run_id))
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and (ended.status, ended.output) == (
        RunStatus.SUCCESS,
        "NOTES: lighthouses",
    )


# --------------------------------------------------------------------------- a task that asks
@tool(side_effects="irreversible")
def publish(topic: str) -> str:
    """Publish the report on a topic (asks a person first)."""
    return f"published {topic}"


async def publisher(input: str, agent: Runtime) -> str:
    """Publish a report on a topic."""
    said = await agent.tools.call(publish, topic=input)
    return f"DONE: {said}"


def publishing(harness: Harness, model: Model) -> Agent:
    wrapped = arm(harness.wrap(publisher, id="publisher"), [publish])
    return harness.wrap(ReAct("You help.", model, subagents=[wrapped]), id="chat")


PUBLISH: Final = ("start_async_task", {"description": "tides", "subagent_type": "publisher"})


async def asked_in_the_chat(harness: Harness) -> Any:
    """The turn of the chat's own that asks the question the task waits on: paused on it."""

    async def paused() -> Any:
        runs = [r async for r in harness.runs.iterate(agent_id="chat", thread_id=THREAD)]
        found = [r for r in runs if r.status is RunStatus.PAUSED]
        return found[0] if found else None

    await until(paused)
    found = await paused()
    record = await harness.runs.get(found.run_id)
    assert record is not None and background.NOTIFIED in record.metadata
    return record


async def test_an_approval_inside_the_task_is_asked_through_the_chat(
    harness: Harness,
) -> None:
    """The task's call that asks a person pauses its run (``waiting``); the chat, idle, asks
    it in a turn of its own — as a sub-agent's question is asked through its parent, the
    interrupt naming the task — and the answer resumes the task, whose result then comes back
    to the chat."""
    # how many model turns tell the two outcomes depends on when the task ends: enough of them
    model = Model(turns=[PUBLISH, "Started.", *["Noted."] * 6])
    agent = publishing(harness, model)
    first = await chatted(agent, "publish the tides report")
    task = await task_run(harness, first.run_id)
    asking = await asked_in_the_chat(harness)
    waiting = await harness.runs.get(task.run_id)
    assert waiting is not None and waiting.status is RunStatus.PAUSED
    asked = asking.awaiting
    assert asked.tool_call is not None and asked.tool_call.tool == "publish"
    named = asked.payload["subagent"]
    assert (named["run_id"], named["background"]) == (task.run_id, True)
    done = await agent.resume(asked.interrupt_id, "approve", reviewer=USER)
    assert done.status is RunStatus.SUCCESS, done.error
    await until(lambda: _final(harness, task.run_id))
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.output == "DONE: published tides"
    await until(lambda: _all_final(harness))
    # the model was told both outcomes: the question first, then the result
    statuses = [u["status"] for u in told(model.seen[-1])]
    assert statuses == ["waiting", "success"]


async def test_an_answer_reaching_another_worker_resumes_the_task_there(
    workers: tuple[Harness, Harness],
) -> None:
    """The task's question is answered on another server sharing the run store: the task
    resumes there, held by it, and its result comes back to the chat from there."""
    first_h, second_h = workers
    # how many model turns tell the two outcomes depends on when the task ends: enough of them
    model = Model(turns=[PUBLISH, "Started.", *["Noted."] * 6])
    here, there = (publishing(h, model) for h in workers)
    first = await chatted(here, "publish the tides report")
    task = await task_run(first_h, first.run_id)
    asking = await asked_in_the_chat(first_h)
    await until(lambda: not first_h.background._at_work)  # nothing at work on the first now
    done = await there.resume(asking.awaiting.interrupt_id, "approve", reviewer=USER)
    assert done.status is RunStatus.SUCCESS, done.error
    await until(lambda: _final(second_h, task.run_id))
    ended = await second_h.runs.get(task.run_id)
    assert ended is not None and ended.output == "DONE: published tides"
    assert not first_h.background._at_work


@pytest.fixture
async def workers() -> AsyncIterator[tuple[Harness, Harness]]:
    """Two servers: two harnesses sharing the run store."""
    store = LocalRuns()
    pair = (Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store))
    yield pair
    for h in pair:
        await h.aclose()


async def test_the_harness_closing_cuts_a_task_at_work_off(gate: Gate) -> None:
    """The harness closes while a task works here: its run ends ``ERROR``, cut off, saying so —
    never left ``RUNNING`` (a killed server's lapses: ``test_killed_server``)."""
    harness = Harness(config=Settings())
    agent = chat(harness, gate, Model(turns=[START, "Started."]))
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    await until(lambda: gate.started)
    await harness.aclose()
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.ERROR and ended.error is not None
    assert (ended.error.code, ended.error.message) == (
        background.UNFINISHED_CODE,
        background.CUT_OFF,
    )


# --------------------------------------------------------------------------- more of its life
async def test_a_turn_out_of_time_leaves_its_tasks_working(harness: Harness, gate: Gate) -> None:
    """A turn's clock is not its tasks': a turn that runs out of time ends the children in its
    call, not the tasks it started in the background."""
    model = Model(turns=[START, "never said"], hold_from=1)
    research = harness.wrap(gate.researcher, id="research")
    target = ReAct("You help.", model, subagents=[research])
    agent = harness.wrap(target, id="chat", run_timeout=0.5)
    turn = await chatted(agent, "research lighthouses")
    assert turn.status is RunStatus.TIMEOUT
    task = await task_run(harness, turn.run_id)
    assert task.status is RunStatus.RUNNING
    gate.opened.set()
    await until(lambda: _final(harness, task.run_id))


async def test_a_tasks_run_ended_elsewhere_stops_its_attempt_here(
    harness: Harness, gate: Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task's run cancelled where its attempt is not (another replica's cancel) is seen
    here within a read (``background.WATCH_SECONDS``): its attempt stops."""
    monkeypatch.setattr(background, "WATCH_SECONDS", 0.05)
    agent = chat(harness, gate, Model(turns=[START, "Started."]))
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    await until(lambda: gate.started)
    await asyncio.sleep(0.2)  # a read or two while it works
    await harness.runs.cancel(task.run_id, reason="on another replica")
    await until(lambda: not harness.background.at_work(task.run_id))


async def test_cancel_as_the_answer_to_the_tasks_question_cancels_it(harness: Harness) -> None:
    model = Model(turns=[PUBLISH, "Started.", *["Noted."] * 4])
    agent = publishing(harness, model)
    first = await chatted(agent, "publish the tides report")
    task = await task_run(harness, first.run_id)
    asking = await asked_in_the_chat(harness)
    done = await agent.resume(asking.awaiting.interrupt_id, "cancel", reviewer=USER)
    assert done.status is RunStatus.CANCELLED
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED


async def test_new_instructions_from_another_server_replace_the_tasks_run_there(
    workers: tuple[Harness, Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``update_async_task`` (``interrupt``) on a server where the task's run is not at work:
    that run is cancelled in the run store — the server it works on stops it at its next read
    — and the follow-up goes on as the task, here."""
    monkeypatch.setattr(background, "WATCH_SECONDS", 0.05)
    first_h, second_h = workers
    gates = (Gate(), Gate())
    turns = [START, reports_the_task, with_task("update_async_task", message="tides"), "Sent."]
    model = Model(turns=turns)
    here, there = (chat(h, g, model) for h, g in zip(workers, gates, strict=True))
    first = await chatted(here, "research lighthouses")
    task = await task_run(first_h, first.run_id)
    await until(lambda: gates[0].started)
    updated = await chatted(there, "research tides instead")
    assert updated.status is RunStatus.SUCCESS, updated.error
    stopped = await second_h.runs.get(task.run_id)
    assert stopped is not None and stopped.status is RunStatus.CANCELLED
    await until(lambda: not first_h.background._at_work)  # stopped where it worked
    await until(lambda: gates[1].started == ["tides"])  # the follow-up, on this server
    for g in gates:
        g.opened.set()


async def _paused(harness: Harness, run_id: str) -> bool:
    found = await harness.runs.get(run_id)
    return found is not None and found.status is RunStatus.PAUSED


async def test_what_breaks_the_harness_around_a_task_is_logged_not_raised(
    harness: Harness, gate: Gate, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the harness broke")

    gate.opened.set()
    agent = chat(harness, gate, Model(turns=[START, "Started.", START, "Again."]))
    monkeypatch.setattr(background, "delivered", broken)
    first = await chatted(agent, "research lighthouses")
    task = await task_run(harness, first.run_id)
    await until(lambda: _final(harness, task.run_id))
    await until(lambda: not harness.background._at_work)
    assert "its outcome not delivered" in caplog.text
    monkeypatch.setattr(pipeline, "attempt", broken)
    record = await harness.runs.get(task.run_id)
    assert record is not None
    harness.background.run(agent, record)
    await until(lambda: not harness.background._at_work)
    assert f"background task {task.run_id}" in caplog.text


async def test_two_tasks_of_one_deep_agents_sub_agent_run_as_one_agent(harness: Harness) -> None:
    notes = {
        "name": "notes",
        "description": "Takes notes on a topic.",
        "system_prompt": "You take notes.",
        "model": Model(turns=["NOTES: a", "NOTES: b"]),
        "tools": [],
    }
    both = [
        ("start_async_task", {"description": "a", "subagent_type": "notes"}),
        ("start_async_task", {"description": "b", "subagent_type": "notes"}),
    ]
    agent = harness.wrap(
        ReAct("You help.", Model(turns=[both, "Started.", *["ok"] * 4]), subagents=[notes]),
        id="chat",
    )
    first = await chatted(agent, "take notes")
    tasks = [c async for c in harness.runs.iterate(parent_run_id=first.run_id)]
    assert {t.agent_id for t in tasks} == {"chat.notes"} and len(tasks) == 2
    await until(lambda: _all_ended(harness, [t.run_id for t in tasks]))


async def test_outside_a_harness_run_the_tasks_middleware_steps_aside(gate: Gate) -> None:
    from trellis.harness.bridges.background_tasks import BackgroundTasks

    tasks = BackgroundTasks({"research": None}, described={"research": "Research."})
    assert await tasks.abefore_agent({"messages": []}, None) is None
    assert await tasks.abefore_model({"messages": []}, None) is None
    assert await tasks.aafter_model({"messages": []}, None) is None


def test_an_update_strategy_is_deep_agents_own() -> None:
    queued = {"name": "notes", "description": "Notes.", "update_strategy": "enqueue"}
    graph = ReAct("You help.", Model(turns=[]), subagents=[queued])
    assert _found_in(graph).offered.strategies == {"notes": "enqueue"}
    assert "update_strategy" not in queued  # Deep Agents is given its own spec
    spec = {"name": "notes", "description": "Notes.", "update_strategy": "later"}
    with pytest.raises(ConfigurationError, match="notes: update_strategy is 'interrupt' or"):
        ReAct("You help.", Model(turns=[]), subagents=[spec])


async def test_cancelling_a_paused_parent_with_no_framework_work_cancels_its_child(
    harness: Harness,
) -> None:
    """``subagents.cancel_children`` on an agent whose framework keeps nothing going for a
    child (a function's: no ``child_cancelled``): the child's run is cancelled all the same."""

    async def asks(input: str, agent: Runtime) -> str:
        """Ask which."""
        return str(await agent.ask("Which one?", options=["a", "b"]))

    child = harness.wrap(asks, id="asker")

    async def boss(input: str, agent: Runtime) -> str:
        return str((await child.run("which?", user=agent.user)).answer)

    parent = harness.wrap(boss, id="boss")
    paused = await parent.run("go", user=USER)
    assert paused.status is RunStatus.PAUSED
    await parent.cancel(paused.run_id, reason="enough")
    [kid] = [c async for c in harness.runs.iterate(parent_run_id=paused.run_id)]
    assert kid.status is RunStatus.CANCELLED


async def test_an_answer_handed_to_a_task_no_longer_waiting_on_it_is_left(harness: Harness) -> None:
    """The task's question answered twice (asked in two turns), or cancelled by its answer:
    the second answer, and a cancel, start no attempt of it."""
    model = Model(turns=[PUBLISH, "Started.", *["Noted."] * 4])
    agent = publishing(harness, model)
    first = await chatted(agent, "publish the tides report")
    task = await task_run(harness, first.run_id)
    asking = await asked_in_the_chat(harness)
    waiting = await harness.runs.get(task.run_id)
    assert waiting is not None and waiting.awaiting is not None
    publisher = harness.agents["publisher"]
    cancel = InterruptResolution(
        interrupt_id=waiting.awaiting.interrupt_id,
        run_id=task.run_id,
        decision=InterruptDecision.CANCEL,
        reviewer=USER,
    )
    await background.answered(agent, publisher, waiting, cancel)
    ended = await harness.runs.get(task.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED
    assert not harness.background.at_work(task.run_id)
    await background.answered(agent, publisher, waiting, cancel)  # waits on nothing now
    assert not harness.background.at_work(task.run_id)
    del asking
