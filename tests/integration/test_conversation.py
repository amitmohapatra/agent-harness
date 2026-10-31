"""A conversation across runs, on every adapter (``BUILDERS``): a follow-up has the earlier
turns with memory on (memory's window, or the framework's session) and off (the run store's
questions and answers, or the session the framework kept); a message sent to a busy
conversation waits (``queue``), cancels the working run and continues from it
(``interrupt``), or is given to it at its next step (``steer``, where the target can take one —
else it waits, said once); every message keeps one id, the one the stream says and memory
keeps, across a resume."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import conversation
from trellis.harness.bounded import LRU
from trellis.harness.conversation import (
    Mode,
    Turns,
    answer_id,
    earlier,
    identified,
    message_id,
    question_id,
    rendered,
    result_id,
)
from trellis.harness.history import Called, Said, Speaker, recorded, said_of
from trellis.harness.identity import conversation_key

THREAD = "chat-7"
#: the targets whose model a test reads (a function has none: its own test)
MODELLED = sorted(set(BUILDERS) - {"function"})


def forget(seen: list[Any]) -> None:
    """What the model was sent so far, forgotten: a test reads one turn's."""
    for model in seen:
        for name in ("seen", "inputs", "requests"):
            kept = getattr(model, name, None)
            if isinstance(kept, list):
                kept.clear()
        if isinstance(model, Path):
            model.unlink(missing_ok=True)
            Path(f"{model}.added").unlink(missing_ok=True)


def told(seen: list[Any]) -> str:
    """Everything the model was sent since :func:`forget` — for Claude, what its CLI was
    started with (the session it resumed and that session's earlier prompts, the context) and
    what the hooks added after its calls."""
    model = seen[-1]
    if isinstance(model, Path):
        record = json.loads(model.read_text()) if model.exists() else {}
        added = Path(f"{model}.added")
        return json.dumps([record, added.read_text() if added.exists() else ""])
    if hasattr(model, "said"):
        return model.said()
    return "\n".join(str(getattr(m, "content", m)) for call in model.seen for m in call)


def requests(seen: list[Any]) -> list[str]:
    """Each model call's input, as text (Claude: what its CLI was started with last)."""
    model = seen[-1]
    if isinstance(model, Path):
        return [told(seen)]
    for name in ("seen", "inputs", "requests"):  # a chat model's ``requests`` are its kwargs
        kept = getattr(model, name, None)
        if isinstance(kept, list):
            return [json.dumps(r, default=str) for r in kept]
    raise AssertionError(f"no requests kept by {model!r}")


# --------------------------------------------------------------------------- follow-ups
@pytest.mark.parametrize("memory", ["on", "off"])
@pytest.mark.parametrize("framework", MODELLED)
async def test_a_follow_up_has_the_conversation(
    memory_harness: Harness, tmp_path: Path, framework: str, memory: str
) -> None:
    seen: list[Any] = []
    target, tools = await BUILDERS[framework](memory_harness, [], tmp_path, [], seen=seen)
    without = {"memory"} if memory == "off" else set()
    agent = arm(memory_harness.wrap(target, id=f"follow-{framework}", without=without), tools)
    first = await agent.run("The locker code is 4242.", user="ada", thread=THREAD)
    assert first.status is RunStatus.SUCCESS, first.error
    await memory_harness.writes.drain()
    forget(seen)
    second = await agent.run("What is the locker code?", user="ada", thread=THREAD)
    assert second.status is RunStatus.SUCCESS, second.error
    assert "4242" in told(seen)  # the earlier turn reached the model
    if framework == "claude_agent_sdk":  # the session the first turn ran in, continued
        record = json.loads(seen[-1].read_text())
        assert record["resume"] and record["earlier"] == ["The locker code is 4242."]
    forget(seen)
    other = await agent.run("What is the locker code?", user="bob", thread=THREAD)
    assert other.status is RunStatus.SUCCESS, other.error
    assert "4242" not in told(seen)  # another user's thread named alike is not this one


async def test_a_function_reads_the_earlier_turns_as_its_history(harness: Harness) -> None:
    given: list[tuple[list[Said] | None, str | None]] = []

    async def answer(input: str, agent: Runtime) -> str:
        given.append((agent.history, agent.context))
        return f"noted: {input}"

    agent = harness.wrap(answer, id="fn-history")
    await agent.run("The locker code is 4242.", user="ada", thread=THREAD)
    await agent.run("What is the locker code?", user="ada", thread=THREAD)
    await agent.run("alone", user="ada")  # no conversation: no history
    (first, _), (second, context), (alone, _) = given
    assert first is None and alone is None and context is None
    assert second is not None
    assert [(s.role, s.content) for s in second] == [
        (Speaker.USER, "The locker code is 4242."),
        (Speaker.ASSISTANT, "noted: The locker code is 4242."),
    ]
    key = conversation_key("default", "ada", THREAD)
    one = runs_in_order(harness)[0]
    assert [s.id for s in second] == [question_id(key, one), answer_id(key, one)]


def runs_in_order(harness: Harness) -> list[str]:
    """The ids of the runs the in-process store keeps, oldest first."""
    runs = harness.runs._runs.values()  # type: ignore[attr-defined]
    return [r.run_id for r in sorted(runs, key=lambda r: r.created_at)]


async def test_the_earlier_turns_are_the_run_stores_answered_ones_in_order(
    harness: Harness,
) -> None:
    seen: list[list[Said]] = []

    async def answer(input: str, agent: Runtime) -> str:
        if input == "fail":
            raise RuntimeError("no")
        if input == "three":
            seen.append(await earlier(agent))
        return f"re: {input}"

    agent = harness.wrap(answer, id="fn-turns")
    for text in ("one", "fail", "two"):
        await agent.run(text, user="ada", thread=THREAD)
    await agent.run("other user", user="bob", thread=THREAD)
    await agent.run("other thread", user="ada", thread="elsewhere")
    await agent.run("three", user="ada", thread=THREAD)
    runs = runs_in_order(harness)
    one, two = runs[0], runs[2]
    key = conversation_key("default", "ada", THREAD)
    assert [(s.role.value, s.content, s.id) for s in seen[0]] == [
        ("user", "one", question_id(key, one)),
        ("assistant", "re: one", answer_id(key, one)),
        ("user", "two", question_id(key, two)),
        ("assistant", "re: two", answer_id(key, two)),
    ]  # a failed turn, another user's, another thread's and the run itself left out
    assert rendered(seen[0]).splitlines()[1:] == [
        "user: one",
        "assistant: re: one",
        "user: two",
        "assistant: re: two",
    ]


async def test_no_earlier_turns_outside_a_conversation_or_inside_a_child() -> None:
    from types import SimpleNamespace

    alone = SimpleNamespace(parent=None, thread="run_1", run_id="run_1")
    child = SimpleNamespace(parent=object(), thread="t", run_id="run_2")
    assert await earlier(alone) == [] and await earlier(child) == []  # type: ignore[arg-type]


async def test_a_framework_keeping_its_own_thread_gets_no_earlier_turns(harness: Harness) -> None:
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    from tests.support.chat_model import ScriptedChatModel

    model = ScriptedChatModel(turns=["noted", "4242"])
    agent = harness.wrap(create_agent(model, tools=[], checkpointer=InMemorySaver()), id="own")
    await agent.run("The locker code is 4242.", user="ada", thread=THREAD)
    await agent.run("What is the locker code?", user="ada", thread=THREAD)
    # the checkpointer replays the thread once; the run store's turns are not added beside it
    said = [str(m.content) for m in model.seen[1]]
    assert said.count("The locker code is 4242.") == 1


async def test_a_failing_run_store_read_is_a_warning(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(runtime: Any) -> list[Said]:
        raise RuntimeError("the store is down")

    monkeypatch.setattr(conversation, "earlier", broken)
    events = []

    async def answer(input: str, agent: Runtime) -> str:
        return "ok"

    agent = harness.wrap(answer, id="fn-down")
    async for event in agent.stream("hi", user="ada", thread=THREAD):
        events.append(event)
    warned = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any(w.get("code") == "earlier_turns" for w in warned)
    assert events[-1].type is RunEventType.RUN_FINISHED


# --------------------------------------------------------------------------- the sessions
@pytest.mark.parametrize("framework", ["claude_agent_sdk", "kimi"])
async def test_the_conversations_session_is_continued_only_where_it_is_kept(
    harness: Harness, tmp_path: Path, framework: str
) -> None:
    seen: list[Any] = []
    target, tools = await BUILDERS[framework](harness, [], tmp_path, [], seen=seen)
    agent = arm(harness.wrap(target, id=f"sess-{framework}"), tools)
    await agent.run("The locker code is 4242.", user="ada", thread=THREAD)
    key = conversation_key("default", "ada", THREAD)
    session, _ = agent.sessions[key]
    forget(seen)
    await agent.run("What is the locker code?", user="ada", thread=THREAD)
    assert agent.sessions[key][0] == session  # the same session, one turn on
    assert "4242" in told(seen)
    # another process holds no session (its agent kept none): the run store's turns instead
    agent.sessions = LRU()
    forget(seen)
    await agent.run("And again?", user="ada", thread=THREAD)
    assert "4242" in told(seen)
    if framework == "claude_agent_sdk":
        record = json.loads(seen[-1].read_text())
        assert record["resume"] is None and "Earlier in this conversation" in record["context"]
    await agent.run("no thread", user="ada")  # a run of its own keeps no session for later
    assert list(agent.sessions) == [key]


@pytest.mark.parametrize("framework", ["claude_agent_sdk", "kimi"])
async def test_a_continued_session_with_memory_on_gets_no_memory_window_beside_it(
    memory_harness: Harness, tmp_path: Path, framework: str
) -> None:
    """The session the framework continues replays the earlier turns; memory's window of the
    same turns is not put into the context beside it (ADR 0005 §4: one owner of the history).
    Where no session is kept (another process), the window is the history again."""
    seen: list[Any] = []
    target, tools = await BUILDERS[framework](memory_harness, [], tmp_path, [], seen=seen)
    agent = arm(memory_harness.wrap(target, id=f"sess-mem-{framework}"), tools)
    earlier_turn = "The locker code is 4242."
    await agent.run(earlier_turn, user="ada", thread=THREAD)
    await memory_harness.writes.drain()
    forget(seen)
    second = await agent.run("What is the locker code?", user="ada", thread=THREAD)
    assert second.status is RunStatus.SUCCESS, second.error
    # the continued session carries the earlier turn once; the window adds no second copy
    assert told(seen).count(earlier_turn) == 1
    await memory_harness.writes.drain()
    agent.sessions = LRU()  # another process: no session kept, memory's window instead
    forget(seen)
    third = await agent.run("And again?", user="ada", thread=THREAD)
    assert third.status is RunStatus.SUCCESS, third.error
    assert told(seen).count(earlier_turn) == 1


# --------------------------------------------------------------------------- busy conversations
@pytest.mark.parametrize("mode", [m.value for m in Mode])
@pytest.mark.parametrize("framework", sorted(BUILDERS))
async def test_a_message_to_a_busy_conversation(
    harness: Harness,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    framework: str,
    mode: str,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    held: list[str] = []

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        held.append("started")
        started.set()
        await release.wait()
        return "held"

    seen: list[Any] = []
    plan = [("hold", {})]
    target, tools = await BUILDERS[framework](harness, [hold], tmp_path, plan, seen=seen)
    agent = arm(harness.wrap(target, id=f"busy-{framework}"), tools)
    first = asyncio.create_task(agent.run("Look it up.", user="ada", thread=THREAD))
    await asyncio.wait_for(started.wait(), 60)
    second = asyncio.create_task(agent.run("Also say hello.", user="ada", thread=THREAD, mode=mode))
    await asyncio.sleep(0.3)  # it arrives while the first works
    if mode == "interrupt":
        a = await asyncio.wait_for(first, 60)
        assert a.status is RunStatus.CANCELLED  # the newer message took over
        release.set()
        b = await asyncio.wait_for(second, 60)
        assert b.status is RunStatus.SUCCESS, b.error
        if seen:  # (a function has no model: its history is its own test's)
            # the conversation went on from the cancelled turn: the newer turn's model call
            # has the cancelled question
            assert any("Look it up." in r and "Also say hello." in r for r in requests(seen))
        return
    assert not second.done()  # it waits, or the first run took it
    release.set()
    a, b = await asyncio.wait_for(asyncio.gather(first, second), 60)
    assert a.status is RunStatus.SUCCESS and b.status is RunStatus.SUCCESS, (a.error, b.error)
    assert b.run_id != a.run_id
    steered = mode == "steer" and agent.adapter.steers(agent.target)
    if steered:
        assert b.answer == a.answer and held == ["started"]  # the first run answered both
        assert "Also say hello." in told(seen)
        record = await harness.runs.get(b.run_id, tenant="default")
        assert record is not None and record.output == a.answer
    else:
        # its own turn, after the first: its model calls the tool again — Kimi's continues
        # the first turn's session, whose history holds the call's result already
        assert held == ["started"] * (1 if framework == "kimi" else 2)
    unsteered = "waits for the working run to end" in caplog.text
    assert unsteered is (mode == "steer" and not steered)


async def test_a_steered_message_is_in_the_runs_transcript_with_its_own_id(
    memory_service: FakeMemoryService,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        started.set()
        await release.wait()
        return "held"

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        model = PlannedChat([("hold", {})])
        agent = arm(h.wrap(ReAct(system_prompt="You work.", model=model), id="steered"), [hold])
        first = asyncio.create_task(agent.run("Look it up.", user="ada", thread=THREAD))
        await asyncio.wait_for(started.wait(), 30)
        events: list[Any] = []

        async def follow() -> None:
            async for event in agent.stream(
                "Also say hello.", user="ada", thread=THREAD, mode="steer"
            ):
                events.append(event)

        second = asyncio.create_task(follow())
        await asyncio.sleep(0.2)
        release.set()
        a = await first
        await second
        await h.writes.drain()
    key = conversation_key("acme", "ada", THREAD)
    b_run = events[0].run_id
    stored = memory_service.threads[key]
    ids = {m["content"]: (m.get("custom_metadata") or {}).get("message_id") for m in stored}
    assert ids["Also say hello."] == question_id(key, b_run)  # steered: its own run's id
    assert ids["Look it up."] == question_id(key, a.run_id)
    # the steered run streams the first run's answer under that answer's id, and ends
    texts = [e for e in events if e.type is RunEventType.TEXT_MESSAGE_START]
    assert [e.message_id for e in texts] == [answer_id(key, a.run_id)]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].data["result"] == a.answer


async def test_a_steer_the_working_run_never_takes_runs_as_its_own_turn(
    harness: Harness,
) -> None:
    model = PlannedChat([], stall=0.5)  # its one model call: under way as the message comes
    agent = harness.wrap(ReAct(system_prompt="You work.", model=model), id="late")
    first = asyncio.create_task(agent.run("Look it up.", user="ada", thread=THREAD))
    await asyncio.sleep(0.2)
    b = await agent.run("Also say hello.", user="ada", thread=THREAD, mode="steer")
    a = await first
    assert a.status is RunStatus.SUCCESS and b.status is RunStatus.SUCCESS
    # no model call after the message came: the first run did not take it, so it ran itself,
    # its question the last the model was asked, the earlier turn before it
    said = [m["content"] for m in model.requests[-1]["messages"] if m["role"] == "user"]
    assert said == ["Look it up.", "Also say hello."]
    assert len(model.requests) == 2


async def test_a_steered_message_is_in_the_follow_ups_history_before_the_answer_it_got(
    harness: Harness,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        started.set()
        await release.wait()
        return "held"

    model = PlannedChat([("hold", {})])
    agent = arm(harness.wrap(ReAct(system_prompt="You work.", model=model), id="told"), [hold])
    first = asyncio.create_task(agent.run("Look it up.", user="ada", thread=THREAD))
    await asyncio.wait_for(started.wait(), 30)
    second = asyncio.create_task(agent.run("Also hi.", user="ada", thread=THREAD, mode="steer"))
    await asyncio.sleep(0.2)
    release.set()
    a, b = await asyncio.gather(first, second)
    assert b.answer == a.answer
    seen: list[list[Said]] = []

    async def look(input: str, agent: Runtime) -> str:
        seen.append(await earlier(agent))
        return "ok"

    looking = harness.wrap(look, id="told-look")
    looking.id = agent.id  # read the same agent's turns
    await looking.run("And now?", user="ada", thread=THREAD)
    assert [(s.role.value, s.content) for s in seen[0]] == [
        ("user", "Look it up."),
        ("user", "Also hi."),
        ("assistant", a.answer),
    ]  # the steered message once, before the answer that answered both


async def test_a_steered_runs_structured_answer_is_its_result_without_text(
    harness: Harness,
) -> None:
    from trellis.contracts import RunOutcome
    from trellis.harness import pipeline
    from trellis.harness.result import Result

    async def answer(input: str, agent: Runtime) -> str:
        return "unused"

    agent = harness.wrap(answer, id="structured")
    record = await agent._opened("q", user="ada", thread=THREAD, tenant=None)
    events: list[Any] = []
    answered = Result(run_id="run_x", status=RunStatus.SUCCESS, answer={"total": 3})
    result = await pipeline._answered_by(agent, record, answered, events.append)
    assert result.answer == {"total": 3} and result.run_id == record.run_id
    kinds = [e.type for e in events]
    assert RunEventType.TEXT_MESSAGE_START not in kinds
    assert events[-1].outcome is RunOutcome.SUCCESS


async def test_turns_give_each_message_to_the_run_holding_the_turn_once() -> None:
    turns = Turns()
    async with turns.taken("c", "run_1", steers=True) as live:
        assert live is not None and turns.live("c") is live
        waiting = asyncio.create_task(live.steered("hello", "run_2"))
        await asyncio.sleep(0)
        assert turns.steered("c", "run_other") == []  # not the run holding the turn
        [taken] = turns.steered("c", "run_1")
        assert (taken.text, taken.run_id, taken.taken) == ("hello", "run_2", True)
        assert turns.steered("c", "run_1") == []  # once
    assert await waiting is True and turns.live("c") is None
    assert turns.steered("c", "run_1") == []
    async with turns.taken("c") as nobody:  # a turn without a run (as before)
        assert nobody is None


# --------------------------------------------------------------------------- message ids
async def test_every_message_keeps_one_id_on_the_stream_and_in_memory_across_a_resume(
    memory_service: FakeMemoryService,
) -> None:
    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return f"refunded {order}"

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        model = PlannedChat([("refund", {"order": "o-1"})])
        agent = arm(h.wrap(ReAct(system_prompt="You work.", model=model), id="ids"), [refund])
        events = [e async for e in agent.stream("Refund o-1.", user="ada", thread=THREAD)]
        paused = events[-1]
        assert paused.outcome is not None and paused.outcome.value == "interrupt"
        interrupt = paused.data["interrupt"]["interrupt_id"]
        done = await agent.resume(interrupt, "approve", reviewer="lead")
        assert done.status is RunStatus.SUCCESS
        await h.writes.drain()
    key = conversation_key("acme", "ada", THREAD)
    run = done.run_id
    stored = memory_service.threads[key]
    meta = [m.get("custom_metadata") or {} for m in stored]
    assert all(m["run_id"] == run for m in meta)  # each linked to its run
    questions = {m["message_id"] for s, m in zip(stored, meta, strict=True) if s["role"] == "USER"}
    assert questions == {question_id(key, run)}  # the same question in both attempts
    results = [(s, m) for s, m in zip(stored, meta, strict=True) if s["role"] == "TOOL"]
    assert results
    for said, m in results:
        assert m["message_id"] == result_id(key, said["tool_call_id"])
        assert m["tool_call_id"] == said["tool_call_id"]
    answers = [m for s, m in zip(stored, meta, strict=True) if s["role"] == "ASSISTANT"]
    assert answers[-1]["message_id"] == answer_id(key, run)
    calls = [m for s, m in zip(stored, meta, strict=True) if s.get("tool_calls")]
    assert calls and all(m["tool_call_id"] == calls[0]["tool_call_id"] for m in calls)


async def test_the_stream_says_the_ids_memory_keeps(
    memory_service: FakeMemoryService,
) -> None:
    @tool(side_effects="read")
    def lookup(sku: str) -> str:
        """Units of a SKU."""
        return f"{sku}: 3"

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        model = PlannedChat([("lookup", {"sku": "a"})])
        agent = arm(h.wrap(ReAct(system_prompt="You work.", model=model), id="sids"), [lookup])
        events = [e async for e in agent.stream("How many a?", user="ada", thread=THREAD)]
        await h.writes.drain()
    key = conversation_key("acme", "ada", THREAD)
    run = events[0].run_id
    text = {e.message_id for e in events if e.type is RunEventType.TEXT_MESSAGE_START}
    result = [e for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert text == {answer_id(key, run)}
    assert [e.message_id for e in result] == [result_id(key, str(e.tool_call_id)) for e in result]
    stored = {
        (m.get("custom_metadata") or {}).get("message_id") for m in memory_service.threads[key]
    }
    assert text <= stored and {e.message_id for e in result} <= stored


def test_ids_are_the_conversations_and_survive_a_cleared_result() -> None:
    from types import SimpleNamespace

    runtime = SimpleNamespace(conversation="conv_a", run_id="run_1", attempt=2)
    call = Called("c1", "lookup", {"q": "x"})
    said = [
        Said(Speaker.USER, "q"),
        Said(Speaker.ASSISTANT, "", tool_calls=(call,)),
        Said(Speaker.TOOL, "a long result", tool_call_id="c1"),
        Said(Speaker.ASSISTANT, "thinking"),
        Said(Speaker.USER, "steered", id="msg_given"),
        Said(Speaker.ASSISTANT, "the answer"),
    ]
    ids = [s.id for s in identified(said, runtime)]  # type: ignore[arg-type]
    assert ids == [
        question_id("conv_a", "run_1"),
        message_id("conv_a", "call:c1"),
        result_id("conv_a", "c1"),
        message_id("conv_a", "run_1:2:3"),
        "msg_given",
        answer_id("conv_a", "run_1"),
    ]
    cleared = [*said[:2], Said(Speaker.TOOL, "[cleared]", tool_call_id="c1"), *said[3:]]
    assert [s.id for s in identified(cleared, runtime)] == ids  # type: ignore[arg-type]
    other = SimpleNamespace(conversation="conv_b", run_id="run_1", attempt=2)
    assert not set(ids[:4]) & {s.id for s in identified(said, other)}  # type: ignore[arg-type]
    kept = recorded(identified(said, runtime)[2])  # type: ignore[arg-type]
    assert said_of(kept).id == result_id("conv_a", "c1")  # through the write spool


async def test_a_langgraph_transcript_keeps_a_steered_message_as_said() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from trellis.harness.adapters.langgraph import added
    from trellis.harness.middleware import STEERED

    messages = [
        HumanMessage("question"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "hold", "args": {}}]),
        ToolMessage("held", tool_call_id="c1"),
        HumanMessage("also", id="msg_x", response_metadata={STEERED: "run_2"}),
        AIMessage("done"),
    ]
    said = added(messages)
    assert [(s.role, s.content, s.id) for s in said if s.role is Speaker.USER] == [
        (Speaker.USER, "also", "msg_x")
    ]
    assert said[-1].content == "done"


def test_unsteered_targets_say_so_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        conversation.steering("agent-x", "acp")
        conversation.steering("agent-x", "acp")
    assert caplog.text.count("waits for the working run to end") == 1


@pytest.mark.parametrize("awaited", [False, True])
async def test_the_runs_own_model_input_filter_goes_first(harness: Harness, awaited: bool) -> None:
    from agents import Agent as OpenAIAgent
    from agents import RunConfig
    from agents.run_config import ModelInputData

    from tests.support.openai_model import ScriptedModel

    def own(data: Any) -> ModelInputData:
        given = data.model_data
        return ModelInputData(input=[*given.input], instructions="Answer briefly.")

    async def own_async(data: Any) -> ModelInputData:
        return own(data)

    model = ScriptedModel(["ok"])
    agent = harness.wrap(OpenAIAgent(name="f", model=model), id=f"filter-{awaited}")
    config = RunConfig(call_model_input_filter=own_async if awaited else own)
    result = await agent.run("hi", user="ada", framework_options={"run_config": config})
    assert result.status is RunStatus.SUCCESS
    assert model.system == ["Answer briefly."]  # theirs ran, ours after it
