"""Where memory keeps the conversation, its window reaches the framework as the framework's own
messages — the question, the answer, a tool call with its result — as a checkpoint would replay
them, and the turn just run is in the next turn's window at once (the run writes it before it
returns): no drain, no worker, no blob of text in the system message."""

# ruff: noqa: RUF001 - literal multilingual fixtures.
from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from benchmark.thread.__main__ import own_thread
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import snapshots
from trellis.harness.agent import TURN_WRITE_SECONDS
from trellis.harness.history import ASKED_WITH
from trellis.harness.identity import conversation_key

pytestmark = pytest.mark.asyncio
THREAD = "conv-1"
#: the memory service's thread user u1's conversation THREAD writes to (the fake's tenant)
KEPT = conversation_key("acme", "u1", THREAD)


@tool
def stock(sku: str) -> int:
    """Units in stock."""
    return 7


def _said(request: dict[str, Any]) -> list[tuple[str, str]]:
    """A chat request's messages after its system message: (role, content) — a question's own
    words, without the memory context sent as its first part."""
    return [
        (m["role"], _own(m.get("content"))) for m in request["messages"] if m["role"] != "system"
    ]


#: how the memory context sent with the question opens (``middleware.ASKED_WITH``)
OPENED = ASKED_WITH.split("{context}")[0]


def _own(content: Any) -> str:
    return content[-1]["text"] if isinstance(content, list) else content or ""


async def test_the_window_is_the_conversation_itself_and_the_last_turn_is_in_it(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["Noted: QX-4471.", "Tuesdays it is.", "QX-4471."])
    agent = memory_harness.wrap(create_agent(model, tools=[], system_prompt="s"), id="support")
    for n, text in enumerate(
        ("My order is QX-4471.", "I prefer Tuesdays.", "What is my order number?"), start=1
    ):
        result = await agent.run(text, user="u1", thread=THREAD)  # no drain between turns
        assert result.status is RunStatus.SUCCESS
        # written before the run returned, once
        assert len(memory_service.threads[KEPT]) == 2 * n
        assert len(memory_service.named("messages")) == n
    last = model.script.requests[-1]
    assert _said(last) == [
        ("user", "My order is QX-4471."),
        ("assistant", "Noted: QX-4471."),
        ("user", "I prefer Tuesdays."),
        ("assistant", "Tuesdays it is."),
        ("user", "What is my order number?"),
    ]
    # the instructions alone are the system message; the long-term context comes with the
    # question, and no transcript in it
    assert [m["content"] for m in last["messages"] if m["role"] == "system"] == ["s"]
    context = last["messages"][-1]["content"][0]["text"]
    assert context.startswith(OPENED + memory_service.context_text)
    assert "Recent conversation" not in context
    asked = memory_service.named("context")
    assert all(c.body["window_format"] == "messages" for c in asked)
    assert memory_service.violations == []


async def test_a_tool_call_comes_back_with_its_result(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), "7 units.", "You asked about a."])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="stock"), [stock])
    await agent.run("How many a?", user="u1", thread=THREAD)
    await agent.run("Which sku did I ask about?", user="u1", thread=THREAD)
    messages = [m for m in model.script.requests[-1]["messages"] if m["role"] != "system"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "user"]
    [call] = messages[1]["tool_calls"]
    assert call["function"]["name"] == "stock" and '"sku": "a"' in call["function"]["arguments"]
    assert messages[2]["tool_call_id"] == call["id"]
    # out of date already (a tool's data is live unless it says otherwise); when it was
    # fetched is the run record's to say, not the result's
    said = messages[2]["content"]
    assert said == snapshots.STALE.format(tool="stock") + "\n7"
    assert messages[3]["content"] == "7 units."


@pytest.mark.parametrize("kept", ["window", "checkpointer"])
async def test_each_turn_sends_the_last_ones_history_unchanged_and_the_context_with_the_question(
    memory_harness: Harness, memory_service: FakeMemoryService, kept: str
) -> None:
    """A provider's prompt cache keeps what a request shares with the one before it, from its
    start: the context, new for every question, comes after the history (with the question),
    so the next turn's request repeats this one's history as it was sent (audit K1)."""
    model = ScriptedChat(
        [("stock", {"sku": "a"}), "7 units.", ("stock", {"sku": "b"}), "7 too.", "a and b."]
    )
    target: Any = (
        ReAct(system_prompt="s", model=model)
        if kept == "window"
        else create_agent(model, tools=[stock], system_prompt="s", checkpointer=InMemorySaver())
    )
    agent = memory_harness.wrap(target, id=f"k1-{kept}")
    if kept == "window":
        agent = arm(agent, [stock])
    questions = ("How many a?", "And b?", "Which did I ask about?")
    firsts = []
    for text in questions:
        memory_service.context_text = f"Facts for: {text}"
        before = len(model.script.requests)
        result = await agent.run(text, user="u1", thread=THREAD)
        assert result.status is RunStatus.SUCCESS, result.error
        firsts.append(model.script.requests[before]["messages"])
    for messages, text in zip(firsts, questions, strict=True):
        assert [m["role"] for m in messages].count("system") == 1  # the instructions alone
        assert messages[0] == {"role": "system", "content": "s"}
        context, question = messages[-1]["content"]
        assert messages[-1]["role"] == "user" and question == {"type": "text", "text": text}
        assert context["text"].startswith(OPENED + f"Facts for: {text}")
    for earlier, later in itertools.pairwise(firsts):
        assert later[: len(earlier) - 1] == earlier[:-1]  # the history, as it was sent


async def test_multilingual_turns_reach_the_model_as_said(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["好的，已记录。", "QX-4471。"])
    agent = memory_harness.wrap(create_agent(model, tools=[]), id="zh")
    await agent.run("我的订单号是 QX-4471，请记住。", user="u1", thread=THREAD)
    await agent.run("我的订单号是多少？", user="u1", thread=THREAD)
    assert _said(model.script.requests[-1]) == [
        ("user", "我的订单号是 QX-4471，请记住。"),
        ("assistant", "好的，已记录。"),
        ("user", "我的订单号是多少？"),
    ]


@pytest.mark.parametrize("durable", [True, False], ids=["durable", "in-ram"])
async def test_a_framework_that_keeps_the_thread_replays_it_once(
    memory_harness: Harness, memory_service: FakeMemoryService, durable: bool
) -> None:
    model = ScriptedChat(["one", "two"])
    graph = create_agent(model, tools=[], checkpointer=own_thread() if durable else InMemorySaver())
    agent = memory_harness.wrap(graph, id=f"kept-{durable}")
    await agent.run("first", user="u1", thread=THREAD)
    await agent.run("second", user="u1", thread=THREAD)
    asked = memory_service.named("context")
    # a durable store owns the thread: no window; one kept in RAM gets memory's window as
    # text (a restart loses the store), never as messages next to the ones it replays
    assert all(c.body["window"] is not durable for c in asked)
    assert all("window_format" not in c.body for c in asked)
    # its own thread replays the first turn, once
    assert _said(model.script.requests[-1]) == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
    ]


async def test_a_function_target_gets_the_window_in_its_context_text(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    seen: list[str | None] = []

    async def fn(input: str, agent: Runtime) -> str:
        seen.append(agent.context)
        return "ok"

    agent = memory_harness.wrap(fn, id="fn")
    await agent.run("hi", user="u1", thread=THREAD)
    [call] = memory_service.named("context")
    assert call.body["window"] is True and "window_format" not in call.body
    assert seen == [memory_service.context_text]


async def test_an_openai_agent_reads_the_thread_through_its_session_as_input_items(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel([("lookup", {"q": "a"}), "found a", "you asked about a"])

    @tool
    def lookup(q: str) -> str:
        """Look it up."""
        return f"result {q}"

    agent = arm(memory_harness.wrap(OpenAIAgent(name="o", model=model), id="oa"), [lookup])
    await agent.run("find a", user="u1", thread=THREAD)
    await agent.run("what did I ask?", user="u1", thread=THREAD)
    items = model.inputs[-1]
    said = [i for i in items if i.get("role") != "system"]
    kinds = [i.get("type") or i.get("role") for i in said]
    assert kinds == ["user", "function_call", "function_call_output", "assistant", "user"]
    call, output = said[1], said[2]
    assert call["name"] == "lookup" and output["call_id"] == call["call_id"]
    assert output["output"] == "result a"  # its fetch time is in its run record
    # memory's thread is the session: the context carries no window besides it
    assert all("window_format" not in c.body for c in memory_service.named("context"))


async def test_turns_run_at_once_on_one_thread_are_each_recorded_whole(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([f"answer {n}" for n in range(4)])
    agent = memory_harness.wrap(create_agent(model, tools=[]), id="busy")
    await asyncio.gather(*(agent.run(f"question {n}", user="u1", thread=THREAD) for n in range(4)))
    batches = memory_service.named("messages")
    assert len(batches) == 4
    for batch in batches:
        user, assistant = batch.body["messages"]
        assert user["role"] == "USER" and assistant["role"] == "ASSISTANT"
    stored = [m["content"] for m in memory_service.threads[KEPT]]
    assert len(stored) == 8 and all(
        q.startswith("question") and a.startswith("answer")
        for q, a in zip(stored[::2], stored[1::2], strict=True)
    )


async def test_with_memory_down_the_run_goes_on_and_the_turn_is_queued() -> None:
    """As memory fails: a warning, no context, the transcript queued - no attempt to write it
    before the run returns (a first turn has no earlier turns for the run store to give)."""
    service = FakeMemoryService(fail={"context", "messages"})
    async with Harness(config=Settings(), memory=service.client()) as h:
        model = ScriptedChat(["ok"])
        agent = h.wrap(create_agent(model, tools=[]), id="down")
        events = [e async for e in agent.stream("hi", user="u1", thread=THREAD)]
        warnings = [
            e.data["code"]
            for e in events
            if e.type is RunEventType.CUSTOM and e.data["name"] == "warning"
        ]
        assert warnings == ["memory_unavailable"]
        assert events[-1].type is RunEventType.RUN_FINISHED
        assert _said(model.script.requests[-1]) == [("user", "hi")]
        await h.writes.drain()
        assert h.writes.failed == 1  # the transcript, tried by the queue alone and reported


async def test_with_memory_down_mid_conversation_the_earlier_turns_come_from_the_run_store() -> (
    None
):
    """Memory answered the first turn and fails on the second (an outage, a circuit open):
    the conversation is not lost - its earlier turns come from the run store, each one's
    question and answer, as with memory off."""
    service = FakeMemoryService()
    async with Harness(config=Settings(), memory=service.client()) as h:
        model = ScriptedChat(["noted: tides", "you said tides"])
        agent = h.wrap(create_agent(model, tools=[]), id="outage")
        await agent.run("remember tides", user="u1", thread=THREAD)
        service.fail = {"context", "messages"}
        events = [e async for e in agent.stream("what did I say?", user="u1", thread=THREAD)]
        warnings = [
            e.data["code"]
            for e in events
            if e.type is RunEventType.CUSTOM and e.data["name"] == "warning"
        ]
        assert "memory_unavailable" in warnings
        assert _said(model.script.requests[-1]) == [
            ("user", "remember tides"),
            ("assistant", "noted: tides"),
            ("user", "what did I say?"),
        ]


async def test_with_memory_down_a_thread_the_framework_keeps_is_not_given_twice() -> None:
    """The graph keeps its own thread (a durable checkpointer): memory failing adds nothing."""
    service = FakeMemoryService()
    async with Harness(config=Settings(), memory=service.client()) as h:
        model = ScriptedChat(["noted: tides", "you said tides"])
        graph = create_agent(model, tools=[], checkpointer=own_thread())
        agent = h.wrap(graph, id="kept-outage")
        await agent.run("remember tides", user="u1", thread=THREAD)
        service.fail = {"context", "messages"}
        await agent.run("what did I say?", user="u1", thread=THREAD)
        said = _said(model.script.requests[-1])
        assert said.count(("user", "remember tides")) == 1
        assert said[-1] == ("user", "what did I say?")


async def test_a_turn_whose_write_fails_is_queued_and_delivered() -> None:
    service = FakeMemoryService(fail_times={"messages": 1})
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["ok"]), tools=[]), id="flaky")
        result = await agent.run("hi", user="u1", thread=THREAD)
        assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert [m["content"] for m in service.threads[KEPT]] == ["hi", "ok"]


async def test_a_slow_write_does_not_hold_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness import agent as agent_module
    from trellis.harness.clients.memory import RunMemory

    service = FakeMemoryService()
    slow = asyncio.Event()
    plain = RunMemory.record_messages
    calls = 0

    async def stalled(self: RunMemory, *args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await slow.wait()  # the first attempt never answers
        await plain(self, *args)

    monkeypatch.setattr(RunMemory, "record_messages", stalled)
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.05)
    assert TURN_WRITE_SECONDS > 0.05
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["ok"]), tools=[]), id="slow")
        result = await agent.run("hi", user="u1", thread=THREAD)
        assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert [m["content"] for m in service.threads[KEPT]] == ["hi", "ok"]
