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
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from benchmark.thread.__main__ import own_thread
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService, pushed
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from tests.support.today import dated, kept_dated
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import snapshots
from trellis.harness import writes as writes_module
from trellis.harness.agent import TURN_WRITE_SECONDS
from trellis.harness.history import ASKED_WITH, Speaker, from_window
from trellis.harness.identity import conversation_key
from trellis.memory.models import WindowMessage

pytestmark = pytest.mark.asyncio
THREAD = "conv-1"
#: the memory service's thread user u1's conversation THREAD writes to (the fake's tenant)
KEPT = conversation_key("acme", "u1", THREAD)


@tool(side_effects="read")
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
    assert [m["content"] for m in last["messages"] if m["role"] == "system"] == [dated("s")]
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
    own = StructuredTool.from_function(stock.fn, metadata={"side_effects": "read"})
    target: Any = (
        ReAct(system_prompt="s", model=model)
        if kept == "window"
        else create_agent(model, tools=[own], system_prompt="s", checkpointer=InMemorySaver())
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
    # a ReAct is told what the harness keeps of a large result; a graph of its own is not
    said = kept_dated("s") if kept == "window" else dated("s")
    # a graph keeping its own thread is told, after the question, that the earlier live result
    # is out of date (a window's replay marks it on the result itself): the request's copy only
    stale = f"\n\n{snapshots.note(['stock'])}" if kept == "checkpointer" else ""
    for n, (messages, text) in enumerate(zip(firsts, questions, strict=True)):
        assert [m["role"] for m in messages].count("system") == 1  # the instructions alone
        assert messages[0] == {"role": "system", "content": said}
        context, question = messages[-1]["content"]
        asked = text + (stale if n else "")
        assert messages[-1]["role"] == "user" and question == {"type": "text", "text": asked}
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
    assert seen == [pushed(memory_service.context_text)]


async def test_a_child_runs_question_is_its_parents_words_not_the_users(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A sub-agent's question is what the agent that handed it the work wrote: kept as an
    agent's message (the service's AGENT role), so the service never reads it as the person's
    own words (their standing rules, their facts) and their conversation's window never
    replays it as a turn of theirs. The person's own question stays theirs."""

    async def specialist(input: str, agent: Runtime) -> str:
        return "3 units"

    child = memory_harness.wrap(specialist, id="specialist")

    async def desk(input: str, agent: Runtime) -> str:
        handed = await child.run("Count SKU-1. Research only; never reorder.", user=agent.user)
        return f"specialist says {handed.answer}"

    parent = memory_harness.wrap(desk, id="desk")
    await parent.run("How many SKU-1?", user="u1", thread=THREAD)
    roles = [(m["role"], m["content"]) for m in memory_service.threads[KEPT]]
    assert ("USER", "How many SKU-1?") in roles
    assert ("AGENT", "Count SKU-1. Research only; never reorder.") in roles
    assert ("USER", "Count SKU-1. Research only; never reorder.") not in roles
    # the person's conversation, as its next turn's window replays it: their turn, not the hand-off
    window = from_window([WindowMessage.model_validate(m) for m in memory_service._window(KEPT)])
    assert [s.content for s in window if s.role is Speaker.USER] == ["How many SKU-1?"]


async def test_an_openai_agent_reads_the_thread_through_its_session_as_input_items(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel([("lookup", {"q": "a"}), "found a", "you asked about a"])

    @tool(side_effects="read")
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
    # lookup is live (it declares no max age): its earlier value reads as out of date, as the
    # memory window's replay says it, from its run record's fetch time (the text is unstamped)
    assert output["output"] == snapshots.STALE.format(tool="lookup") + "\nresult a"
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


async def test_a_later_turn_is_stored_after_an_earlier_one_still_on_its_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first turn's write is slow and queued; the second turn, said meanwhile, is not
    written past it: memory keeps a conversation's messages in the order they arrive."""
    from trellis.harness import agent as agent_module
    from trellis.harness.clients.memory import RunMemory

    service = FakeMemoryService()
    plain = RunMemory.record_messages
    first = asyncio.Event()

    async def slow_first(self: RunMemory, messages: Any, run_id: str, attempt: int) -> None:
        if messages[0].content == "one" and not first.is_set():
            await asyncio.sleep(0.3)  # past the run's wait: queued, and still on its way
            first.set()
        await plain(self, messages, run_id, attempt)

    monkeypatch.setattr(RunMemory, "record_messages", slow_first)
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.05)
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["1", "2"]), tools=[]), id="ordered")
        for said in ("one", "two"):
            result = await agent.run(said, user="u1", thread=THREAD)
            assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert [m["content"] for m in service.threads[KEPT]] == ["one", "1", "two", "2"]


async def test_an_earlier_turn_sent_again_is_sent_as_it_was_under_its_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first turn's write is slow; the second turn sends it again (in order, before its
    own): under the first turn's key it must be the first turn's write, byte for byte — its
    own scope — or the service refuses the key reused with another payload and the turn is
    lost (seen under load: "Idempotency-Key reused with a different payload")."""
    from trellis.harness import agent as agent_module
    from trellis.harness.clients.memory import RunMemory

    service = FakeMemoryService()
    plain = RunMemory.record_messages
    first = asyncio.Event()

    async def slow_first(self: RunMemory, messages: Any, run_id: str, attempt: int) -> None:
        if messages[0].content == "one" and not first.is_set():
            await asyncio.sleep(0.3)  # past the run's wait: queued, and still on its way
            first.set()
        await plain(self, messages, run_id, attempt)

    monkeypatch.setattr(RunMemory, "record_messages", slow_first)
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.05)
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["1", "2"]), tools=[]), id="resent")
        for said in ("one", "two"):
            result = await agent.run(said, user="u1", thread=THREAD)
            assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
    sent: dict[str, Any] = {}
    for call in service.calls:
        if call.name == "messages" and call.idempotency_key:
            assert sent.setdefault(call.idempotency_key, call.body) == call.body, (
                call.idempotency_key
            )


async def test_a_later_turn_never_waits_for_a_stuck_earlier_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first turn's own write never answers (its worker stuck on it): the second turn's
    write sends the first turn again, then its own, so both are in memory in order without
    waiting for it; the stuck write, when it comes, stores nothing twice."""
    from trellis.harness import agent as agent_module
    from trellis.harness.clients.memory import RunMemory

    service = FakeMemoryService()
    plain = RunMemory.record_messages
    stuck = asyncio.Event()
    calls: list[str] = []

    async def first_one_stuck(self: RunMemory, messages: Any, run_id: str, attempt: int) -> None:
        calls.append(messages[0].content)
        if messages[0].content == "one" and calls.count("one") == 1:
            await stuck.wait()  # the first write of the first turn never answers
        await plain(self, messages, run_id, attempt)

    monkeypatch.setattr(RunMemory, "record_messages", first_one_stuck)
    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.05)
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["1", "2"]), tools=[]), id="unstuck")
        for said in ("one", "two"):
            assert (await agent.run(said, user="u1", thread=THREAD)).status is RunStatus.SUCCESS
        async with asyncio.timeout(5):
            while [m["content"] for m in service.threads.get(KEPT, [])] != ["one", "1", "two", "2"]:  # noqa: ASYNC110 - the write queue's own workers
                await asyncio.sleep(0.01)
        stuck.set()
        await h.writes.drain()
        assert [m["content"] for m in service.threads[KEPT]] == ["one", "1", "two", "2"]


async def test_a_turn_stored_but_answered_past_the_deadline_is_stored_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service stored the turn, then its request deadline answered ``504 TIMEOUT``: the
    write is sent again under the same ``Idempotency-Key`` and the same message ids, and the
    thread holds the turn once."""
    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.0)
    service = FakeMemoryService(deadline_after={"messages": 1})
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["ok"]), tools=[]), id="late")
        result = await agent.run("hi", user="u1", thread=THREAD)
        assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert h.writes.failed == 0
    sent = service.named("messages")
    assert len(sent) == 2 and sent[0].idempotency_key == sent[1].idempotency_key
    assert sent[0].body == sent[1].body
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
    assert TURN_WRITE_SECONDS == 5.0  # what a run waits at most for its turn to be written
    async with Harness(config=Settings(), memory=service.client()) as h:
        agent = h.wrap(create_agent(ScriptedChat(["ok"]), tools=[]), id="slow")
        result = await agent.run("hi", user="u1", thread=THREAD)
        assert result.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert [m["content"] for m in service.threads[KEPT]] == ["hi", "ok"]
