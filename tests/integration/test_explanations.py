"""Explanations: a follow-up about an earlier answer ("where did that come from?", "what did you
do?", "how long did it take?") is answered from the run's own record, which the harness keeps
and renders for the audience its key is — the end user (a service key) sees the steps in plain
words, the sources, the time, their own memories and the approvals by role; an administrator
sees everything — and never a source the run's tools did not return."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool

from tests.support.arm import arm
from tests.support.memory import NOW, FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.runlog import LoggedRuns
from trellis import Ask, Harness, Hooks, ReAct, Settings, tool
from trellis.contracts import RunStatus, ToolCall
from trellis.harness.clients.memory import CONVERSATION_BUDGET
from trellis.harness.explain import RECORD, measured
from trellis.harness.runs import LocalRuns
from trellis.harness.runtime import EARLIER
from trellis.harness.tools.base import Citation
from trellis.runs import Job

pytestmark = pytest.mark.asyncio
THREAD = "conv-explain"
SYSTEM = "You answer stock price questions."
NASDAQ = Citation(name="Nasdaq Last Sale feed", date="2026-10-27", link="https://nasdaq.test/acme")
#: what the client view never holds (the tool's name and arguments, its raw output, ids)
TOOL = "stock_quote"


@tool(side_effects="read", response_format="content_and_artifact")
def stock_quote(symbol: str) -> tuple[str, Citation]:
    """The latest closing quote of a stock symbol.

    Args:
        symbol: the ticker, e.g. ACME
    """
    return json.dumps({"symbol": symbol, "close": 123.45, "raw_feed_id": "feed-77"}), NASDAQ


@tool(side_effects="read", response_format="content_and_artifact")
def intraday(symbol: str, page: int) -> tuple[str, list[dict[str, str]]]:
    """One page of the day's trades of a stock symbol, from the exchange's tape."""
    ticks = ", ".join(f'{{"t": "15:{m:02d}", "p": {120 + m % 7}}}' for m in range(60))
    return f"[{ticks}]" * 4, [{"name": "Exchange tape", "date": "2026-10-27"}]


def memory(fake: FakeMemoryService, memory_id: str, user: str, content: str, **kw: str) -> None:
    fake.memories[memory_id] = {
        "content": content,
        "user": user,
        "visibility": kw.get("visibility", "USER"),
        "memory_type": "SEMANTIC",
        "current": True,
        "superseded_by": None,
        "supersedes": None,
        "created_at": NOW,
    }


def looking(h: Harness, model: Any, tools: list[Any], *, id: str = "stock", **wrap: Any) -> Any:
    """A ``ReAct`` agent of the ``lookup`` arm: it reads its earlier answers' records."""
    with measured("lookup"):
        return arm(h.wrap(ReAct(SYSTEM, model), id=id, **wrap), tools)


def tool_results(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m["role"] == "tool"]


async def two_turns(
    h: Harness, second: str, *, turns: list[Any] | None = None, tools: list[Any] | None = None
) -> tuple[ScriptedChat, str]:
    """Turn 1 asks the quote (one cited call); turn 2 asks ``second``, and the model reads the
    earlier answer's record: what the model was given of it."""
    model = ScriptedChat(
        turns
        or [(TOOL, {"symbol": "ACME"}), "ACME closed at 123.45 USD.", (EARLIER, {}), "From Nasdaq."]
    )
    with measured("lookup"):
        agent = arm(
            h.wrap(ReAct(SYSTEM, model, context_window=16_000), id="stock"), tools or [stock_quote]
        )
    first = await agent.run("What did ACME close at yesterday?", user="u1", thread=THREAD)
    assert first.status is RunStatus.SUCCESS, first.error
    later = await agent.run(second, user="u1", thread=THREAD)
    assert later.status is RunStatus.SUCCESS, later.error
    return model, tool_results(model.requests[-1])[-1]


@pytest.fixture
async def admin(memory_service: FakeMemoryService) -> AsyncIterator[Harness]:
    memory_service.role = "admin"
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        yield h


# --------------------------------------------------------------------------- the answer's sources
async def test_an_answer_carries_the_sources_its_tools_returned_and_no_other(
    harness: Harness,
) -> None:
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "ACME closed at 123.45 USD, says Reuters."])
    agent = looking(harness, model, [stock_quote])
    result = await agent.run("What did ACME close at?", user="u1")
    # the model named Reuters; no tool returned it: it is not a source
    assert result.sources == [NASDAQ]
    # what the model read is the content alone, never the artifact
    assert tool_results(model.requests[-1]) == [
        json.dumps({"symbol": "ACME", "close": 123.45, "raw_feed_id": "feed-77"})
    ]


async def test_an_artifact_that_names_no_source_is_not_one(harness: Harness) -> None:
    @tool(side_effects="read", response_format="content_and_artifact")
    def lookup(symbol: str) -> tuple[str, Any]:
        """A lookup whose artifact is a raw payload."""
        return "123.45", {"rows": [1, 2]}

    model = ScriptedChat([("lookup", {"symbol": "ACME"}), "123.45"])
    agent = looking(harness, model, [lookup])
    assert (await agent.run("ACME?", user="u1")).sources == []


async def test_a_content_and_artifact_tool_that_returns_no_pair_fails_its_call(
    harness: Harness,
) -> None:
    @tool(side_effects="read", response_format="content_and_artifact")
    def broken(symbol: str) -> str:
        """Returns the content alone, by mistake."""
        return "123.45"

    model = ScriptedChat([("broken", {"symbol": "ACME"}), "It failed."])
    agent = looking(harness, model, [broken])
    result = await agent.run("ACME?", user="u1")
    assert result.sources == []
    [said] = tool_results(model.requests[-1])
    assert "content_and_artifact" in said and "(content, artifact)" in said


async def test_a_langchain_tools_artifact_is_its_source(harness: Harness) -> None:
    @lc_tool(response_format="content_and_artifact")
    def quote(symbol: str) -> tuple[str, Citation]:
        """The latest closing quote of a stock symbol."""
        return "123.45", NASDAQ

    quote.metadata = {"side_effects": "read"}
    model = ScriptedChat([("quote", {"symbol": "ACME"}), "123.45"])
    agent = harness.wrap(ReAct(SYSTEM, model, tools=[quote]), id="stock")
    assert (await agent.run("ACME?", user="u1")).sources == [NASDAQ]


# --------------------------------------------------------------------------- each question
QUESTIONS = {
    "steps": ("What steps did you take?", ["What I did (1 step(s)):", "1. The latest closing"]),
    "citation": ("Where did this come from? Cite it.", ["Nasdaq Last Sale feed, 2026-10-27"]),
    "how_known": ("How do you know this is accurate?", ["https://nasdaq.test/acme", "Basis:"]),
    "reasoning": ("Why did you conclude this?", ["Basis: the answer rests on what step(s) 1"]),
    "sure": ("How sure are you? What could be wrong?", ["Limits: it rests on a single source"]),
    "about_me": ("What do you know about me? Did you use my history?", ["- I hold 10 ACME."]),
    "approved": ("Did anyone approve this?", ["Approvals: none was needed."]),
    "how_long": ("How long did it take?", ["It took", "seconds of work"]),
    "how_many": ("How many tool calls did you make?", ["(1 step(s))"]),
    "which_tools": ("Which tools did you use?", ["The latest closing quote of a stock symbol"]),
}


@pytest.mark.parametrize("kind", QUESTIONS)
async def test_each_question_about_an_earlier_answer_is_answered_from_its_record(
    memory_harness: Harness, memory_service: FakeMemoryService, kind: str
) -> None:
    memory(memory_service, "mem_1", "u1", "I hold 10 ACME.")
    memory_service.shown = ["mem_1"]
    question, expected = QUESTIONS[kind]
    model, seen = await two_turns(memory_harness, question)
    for part in expected:
        assert part in seen, (part, seen)
    # the window carries the earlier question and answer: the record does not say them twice
    assert "Earlier question" not in seen and "Answer given" not in seen
    # the client's view: no tool name, argument, raw output or id
    for leak in (TOOL, '"symbol"', "feed-77", "mem_1", "run_", "raw_feed_id"):
        assert leak not in seen, leak
    # the earlier turn reached the model whole as well: its question, call and result
    roles = [m["role"] for m in model.requests[2]["messages"] if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


async def test_an_administrator_sees_everything(
    admin: Harness, memory_service: FakeMemoryService
) -> None:
    memory(memory_service, "mem_1", "u1", "I hold 10 ACME.")
    memory(memory_service, "mem_9", "bob", "Bob trades ACME daily.", visibility="WORKSPACE")
    memory_service.shown = ["mem_1", "mem_9"]
    _, seen = await two_turns(admin, "Which tools did you use, with what?")
    shown = json.loads(seen)
    [step] = shown["record"]["steps"]
    assert step["tool"] == TOOL and step["args"] == {"symbol": "ACME"}
    assert step["sources"] == [NASDAQ.model_dump()]
    assert shown["run"]["run_id"].startswith("run_") and shown["run"]["user_id"] == "u1"
    # what the window carries (the input, the output, the tool's result) is not said twice
    assert "output" not in step and "input" not in shown["run"] and "output" not in shown["run"]
    assert {m["memory_id"] for m in shown["memories"]} == {"mem_1", "mem_9"}
    assert shown["record"]["spent"]["tokens"] > 0


async def test_another_persons_memories_are_never_in_the_clients_view(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory(memory_service, "mem_1", "u1", "I hold 10 ACME.")
    memory(memory_service, "mem_9", "bob", "Bob's salary is 9000.", visibility="WORKSPACE")
    memory(memory_service, "mem_8", "eve", "Eve's private note.")  # not readable: 404
    memory_service.shown = ["mem_1", "mem_9", "mem_8"]
    _, seen = await two_turns(memory_harness, "What do you know about me?")
    assert "- I hold 10 ACME." in seen
    assert "Bob" not in seen and "9000" not in seen and "Eve" not in seen


async def test_another_users_and_another_agents_runs_in_the_thread_are_not_earlier_answers(
    harness: Harness,
) -> None:
    model = ScriptedChat(["Hi Bob.", "Other agent.", (EARLIER, {}), "None yet."])
    agent = looking(harness, model, [stock_quote])
    other = harness.wrap(ReAct(SYSTEM, model), id="other")
    await agent.run("Hello", user="bob", thread=THREAD)  # another person, the same thread id
    await other.run("Hello", user="u1", thread=THREAD)  # another agent
    await agent.run("What did you do before?", user="u1", thread=THREAD)
    assert tool_results(model.requests[-1]) == ["There is no earlier answer in this conversation."]


async def test_an_answer_further_back_than_there_are_is_said_so(harness: Harness) -> None:
    model = ScriptedChat(["One.", (EARLIER, {"back": 3}), "Only one."])
    agent = looking(harness, model, [stock_quote])
    await agent.run("First?", user="u1", thread=THREAD)
    await agent.run("And three answers ago?", user="u1", thread=THREAD)
    assert tool_results(model.requests[-1]) == [
        "There are only 1 earlier answer(s) in this conversation."
    ]


async def test_a_run_with_no_conversation_has_no_earlier_answers_to_read(
    harness: Harness,
) -> None:
    model = ScriptedChat(["Hi."])
    agent = looking(harness, model, [stock_quote])
    await agent.run("Hello", user="u1")
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == [TOOL]


async def test_a_run_whose_record_was_not_kept_says_it_does_not_know_its_steps(
    harness: Harness,
) -> None:
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "123.45", (EARLIER, {}), "I do not know."])
    agent = looking(harness, model, [stock_quote])
    first = await agent.run("ACME?", user="u1", thread=THREAD)
    agent.records.pop(first.run_id)  # forgotten (an older release kept none)
    await agent.run("What did you do?", user="u1", thread=THREAD)
    [seen] = tool_results(model.requests[-1])
    assert "I have no record of the steps behind that answer, so I do not know them." in seen
    assert 'Answer given: "123.45"' in seen


# --------------------------------------------------------------------------- large and long turns
async def test_in_the_sized_arm_a_large_long_turn_reaches_the_next_one_whole(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    pages = [("intraday", {"symbol": "ACME", "page": n}) for n in range(12)]
    model = ScriptedChat([*pages, "ACME traded 120 to 126.", "12 calls, from the tape."])
    with measured("sized", share=0.5, cap=40_000):
        target = ReAct(SYSTEM, model, context_window=128_000)
        agent = arm(memory_harness.wrap(target, id="tape"), [intraday])
    await agent.run("How did ACME trade today?", user="u1", thread=THREAD)
    await agent.run("How many tool calls did you make?", user="u1", thread=THREAD)
    asked = memory_service.named("context")[-1]
    # a window sized to the model (its share, at most the cap): the whole earlier turn
    assert asked.body[CONVERSATION_BUDGET] == 40_000
    said = [m["role"] for m in model.requests[-1]["messages"] if m["role"] != "system"]
    assert said[0] == "user" and said.count("tool") == 12 and said[-1] == "user"
    assert EARLIER not in [t["function"]["name"] for t in model.requests[-1]["tools"]]


async def test_by_default_the_window_is_the_services_and_no_record_is_offered(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    pages = [("intraday", {"symbol": "ACME", "page": n}) for n in range(12)]
    model = ScriptedChat([*pages, "ACME traded 120 to 126.", "I do not know."])
    agent = arm(
        memory_harness.wrap(ReAct(SYSTEM, model, context_window=128_000), id="tape"), [intraday]
    )
    await agent.run("How did ACME trade today?", user="u1", thread=THREAD)
    await agent.run("How many tool calls did you make?", user="u1", thread=THREAD)
    assert all(CONVERSATION_BUDGET not in c.body for c in memory_service.named("context"))
    # the service's own window: the newest turn whole even past its budget (over_budget)
    said = [m["role"] for m in model.requests[-1]["messages"] if m["role"] != "system"]
    assert said.count("tool") == 12
    assert EARLIER not in [t["function"]["name"] for t in model.requests[-1]["tools"]]


async def test_in_the_lookup_arm_a_long_turns_record_counts_every_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    pages = [("intraday", {"symbol": "ACME", "page": n}) for n in range(12)]
    model = ScriptedChat([*pages, "ACME traded 120 to 126.", (EARLIER, {}), "12 steps."])
    with measured("lookup"):
        target = ReAct(SYSTEM, model, context_window=128_000)
        agent = arm(memory_harness.wrap(target, id="tape"), [intraday])
    await agent.run("How did ACME trade today?", user="u1", thread=THREAD)
    await agent.run("How many tool calls did you make?", user="u1", thread=THREAD)
    assert all(CONVERSATION_BUDGET not in c.body for c in memory_service.named("context"))
    [seen] = tool_results(model.requests[-1])[-1:]
    assert "What I did (12 step(s)):" in seen and "12. One page of the day's trades" in seen
    assert seen.count("Exchange tape") == 13  # each step, and the one source once
    assert "Limits: it rests on a single source." in seen


async def test_an_older_memory_service_is_asked_for_its_own_window_from_then_on(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.sized_windows = False
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "123.45", "Again: 123.45."])
    with measured("sized", share=0.25, cap=8000):
        target = ReAct(SYSTEM, model, context_window=16_000)
        agent = arm(memory_harness.wrap(target, id="stock"), [stock_quote])
    await agent.run("ACME?", user="u1", thread=THREAD)
    await agent.run("Again?", user="u1", thread=THREAD)
    asked = memory_service.named("context")
    # the first ask was refused for the field, then asked as before, and so was the next turn
    assert [CONVERSATION_BUDGET in c.body for c in asked] == [True, False, False]
    assert memory_harness.memory is not None and not memory_harness.memory.sized_windows
    roles = [m["role"] for m in model.requests[-1]["messages"] if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


async def test_a_context_refused_for_another_reason_is_no_context_and_no_downgrade(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.failures["context"] = 422
    model = ScriptedChat(["Without memory."])
    with measured("sized", share=0.25, cap=8000):
        agent = memory_harness.wrap(ReAct(SYSTEM, model, context_window=16_000), id="stock")
    result = await agent.run("ACME?", user="u1", thread=THREAD)
    assert result.status is RunStatus.SUCCESS
    assert memory_harness.memory is not None and memory_harness.memory.sized_windows


async def test_insufficient_memory_lets_the_model_use_its_own_earlier_calls(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.evidence_status = "INSUFFICIENT"
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "123.45", "From the Nasdaq feed."])
    agent = arm(memory_harness.wrap(ReAct(SYSTEM, model), id="stock"), [stock_quote])
    await agent.run("ACME?", user="u1", thread=THREAD)
    await agent.run("Where did that come from?", user="u1", thread=THREAD)
    note = [m["content"] for m in model.requests[-1]["messages"] if m["role"] == "system"][-1]
    assert "Long-term memory holds nothing about this question" in note
    assert "your own earlier tool calls and their results included" in note
    assert "If the answer depends on something the user said that is not here" in note
    assert any(m["role"] == "tool" for m in model.requests[-1]["messages"])


async def test_memory_off_answers_from_the_record_alone(harness: Harness) -> None:
    model, seen = await two_turns(harness, "Where did that come from?")
    assert "Nasdaq Last Sale feed, 2026-10-27, https://nasdaq.test/acme" in seen
    assert "Your saved notes I used: none." in seen and "Memory:" not in seen
    # no memory: the model keeps no conversation of its own, and the record is what it has
    roles = [m["role"] for m in model.requests[2]["messages"] if m["role"] != "system"]
    assert roles == ["user"]


# --------------------------------------------------------------------------- approvals
class FinanceAsks(Hooks):
    async def before_tool(self, call: ToolCall) -> Ask | None:
        return Ask("Approve the quote?", assignee="role:finance") if call.tool == TOOL else None


async def test_an_approval_given_at_a_pause_is_in_the_record_by_role_not_by_name(
    harness: Harness,
) -> None:
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "123.45", (EARLIER, {}), "Finance did."])
    agent = looking(harness, model, [stock_quote], hooks=[FinanceAsks()])
    paused = await agent.run("ACME?", user="u1", thread=THREAD)
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="priya")
    assert done.status is RunStatus.SUCCESS and done.sources == [NASDAQ]
    await agent.run("Did anyone approve this?", user="u1", thread=THREAD)
    [seen] = tool_results(model.requests[-1])
    assert "- a reviewer (finance) approved step 1." in seen
    assert "priya" not in seen


async def test_an_approval_by_a_named_person_and_a_rejection_are_said_by_role(
    harness: Harness,
) -> None:
    class Named(Hooks):
        async def before_tool(self, call: ToolCall) -> Ask | None:
            return Ask("Approve?", assignee="user:priya") if call.tool == TOOL else None

    model = ScriptedChat(
        [(TOOL, {"symbol": "ACME"}), "Not allowed.", (EARLIER, {}), "It was rejected."]
    )
    agent = looking(harness, model, [stock_quote], hooks=[Named()])
    paused = await agent.run("ACME?", user="u1", thread=THREAD)
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="priya", comment="no")
    await agent.run("Did anyone approve this?", user="u1", thread=THREAD)
    [seen] = tool_results(model.requests[-1])
    assert "- a reviewer rejected a step that did not run." in seen
    assert "priya" not in seen and "user:" not in seen


# --------------------------------------------------------------------------- the record survives
class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_the_record_survives_a_crash_and_is_read_by_the_worker_that_took_over() -> None:
    crashed: list[BaseException] = [Crash()]

    @tool(side_effects="write", response_format="content_and_artifact")
    def order(symbol: str) -> tuple[str, Citation]:
        """Place an order for a stock."""
        return "ordered", Citation(name="Broker confirmations", date="2026-10-28")

    @tool(side_effects="read")
    def check(symbol: str) -> str:
        """Check an order's state."""
        if crashed:
            raise crashed.pop()
        return "filled"

    store = LocalRuns()
    model = ScriptedChat(
        [
            ("order", {"symbol": "ACME"}),
            ("check", {"symbol": "ACME"}),  # the first worker dies here
            ("check", {"symbol": "ACME"}),  # the second: the order is replayed, not placed
            "Ordered and filled.",
            (EARLIER, {}),
            "From the broker.",
        ]
    )
    first, second = (Harness(config=Settings(), runs=store) for _ in range(2))
    try:
        dying, taking_over = (
            looking(h, model, [order, check], id="orders") for h in (first, second)
        )
        handle = await dying.start("Buy ACME", user="u1", thread=THREAD)
        worker = first.worker([dying])
        claimed = await store.claim(worker.worker_id, [dying.id])
        assert claimed is not None
        with pytest.raises(Crash):
            await dying.execute(
                Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
            )
        held, _ = store._leases[handle.id]
        store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
        assert await second.worker([taking_over]).run_once()
        assert (await handle.result(wait=5)).status is RunStatus.SUCCESS
        await taking_over.run("Where did that come from?", user="u1", thread=THREAD)
        [seen] = tool_results(model.requests[-1])
        assert "1. Place an order for a stock. — from Broker confirmations, 2026-10-28" in seen
        assert "2. Check an order's state." in seen
    finally:
        await first.aclose()
        await second.aclose()


async def test_with_agent_runs_the_record_is_in_the_runs_log_for_any_replica() -> None:
    runs = LoggedRuns()
    model = ScriptedChat([(TOOL, {"symbol": "ACME"}), "123.45", (EARLIER, {}), "Nasdaq."])
    async with (
        Harness(config=Settings(), runs=runs) as h,
        Harness(config=Settings(), runs=runs) as replica,
    ):
        agent = looking(h, model, [stock_quote])
        first = await agent.run("ACME?", user="u1", thread=THREAD)
        kept = [e for e in runs.logged(first.run_id) if (e.data or {}).get("name") == RECORD]
        assert len(kept) == 1 and not agent.records
        other = looking(replica, model, [stock_quote])
        await other.run("Where did that come from?", user="u1", thread=THREAD)
    [seen] = tool_results(model.requests[-1])
    assert "Nasdaq Last Sale feed, 2026-10-27, https://nasdaq.test/acme" in seen


# --------------------------------------------------------------------------- languages
async def test_steps_in_another_language_are_kept_as_written(harness: Harness) -> None:
    @tool(side_effects="read", response_format="content_and_artifact")
    def bhav(symbol: str) -> tuple[str, Citation]:
        """किसी शेयर का पिछला बंद भाव। विवरण के लिए एक्सचेंज देखें।"""
        return "123.45", Citation(name="राष्ट्रीय शेयर बाज़ार", date="2026-10-27")

    model = ScriptedChat([("bhav", {"symbol": "ACME"}), "123.45 रुपये।", (EARLIER, {}), "एनएसई से।"])
    agent = looking(harness, model, [bhav])
    await agent.run("ACME का भाव क्या था?", user="u1", thread=THREAD)
    await agent.run("यह जानकारी कहाँ से आई?", user="u1", thread=THREAD)
    [seen] = tool_results(model.requests[-1])
    assert "1. किसी शेयर का पिछला बंद भाव। — from राष्ट्रीय शेयर बाज़ार, 2026-10-27" in seen
    assert 'Earlier question: "ACME का भाव क्या था?"' in seen and "bhav" not in seen
