"""Every shape a ``ReAct`` run's end takes, on ``ReAct`` alone (``create_agent``'s loop with the
harness's middleware) and on ``ReAct`` that is Deep Agents from its first step (``backend=``):
the answer streamed and returned as one text, unicode kept as it is; an empty reply; a tool that
fails; a structured answer; a large result read to its end and queried; a write approved and
run once; a cancelled run; a run past its time limit; a run past its budget; and a run whose
memory service is down."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from pydantic import BaseModel

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Budget, Harness, ReAct, tool
from trellis.contracts import InterruptReason, RunEventType, RunStatus
from trellis.harness.automatic import JournalBackend
from trellis.harness.budget import NO_ANSWER

#: An answer in several scripts, an emoji and a right-to-left word among them.
UNICODE = "Stock: 7 — 在庫あり ✓, ¡sí! 🚀 שלום"


@pytest.fixture(params=["alone", "deep"])
def build(request: pytest.FixtureRequest) -> Any:
    """``ReAct`` as the parameter says: alone, or given a backend (Deep Agents from its first
    step)."""

    def react(model: Any, **given: Any) -> Any:
        if request.param == "deep":
            given.setdefault("backend", JournalBackend())
        return ReAct(system_prompt="s", model=model, **given)

    return react


@tool(side_effects="read")
def stock(sku: str) -> str:
    """Units of a SKU in stock."""
    return {"a": "7 — 在庫 ✓"}.get(sku, "0")


async def test_the_streamed_text_is_the_answer_and_unicode_is_kept(
    harness: Harness, build: Any
) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), UNICODE])
    agent = arm(harness.wrap(build(model), id="streamed", without={"schedules"}), [stock])
    events = [e async for e in agent.stream("how many a?", user="u")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED, finished
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == UNICODE == finished.data["result"]
    [result] = [e for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert "7 — 在庫 ✓" in str(result.data)
    # the model read the tool's text as it is
    assert model.requests[1]["messages"][-1]["content"] == "7 — 在庫 ✓"
    record = await harness.runs.get(finished.run_id)
    assert record is not None and record.output == UNICODE
    assert model.turns == []  # every planned call made, nothing asked twice


@pytest.mark.parametrize("before", [[], [("stock", {"sku": "a"})]], ids=["first", "after_a_call"])
async def test_an_empty_reply_ends_the_run_saying_it_had_no_text(
    harness: Harness, build: Any, before: list[Any]
) -> None:
    model = ScriptedChat([*before, ""])
    agent = arm(harness.wrap(build(model), id="empty", without={"schedules"}), [stock])
    result = await agent.run("how many a?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None, result
    assert result.error.code == NO_ANSWER and "had no text" in result.error.message


async def test_a_tool_that_fails_is_read_by_the_model_and_the_run_answers(
    harness: Harness, build: Any
) -> None:
    @tool(side_effects="read")
    def prices(sku: str) -> str:
        """The price of a SKU."""
        raise ValueError(f"no price list for {sku} — 価格なし")

    model = ScriptedChat([("prices", {"sku": "a"}), "I could not read the price."])
    agent = arm(harness.wrap(build(model), id="failing", without={"schedules"}), [prices])
    events = [e async for e in agent.stream("price of a?", user="u")]
    assert events[-1].type is RunEventType.RUN_FINISHED
    assert events[-1].data["result"] == "I could not read the price."
    told = model.requests[1]["messages"][-1]
    assert told["role"] == "tool" and "no price list for a — 価格なし" in told["content"]


class Answer(BaseModel):
    sku: str
    units: int
    note: str


async def test_a_structured_answer_is_the_runs_output(harness: Harness, build: Any) -> None:
    model = ScriptedChat(
        [("stock", {"sku": "a"}), ("Answer", {"sku": "a", "units": 7, "note": "在庫 ✓"})]
    )
    agent = arm(harness.wrap(build(model, response_format=Answer), id="structured"), [stock])
    result = await agent.run("how many a?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == Answer(sku="a", units=7, note="在庫 ✓")
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == {"sku": "a", "units": 7, "note": "在庫 ✓"}


@tool(side_effects="read")
def skus() -> str:
    """Every SKU, one row a line."""
    return "sku,units\n" + "\n".join(f"SKU-{n:05d},{n % 7}" for n in range(20_000))


async def _kept(harness: Harness, build: Any, then: list[Any]) -> tuple[ScriptedChat, Any]:
    """A run whose first call returns :func:`skus` (the result kept, a preview in its place),
    the model's next calls ``then`` made with the path the preview gives; the model, and the
    run's result."""
    probe = ScriptedChat([("skus", {}), "seen"])
    await arm(harness.wrap(build(probe), id="probe", without={"schedules"}), [skus]).run(
        "x", user="u"
    )
    preview = str(probe.requests[1]["messages"][-1]["content"])
    assert len(preview) < 10_000 and "SKU-00000,0" in preview, preview[:500]
    [path] = set(re.findall(r"/large_tool_results/[\w-]+", preview))
    # a model of the same script: its call has the same id, so its result the same path
    model = ScriptedChat([("skus", {}), *[(n, {**a, "file_path": path}) for n, a in then], "ok"])
    agent = arm(harness.wrap(build(model), id="kept", without={"schedules"}), [skus])
    return model, await agent.run("x", user="u")


async def test_a_large_result_is_read_to_its_last_row(harness: Harness, build: Any) -> None:
    model, result = await _kept(harness, build, [("read_file", {"offset": 19_995, "limit": 10})])
    assert result.status is RunStatus.SUCCESS, result.error
    read = model.requests[2]["messages"][-1]["content"]
    assert "SKU-19999,0" in read and "SKU-19994" in read, read


async def test_a_large_result_is_queried_over_every_row(harness: Harness, build: Any) -> None:
    sql = "SELECT count(*) AS n, sum(units) AS total FROM result"
    model, result = await _kept(harness, build, [("query_result", {"sql": sql})])
    assert result.status is RunStatus.SUCCESS, result.error
    counted = model.requests[2]["messages"][-1]["content"]
    total = sum(n % 7 for n in range(20_000))
    assert "20000" in counted and str(total) in counted, counted


async def test_a_write_waits_for_approval_and_runs_once(harness: Harness, build: Any) -> None:
    ran: list[str] = []

    @tool(side_effects="write")
    def reorder(sku: str) -> str:
        """Reorder a SKU."""
        ran.append(sku)
        return f"reordered {sku} ✓"

    model = ScriptedChat([("reorder", {"sku": "a"}), "Reordered."])
    agent = arm(harness.wrap(build(model), id="approved", without={"schedules"}), [reorder])
    paused = await agent.run("reorder a", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    assert paused.interrupt.reason is InterruptReason.APPROVAL and ran == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="ops")
    assert done.status is RunStatus.SUCCESS and done.answer == "Reordered.", done.error
    assert ran == ["a"]
    assert model.requests[-1]["messages"][-1]["content"] == "reordered a ✓"


async def test_a_rejected_write_never_runs_and_the_model_reads_why(
    harness: Harness, build: Any
) -> None:
    ran: list[str] = []

    @tool(side_effects="write")
    def reorder(sku: str) -> str:
        """Reorder a SKU."""
        ran.append(sku)
        return "reordered"

    model = ScriptedChat([("reorder", {"sku": "a"}), "Not reordered."])
    agent = arm(harness.wrap(build(model), id="rejected", without={"schedules"}), [reorder])
    paused = await agent.run("reorder a", user="u")
    assert paused.interrupt is not None, paused
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", reviewer="ops", comment="stock is enough"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "Not reordered.", done.error
    assert ran == []
    assert "stock is enough" in model.requests[-1]["messages"][-1]["content"]


def _slow(started: asyncio.Event) -> Any:
    @tool(side_effects="read")
    async def slow(sku: str) -> str:
        """A slow read."""
        started.set()
        await asyncio.sleep(30)
        return "late"

    return slow


async def test_a_cancelled_run_ends_cancelled_with_its_reason(harness: Harness, build: Any) -> None:
    started = asyncio.Event()
    model = ScriptedChat([("slow", {"sku": "a"}), "never"])
    agent = arm(harness.wrap(build(model), id="cancelled", without={"schedules"}), [_slow(started)])
    events: list[Any] = []

    async def watch() -> None:
        async for event in agent.stream("x", user="u"):
            events.append(event)

    streaming = asyncio.create_task(watch())
    async with asyncio.timeout(10):
        await started.wait()
    record = await agent.cancel(events[0].run_id, reason="changed my mind — 取消")
    assert record.status is RunStatus.CANCELLED, record
    async with asyncio.timeout(10):
        await streaming
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.outcome is not None
    assert finished.outcome.value == "cancelled", finished
    assert finished.data["reason"] == "changed my mind — 取消"
    assert model.turns == ["never"]  # the model was not asked again


async def test_a_run_past_its_time_limit_ends_timeout(harness: Harness, build: Any) -> None:
    started = asyncio.Event()
    model = ScriptedChat([("slow", {"sku": "a"}), "never"])
    agent = arm(harness.wrap(build(model), id="timeout", without={"schedules"}), [_slow(started)])
    async with asyncio.timeout(10):
        result = await agent.run("x", user="u", run_timeout=0.5)
    assert result.status is RunStatus.TIMEOUT, result


async def test_a_run_past_its_budget_answers_with_what_it_has(harness: Harness, build: Any) -> None:
    calls: list[str] = []

    @tool(side_effects="read")
    def count(sku: str) -> str:
        """Count a SKU."""
        calls.append(sku)
        return "1"

    model = ScriptedChat([("count", {"sku": s}) for s in "abc"] + ["counted"])
    agent = arm(harness.wrap(build(model), id="budget", without={"schedules"}), [count])
    result = await agent.run("count", user="u", budget=Budget(max_tool_calls=2))
    assert calls == ["a", "b"], calls  # the third call is over the budget: never run
    # told so, offered no tool, the model answers with what it has: the run is partial
    assert result.status is RunStatus.PARTIAL and result.answer == "counted", result
    assert result.error is not None and result.error.code == "BUDGET_EXCEEDED"
    last = model.requests[-1]
    assert "budget is spent" in last["messages"][-1]["content"]
    assert last["tool_choice"] == "none"


async def test_a_run_with_its_memory_service_down_still_answers(
    memory_harness: Harness, memory_service: FakeMemoryService, build: Any
) -> None:
    for down in ("context", "agent_tools", "tool_hints", "messages", "record_tool"):
        memory_service.failures[down] = 503
    model = ScriptedChat([("stock", {"sku": "a"}), UNICODE])
    agent = arm(memory_harness.wrap(build(model), id="no-memory", without={"schedules"}), [stock])
    events = [e async for e in agent.stream("how many a?", user="u")]
    assert events[-1].type is RunEventType.RUN_FINISHED, events[-1]
    assert events[-1].data["result"] == UNICODE
    warnings = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any("no memory context" in str(w.get("message")) for w in warnings), warnings
