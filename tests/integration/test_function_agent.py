"""The whole pipeline on the smallest target: an async function. Pauses, resumes, tiers,
replay, streaming, errors, cancellation — everything framework-independent."""

from __future__ import annotations

from typing import Any

import pytest

from trellis import Harness, Runtime, tool
from trellis.contracts import (
    ConfigurationError,
    InterruptReason,
    RunEventType,
    RunOutcome,
    RunStatus,
)

calls: list[str] = []


@tool(side_effects="irreversible")
async def refund(order: str, amount: float) -> str:
    """Refund an order."""
    calls.append(f"refund:{order}")
    return f"refunded {amount} on {order}"


@tool(side_effects="write")
def note(text: str) -> str:
    """Write a note."""
    calls.append(f"note:{text}")
    return "noted"


@tool(side_effects="read")
def lookup(order: str) -> dict[str, Any]:
    """Look an order up."""
    calls.append(f"lookup:{order}")
    return {"order": order, "amount": 40}


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()


async def test_a_run_returns_the_answer(harness: Harness) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input.upper()

    result = await harness.wrap(echo, id="echo").run("hi", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "HI"
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == "HI" and record.user_id == "u1"


async def test_ask_pauses_and_resume_returns_the_answer_where_it_was_asked(
    harness: Harness,
) -> None:
    async def planner(input: str, agent: Runtime) -> str:
        colour = await agent.ask("Which colour?", options=["red", "blue"], assignee="role:design")
        size = await agent.ask("Which size?", expects={"type": "integer"})
        return f"{colour}/{size}"

    agent = harness.wrap(planner, id="planner")
    first = await agent.run("paint", user="u1")
    assert first.status is RunStatus.PAUSED and first.interrupt is not None
    assert first.interrupt.reason is InterruptReason.CHOICE and first.interrupt.ui == "choice"
    assert first.interrupt.assignee == "role:design"
    assert [r.run_id for r in await harness.inbox("role:design")] == [first.run_id]
    assert await harness.inbox("user:u1") == []
    second = await agent.resume(
        first.interrupt.interrupt_id, "answer", answer="blue", reviewer="u1"
    )
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert second.interrupt.ui == "form" and second.run_id == first.run_id
    done = await agent.resume(second.interrupt.interrupt_id, "answer", answer=3, reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "blue/3"


async def test_the_risk_tiers(harness: Harness) -> None:
    async def worker(input: str, agent: Runtime) -> Any:
        found = await agent.tools.call("lookup", order="o1")
        await agent.tools.call("note", text="checked")
        return await agent.tools.call("refund", order="o1", amount=found["amount"])

    agent = harness.wrap(worker, id="refunds", tools=[lookup, note, refund])
    events = [e async for e in agent.stream("refund o1", user="u1")]
    names = [e.data.get("name") for e in events if e.type is RunEventType.CUSTOM]
    assert names == ["tool_notice"]  # the write is announced; the read is not
    finished = events[-1]
    assert finished.outcome is RunOutcome.INTERRUPT
    interrupt = finished.data["interrupt"]
    assert interrupt["reason"] == "APPROVAL" and interrupt["tool_call"]["args"] == {
        "order": "o1",
        "amount": 40,
    }
    assert calls == ["lookup:o1", "note:checked"]

    done = await agent.resume(interrupt["interrupt_id"], "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "refunded 40.0 on o1"
    # the re-run replayed the read and the write from the journal: each ran once
    assert calls == ["lookup:o1", "note:checked", "refund:o1"]


async def test_an_approver_can_edit_or_reject(harness: Harness) -> None:
    async def worker(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("refund", order="o1", amount=500)

    agent = harness.wrap(worker, id="refunds", tools=[refund])
    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    edited = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"order": "o1", "amount": 50}, reviewer="boss"
    )
    assert edited.answer == "refunded 50.0 on o1"

    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    rejected = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="boss")
    assert rejected.status is RunStatus.SUCCESS
    assert rejected.answer == "refund was not run: the approver rejected it"

    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    told = await agent.resume(
        paused.interrupt.interrupt_id, "reject", answer="over the limit", reviewer="boss"
    )
    assert told.answer == "refund was not run: the approver rejected it (over the limit)"


async def test_what_is_asked_decides_how_it_is_shown_and_the_user_answers_by_default(
    harness: Harness,
) -> None:
    async def reviewer(input: str, agent: Runtime) -> Any:
        rows = await agent.ask("Check these lines", table=[{"sku": "a", "qty": 2}])
        text = await agent.ask(
            "Accept the rewrite?", diff=("old text", "new text"), expects={"type": "string"}
        )
        plain = await agent.ask("Anything else?")
        return [rows, text, plain]

    agent = harness.wrap(reviewer, id="reviewer")
    paused = await agent.run("x", user="u1")
    first = paused.interrupt
    assert first is not None and first.ui == "table" and first.reason is InterruptReason.QUESTION
    assert first.payload == {"table": [{"sku": "a", "qty": 2}]}
    assert first.assignee == "user:u1"  # the run's user, when nobody else is named
    assert [r.run_id for r in await harness.inbox("user:u1")] == [paused.run_id]
    paused = await agent.resume(first.interrupt_id, "answer", answer="ok", reviewer="u1")
    second = paused.interrupt
    assert second is not None and second.ui == "diff" and second.reason is InterruptReason.REVIEW
    assert second.payload == {"diff": {"before": "old text", "after": "new text"}}
    paused = await agent.resume(
        second.interrupt_id, "edit", answer={"text": "newer"}, reviewer="u1"
    )
    third = paused.interrupt
    assert third is not None and third.ui == "form"
    done = await agent.resume(third.interrupt_id, "answer", answer="no", reviewer="u1")
    assert done.answer == ["ok", {"text": "newer"}, "no"]


async def test_a_cancel_ends_the_run(harness: Harness) -> None:
    async def asker(input: str, agent: Runtime) -> str:
        return await agent.ask("continue?")

    agent = harness.wrap(asker, id="asker")
    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    cancelled = await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="u1")
    assert cancelled.status is RunStatus.CANCELLED
    record = await harness.runs.get(paused.run_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_resume_must_answer_the_open_question(harness: Harness) -> None:
    async def asker(input: str, agent: Runtime) -> str:
        return await agent.ask("continue?")

    agent = harness.wrap(asker, id="asker")
    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    with pytest.raises(ConfigurationError, match="waits on"):
        await agent.resume(f"{paused.run_id}.9.9", "answer", answer="y", reviewer="u1")
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="y", reviewer="u1")
    with pytest.raises(ConfigurationError, match="SUCCESS"):
        await agent.resume(paused.interrupt.interrupt_id, "answer", answer="y", reviewer="u1")


async def test_a_failing_agent_is_an_error_result(harness: Harness) -> None:
    async def broken(input: str, agent: Runtime) -> str:
        raise ValueError("bad input")

    result = await harness.wrap(broken, id="broken").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == "bad input"
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.status is RunStatus.ERROR


async def test_a_failing_tool_is_shown_to_the_agent_not_raised(harness: Harness) -> None:
    @tool(side_effects="read")
    def flaky() -> str:
        raise RuntimeError("timeout upstream")

    async def worker(input: str, agent: Runtime) -> str:
        return await agent.tools.call("flaky")

    result = await harness.wrap(worker, id="flaky", tools=[flaky]).run("x", user="u1")
    assert result.answer == "flaky failed: timeout upstream"


async def test_streaming_and_closing_the_stream_cancels_the_run(harness: Harness) -> None:
    async def asker(input: str, agent: Runtime) -> str:
        agent.log("thinking", step=1)
        return await agent.ask("go?")

    agent = harness.wrap(asker, id="asker")
    stream = agent.stream("x", user="u1")
    first = await anext(stream)
    assert first.type is RunEventType.RUN_STARTED
    log = await anext(stream)
    assert log.data == {"name": "log", "message": "thinking", "step": 1}
    await stream.aclose()


async def test_wrap_refuses_what_it_cannot_do(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return input

    with pytest.raises(ConfigurationError, match="cannot wrap"):
        harness.wrap(object(), id="o")
    harness.wrap(fn, id="dup")
    with pytest.raises(ConfigurationError, match="already wrapped"):
        harness.wrap(fn, id="dup")
    with pytest.raises(ConfigurationError, match="user="):
        await harness.wrap(fn, id="nouser").run("x", user="")


async def test_a_tool_outside_a_run_is_refused() -> None:
    [resolved] = await refund.resolve()
    from trellis.harness.tools.bridge import call

    with pytest.raises(Exception, match="inside a Harness run"):
        await call(resolved, {"order": "o", "amount": 1})


async def test_a_resume_without_the_journal_still_answers_the_right_question(
    harness: Harness,
) -> None:
    """Another process resumes: the store kept the interrupt and the answer, not the journal."""

    async def asker(input: str, agent: Runtime) -> str:
        return await agent.ask("Which colour?")

    agent = harness.wrap(asker, id="asker")
    paused = await agent.run("x", user="u1")
    assert paused.interrupt is not None
    record, resolution = await agent._resolution(
        paused.interrupt.interrupt_id, "answer", "blue", "u1", tenant="default"
    )
    forgetful = record.model_copy(update={"metadata": {}})
    done = await agent._continue(forgetful, resolution)
    assert done.status is RunStatus.SUCCESS and done.answer == "blue"
