"""Everything the harness hands back is the platform's contract: a run's interrupt, error and
status, its events, its run record, its schedule, the inbox's interrupts, the errors it raises
— each the agent-contracts type (or, where a service's stored record is richer, a record
consistent with it), and each survives its own JSON round trip, the form it crosses a process
boundary in."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tests.support.memory import FakeMemoryService
from trellis import Harness, Result, Runtime, tool
from trellis.contracts import (
    AgentError,
    ConfigurationError,
    Feedback,
    Interrupt,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    Schedule,
    ToolError,
)
from trellis.harness.tools.sources import FunctionTool


def round_trips(value: BaseModel) -> None:
    """``value`` validates as its own type from its JSON form, and is equal to itself."""
    assert type(value).model_validate_json(value.model_dump_json()) == value


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


async def refunds(input: str, agent: Runtime) -> Any:
    if input == "fail":
        raise ValueError("no such order")
    return await agent.tools.call("refund", order=input)


async def test_a_runs_result_carries_contracts_types(memory_harness: Harness) -> None:
    agent = memory_harness.wrap(refunds, id="refunds", tools=[refund])
    paused = await agent.run("o-1", user="ada")
    assert isinstance(paused, Result) and isinstance(paused.status, RunStatus)
    assert isinstance(paused.interrupt, Interrupt)
    round_trips(paused.interrupt)
    round_trips(paused)
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
    assert done.status is RunStatus.SUCCESS and done.answer == "refunded o-1"
    failed = await agent.run("fail", user="ada")
    assert isinstance(failed.error, AgentError) and failed.error.code == "ValueError"
    round_trips(failed.error)

    record = await memory_harness.runs.get(done.run_id)
    assert isinstance(record, RunRecord)
    round_trips(record)
    again = await agent.run("o-2", user="ada")
    [waiting] = [r for r in await memory_harness.inbox() if r.run_id == again.run_id]
    assert isinstance(waiting.awaiting, Interrupt)


async def test_every_streamed_event_is_a_contracts_run_event(memory_harness: Harness) -> None:
    agent = memory_harness.wrap(refunds, id="refunds", tools=[refund])
    events = [e async for e in agent.stream("o-1", user="ada")]
    assert events and all(isinstance(e, RunEvent) for e in events)
    for event in events:
        round_trips(event)
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED
    assert Interrupt.model_validate(finished.data["interrupt"]).question.startswith("Approve")


async def test_a_queued_run_a_schedule_and_feedback_are_contracts_records(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = memory_harness.wrap(refunds, id="refunds", tools=[refund])
    handle = await agent.start("o-3", user="ada")
    queued = await handle.status()
    assert isinstance(queued, RunRecord) and queued.status is RunStatus.QUEUED
    schedule = await agent.schedule("daily", "o-4", on_behalf_of="ada")
    assert isinstance(schedule, Schedule)
    round_trips(schedule)

    done = await agent.run("fail", user="ada")
    stored = await memory_harness.feedback(done.run_id, "correct", correction="o-5")
    assert stored is not None
    # the memory service's stored record is the contracts Feedback, plus how it was reviewed
    contract = Feedback.model_validate(stored.model_dump(include=set(Feedback.model_fields)))
    assert (contract.target_id, contract.verdict.value, contract.source.value) == (
        done.run_id,
        "correct",
        "human",
    )
    await memory_harness.writes.drain()
    # an approval's feedback, as the harness sends it, is a contracts Feedback
    paused = await agent.run("o-6", user="ada")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", answer="no", reviewer="cfo")
    await memory_harness.writes.drain()
    [sent] = [
        c.body for c in memory_service.named("feedback") if c.body.get("target_kind") == "tool_call"
    ]
    assert Feedback.model_validate(sent).verdict.value == "reject"


async def test_the_errors_it_raises_are_contracts_errors(harness: Harness) -> None:
    with pytest.raises(ConfigurationError):
        harness.wrap(object(), id="nothing")
    agent = harness.wrap(refunds, id="refunds", tools=[refund])
    with pytest.raises(ConfigurationError):
        await agent.resume("run_missing.1.1", "approve", reviewer="cfo")
    assert isinstance(refund, FunctionTool)
    with pytest.raises(ToolError):  # a harness tool outside a run
        from trellis.harness.tools import bridge

        [made] = await refund.resolve()
        await bridge.call(made, {"order": "o-1"})
