"""An interrupt's decision is feedback on the tool call exactly when the resume took effect,
sent with a fixed id the client may retry and the service stores once."""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.memory import FakeMemoryService
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.contracts.ids import stable_id
from trellis.harness.clients.memory import Memory
from trellis.harness.clients.runs import RunStoreError


@tool(side_effects="irreversible")
def wipe(disk: str) -> str:
    """Wipe a disk."""
    return f"wiped {disk}"


async def _wipes(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("wipe", disk="d1")


async def _asks(input: str, agent: Runtime) -> Any:
    return await agent.ask("Which customer?")


def _decided(memory_service: FakeMemoryService) -> list[dict[str, Any]]:
    return [f for f in memory_service.stored_feedback.values() if f["target_kind"] == "tool_call"]


@pytest.mark.parametrize(
    ("decision", "answer", "verdict"),
    [("approve", None, "approve"), ("reject", None, "reject"), ("edit", {"disk": "d2"}, "edit")],
)
async def test_each_decision_on_a_tool_call_is_feedback_once_with_a_fixed_id(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    decision: str,
    answer: Any,
    verdict: str,
) -> None:
    agent = memory_harness.wrap(_wipes, id="ops", tools=[wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, decision, answer=answer, reviewer="boss")
    await memory_harness.writes.drain()
    [sent] = [c for c in memory_service.named("feedback") if c.body["target_kind"] == "tool_call"]
    expected = stable_id(paused.run_id, paused.interrupt.interrupt_id, prefix="fb_")
    assert sent.body["feedback_id"] == expected
    assert sent.idempotency_key == expected, "the id is the key, so the client may retry"
    assert sent.body["verdict"] == verdict and sent.body["reviewer"] == "boss"
    assert sent.body["metadata"]["tool"] == "wipe"
    assert sent.body["metadata"]["args"] == {"disk": "d1"}
    assert sent.body["metadata"]["decision_seconds"] >= 0
    assert sent.body.get("correction") == (answer if decision == "edit" else None)
    assert memory_harness.writes.failed == 0


@pytest.mark.parametrize("decision", ["answer", "cancel"])
async def test_an_answer_or_a_cancel_judges_no_tool_call(
    memory_harness: Harness, memory_service: FakeMemoryService, decision: str
) -> None:
    agent = memory_harness.wrap(_asks, id="ops")
    paused = await agent.run("go", user="u")
    assert paused.interrupt is not None
    await agent.resume(
        paused.interrupt.interrupt_id,
        decision,
        answer="Acme" if decision == "answer" else None,
        reviewer="boss",
    )
    await memory_harness.writes.drain()
    assert _decided(memory_service) == []


async def test_a_cancelled_tool_call_is_not_feedback(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = memory_harness.wrap(_wipes, id="ops", tools=[wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    result = await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="boss")
    await memory_harness.writes.drain()
    assert result.status is RunStatus.CANCELLED
    assert _decided(memory_service) == []


async def test_a_resume_the_run_store_refuses_sends_no_feedback(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another reviewer answered first: the store refuses this one, and the memory service
    must not learn from a decision that never took effect."""
    agent = memory_harness.wrap(_wipes, id="ops", tools=[wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None

    async def refused(resolution: Any) -> Any:
        raise RunStoreError("run is RUNNING, not waiting on an answer")

    monkeypatch.setattr(memory_harness.runs, "resumed", refused)
    with pytest.raises(RunStoreError):
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="boss")
    await memory_harness.writes.drain()
    assert _decided(memory_service) == []
    assert [
        c for c in memory_service.named("feedback") if c.body["target_kind"] == "tool_call"
    ] == []


async def test_a_second_answer_to_the_same_pause_sends_nothing_more(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = memory_harness.wrap(_wipes, id="ops", tools=[wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="boss")
    with pytest.raises(Exception, match="not paused"):
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="other")
    await memory_harness.writes.drain()
    [only] = _decided(memory_service)
    assert only["verdict"] == "reject"


async def test_a_brief_memory_outage_is_retried_and_counted_once(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.fail_times["feedback"] = 2
    async with Harness(config=Settings(memory_url="http://memory.test")) as h:
        h.memory = Memory("http://memory.test", None, client=memory_service.client(max_retries=3))
        agent = h.wrap(_wipes, id="ops", tools=[wipe])
        paused = await agent.run("wipe d1", user="u")
        assert paused.interrupt is not None
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="boss")
        await h.writes.drain()
        assert h.writes.failed == 0
    assert memory_service.fail_times["feedback"] == 0, "both failures were retried through"
    [stored] = _decided(memory_service)
    assert stored["verdict"] == "approve"


async def test_a_lasting_memory_outage_is_reported_and_the_run_still_continues(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail.add("feedback")
    agent = memory_harness.wrap(_wipes, id="ops", tools=[wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="boss")
    await memory_harness.writes.drain()
    assert result.status is RunStatus.SUCCESS and result.answer == "wiped d1"
    # the decision's feedback and the run's outcome both went to the downed route
    assert memory_harness.writes.failed == 2
    assert _decided(memory_service) == []
