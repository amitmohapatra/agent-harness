"""OpenAI Agents SDK: the real ``Runner`` with a scripted ``Model``."""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent, function_tool

from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Harness, current, tool
from trellis.contracts import InterruptReason, RunEventType, RunStatus

done: list[str] = []


@tool(side_effects="irreversible")
def ship(order: str) -> str:
    """Ship an order."""
    done.append(order)
    return f"shipped {order}"


@tool(side_effects="read")
async def confirm(order: str) -> Any:
    """Ask the customer to confirm an order."""
    runtime = current()
    assert runtime is not None
    return await runtime.ask(f"Ship {order}?", options=["yes", "no"])


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


async def test_the_agent_gets_the_harness_tools_next_to_its_own(harness: Harness) -> None:
    @function_tool
    def eta(order: str) -> str:
        return "tomorrow"

    model = ScriptedModel(
        [("eta", {"order": "o1"}), ("ship", {"order": "o1"}), "shipped, arriving tomorrow"]
    )
    target = Agent(name="shipper", instructions="You ship orders.", model=model, tools=[eta])
    agent = harness.wrap(target, id="shipper", tools=[ship], approve={"ship": False})
    result = await agent.run("ship o1", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "shipped, arriving tomorrow"
    assert done == ["o1"]
    assert [t.name for t in target.tools] == ["eta"]  # the team's agent is untouched


async def test_an_approval_pauses_and_a_resume_reruns_against_the_journal(harness: Harness) -> None:
    model = ScriptedModel([("ship", {"order": "o1"}), ("ship", {"order": "o1"}), "shipped"])
    agent = harness.wrap(Agent(name="shipper", model=model), id="shipper", tools=[ship])
    paused = await agent.run("ship o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.reason is InterruptReason.APPROVAL and done == []
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and done == ["o1"]


async def test_ask_inside_a_tool_returns_the_answer_on_resume(harness: Harness) -> None:
    model = ScriptedModel(
        [("confirm", {"order": "o1"}), ("confirm", {"order": "o1"}), "customer said yes"]
    )
    agent = harness.wrap(Agent(name="c", model=model), id="confirmer", tools=[confirm])
    paused = await agent.run("check o1", user="u1")
    assert paused.interrupt is not None and paused.interrupt.options == ["yes", "no"]
    finished = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u1"
    )
    assert finished.answer == "customer said yes"
    tool_output = model.inputs[-1][-1]
    assert tool_output["type"] == "function_call_output" and tool_output["output"] == "yes"


async def test_the_sdks_own_approval_pauses_and_resumes_its_run_state(harness: Harness) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return "sent"

    model = ScriptedModel([("send", {"order": "o9"}), "sent it"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.args == {
        "order": "o9"
    }
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "sent it"
    assert shipped == ["o9"]
    assert len(model.inputs) == 2  # continued from the SDK's state, not re-run


async def test_a_rejected_sdk_approval_does_not_run_the_tool(harness: Harness) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        raise AssertionError("must not run")

    model = ScriptedModel([("send", {"order": "o9"}), "not sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.interrupt is not None
    finished = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u1")
    assert finished.answer == "not sent"


async def test_streaming_emits_text_deltas(harness: Harness) -> None:
    agent = harness.wrap(Agent(name="s", model=ScriptedModel(["all good here"])), id="streamer")
    events = [e async for e in agent.stream("hi", user="u1")]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == "all good here" and len(deltas) == 3


async def test_the_context_is_a_system_message_and_memory_tools_are_added(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel([("memory_search", {"query": "prefs"}), "email"])
    agent = memory_harness.wrap(Agent(name="m", model=model), id="m", memory="read_write")
    assert (await agent.run("how to reach me?", user="u1")).answer == "email"
    assert model.inputs[0][0] == {"role": "system", "content": memory_service.context_text}
    assert memory_service.named("call_agent_tool")[0].path["name"] == "memory_search"
