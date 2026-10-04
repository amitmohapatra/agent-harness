"""OpenAI Agents SDK: the real ``Runner`` with a scripted ``Model``."""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent, function_tool

from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
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
    agent = harness.wrap(target, id="shipper", tools=[tool(ship.fn, side_effects="write")])
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
    agent = memory_harness.wrap(Agent(name="m", model=model), id="m")
    assert (await agent.run("how to reach me?", user="u1")).answer == "email"
    assert model.inputs[0][0] == {"role": "system", "content": memory_service.context_text}
    assert memory_service.named("call_agent_tool")[0].path["name"] == "memory_search"


async def test_the_hints_narrow_the_tools_each_turn_and_the_teams_own_stay(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @function_tool
    def eta(order: str) -> str:
        return "tomorrow"

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}"

        return tool(lookup, name=f"t{i}", side_effects="read")

    memory_service.candidates = ["t2"]
    memory_service.candidates_for = {"find t4": ["t4"]}
    model = ScriptedModel([("tool_search", {"task": "find t4"}), ("t4", {"key": "k"}), "done"])
    target = Agent(name="n", model=model, tools=[eta])
    agent = memory_harness.wrap(target, id="n", tools=[make(i) for i in range(6)])
    assert (await agent.run("find t2", user="u1")).answer == "done"
    memory_tools = MEMORY_TOOLS
    assert model.tools[0] == ["eta", "t2", *memory_tools]
    assert model.tools[1] == ["eta", "t2", "t4", *memory_tools]  # tool_search offered t4


async def test_two_sdk_approvals_in_one_turn_are_answered_one_at_a_time(
    harness: Harness,
) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return f"sent {order}"

    model = ScriptedModel([[("send", {"order": "a"}), ("send", {"order": "b"})], "both sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    first = await agent.run("send a and b", user="u1")
    assert first.interrupt is not None and first.interrupt.tool_call is not None
    assert first.interrupt.tool_call.args == {"order": "a"}
    # approving the first leaves the second waiting: the run pauses on it next
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="u1")
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert second.interrupt.tool_call is not None
    assert second.interrupt.tool_call.args == {"order": "b"}
    assert shipped == ["a"]
    done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "both sent"
    assert shipped == ["a", "b"]
