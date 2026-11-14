"""Handoffs between ``ReAct`` agents through ``langgraph-swarm`` (its ``create_handoff_tool``,
``SwarmState`` and ``add_active_agent_router``, as the library gives them): the hand-over is a
governed, recorded call that asks nobody, the agent handed to answers — governed as ever —, the
conversation's next turn is its own, and a pause resumes in place."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool
from langgraph.graph import StateGraph
from langgraph_swarm import SwarmState, add_active_agent_router, create_handoff_tool

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.adapters.base import HELPER
from trellis.harness.middleware import RunCheckpointer, spec_of

refunded: list[str] = []


@lc_tool
def refund(order: str) -> str:
    """Refund an order."""
    refunded.append(order)
    return f"refunded {order}"


refund.metadata = {"side_effects": "irreversible"}


@pytest.fixture(autouse=True)
def nothing_refunded() -> None:
    refunded.clear()


def team(harness: Harness, front: list[Any], back: list[Any]) -> tuple[Any, Any, Any]:
    """A front desk that hands refunds to billing, as a langgraph-swarm graph of two ``ReAct``
    agents, wrapped: the agent, and each member's scripted model."""
    desk, billing = ScriptedChat(front), ScriptedChat(back)
    members = {
        "desk": ReAct("You are the front desk.", desk, tools=[handoff("billing")]),
        "billing": ReAct("You do refunds.", billing, tools=[refund, handoff("desk")]),
    }
    builder = StateGraph(SwarmState)
    for name, member in members.items():
        builder.add_node(name, member, destinations=tuple(set(members) - {name}))
    swarm = add_active_agent_router(builder, route_to=list(members), default_active_agent="desk")
    graph = swarm.compile(checkpointer=RunCheckpointer())
    return harness.wrap(graph, id="support"), desk, billing


def handoff(to: str) -> Any:
    return create_handoff_tool(agent_name=to)


def calls(events: list[Any]) -> list[tuple[str, str]]:
    return [
        (e.data["tool"], e.data["status"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


def test_a_handoff_tool_is_a_helper_that_only_hands_the_work_on() -> None:
    spec = spec_of(handoff("billing"))
    assert spec.name == "transfer_to_billing"
    assert HELPER in spec.tags and spec.side_effects == "read"
    assert HELPER not in spec_of(refund).tags  # any other tool: as it declares itself


async def test_a_handoff_is_recorded_asks_nobody_and_the_other_agent_answers(
    harness: Harness,
) -> None:
    agent, desk, billing = team(harness, [("transfer_to_billing", {})], ["Billing here."])
    events = [e async for e in agent.stream("I need a refund", user="u", thread="t")]
    assert calls(events) == [("transfer_to_billing", "ok")]
    assert not [e for e in events if e.type is RunEventType.INTERRUPT]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == "Billing here."
    assert len(desk.requests) == 1 and len(billing.requests) == 1
    # billing reads the conversation so far, the hand-over included
    said = [m["role"] for m in billing.requests[0]["messages"]]
    assert said == ["system", "user", "assistant", "tool"]


async def test_the_next_turn_is_the_agent_handed_to_and_says_so(harness: Harness) -> None:
    agent, desk, _ = team(
        harness, [("transfer_to_billing", {})], ["Billing here.", "Still billing."]
    )
    await agent.run("I need a refund", user="u", thread="t")
    events = [e async for e in agent.stream("are you there?", user="u", thread="t")]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == "Still billing."
    assert len(desk.requests) == 1  # the front desk was not asked again
    active = [
        e.data["agent"]
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "active_agent"
    ]
    assert active == ["billing"]
    # another conversation starts at the front desk
    other = await agent.run("hello", user="u", thread="t2")
    assert other.status is RunStatus.ERROR  # its model has no turn left: it was asked
    assert len(desk.requests) == 2


async def test_the_agent_handed_to_is_governed_and_its_pause_resumes_in_place(
    harness: Harness,
) -> None:
    agent, desk, billing = team(
        harness,
        [("transfer_to_billing", {})],
        [("refund", {"order": "o1"}), "Refunded o1."],
    )
    paused = await agent.run("refund o1", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "refund" and refunded == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "Refunded o1."
    assert refunded == ["o1"]
    # in place: neither the hand-over nor billing's first step was asked of a model again
    assert len(desk.requests) == 1 and len(billing.requests) == 2
