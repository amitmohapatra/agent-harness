"""Sub-agents against agent-runs (and its ticker) and the memory service: a ``ReAct`` planner
queued for workers delegates to two children at once; one asks a person, whose answer comes
from the inbox through the planner; the worker dies inside that child after its side effect,
and the planner's next attempt continues the child without repeating it."""

from __future__ import annotations

import uuid

import pytest

from tests.live.conftest import needs_memory, needs_runs
from tests.live.test_live_reliability import Crash, claimed_again, died
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness.subagents import SUBAGENT

pytestmark = [pytest.mark.live, needs_runs, needs_memory]


async def test_a_planner_with_two_children_one_asking_survives_a_killed_worker(
    harness: Harness,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    looked: list[str] = []
    charged: list[str] = []
    crashes = [Crash()]

    @tool(name=f"flights_{suffix}", side_effects="read")
    async def flights(city: str) -> str:
        """The direct flights to a city."""
        looked.append(city)
        return f"{city}: 2 direct flights a day"

    @tool(name=f"book_{suffix}", side_effects="write")
    async def book(city: str, budget: str) -> str:
        """Book a hotel room (charges the card)."""
        charged.append(f"{city} {budget}")
        return f"booked {city} ({budget})"

    @tool(name=f"confirm_{suffix}", side_effects="read")
    async def confirm(booking: str) -> str:
        """Confirm a booking (the worker dies here, once)."""
        if crashes:
            raise crashes.pop()
        return f"confirmed: {booking}"

    async def booker(input: str, agent: Runtime) -> str:
        budget = await agent.ask("Which budget?", options=["low", "high"])
        booking = await agent.tools.call(book.spec.name, city=input, budget=budget)
        return str(await agent.tools.call(confirm.spec.name, booking=booking))

    scout_model = ScriptedChat([(flights.spec.name, {"city": "Oslo"}), "Oslo: 2 a day"])
    scout = harness.wrap(
        ReAct(system="You find flights.", model=scout_model),
        id=f"live-scout-{suffix}",
        tools=[flights],
    )
    hotel = harness.wrap(booker, id=f"live-booker-{suffix}", tools=[book, confirm])
    model = ScriptedChat(
        [[(scout.id, {"message": "Oslo"}), (hotel.id, {"message": "Oslo"})], "Planned."]
    )
    planner = harness.wrap(
        ReAct(system="You plan trips.", model=model),
        id=f"live-planner-{suffix}",
        tools=[scout.as_tool(), hotel.as_tool()],
    )
    handle = await planner.start("Plan a weekend in Oslo", user="live-user")
    assert await claimed_again(harness, planner)
    paused = await handle.result(timeout=30)
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    assert paused.interrupt.question == "Which budget?"
    asker = (paused.interrupt.payload or {})[SUBAGENT]
    assert asker["agent_id"] == hotel.id
    # the person's inbox holds the question once, on the planner's run
    inbox = await harness.inbox("user:live-user")
    [entry] = [s for s in inbox if s.run_id in (handle.run_id, asker["run_id"])]
    assert entry.run_id == handle.run_id and entry.awaiting is not None
    resumed = await planner.resume(
        entry.awaiting.interrupt_id, "answer", answer="low", reviewer="live-user"
    )
    assert resumed.status is RunStatus.QUEUED
    await died(harness, planner, handle)  # inside the booker, after it booked
    assert charged == ["Oslo low"]
    assert await claimed_again(harness, planner)
    done = await handle.result(timeout=30)
    assert done.status is RunStatus.SUCCESS and done.answer == "Planned.", done.error
    assert charged == ["Oslo low"] and looked == ["Oslo"]  # no side effect repeated
    told = [m["content"] for m in model.requests[-1]["messages"] if m["role"] == "tool"]
    assert told == ["Oslo: 2 a day", "confirmed: booked Oslo (low)"]
    kids = [s async for s in harness.runs.iterate(parent_run_id=handle.run_id)]
    assert sorted(k.agent_id for k in kids) == sorted([scout.id, hotel.id])
    for kid in kids:
        record = await harness.runs.get(kid.run_id)
        assert record is not None and record.status is RunStatus.SUCCESS
        assert record.user_id == "live-user" and record.thread_id == handle.run_id
