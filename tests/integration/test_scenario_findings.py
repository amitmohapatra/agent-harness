"""Findings of the scenario suite (tests/scenarios) in their smallest form, in process: SCN-1 (a
burst to one chat is answered in the order sent) and SCN-8 (a message cancelled while it waits
its turn never runs). SCN-2, 3, 4 and 6 are in test_scenario_fixes.py; root causes:
docs/testing/TEST_PLAN.md, "Scenario findings"."""

from __future__ import annotations

import asyncio
from typing import Any

from trellis import Harness, Runtime


async def test_scn1_a_burst_to_one_chat_is_answered_in_the_order_sent(harness: Harness) -> None:
    """Five messages sent at once to one conversation (each ``agent.run`` called in order) are
    answered in the order sent: the turn is taken only after the run's record was opened in
    the run store and held — awaits that end in any order (``pipeline._turned`` takes
    ``Turns.taken`` after ``Agent._opened`` and ``holds.holding``)."""
    order: list[str] = []

    async def echo(input: Any, agent: Runtime) -> str:
        order.append(str(input))
        return str(input)

    agent = harness.wrap(echo, id="scn1")
    # the run store answers the starts in another order than they were sent, as a store over a
    # network does under load (agent-runs: seen with every adapter, bursts of 2, 5 and 20)
    start = harness.runs.start
    delays = iter([0.05, 0.04, 0.03, 0.02, 0.01])

    async def jittered(record: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(next(delays))
        return await start(record, *args, **kwargs)

    harness.runs.start = jittered  # type: ignore[method-assign]
    sent = [f"m{n}" for n in range(5)]
    await asyncio.gather(*(agent.run(m, user="ada", thread="t1") for m in sent))
    assert order == sent, order


async def test_scn8_a_message_cancelled_while_it_waits_its_turn_ends_cancelled(
    harness: Harness,
) -> None:
    """The second message of a busy conversation waits for the first (in process); cancelled
    while it waits, it ends ``CANCELLED`` and never runs. Today ``agent.cancel`` cancels it in the
    run store only (it is not in ``Agent.running`` until it has the turn), and it runs when the
    turn comes: its finish is refused (the store has it ``CANCELLED``) and ``agent.run`` raises."""
    ran: list[str] = []

    async def slow(input: Any, agent: Runtime) -> str:
        ran.append(str(input))
        await asyncio.sleep(0.3)
        return str(input)

    agent = harness.wrap(slow, id="scn8")
    first = asyncio.create_task(agent.run("first", user="ada", thread="t1"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(agent.run("second", user="ada", thread="t1"))
    await asyncio.sleep(0.05)
    waiting = [r async for r in harness.runs.iterate(agent_id="scn8")]
    [queued] = [r for r in waiting if r.run_id not in agent.running]
    await agent.cancel(queued.run_id, reason="never mind")
    await first
    done = await second
    assert done.status.value == "CANCELLED" and ran == ["first"], (done.status, ran)
