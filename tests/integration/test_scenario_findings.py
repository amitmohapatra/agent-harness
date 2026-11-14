"""Findings of the scenario suite (tests/scenarios) in their smallest form, in process: SCN-1 (a
burst to one chat is answered in the order sent) and SCN-8 (a message cancelled while it waits
its turn never runs). SCN-2, 3, 4 and 6 are in test_scenario_fixes.py; root causes:
docs/testing/TEST_PLAN.md, "Scenario findings"."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

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


@pytest.mark.xfail(
    strict=True,
    reason="SCN-13: scn-fixes (harness.py Harness._lost stops a run only when the store has it "
    "CANCELLED, not when the store ended it holder_lost)",
)
async def test_scn13_a_run_the_store_ended_stops_where_it_works(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run this process works on, ended by the run store meanwhile (its hold lapsed while the
    network to the store was cut: ``ERROR``, ``holder_lost``), stops here at its next refused
    heartbeat: what it would have done after that — a write — is not done. Today only a run the
    store has ``CANCELLED`` is stopped; one it ended goes on to its end, side effects and all
    (seen on the real stack: a note written 30 s after agent-runs had ended the run)."""
    from trellis.contracts import AgentError, ErrorCategory, RunStatus
    from trellis.harness import holding

    monkeypatch.setattr(holding, "HOLD_SECONDS", 0.3)
    started = asyncio.Event()
    went_on: list[str] = []

    async def slow(input: Any, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(1.5)
        went_on.append("the write after the slow step")
        return "done"

    agent = harness.wrap(slow, id="scn13")
    turn = asyncio.create_task(agent.run("go", user="ada", thread="t1"))
    await started.wait()
    [run] = [r async for r in harness.runs.iterate(agent_id="scn13")]
    lost = AgentError(
        code="holder_lost", category=ErrorCategory.DEPENDENCY, message="its holder stopped"
    )
    await harness.runs.finish(run.run_id, RunStatus.ERROR, error=lost)  # as the store's sweep does
    await asyncio.wait({turn})
    assert went_on == [], went_on
