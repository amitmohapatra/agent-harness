"""Findings of the scenario suite (tests/scenarios) in their smallest form, in process: SCN-1 (a
burst to one chat is answered in the order sent) and SCN-8 (a message cancelled while it waits
its turn never runs), LR-02 (a 200-step task keeps every request in a small window). SCN-2, 3,
4 and 6 are in test_scenario_fixes.py; root causes: docs/testing/TEST_PLAN.md, "Scenario
findings"."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import convert_to_messages
from langchain_core.messages.utils import count_tokens_approximately

# the brain's model is built on LangChain's ChatOpenAI as imported here, before a test's
# fixtures stand in for it (``conftest.openai_on_httpx``)
from tests.scenarios.brains import CLI as CLAUDE_CLI
from tests.scenarios.brains import SYSTEM, BrainChat, parse, token
from tests.scenarios.kit import TOOLS
from tests.support.arm import arm
from trellis import Harness, ReAct, Runtime
from trellis.contracts import RunEventType


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


async def test_lr02_a_two_hundred_step_task_on_a_small_window_keeps_every_request_in_it(
    memory_harness: Harness,
) -> None:
    """``ReAct(context_window=8000)`` on a task of 200 lookups, in one turn: every step runs
    once, it answers with the last result, and no request is over the window — counted as the
    harness counts it, and as the scenario suite measures it (a token per 4 characters of the
    messages, within its margin). The latest turn's older steps are said in a note once they do
    not fit (``middleware._steps_said``); before, the turn was sent whole: ~19,000 tokens
    (LR-02, ``test_s13::test_a_two_hundred_step_task[react_fitted]``)."""
    chat = BrainChat()
    target = ReAct(SYSTEM, chat, max_steps=230, context_window=8_000)
    agent = arm(memory_harness.wrap(target, id="lr02"), TOOLS)
    asked = f"{token('lr', 1)} loop:200:lookup"
    events = [e async for e in agent.stream(asked, user="ada", thread="t1")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.error is None, finished
    said = parse(finished.data["result"])
    assert said.mine == token("lr", 1) and said.last == "value-of-k199", said
    assert len([e for e in events if e.type is RunEventType.TOOL_CALL_START]) == 200
    counted = [
        count_tokens_approximately(convert_to_messages(r["messages"]), tools=r.get("tools"))
        for r in chat.requests
    ]
    assert max(counted) <= int(8_000 * 0.95), max(counted)
    biggest = max(len(json.dumps(r["messages"])) for r in chat.requests)
    assert biggest / 4 <= 8_000 * 1.25, biggest


@pytest.mark.parametrize("memory", [False, True])
async def test_claude_a_burst_to_one_chat_each_turn_continues_the_turn_before(
    harness: Harness, memory_harness: Harness, memory: bool, tmp_path: Path
) -> None:
    """Three messages sent at once to one Claude conversation: each turn continues the session
    the turn before it ended in, so the third has heard the first two. The turns still waiting
    in this process's line (created after the latest turn served here, held here, not
    ``QUEUED``) were taken for turns another worker served (``pipeline._served_elsewhere``):
    the kept session was dropped and the turn began a new one, missing the turn before (s01's
    Claude burst of 5 and 20: turn 3 saw turn 1 only)."""
    from claude_agent_sdk import ClaudeAgentOptions

    h = memory_harness if memory else harness
    # the run store lists a thread's runs as agent-runs does: newest created first (keyset
    # ``created_at, run_id`` descending), the turns still waiting ahead of the one served
    listed = h.runs.iterate

    async def newest_created_first(**query: Any) -> Any:
        runs = []
        async for summary in listed(**query):
            record = await h.runs.get(summary.run_id, tenant=query.get("tenant"))
            assert record is not None
            runs.append((record.created_at, summary.run_id, summary))
        for *_, found in sorted(runs, key=lambda r: (r[0], r[1]), reverse=True):
            yield found

    h.runs.iterate = newest_created_first  # type: ignore[method-assign]
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps([{"text": "ANS {heard} CTX {context} LAST {last}"}]),
        "FAKE_CLAUDE_SESSIONS": str(tmp_path),
        "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
    }
    agent = h.wrap(ClaudeAgentOptions(cli_path=CLAUDE_CLI, env=env), id="burst")
    sent = [token("b", n) for n in range(1, 4)]
    done = await asyncio.gather(*(agent.run(f"{t} burst", user="ada", thread="t1") for t in sent))
    said = [parse(d.answer) for d in done]
    assert [s.mine for s in said] == sent, said
    assert set(sent[:2]) <= said[2].seen, said[2]  # the third turn heard the first two
