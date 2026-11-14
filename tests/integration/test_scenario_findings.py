"""The scenario suite's findings (tests/scenarios), each isolated to its smallest failing form here,
in process (no service; SCN-2 and SCN-3 need agent-runs' event log: tests/scenarios/test_s15_findings.py): ``xfail(strict=True)`` until fixed, so ``make test`` stays green and the
day a fix lands the test passes — and must be unmarked. Root causes and proposed fixes:
docs/testing/TEST_PLAN.md, "Scenario findings"."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from trellis import Harness, Runtime


@pytest.mark.xfail(
    strict=True, reason="SCN-1: a burst to one chat is not answered in the order sent"
)
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


@pytest.mark.xfail(strict=True, reason="SCN-4: a turn written late leaves the next turn without it")
async def test_scn4_the_next_turn_has_the_turn_before_when_memory_writes_slowly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A memory service that takes longer than ``TURN_WRITE_SECONDS`` (5 s; 0.2 here) to keep a
    turn: the run returns with its transcript queued (``Agent.recorded_run``, "the turn could not
    be written before the run returned; it is queued"), and the user's next message — sent at once,
    as people do — is answered from a window without it. Seen on the real stack under load: POST
    /v1/messages taking 3-7 s, a chat's third turn answered without its second."""
    import httpx

    from tests.support.memory import URL, FakeMemoryService
    from tests.support.models import ScriptedChat
    from trellis import ReAct, Settings
    from trellis.harness import agent as agent_module
    from trellis.memory import MemoryClient

    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.2)
    service = FakeMemoryService()

    async def slow(request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("POST", "/v1/messages"):
            await asyncio.sleep(0.6)
        return service._checked(request)

    client = MemoryClient(
        URL,
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(base_url=URL, transport=httpx.MockTransport(slow)),
    )
    async with Harness(config=Settings(), memory=client) as h:
        model = ScriptedChat(["Noted.", "I do not know."])
        agent = h.wrap(ReAct(system_prompt="You help.", model=model), id="scn4")
        first = await agent.run("The locker code is 4242.", user="ada", thread="t1")
        assert first.status.value == "SUCCESS"
        second = await agent.run("What is the locker code?", user="ada", thread="t1")
        assert second.status.value == "SUCCESS"
        assert "4242" in str(model.requests[-1]["messages"]), "the turn before is not in the window"


@pytest.mark.xfail(
    strict=True,
    reason="SCN-6: a write is lost when the memory service is down longer than its retries",
)
async def test_scn6_a_turn_is_kept_once_memory_comes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory service refuses every write for a while (down, or past its deadlines under
    load: the client's circuit opens) and then comes back: the turn made meanwhile is kept once
    it is back. Today the write gets ``WRITE_ATTEMPTS`` (3) tries with a short backoff and, with no
    spool (none in a container: ``settings.default_spool``), is given up "failed and is lost"
    (``writes._attempt``); seen on the real stack as transcripts, tool records and outcomes lost
    while the memory service answered 504s."""
    import httpx

    from tests.support.memory import URL, FakeMemoryService
    from tests.support.models import ScriptedChat
    from trellis import ReAct, Settings
    from trellis.harness import agent as agent_module
    from trellis.harness import writes as writes_module
    from trellis.memory import MemoryClient

    monkeypatch.setattr(agent_module, "TURN_WRITE_SECONDS", 0.1)
    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.05)
    service = FakeMemoryService()
    down = {"on": False}

    async def flaky(request: httpx.Request) -> httpx.Response:
        if down["on"] and (request.method, request.url.path) == ("POST", "/v1/messages"):
            problem = {
                "title": "unavailable",
                "status": 503,
                "code": "DEPENDENCY_UNAVAILABLE",
                "retryable": True,
            }
            return httpx.Response(
                503, json=problem, headers={"content-type": "application/problem+json"}
            )
        return service._checked(request)

    client = MemoryClient(
        URL,
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(base_url=URL, transport=httpx.MockTransport(flaky)),
    )
    async with Harness(config=Settings(), memory=client) as h:
        agent = h.wrap(ReAct(system_prompt="You help.", model=ScriptedChat(["Noted."])), id="scn6")
        down["on"] = True
        done = await agent.run("The locker code is 4242.", user="ada", thread="t1")
        assert done.status.value == "SUCCESS"
        await asyncio.sleep(1.0)  # down for longer than the write's retries
        down["on"] = False
        await h.writes.drain()
        kept = [c for c in service.named("messages") if "4242" in str(c.body)]
        assert kept, "the turn made while the memory service was down is lost"
