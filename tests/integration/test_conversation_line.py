"""A conversation's turns on two servers: two harnesses sharing one run store stand for two
server processes behind a load balancer (the real ones — separate processes over agent-runs
and PostgreSQL, every adapter — are ``tests/live/test_live_chat_order.py``). Whichever server
each message reaches, the conversation's messages are answered one at a time, in the order they
arrived, each turn seeing the earlier ones; ``interrupt`` cancels the run working on the other
server, ``steer`` waits its turn there (said once); a turn paused for a person holds nobody up,
and continues after the turns that arrived while it waited; a worker's queued turn and the
servers' turns take turns too; other conversations — another chat of the user, another user's
thread named alike — never wait. A burst of messages is answered in the order it was sent, however
long the run store takes to answer each start."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.runlog import LoggedRuns
from trellis import Harness, Runtime, Settings, current, tool
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunStart,
    RunStatus,
)
from trellis.harness import holding
from trellis.harness.agent import Agent
from trellis.harness.conversation import EARLIER_RUNS
from trellis.harness.line import Line, placed
from trellis.harness.runs import LocalRuns
from trellis.runs import Page, ResolutionEntry, RunsError

THREAD = "chat-9"

#: what each turn did, in order: ("start", question, the earlier turns it saw) and ("end", ...)
Log = list[tuple[Any, ...]]


@pytest.fixture
async def servers(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Harness, Harness]]:
    monkeypatch.setattr(holding, "HOLD_SECONDS", 1)  # a run cancelled elsewhere stops within 1/3 s
    runs = LocalRuns()
    async with (
        Harness(config=Settings(), runs=runs) as a,
        Harness(config=Settings(), runs=runs) as b,
    ):
        yield a, b
    await runs.aclose()


def chat(h: Harness, log: Log, gates: dict[str, asyncio.Event]) -> Agent:
    """The same agent on a server: each turn logs what it saw, and waits at its gate."""

    async def answer(input: str, agent: Runtime) -> str:
        log.append(("start", input, [s.content for s in agent.history or []]))
        try:
            if input in gates:
                await gates[input].wait()
        except asyncio.CancelledError:
            log.append(("cancelled", input))
            raise
        log.append(("end", input))
        return f"re: {input}"

    return h.wrap(answer, id="chat")


async def started(log: Log, question: str) -> None:
    async with asyncio.timeout(10):
        while not any(e[:2] == ("start", question) for e in log):  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_a_chats_messages_on_two_servers_are_answered_one_at_a_time_in_order(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    questions = ["one", "two", "three", "four", "five"]
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    rest = []
    for n, question in enumerate(questions[1:]):
        server = (b, a)[n % 2]
        rest.append(asyncio.create_task(server.run(question, user="ada", thread=THREAD)))
        await asyncio.sleep(0.05)  # each arrives after the one before
    assert [e[1] for e in log] == ["one"], "the others wait while the first works"
    gates["one"].set()
    results = await asyncio.gather(first, *rest)
    assert all(r.status is RunStatus.SUCCESS for r in results), [r.error for r in results]
    assert [e[:2] for e in log] == [(p, q) for q in questions for p in ("start", "end")]
    for n, question in enumerate(questions):
        seen = next(e[2] for e in log if e[:2] == ("start", question))
        assert seen == [x for q in questions[:n] for x in (q, f"re: {q}")]


async def test_other_chats_and_another_users_thread_named_alike_never_wait(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    other_chat = await asyncio.wait_for(b.run("elsewhere", user="ada", thread="chat-2"), 10)
    other_user = await asyncio.wait_for(b.run("mine", user="bob", thread=THREAD), 10)
    assert other_chat.answer == "re: elsewhere" and other_user.answer == "re: mine"
    assert next(e[2] for e in log if e[1] == "mine") == []  # isolated: not ada's turn
    gates["one"].set()
    assert (await first).status is RunStatus.SUCCESS


async def test_an_interrupt_on_one_server_cancels_the_run_working_on_the_other(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event()}  # never opened: only the cancel ends it
    a, b = (chat(h, log, gates) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    second = await asyncio.wait_for(b.run("two", user="ada", thread=THREAD, mode="interrupt"), 10)
    stopped = await asyncio.wait_for(first, 10)
    assert stopped.status is RunStatus.CANCELLED
    assert ("cancelled", "one") in log  # its server stopped it
    assert second.status is RunStatus.SUCCESS
    assert next(e[2] for e in log if e[1] == "two") == ["one"]  # it went on from the question
    record = await servers[0].runs.get(stopped.run_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_steer_to_a_run_on_the_other_server_waits_its_turn_said_once(
    servers: tuple[Harness, Harness], caplog: pytest.LogCaptureFixture
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        second = asyncio.create_task(b.run("two", user="ada", thread=THREAD, mode="steer"))
        await asyncio.sleep(0.3)
        assert [e[1] for e in log] == ["one"]
        gates["one"].set()
        results = await asyncio.gather(first, second)
    assert [r.answer for r in results] == ["re: one", "re: two"]  # its own turn, after
    assert caplog.text.count("works in another process") == 1


async def test_a_turn_paused_for_a_person_holds_nobody_up_and_continues_after_them(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    gates = {"two": asyncio.Event()}

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order (a person approves it first)."""
        log.append(("refunded", order))
        return f"refunded {order}"

    async def answer(input: str, agent: Runtime) -> str:
        log.append(("start", input))
        if input == "refund o-1":
            return str(await agent.tools.call("refund", order="o-1"))
        await gates[input].wait()
        log.append(("end", input))
        return f"re: {input}"

    a, b = (arm(h.wrap(answer, id="chat"), [refund]) for h in servers)
    asked = await a.run("refund o-1", user="ada", thread=THREAD)
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    second = asyncio.create_task(b.run("two", user="ada", thread=THREAD))
    await started(log, "two")  # the paused turn holds nobody up
    resumed = asyncio.create_task(
        a.resume(asked.interrupt.interrupt_id, InterruptDecision.APPROVE, reviewer="lead")
    )
    await asyncio.sleep(0.3)
    assert ("refunded", "o-1") not in log, "it continues after the turn working now"
    gates["two"].set()
    assert (await second).status is RunStatus.SUCCESS
    assert (await resumed).status is RunStatus.SUCCESS
    assert log.index(("end", "two")) < log.index(("refunded", "o-1"))


async def test_a_workers_queued_turn_and_the_servers_turns_take_turns(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event(), "two": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    await a.start("one", user="ada", thread=THREAD)
    working = asyncio.create_task(servers[0].worker([a]).run_once())
    await started(log, "one")
    second = asyncio.create_task(b.run("two", user="ada", thread=THREAD))
    await asyncio.sleep(0.3)
    assert [e[1] for e in log] == ["one"], "the server's turn waits for the worker's"
    await a.start("three", user="ada", thread=THREAD)
    gates["one"].set()
    assert await working is True
    await started(log, "two")
    assert await servers[0].worker([a]).run_once() is False, "the queued turn waits too"
    gates["two"].set()
    assert (await second).status is RunStatus.SUCCESS
    assert await servers[0].worker([a]).run_once() is True
    assert [e[1] for e in log if e[0] == "start"] == ["one", "two", "three"]


async def test_a_run_continued_after_its_answer_is_placed_when_it_continued(
    servers: tuple[Harness, Harness],
) -> None:
    """Its place is when it last started running: the line's order is the store's."""
    runs = servers[0].runs

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return order

    async def answer(input: str, agent: Runtime) -> str:
        return str(await agent.tools.call("refund", order=input))

    agent = arm(servers[0].wrap(answer, id="chat"), [refund])
    asked = await agent.run("o-1", user="ada", thread=THREAD)
    assert asked.interrupt is not None
    record = await runs.get(asked.run_id)
    assert record is not None and await placed(runs, record) == record.created_at
    await runs.resume(
        InterruptResolution(
            interrupt_id=asked.interrupt.interrupt_id,
            run_id=asked.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="lead",
        )
    )
    continued = await runs.get(asked.run_id)
    assert continued is not None
    place = await placed(runs, continued)
    assert place > record.created_at and place - continued.updated_at < timedelta(seconds=1)
    assert await Line(runs, continued).ahead() == []


class Laggy(LocalRuns):
    """A run store whose starts answer late, the earlier ones the later — as agent-runs over
    HTTP, where a start sent later may be taken first."""

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0

    async def start(self, start: RunStart, *, queue: bool = False) -> Any:
        self.starts += 1
        await asyncio.sleep(max(0.0, 0.03 - 0.0015 * self.starts))
        return await super().start(start, queue=queue)


@pytest.mark.parametrize("burst", [5, 20])
@pytest.mark.parametrize("framework", sorted(BUILDERS))
async def test_a_burst_of_messages_is_answered_in_the_order_it_was_sent(
    tmp_path: Any, framework: str, burst: int
) -> None:
    turns: list[tuple[str, list[str] | None]] = []

    @tool(side_effects="read")
    async def step() -> str:
        """Do the work of this turn."""
        runtime = current()
        assert runtime is not None
        history = runtime.history
        turns.append((str(runtime.task), None if history is None else [s.content for s in history]))
        return f"worked on {runtime.task}"

    questions = [f"message {n}" for n in range(burst)]
    runs = Laggy()
    async with Harness(config=Settings(), runs=runs) as h:
        target, tools = await BUILDERS[framework](h, [step], tmp_path, [("step", {})])
        agent = arm(h.wrap(target, id=f"burst-{framework}"), tools)
        sent = [asyncio.create_task(agent.run(q, user="ada", thread=THREAD)) for q in questions]
        results = await asyncio.wait_for(asyncio.gather(*sent), 300)
    records = [await runs.get(r.run_id) for r in results]
    await runs.aclose()
    assert all(r.status is RunStatus.SUCCESS for r in results), [r.error for r in results]
    answered = [r.updated_at for r in records if r is not None]
    assert answered == sorted(answered) and len(answered) == burst  # in the order sent
    stepped = [question for question, _ in turns]
    if framework == "kimi":  # its session holds the first turn's result: it calls step once
        assert stepped == questions[: len(stepped)]
    else:
        assert stepped == questions
    for n, (_, history) in enumerate(turns):
        if history is not None:  # (a session the framework continued holds it instead)
            asked = [said for said in history if said.startswith("message ")]
            assert asked == questions[max(0, n - EARLIER_RUNS) : n]


async def test_a_run_cancelled_elsewhere_stops_at_its_next_event() -> None:
    """With agent-runs' event log, the run's next event is refused once it was cancelled on
    another server: it stops then, long before its next heartbeat."""
    runs = LoggedRuns()
    cancelled: list[str] = []

    async def chatty(input: str, agent: Runtime) -> str:
        try:
            while True:  # it says something every 20 ms, and never ends on its own
                agent.events.warning("working", input)
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            cancelled.append(input)
            raise

    async with (
        Harness(config=Settings(), runs=runs) as a,
        Harness(config=Settings(), runs=runs) as b,
    ):
        a.wrap(chatty, id="other")  # another agent of the server: not the run's
        working = asyncio.create_task(
            a.wrap(chatty, id="chat").run("one", user="ada", thread=THREAD)
        )
        async with asyncio.timeout(10):
            while not runs.logs:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        [run_id] = runs.logs
        await b.wrap(chatty, id="chat").cancel(run_id, reason="from b")
        stopped = await asyncio.wait_for(working, 5)  # a hold heartbeat is 20 s away
    assert stopped.status is RunStatus.CANCELLED and cancelled == ["one"]
    await runs.aclose()


async def test_a_refused_hold_stops_only_a_run_cancelled_in_the_store(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs = LocalRuns()
    async with Harness(config=Settings(), runs=runs) as h:
        await h._lost("run_nowhere", "default")  # not in the store: nothing to stop
        record = await runs.start(RunStart(tenant_id="default", agent_id="chat", user_id="ada"))
        await h._lost(record.run_id, "default")  # still running (held by another): goes on

        async def unreachable(*args: Any, **kwargs: Any) -> None:
            raise RunsError("agent-runs is away", code="DEPENDENCY_UNAVAILABLE", status=503)

        h.runs.get = unreachable  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="trellis.harness"):
            await h._lost(record.run_id, "default")
    assert "not read back after its hold was refused" in caplog.text
    await runs.aclose()


async def test_a_place_reads_every_page_of_answers_and_a_run_ended_is_behind_nobody(
    servers: tuple[Harness, Harness],
) -> None:
    runs = servers[0].runs
    started = RunStart(tenant_id="default", agent_id="chat", user_id="ada", thread_id=THREAD)
    record = await runs.start(started)
    answers = [
        ResolutionEntry(
            interrupt=Interrupt(tenant_id="default", run_id=record.run_id, question=f"q{n}"),
            resolution=InterruptResolution(
                interrupt_id=f"i{n}", run_id=record.run_id, decision=InterruptDecision.APPROVE
            ),
            attempt=n + 1,
            recorded_at=record.created_at + timedelta(seconds=n + 1),
        )
        for n in range(3)
    ]
    answered = record.model_copy(update={"last_resolution": answers[-1].resolution})
    pages = Page[ResolutionEntry]

    async def paged(run_id: str, *, cursor: str | None = None, **kw: Any) -> Any:
        n = int(cursor or 0)
        return pages(items=answers[n : n + 1], next_cursor=str(n + 1) if n < 2 else None)

    runs.resolutions = paged  # type: ignore[method-assign]
    assert await placed(runs, answered) == answers[-1].recorded_at
    ended = await runs.finish(record.run_id, RunStatus.SUCCESS)
    same = await runs.start(started.model_copy(update={"run_id": "run_next"}))
    assert same.status is RunStatus.RUNNING
    assert await Line(runs, ended).ahead() == []  # it no longer runs: nothing is ahead of it
