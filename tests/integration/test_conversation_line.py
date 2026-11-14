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
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.runlog import LoggedRuns
from trellis import Harness, Runtime, Settings, current, tool
from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunStart,
    RunStatus,
)
from trellis.contracts.ids import new_id
from trellis.harness import holding
from trellis.harness import runs as local_runs
from trellis.harness.agent import Agent
from trellis.harness.bounded import LRU
from trellis.harness.conversation import EARLIER_RUNS, earlier
from trellis.harness.identity import conversation_key
from trellis.harness.line import Followers, Line
from trellis.harness.runlog import RunLog
from trellis.harness.runs import LocalRuns, place
from trellis.runs import ConflictError, Page, ResolutionEntry, RunsError

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
    """Its place is when it last started running: the line's order is the store's
    (``placed_at``), read as the store gives it."""
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
    assert record is not None
    assert place(record) == (record.created_at, record.run_id)
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
    at, _ = place(continued)
    assert at > record.created_at and at - continued.updated_at < timedelta(seconds=1)
    assert await Line(runs, continued, LRU(), servers[0].followers).ahead() == []


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


async def test_a_run_the_store_cancelled_unheld_stops_at_its_next_event() -> None:
    """A run cancelled before the store had its server's hold (its first heartbeat still on its
    way) is ``CANCELLED`` there at once: with agent-runs' event log its next event is refused,
    and it stops then, long before its next heartbeat."""
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
        await runs.release(run_id, a.holds.name)  # the store has no hold of it (yet)
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


class Counted(LocalRuns):
    """The in-process store, counting the reads a line makes: ``get`` by run, ``resolutions``
    by run."""

    def __init__(self) -> None:
        super().__init__()
        self.got: list[str] = []
        self.answers_read: list[str] = []

    async def get(self, run_id: str, *, tenant: str | None = None) -> Any:
        self.got.append(run_id)
        return await super().get(run_id, tenant=tenant)

    async def resolutions(self, run_id: str, **kwargs: Any) -> Page[ResolutionEntry]:
        self.answers_read.append(run_id)
        return await super().resolutions(run_id, **kwargs)


def turn_of(user: str, thread: str = THREAD) -> RunStart:
    """A turn of ``user``'s conversation ``thread``, keyed as the harness keys it."""
    key = f"thread:{conversation_key('default', user, thread)}"
    return RunStart(
        tenant_id="default", agent_id="chat", user_id=user, thread_id=thread, concurrency_key=key
    )


async def test_arch15_a_line_is_ordered_by_the_stores_places_reading_no_answers() -> None:
    """ARCH-15: the place is the store's (``placed_at``): a line of runs answered three times
    over is ordered without reading one answer — the run continued after its answers is
    behind the run that came in while it waited — and a run ended is behind nobody."""
    runs = Counted()
    answered = await runs.start(turn_of("ada"))
    waited = None
    for n in range(3):
        asked = Interrupt(tenant_id="default", run_id=answered.run_id, question=f"q{n}")
        await runs.pause(asked)
        waited = waited or await runs.start(turn_of("ada"))  # it came in while the first waited
        await runs.resume(
            InterruptResolution(
                interrupt_id=asked.interrupt_id,
                run_id=answered.run_id,
                decision=InterruptDecision.APPROVE,
            )
        )
    assert waited is not None
    mine = await runs.start(turn_of("ada"))
    runs.answers_read.clear()
    line = Line(runs, mine, LRU(), Followers(lambda: None))
    assert await line.ahead() == [waited.run_id, answered.run_id]
    assert runs.answers_read == []  # (it read 3 pages of answers before)
    ended = await runs.finish(mine.run_id, RunStatus.SUCCESS)
    assert await Line(runs, ended, LRU(), Followers(lambda: None)).ahead() is None  # it left
    await runs.aclose()


async def test_arch15_a_line_run_without_a_place_is_a_clear_error_never_an_order() -> None:
    """A run store that sends no ``placed_at`` (older than this harness) cannot order a line:
    the line says so, naming the run, rather than guessing an order."""
    runs = LocalRuns()
    ahead = await runs.start(turn_of("ada"))
    mine = await runs.start(turn_of("ada"))
    get = runs.get

    async def unplaced(run_id: str, **kwargs: Any) -> Any:
        found = await get(run_id, **kwargs)
        return found.model_copy(update={"placed_at": None}) if found else None

    runs.get = unplaced  # type: ignore[method-assign]
    named = f"run ({ahead.run_id}|{mine.run_id}) of concurrency_key .* without placed_at"
    with pytest.raises(ConfigurationError, match=named):
        await Line(runs, mine, LRU(), Followers(lambda: None)).ahead()
    await runs.aclose()


async def test_secr11_a_run_claimed_between_the_lines_listings_is_still_ahead() -> None:
    """The line lists its key's queued runs, then its running ones: a queued run a worker
    claims in between (``QUEUED`` to ``RUNNING``, its place unchanged) is in one listing or
    both, never in neither; the run itself, running throughout, is in the second."""
    runs = LocalRuns()
    queued = await runs.start(turn_of("ada"), queue=True)
    mine = await runs.start(turn_of("ada"))
    iterate = runs.iterate
    listed: list[RunStatus | None] = []

    async def claiming(**kwargs: Any) -> AsyncIterator[Any]:
        async for summary in iterate(**kwargs):
            yield summary
        listed.append(kwargs.get("status"))
        if len(listed) == 1:
            assert await runs.claim("worker-1", ["chat"]) is not None

    runs.iterate = claiming  # type: ignore[method-assign]
    assert await Line(runs, mine, LRU(), Followers(lambda: None)).ahead() == [queued.run_id]
    assert listed == [RunStatus.QUEUED, RunStatus.RUNNING]
    await runs.aclose()


async def test_an_interrupt_or_a_steer_to_an_idle_chat_just_runs(
    servers: tuple[Harness, Harness],
) -> None:
    log: Log = []
    agent = chat(servers[0], log, {})
    for mode in ("interrupt", "steer"):
        result = await agent.run(mode, user="ada", thread=THREAD, mode=mode)
        assert result.answer == f"re: {mode}"


async def test_a_log_whose_run_ended_takes_nothing_more_whoever_listens() -> None:
    """A run's log refused (the run no longer runs) closes; with nobody to tell, that is all."""

    class Ended:
        async def append_events(self, *args: Any, **kwargs: Any) -> None:
            raise ConflictError("the run takes no events now", code="CONFLICT", status=409)

    log = RunLog(Ended(), "run_x", tenant="t", worker_id=None, events=None)  # type: ignore[arg-type]
    log(RunEvent(tenant_id="t", run_id="run_x", type=RunEventType.RUN_STARTED, sequence=1))
    await log.flush()
    assert log.closed


async def keep_claiming(h: Harness, agent: Agent) -> None:
    """A worker of ``h`` claiming ``agent``'s queued runs as they come."""
    worker = h.worker([agent])
    while True:
        if not await worker.run_once():
            await asyncio.sleep(0.02)


async def thread_run(runs: Any, question: str) -> str:
    """The id of the run of :data:`THREAD` asked ``question``, once the store has it."""
    async with asyncio.timeout(10):
        while True:
            async for summary in runs.iterate(thread_id=THREAD):
                record = await runs.get(summary.run_id)
                if record is not None and record.input == question:
                    return record.run_id
            await asyncio.sleep(0.01)


async def test_a_queued_turn_holds_its_place_before_the_messages_sent_after_it(
    servers: tuple[Harness, Harness],
) -> None:
    """A turn queued for a worker while another works is answered before the messages sent
    after it: a stream of messages cannot keep it waiting."""
    log: Log = []
    gates = {"one": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    await a.start("queued two", user="ada", thread=THREAD)
    third = asyncio.create_task(b.run("three", user="ada", thread=THREAD))
    await thread_run(servers[1].runs, "three")
    claiming = asyncio.create_task(keep_claiming(servers[0], a))
    try:
        gates["one"].set()
        results = await asyncio.wait_for(asyncio.gather(first, third), 20)
        async with asyncio.timeout(10):
            while not any(e[:2] == ("end", "queued two") for e in log):  # noqa: ASYNC110
                await asyncio.sleep(0.02)
    finally:
        claiming.cancel()
    assert all(r.status is RunStatus.SUCCESS for r in results)
    assert [e[1] for e in log if e[0] == "start"] == ["one", "queued two", "three"]


@pytest.mark.parametrize("where", ["here", "elsewhere"])
async def test_a_run_cancelled_while_it_waits_its_turn_never_runs(
    servers: tuple[Harness, Harness], where: str
) -> None:
    log: Log = []
    gates = {"one": asyncio.Event()}
    a, b = (chat(h, log, gates) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    second = asyncio.create_task(a.run("two", user="ada", thread=THREAD))
    waiting = await thread_run(servers[0].runs, "two")
    await (a if where == "here" else b).cancel(waiting, reason="not needed")
    ended = await asyncio.wait_for(second, 5)  # at once, not when the turn ahead ends
    assert ended.status is RunStatus.CANCELLED and ended.run_id == waiting
    gates["one"].set()
    assert (await first).status is RunStatus.SUCCESS
    assert [e[1] for e in log if e[0] == "start"] == ["one"]  # it never ran


async def test_a_run_waiting_behind_another_servers_run_follows_its_events_not_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behind a run another server works on, a run waits on that run's event log (it says
    when the run ends or pauses) instead of reading the line again and again."""
    runs = LoggedRuns()
    listed: list[str] = []
    iterate = runs.iterate

    def counted(**kwargs: Any) -> Any:
        listed.append(str(kwargs.get("status")))
        return iterate(**kwargs)

    monkeypatch.setattr(runs, "iterate", counted)
    log: Log = []
    gates = {"one": asyncio.Event()}
    async with (
        Harness(config=Settings(), runs=runs) as h1,
        Harness(config=Settings(), runs=runs) as h2,
    ):
        a, b = chat(h1, log, gates), chat(h2, log, gates)
        first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
        await started(log, "one")
        second = asyncio.create_task(b.run("two", user="ada", thread=THREAD))
        await thread_run(runs, "two")
        await asyncio.sleep(0.2)
        before = len(listed)
        await asyncio.sleep(2)  # the turn ahead works on
        during = len(listed) - before
        gates["one"].set()
        results = await asyncio.wait_for(asyncio.gather(first, second), 10)
    assert [r.answer for r in results] == ["re: one", "re: two"]
    assert during == 0, f"{during} listings of the line while it waited"
    await runs.aclose()


async def test_conv04_an_interrupt_from_another_server_never_overlaps_the_turn_it_stops(
    servers: tuple[Harness, Harness],
) -> None:
    """CONV-04: a turn stuck in one long, silent tool call on one server is interrupted from
    the other: its call is cancelled and its run ended before the interrupting turn starts.
    The message says in its run which run it interrupts; the server holding that run reads it
    as it renews its hold, stops the call and ends the run — ``RUNNING`` until then, so the
    message, waiting in the line behind it, never works beside it."""
    log: Log = []
    never = asyncio.Event()

    @tool(side_effects="read")
    async def wait() -> str:
        """Wait for something that never comes."""
        log.append(("tool", "started"))
        try:
            await never.wait()
        except asyncio.CancelledError:
            record = await servers[0].runs.get(first_run[0])
            log.append(("tool", "cancelled", record.status if record else None))
            raise
        return "done"

    async def answer(input: str, agent: Runtime) -> str:
        log.append(("start", input))
        if input == "one":
            first_run.append(agent.run_id)
            await agent.tools.call("wait")
        return f"re: {input}"

    first_run: list[str] = []
    a, b = (arm(h.wrap(answer, id="chat"), [wait]) for h in servers)
    first = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    async with asyncio.timeout(10):
        while ("tool", "started") not in log:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
    second = await asyncio.wait_for(b.run("two", user="ada", thread=THREAD, mode="interrupt"), 10)
    stopped = await asyncio.wait_for(first, 10)
    assert (stopped.status, second.status) == (RunStatus.CANCELLED, RunStatus.SUCCESS)
    # the call was cancelled while its run still held the turn, and only then did "two" start
    assert log == [
        ("start", "one"),
        ("tool", "started"),
        ("tool", "cancelled", RunStatus.RUNNING),
        ("start", "two"),
    ]
    ended = await servers[0].runs.get(stopped.run_id)
    assert ended is not None and ended.status is RunStatus.CANCELLED


async def test_conv04_a_cancel_from_another_server_stops_the_call_at_the_next_heartbeat(
    servers: tuple[Harness, Harness],
) -> None:
    """CONV-04 / SEC-R-3: ``agent.cancel`` on one server, of a run stuck in a silent tool call
    on the other: the run store cancels it at once, the holding server's next heartbeat is
    refused, and it stops the call then — within one heartbeat, not when its hold lapses."""
    runs = servers[0].runs
    beats: list[str] = []
    heartbeat = runs.heartbeat

    async def counted(run_id: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return await heartbeat(run_id, *args, **kwargs)
        finally:
            beats.append(run_id)

    runs.heartbeat = counted  # type: ignore[method-assign]
    log: Log = []
    a, b = (chat(h, log, {"one": asyncio.Event()}) for h in servers)
    working = asyncio.create_task(a.run("one", user="ada", thread=THREAD))
    await started(log, "one")
    run_id = await thread_run(runs, "one")
    cancelled = await b.cancel(run_id, reason="stop")
    before = len(beats)
    stopped = await asyncio.wait_for(working, 10)
    assert cancelled.status is RunStatus.CANCELLED and stopped.status is RunStatus.CANCELLED
    assert ("cancelled", "one") in log
    assert len(beats) - before <= 1  # the first heartbeat after the cancel stopped it


async def held_elsewhere(runs: LocalRuns, user: str, thread: str) -> str:
    """A turn of ``user``'s conversation ``thread`` that another server works on: its run."""
    key = f"thread:{conversation_key('default', user, thread)}"
    start = RunStart(tenant_id="default", agent_id="chat", user_id=user)
    start = start.model_copy(update={"thread_id": thread, "concurrency_key": key})
    run_id = (await runs.start(start)).run_id
    await runs.heartbeat(run_id, "another-server", lease_seconds=600)
    return run_id


async def test_secr11_another_users_turns_on_a_thread_named_alike_are_never_read_or_waited_for() -> (
    None
):
    """SEC-R-11: the line is listed by its own key (``thread:<conversation>``), not by thread
    name: 200 turns of another user on a thread named alike, running on another server, are
    neither read (no ``get``) nor waited for."""
    runs = Counted()
    bobs = {await held_elsewhere(runs, "bob", THREAD) for _ in range(200)}
    mine = await runs.start(turn_of("ada"))
    runs.got.clear()
    assert await Line(runs, mine, LRU(), Followers(lambda: None)).ahead() == []
    assert bobs.isdisjoint(runs.got) and runs.answers_read == []
    await runs.finish(mine.run_id, RunStatus.SUCCESS)
    async with Harness(config=Settings(), runs=runs) as h:
        agent = chat(h, [], {})
        answers = [
            (await asyncio.wait_for(agent.run(f"ada {n}", user="ada", thread=THREAD), 10)).answer
            for n in range(2)
        ]
    assert answers == ["re: ada 0", "re: ada 1"]
    assert bobs.isdisjoint(runs.got)  # bob's turns held none up, and none was read
    await runs.aclose()


async def test_perf1_runs_waiting_behind_runs_elsewhere_follow_few_streams_and_stay_held() -> None:
    """PERF-1: 200 chats each with a message waiting behind a run another server holds: this
    server follows at most 32 event streams at once (the others read their line again after a
    pause), every waiting run's hold heartbeat is still sent, and all are answered."""
    runs = LoggedRuns()
    following = [0, 0]  # open now, the most at once
    stream, heartbeat = runs.stream_events, runs.heartbeat
    held: set[str] = set()

    async def counted(run_id: str, **kwargs: Any) -> Any:
        following[0] += 1
        following[1] = max(following)
        try:
            async for entry in stream(run_id, **kwargs):
                yield entry
        finally:
            following[0] -= 1

    async def beating(run_id: str, *args: Any, **kwargs: Any) -> Any:
        held.add(run_id)
        return await heartbeat(run_id, *args, **kwargs)

    # a turn of each chat held by another server
    elsewhere = [await held_elsewhere(runs, "ada", f"chat-{n}") for n in range(200)]
    runs.stream_events, runs.heartbeat = counted, beating  # type: ignore[method-assign]
    async with Harness(config=Settings(), runs=runs) as h:
        agent = chat(h, [], {})
        sent = [
            asyncio.create_task(agent.run("hello", user="ada", thread=f"chat-{n}"))
            for n in range(200)
        ]
        async with asyncio.timeout(30):
            while len(held) < 200:  # every waiting run was held  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        assert 0 < following[1] <= 32
        for run_id in elsewhere:
            await runs.finish(run_id, RunStatus.SUCCESS)
        results = await asyncio.wait_for(asyncio.gather(*sent), 60)
    assert all(r.status is RunStatus.SUCCESS for r in results) and following[1] <= 32
    await runs.aclose()


async def test_perf6_a_turn_reads_its_earlier_turns_in_three_listings_however_many_wait(
    servers: tuple[Harness, Harness],
) -> None:
    """PERF-6: a turn with 1,000 newer messages of its chat waiting reads the earlier turns
    with one listing a status (answered, cut short, cancelled), never paging past the waiting
    ones."""
    runs = servers[0].runs
    log: Log = []
    agent = chat(servers[0], log, {})
    assert (await agent.run("first", user="ada", thread=THREAD)).answer == "re: first"
    key = f"thread:{conversation_key('default', 'ada', THREAD)}"
    waiting = RunStart(tenant_id="default", agent_id="chat", user_id="ada", thread_id=THREAD)
    for _ in range(1000):  # newer messages, waiting their turn on another server
        await runs.start(waiting.model_copy(update={"run_id": new_id("run_")}))
    listings: list[Any] = []
    iterate = runs.iterate

    def counted(**kwargs: Any) -> Any:
        listings.append(kwargs.get("status"))
        return iterate(**kwargs)

    runs.iterate = counted  # type: ignore[method-assign]
    record = await runs.start(waiting.model_copy(update={"run_id": "run_asking"}))
    runtime = SimpleNamespace(
        agent=agent,
        parent=None,
        thread=THREAD,
        run_id=record.run_id,
        tenant="default",
        agent_id="chat",
        user="ada",
        conversation=key.removeprefix("thread:"),
    )
    said = await earlier(runtime)  # type: ignore[arg-type]
    assert [s.content for s in said] == ["first", "re: first"]
    assert len(listings) <= 3 and None not in listings


async def test_arch5_a_chats_turn_nobody_ever_held_is_ended_and_frees_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARCH-5: a server killed between recording a chat's turn and holding it leaves a run
    nobody holds: the store ends it after the grace it gives an answered run (60 s; 0.2 s
    here) — agent-runs' ticker, this store's sweep at a claim — and the chat's next message
    runs."""
    monkeypatch.setattr(local_runs, "RESUME_GRACE_SECONDS", 0.2)
    runs = LocalRuns()
    key = f"thread:{conversation_key('default', 'ada', THREAD)}"
    start = RunStart(tenant_id="default", agent_id="chat", user_id="ada", thread_id=THREAD)
    orphan = await runs.start(start.model_copy(update={"concurrency_key": key}))
    keyless = await runs.start(start.model_copy(update={"run_id": "run_keyless"}))
    async with Harness(config=Settings(), runs=runs) as h:
        sent = asyncio.create_task(chat(h, [], {}).run("next", user="ada", thread=THREAD))
        async with asyncio.timeout(10):
            while (await runs.get(orphan.run_id)).status is RunStatus.RUNNING:  # type: ignore[union-attr]
                await runs.claim("sweeper", ["nobody"])  # this store sweeps lapsed holds at a claim
                await asyncio.sleep(0.02)
        answered = await asyncio.wait_for(sent, 10)
    assert answered.answer == "re: next"
    ended, kept = await runs.get(orphan.run_id), await runs.get(keyless.run_id)
    assert ended is not None and ended.error is not None and ended.error.code == "holder_lost"
    assert kept is not None and kept.status is RunStatus.RUNNING  # a run with no key: as before
    await runs.aclose()


async def test_arch19_a_chats_messages_keep_their_order_whether_or_not_the_tenant_is_named() -> (
    None
):
    """ARCH-19: two callers of one chat, one naming the tenant and one not, send in order."""
    turns: list[str] = []

    async def answer(input: str, agent: Runtime) -> str:
        turns.append(input)
        return input

    questions = [f"message {n}" for n in range(10)]
    runs = Laggy()
    async with Harness(config=Settings(), runs=runs) as h:
        agent = h.wrap(answer, id="chat")
        sent = [
            asyncio.create_task(
                agent.run(q, user="ada", thread=THREAD, tenant=(None, "default")[n % 2])
            )
            for n, q in enumerate(questions)
        ]
        await asyncio.wait_for(asyncio.gather(*sent), 30)
    assert turns == questions
    await runs.aclose()
