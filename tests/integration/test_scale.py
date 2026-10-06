"""W3 scale: a run's events in the run store's log (so any replica streams any run), read by
``agent.events`` and by an AG-UI reconnect on another replica; a 429 from agent-runs as one
clear retryable error; the queue's order (priority, a conversation's runs one at a time) and the
same in process; a schedule's limits; queue wait and admission metrics; JSON logs."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.support.runlog import LoggedRuns
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunEvent, RunEventType, RunOutcome, RunStatus
from trellis.harness import logs, telemetry
from trellis.harness.agent import Throttled
from trellis.harness.agui.sse import decode
from trellis.harness.runlog import BATCH, RunLog
from trellis.harness.runs import LocalRuns
from trellis.runs import LeaseLostError, RateLimitedError


@tool(side_effects="write")
def charge(order: str) -> str:
    """Charge an order."""
    return f"charged {order}"


async def charging(input: str, agent: Runtime) -> Any:
    receipt = await agent.tools.call("charge", order=input)
    size = await agent.ask("Which size?", options=["S", "L"])
    return f"{receipt}; {size}"


@pytest.fixture
async def logged() -> AsyncIterator[tuple[Harness, LoggedRuns]]:
    runs = LoggedRuns()
    async with Harness(config=Settings(), runs=runs) as h:
        yield h, runs


# --------------------------------------------------------------------------- the run's log


async def test_every_event_is_in_the_runs_log_before_its_pause_or_ending_is_recorded(
    logged: tuple[Harness, LoggedRuns],
) -> None:
    h, runs = logged
    agent = h.wrap(charging, id="billing", tools=[charge])
    streamed: list[RunEvent] = []
    async for event in agent.stream("o-1", user="ada"):
        streamed.append(event)
        if event.type is RunEventType.RUN_FINISHED:  # heard once the pause is recorded
            assert runs._runs[event.run_id].status is RunStatus.PAUSED
    paused = streamed[-1].data["interrupt"]
    first = runs.logged(paused["run_id"])
    assert first == streamed  # the same events, in order, the last ones before the pause
    assert first[-1].outcome is RunOutcome.INTERRUPT
    done = await agent.resume(paused["interrupt_id"], "answer", answer="L", reviewer="lee")
    assert done.answer == "charged o-1; L"
    both = runs.logged(done.run_id)
    assert [e.attempt for e in both].count(2) == len(both) - len(first)
    assert both[-1].type is RunEventType.RUN_FINISHED and both[-1].outcome is RunOutcome.SUCCESS
    [decided] = [e.data for e in both if e.data.get("name") == "decision"]
    assert (decided["interrupt_id"], decided["reviewer"]) == (paused["interrupt_id"], "lee")


async def test_agent_events_reads_a_run_from_the_log_wherever_it_ran(
    logged: tuple[Harness, LoggedRuns],
) -> None:
    h, _ = logged
    agent = h.wrap(charging, id="billing", tools=[charge])
    paused = await agent.run("o-2", user="ada")
    assert paused.interrupt is not None
    follower = asyncio.create_task(_collect(agent.events(paused.run_id)))
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="S", reviewer="lee")
    events = await asyncio.wait_for(follower, 5)
    outcomes = [e.outcome for e in events if e.type is RunEventType.RUN_FINISHED]
    assert outcomes == [RunOutcome.INTERRUPT, RunOutcome.SUCCESS]  # every attempt, to the end
    later = [e async for e in agent.events(paused.run_id, after=len(events) - 1)]
    assert later == events[-1:]


async def test_a_worker_appends_its_runs_events_fenced_by_its_lease(
    logged: tuple[Harness, LoggedRuns],
) -> None:
    h, runs = logged
    agent = h.wrap(charging, id="billing", tools=[charge])
    handle = await agent.start("o-3", user="ada", thread="chat-1", priority=5)
    worker = h.worker([agent], concurrency=1)
    assert await worker.run_once()
    record = await handle.status()
    assert (record.priority, record.concurrency_key) == (5, "thread:chat-1")
    assert {w for r, w, _ in runs.appends if r == handle.run_id} == {worker.worker_id}
    assert runs.logged(handle.run_id)[-1].outcome is RunOutcome.INTERRUPT


async def test_the_log_is_best_effort_and_a_lost_lease_stops_it(caplog: Any) -> None:
    class Refusing:
        def __init__(self, error: Exception) -> None:
            self.error = error
            self.calls = 0

        async def append_events(self, run_id: str, events: Sequence[RunEvent], **kw: Any) -> Any:
            self.calls += 1
            raise self.error

        def stream_events(self, run_id: str, **kw: Any) -> AsyncIterator[Any]:
            raise NotImplementedError

    from trellis.harness.events import RunEvents
    from trellis.harness.identity import Identity

    identity = Identity(tenant="t", user="u", agent_id="a", run_id="run_1", thread=None)
    events = RunEvents(identity.context())
    heard: list[RunEvent] = []
    events.listen(heard.append)
    down = Refusing(ConnectionError("agent-runs is down"))
    log = RunLog(down, "run_1", tenant="t", worker_id=None, events=events)  # type: ignore[arg-type]
    events.record(log)
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        for n in range(3):
            events.custom("step", n=n)
        await log.flush()
    warnings = [e for e in heard if e.data.get("code") == "events_undelivered"]
    assert len(warnings) == 1 and "agent-runs is down" in warnings[0].data["message"]
    assert caplog.text.count("not in agent-runs' event log") == 1
    lost = Refusing(LeaseLostError("gone", code="LEASE_LOST", status=409))
    log = RunLog(lost, "run_1", tenant="t", worker_id="w", events=events)  # type: ignore[arg-type]
    for event in heard[:2]:
        log(event)
    await log.close()
    log(heard[0])  # closed: nothing more is sent
    await log.flush()
    assert lost.calls == 1 and log.closed
    assert BATCH == 500


async def test_in_process_events_follow_a_run_here_until_it_ends(harness: Harness) -> None:
    agent = harness.wrap(charging, id="billing", tools=[charge])
    paused = await agent.run("o-4", user="ada")
    assert paused.interrupt is not None
    follower = asyncio.create_task(_collect(agent.events(paused.run_id)))
    await asyncio.sleep(0)
    await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="lee")
    assert await asyncio.wait_for(follower, 5) == []  # cancelled while it waited: no attempt
    assert [e async for e in agent.events(paused.run_id)] == []  # it ended
    with pytest.raises(Exception, match="no run run_nope"):
        _ = [e async for e in agent.events("run_nope")]
    again = await agent.run("o-5", user="ada")
    assert again.interrupt is not None
    followers = [asyncio.create_task(_collect(agent.events(again.run_id))) for _ in range(2)]
    await asyncio.sleep(0)
    await agent.cancel(again.run_id)
    assert await asyncio.wait_for(asyncio.gather(*followers), 5) == [[], []]
    assert not agent._watching


async def test_in_process_events_include_the_attempt_running_now(harness: Harness) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        started.set()
        await release.wait()
        return "done"

    agent = harness.wrap(slow, id="slow")
    running = asyncio.create_task(agent.run("go", user="ada"))
    await started.wait()
    [run_id] = list(agent.running)
    follower = asyncio.create_task(_collect(agent.events(run_id)))
    await asyncio.sleep(0)
    release.set()
    events = await asyncio.wait_for(follower, 5)
    assert (await running).answer == "done"
    assert events[-1].type is RunEventType.RUN_FINISHED


async def _collect(events: AsyncIterator[RunEvent]) -> list[RunEvent]:
    return [e async for e in events]


# --------------------------------------------------------------------------- AG-UI, any replica


async def test_a_reconnect_to_another_replica_reads_the_run_from_the_log() -> None:
    async def billed(input: str, agent: Runtime) -> Any:
        if input == "boom":
            raise RuntimeError("boom")
        return await agent.tools.call("charge", order=input)

    runs = LoggedRuns()
    replicas = []
    for _ in range(2):
        h = Harness(config=Settings(), runs=runs)
        app = FastAPI()
        h.wrap(billed, id="chat", tools=[charge]).serve_chat(
            app, identity=lambda r: r.headers.get("x-user", "u1")
        )
        replicas.append((h, app))
    (_, app_one), (_, app_two) = replicas
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app_one), base_url="http://a") as a,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app_two), base_url="http://b") as b,
    ):
        payload = {"threadId": "t1", "messages": [{"id": "m1", "role": "user", "content": "o"}]}
        served = decode((await a.post("/agui/run", json=payload)).text)
        run_id = served[0][1]["runId"]
        elsewhere = decode((await b.get(f"/agui/runs/{run_id}/events")).text)
        assert [e for _, e in elsewhere] == [e for _, e in served]  # the same events
        assert [n for n, _ in elsewhere][:2] == [1, 2]  # numbered by their position in the log
        last = {"Last-Event-ID": str(elsewhere[-2][0])}
        assert (
            decode((await b.get(f"/agui/runs/{run_id}/events", headers=last)).text)
            == (elsewhere[-1:])
        )
        boom = {**payload, "messages": [{"id": "m2", "role": "user", "content": "boom"}]}
        failed = decode((await a.post("/agui/run", json=boom)).text)
        logged = decode((await b.get(f"/agui/runs/{failed[0][1]['runId']}/events")).text)
        assert [e for _, e in logged] == [e for _, e in failed]  # RUN_ERROR folded, as served
        assert (await b.get("/agui/runs/run_other/events")).status_code == 404
        theirs = await b.get(f"/agui/runs/{run_id}/events", headers={"x-user": "u2"})
        assert theirs.status_code == 404  # only the run's own user
    for h, _ in replicas:
        await h.aclose()


async def test_without_a_log_a_reconnect_must_reach_the_replica_that_served_it(
    harness: Harness,
) -> None:
    app = FastAPI()
    harness.wrap(charging, id="chat", tools=[charge]).serve_chat(app, identity=lambda r: "u1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c") as c:
        assert (await c.get("/agui/runs/run_x/events")).status_code == 404


# --------------------------------------------------------------------------- admission (429)


class Limited(LocalRuns):
    """agent-runs answering 429 after the SDK's retries."""

    async def start(self, start: Any, *, queue: bool = False) -> Any:
        raise RateLimitedError(
            "budget spent", code="RATE_LIMIT", status=429, retryable=True, retry_after=7.0
        )

    async def resume(self, resolution: Any, *, tenant: str | None = None) -> Any:
        raise RateLimitedError("budget spent", code="RATE_LIMIT", status=429, retry_after=None)


async def test_a_rate_limited_start_is_one_clear_retryable_error_and_a_429_on_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted: list[str] = []
    monkeypatch.setattr(telemetry.metrics, "rate_limited", counted.append)
    async with Harness(config=Settings(), runs=Limited()) as h:
        agent = h.wrap(charging, id="billing", tools=[charge])
        with pytest.raises(Throttled, match="try again after 7s") as raised:
            await agent.run("o", user="ada")
        error = raised.value.to_error()
        assert error.retryable and error.category.value == "RATE_LIMIT"
        assert error.details == {"retry_after": 7.0} and error.source is not None
        with pytest.raises(Throttled):
            await agent.start("o", user="ada")
        app = FastAPI()
        agent.serve_chat(app, identity=lambda r: "u1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://chat"
        ) as http:
            payload = {"threadId": "t", "messages": [{"id": "m", "role": "user", "content": "o"}]}
            response = await http.post("/agui/run", json=payload)
        assert response.status_code == 429 and response.headers["retry-after"] == "7"
        assert response.json()["code"] == "RATE_LIMIT" and response.json()["retryable"]
    assert counted == ["start", "start", "start"]


async def test_a_rate_limited_resume_says_to_try_later() -> None:
    runs = Limited()
    async with Harness(config=Settings(), runs=runs) as h:
        agent = h.wrap(charging, id="billing", tools=[charge])
        record = await LocalRuns.start(
            runs, await agent._start("o", user="ada", thread=None, tenant=None)
        )
        from trellis.contracts import Interrupt

        asked = Interrupt(tenant_id=record.tenant_id, run_id=record.run_id, question="Go?")
        await runs.pause(asked.model_copy(update={"interrupt_id": f"{record.run_id}.1.1"}))
        with pytest.raises(Throttled, match="try again later"):
            await agent.resume(f"{record.run_id}.1.1", "answer", answer="x", reviewer="a")


# --------------------------------------------------------------------------- the queue's order


async def test_claims_take_the_highest_priority_and_one_run_of_a_conversation_at_a_time(
    harness: Harness,
) -> None:
    order: list[str] = []

    async def note(input: str, agent: Runtime) -> str:
        order.append(input)
        return input

    agent = harness.wrap(note, id="noting")
    await agent.start("low", user="u", priority=-1)
    await agent.start("first in chat", user="u", thread="chat")
    await agent.start("second in chat", user="u", thread="chat")
    await agent.start("urgent", user="u", priority=10, concurrency_key="mine")
    runs = harness.runs
    assert isinstance(runs, LocalRuns)
    claimed = [await runs.claim("w", ["noting"]) for _ in range(4)]
    picked = [c.run.input if c is not None else None for c in claimed]
    # the chat's second run waits while its first is running
    assert picked == ["urgent", "first in chat", "low", None]
    assert claimed[1] is not None
    await runs.finish(claimed[1].run.run_id, RunStatus.SUCCESS, worker_id="w")
    last = await runs.claim("w", ["noting"])
    assert last is not None and last.run.input == "second in chat"


async def test_a_second_message_to_a_busy_conversation_waits_for_the_first(
    harness: Harness,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def chat(input: str, agent: Runtime) -> str:
        order.append(f"start {input}")
        if input == "one":
            started.set()
            await release.wait()
        order.append(f"end {input}")
        return input

    agent = harness.wrap(chat, id="chat")
    first = asyncio.create_task(agent.run("one", user="u", thread="t"))
    await started.wait()
    second = asyncio.create_task(agent.run("two", user="u", thread="t"))
    other = await agent.run("three", user="u", thread="elsewhere")  # another conversation runs
    await asyncio.sleep(0.01)
    assert order == ["start one", "start three", "end three"]
    release.set()
    assert [(await first).answer, (await second).answer] == ["one", "two"]
    assert order[3:] == ["end one", "start two", "end two"]
    assert other.answer == "three" and not agent.turns._held


async def test_a_schedule_carries_the_agents_limit_and_version_to_every_fire(
    harness: Harness,
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return input

    agent = harness.wrap(fn, id="briefing", timeout=60, version="2026.10")
    schedule = await agent.schedule("manual", "brief", on_behalf_of="ada")
    assert (schedule.timeout_seconds, schedule.agent_version) == (60, "2026.10")


async def test_queue_wait_is_measured_when_a_worker_claims(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    waited: list[tuple[str, float]] = []
    monkeypatch.setattr(
        telemetry.metrics, "queue_waited", lambda agent, s: waited.append((agent, s))
    )
    agent = harness.wrap(charging, id="billing", tools=[charge])
    handle = await agent.start("o-6", user="ada")
    worker = harness.worker([agent], concurrency=1)
    assert await worker.run_once()
    paused = await handle.result(timeout=5)
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="S", reviewer="lee")
    assert await worker.run_once()
    assert [a for a, _ in waited] == ["billing", "billing"] and all(s >= 0 for _, s in waited)


# --------------------------------------------------------------------------- logs


def test_a_worker_logs_json_lines_unless_a_terminal_reads_them() -> None:
    line = logs.JSONFormatter().format(
        logging.LogRecord("trellis.run", logging.WARNING, "f", 1, "run %s failed", ("r1",), None)
    )
    assert json.loads(line) | {"time": "t"} == {
        "time": "t",
        "level": "WARNING",
        "logger": "trellis.run",
        "message": "run r1 failed",
    }
    record = logging.LogRecord("trellis.run", logging.ERROR, "f", 1, "boom", None, None)
    record.run_id = "r2"
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()
    parsed = json.loads(logs.JSONFormatter().format(record))
    assert parsed["run_id"] == "r2" and "ValueError: bad" in parsed["exception"]

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    root = logging.getLogger()
    kept = root.handlers[:]
    for stream, formatter in ((io.StringIO(), logs.JSONFormatter), (Terminal(), logging.Formatter)):
        root.handlers.clear()
        logs.configure(stream)
        assert type(root.handlers[0].formatter or logging.Formatter()) is formatter
    root.handlers[:] = kept
