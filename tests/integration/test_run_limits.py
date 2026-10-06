"""A run's time, its version and its cancellation, through the public API: ``timeout=`` (working
time, across attempts and a crash) and ``deadline=`` end a run ``TIMEOUT``; ``ReAct``'s
``model_timeout`` bounds each model call; ``version=`` is recorded with every run, and a resume
on another version says so; ``agent.cancel`` stops a run in this process, ``CANCELLED`` with
the reason."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.support.adapters import BUILDERS
from tests.support.models import ScriptedChat
from tests.support.planned import Call
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, RunEventType, RunOutcome, RunStatus, ToolError
from trellis.harness.a2a import remote
from trellis.harness.runs import LocalRuns
from trellis.runs import ConflictError, Job, NotFoundError


class Crash(BaseException):
    """The worker process dies: nothing is written, the lease lapses."""


def lapse(store: LocalRuns, run_id: str) -> None:
    worker, _ = store._leases[run_id]
    store._leases[run_id] = (worker, datetime.now(UTC) - timedelta(seconds=1))


async def slow(input: Any, agent: Runtime) -> str:
    await asyncio.sleep(float(input))
    return "done"


# --------------------------------------------------------------------------- run time


async def test_a_run_past_its_time_limit_ends_timeout(harness: Harness) -> None:
    agent = harness.wrap(slow, id="slow")
    result = await agent.run("5", user="u", timeout=0.05)
    assert result.status is RunStatus.TIMEOUT and result.error is not None
    assert result.error.code == "run_timeout" and not result.error.retryable
    assert result.error.message == "the run worked past its time limit of 0.05s"
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.status is RunStatus.TIMEOUT
    assert record.timeout_seconds == 0.05
    assert (await agent.run("0", user="u", timeout=5)).answer == "done"


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_an_agents_time_limit_ends_its_runs_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    @tool(side_effects="read")
    async def wait(seconds: int) -> str:
        """Wait a while."""
        await asyncio.sleep(seconds)
        return "waited"

    plan: list[Call] = [("wait", {"seconds": 5})]
    target, tools = await BUILDERS[framework](harness, [wait], tmp_path, plan)
    agent = harness.wrap(target, id=f"slow-{framework}", tools=tools, timeout=0.3)
    result = await agent.run("wait", user="u")
    assert result.status is RunStatus.TIMEOUT and result.error is not None
    assert result.error.message == "the run worked past its time limit of 0.3s"
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.timeout_seconds == 0.3  # agent-runs enforces it too


async def test_the_agents_time_limit_holds_on_every_entry(harness: Harness) -> None:
    """``serve_chat``, ``serve_a2a``, ``h.evaluate`` and a scheduled run start an attempt the
    one way ``agent.run`` does: the agent's limit applies to each; a run's own overrides it."""
    agent = harness.wrap(slow, id="slow", timeout=0.1)
    limit = "the run worked past its time limit of 0.1s"
    assert (await agent.run("0.2", user="u", timeout=5)).answer == "done"
    app = FastAPI()
    agent.serve_chat(app)
    agent.serve_a2a(app, "http://limits.test/a2a")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://limits.test"
    ) as http:
        chat = {"threadId": "t1", "messages": [{"id": "m1", "role": "user", "content": "5"}]}
        response = await http.post("/agui/run", json=chat)
        assert limit in response.text
        async with remote("http://limits.test/a2a", tenant="default", user="u", client=http) as a2a:
            with pytest.raises(ToolError, match="TASK_STATE_FAILED"):
                await a2a("5")
    report = await harness.evaluate(agent, [{"input": "5"}], [])
    assert [(i.status, i.error) for i in report.items] == [("error", limit)]
    schedule = await agent.schedule("0 7 * * *", "5", on_behalf_of="ada")
    runs = harness.runs
    assert isinstance(runs, LocalRuns)
    runs._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    assert await harness.worker([agent]).run_once() is True
    [fired] = [r for r in runs._runs.values() if r.metadata.get("schedule_id")]
    assert fired.timeout_seconds is None  # a schedule names no limit: the agent's applies
    assert fired.status is RunStatus.TIMEOUT and fired.error is not None
    assert fired.error.message == limit
    timed_out = [r for r in runs._runs.values() if r.status is RunStatus.TIMEOUT]
    assert len(timed_out) == 4  # the chat run, the A2A task, the item and the scheduled run


def test_an_agents_time_limit_is_a_number_of_seconds_over_zero(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="slow: a timeout is a number of seconds over 0"):
        harness.wrap(slow, id="slow", timeout=0)


async def test_a_run_past_its_deadline_ends_timeout_on_the_stream(harness: Harness) -> None:
    deadline = datetime.now(UTC) - timedelta(seconds=1)
    agent = harness.wrap(slow, id="slow")
    events = [e async for e in agent.stream("5", user="u", timeout=60, deadline=deadline)]
    error, finished = events[-2:]
    assert error.type is RunEventType.RUN_ERROR and error.error is not None
    assert error.error.code == "run_deadline"
    assert finished.outcome is RunOutcome.TIMEOUT
    record = await harness.runs.get(finished.run_id)
    assert record is not None and record.deadline == deadline


async def test_working_time_counts_across_a_crash(harness: Harness) -> None:
    crashes = [Crash()]

    async def billing(input: str, agent: Runtime) -> str:
        await asyncio.sleep(0.3)
        if crashes:
            raise crashes.pop()
        return "billed"

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(billing, id="billing")
    handle = await agent.start("x", user="u", timeout=0.5)
    worker = harness.worker([agent])
    claimed = await store.claim(worker.worker_id, [agent.id])
    assert claimed is not None and claimed.run.timeout_seconds == 0.5
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    lapse(store, handle.run_id)
    assert await worker.run_once()  # the next attempt: 0.2 s left of the run's 0.5
    done = await handle.result(timeout=5)
    assert done.status is RunStatus.TIMEOUT and done.error is not None
    assert done.error.code == "run_timeout"
    record = await handle.status()
    assert record.attempt == 2 and record.worked_seconds >= 0.5


async def test_a_call_may_take_what_is_left_of_its_own_time_and_the_runs(
    harness: Harness,
) -> None:
    seen: dict[str, float | None] = {}

    @tool(side_effects="read", timeout=30)
    def look(sku: str) -> str:
        """Look a SKU up."""
        from trellis import current

        runtime = current()
        assert runtime is not None
        seen["call"] = runtime.remaining()
        return sku

    async def looking(input: str, agent: Runtime) -> str:
        seen["run"] = agent.remaining()
        return await agent.tools.call("look", sku="A")

    agent = harness.wrap(looking, id="looking", tools=[look])
    await agent.run("x", user="u", timeout=10)
    assert seen["call"] is not None and seen["call"] <= 10  # the run's time is the tighter
    await agent.run("x", user="u")
    assert seen["run"] is None and seen["call"] is not None and 29 < seen["call"] <= 30


# --------------------------------------------------------------------------- model calls


class Slow(ScriptedChat):
    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        await asyncio.sleep(5)
        return await super().complete(messages, **body)


class GivesUp(ScriptedChat):
    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        raise TimeoutError("the model's own client gave up")


async def test_a_model_call_past_its_timeout_fails_the_run_retryably(harness: Harness) -> None:
    agent = harness.wrap(ReAct(system="s", model=Slow(["hi"]), model_timeout=0.05), id="slow")
    result = await agent.run("q", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == "MODEL_ERROR" and result.error.retryable
    assert result.error.message == "the model did not answer within 0.05s"
    gives_up = harness.wrap(ReAct(system="s", model=GivesUp([])), id="gives-up")
    failed = await gives_up.run("q", user="u")
    assert failed.error is not None and failed.error.message == "the model did not answer"


# --------------------------------------------------------------------------- version


async def test_runs_carry_the_agents_version_and_a_resume_on_another_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return await agent.ask("Go on?")

    async with Harness(config=Settings(agent_version="2026.10")) as h:
        old = h.wrap(asks, id="asks")
        paused = await old.run("x", user="u")
        record = await h.runs.get(paused.run_id)
        assert record is not None and record.agent_version == "2026.10"
        async with Harness(config=Settings()) as deployed:
            deployed.runs = h.runs
            new = deployed.wrap(asks, id="asks", version="2026.11")
            assert paused.interrupt is not None
            with caplog.at_level(logging.WARNING, logger="trellis.run"):
                done = await new.resume(
                    paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u"
                )
    assert done.answer == "yes"
    assert f"run {paused.run_id} was started by asks 2026.10 and continues on 2026.11" in (
        caplog.text
    )


# --------------------------------------------------------------------------- cancel


async def test_a_run_in_this_process_is_cancelled_with_its_reason(harness: Harness) -> None:
    started = asyncio.Event()
    runs: list[str] = []

    async def waits(input: str, agent: Runtime) -> str:
        runs.append(agent.run_id)
        started.set()
        await asyncio.sleep(30)
        return "never"

    agent = harness.wrap(waits, id="waits")
    running = asyncio.create_task(agent.run("x", user="u"))
    await started.wait()
    record = await agent.cancel(runs[0], reason="a duplicate of run 7")
    assert record.status is RunStatus.CANCELLED
    assert (await running).status is RunStatus.CANCELLED

    started.clear()
    events: list[Any] = []

    async def watch() -> None:
        async for event in agent.stream("x", user="u"):
            events.append(event)

    streaming = asyncio.create_task(watch())
    await started.wait()
    await agent.cancel(runs[1])
    await streaming
    assert events[-1].outcome is RunOutcome.CANCELLED
    assert events[-1].data == {"reason": "cancelled"}


async def test_a_cancelled_queued_run_ends_cancelled_and_its_caller_too(
    harness: Harness,
) -> None:
    started = asyncio.Event()

    async def waits(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    agent = harness.wrap(waits, id="waits")
    handle = await agent.start("x", user="u")
    working = asyncio.create_task(harness.worker([agent]).run_once())
    await started.wait()
    assert (await handle.cancel(reason="not needed")).status is RunStatus.CANCELLED
    assert await working

    started.clear()
    runs: list[str] = []

    async def noted(input: str, agent: Runtime) -> str:
        runs.append(agent.run_id)
        return await waits(input, agent)

    other = harness.wrap(noted, id="noted")
    running = asyncio.create_task(other.run("x", user="u"))
    await started.wait()
    running.cancel()  # its caller goes away too: the caller's cancel wins
    record = await other.cancel(runs[0], reason="dup")
    assert record.status is RunStatus.CANCELLED
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_a_waiting_run_is_cancelled_at_once_and_an_ended_one_is_not(
    harness: Harness,
) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return await agent.ask("Go on?")

    agent = harness.wrap(asks, id="asks")
    paused = await agent.run("x", user="u")
    assert (await agent.cancel(paused.run_id)).status is RunStatus.CANCELLED
    handle = await agent.start("x", user="u")
    assert (await handle.cancel(reason="not needed")).status is RunStatus.CANCELLED
    assert not await harness.worker([agent]).run_once()  # nothing left to claim
    with pytest.raises(ConflictError, match="already ended CANCELLED"):
        await agent.cancel(paused.run_id)
    with pytest.raises(NotFoundError):
        await agent.cancel("run_elsewhere")


async def test_a_run_cancelled_elsewhere_is_stopped_by_its_worker(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.runs import worker as claim_loop

    monkeypatch.setattr(claim_loop, "_sleep", lambda seconds: asyncio.sleep(0.01))
    started = asyncio.Event()

    async def waits(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    agent = harness.wrap(waits, id="waits")
    handle = await agent.start("x", user="u", timeout=60)
    working = asyncio.create_task(harness.worker([agent]).run_once())
    await started.wait()
    # another process cancels it: only agent-runs (here, the store) is told
    await harness.runs.cancel(handle.run_id, reason="dup", tenant=handle.tenant)
    assert await asyncio.wait_for(working, 5)
    assert (await handle.status()).status is RunStatus.CANCELLED


async def test_a_cancelled_run_keeps_its_transcript_and_says_nothing_of_the_agent(
    memory_harness: Harness, memory_service: Any
) -> None:
    started = asyncio.Event()
    runs: list[str] = []

    async def waits(input: str, agent: Runtime) -> str:
        runs.append(agent.run_id)
        started.set()
        await asyncio.sleep(30)
        return "never"

    agent = memory_harness.wrap(waits, id="waits")
    running = asyncio.create_task(agent.run("reconcile March", user="u"))
    await started.wait()
    await agent.cancel(runs[0], reason="dup")
    assert (await running).status is RunStatus.CANCELLED
    await memory_harness.writes.drain()
    [batch] = memory_service.named("messages")
    assert [m["content"] for m in batch.body["messages"]] == ["reconcile March"]
    assert memory_service.named("feedback") == []  # no outcome: a cancel says nothing
