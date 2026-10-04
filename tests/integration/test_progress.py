"""A worker that dies mid-run: the journal saved as a progress checkpoint on the heartbeat
after each side-effecting tool call means the next attempt replays those calls instead of
running them again."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness import runtime as runtime_module
from trellis.harness.clients.runs import LeaseLost, LocalRuns, RunStoreError

paid: list[int] = []
looked: list[str] = []


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an invoice."""
    paid.append(amount)
    return f"paid {amount}"


@tool(side_effects="read")
def balance(account: str) -> int:
    """An account's balance."""
    looked.append(account)
    return 100


class Crash(BaseException):
    """The worker process dies: nothing is written, the lease lapses."""


@pytest.fixture(autouse=True)
def _reset() -> None:
    paid.clear()
    looked.clear()


def lapse(store: LocalRuns, run_id: str) -> None:
    """The dead worker's lease runs out; the next claim puts the run back on the queue."""
    worker, _ = store._leases[run_id]
    store._leases[run_id] = (worker, datetime.now(UTC) - timedelta(seconds=1))


async def crash_once(store: LocalRuns, agent: Any, handle: Any) -> None:
    worker = agent.harness.worker([agent])
    record = await store.claim(worker.worker_id, [agent.id], 60)
    assert record is not None
    with pytest.raises(Crash):
        await agent._claimed(record, worker.worker_id, lease_seconds=60)
    lapse(store, handle.run_id)


async def test_a_crash_after_a_write_does_not_run_the_write_again(harness: Harness) -> None:
    crashes = [Crash()]

    async def billing(input: str, agent: Runtime) -> str:
        receipt = await agent.tools.call("pay", amount=5)
        if crashes:
            raise crashes.pop()
        return receipt

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(billing, id="billing", tools=[pay])
    handle = await agent.start("invoice 7", user="u")
    await crash_once(store, agent, handle)
    record = await handle.status()
    assert record.status is RunStatus.RUNNING and record.checkpoint is not None
    assert await harness.worker([agent]).run_once()  # the next attempt
    done = await handle.result(timeout=5)
    assert done.status is RunStatus.SUCCESS and done.answer == "paid 5"
    assert paid == [5]  # replayed from the checkpoint, not paid twice
    assert (await handle.status()).attempt == 2


async def test_a_resumed_react_run_replays_its_model_steps_and_its_writes(
    harness: Harness,
) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """Crash the worker."""
        if crashes:
            raise crashes.pop()
        return "fine"

    model = ScriptedChat([("pay", {"amount": 3}), ("boom", {}), "paid 3"])
    target = ReAct(system="You pay.", model=model)
    agent = harness.wrap(target, id="payer", tools=[pay, boom])
    store = harness.runs
    assert isinstance(store, LocalRuns)
    handle = await agent.start("pay 3", user="u")
    await crash_once(store, agent, handle)
    # the model's script from the step after pay (that one came from the checkpoint)
    model.turns = [("boom", {}), "paid 3"]
    asked_before = len(model.requests)
    assert await harness.worker([agent]).run_once()
    done = await handle.result(timeout=5)
    assert done.answer == "paid 3" and paid == [3]
    # the first model step (it chose pay) came from the checkpoint: only the steps after it
    assert len(model.requests) - asked_before == 2


async def test_reads_are_saved_at_most_every_progress_interval(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[dict[str, Any] | None] = []
    store = harness.runs
    assert isinstance(store, LocalRuns)
    original = store.heartbeat

    async def heartbeat(run_id: str, worker_id: str, lease: float, **kw: Any) -> None:
        saved.append(kw.get("checkpoint"))
        await original(run_id, worker_id, lease, **kw)

    monkeypatch.setattr(store, "heartbeat", heartbeat)

    async def reader(input: str, agent: Runtime) -> str:
        for account in ("a", "b", "c"):
            await agent.tools.call("balance", account=account)
        return await agent.tools.call("pay", amount=1)

    agent = harness.wrap(reader, id="reader", tools=[balance, pay])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(timeout=5)).status is RunStatus.SUCCESS
    # the first read saves, the next two are within the interval, the write always saves
    assert len(saved) == 2
    assert saved[-1] is not None and len(saved[-1]["calls"]) == 4


async def test_an_in_process_run_saves_no_progress(harness: Harness) -> None:
    async def billing(input: str, agent: Runtime) -> str:
        return await agent.tools.call("pay", amount=1)

    agent = harness.wrap(billing, id="billing", tools=[pay])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS and paid == [1]


async def test_a_progress_checkpoint_too_large_or_refused_is_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runtime_module, "MAX_CHECKPOINT_BYTES", 10)

    async def billing(input: str, agent: Runtime) -> str:
        await agent.tools.call("pay", amount=1)
        return await agent.tools.call("pay", amount=2)

    async with Harness(config=Settings()) as h:
        agent = h.wrap(billing, id="billing", tools=[pay])
        handle = await agent.start("x", user="u")
        assert await h.worker([agent]).run_once()
        assert (await handle.result(timeout=5)).answer == "paid 2"
    assert caplog.text.count("too large to save as progress") == 1  # said once per run

    monkeypatch.setattr(runtime_module, "MAX_CHECKPOINT_BYTES", 1024 * 1024)
    async with Harness(config=Settings()) as h:

        async def refused(*args: Any, **kwargs: Any) -> None:
            if "checkpoint" in kwargs:
                raise RunStoreError("413 PAYLOAD_TOO_LARGE")

        monkeypatch.setattr(h.runs, "heartbeat", refused)
        agent = h.wrap(billing, id="billing", tools=[pay])
        handle = await agent.start("x", user="u")
        assert await h.worker([agent]).run_once()
        assert (await handle.result(timeout=5)).answer == "paid 2"
    assert "progress of run" in caplog.text and "413" in caplog.text


async def test_a_lease_lost_while_saving_progress_stops_the_run(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lost(*args: Any, **kwargs: Any) -> None:
        raise LeaseLost("w no longer holds the run")

    monkeypatch.setattr(harness.runs, "heartbeat", lost)

    async def billing(input: str, agent: Runtime) -> str:
        await agent.tools.call("pay", amount=1)
        return await agent.tools.call("pay", amount=2)

    agent = harness.wrap(billing, id="billing", tools=[pay])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert paid == [1]  # nothing after the lost lease
    assert (await handle.status()).status is RunStatus.RUNNING  # nothing written
