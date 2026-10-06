"""A worker that dies mid-run: the journal saved as a progress checkpoint on the heartbeat
after each side-effecting tool call means the next attempt replays those calls instead of
running them again — a journal larger than a checkpoint may be too, stored as a run artifact
the checkpoint names."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, current, tool
from trellis.contracts import HarnessError, Interrupt, RunRecord, RunStatus
from trellis.harness.journal import JOURNAL_REF, MAX_CHECKPOINT_BYTES, content_key
from trellis.harness.runs import LocalRuns
from trellis.runs import Job, Lease, LeaseLostError, PayloadTooLargeError

paid: list[int] = []
looked: list[str] = []
exported: list[int] = []
#: a tool output larger than a checkpoint may be
REPORT = "x" * (MAX_CHECKPOINT_BYTES + 1)


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


@tool(side_effects="write")
def export(rows: int) -> str:
    """Export a report."""
    exported.append(rows)
    return REPORT


class Crash(BaseException):
    """The worker process dies: nothing is written, the lease lapses."""


class Bounded(LocalRuns):
    """The run store refusing a checkpoint over agent-runs' bound, as agent-runs does."""

    async def heartbeat(
        self, run_id: str, worker_id: str, *, checkpoint: dict[str, Any] | None = None, **kw: Any
    ) -> Lease:
        bounded(checkpoint)
        return await super().heartbeat(run_id, worker_id, checkpoint=checkpoint, **kw)

    async def pause(
        self,
        interrupt: Interrupt,
        *,
        checkpoint: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> RunRecord:
        bounded(checkpoint)
        return await super().pause(interrupt, checkpoint=checkpoint, worker_id=worker_id)


def bounded(checkpoint: dict[str, Any] | None) -> None:
    if len(json.dumps(checkpoint, separators=(",", ":"))) > MAX_CHECKPOINT_BYTES:
        raise PayloadTooLargeError("checkpoint too large", code="PAYLOAD_TOO_LARGE", status=413)


@pytest.fixture(autouse=True)
def _reset() -> None:
    paid.clear()
    looked.clear()
    exported.clear()


def lapse(store: LocalRuns, run_id: str) -> None:
    """The dead worker's lease runs out; the next claim puts the run back on the queue."""
    worker, _ = store._leases[run_id]
    store._leases[run_id] = (worker, datetime.now(UTC) - timedelta(seconds=1))


async def crash_once(store: LocalRuns, agent: Any, handle: Any) -> None:
    worker = agent.harness.worker([agent])
    claimed = await store.claim(worker.worker_id, [agent.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
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

    async def heartbeat(run_id: str, worker_id: str, **kw: Any) -> Lease:
        saved.append(kw.get("checkpoint"))
        return await original(run_id, worker_id, **kw)

    monkeypatch.setattr(store, "heartbeat", heartbeat)

    async def reader(input: str, agent: Runtime) -> str:
        for account in ("a", "b", "c"):
            await agent.tools.call("balance", account=account)
        return await agent.tools.call("pay", amount=1)

    agent = harness.wrap(reader, id="reader", tools=[balance, pay])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(timeout=5)).status is RunStatus.SUCCESS
    # the first read saves, the next two are within the interval, the write saves before it
    # runs (marked started) and after
    assert len(saved) == 3
    assert saved[1] is not None and saved[1]["started"] == {
        content_key("call", "pay", {"amount": 1}): 1
    }
    assert saved[-1] is not None and len(saved[-1]["calls"]) == 4


async def test_an_in_process_run_saves_no_progress(harness: Harness) -> None:
    async def billing(input: str, agent: Runtime) -> str:
        return await agent.tools.call("pay", amount=1)

    agent = harness.wrap(billing, id="billing", tools=[pay])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS and paid == [1]


async def test_a_refused_progress_save_is_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def billing(input: str, agent: Runtime) -> str:
        await agent.tools.call("pay", amount=1)
        return await agent.tools.call("pay", amount=2)

    async with Harness(config=Settings()) as h:

        async def refused(*args: Any, **kwargs: Any) -> None:
            if "checkpoint" in kwargs:
                raise PayloadTooLargeError("413 PAYLOAD_TOO_LARGE", status=413)

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
        raise LeaseLostError("w no longer holds the run", code="LEASE_LOST", status=409)

    monkeypatch.setattr(harness.runs, "heartbeat", lost)

    async def billing(input: str, agent: Runtime) -> str:
        await agent.tools.call("pay", amount=1)
        return await agent.tools.call("pay", amount=2)

    agent = harness.wrap(billing, id="billing", tools=[pay])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert paid == []  # nothing after the lost lease: the write's save before it runs failed
    assert (await handle.status()).status is RunStatus.RUNNING  # nothing written


async def reporter(input: str, agent: Runtime) -> str:
    report = await agent.tools.call("export", rows=3)
    sent = await agent.ask("Send it?", options=["yes", "no"])
    return f"{len(report)} {sent}"


async def test_a_journal_larger_than_a_checkpoint_survives_a_crash() -> None:
    crashes = [Crash()]

    async def exporting(input: str, agent: Runtime) -> int:
        report = await agent.tools.call("export", rows=3)
        if crashes:
            raise crashes.pop()
        return len(report)

    store = Bounded()
    async with Harness(config=Settings(), runs=store) as h:
        agent = h.wrap(exporting, id="exporting", tools=[export])
        handle = await agent.start("x", user="u")
        await crash_once(store, agent, handle)
        record = await handle.status()
        assert record.checkpoint is not None and set(record.checkpoint) == {JOURNAL_REF}
        assert await h.worker([agent]).run_once()
        done = await handle.result(timeout=5)
    assert done.answer == len(REPORT) and exported == [3]  # replayed, not exported again


async def test_a_journal_larger_than_a_checkpoint_survives_a_pause() -> None:
    async with Harness(config=Settings(), runs=Bounded()) as h:
        agent = h.wrap(reporter, id="reporter", tools=[export])
        handle = await agent.start("x", user="u")
        worker = h.worker([agent])
        assert await worker.run_once()
        paused = await handle.result(timeout=5)
        assert paused.interrupt is not None
        record = await handle.status()
        assert record.checkpoint is not None and set(record.checkpoint) == {JOURNAL_REF}
        await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
        assert await worker.run_once()
        done = await handle.result(timeout=5)
    assert done.answer == f"{len(REPORT)} yes" and exported == [3]

    async with Harness(config=Settings(), runs=Bounded()) as h:  # resumed in its own process
        agent = h.wrap(reporter, id="reporter", tools=[export])
        asked = await agent.run("x", user="u")
        assert asked.interrupt is not None
        done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="no", reviewer="u")
    assert done.answer == f"{len(REPORT)} no" and exported == [3, 3]


async def test_a_journal_whose_artifact_is_gone_leaves_the_run_waiting(harness: Harness) -> None:
    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(reporter, id="reporter", tools=[export])
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None
    store._artifacts.clear()
    with pytest.raises(HarnessError, match="is gone"):
        await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    record = await store.get(paused.run_id)
    assert record is not None and record.status is RunStatus.PAUSED


# --------------------------------------------------------------------------- a call in flight


async def test_a_write_running_when_its_worker_died_is_unknown_and_not_run_again(
    harness: Harness,
) -> None:
    crashes = [Crash()]
    wired: list[int] = []

    @tool(side_effects="write")
    async def wire(amount: int) -> str:
        """Wire money."""
        wired.append(amount)
        if crashes:
            raise crashes.pop()  # the worker dies while the bank works on it
        return "wired"

    async def billing(input: str, agent: Runtime) -> str:
        return await agent.tools.call("wire", amount=5)

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(billing, id="billing", tools=[wire])
    handle = await agent.start("x", user="u")
    await crash_once(store, agent, handle)
    assert await harness.worker([agent]).run_once()
    done = await handle.result(timeout=5)
    assert done.answer == (
        "wire was interrupted by a crash; it may or may not have taken effect: check before "
        "calling it again"
    )
    assert wired == [5]  # not wired twice


async def test_an_idempotent_write_runs_again_after_a_crash_with_the_same_key(
    harness: Harness,
) -> None:
    keys: list[str | None] = []
    crashes = [Crash()]

    @tool(side_effects="write", idempotent=True)
    async def upsert(row: int) -> str:
        """Store a row: the service applies a repeated key once."""
        runtime = current()
        keys.append(runtime.idempotency_key if runtime else None)
        if crashes:
            raise crashes.pop()
        return "stored"

    async def saving(input: str, agent: Runtime) -> str:
        return await agent.tools.call("upsert", row=1)

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(saving, id="saving", tools=[upsert])
    handle = await agent.start("x", user="u")
    await crash_once(store, agent, handle)
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(timeout=5)).answer == "stored"
    first, again = keys
    assert first is not None and first == again  # the service sees one request, twice


async def test_a_write_that_asks_or_fails_runs_again_on_resume(harness: Harness) -> None:
    ran: list[str] = []

    @tool(side_effects="write")
    async def post(text: str) -> str:
        """Post a message, after a person confirms the wording."""
        ran.append(text)
        runtime = current()
        assert runtime is not None
        if text == "draft":
            raise ValueError("the board is read-only")
        return f"posted {await runtime.ask(f'Post {text!r}?')}"

    async def poster(input: str, agent: Runtime) -> list[Any]:
        return [await agent.tools.call("post", text=t) for t in ("draft", "hi")]

    agent = harness.wrap(poster, id="poster", tools=[post])
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    # neither the failed call nor the one that asked reads as interrupted by a crash
    assert done.answer == ["post failed: the board is read-only", "posted yes"]
    assert ran == ["draft", "hi", "draft", "hi"]
