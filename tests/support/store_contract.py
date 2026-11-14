"""What every run store does the same way — ``LocalRuns``, ``FileRuns`` and agent-runs behind
``RunsClient`` — checked as behaviour, through ``RunStore``: the place a run of a
``concurrency_key`` takes (``placed_at``, on its record and its summary), the listing of exactly
one key's runs in one tenant, a claim that respects the places, a cancel of a run its process
holds, and what a hold gives away. One suite, run against each store (``tests/integration/
test_store_contract.py`` in process; ``tests/live/test_live_chat_order.py`` on agent-runs)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest

from trellis.contracts import Interrupt, InterruptDecision, InterruptResolution, RunStart, RunStatus
from trellis.harness.runs import RunStore, place
from trellis.runs import AuthorizationError, LeaseLostError

TENANT: Final = "default"
#: Another tenant.
B: Final = "other-tenant"


def starting(agent: str, key: str) -> RunStart:
    """A run of ``agent`` with the ``concurrency_key`` ``key``."""
    return RunStart(tenant_id=TENANT, agent_id=agent, user_id="ada", concurrency_key=key)


async def runs_of_a_key_are_placed_in_the_order_they_took_their_place(runs: RunStore) -> None:
    """A run started after another of its key is placed after it; one continued after its
    answer takes its place again, after the runs that arrived while it waited."""
    key = f"key-{uuid.uuid4().hex}"
    first = await runs.start(starting("contract", key))
    second = await runs.start(starting("contract", key))
    assert first.placed_at == first.created_at  # placed when it was started
    assert place(first) < place(second)
    asked = Interrupt(tenant_id=TENANT, run_id=first.run_id, question="Go on?")
    await runs.pause(asked)
    answer = InterruptResolution(
        interrupt_id=asked.interrupt_id, run_id=first.run_id, decision=InterruptDecision.APPROVE
    )
    continued = await runs.resume(answer, tenant=TENANT)
    assert continued.created_at == first.created_at  # written once
    [entry] = (await runs.resolutions(first.run_id, tenant=TENANT)).items
    assert continued.placed_at == entry.recorded_at  # placed again when it was answered
    assert place(continued) > place(second)
    read = await runs.get(first.run_id, tenant=TENANT)
    assert read is not None and read.placed_at == continued.placed_at
    keyless = await runs.start(RunStart(tenant_id=TENANT, agent_id="contract", user_id="ada"))
    assert keyless.placed_at is None
    await runs.finish(keyless.run_id, RunStatus.SUCCESS, tenant=TENANT)
    for run in (first, second):
        await runs.finish(run.run_id, RunStatus.SUCCESS, tenant=TENANT)


async def a_queued_run_is_claimed_once_no_run_placed_before_it_runs(runs: RunStore) -> None:
    """A queued run of a key waits for the run of its key placed before it, kept in its
    caller's process; one placed after it does not hold it back (it waits for the queued run
    in its own process)."""
    agent, key = f"contract-{uuid.uuid4().hex}", f"key-{uuid.uuid4().hex}"
    before = await runs.start(starting(agent, key))
    queued = await runs.start(starting(agent, key), queue=True)
    after = await runs.start(starting(agent, key))
    assert await runs.claim("worker-1", [agent], tenant=TENANT) is None
    await runs.finish(before.run_id, RunStatus.SUCCESS, tenant=TENANT)
    claimed = await runs.claim("worker-1", [agent], tenant=TENANT)
    assert claimed is not None and claimed.run.run_id == queued.run_id
    await runs.finish(queued.run_id, RunStatus.SUCCESS, worker_id="worker-1", tenant=TENANT)
    await runs.finish(after.run_id, RunStatus.SUCCESS, tenant=TENANT)


async def a_run_its_process_holds_is_cancelled_at_once(runs: RunStore) -> None:
    """A cancel ends a run kept in its caller's process at once, held or not; its holder's
    next heartbeat is refused (the holder stops working it then)."""
    held = await runs.start(RunStart(tenant_id=TENANT, agent_id="contract", user_id="ada"))
    hold = await runs.heartbeat(held.run_id, "server-1", lease_seconds=30, tenant=TENANT)
    assert not hold.cancel_requested
    cancelled = await runs.cancel(held.run_id, reason="enough", tenant=TENANT)
    assert cancelled.status is RunStatus.CANCELLED
    with pytest.raises(LeaseLostError):
        await runs.heartbeat(held.run_id, "server-1", lease_seconds=30, tenant=TENANT)


async def a_hold_is_short_and_a_refused_one_names_nobody(runs: RunStore) -> None:
    """A hold lasts at most 300 s whatever was asked (its lapse is how a dead process's run
    ends), and the refusal another holder gets does not say who holds the run."""
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id="contract", user_id="ada"))
    hold = await runs.heartbeat(run.run_id, "server-1@secret", lease_seconds=3600, tenant=TENANT)
    assert hold.expires_at - datetime.now(UTC) <= timedelta(seconds=300)
    with pytest.raises(LeaseLostError) as refused:
        await runs.heartbeat(run.run_id, "server-2", lease_seconds=30, tenant=TENANT)
    assert "server-1@secret" not in str(refused.value)
    await runs.finish(run.run_id, RunStatus.SUCCESS, tenant=TENANT)


async def a_keys_line_is_listed_by_its_key_in_its_tenant_with_each_place(runs: RunStore) -> None:
    """Listing by ``concurrency_key`` gives exactly that key's runs of the tenant asked for,
    each summary with the record's ``placed_at``: a run of another key on the same thread
    name, and another tenant's run of the same key, are not listed."""
    key, thread = f"key-{uuid.uuid4().hex}", f"thread-{uuid.uuid4().hex}"
    mine = await runs.start(starting("contract", key).model_copy(update={"thread_id": thread}))
    other_key = starting("contract", f"{key}-other").model_copy(update={"thread_id": thread})
    other = await runs.start(other_key)
    listed = [
        summary
        async for summary in runs.iterate(
            concurrency_key=key, status=RunStatus.RUNNING, tenant=TENANT
        )
    ]
    assert [(s.run_id, s.placed_at) for s in listed] == [(mine.run_id, mine.placed_at)]
    try:  # another tenant's run of the key: a key of this tenant may not start one there
        theirs = await runs.start(starting("contract", key).model_copy(update={"tenant_id": B}))
    except AuthorizationError:
        theirs = None
    if theirs is not None:
        ours = [s.run_id async for s in runs.iterate(concurrency_key=key, tenant=TENANT)]
        assert ours == [mine.run_id]
        assert [s.run_id async for s in runs.iterate(concurrency_key=key, tenant=B)] == [
            theirs.run_id
        ]
        await runs.finish(theirs.run_id, RunStatus.SUCCESS, tenant=B)
    for run in (mine, other):
        await runs.finish(run.run_id, RunStatus.SUCCESS, tenant=TENANT)


#: The contract, each check by what it says.
CHECKS: Final[tuple[Callable[[RunStore], Awaitable[None]], ...]] = (
    runs_of_a_key_are_placed_in_the_order_they_took_their_place,
    a_keys_line_is_listed_by_its_key_in_its_tenant_with_each_place,
    a_queued_run_is_claimed_once_no_run_placed_before_it_runs,
    a_run_its_process_holds_is_cancelled_at_once,
    a_hold_is_short_and_a_refused_one_names_nobody,
)
