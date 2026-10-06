"""Composition and selection against the running services: a Harness given its own run store
(agent-runs' client the test built) and its own governance, whose queued run the team's own
scheduler loop claims from agent-runs and continues with ``agent.execute`` — paused for an
approval, queued again, finished; and a run ``without={"memory"}`` that leaves no record in
the memory service, beside one that does."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import pytest

from tests.live.conftest import LiveHarness, needs_memory, needs_runs, settings
from tests.live.support import eventually, memory_scope
from trellis import Agent, Harness, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness.governance import Decision, Governance
from trellis.memory import MemoryClient
from trellis.memory.errors import NotFoundError
from trellis.runs import Job, RunsClient

pytestmark = [pytest.mark.live, needs_runs, needs_memory]

LEASE_SECONDS = 30


class Watched(Governance):
    """The team's governance (the tools' own risks): every decision kept."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Decision] = []

    async def check(
        self, tool: str, args: Mapping[str, Any], *, side_effects: str = "write"
    ) -> Decision:
        decision = await super().check(tool, args, side_effects=side_effects)
        self.seen.append(decision)
        return decision


async def scheduled(runs: RunsClient, agent: Agent, worker_id: str) -> Any:
    """The team's own scheduler: one claim from agent-runs, its run continued by the agent."""
    claimed = await runs.claim(worker_id, [agent.id], lease_seconds=LEASE_SECONDS)
    if claimed is None:
        return None
    job = Job(record=claimed.run, worker_id=worker_id, lease_seconds=LEASE_SECONDS, store=runs)
    return await agent.execute(job)


@pytest.mark.timeout(120)
async def test_own_run_store_and_scheduler_loop_against_agent_runs() -> None:
    suffix = uuid.uuid4().hex[:8]
    refunded: list[str] = []

    @tool(name=f"refund_{suffix}", side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        refunded.append(order)
        return f"refunded {order}"

    async def refunds(input: str, agent: Runtime) -> Any:
        return await agent.tools.call(refund.spec.name, order=input)

    s = settings()
    runs = RunsClient(s.runs_url, api_key=s.api_key)
    governance = Watched()
    memory = MemoryClient(s.memory_url, api_key=s.api_key)
    try:
        async with Harness(config=s, runs=runs, memory=memory, governance=governance) as h:
            assert h.runs is runs
            agent = h.wrap(refunds, id=f"live-blocks-{suffix}", tools=[refund], timeout=120)
            handle = await agent.start("o1", user=f"live-user-{suffix}")
            worker = f"my-scheduler-{suffix}"

            async def claimed() -> bool:
                paused = await scheduled(runs, agent, worker)
                return paused is not None

            assert await eventually(claimed, within=30)
            paused = await handle.result(timeout=30)
            assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
            assert [d.action.value for d in governance.seen] == ["ask"] and refunded == []
            queued = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
            assert queued.status is RunStatus.QUEUED  # back to agent-runs' queue
            assert await eventually(claimed, within=30)
            done = await handle.result(timeout=30)
            record = await runs.get(handle.run_id, tenant=await h.tenant())
    finally:
        await runs.aclose()
        await memory.aclose()
    assert done.status is RunStatus.SUCCESS and done.answer == "refunded o1"
    assert refunded == ["o1"]  # once, after the approval
    assert record is not None and record.timeout_seconds == 120  # the agent's limit, kept


@pytest.mark.timeout(120)
async def test_a_run_without_memory_leaves_no_record_in_memory(harness: LiveHarness) -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"

    @tool(name=f"stock_{suffix}", side_effects="read")
    def stock(sku: str) -> int:
        """Units in stock."""
        return 42

    async def answer(input: str, agent: Runtime) -> str:
        return f"{await agent.tools.call(stock.spec.name, sku='A-1')} units"

    agent = harness.wrap(answer, id=f"live-select-{suffix}", tools=[stock])
    scope = await memory_scope(harness, user=user, agent_id=agent.id, thread=f"off-{suffix}")
    off = await agent.run("stock?", user=user, thread=f"off-{suffix}", without={"memory"})
    on = await agent.run("stock?", user=user, thread=f"on-{suffix}")
    await harness.writes.drain()
    assert off.answer == on.answer == "42 units"
    with pytest.raises(NotFoundError, match="Thread not found"):
        await scope.history()  # nothing recorded for the run without memory: not even its thread
    remembered = await memory_scope(harness, user=user, agent_id=agent.id, thread=f"on-{suffix}")

    async def recorded() -> bool:
        return [m.role for m in await remembered.history()] == ["USER", "ASSISTANT"]

    assert await eventually(recorded)

    async def counted() -> bool:  # the one call with memory on, not two
        [entry] = await remembered.advanced.tools.catalog(names=[stock.spec.name])
        return entry.stats.calls == 1

    assert await eventually(counted)
