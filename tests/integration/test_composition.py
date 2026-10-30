"""A Harness is the blocks it is given: a run store, a memory client, the gateway, governance —
each used as it is, the others built from the environment, ``False`` leaving one off. So
``ReAct`` (or any target) runs with the team's own blocks, and the team's own scheduler or
worker continues its runs (``agent.execute``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from tests.support.arm import arm, with_governance
from tests.support.gateway import FakeGateway
from tests.support.memory import FakeMemoryService
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.governance import Decision, Governance
from trellis.harness.runs import LocalRuns
from trellis.runs import Job, Worker

#: A deployment whose environment names every service.
EVERYTHING = Settings(
    memory_url="http://memory.env",
    runs_url="http://runs.env",
    llm_gateway="bifrost",
    llm_base_url="http://gateway.env/v1",
    api_key="key",
)


class Closing:
    """Counts the closes of the clients it watches."""

    def __init__(self) -> None:
        self.closed: list[str] = []

    def watch(self, name: str, client: Any) -> Any:
        closing = client.aclose

        async def aclose() -> None:
            self.closed.append(name)
            await closing()

        client.aclose = aclose
        return client


async def test_the_blocks_given_are_used_as_they_are_and_left_to_their_owner(
    memory_service: FakeMemoryService,
) -> None:
    closing = Closing()
    store = LocalRuns()
    memory = closing.watch("memory", memory_service.client())
    gateway = closing.watch("gateway", FakeGateway().gateway())
    governance = Governance()
    async with with_governance(
        Harness(config=EVERYTHING, runs=store, memory=memory, gateway=gateway), governance
    ) as h:
        assert h.runs is store and h.gateway is gateway
        assert h.memory is not None and h.memory.client is memory
        assert h._governance("acme") is governance is h._governance("default")  # every tenant
    assert closing.closed == []  # the blocks given are their owner's to close


async def test_the_blocks_not_given_come_from_the_environment_and_are_closed() -> None:
    closing = Closing()
    h = Harness(config=EVERYTHING)
    assert h.memory is not None and h.gateway is not None
    assert type(h.runs).__name__ == "RunsClient"
    closing.watch("memory", h.memory.client)
    closing.watch("gateway", h.gateway)
    closing.watch("runs", h.runs)
    await h.aclose()
    assert sorted(closing.closed) == ["gateway", "memory", "runs"]
    assert h._governance("acme") is h._governance("acme") is not h._governance("other")


async def test_false_leaves_a_block_off_whatever_the_environment_names() -> None:
    async with Harness(config=EVERYTHING, runs=False, memory=False, gateway=False) as h:
        assert h.memory is None and h.gateway is None and isinstance(h.runs, LocalRuns)
        assert await h._tenant() == "default"  # no memory service to ask who the key is
    with pytest.raises(ConfigurationError, match="RUNS_URL needs MEMORY_URL"):
        Harness(config=EVERYTHING, memory=False)  # agent-runs from the environment, no memory
    keyless = EVERYTHING.model_copy(update={"api_key": None})
    with pytest.raises(ConfigurationError, match="need TRELLIS_API_KEY"):
        Harness(config=keyless, runs=False)  # the memory service from the environment
    async with Harness(config=keyless, runs=LocalRuns(), memory=False) as h:  # nothing asks
        assert h.memory is None


# --------------------------------------------------------------------------- ReAct with blocks


class Watched(Governance):
    """The team's governance: every decision it makes, kept."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Decision] = []

    async def check(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        side_effects: str = "write",
        agent_id: str | None = None,
    ) -> Decision:
        decision = await super().check(tool, args, side_effects=side_effects, agent_id=agent_id)
        self.seen.append(decision)
        return decision


async def test_react_runs_with_the_teams_blocks_and_its_own_scheduler() -> None:
    """``Harness(<blocks>).wrap(ReAct(...))``: the team's run store, its governance, no memory;
    its own loop claims the queued run and hands it to ``agent.execute``, and after the
    approval its own ``trellis.runs.Worker`` continues it."""
    refunded: list[str] = []

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        refunded.append(order)
        return f"refunded {order}"

    store, governance = LocalRuns(), Watched()
    async with with_governance(
        Harness(config=Settings(), runs=store, memory=False), governance
    ) as h:
        model = PlannedChat([("refund", {"order": "o1"})])
        agent = arm(h.wrap(ReAct(system_prompt="You refund.", model=model), id="refunds"), [refund])
        handle = await agent.start("refund o1", user="ada")

        claimed = await store.claim("my-loop", [agent.id], lease_seconds=30)  # the team's loop
        assert claimed is not None
        job = Job(record=claimed.run, worker_id="my-loop", lease_seconds=30, store=store)
        paused = await agent.execute(job)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert [d.action.value for d in governance.seen] == ["ask"] and refunded == []

        queued = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
        assert queued.status is RunStatus.QUEUED  # back to the team's queue
        assert await Worker(store, agent.execute, [agent.id]).run_once()
        done = await handle.result(wait=5)
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. refunded o1"
    assert refunded == ["o1"]  # once, after the approval
