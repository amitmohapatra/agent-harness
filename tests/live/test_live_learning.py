"""What the memory service learns from the harness's records shows up in the next run's
context: a procedure from successful runs, tool statistics and approval patterns from the
calls and their approvals, a pinned profile block an agent edited, feedback on a run — plus
the agent's model key registered once at startup."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_memory
from tests.live.support import eventually, memory_scope
from trellis import Runtime, tool
from trellis.contracts import RunStatus

pytestmark = [pytest.mark.live, needs_memory]

TASK = "Restock SKU A-1 if it is low"
#: Approvals on one (tool, argument shape) before the service suggests a rule.
APPROVALS = 5


async def test_procedures_tool_stats_approvals_and_profile_are_learned() -> None:
    suffix = uuid.uuid4().hex[:8]
    lookup_name, reorder_name = f"lookup_{suffix}", f"reorder_{suffix}"
    user = f"live-user-{suffix}"

    @tool(name=lookup_name, side_effects="read")
    def lookup(sku: str) -> int:
        """Units of a SKU in stock."""
        return 3

    @tool(name=reorder_name, side_effects="irreversible")
    def reorder(sku: str, qty: int) -> str:
        """Order more units of a SKU."""
        return f"ordered {qty} x {sku}"

    async def restock(input: str, agent: Runtime) -> str:
        units = await agent.tools.call(lookup_name, sku="A-1")
        if units < 10:
            await agent.tools.call(reorder_name, sku="A-1", qty=20)
            return "reordered 20"
        return "enough"

    async with live_harness() as h:
        agent = h.wrap(
            restock, id=f"live-restock-{suffix}", memory="read_write", tools=[lookup, reorder]
        )
        scope = memory_scope(h, user=user, agent_id=agent.id)
        run_ids: list[str] = []
        for n in range(APPROVALS):
            paused = await agent.run(TASK, user=user, thread=f"live-restock-{suffix}-{n}")
            assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
            done = await agent.resume(
                paused.interrupt.interrupt_id, "approve", reviewer=f"live-lead-{suffix}"
            )
            assert done.status is RunStatus.SUCCESS and done.answer == "reordered 20"
            run_ids.append(done.run_id)
        await h.feedback(run_ids[0], "confirm")
        await h.writes.drain()
        assert h.writes.failed == 0

        async def learned_procedure() -> bool:
            bundle = await scope.context(TASK)
            return any(
                reorder_name in str(p.steps) and lookup_name in str(p.steps)
                for p in bundle.procedures
            )

        assert await eventually(learned_procedure, within=120, every=3)

        async def counted() -> bool:
            entries = await scope.advanced.tools.catalog(names=[lookup_name, reorder_name])
            stats = {e.name: e.stats for e in entries}
            return (
                stats.get(lookup_name) is not None
                and stats[lookup_name].calls == APPROVALS  # replays are not calls
                and stats.get(reorder_name) is not None
                and stats[reorder_name].approvals == APPROVALS
            )

        assert await eventually(counted)

        async def suggested() -> bool:
            found = await scope.advanced.tools.approval_suggestions(tool=reorder_name)
            return any(s.suggestion == "auto_approve" and s.support >= APPROVALS for s in found)

        assert await eventually(suggested)

        # feedback on a run is stored with the run, as a person's verdict
        [stored] = await scope.feedback.list_for("run", run_ids[0])
        assert stored.verdict == "confirm" and stored.source == "human"


async def test_a_profile_block_an_agent_edits_is_in_the_next_context() -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"
    preference = f"Prefers deliveries on Fridays ({suffix})."

    async def assistant(input: str, agent: Runtime) -> Any:
        if input == "remember":
            return await agent.tools.call("profile_edit", block="user", old="", new=preference)
        return agent.context

    async with live_harness() as h:
        agent = h.wrap(assistant, id=f"live-assistant-{suffix}", memory="read_write")
        assert (await agent.run("remember", user=user)).status is RunStatus.SUCCESS
        later = await agent.run("when do I like deliveries?", user=user)
        assert preference in str(later.answer)
        blocks = await memory_scope(h, user=user, agent_id=agent.id).profile()
        assert any(preference in b.text for b in blocks)


@pytest.mark.skipif(
    not os.environ.get("TRELLIS_MEMORY_MODEL_KEY"), reason="needs TRELLIS_MEMORY_MODEL_KEY"
)
async def test_the_model_key_is_registered_for_the_agent() -> None:
    suffix = uuid.uuid4().hex[:8]

    async def echo(input: str, agent: Runtime) -> str:
        return input

    async with live_harness() as h:
        agent = h.wrap(echo, id=f"live-keyed-{suffix}", memory="read")
        await agent.run("hello", user=f"live-user-{suffix}")
        await h.writes.drain()
        assert h.writes.failed == 0
        assert h.memory is not None  # the key is the agent's, read in the agent's scope
        status = await h.memory.scoped(h.settings.tenant, agent.id).ctx.advanced.model_keys.status()
    assert status.registered and not status.revoked
