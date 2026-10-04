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


# longer than the suite's 120 s: the learning job is waited for up to 120 s after the runs it learns from
@pytest.mark.timeout(300)
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
        agent = h.wrap(restock, id=f"live-restock-{suffix}", tools=[lookup, reorder])
        scope = await memory_scope(h, user=user, agent_id=agent.id)
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
            """The procedure the successful runs taught it, in the next run's own context."""
            pushed = await scope.context(TASK, tools=[lookup_name, reorder_name])
            return reorder_name in pushed.rendered and lookup_name in pushed.rendered

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

        # however often a reorder was approved, an irreversible tool is never offered
        # "approve automatically": the next order is still asked about
        found = await scope.advanced.tools.approval_suggestions(tool=reorder_name)
        assert not any(s.suggestion == "auto_approve" for s in found)

        # feedback on a run is stored with the run as a person's verdict, waiting for the
        # tenant administrator (the memory service's ADR 0028) beside the run's own
        # ``system`` outcome, which was applied as it arrived
        stored = await scope.feedback.list_for("run", run_ids[0])
        [human] = [f for f in stored if f.source == "human"]
        assert human.verdict == "confirm" and human.review is not None
        assert human.review.state == "pending"
        assert all(f.review is None for f in stored if f.source == "system")


async def test_a_profile_block_an_agent_edits_is_in_the_next_context() -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"
    preference = f"Prefers deliveries on Fridays ({suffix})."

    async def assistant(input: str, agent: Runtime) -> Any:
        if input == "remember":
            return await agent.tools.call("profile_edit", block="user", old="", new=preference)
        return agent.context

    async with live_harness() as h:
        agent = h.wrap(assistant, id=f"live-assistant-{suffix}")
        assert (await agent.run("remember", user=user)).status is RunStatus.SUCCESS
        later = await agent.run("when do I like deliveries?", user=user)
        assert preference in str(later.answer)
        blocks = await (await memory_scope(h, user=user, agent_id=agent.id)).profile()
        assert any(preference in b.text for b in blocks)


@pytest.mark.skipif(not os.environ.get("BIFROST_VIRTUAL_KEY"), reason="needs BIFROST_VIRTUAL_KEY")
async def test_the_virtual_key_is_registered_as_the_agents_model_key() -> None:
    suffix = uuid.uuid4().hex[:8]

    async def echo(input: str, agent: Runtime) -> str:
        return input

    async with live_harness() as h:
        agent = h.wrap(echo, id=f"live-keyed-{suffix}")
        await agent.run("hello", user=f"live-user-{suffix}")
        await h.writes.drain()
        assert h.writes.failed == 0
        assert h.memory is not None  # the key is the agent's, read in the agent's scope
        scope = h.memory.scoped(await h.tenant(), agent.id)
        status = await scope.ctx.advanced.model_keys.status()
    assert status.registered and not status.revoked


async def test_a_document_added_for_a_user_is_in_their_next_context() -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"
    policy = f"Returns are accepted within 45 days of delivery (policy {suffix})."

    async def assistant(input: str, agent: Runtime) -> Any:
        return agent.context

    async with live_harness() as h:
        info = await h.add_document(
            ("returns.txt", policy.encode(), "text/plain"), user=user, title="Returns"
        )
        assert info.status == "READY"
        agent = h.wrap(assistant, id=f"live-docs-{suffix}")
        later = await agent.run("How many days do I have to return an order?", user=user)
        assert "45 days" in str(later.answer)
