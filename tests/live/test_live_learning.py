"""What the memory service learns from the harness's records shows up in the next run's
context: a procedure from runs a person confirmed (a run's own report of success teaches
nothing; a person's verdict applies once the tenant's administrator accepts it), tool
statistics and approval patterns from the calls and their approvals (a rule to approve
automatically needs two reviewers), a pinned profile block an agent edited, feedback on a run
— plus the agent's model key registered once at startup."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_memory
from tests.live.support import eventually, memory_scope
from tests.support.arm import arm
from trellis import Runtime, tool
from trellis.contracts import RunStatus
from trellis.memory import MemoryClient

#: about what a write does once it runs, not whether it asks: ``writes_unasked``
pytestmark = [pytest.mark.live, needs_memory, pytest.mark.usefixtures("writes_unasked")]

TASK = "Restock SKU A-1 if it is low"
#: Approvals on one (tool, argument shape) before the service suggests a rule.
APPROVALS = 5
#: The tenant administrator's memory key: accepts a person's verdict on a run.
ADMIN_KEY = os.environ.get("TRELLIS_ADMIN_KEY", "").strip() or None


# longer than the suite's 120 s: the learning job is waited for up to 120 s after the runs it learns from
@pytest.mark.timeout(300)
@pytest.mark.skipif(
    ADMIN_KEY is None, reason="TRELLIS_ADMIN_KEY (the tenant administrator's memory key) is not set"
)
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
        agent = arm(h.wrap(restock, id=f"live-restock-{suffix}"), [lookup, reorder])
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
        # a person confirms three runs; the administrator accepts two: the procedure is
        # learned from those (two runs that followed it), the third verdict still waits
        await h.feedback(run_ids[0], "confirm")
        async with MemoryClient(os.environ["MEMORY_URL"], api_key=ADMIN_KEY) as client:
            admin = client.bind(tenant_id=await h._tenant())
            for run_id in run_ids[1:3]:
                said = await h.feedback(run_id, "confirm")
                assert said is not None
                await admin.feedback.approve(said.feedback_id)
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
        stored = (await scope.feedback.page_for("run", run_ids[0])).items
        [human] = [f for f in stored if f.source == "human"]
        assert human.verdict == "confirm" and human.review is not None
        assert human.review.state == "pending"
        assert all(f.review is None for f in stored if f.source == "system")


# longer than the suite's 120 s: the decisions are learned from in the background
@pytest.mark.timeout(300)
async def test_a_learned_approval_rule_is_applied_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write the harness asks about (no rule says otherwise), approved by people every time —
    two of them (one person's approvals, however many, are no team's rule): the memory service
    learns from the agent's decisions to stop asking, and governance accepts that suggestion
    for the agent on its next refresh — the next call runs with no pause. The same decisions
    on a tool its administrator set to always ask learn no such rule: what an administrator
    decided is never relaxed by learning (the memory service asks more, never less, of it)."""
    from trellis.harness.governance import catalog as catalog_module
    from trellis.harness.governance import decision

    # the product's default here, not the module's ``writes_unasked``: a write asks
    monkeypatch.setitem(decision.ACTIONS, "write", decision.Action.ASK)
    suffix = uuid.uuid4().hex[:8]
    order_name, ruled_name = f"order_{suffix}", f"ruled_order_{suffix}"
    user = f"live-user-{suffix}"

    @tool(name=order_name, side_effects="write")
    def order(sku: str, qty: int) -> str:
        """Order units of a SKU."""
        return f"ordered {qty} x {sku}"

    @tool(name=ruled_name, side_effects="write")
    def ruled_order(sku: str, qty: int) -> str:
        """Order units of a SKU, as the administrator's rule says: asked every time."""
        return f"ordered {qty} x {sku} (ruled)"

    def buyer(name: str) -> Any:
        async def buy(input: str, agent: Runtime) -> str:
            return str(await agent.tools.call(name, sku="A-1", qty=2))

        return buy

    async with live_harness() as h:
        agent = arm(h.wrap(buyer(order_name), id=f"live-buyer-{suffix}"), [order])
        ruled = arm(h.wrap(buyer(ruled_name), id=f"live-ruled-{suffix}"), [ruled_order])
        scope = await memory_scope(h, user=user, agent_id=agent.id)
        ruled_scope = await memory_scope(h, user=user, agent_id=ruled.id)
        await ruled_scope.advanced.tools.put_catalog(
            [{"name": ruled_name, "side_effects": "write", "approve_when": "true"}]
        )
        for each in (agent, ruled):
            for n in range(APPROVALS):
                paused = await each.run("order two", user=user, thread=f"{each.id}-{n}")
                assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
                reviewer = f"live-lead-{suffix}" if n % 2 == 0 else f"live-buyer-lead-{suffix}"
                done = await each.resume(
                    paused.interrupt.interrupt_id, "approve", reviewer=reviewer
                )
                assert done.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert h.writes.failed == 0

        async def suggested() -> bool:
            found = await scope.advanced.tools.approval_suggestions(tool=order_name)
            return any(s.suggestion == "auto_approve" for s in found)

        assert await eventually(suggested, within=120, every=3)
        # the administrator's rule stays: the same decisions on its tool suggest nothing
        # that would ask less
        found = await ruled_scope.advanced.tools.approval_suggestions(tool=ruled_name)
        assert not any(s.suggestion == "auto_approve" for s in found), found
        # governance's next refresh (now, not in 30 s) accepts the suggestion for this agent
        later = catalog_module._now() + catalog_module.GOVERNANCE_TTL_SECONDS + 1
        monkeypatch.setattr(catalog_module, "_now", lambda: later)
        ran = await agent.run("order two", user=user, thread=f"buy-{suffix}-last")
        assert ran.status is RunStatus.SUCCESS and ran.answer == "ordered 2 x A-1"
        [entry] = await scope.advanced.tools.catalog(names=[order_name])
        assert entry.approve_when and entry.approve_when != "true"  # the learned rule
        found = await scope.advanced.tools.approval_suggestions(tool=order_name)
        assert any(s.accepted for s in found)
        still = await ruled.run("order two", user=user, thread=f"ruled-{suffix}-last")
        assert still.status is RunStatus.PAUSED, "the administrator's rule still asks"


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


@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY")
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
        scope = h.memory.scoped(await h._tenant(), agent.id)
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
