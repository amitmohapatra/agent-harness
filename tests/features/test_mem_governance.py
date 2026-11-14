"""MEM-09 and the catalog rules, against the real memory service: when a call needs a person
is a rule in the memory service's tool catalog, and what people decide is learned there.

* "always allow": a person's ``always`` for a write is a tool permission in the memory
  service, honoured in every later conversation of that user and by no other user, never
  offered for an irreversible tool;
* learned rules start from asking: a tool with no rule asks; once two reviewers approved five
  calls of one argument shape the service suggests auto-approving that shape; accepted (by the
  agent it was learned from), that shape runs unasked and any other shape still asks;
* an administrator's catalog entry decides: a ``risk`` overrides the tool's own declaration,
  and ``approve_when`` asks exactly when it holds — read by an agent already running within
  the catalog's refresh.

The tools are named per test: the catalog is the tenant's, shared by every test."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.features.conftest import ADAPTERS, eventually, harness, memory_sql
from tests.features.kit import armed, book, ledger, scope_of
from trellis import tool
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(400)


def write_tool(name: str) -> Any:
    def order(sku: str, qty: int) -> str:
        """Order more of a SKU."""
        book(f"{name}:{sku}:{qty}")
        return f"ordered {qty} of {sku}"

    return tool(order, name=name, side_effects="write")


def options(result: Any) -> list[str]:
    return [o.value for o in (result.interrupt.options or [])]


@pytest.mark.plan("MEM-09", "APR-03")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_always_allow_is_kept_for_the_user_in_every_conversation(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    name, user = f"order_{uid}", f"ola-{uid}"
    plan = [(name, {"sku": "A-1", "qty": 2})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"always-{adapter}-{uid}",
                               tools=[write_tool(name)])
        first = await agent.run("Order two A-1.", user=user, thread="c1")
        assert first.interrupt is not None, first
        assert "always" in options(first), options(first)  # offered: memory keeps it
        done = await agent.resume(
            first.interrupt.interrupt_id, "approve", reviewer=user, remember="always"
        )
        assert done.status is RunStatus.SUCCESS, done
        await h.writes.drain()

    async def kept() -> list[dict]:
        return memory_sql(
            "select user_id, tool from tool_permissions where user_id = :1 and tool = :2",
            user,
            name,
        )

    assert len(await eventually(kept, within=60)) == 1  # one permission, in memory
    async with harness() as h:  # another process, another conversation of the same user
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"always-{adapter}-{uid}",
                               tools=[write_tool(name)])
        again = await agent.run("Order two A-1 again.", user=user, thread="c2")
        assert again.status is RunStatus.SUCCESS and again.interrupt is None, again
        stranger = await agent.run("Order two A-1.", user=f"pat-{uid}", thread="c1")
        assert stranger.interrupt is not None, stranger  # nobody else's
        await agent.resume(stranger.interrupt.interrupt_id, "reject", reviewer="pat")
    assert ledger().count(f"{name}:A-1:2") == 2


@pytest.mark.plan("MEM-09")
async def test_always_is_never_offered_for_an_irreversible_tool(tmp_path: Path, uid: str) -> None:
    async with harness() as h:
        plan = [("wire", {"amount": 1, "to": "x"})]
        agent, _ = await armed("react", h, tmp_path, plan, id=f"irrev-{uid}")
        first = await agent.run("Pay x.", user=f"quin-{uid}", thread="c1")
        assert first.interrupt is not None
        assert "always" not in options(first) and "run" not in options(first), options(first)
        await agent.resume(first.interrupt.interrupt_id, "reject", reviewer="quin")


@pytest.mark.plan("MEM-09", "APR-03")
async def test_a_learned_rule_starts_from_asking_and_is_applied_once_accepted(
    tmp_path: Path, uid: str
) -> None:
    name, agent_id = f"stock_{uid}", f"learner-{uid}"
    small = [(name, {"sku": "A-1", "qty": 3})]
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, small, id=agent_id, tools=[write_tool(name)])
        for n in range(6):  # six conversations, each asked once, approved by two people
            paused = await agent.run(f"Order three A-1 ({n}).", user=f"u{n}-{uid}", thread="c")
            assert paused.interrupt is not None, (n, paused)  # no rule: it asks
            done = await agent.resume(
                paused.interrupt.interrupt_id, "approve", reviewer=("rae", "sol")[n % 2]
            )
            assert done.status is RunStatus.SUCCESS, done
        await h.writes.drain()
        ctx = await scope_of(h, "u0-" + uid, agent=agent_id)

        async def suggested() -> list[Any]:
            got = await ctx.advanced.tools.approval_suggestions(tool=name)
            return [s for s in got if s.suggestion == "auto_approve"]

        suggestions = await eventually(suggested, within=200, every=5)
        assert suggestions, await ctx.advanced.tools.approval_suggestions(tool=name)
        entry = await ctx.advanced.tools.accept_suggestion(suggestions[0].id)
        assert entry.approve_when and "shape" in entry.approve_when, entry
    async with harness() as h:  # a new process reads the learned rule
        plan = [(name, {"sku": "A-1", "qty": 4}), (name, {"sku": "A-1", "qty": 40_000})]
        agent, _ = await armed("react", h, tmp_path, plan, id=agent_id, tools=[write_tool(name)])
        result = await agent.run("Order four, then forty thousand.", user=f"v-{uid}", thread="c")
        assert result.interrupt is not None, result  # the other shape still asks
        assert result.interrupt.tool_call is not None
        assert result.interrupt.tool_call.args["qty"] == 40_000
        assert f"{name}:A-1:4" in ledger()  # the learned shape ran unasked
        await agent.resume(result.interrupt.interrupt_id, "reject", reviewer="rae")


@pytest.mark.plan("MEM-09", "HAR-GOV")
async def test_the_catalogs_risk_and_rule_decide_and_reach_a_running_agent(
    tmp_path: Path, uid: str
) -> None:
    """An administrator's ``risk`` makes a declared read ask; an ``approve_when`` asks exactly
    when it holds; a rule changed while the agent runs is read within the refresh (30 s)."""
    read_name, write_name = f"peek_{uid}", f"buy_{uid}"

    def peek(sku: str) -> str:
        """Peek at a SKU."""
        book(f"peek:{sku}")
        return f"peeked {sku}"

    peek_tool = tool(peek, name=read_name, side_effects="read")
    buy = write_tool(write_name)
    async with harness() as h:
        admin = await scope_of(h, f"adm-{uid}", agent=f"cat-{uid}")
        await admin.advanced.tools.put_catalog(
            [
                {"name": read_name, "side_effects": "irreversible"},
                {"name": write_name, "approve_when": "qty > 100"},
            ]
        )
        plan = [(read_name, {"sku": "A-1"}), (write_name, {"sku": "A-1", "qty": 5}),
                (write_name, {"sku": "A-1", "qty": 500})]
        agent, _ = await armed("react", h, tmp_path, plan, id=f"cat-{uid}",
                               tools=[peek_tool, buy])
        first = await agent.run("Peek, buy 5, buy 500.", user=f"wu-{uid}", thread="c")
        assert first.interrupt is not None and first.interrupt.tool_call.tool == read_name
        second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="wu")
        assert second.interrupt is not None, second  # 5 ran unasked: the rule did not hold
        assert second.interrupt.tool_call.args["qty"] == 500
        assert "qty > 100" in second.interrupt.question
        done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="wu")
        assert done.status is RunStatus.SUCCESS
        assert ledger()[-3:] == ["peek:A-1", f"{write_name}:A-1:5", f"{write_name}:A-1:500"]
        # the rule tightened while this agent lives: within the refresh, 5 asks too
        await admin.advanced.tools.put_catalog([{"name": write_name, "approve_when": "qty > 1"}])
        await asyncio.sleep(31)
        agent2, _ = await armed("react", h, tmp_path, [(write_name, {"sku": "A-1", "qty": 5})],
                                id=f"cat2-{uid}", tools=[peek_tool, buy])
        later = await agent2.run("Buy 5.", user=f"wu-{uid}", thread="c9")
        assert later.interrupt is not None, later
        assert "qty > 1" in later.interrupt.question
        await agent2.resume(later.interrupt.interrupt_id, "reject", reviewer="wu")
