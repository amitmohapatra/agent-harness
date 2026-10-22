"""Learned approval rules, end to end against a running memory service (its ADR 0042): a call
people keep approving stops pausing once the service accepts the rule on its own; an
irreversible tool keeps pausing whatever the decisions; revoking the rule through the catalog
brings the pause back. No model: a function agent calls the tool."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from tests.live.conftest import needs_memory
from trellis import Harness, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness.governance import catalog

pytestmark = [pytest.mark.live, needs_memory]

#: The decisions on one argument shape after which the service accepts the rule.
SUPPORT = 5


@pytest.fixture(autouse=True)
def catalog_read_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Governance re-reads the catalog every call, not every 30 s: the test does not wait."""
    monkeypatch.setattr(catalog, "GOVERNANCE_TTL_SECONDS", 0.0)


async def decided(harness: Harness, agent: Any, user: str, n: int) -> list[bool]:
    """``n`` runs, each approved when it pauses: whether each paused."""
    paused: list[bool] = []
    for i in range(n):
        result = await agent.run(f"ship order o-{i}", user=user)
        paused.append(result.status is RunStatus.PAUSED)
        if result.status is RunStatus.PAUSED:
            done = await agent.resume(result.run_id, "approve", reviewer="ops")
            assert done.status is RunStatus.SUCCESS
        await harness.writes.drain()  # the decision is fed back
    return paused


@pytest.mark.parametrize("risk", ["write", "irreversible"])
async def test_approving_the_same_call_stops_its_pause_unless_the_tool_is_irreversible(
    harness: Harness, risk: str
) -> None:
    marker = uuid.uuid4().hex[:8]
    name = f"ship_{marker}"

    @tool(name=name, side_effects=risk)  # type: ignore[arg-type]
    def ship(order: str) -> str:
        """Ship an order."""
        return f"shipped {order}"

    async def shipper(input: str, agent: Runtime) -> Any:
        return await agent.tools.call(name, order=input.rsplit(maxsplit=1)[-1])

    user = f"live-{marker}"
    agent = harness.wrap(shipper, id=f"shipper-{marker}", tools=[ship])
    tenant = await harness.tenant()
    assert harness.memory is not None
    scope = harness.memory.scoped(tenant)
    await scope.ctx.advanced.tools.put_catalog([{"name": name, "approve_when": "true"}])

    assert await decided(harness, agent, user, SUPPORT) == [True] * SUPPORT
    [after] = await decided(harness, agent, user, 1)
    rules = await scope.ctx.advanced.tools.auto_accepted_rules()
    mine = [r for r in rules if r.tool == name]
    if risk == "irreversible":
        assert after is True, "the strictest tier is never relaxed on the service's word"
        assert mine == []
        return
    assert after is False, "the rule the decisions support was accepted: no pause"
    assert [(r.suggestion, r.in_effect) for r in mine] == [("auto_approve", True)]

    # a person revokes it through the catalog: the pause is back, and stays back
    await scope.ctx.advanced.tools.put_catalog([{"name": name, "approve_when": "true"}])
    assert await decided(harness, agent, user, 2) == [True, True]
    [revoked] = [r for r in await scope.ctx.advanced.tools.auto_accepted_rules() if r.tool == name]
    assert revoked.in_effect is False
