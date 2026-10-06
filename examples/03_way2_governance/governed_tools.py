"""Way 2, governance alone: the run, announce or ask decision on your own tools — no harness, any
framework. The same ``Governance`` a wrapped agent uses decides every call.

* ``check(tool, args, side_effects=)`` — the decision for one call: ``read`` runs, ``write`` is
  announced, ``irreversible`` asks; an administrator's rule in the tool catalog
  (``approve_when``, here ``amount > 100``) replaces that, by the call's arguments;
* ``governed(fn, gov, on_ask=..., hooks=[...])`` — your function, every call checked first: a
  call that asks goes to ``on_ask`` (your approval UI; ``True`` runs it, ``False`` rejects it, a
  dict runs it with edited arguments), a hook's ``Deny`` refuses it;
* ``decided(...)`` — the reviewer's decision, which the memory service learns rules from.

Offline the catalog is the scripted memory service's; with ``MEMORY_URL`` and
``TRELLIS_API_KEY`` set, ``Governance.from_env()`` reads the real one.

    python -m examples.03_way2_governance.governed_tools
"""

from __future__ import annotations

import asyncio
import os

from examples._support.memory import ScriptedMemory

from trellis import Deny, Hooks
from trellis.contracts import ToolCall
from trellis.harness.governance import Decision, Denied, Governance, Rejected, governed
from trellis.harness.governance.catalog import MemoryCatalog

TENANT = "default"


def governance() -> Governance:
    """The catalog's rules: the real service's with ``MEMORY_URL``, else the scripted one's."""
    if os.environ.get("MEMORY_URL"):
        return Governance.from_env(agent_id="billing", tenant=TENANT)
    catalog = {"refund": {"risk": "write", "approve_when": "amount > 100"}}
    client = ScriptedMemory(catalog=catalog).client()
    return Governance(
        MemoryCatalog(client.bind(tenant_id=TENANT)), tenant=TENANT, agent_id="billing"
    )


async def refund(order: str, amount: float) -> str:
    """Your tool, as it is."""
    return f"refunded {amount:g} on {order}"


class NoDisputedRefunds(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | None:
        return (
            Deny("no refunds on order o-0: it is disputed") if call.args["order"] == "o-0" else None
        )


async def approve_below_500(decision: Decision) -> bool:
    """Your approval UI: here, a rule standing in for a person."""
    print("  asked:", decision.question)
    return float(decision.args["amount"]) < 500


async def main() -> None:
    gov = governance()
    for amount in (40, 250):
        decision = await gov.check(
            "refund", {"order": "o-7", "amount": amount}, side_effects="write"
        )
        print(f"refund {amount}: {decision.action.value} ({decision.reason or decision.risk})")

    run = governed(
        refund, gov, side_effects="write", on_ask=approve_below_500, hooks=[NoDisputedRefunds()]
    )
    print(await run(order="o-7", amount=40))  # announced: runs
    print(await run(order="o-8", amount=250))  # the rule asks; approved
    for order, amount in (("o-9", 900), ("o-0", 10)):
        try:
            await run(order=order, amount=amount)
        except Rejected as rejected:
            print("rejected:", rejected.decision.question)
        except Denied as denied:
            print("denied:", denied.outcome.output)

    asked = await gov.check("refund", {"order": "o-8", "amount": 250})
    await gov.decided(asked, "approve", reviewer="lee", run_id="run_example", user="ada")
    await gov.aclose()


if __name__ == "__main__":
    asyncio.run(main())
