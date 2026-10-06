"""Scenario: an agent on the gateway's MCP tools — many read-only ones behind Code Mode, one
that pays out behind an administrator's rule — then evaluated on what it said and what it did.

* Code Mode: three MCP servers whose tools all only read (the tool catalog says so) are offered
  as Bifrost's Code Mode meta-tools, so the model writes one script (``execute_tool_code``)
  instead of many calls; the script's nested calls are read back from the gateway's log and
  recorded;
* governance: ``erp-refund`` is governed by the catalog's ``approve_when`` (``amount > 100``,
  an administrator's rule): a refund of 250 waits for a person, one of 40 would not;
* evaluation: ``h.evaluate`` over a small dataset, scored by ``contains`` and by the run's
  trajectory (``called``, ``tool_sequence``).

Offline the gateway and the memory service (its catalog) are scripted in this process.

    python -m examples.06_scenarios.gateway_code_mode_governed_evals
"""

from __future__ import annotations

import asyncio

from examples._support.gateway import McpTool
from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, react_model

from trellis import Harness, ReAct
from trellis.harness.evals import called, contains, tool_sequence


def customer(name: str) -> str:
    """A customer's tier."""
    return f"{name}: gold"


def policy(topic: str) -> str:
    """The policy on a topic."""
    return f"{topic}: gold customers are refunded in full"


def manual(section: str) -> str:
    """A section of the operations manual."""
    return f"{section}: refund to the original card"


def refund(order: str, amount: int) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


TOOLS = {
    "crm-customer": McpTool(customer, read_only=True),
    "wiki-policy": McpTool(policy, read_only=True),
    "docs-manual": McpTool(manual, read_only=True),
    "erp-refund": McpTool(refund, destructive=True),
}
#: the catalog: the Code Mode servers' tools only read; refunds over 100 need a person
CATALOG = {
    "crm-customer": {"risk": "read"},
    "wiki-policy": {"risk": "read"},
    "docs-manual": {"risk": "read"},
    "erp-refund": {"risk": "write", "approve_when": "amount > 100"},
}
SCRIPT = "print(crm.customer(name='acme'))\nprint(wiki.policy(topic='refunds'))"
DATASET = [
    {"input": "What is acme's tier?", "expected": "gold"},
    {"input": "How do we refund acme?", "expected": "in full"},
]


async def main() -> None:
    model = react_model(
        [
            ("execute_tool_code", {"code": SCRIPT}),
            ("erp-refund", {"order": "o-7", "amount": 250}),
            "acme is gold: refunded 250 on o-7 in full.",
            ("execute_tool_code", {"code": "print(crm.customer(name='acme'))"}),
            "acme is a gold customer.",
            ("execute_tool_code", {"code": "print(wiki.policy(topic='refunds'))"}),
            "Gold customers are refunded in full.",
        ]
    )
    memory = ScriptedMemory(catalog=CATALOG)
    blocks = offline_blocks(
        memory=memory,
        mcp=TOOLS,
        code_mode=frozenset({"crm", "wiki", "docs"}),
    )
    async with Harness(**blocks) as h:
        agent = h.wrap(ReAct(system="You handle refunds for support.", model=model), id="refunds")

        result = await agent.run("acme wants order o-7 (250 EUR) refunded.", user="ada")
        while result.interrupt is not None:  # amount > 100: the administrator's rule asks
            print("asks:", result.interrupt.question)
            result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="lee")
        print(result.status.value, result.answer)

        report = await h.evaluate(
            agent,
            DATASET,
            [
                contains(),
                called("execute_tool_code"),
                tool_sequence(["execute_tool_code"], exact=True),
            ],
            run_name="refunds-code-mode",
            concurrency=1,
        )
        print(report)
    # the scripts, the refund and the calls the scripts made (read back from the gateway's log)
    print("tool calls recorded in memory:", memory.written["record_tool"])


if __name__ == "__main__":
    asyncio.run(main())
