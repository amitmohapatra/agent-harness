"""People in the loop: an approval rule in code, labelled choices, a tool whose result comes from
outside, an approval remembered for the run, a comment — on a ``ReAct`` agent, answered by a
scripted reviewer (``trellis.testing.Reviewer``) as a test would.

* ``refund`` has an approval function: up to 20 it is approved by the rule (no person), over 500
  it asks finance on its own screen (``Ask(..., component=, props=)``), else governance decides
  (it is ``irreversible``: it asks);
* ``pick_plans`` asks which plans to offer: labelled options, several picks;
* ``sign`` is external: the run pauses with the call, and the signature system's answer is what
  the model reads (``agent.resume(run_id, result=...)``).

    .venv/bin/python examples/approvals.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _offline import react_model

import trellis
from trellis import Ask, Harness, ReAct, tool
from trellis.contracts import Option, RunStatus
from trellis.testing import Decide, Reviewer


def refund_rule(args: dict[str, Any]) -> Ask | bool | None:
    amount = args["amount"]
    if amount <= 20:
        return True
    if amount > 500:
        return Ask(
            f"Refund {amount} EUR?",
            assignee="role:finance",
            component="refund-review",
            props={"amount": amount},
        )
    return None


@tool(side_effects="irreversible", approval=refund_rule)
def refund(order: str, amount: int) -> str:
    """Refund an order."""
    return f"refunded {amount} EUR on {order}"


@tool(side_effects="read")
async def pick_plans(customer: str) -> str:
    """Ask sales which plans to offer a customer."""
    runtime = trellis.current()
    assert runtime is not None
    picked = await runtime.ask(
        f"Which plans for {customer}?",
        options=[Option(value="basic", label="Basic, 10 EUR"), Option(value="pro", label="Pro")],
        multiple=True,
        assignee="role:sales",
    )
    return ", ".join(picked)


@tool(side_effects="write", external=True)
def sign(contract: str) -> str:
    """Have a contract signed in the e-signature system (a person signs it)."""
    raise NotImplementedError  # never runs: the result comes from outside


async def main() -> None:
    async with Harness() as h:
        model = react_model(
            [
                ("refund", {"order": "o1", "amount": 15}),
                ("refund", {"order": "o2", "amount": 900}),
                ("pick_plans", {"customer": "acme"}),
                ("sign", {"contract": "c-7"}),
                "Refunded o1 and o2, offered basic and pro, contract c-7 signed.",
            ]
        )
        agent = h.wrap(
            ReAct(system="You handle accounts.", model=model),
            id="accounts",
            tools=[refund, pick_plans, sign],
        )
        reviewer = Reviewer(
            {
                "refund": Decide("approve", comment="checked on refund-review"),
                "Which plans for acme?": ["basic", "pro"],
            },
            name="lee",
        )
        result = await agent.run("Settle acme's account", user="ada")
        while result.status is RunStatus.PAUSED and result.interrupt is not None:
            asked = result.interrupt
            print("waiting:", asked.question, f"({asked.component or asked.ui})")
            if asked.tool_call is not None and asked.tool_call.tool == "sign":
                # the e-signature system answers the external call: what the model reads
                result = await agent.resume(result.run_id, result="signed by ada, 10:42")
            else:
                result = await reviewer.answer(agent, result)
        print(result.status.value, result.answer)
        print("answered:", reviewer.answered)


if __name__ == "__main__":
    asyncio.run(main())
