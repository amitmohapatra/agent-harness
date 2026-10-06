"""People in the loop: an approval rule in code, labelled choices, a tool whose result comes from
outside, an approval remembered for the run, a comment — on a ``ReAct`` agent, answered by a
scripted reviewer (``trellis.testing.Reviewer``) as a test would.

* ``refund`` has an approval rule in code, a ``before_tool`` hook: up to 20 it runs unasked (the
  tool is ``write``: governance only announces it), over 500 it asks finance on its own screen
  (``Ask(..., assignee=, component=, props=)``), anything between is asked of whoever answers;
* ``pick_plans`` asks which plans to offer: labelled options, several picks;
* ``sign``'s result comes from outside the run: the tool asks (``ask(..., expects=)``), the run
  pauses inside it, and the signature system's answer is what the tool returns and the model
  reads (``agent.resume(run_id, "answer", answer=...)``).

    .venv/bin/python examples/approvals.py
"""

from __future__ import annotations

import asyncio

from _offline import react_model

import trellis
from trellis import Ask, Harness, Hooks, ReAct, tool
from trellis.contracts import Option, RunStatus, ToolCall
from trellis.testing import Decide, Reviewer


class RefundRule(Hooks):
    """When a refund needs a person, by its amount."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        amount = call.args.get("amount", 0) if call.tool == "refund" else 0
        if amount <= 20:
            return None  # small: no person
        if amount > 500:
            return Ask(
                f"Refund {amount} EUR?",
                assignee="role:finance",
                component="refund-review",
                props={"amount": amount},
            )
        return Ask(f"Refund {amount} EUR?")


@tool(side_effects="write")
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


@tool(side_effects="write")
async def sign(contract: str) -> str:
    """Have a contract signed in the e-signature system (a person signs it)."""
    runtime = trellis.current()
    assert runtime is not None
    # the run pauses here; what the e-signature system answers is what this tool returns
    return await runtime.ask(
        f"Signed {contract}?",
        expects={"type": "string"},  # what this tool returns
        component="e-signature",
        props={"contract": contract},
    )


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
            hooks=[RefundRule()],
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
            if asked.component == "e-signature":
                # the e-signature system answers: what sign returns and the model reads
                result = await agent.resume(
                    result.run_id, "answer", answer="signed by ada, 10:42", reviewer="e-signature"
                )
            else:
                result = await reviewer.answer(agent, result)
        print(result.status.value, result.answer)
        print("answered:", reviewer.answered)


if __name__ == "__main__":
    asyncio.run(main())
