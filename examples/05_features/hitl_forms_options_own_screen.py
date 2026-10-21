"""People in the loop: an approval rule, a reviewer and screen of your own, labelled choices, a
form, a tool whose result comes from outside, a comment — on a ``ReAct`` agent, answered by a
scripted reviewer (``trellis.testing.Reviewer``) as a test would.

* ``refund``: when it needs a person is the tool catalog's rule (``approve_when: amount > 20``;
  up to 20 it runs unasked, announced — the tool is ``write``); who reviews a large one, and on
  which screen, is what no rule says: a ``before_tool`` hook asks finance on its own screen over
  500 (``Ask(..., assignee=, component=, props=)``);
* ``pick_plans`` asks which plans to offer: labelled options, several picks;
* ``ship_to`` asks for an address: a form (``form=`` a pydantic model, ``ui_schema=`` its widget
  hints), whose answer is read back into the model;
* ``sign``'s result comes from outside the run: the tool asks (``ask(..., expects=)``), the run
  pauses inside it, and the signature system's answer is what the tool returns and the model
  reads (``agent.resume(run_id, "answer", answer=...)``).

    python -m examples.05_features.hitl_forms_options_own_screen
"""

from __future__ import annotations

import asyncio

from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, react_model
from pydantic import BaseModel

import trellis
from trellis import Ask, Harness, Hooks, ReAct, tool
from trellis.contracts import Option, RunStatus, ToolCall
from trellis.testing import Decide, Reviewer


class Address(BaseModel):
    street: str
    city: str


#: When a refund needs a person: the catalog's rule (an administrator's, or a deploy script's
#: ``put_catalog``), here held by the scripted memory service.
CATALOG = {"refund": {"risk": "write", "approve_when": "amount > 20"}}


class FinanceReviews(Hooks):
    """Who reviews a large refund, and on which screen — what the catalog's rule cannot say."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        amount = call.args.get("amount", 0) if call.tool == "refund" else 0
        if amount > 500:
            return Ask(
                f"Refund {amount} EUR?",
                assignee="role:finance",
                component="refund-review",
                props={"amount": amount},
            )
        return None  # the rest: governance, by the catalog's rule


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


@tool(side_effects="read")
async def ship_to(customer: str) -> str:
    """Ask the customer's account manager where to ship."""
    runtime = trellis.current()
    assert runtime is not None
    address = await runtime.ask(
        f"Where do we ship {customer}'s welcome kit?",
        form=Address,
        ui_schema={"street": {"ui:widget": "textarea"}},
    )
    return f"{address.street}, {address.city}"


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
    memory = ScriptedMemory(catalog=CATALOG)
    async with Harness(**offline_blocks(memory=memory)) as h:
        model = react_model(
            [
                ("refund", {"order": "o1", "amount": 15}),
                ("refund", {"order": "o2", "amount": 900}),
                ("pick_plans", {"customer": "acme"}),
                ("ship_to", {"customer": "acme"}),
                ("sign", {"contract": "c-7"}),
                "Refunded o1 and o2, offered basic and pro, contract c-7 signed.",
            ]
        )
        agent = h.wrap(
            ReAct(system="You handle accounts.", model=model),
            id="accounts",
            tools=[refund, pick_plans, ship_to, sign],
            hooks=[FinanceReviews()],
        )
        reviewer = Reviewer(
            {
                "refund": Decide("approve", comment="checked on refund-review"),
                "Which plans for acme?": ["basic", "pro"],
                "Where do we ship acme's welcome kit?": {"street": "1 Main St", "city": "Oslo"},
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
