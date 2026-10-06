"""Hooks: a guardrail and redaction of your own, on a ``ReAct`` agent.

* ``Refunds`` — no refund over 1000 (``Deny``: the model reads why), a person for one over 100
  (``Ask``: the run pauses for ``role:support-lead``), amounts rounded to cents (``Rewrite``);
* ``Cards`` — card numbers masked in every tool result (``after_tool``: what the model reads,
  the journal and memory keep) and in every message sent to the model (``before_model``).

    .venv/bin/python examples/hooks.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any

from _offline import react_model

from trellis import Ask, Deny, Harness, Hooks, ModelCall, ReAct, Rewrite, tool
from trellis.contracts import ToolCall, ToolOutcome

CARD = re.compile(r"\b(\d{4})\d{8}(\d{4})\b")


def mask(value: Any) -> Any:
    """Card numbers as ``1234********5678``."""
    if isinstance(value, str):
        return CARD.sub(r"\1********\2", value)
    if isinstance(value, dict):
        return {key: mask(item) for key, item in value.items()}
    return value


class Refunds(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | Ask | Rewrite | None:
        if call.tool != "refund":
            return None
        amount = float(call.args["amount"])
        if amount > 1000:
            return Deny("refunds over 1000 go through finance")
        if amount > 100:
            return Ask(f"Refund {amount:g}?", assignee="role:support-lead")
        return Rewrite({**call.args, "amount": round(amount, 2)})


class Cards(Hooks):
    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome.model_copy(update={"output": mask(outcome.output)})

    async def before_model(self, call: ModelCall) -> ModelCall:
        return dataclasses.replace(call, messages=[mask(m) for m in call.messages])


@tool(side_effects="read")
def order(number: str) -> str:
    """An order: its amount and the card it was paid with."""
    return f"order {number}: 250 EUR, card 4111111111111111"


@tool(side_effects="write")
def refund(number: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount:g} EUR on {number}"


async def main() -> None:
    async with Harness(hooks=[Cards()]) as h:  # every agent: no card number reaches a model
        model = react_model(
            [
                ("order", {"number": "o1"}),
                ("refund", {"number": "o1", "amount": 250}),
                "Refunded 250 EUR on o1.",
            ]
        )
        target = ReAct(system="You handle refunds. Look the order up first.", model=model)
        agent = h.wrap(target, id="refunds", tools=[order, refund], hooks=[Refunds()])
        paused = await agent.run("Refund order o1, card 4111111111111111.", user="ada")
        assert paused.interrupt is not None  # over 100: a person approves it
        print("waiting:", paused.interrupt.question, "-", paused.interrupt.assignee)
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
        print(done.status.value, done.answer)


if __name__ == "__main__":
    asyncio.run(main())
