"""Approval rules and hooks, on a ``ReAct`` agent — each in its one place:

* **when a call needs a person** is a rule in the tool catalog (``approve_when``, set by an
  administrator or a deploy script; here the scripted memory service holds it): a refund over
  100 asks. Rules live there and only there, and apply to every framework and process;
* **what a rule cannot say** is a hook — ``Refunds``: no refund over 1000 at all (``Deny``:
  the model reads why), amounts rounded to cents (``Rewrite``); ``Cards``: card numbers masked
  in every tool result (``after_tool``: what the model reads, the journal and memory keep) and
  in every message sent to the model (``before_model``).

    python -m examples.02_way1_react.hooks_guardrails
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any

from examples._support.memory import ScriptedMemory
from examples._support.offline import offline_blocks, react_model

from trellis import Deny, Harness, Hooks, ModelCall, ReAct, Rewrite, tool
from trellis.contracts import ToolCall, ToolOutcome

CARD = re.compile(r"\b(\d{4})\d{8}(\d{4})\b")


def mask(value: Any) -> Any:
    """Card numbers as ``1234********5678``."""
    if isinstance(value, str):
        return CARD.sub(r"\1********\2", value)
    if isinstance(value, dict):
        return {key: mask(item) for key, item in value.items()}
    return value


#: The catalog's rule (what ``put_catalog`` writes in the memory service): over 100 asks.
CATALOG = {"refund": {"risk": "write", "approve_when": "amount > 100"}}


class Refunds(Hooks):
    """What no rule says: a refusal, and a rewrite. When a refund asks is the catalog's."""

    async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
        if call.tool != "refund":
            return None
        amount = float(call.args["amount"])
        if amount > 1000:
            return Deny("refunds over 1000 go through finance")
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
    memory = ScriptedMemory(catalog=CATALOG)
    async with Harness(**offline_blocks(memory=memory)) as h:
        model = react_model(
            [
                ("order", {"number": "o1"}),
                ("refund", {"number": "o1", "amount": 250}),
                "Refunded 250 EUR on o1.",
            ]
        )
        target = ReAct(
            system_prompt="You handle refunds. Look the order up first.",
            model=model,
            tools=[order, refund],
        )
        # in order: no card number reaches a model, then the refund rules
        agent = h.wrap(target, id="refunds", hooks=[Cards(), Refunds()])
        paused = await agent.run("Refund order o1, card 4111111111111111.", user="ada")
        assert paused.interrupt is not None  # over 100: the catalog's rule asks
        print("waiting:", paused.interrupt.question)
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
        print(done.status.value, done.answer)


if __name__ == "__main__":
    asyncio.run(main())
