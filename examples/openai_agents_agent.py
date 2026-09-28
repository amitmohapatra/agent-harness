"""The OpenAI Agents SDK through the harness: the six moments, against a scripted model.

Runs as-is, with no services and no API keys. Point ``model=`` at a ``BifrostModelClient`` and
the same code talks to the gateway; nothing else changes.

    pip install "trellis-harness[openai-agents]"
    python examples/openai_agents_agent.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, function_tool
from trellis.contracts import (
    AgentExecutionContext,
    InterruptDecision,
    InterruptResolution,
    ModelResponse,
    ModelUsage,
    RunEventType,
)

from trellis.harness import AgentHarness, CollectingEventSink
from trellis.harness.interrupts import ApprovalRequired, interrupt_from_signal

CALLED: list[dict[str, Any]] = []


@function_tool
def lookup(sku: str) -> str:
    """Look up the stock of one SKU."""
    CALLED.append({"sku": sku})
    return f"{sku}: 3 on hand, reorder point 50"


@function_tool(needs_approval=True)
def reorder(sku: str, quantity: int) -> str:
    """Place a reorder. The SDK's own approval flow holds this for a person."""
    CALLED.append({"sku": sku, "quantity": quantity})
    return f"ordered {quantity} of {sku}"


class ScriptedModel:
    """Calls one tool, then answers once it sees the tool's result."""

    def __init__(self, tool_name: str, args: dict[str, Any], answer: str) -> None:
        self.tool_name, self.args, self.answer = tool_name, args, answer

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        if any(m.get("role") == "tool" for m in request.messages or ()):
            return ModelResponse(
                text=self.answer,
                model="scripted",
                usage=ModelUsage(input_tokens=51, output_tokens=9),
            )
        return ModelResponse(
            text=None,
            model="scripted",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(self.args),
                    },
                }
            ],
        )

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def turn(agent_id: str, thread_id: str) -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id="acme",
        user_id="u1",
        agent_id=agent_id,
        thread_id=thread_id,
        turn_id=f"trn_{thread_id}",
    )


async def main() -> None:
    sink = CollectingEventSink()
    harness = AgentHarness(
        model=ScriptedModel("lookup", {"sku": "SKU-1"}, "SKU-1 is 47 below its reorder point."),
        event_sinks=[sink],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        # In a real deployment:
        #   AgentHarness(memory=MemoryClient(...), model=BifrostModelClient(...))
    )

    # -- an agent you already have, wrapped ------------------------------------------------
    agent = Agent(
        name="inventory",
        instructions=harness.openai_agents.instructions("You answer stock questions briefly."),
        tools=[lookup],
    )
    run = harness.openai_agents.wrap(agent, agent_id="inventory-agent")
    context = turn("inventory-agent", "chat-1")
    result = await run("how much stock of SKU-1?", context=context)

    print("\n--- the answer ---")
    print(result.data)
    print(f"tool ran with: {CALLED}")

    print("\n--- what a UI saw ---")
    for event in sink.for_run(context.agent_run_id):
        print(
            f"  {event.sequence:>2}  {event.type.value:<20} "
            f"{event.tool_call_id or event.step or ''}"
        )

    # -- the SDK's own needs_approval, as one harness Interrupt ----------------------------
    print("\n--- a tool the SDK holds for a person ---")
    approvals = AgentHarness(
        model=ScriptedModel("reorder", {"sku": "SKU-1", "quantity": 50}, "The reorder is placed."),
        event_sinks=[sink],
        defaults={"tenant_id": "acme", "user_id": "u1"},
    )
    ordering = approvals.openai_agents.agent(agent_id="reorder-agent", tools=[reorder])
    ordering_turn = turn("reorder-agent", "chat-2")
    try:
        await ordering("reorder 50 of SKU-1", context=ordering_turn)
    except ApprovalRequired as paused:
        interrupt = interrupt_from_signal(paused, ordering_turn)
        print(f"  paused: {interrupt.question}")
        print(f"  the call: {interrupt.tool_call.tool} {interrupt.tool_call.args}")
        print(f"  the tool has not run: {CALLED[-1]}")
        # An EDIT decision is refused rather than silently approved: the SDK approves or
        # rejects a call, it does not rewrite one.
        try:
            approvals.openai_agents.apply_resolution(
                _PendingState(),
                InterruptResolution(
                    interrupt_id=interrupt.interrupt_id,
                    run_id=interrupt.run_id,
                    decision=InterruptDecision.EDIT,
                    payload={"quantity": 10},
                ),
            )
        except ValueError as refused:
            print(f"  an edit is refused, with the reason: {refused}")

    finishes = [
        e.outcome.value
        for e in sink.for_run(ordering_turn.agent_run_id)
        if e.type is RunEventType.RUN_FINISHED
    ]
    print(f"  the run finished: {finishes}")

    await harness.aclose()
    await approvals.aclose()


class _PendingState:
    """Stands in for the SDK's ``RunState`` so the example can show the refusal."""

    def get_interruptions(self) -> list[str]:
        return ["the pending call"]

    def approve(self, item: str, always_approve: bool = False) -> None: ...

    def reject(self, item: str, always_reject: bool = False) -> None: ...


if __name__ == "__main__":
    asyncio.run(main())
