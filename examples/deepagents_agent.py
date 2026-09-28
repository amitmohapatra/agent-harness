"""Deep Agents through the harness: the six moments, against a scripted model.

Runs as-is, with no services and no API keys: the model is a scripted client behind the
harness's own model port, so the example is honest about what it shows (the bindings) without
pretending to show a real model's judgement. Point ``model=`` at a ``BifrostModelClient`` and
the same code talks to the gateway.

    pip install "trellis-harness[deepagents]"
    python examples/deepagents_agent.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool
from trellis.contracts import (
    AgentExecutionContext,
    InterruptDecision,
    InterruptResolution,
    ModelResponse,
    ModelUsage,
    RunEventType,
    RunOutcome,
)

from trellis.harness import AgentHarness, CallablePolicyProvider, CollectingEventSink
from trellis.harness.interrupts import ApprovalRequired, interrupt_from_signal

CALLED: list[dict[str, Any]] = []


@tool
def lookup(sku: str) -> str:
    """Look up the stock of one SKU."""
    CALLED.append({"sku": sku})
    return f"{sku}: 3 on hand, reorder point 50"


class ScriptedModel:
    """The harness model port: asks for the tool once, then answers.

    A rule rather than a fixed script, because a resumed run re-plans from scratch — which is
    exactly what the approval part of this example exercises.
    """

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        if any(m.get("role") == "tool" for m in request.messages or ()):
            return ModelResponse(
                text="SKU-1 is 47 below its reorder point.",
                model="scripted",
                usage=ModelUsage(input_tokens=42, output_tokens=11),
            )
        return ModelResponse(
            text=None,
            model="scripted",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"sku": "SKU-1"})},
                }
            ],
        )

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def context(thread_id: str) -> AgentExecutionContext:
    """A context that names its turn, so a resumed run is the same run."""
    return AgentExecutionContext.create(
        tenant_id="acme",
        user_id="u1",
        agent_id="inventory-agent",
        thread_id=thread_id,
        turn_id=f"trn_{thread_id}",
    )


async def main() -> None:
    sink = CollectingEventSink()

    async def hold_writes(_context: Any, call: Any) -> Any:
        """A policy: reading stock is fine, anything else waits for a person."""
        return True if call.tool == "lookup" else "require_approval"

    harness = AgentHarness(
        model=ScriptedModel(),
        policy=CallablePolicyProvider(tool=hold_writes),
        event_sinks=[sink],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        # In a real deployment:
        #   AgentHarness(memory=MemoryClient(...), model=BifrostModelClient(...))
    )

    run = harness.deepagents.agent(agent_id="inventory-agent", tools=[lookup])
    turn = context("chat-1")
    result = await run(
        {"messages": [{"role": "user", "content": "how much stock of SKU-1?"}]}, context=turn
    )

    print("\n--- the answer ---")
    print(result.data["messages"][-1].content)
    print(f"tool ran with: {CALLED}")

    print("\n--- what a UI saw ---")
    for event in sink.for_run(turn.agent_run_id):
        detail = event.tool_call_id or event.step or ""
        print(f"  {event.sequence:>2}  {event.type.value:<18} {detail}")

    # -- and the same run, with a tool the policy holds for a person ----------------------
    print("\n--- a tool that needs approval ---")

    @tool
    def reorder(sku: str, quantity: int) -> str:
        """Place a reorder. Costs money, so a person approves it."""
        CALLED.append({"sku": sku, "quantity": quantity})
        return f"ordered {quantity} of {sku}"

    class OrdersThenReports(ScriptedModel):
        async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
            if any(m.get("role") == "tool" for m in request.messages or ()):
                return ModelResponse(text="The reorder is placed.", model="scripted")
            return ModelResponse(
                text=None,
                model="scripted",
                tool_calls=[
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "reorder",
                            "arguments": json.dumps({"sku": "SKU-1", "quantity": 50}),
                        },
                    }
                ],
            )

    approvals = AgentHarness(
        model=OrdersThenReports(),
        policy=CallablePolicyProvider(tool=hold_writes),
        event_sinks=[sink],
        defaults={"tenant_id": "acme", "user_id": "u1"},
    )
    ordering = approvals.deepagents.agent(agent_id="reorder-agent", tools=[reorder])
    ordering_turn = AgentExecutionContext.create(
        tenant_id="acme",
        user_id="u1",
        agent_id="reorder-agent",
        thread_id="chat-2",
        turn_id="trn_chat-2",
    )
    payload = {"messages": [{"role": "user", "content": "reorder 50 of SKU-1"}]}
    try:
        await ordering(payload, context=ordering_turn)
    except ApprovalRequired as paused:
        interrupt = interrupt_from_signal(paused, ordering_turn)
        print(f"  paused: {interrupt.question}")
        print(f"  the call a person is being asked about: {interrupt.tool_call.args}")
        resumed = await approvals.resume(
            interrupt,
            InterruptResolution(
                interrupt_id=interrupt.interrupt_id,
                run_id=interrupt.run_id,
                decision=InterruptDecision.APPROVE,
                reviewer="u1",
            ),
            context=ordering_turn,
            agent=ordering,
            payload=payload,
        )
        print(f"  approved, and the run continued: {resumed.data['messages'][-1].content}")
        print(f"  the tool ran once: {CALLED[-1]}")

    finishes = [
        e.outcome
        for e in sink.for_run(ordering_turn.agent_run_id)
        if e.type is RunEventType.RUN_FINISHED
    ]
    print(f"  the run finished: {[o.value for o in finishes]}")
    assert finishes[0] is RunOutcome.INTERRUPT and finishes[-1] is RunOutcome.SUCCESS

    await harness.aclose()
    await approvals.aclose()


if __name__ == "__main__":
    asyncio.run(main())
