"""The production shape: Memory Service + OpenTelemetry + Langfuse, all by configuration.

Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and optionally LANGFUSE_HOST) to enable
Langfuse; without them the example runs with Langfuse disabled and everything else intact.

    python examples/with_memory_and_langfuse.py
"""

from __future__ import annotations

import asyncio
import os

from universal_agent_harness import AgentExecutionContext, AgentHarness, AgentResult

MEMORY_URL = os.environ.get("MEMORY_SERVICE_URL")
LANGFUSE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


class EchoModel:
    """Stand-in for a real provider client (anything with ``ainvoke`` works)."""

    async def ainvoke(self, prompt, **kwargs):
        return {
            "text": f"answer to: {prompt}",
            "model": "demo-model",
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "cost_usd": 0.0002},
        }


def build_harness() -> AgentHarness:
    memory = None
    if MEMORY_URL:
        from universal_memory import MemoryClient  # noqa: PLC0415 - optional in this example

        memory = MemoryClient(MEMORY_URL, api_key=os.environ.get("MEMORY_API_KEY"))

    config = {
        "memory": {"enabled": bool(memory)},
        "observability": {
            "langfuse": {
                "enabled": LANGFUSE_ON,
                # keys and host come from LANGFUSE_* environment variables
                "sampling": {"sample_rate": 1.0, "error_sample_rate": 1.0},
                "capture": {
                    # conservative on purpose: ids, counts and costs, no content
                    "raw_prompts": False,
                    "raw_model_outputs": False,
                },
            }
        },
        "evaluation_events": {"enabled": True},
    }
    return AgentHarness(memory=memory, model=EchoModel(), config=config,
                        defaults={"tenant_id": "acme", "user_id": "u1"})


async def main() -> None:
    harness = build_harness()
    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="planner-agent", user_id="u1",
        thread_id="chat-42", turn_id="turn-1",
    )

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def inventory_agent(state, agent) -> AgentResult:
        response = await agent.model.invoke(state["question"])
        return AgentResult.ok(response.text, metrics={"tokens": float(response.usage.tokens or 0)})

    @harness.agent(agent_id="planner-agent")
    async def planner(state, agent) -> AgentResult:
        inventory = await inventory_agent(state)          # a nested run: lineage is automatic
        return AgentResult.ok({"plan": inventory.data})

    result = await planner({"question": "how much stock of SKU-1?"}, context=context)
    print("result:", result.data)
    print("memory:", "enabled" if MEMORY_URL else "disabled (set MEMORY_SERVICE_URL)")
    print("langfuse:", "enabled" if LANGFUSE_ON else "disabled (set LANGFUSE_* keys)")

    await harness.aclose()   # drains memory writeback and flushes telemetry


if __name__ == "__main__":
    asyncio.run(main())
