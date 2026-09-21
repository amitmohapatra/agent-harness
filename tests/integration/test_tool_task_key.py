"""Tool memory is keyed on what the turn was *for*.

The service normalises the task into a typed-placeholder pattern — "how much stock of
{entity}?" — so two phrasings of the same question mine the same trajectory. It was being
sent ``context.task_id``, an opaque identifier, which meant the pattern matched only other
runs of that same id or, far more often, was empty: no task_id, no pattern, no procedure,
ever. Verified against the running service: with the query sent instead, three runs of one
question shape produce a procedure with support=3 that is retrieved for an unseen entity.
"""

from __future__ import annotations

from universal_agent_harness import AgentExecutionContext, AgentHarness


async def stock_db(sku: str) -> dict:
    return {"sku": sku, "on_hand": 95}


def _harness(memory):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    harness.register_tool(stock_db, name="stock_db")
    return harness


async def test_the_query_is_what_keys_the_procedure(memory, context):
    harness = _harness(memory)

    @harness.agent(agent_id="inv")
    async def node(state, agent):
        await agent.tools.call("stock_db", sku="SKU-1")
        return "ok"

    await node({"question": "how much stock of SKU-1 do we have?"}, context=context)
    assert memory.of("tools.record")[0]["task"] == "how much stock of SKU-1 do we have?"
    await harness.aclose()


async def test_an_explicit_task_on_the_call_wins(memory, context):
    from universal_agent_contracts.tool import ToolCall

    harness = _harness(memory)

    @harness.agent(agent_id="inv")
    async def node(state, agent):
        await agent.tools.call(
            ToolCall(tool="stock_db", args={"sku": "SKU-1"}, task="reorder review")
        )
        return "ok"

    await node({"question": "how much stock?"}, context=context)
    assert memory.of("tools.record")[0]["task"] == "reorder review"
    await harness.aclose()


async def test_no_query_falls_back_to_the_task_id_not_to_nothing(memory):
    harness = _harness(memory)

    @harness.agent(agent_id="inv")
    async def node(state, agent):
        await agent.tools.call("stock_db", sku="SKU-1")
        return "ok"

    ctx = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", thread_id="tk-1", user_id="u1", task_id="task-42"
    )
    await node({"unrelated": 1}, context=ctx)
    assert memory.of("tools.record")[0]["task"] == "task-42"
    await harness.aclose()
