"""Memory Service integration (§10, §11, §42, §77)."""

from __future__ import annotations

import asyncio

import pytest

from universal_agent_harness import AgentHarness, AgentResult, MemoryObservation, MemoryPolicy
from universal_agent_harness.contracts.errors import MemoryUnavailableError


async def test_context_is_retrieved_before_the_agent_runs(harness, memory, context):
    async def agent(payload, agent_runtime):
        assert agent_runtime.memory_context is not None
        assert agent_runtime.memory_context.rendered.startswith("remembered:")
        return "ok"

    await harness.wrap(agent, agent_id="inv")({"question": "how much stock?"}, context=context)

    assert len(memory.retrievals) == 1
    retrieval = memory.retrievals[0]
    assert retrieval["query"] == "how much stock?"
    assert retrieval["scope"]["tenant_id"] == "acme"
    assert retrieval["scope"]["agent_id"] == "inv"
    assert retrieval["scope"]["thread_id"] == "chat-1"


async def test_no_query_means_no_retrieval(harness, memory, context):
    async def agent(payload):
        return "ok"

    await harness.wrap(agent, agent_id="inv")({"unrelated": 1}, context=context)
    assert memory.retrievals == []


async def test_observations_are_written_after_the_result(harness, memory, context):
    async def agent(payload):
        return AgentResult.ok(
            "stock is low",
            memory_observations=[MemoryObservation(content="SKU-1 is short by 40 units")],
        )

    await harness.wrap(agent, agent_id="inv")("how much stock?", context=context)

    contents = [o["content"] for o in memory.observations]
    assert "SKU-1 is short by 40 units" in contents
    assert "how much stock?" in contents  # observe_input
    assert "stock is low" in contents  # observe_output


async def test_writeback_does_not_block_the_result(memory, context):
    harness = AgentHarness(memory=memory, defaults={"tenant_id": "acme"})

    async def agent(payload):
        return "answer"

    result = await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert result.data == "answer"  # returned before the writes land
    await harness.drain()
    assert memory.observations  # ...which then happen


async def test_observation_keys_are_stable_across_retries(harness, memory, context):
    async def agent(payload):
        return "same answer"

    wrapped = harness.wrap(agent, agent_id="inv")
    await wrapped("q", context=context)
    await wrapped("q", context=context)

    keys = [o["idempotency_key"] for o in memory.observations]
    assert len(keys) == 4  # two runs x (input, output)
    assert len(set(keys)) == 2  # ...but only two distinct keys (§42)


async def test_claims_are_observed_when_the_policy_asks(harness, memory, context):
    from universal_agent_harness import Claim

    async def agent(payload):
        return AgentResult.ok("x", claims=[Claim(claim_id="c1", text="stock is low")])

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    claim_writes = [o for o in memory.observations if o["content"] == "stock is low"]
    assert claim_writes, "the claim should have been written"
    # the kind says where it came from (the agent's result); the hint says what it is
    assert claim_writes[0]["kind"] == "AGENT_RESULT"
    assert claim_writes[0]["hints"]["memory_type"] == "SEMANTIC"


async def test_memory_policy_can_be_narrowed_per_agent(harness, memory, context):
    async def agent(payload):
        return "answer"

    await harness.wrap(
        agent,
        agent_id="private-agent",
        memory_policy=MemoryPolicy(retrieve_before=False, observe_input=False, observe_output=False),
    )("q", context=context)

    assert memory.retrievals == []
    assert memory.observations == []


async def test_private_by_default_marks_observations_run_visible(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"private_by_default": True, "writeback": False}},
    )

    async def agent(payload):
        return "secret working note"

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert all(o["hints"].get("visibility") == "RUN" for o in memory.observations)


async def test_retrieval_failure_degrades_by_default(harness, memory, context):
    memory.fail_retrieval = True

    async def agent(payload, runtime):
        assert runtime.memory_context is None
        return "answered without memory"

    result = await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert result.data == "answered without memory"
    assert any(w.code == "MEMORY_DEGRADED" for w in result.warnings)


async def test_fail_closed_turns_a_memory_outage_into_an_error(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"failure_mode": "fail_closed", "writeback": False}},
    )
    memory.fail_retrieval = True

    async def agent(payload):
        return "should not run"

    with pytest.raises(MemoryUnavailableError):
        await harness.wrap(agent, agent_id="inv")("q", context=context)


async def test_memory_calls_are_bounded_by_the_memory_timeout(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 0.05}},
    )
    memory.retrieval_delay = 1.0

    async def agent(payload, runtime):
        return "no context" if runtime.memory_context is None else "had context"

    result = await asyncio.wait_for(
        harness.wrap(agent, agent_id="inv")("q", context=context), timeout=2
    )
    assert result.data == "no context"


async def test_memory_disabled_gives_a_working_noop_runtime(context):
    harness = AgentHarness(defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        assert runtime.memory.enabled is False
        assert await runtime.memory.retrieve("anything") is None
        return "fine"

    assert (await harness.wrap(agent, agent_id="inv")("q", context=context)).data == "fine"


async def test_failed_write_warns_but_keeps_the_turn_s_result(memory, context):
    """A post-hoc write failing must not destroy a result the agent already produced (§10),
    but it must never be silent either (§77)."""
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    memory.fail_observation = True

    async def agent(payload):
        return "answer"

    result = await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert result.data == "answer"
    assert any(w.code == "MEMORY_WRITE_FAILED" for w in result.warnings)


async def test_fail_closed_propagates_a_failed_write(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False, "failure_mode": "fail_closed"}},
    )
    memory.fail_observation = True

    async def agent(payload):
        return "answer"

    with pytest.raises(ConnectionError):
        await harness.wrap(agent, agent_id="inv")("q", context=context)
