"""The harness labels its own runs, so tool memory does not have to guess.

Without a label the service treats a run as a *weak positive* only once it is older than
its window — hours — and never learns anything from a turn that failed for a reason other
than a failing tool call. The harness knows at the end of the turn; these tests assert it
says so, on both paths.
"""

from __future__ import annotations

import pytest

from universal_agent_harness import AgentHarness, AgentResponse, MemoryPolicy


async def test_a_successful_turn_is_labelled_a_success(harness, memory, context):
    async def agent(payload):
        return "answered"

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    await harness.drain()

    labels = memory.of("runs.outcome")
    assert len(labels) == 1
    assert labels[0]["success"] is True
    assert labels[0]["run_id"]  # the run it describes, not the parent


async def test_a_failed_turn_is_labelled_a_failure(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )

    async def agent(payload):
        raise RuntimeError("the tool chain did not work out")

    with pytest.raises(RuntimeError):
        await harness.wrap(agent, agent_id="inv")("q", context=context)
    await harness.drain()

    labels = memory.of("runs.outcome")
    assert len(labels) == 1, "a raising turn must still be labelled"
    assert labels[0]["success"] is False
    assert "UNKNOWN" in (labels[0]["note"] or "")


async def test_an_error_returned_as_a_result_is_also_a_failure(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )

    async def agent(payload):
        raise ValueError("bad input")

    result = await harness.wrap(agent, agent_id="inv", error_mode="result")("q", context=context)
    assert result.status == "ERROR"
    await harness.drain()

    labels = memory.of("runs.outcome")
    assert labels and all(label["success"] is False for label in labels)


async def test_the_label_can_be_turned_off(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )

    async def agent(payload):
        return "answered"

    await harness.wrap(agent, agent_id="inv", memory_policy=MemoryPolicy(record_outcome=False))(
        "q", context=context
    )
    await harness.drain()
    assert memory.of("runs.outcome") == []


async def test_the_label_survives_a_turn_that_wrote_no_observations(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={
            "memory": {
                "writeback": False,
                "observe_input": False,
                "observe_output": False,
                "observe_claims": False,
            }
        },
    )

    async def agent(payload):
        return AgentResponse.ok(None)

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    await harness.drain()
    assert len(memory.of("runs.outcome")) == 1


async def test_the_label_does_not_block_a_turn_that_wrote_nothing(memory, context):
    """The label rides the writeback queue like everything else.

    Recording it inline was a real regression: an agent that observes nothing — a LangGraph
    node returning a state update, say — paid an HTTP round trip before its result came back,
    which is precisely the cost writeback exists to avoid.
    """
    import asyncio

    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        # retrieval off so the measurement is about the write, not the read before it;
        # writeback stays on, as in production
        config={
            "memory": {
                "retrieve_before": False,
                "observe_input": False,
                "observe_output": False,
                "observe_claims": False,
            }
        },
    )

    async def agent(payload):
        return AgentResponse.ok(None)

    started = asyncio.get_running_loop().time()
    await harness.wrap(agent, agent_id="inv")("q", context=context)
    elapsed = asyncio.get_running_loop().time() - started

    assert memory.of("runs.outcome") == [], "the label must not have been written yet"
    assert elapsed < 0.05, f"the turn waited {elapsed * 1000:.0f} ms for a memory write"

    await harness.drain()
    assert len(memory.of("runs.outcome")) == 1, "...and it lands once the queue drains"
