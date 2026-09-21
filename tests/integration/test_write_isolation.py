"""One failing write must not discard the others.

The writes for a turn used to be a single sequence of awaits inside one ``try``, so the
first failure dropped everything after it. Measured live: a turn with ``record_messages``
enabled wrote one message, hit a scope validation error on the second, and silently lost the
question, the answer and every claim — while reporting SUCCESS with no warnings, because the
failure happened on the writeback path after the result was returned.
"""

from __future__ import annotations

from universal_agent_harness import AgentHarness, AgentResponse, Claim


async def test_a_failing_message_write_does_not_lose_the_observations(faulty_memory, context):
    """The message write is cut at the socket; the observations must still land.

    Cutting one path rather than relying on a validation quirk keeps the test about the
    invariant — writes are independent — instead of about any particular error.
    """
    memory = faulty_memory
    memory.faults.drop.add("/v1/messages")
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"writeback": False, "record_messages": True},
            "timeouts": {"memory_seconds": 10},
        },
    )

    async def agent(payload):
        return AgentResponse.ok(
            "reorder 50 units",
            claims=[Claim(claim_id="c1", text="SKU-1 cover is below ten days")],
        )

    result = await harness.wrap(agent, agent_id="inv")("how much stock?", context=context)

    # the turn still succeeded and the caller was told something was lost
    assert result.data == "reorder 50 units"
    assert any(w.code == "MEMORY_WRITE_FAILED" for w in result.warnings)

    # ...and the writes that *could* land, did
    contents = [o["content"] for o in memory.observations]
    assert "how much stock?" in contents, "the question survived the message failure"
    assert "reorder 50 units" in contents, "the answer survived"
    assert "SKU-1 cover is below ten days" in contents, "the claim survived"
    assert memory.of("runs.outcome"), "the run was still labelled"
    await harness.aclose()


async def test_the_error_names_what_was_lost(memory, context):
    from universal_agent_harness.interceptors.memory import MemoryWriteError

    failures = [
        ("message.user", RuntimeError("scope rejected")),
        ("observe[EVENT]", RuntimeError("down")),
    ]
    error = MemoryWriteError(failures)
    assert "2 memory write(s) failed" in str(error)
    assert "message.user" in str(error) and "observe[EVENT]" in str(error)
    assert error.failures == failures


async def test_all_writes_landing_raises_nothing(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )

    async def agent(payload):
        return "answered"

    result = await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert result.warnings == []
    await harness.aclose()
