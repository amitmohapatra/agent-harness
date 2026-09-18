"""Failure injection (§77, §83): every dependency fails in turn, the business result holds.

One test per dependency, each asserting the documented behaviour rather than "it didn't
crash": what the caller gets back, what warning is attached, and what still got recorded.
"""

from __future__ import annotations

import asyncio

import pytest

from universal_agent_harness import AgentHarness, AgentResult, CollectingEvaluationSink


class BrokenTelemetry:
    """A telemetry backend that fails on every call."""

    name = "broken"

    def start_span(self, name, *, kind="internal", attributes=None):
        raise RuntimeError("telemetry backend is down")

    def record_event(self, name, attributes=None):
        raise RuntimeError("telemetry backend is down")

    def record_metric(self, name, value, *, unit="", attributes=None):
        raise RuntimeError("telemetry backend is down")

    def flush(self, timeout_seconds=5.0):
        raise RuntimeError("telemetry backend is down")


async def test_a_completely_broken_telemetry_backend_does_not_fail_the_agent(memory, context):
    from universal_agent_harness.telemetry.composite import CompositeTelemetryProvider

    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        telemetry=CompositeTelemetryProvider([BrokenTelemetry()]),
        config={"memory": {"writeback": False}},
    )

    async def agent(payload):
        return "business value"

    assert (await harness.wrap(agent, agent_id="inv")("q", context=context)).data == "business value"


async def test_memory_read_and_write_both_failing(faulty_memory, context):
    """Both directions cut at the socket: the read and the write really fail to connect."""
    faulty_memory.faults.drop.update({"/v1/context", "/v1/observations"})
    harness = AgentHarness(
        memory=faulty_memory,
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 10}},
    )

    async def agent(payload, runtime):
        assert runtime.memory_context is None
        return "answer anyway"

    result = await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert result.data == "answer anyway"
    codes = {w.code for w in result.warnings}
    assert codes == {"MEMORY_DEGRADED", "MEMORY_WRITE_FAILED"}


async def test_evaluation_sink_failure_is_contained(memory, context):
    class BrokenSink:
        async def emit(self, event):
            raise RuntimeError("evaluation pipeline is down")

    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        evaluation_sink=BrokenSink(),
        config={
            "memory": {"writeback": False},
            "evaluation_events": {"enabled": True, "synchronous": True},
        },
    )

    async def agent(payload):
        return "value"

    # A synchronous sink failure surfaces as an execution error only if the sink is
    # synchronous *and* raising; the default asynchronous mode never touches the caller.
    with pytest.raises(RuntimeError):
        await harness.wrap(agent, agent_id="inv")("q", context=context)

    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        evaluation_sink=BrokenSink(),
        config={"memory": {"writeback": False}, "evaluation_events": {"enabled": True}},
    )
    assert (await harness.wrap(agent, agent_id="inv")("q", context=context)).data == "value"
    await harness.drain()


async def test_artifact_store_failure_surfaces_as_an_error(memory, context):
    class BrokenStore:
        async def put(self, content, **kwargs):
            raise OSError("disk full")

        async def get(self, artifact_id):
            return None

    harness = AgentHarness(
        memory=memory,
        artifacts=BrokenStore(),
        defaults={"tenant_id": "acme"},
        config={"memory": {"writeback": False}},
    )

    async def agent(payload, runtime):
        with pytest.raises(OSError, match="disk full"):
            await runtime.artifacts.put("report")
        return "handled"

    assert (await harness.wrap(agent, agent_id="inv")("q", context=context)).data == "handled"


async def test_slow_memory_does_not_hold_the_turn_past_its_deadline(faulty_memory, context):
    faulty_memory.faults.stall["/v1/context"] = 10.0
    harness = AgentHarness(
        memory=faulty_memory,
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"writeback": False},
            "timeouts": {"default_seconds": 2.0, "memory_seconds": 0.05},
        },
    )

    async def agent(payload):
        return "in time"

    result = await asyncio.wait_for(
        harness.wrap(agent, agent_id="inv")("q", context=context), timeout=3
    )
    assert result.data == "in time"


async def test_everything_disabled_still_runs(context):
    """The degenerate configuration: no memory, no telemetry, no evaluation, no tools."""
    harness = AgentHarness(
        defaults={"tenant_id": "acme"},
        config={
            "memory": {"enabled": False},
            "telemetry": {"enabled": False},
            "artifacts": {"enabled": False},
            "evaluation_events": {"enabled": False},
        },
    )

    async def agent(payload, runtime):
        assert runtime.memory.enabled is False
        return AgentResult.ok("bare metal")

    assert (await harness.wrap(agent, agent_id="inv")("q", context=context)).data == "bare metal"


async def test_evaluation_events_survive_a_partially_broken_sink_set(memory, context):
    from universal_agent_harness.evaluation.events import CompositeEvaluationSink

    class BrokenSink:
        async def emit(self, event):
            raise RuntimeError("down")

    good = CollectingEvaluationSink()
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        evaluation_sink=CompositeEvaluationSink([BrokenSink(), good]),
        config={
            "memory": {"writeback": False},
            "evaluation_events": {"enabled": True, "synchronous": True},
        },
    )

    async def agent(payload):
        return "value"

    await harness.wrap(agent, agent_id="inv")("q", context=context)
    assert len(good.events) == 1  # the healthy sink still received the event
