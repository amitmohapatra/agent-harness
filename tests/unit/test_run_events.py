"""The run event stream: ordered, attempt-aware, redacted, and never in the run's way."""

from __future__ import annotations

import asyncio

import pytest
from trellis.contracts import AgentExecutionContext, RunEventType, RunOutcome

from trellis.harness.events import (
    CollectingEventSink,
    CompositeEventSink,
    FilteringEventSink,
    RunEventStream,
)
from trellis.harness.telemetry.redaction import REDACTED, DefaultRedactor

CTX = AgentExecutionContext.create(
    tenant_id="acme", user_id="u1", agent_id="ref", thread_id="thr_1"
)


class _Broken:
    async def publish(self, event) -> None:
        raise RuntimeError("sink down")


async def test_events_are_numbered_per_attempt_and_carry_the_run() -> None:
    sink = CollectingEventSink()
    stream = RunEventStream(CTX, [sink])
    assert stream.enabled
    await stream.emit(RunEventType.RUN_STARTED, agent_id="ref")
    await stream.emit(RunEventType.STEP_STARTED, step="agent")
    stream.next_attempt()
    await stream.emit(RunEventType.STEP_STARTED, step="agent")
    finished = await stream.emit(RunEventType.RUN_FINISHED, outcome="success")
    assert finished.outcome is RunOutcome.SUCCESS
    seq = [(e.attempt, e.sequence, e.type) for e in sink.events]
    assert seq == [
        (1, 0, RunEventType.RUN_STARTED),
        (1, 1, RunEventType.STEP_STARTED),
        (2, 0, RunEventType.STEP_STARTED),
        (2, 1, RunEventType.RUN_FINISHED),
    ]
    assert all(e.run_id == CTX.agent_run_id and e.thread_id == "thr_1" for e in sink.events)


async def test_an_invalid_event_is_a_harness_bug_and_a_broken_sink_is_skipped() -> None:
    sink = CollectingEventSink()
    stream = RunEventStream(CTX, [_Broken(), sink])
    with pytest.raises(ValueError, match="RUN_FINISHED names its outcome"):
        await stream.emit(RunEventType.RUN_FINISHED)
    with pytest.raises(ValueError, match="names its message"):
        await stream.emit(RunEventType.TEXT_MESSAGE_END)
    event = await stream.emit(RunEventType.RUN_STARTED)
    assert sink.events == [event] and stream.sequence == 1
    assert not RunEventStream(CTX).enabled


async def test_payloads_are_redacted_like_spans_before_any_sink_sees_them() -> None:
    sink = CollectingEventSink()
    stream = RunEventStream(CTX, [sink], redactor=DefaultRedactor())
    await stream.emit(
        RunEventType.TOOL_CALL_ARGS,
        tool_call_id="c1",
        args={"api_key": "sk-live-1234567890abcdef", "order_id": 91, "note": "ok"},
    )
    args = sink.events[0].data["args"]
    assert args["api_key"] == REDACTED and args["order_id"] == 91 and args["note"] == "ok"
    bare = RunEventStream(CTX, [CollectingEventSink()])
    plain = await bare.emit(RunEventType.TOOL_CALL_ARGS, tool_call_id="c1", args={"token": "t"})
    assert plain.data["args"] == {"token": "t"}  # no redactor: the harness's own concern


async def test_subscribers_are_keyed_by_tenant_and_run_and_replayed_when_late() -> None:
    sink = CollectingEventSink()
    stream = RunEventStream(CTX, [sink])
    await stream.emit(RunEventType.RUN_STARTED)
    late = sink.subscribe("acme", CTX.agent_run_id)  # subscribed after the first event: replayed
    assert (await late.get()).type is RunEventType.RUN_STARTED
    other_tenant = sink.subscribe("globex", CTX.agent_run_id)  # same run id, another tenant
    fresh = sink.subscribe("acme", CTX.agent_run_id, replay=False)
    await stream.emit(RunEventType.RUN_FINISHED, outcome="success")
    assert (await late.get()).type is RunEventType.RUN_FINISHED
    assert await late.get() is None  # the run is over
    assert (await fresh.get()).type is RunEventType.RUN_FINISHED  # new events only
    assert other_tenant.empty()
    sink.unsubscribe("acme", CTX.agent_run_id, late)
    sink.unsubscribe("acme", CTX.agent_run_id, fresh)
    assert sink.for_run(CTX.agent_run_id)[-1].type is RunEventType.RUN_FINISHED
    assert sink.for_run(CTX.agent_run_id, tenant_id="globex") == []


async def test_the_collecting_sink_is_bounded() -> None:
    sink = CollectingEventSink(max_runs=2, max_events=3)
    for n in range(3):
        ctx = AgentExecutionContext.create(tenant_id="acme", agent_id="ref", turn_id=f"t{n}")
        stream = RunEventStream(ctx, [sink])
        for _ in range(5):
            await stream.emit(RunEventType.STEP_STARTED, step="agent")
    assert len(sink.events) == 6  # two runs remembered, three events each
    assert all(e.sequence >= 2 for e in sink.events)  # the oldest events of a run went first


async def test_composite_and_filtering_sinks() -> None:
    finished_only = CollectingEventSink()
    everything = CollectingEventSink()
    stream = RunEventStream(
        CTX,
        [
            CompositeEventSink(
                [_Broken(), everything, FilteringEventSink(finished_only, ["RUN_FINISHED"])]
            )
        ],
    )
    await stream.emit(RunEventType.RUN_STARTED)
    await stream.emit(RunEventType.RUN_FINISHED, outcome="error")
    assert everything.types() == [RunEventType.RUN_STARTED, RunEventType.RUN_FINISHED]
    assert finished_only.types() == [RunEventType.RUN_FINISHED]
    with pytest.raises(ValueError):
        FilteringEventSink(finished_only, ["NOPE"])
    await asyncio.sleep(0)
