"""The contract every target keeps, one suite run against each of them (``BUILDERS``: the
function target, ``ReAct``, a ``create_agent`` graph, Deep Agents, the OpenAI Agents SDK and
the Claude Agent SDK driving the scripted CLI):

* the run's records land in the memory service — each tool call, the transcript, the run's
  outcome — and a write the service refuses is never only a warning (counted, traced,
  reported on the run, logged);
* a model call that stalls ends the run with a clear ``TIMEOUT`` step error
  (``MODEL_TIMEOUT``), recorded on the run and in its trace — not a generic reply, not a hang.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings, tool
from trellis.contracts import ErrorCategory, RunEvent, RunEventType, RunStatus
from trellis.harness import runtime as runtime_module
from trellis.harness import telemetry

TARGETS = sorted(BUILDERS)


@tool(side_effects="read")
def lookup(sku: str) -> str:
    """Units of a SKU in stock."""
    return f"{sku}: 3 units"


async def _run(
    h: Harness,
    framework: str,
    tmp: Path,
    *,
    stall: float = 0.0,
    events: list[RunEvent] | None = None,
) -> Any:
    plan = [("lookup", {"sku": "A-1"})]
    target, tools = await BUILDERS[framework](h, [lookup], tmp, plan, stall=stall)
    agent = arm(h.wrap(target, id=f"contract-{framework}"), tools)
    if events is None:
        return await agent.run("How many A-1?", user="u1")
    async for event in agent.stream("How many A-1?", user="u1"):
        events.append(event)
    record = await h.runs.get(events[0].run_id, tenant=await h._tenant())
    return record


@pytest.mark.parametrize("framework", TARGETS)
async def test_the_records_land_for_every_target(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, framework: str
) -> None:
    result = await _run(memory_harness, framework, tmp_path)
    assert result.status is RunStatus.SUCCESS, result.error
    await memory_harness.writes.drain()
    calls = memory_service.calls
    recorded = [c.body for c in calls if c.name == "record_tool"]
    assert [r["tool"] for r in recorded] == ["lookup"]
    assert recorded[0]["status"] == "ok" and recorded[0]["latency_ms"] is not None
    transcript = [c.body for c in calls if c.name == "messages"]
    assert transcript and transcript[0]["messages"][0]["content"] == "How many A-1?"
    outcome = [c.body for c in calls if c.name == "feedback" and c.body.get("source") == "system"]
    assert [o["verdict"] for o in outcome] == ["confirm"]
    assert memory_harness.writes.failed == memory_harness.writes.spooled == 0


@pytest.fixture
def first_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> pytest.LogCaptureFixture:
    """The log, as a fresh process logs: each kind of failed write's first failure an error."""
    from trellis.harness import writes

    monkeypatch.setattr(writes, "_ERRORS_SAID", set())
    return caplog


@pytest.mark.parametrize("framework", TARGETS)
async def test_a_refused_record_is_never_only_a_warning(
    memory_service: FakeMemoryService,
    tmp_path: Path,
    framework: str,
    spans: Any,
    first_errors: pytest.LogCaptureFixture,
) -> None:
    """The write fails after the run answered (the records are written in the background):
    the run is not failed for it, and the failure is counted, in the run's trace and logged as
    an error (once per process and kind of write: ``writes.py``)."""
    memory_service.fail = {"record_tool"}
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        result = await _run(h, framework, tmp_path)
        await h.writes.drain()
        assert result.status is RunStatus.SUCCESS  # the run is not failed for it
        assert h.writes.failed == 1  # counted
    traced = [s for s in spans.get_finished_spans() if s.name == "write_failed memory.record_tool"]
    assert traced and traced[0].status.status_code.name == "ERROR"  # in the run's trace
    assert traced[0].context.trace_id == telemetry.trace_id_of(result.run_id)
    logged = first_errors.records
    errors = [r for r in logged if r.levelname == "ERROR" and "record_tool" in r.message]
    assert len(errors) == 1


@pytest.mark.parametrize("framework", TARGETS)
async def test_a_stalled_model_call_ends_the_run_with_a_timeout_step_error(
    harness: Harness,
    tmp_path: Path,
    framework: str,
    monkeypatch: pytest.MonkeyPatch,
    spans: Any,
) -> None:
    monkeypatch.setattr(runtime_module, "STALL_SECONDS", 0.2)
    events: list[RunEvent] = []
    record = await _run(harness, framework, tmp_path, stall=3.0, events=events)
    assert record.status is RunStatus.ERROR
    error = record.error
    assert error is not None and error.code == "MODEL_TIMEOUT"
    assert error.category is ErrorCategory.TIMEOUT and error.retryable
    assert "the step stalled" in error.message
    said = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any(d.get("code") == "model_timeout" for d in said)
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED]
    assert finished[-1].error is not None and finished[-1].error.code == "MODEL_TIMEOUT"
    traced = [s for s in spans.get_finished_spans() if s.name == "model_timeout"]
    assert traced and traced[0].status.status_code.name == "ERROR"


@pytest.mark.parametrize("framework", TARGETS)
async def test_a_slow_model_within_its_bound_is_no_stall(
    harness: Harness, tmp_path: Path, framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "STALL_SECONDS", 5.0)
    result = await _run(harness, framework, tmp_path, stall=0.05)
    assert result.status is RunStatus.SUCCESS, result.error
