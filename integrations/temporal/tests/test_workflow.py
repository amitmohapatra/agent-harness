"""The same run state machine, inside a real workflow.

``WorkflowEnvironment.start_time_skipping()`` is a real Temporal test server — real signals,
real queries, real payload conversion, real determinism checks — with no cluster and no clock
to wait on. It downloads its server on first use, so every test here skips with a reason when
that is not possible; :mod:`test_state_and_cadence` and :mod:`test_scripted_client` are what
prove the adapters with nothing running at all.
"""

from __future__ import annotations

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from trellis.contracts.runs import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
    ScheduleSpec,
)

from trellis.harness_temporal import AgentRunWorkflow, TemporalRunStore, TemporalScheduler
from trellis.harness_temporal.state import RunEnding
from trellis.harness_temporal.workflow import (
    FINISH_SIGNAL,
    PAUSE_SIGNAL,
    RECORD_QUERY,
    REFUSALS_QUERY,
    RESUME_SIGNAL,
)

#: The test server is fetched on first use, which is slower than the suite's default budget.
pytestmark = [pytest.mark.temporal, pytest.mark.timeout(600)]

TENANT = "acme"
TASK_QUEUE = "trellis-runs-test"


@pytest.fixture
async def env():
    """A time-skipping Temporal test server, or a skip that says why there is none."""
    try:
        environment = await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(
            "temporalio's time-skipping test server could not start "
            f"({type(exc).__name__}: {exc}); it is downloaded on first use"
        )
    try:
        yield environment
    finally:
        await environment.shutdown()


@pytest.fixture
async def worker(env):
    """One worker for the run workflow. No activities: the workflow *is* the record."""
    async with Worker(env.client, task_queue=TASK_QUEUE, workflows=[AgentRunWorkflow]):
        yield env.client


def start(run_id: str) -> RunStart:
    return RunStart(run_id=run_id, tenant_id=TENANT, agent_id="refunds", input="where is it?")


def question(run_id: str) -> Interrupt:
    return Interrupt(
        tenant_id=TENANT,
        run_id=run_id,
        reason=InterruptReason.QUESTION,
        question="which order number?",
    )


async def test_a_run_is_a_workflow_and_its_transitions_are_signals(worker) -> None:
    handle = await worker.start_workflow(
        AgentRunWorkflow.run, start("run_wf_1"), id="run_wf_1", task_queue=TASK_QUEUE
    )

    # ``result_type`` is how a caller asks the converter for the contract rather than the
    # dict it serialised to; the adapter validates what it gets either way.
    running = await handle.query(RECORD_QUERY, result_type=RunRecord)
    assert running is not None and running.status is RunStatus.RUNNING
    assert running.run_id == "run_wf_1", "the workflow id is the run id"

    asked = question("run_wf_1")
    await handle.signal(PAUSE_SIGNAL, asked)
    paused = await handle.query(RECORD_QUERY, result_type=RunRecord)
    assert paused.status is RunStatus.PAUSED
    assert paused.awaiting is not None and paused.awaiting.question == "which order number?"

    await handle.signal(
        RESUME_SIGNAL,
        InterruptResolution(
            interrupt_id=asked.interrupt_id,
            run_id="run_wf_1",
            decision=InterruptDecision.ANSWER,
            answer="A-1029",
        ),
    )
    resumed = await handle.query(RECORD_QUERY, result_type=RunRecord)
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2

    await handle.signal(FINISH_SIGNAL, RunEnding(status=RunStatus.SUCCESS, output="delivered"))
    final = await handle.result()
    assert final.status is RunStatus.SUCCESS and final.output == "delivered"


async def test_a_signal_that_does_not_apply_is_refused_not_a_wedged_run(worker) -> None:
    """A raising signal handler fails the workflow task, which Temporal retries forever. So a
    refusal is recorded on the run and the run keeps answering the right questions."""
    handle = await worker.start_workflow(
        AgentRunWorkflow.run, start("run_wf_2"), id="run_wf_2", task_queue=TASK_QUEUE
    )
    await handle.signal(
        RESUME_SIGNAL,
        InterruptResolution(
            interrupt_id="int_never_asked",
            run_id="run_wf_2",
            decision=InterruptDecision.ANSWER,
        ),
    )
    refusals = await handle.query(REFUSALS_QUERY)
    assert refusals and "not waiting" in refusals[0]
    assert (await handle.query(RECORD_QUERY, result_type=RunRecord)).status is RunStatus.RUNNING

    await handle.signal(FINISH_SIGNAL, RunEnding(status=RunStatus.SUCCESS))
    assert (await handle.result()).status is RunStatus.SUCCESS


async def test_the_run_store_drives_that_workflow_through_the_port(worker) -> None:
    """Every ``RunStore`` method against a real server: start, pause, resume, finish, get."""
    store = TemporalRunStore(worker, task_queue=TASK_QUEUE)

    opened = await store.started(start("run_wf_3"))
    assert opened.status is RunStatus.RUNNING

    again = await store.started(start("run_wf_3"))
    assert again.run_id == "run_wf_3", "a retried start finds its own run (USE_EXISTING)"

    asked = question("run_wf_3")
    paused = await store.paused(asked)
    assert paused.status is RunStatus.PAUSED and paused.awaiting is not None

    fetched = await store.get("run_wf_3")
    assert fetched is not None and fetched.status is RunStatus.PAUSED

    resumed = await store.resumed(
        InterruptResolution(
            interrupt_id=asked.interrupt_id, run_id="run_wf_3", decision=InterruptDecision.APPROVE
        )
    )
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2

    finished = await store.finished("run_wf_3", RunStatus.SUCCESS, output="refunded")
    assert finished.status is RunStatus.SUCCESS and finished.output == "refunded"


async def test_a_run_that_was_never_opened_is_not_a_run(worker) -> None:
    store = TemporalRunStore(worker, task_queue=TASK_QUEUE)
    assert await store.get("run_that_never_existed") is None


async def test_a_schedule_starts_the_same_workflow_with_the_same_request(worker) -> None:
    """Temporal Schedules need a server with schedule support; the time-skipping test server
    does not always have one, so this skips rather than claiming a pass it did not get."""
    scheduler = TemporalScheduler(worker, task_queue=TASK_QUEUE)
    spec = ScheduleSpec(
        tenant_id=TENANT,
        agent_id="refunds",
        name="nightly refunds sweep",
        cadence="weekdays",
        timezone="Europe/Berlin",
        on_behalf_of="user-7",
        input="sweep",
    )
    try:
        created = await scheduler.create(spec)
    except Exception as exc:  # pragma: no cover - server-dependent
        pytest.skip(f"this Temporal test server has no schedule support ({type(exc).__name__})")
    try:
        assert created.cadence == "weekdays" and created.on_behalf_of == "user-7"
        assert created.timezone == "Europe/Berlin"

        listed = await scheduler.list_for_tenant(TENANT)
        assert [s.schedule_id for s in listed] == [created.schedule_id]
        assert await scheduler.list_for_tenant("someone-else") == []

        disabled = await scheduler.set_enabled(created.schedule_id, False)
        assert disabled.enabled is False
    finally:
        await scheduler.delete(created.schedule_id)
        assert await scheduler.get(created.schedule_id) is None
        # Deleting what is already gone is the outcome the caller asked for, not an error.
        await scheduler.delete(created.schedule_id)
