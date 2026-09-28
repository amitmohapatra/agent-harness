"""The parts of the Temporal adapters that need no Temporal at all.

The interesting half of "a workflow per run" is the state machine, and a state machine that
can only be tested against a cluster is a state machine nobody tests. So the transitions, the
cadence mapping, the tenant filter and both adapters' degradation paths run here with no
server, no worker and no task queue. :mod:`test_workflow` is the other half: the same machine
inside a real workflow.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from trellis.contracts.errors import AgentError, ConfigurationError, ErrorCategory
from trellis.contracts.ports import RunStore, Scheduler
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

from trellis.harness_temporal import (
    ENDINGS_WITH_ERROR,
    Cadence,
    RunEnding,
    RunState,
    RunTransitionError,
    TemporalRunStore,
    TemporalRunsUnavailable,
    TemporalScheduler,
    is_manual,
    parse_cadence,
    temporalio_version,
)
from trellis.harness_temporal.client import TemporalConnection
from trellis.harness_temporal.runs import _claims_tenant
from trellis.harness_temporal.state import MAX_REFUSALS

TENANT = "acme"


def start(run_id: str = "run_1", *, tenant: str = TENANT) -> RunStart:
    return RunStart(run_id=run_id, tenant_id=tenant, agent_id="refunds", input="why?")


def question(run_id: str = "run_1", *, tenant: str = TENANT) -> Interrupt:
    return Interrupt(
        tenant_id=tenant, run_id=run_id, reason=InterruptReason.QUESTION, question="which order?"
    )


# --------------------------------------------------------------------------- state machine


def test_a_run_goes_started_paused_resumed_finished() -> None:
    state = RunState(start())
    assert state.record.status is RunStatus.RUNNING
    assert state.record.attempt == 1
    assert not state.done

    asked = question()
    paused = state.pause(asked)
    assert paused.status is RunStatus.PAUSED
    assert paused.awaiting == asked, "a paused run carries the question it is waiting on"

    resumed = state.resume(
        InterruptResolution(
            interrupt_id=asked.interrupt_id, run_id="run_1", decision=InterruptDecision.ANSWER
        )
    )
    assert resumed.status is RunStatus.RUNNING
    assert resumed.awaiting is None
    assert resumed.attempt == 2, "resuming is the next attempt of the same run"

    done = state.finish(RunStatus.SUCCESS, output="issued")
    assert done.status is RunStatus.SUCCESS
    assert done.output == "issued"
    assert state.done


def test_the_workflow_id_is_the_run_id_even_when_the_start_says_otherwise() -> None:
    """A schedule firing: the server appends the nominal time to the action's id, so the
    firing's workflow id is the run id and the schedule's template id is not."""
    state = RunState(start("sch_nightly"), run_id="sch_nightly-2026-09-28T06:00:00Z")
    assert state.record.run_id == "sch_nightly-2026-09-28T06:00:00Z"


def test_a_resolution_for_another_interrupt_is_refused() -> None:
    state = RunState(start())
    state.pause(question())
    with pytest.raises(RunTransitionError, match="different interrupt"):
        state.resume(
            InterruptResolution(
                interrupt_id="int_somebody_else",
                run_id="run_1",
                decision=InterruptDecision.ANSWER,
            )
        )
    assert state.record.status is RunStatus.PAUSED, "a refused answer leaves the run waiting"


def test_a_run_that_is_not_waiting_cannot_be_resumed() -> None:
    state = RunState(start())
    with pytest.raises(RunTransitionError, match="not waiting"):
        state.resume(
            InterruptResolution(
                interrupt_id="int_x", run_id="run_1", decision=InterruptDecision.ANSWER
            )
        )


def test_an_interrupt_from_another_run_or_tenant_is_refused() -> None:
    state = RunState(start())
    with pytest.raises(RunTransitionError, match="another run"):
        state.pause(question("run_2"))
    with pytest.raises(RunTransitionError, match="another run"):
        state.pause(question(tenant="other"))


def test_a_second_ending_is_the_same_ending() -> None:
    """A retried transition must not turn a recorded success into a failure."""
    state = RunState(start())
    state.finish(RunStatus.SUCCESS, output="issued")
    again = state.end(
        RunEnding(status=RunStatus.ERROR, error=AgentError(code="LATE", message="late"))
    )
    assert again.status is RunStatus.SUCCESS and again.output == "issued"
    assert again.error is None


def test_a_finished_run_cannot_pause_and_a_non_ending_cannot_finish() -> None:
    state = RunState(start())
    state.finish(RunStatus.SUCCESS)
    with pytest.raises(RunTransitionError, match="cannot pause"):
        state.pause(question())
    fresh = RunState(start())
    with pytest.raises(RunTransitionError, match="not an ending"):
        fresh.finish(RunStatus.RUNNING)


def test_an_error_travels_only_with_an_ending_that_carries_one() -> None:
    failed = RunState(start())
    failed.end(
        RunEnding(
            status=RunStatus.ERROR,
            error=AgentError(
                code="UPSTREAM", category=ErrorCategory.DEPENDENCY, message="upstream 500"
            ),
        )
    )
    assert failed.record.error is not None

    cancelled = RunState(start())
    cancelled.end(
        RunEnding(status=RunStatus.CANCELLED, error=AgentError(code="IGNORED", message="ignored"))
    )
    assert cancelled.record.error is None, "the contract says a CANCELLED run carries no error"


def test_which_endings_carry_an_error_is_the_contracts_answer_not_a_copy_of_it() -> None:
    """``ENDINGS_WITH_ERROR`` exists because ``RunRecord`` refuses a record that gets this
    wrong, and a hand-maintained copy of somebody else's rule drifts in silence — the first
    symptom would be a validation error on a record this adapter built itself.

    So the truth is derived by construction: ask the contract, for every final status, whether
    it accepts an error. A new failing status in ``trellis-contracts`` fails here instead.
    """
    error = AgentError(code="X", category=ErrorCategory.DEPENDENCY, message="x")
    accepted = set()
    for status in RunStatus:
        if not status.final:
            continue
        try:
            RunRecord(run_id="r", tenant_id=TENANT, agent_id="a", status=status, error=error)
        except ValidationError:
            continue
        accepted.add(status)
    assert accepted == set(ENDINGS_WITH_ERROR), (
        "the adapter's set and the contract's validator disagree about which endings carry "
        f"an error: contract accepts {sorted(s.value for s in accepted)}"
    )


def test_refusals_are_remembered_and_bounded() -> None:
    state = RunState(start())
    for index in range(MAX_REFUSALS + 10):
        state.refuse(RunTransitionError(f"nope {index}"))
    assert len(state.refusals) == MAX_REFUSALS
    assert state.refusals[-1] == f"nope {MAX_REFUSALS + 9}", "newest last"
    assert state.refusals is not state.refusals, "a copy: a query result cannot be mutated"


# --------------------------------------------------------------------------- cadence


@pytest.mark.parametrize(
    ("cadence", "cron"),
    [
        (Cadence.HOURLY, "0 * * * *"),
        (Cadence.DAILY, "0 0 * * *"),
        (Cadence.WEEKLY, "0 0 * * 1"),
        (Cadence.WEEKDAYS, "0 0 * * 1-5"),
    ],
)
def test_every_named_bucket_maps_to_one_cron_expression(cadence: Cadence, cron: str) -> None:
    spec = parse_cadence(cadence.value, timezone="Europe/Berlin")
    assert list(spec.cron_expressions) == [cron]
    assert spec.time_zone_name == "Europe/Berlin"


def test_manual_is_a_schedule_that_never_fires_on_its_own() -> None:
    spec = parse_cadence("manual")
    assert not spec.cron_expressions and not spec.calendars and not spec.intervals
    assert is_manual("MANUAL") and not is_manual("daily")


def test_a_cron_expression_is_passed_through() -> None:
    assert list(parse_cadence("15 9 * * 1-5").cron_expressions) == ["15 9 * * 1-5"]
    assert list(parse_cadence("0 30 9 * * *").cron_expressions) == ["0 30 9 * * *"]


@pytest.mark.parametrize(
    "cadence",
    ["* * * *", "every tuesday", "*/5 * * * * * * *"],
)
def test_a_cadence_that_is_not_a_cron_expression_is_refused_with_its_reason(cadence: str) -> None:
    with pytest.raises(ConfigurationError, match="cron expression"):
        parse_cadence(cadence)


def test_a_cadence_under_the_floor_is_refused_at_create_time_not_discovered_as_a_bill() -> None:
    with pytest.raises(ConfigurationError, match="more often than every 60s"):
        parse_cadence("*/5 * * * * *")
    with pytest.raises(ConfigurationError, match="more often than every 60s"):
        parse_cadence("* * * * * *")
    assert parse_cadence("*/5 * * * * *", floor_seconds=5), "a floor raised deliberately"


# --------------------------------------------------------------------------- ports and shape


def test_both_adapters_satisfy_the_ports_they_claim() -> None:
    runs = TemporalRunStore("localhost:7233")
    scheduler = TemporalScheduler("localhost:7233")
    assert isinstance(runs, RunStore)
    assert isinstance(scheduler, Scheduler)


def test_a_task_queue_is_not_optional() -> None:
    """A worker polls a named queue; an empty one is a run nothing will ever keep alive."""
    with pytest.raises(ValueError, match="task_queue"):
        TemporalRunStore("localhost:7233", task_queue="")
    with pytest.raises(ValueError, match="task_queue"):
        TemporalScheduler("localhost:7233", task_queue="")


def test_the_adapter_reports_the_installed_temporalio() -> None:
    from importlib.metadata import version

    assert temporalio_version() == version("temporalio")


# --------------------------------------------------------------------------- connection


def test_a_connection_needs_a_client_or_an_address() -> None:
    with pytest.raises(ValueError, match="client or a target"):
        TemporalConnection("")


async def test_a_connection_opens_once_and_only_releases_what_it_opened() -> None:
    """Concurrent first calls must share one connection, and a client somebody else passed in
    is theirs to manage."""
    connection = TemporalConnection("localhost:7233")
    assert not connection.connected
    sentinel = object()
    connection._client = sentinel  # type: ignore[assignment]
    assert connection.connected
    assert await connection.client() is sentinel
    await connection.aclose()
    assert not connection.connected, "the connection it opened is the one it drops"

    borrowed = TemporalConnection("localhost:7233")
    borrowed._owns_client = False
    borrowed._client = sentinel  # type: ignore[assignment]
    await borrowed.aclose()
    assert borrowed.connected, "a borrowed client is not ours to close"


# --------------------------------------------------------------------------- degradation


class Unreachable:
    """A connection to a cluster that is not there."""

    namespace = "default"

    async def client(self):
        raise OSError("connection refused")

    async def aclose(self) -> None:
        return None


async def test_an_unreachable_cluster_does_not_fail_the_turn() -> None:
    """Bookkeeping being unreachable is a worse reason to fail a customer's request than
    almost any other, so the record degrades and the run continues."""
    store = TemporalRunStore("localhost:7233", connection=Unreachable())  # type: ignore[arg-type]
    record = await store.started(start())
    assert record.run_id == "run_1" and record.status is RunStatus.RUNNING
    assert await store.get("run_1") is None
    assert await store.list_paused(TENANT) == []
    paused = await store.paused(question())
    assert paused.status is RunStatus.PAUSED, "the caller still gets the record it asked for"
    finished = await store.finished("run_1", RunStatus.SUCCESS, output="ok")
    assert finished.status is RunStatus.SUCCESS


async def test_required_true_inverts_that_and_says_so() -> None:
    store = TemporalRunStore(
        "localhost:7233",
        required=True,
        connection=Unreachable(),  # type: ignore[arg-type]
    )
    with pytest.raises(TemporalRunsUnavailable, match="start run_1"):
        await store.started(start())


async def test_an_empty_inbox_is_asked_for_nothing() -> None:
    store = TemporalRunStore("localhost:7233", connection=Unreachable())  # type: ignore[arg-type]
    assert await store.list_paused(TENANT, limit=0) == []


async def test_a_scheduler_that_cannot_reach_temporal_raises() -> None:
    """Unlike a run record, a schedule the user believes was saved must not be silently lost."""
    scheduler = TemporalScheduler(
        "localhost:7233",
        connection=Unreachable(),  # type: ignore[arg-type]
    )
    spec = ScheduleSpec(
        tenant_id=TENANT, agent_id="refunds", name="nightly", cadence="daily", on_behalf_of="u1"
    )
    with pytest.raises(OSError, match="connection refused"):
        await scheduler.create(spec)


# --------------------------------------------------------------------------- tenant filter


class Listed:
    def __init__(self, memo) -> None:
        self.id = "run_1"
        self._memo = memo

    async def memo(self):
        if isinstance(self._memo, Exception):
            raise self._memo
        return self._memo


async def test_the_memo_filter_asks_rather_than_admits() -> None:
    assert await _claims_tenant(Listed({"tenant_id": TENANT}), TENANT) is True
    assert await _claims_tenant(Listed({"tenant_id": "other"}), TENANT) is False
    assert await _claims_tenant(Listed({}), TENANT) is True, "no claim: ask the record"
    assert await _claims_tenant(Listed(RuntimeError("no memo")), TENANT) is True
