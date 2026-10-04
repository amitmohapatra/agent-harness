"""The in-process run store, ``LocalRuns``: the same behaviour as agent-runs (whose client,
``trellis.runs.RunsClient``, is tested in its SDK), and both are the harness's ``RunStore``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trellis.contracts import (
    AgentError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunStart,
    RunStatus,
    ScheduleSpec,
)
from trellis.harness.runs import LocalRuns
from trellis.runs import ConflictError, Lease, LeaseLostError, NotFoundError


def start(run_id: str = "run_1", agent: str = "a", tenant: str = "t") -> RunStart:
    return RunStart(run_id=run_id, tenant_id=tenant, agent_id=agent, user_id="u", input="hi")


def interrupt(run_id: str = "run_1") -> Interrupt:
    return Interrupt(
        interrupt_id=f"{run_id}.1.1",
        tenant_id="t",
        run_id=run_id,
        question="ok?",
        assignee="role:ops",
    )


def resolution(run_id: str = "run_1") -> InterruptResolution:
    return InterruptResolution(
        interrupt_id=f"{run_id}.1.1", run_id=run_id, decision=InterruptDecision.ANSWER, answer="yes"
    )


async def inbox(runs: LocalRuns, assignee: str | None) -> list[str]:
    listed = runs.iterate(status=RunStatus.PAUSED, assignee=assignee, tenant="t")
    return [summary.run_id async for summary in listed]


# --------------------------------------------------------------------------- in process
async def test_a_run_moves_through_the_contract_state_machine() -> None:
    runs = LocalRuns()
    assert (await runs.start(start())).status is RunStatus.RUNNING
    assert (await runs.start(start())).status is RunStatus.RUNNING  # idempotent
    paused = await runs.pause(interrupt(), checkpoint={"answers": {}})
    assert paused.status is RunStatus.PAUSED and paused.checkpoint == {"answers": {}}
    [waiting] = [s async for s in runs.iterate(status=RunStatus.PAUSED, assignee="role:ops")]
    assert (waiting.run_id, waiting.assignee, waiting.status) == (
        "run_1",
        "role:ops",
        paused.status,
    )
    assert await inbox(runs, "role:other") == []
    assert await inbox(runs, None) == ["run_1"]
    answer = resolution()
    resumed = await runs.resume(answer, tenant="t")
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2
    assert resumed.last_resolution == answer and resumed.checkpoint == {"answers": {}}
    done = await runs.finish("run_1", RunStatus.SUCCESS, output="ok", tenant="t")
    assert done.final and done.output == "ok" and done.checkpoint is None
    with pytest.raises(ConflictError) as refused:
        await runs.finish("run_1", RunStatus.ERROR)
    assert (refused.value.code, refused.value.status) == ("CONFLICT", 409)


async def test_an_answer_to_another_question_is_refused() -> None:
    runs = LocalRuns()
    await runs.start(start())
    await runs.pause(interrupt())
    wrong = resolution().model_copy(update={"interrupt_id": "run_1.1.9"})
    with pytest.raises(ConflictError):
        await runs.resume(wrong)


async def test_queued_runs_are_claimed_once_by_agent_and_leases_expire() -> None:
    runs = LocalRuns()
    await runs.start(start("run_a", agent="a"), queue=True)
    await runs.start(start("run_b", agent="b"), queue=True)
    claimed = await runs.claim("w1", ["b"], lease_seconds=30)
    assert claimed is not None and claimed.run.run_id == "run_b"
    assert claimed.run.status is RunStatus.RUNNING
    assert (claimed.lease.run_id, claimed.lease.worker_id) == ("run_b", "w1")
    assert await runs.claim("w2", ["b"], lease_seconds=30) is None
    lease = await runs.heartbeat("run_b", "w1", lease_seconds=30, tenant="t")
    assert isinstance(lease, Lease) and lease.worker_id == "w1"
    with pytest.raises(LeaseLostError) as lost:
        await runs.heartbeat("run_b", "w2", lease_seconds=30)
    assert not isinstance(lost.value, ConflictError)  # a lost lease is never a conflict
    assert (lost.value.code, lost.value.status) == ("LEASE_LOST", 409)
    lapsed = await runs.claim("w1", ["a"], lease_seconds=-1)  # a lease already over
    assert lapsed is not None
    await runs.heartbeat("run_a", "w1", lease_seconds=-1, checkpoint={"calls": {"k": ["done"]}})
    again = await runs.claim("w3", ["a"])
    assert again is not None and again.run.run_id == "run_a" and again.run.attempt == 2
    assert again.run.checkpoint == {"calls": {"k": ["done"]}}  # progress survives the lapse
    await runs.heartbeat("run_a", "w3")  # no checkpoint: the one there is kept
    kept = await runs.get("run_a")
    assert kept is not None and kept.checkpoint == {"calls": {"k": ["done"]}}


async def test_a_claim_for_a_tenant_takes_only_that_tenants_runs() -> None:
    runs = LocalRuns()
    await runs.start(start("run_other", tenant="other"), queue=True)
    assert await runs.claim("w", ["a"], tenant="t") is None
    claimed = await runs.claim("w", ["a"], tenant="other")
    assert claimed is not None and claimed.run.tenant_id == "other"


async def test_a_resumed_durable_run_goes_back_to_the_queue() -> None:
    runs = LocalRuns()
    await runs.start(start(), queue=True)
    await runs.claim("w", ["a"])
    await runs.pause(interrupt())
    assert (await runs.resume(resolution())).status is RunStatus.QUEUED
    claimed = await runs.claim("w", ["a"])
    assert claimed is not None and claimed.run.last_resolution is not None


async def test_a_cancel_ends_a_paused_run_and_a_lapsed_worker_cannot_write() -> None:
    runs = LocalRuns()
    await runs.start(start())
    await runs.pause(interrupt())
    cancel = resolution().model_copy(update={"decision": InterruptDecision.CANCEL})
    assert (await runs.resume(cancel)).status is RunStatus.CANCELLED
    await runs.start(start("run_2"), queue=True)
    await runs.claim("w1", ["a"])
    with pytest.raises(LeaseLostError):
        await runs.finish("run_2", RunStatus.SUCCESS, worker_id="w2")
    assert (await runs.finish("run_2", RunStatus.SUCCESS, worker_id="w1")).final


async def test_another_tenants_run_is_not_there() -> None:
    runs = LocalRuns()
    await runs.start(start())
    await runs.pause(interrupt())
    assert await runs.get("run_1", tenant="other") is None
    assert await runs.get("run_1", tenant="t") is not None
    assert await runs.get("run_1") is not None  # no tenant named: every tenant's
    with pytest.raises(NotFoundError) as missing:
        await runs.resume(resolution(), tenant="other")
    assert (missing.value.code, missing.value.status) == ("NOT_FOUND", 404)
    with pytest.raises(NotFoundError):
        await runs.finish("run_1", RunStatus.CANCELLED, tenant="other")
    assert [s async for s in runs.iterate(tenant="other")] == []


async def test_listing_filters_and_stops_after_its_pages() -> None:
    runs = LocalRuns()
    for n in range(5):
        await runs.start(start(f"run_{n}", agent="a" if n % 2 else "b"))
    await runs.finish("run_4", RunStatus.SUCCESS)
    newest_first = [s.run_id async for s in runs.iterate()]
    assert newest_first == ["run_4", "run_3", "run_2", "run_1", "run_0"]
    assert [s.run_id async for s in runs.iterate(agent_id="a")] == ["run_3", "run_1"]
    assert [s.run_id async for s in runs.iterate(status=RunStatus.SUCCESS)] == ["run_4"]
    assert [s.run_id async for s in runs.iterate(thread_id="run_2")] == []  # no thread given
    assert [s.run_id async for s in runs.iterate(parent_run_id="run_0")] == []
    assert len([s async for s in runs.iterate(limit=2, max_pages=2)]) == 4


async def test_a_due_schedule_queues_a_run_when_a_worker_claims() -> None:
    runs = LocalRuns()
    schedule = await runs.schedules.create(
        ScheduleSpec(
            tenant_id="t",
            agent_id="a",
            name="daily",
            cadence="0 6 * * *",
            timezone="Europe/Paris",
            on_behalf_of="u",
            input="report",
        )
    )
    assert schedule.next_fire_at is not None and schedule.next_fire_at > datetime.now(UTC)
    assert await runs.claim("w", ["a"]) is None
    runs._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    fired = await runs.claim("w", ["a"])
    assert fired is not None and fired.run.input == "report" and fired.run.user_id == "u"
    assert fired.run.metadata["schedule_id"] == schedule.schedule_id


async def test_local_schedules_upsert_on_agent_person_cadence_and_input() -> None:
    runs = LocalRuns()

    def spec(**changes: object) -> ScheduleSpec:
        fields = {"tenant_id": "t", "agent_id": "a", "name": "n", "cadence": "daily"}
        return ScheduleSpec(**{**fields, "on_behalf_of": "u", "input": {"x": 1}, **changes})

    first = await runs.schedules.create(spec())
    assert (await runs.schedules.create(spec(name="renamed"))).schedule_id == first.schedule_id
    assert (await runs.schedules.create(spec(input={"x": 2}))).schedule_id != first.schedule_id
    assert (await runs.schedules.create(spec(on_behalf_of="v"))).schedule_id != first.schedule_id


async def test_local_artifacts_are_kept_per_tenant() -> None:
    runs = LocalRuns()
    await runs.start(start())
    ref = await runs.artifacts.upload("run_1", b'{"a":1}', tenant="t")
    assert ref.size_bytes == 7 and ref.mime_type == "application/json"
    assert ref.checksum is not None and ref.checksum.startswith("sha256:")
    assert await runs.artifacts.download(ref.artifact_id, tenant="t") == b'{"a":1}'
    assert await runs.artifacts.download(ref.artifact_id) == b'{"a":1}'
    assert await runs.artifacts.download(ref.artifact_id, tenant="other") is None
    assert await runs.artifacts.download("art_missing", tenant="t") is None
    csv = await runs.artifacts.upload("run_1", b"a,b", mime_type="text/csv")
    assert csv.mime_type == "text/csv"
    with pytest.raises(NotFoundError):
        await runs.artifacts.upload("run_1", b"x", tenant="other")


async def test_a_failed_finish_carries_its_error() -> None:
    runs = LocalRuns()
    await runs.start(start())
    error = AgentError(code="Boom", message="it broke")
    failed = await runs.finish("run_1", RunStatus.ERROR, error=error, tenant="t")
    assert failed.error == error
    await runs.aclose()


async def test_a_resume_of_a_run_the_store_never_had_is_refused() -> None:
    with pytest.raises(NotFoundError, match="no run run_1"):
        await LocalRuns().resume(resolution())


async def test_a_manual_schedule_never_fires_on_its_own() -> None:
    runs = LocalRuns()
    manual = await runs.schedules.create(
        ScheduleSpec(tenant_id="t", agent_id="a", name="m", cadence="manual", on_behalf_of="u")
    )
    assert manual.next_fire_at is None
    assert await runs.claim("w", ["a"]) is None
