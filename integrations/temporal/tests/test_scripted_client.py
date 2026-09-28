"""The two reads that are too expensive to prove against a cluster, proved against a script.

``list_paused`` and ``list_for_tenant`` are the adapters' only *assembled* answers: a bound on
how much they ask, a filter on whose runs they return, and a shape read back out of a memo.
Driving those with a scripted client is not a shortcut — it is the only way to assert the
bound (a cluster with three workflows cannot show that the thousandth is not queried) and the
isolation (two tenants' runs must be present at once for the filter to mean anything).

The script is shaped after the real objects: ``memo()`` is a coroutine on all of Temporal's
descriptions, ``list_workflows`` returns an async iterator directly and ``list_schedules`` is a
coroutine returning one. :mod:`test_workflow` proves the same adapters against a real server.
"""

from __future__ import annotations

from typing import Any

import pytest
from temporalio.client import ScheduleActionStartWorkflow, ScheduleState
from temporalio.service import RPCError, RPCStatusCode
from trellis.contracts.runs import (
    Interrupt,
    InterruptReason,
    RunRecord,
    RunStatus,
    Schedule,
    ScheduleSpec,
)

from trellis.harness_temporal import TemporalRunStore, TemporalScheduler
from trellis.harness_temporal.scheduler import SPEC_MEMO_KEY
from trellis.harness_temporal.workflow import RUN_WORKFLOW

TENANT = "acme"


def record(run_id: str, *, tenant: str = TENANT, status: RunStatus = RunStatus.PAUSED) -> RunRecord:
    awaiting = (
        Interrupt(
            tenant_id=tenant, run_id=run_id, reason=InterruptReason.QUESTION, question="which?"
        )
        if status is RunStatus.PAUSED
        else None
    )
    return RunRecord(
        run_id=run_id,
        tenant_id=tenant,
        agent_id="refunds",
        status=status,
        awaiting=awaiting,
    )


# --------------------------------------------------------------------------- run store script


class Execution:
    def __init__(self, workflow_id: str, memo: dict[str, Any] | None) -> None:
        self.id = workflow_id
        self._memo = memo

    async def memo(self) -> dict[str, Any]:
        return self._memo or {}


class Handle:
    def __init__(self, answer: RunRecord | None, asked: list[str], workflow_id: str) -> None:
        self._answer = answer
        self._asked = asked
        self._id = workflow_id

    async def query(self, name: str) -> RunRecord | None:
        self._asked.append(self._id)
        return self._answer


class RunClient:
    """A client that lists the executions it was given and answers each record query."""

    def __init__(self, executions: list[Execution], records: dict[str, RunRecord | None]) -> None:
        self.executions = executions
        self.records = records
        self.queries: list[str] = []
        self.listed: list[tuple[str | None, int | None]] = []

    def list_workflows(self, query: str | None = None, *, limit: int | None = None, **_: Any):
        self.listed.append((query, limit))

        async def iterator():
            for execution in self.executions:
                yield execution

        return iterator()

    def get_workflow_handle(self, workflow_id: str, **_: Any) -> Handle:
        return Handle(self.records.get(workflow_id), self.queries, workflow_id)


class Connection:
    namespace = "default"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def client(self) -> Any:
        return self._client

    async def aclose(self) -> None:
        return None


def store(client: Any, **kwargs: Any) -> TemporalRunStore:
    return TemporalRunStore(
        "unused", task_queue="trellis-runs", connection=Connection(client), **kwargs
    )


async def test_the_inbox_lists_running_workflows_of_this_type_only() -> None:
    client = RunClient([Execution("run_a", {"tenant_id": TENANT})], {"run_a": record("run_a")})
    found = await store(client).list_paused(TENANT)
    assert [r.run_id for r in found] == ["run_a"]
    query, limit = client.listed[0]
    assert query is not None
    assert f"WorkflowType = '{RUN_WORKFLOW}'" in query
    assert "ExecutionStatus = 'Running'" in query
    assert limit == 1000, "the scan is bounded before anything is queried"


async def test_a_running_run_is_not_in_the_paused_inbox() -> None:
    client = RunClient(
        [Execution("run_a", {"tenant_id": TENANT})],
        {"run_a": record("run_a", status=RunStatus.RUNNING)},
    )
    assert await store(client).list_paused(TENANT) == []


async def test_another_tenants_run_never_reaches_this_tenants_inbox() -> None:
    """Twice over: the memo is the cheap filter, and the record itself is the decision — so a
    memo that cannot be read (an older run, another writer) cannot be used to read somebody
    else's inbox."""
    client = RunClient(
        [
            Execution("run_mine", {"tenant_id": TENANT}),
            Execution("run_theirs", {"tenant_id": "other"}),
            Execution("run_unlabelled", None),
        ],
        {
            "run_mine": record("run_mine"),
            "run_theirs": record("run_theirs", tenant="other"),
            "run_unlabelled": record("run_unlabelled", tenant="other"),
        },
    )
    found = await store(client).list_paused(TENANT)
    assert [r.run_id for r in found] == ["run_mine"]
    assert "run_theirs" not in client.queries, "the memo filter saves the round trip"
    assert "run_unlabelled" in client.queries, "an unlabelled run is asked, then refused"


async def test_the_inbox_stops_at_the_limit_it_was_given() -> None:
    executions = [Execution(f"run_{i}", {"tenant_id": TENANT}) for i in range(10)]
    records = {f"run_{i}": record(f"run_{i}") for i in range(10)}
    found = await store(RunClient(executions, records)).list_paused(TENANT, limit=3)
    assert len(found) == 3


async def test_the_scan_is_bounded_even_when_nothing_matches() -> None:
    """``scan_limit`` questions, not ``scan_limit`` matches: an inbox query on a busy cluster
    must cost a known amount."""
    executions = [Execution(f"run_{i}", {"tenant_id": TENANT}) for i in range(50)]
    records = {f"run_{i}": record(f"run_{i}", status=RunStatus.RUNNING) for i in range(50)}
    client = RunClient(executions, records)
    assert await store(client, scan_limit=5).list_paused(TENANT) == []
    assert len(client.queries) == 5


async def test_a_workflow_with_no_record_yet_is_skipped_not_invented() -> None:
    client = RunClient([Execution("run_a", {"tenant_id": TENANT})], {"run_a": None})
    assert await store(client).list_paused(TENANT) == []


# --------------------------------------------------------------------------- scheduler script


class Described:
    def __init__(self, schedule_id: str, memo: dict[str, Any], *, paused: bool = False) -> None:
        self.id = schedule_id
        self._memo = memo
        self.schedule = type("S", (), {"state": ScheduleState(paused=paused)})()
        self.info = type("I", (), {"next_action_times": [], "recent_actions": []})()

    async def memo(self) -> dict[str, Any]:
        return self._memo


class ScheduleHandle:
    def __init__(self, client: ScheduleClient, schedule_id: str) -> None:
        self.id = schedule_id
        self._client = client

    async def describe(self) -> Described:
        described = self._client.schedules.get(self.id)
        if described is None:
            # What the server actually answers, so the adapter's not-found path is the one
            # under test rather than a convenient exception type.
            raise RPCError(f"no schedule {self.id}", RPCStatusCode.NOT_FOUND, b"")
        return described

    async def pause(self, *, note: str | None = None) -> None:
        self._client.notes.append(("pause", note))
        self._client.schedules[self.id] = Described(
            self.id, self._client.schedules[self.id]._memo, paused=True
        )

    async def unpause(self, *, note: str | None = None) -> None:
        self._client.notes.append(("unpause", note))
        self._client.schedules[self.id] = Described(
            self.id, self._client.schedules[self.id]._memo, paused=False
        )

    async def delete(self) -> None:
        self._client.deleted.append(self.id)
        self._client.schedules.pop(self.id, None)


class ScheduleClient:
    def __init__(self) -> None:
        self.schedules: dict[str, Described] = {}
        self.created: list[tuple[str, Any, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.notes: list[tuple[str, str | None]] = []

    async def create_schedule(
        self, schedule_id: str, schedule: Any, *, memo: dict[str, Any] | None = None, **_: Any
    ) -> ScheduleHandle:
        self.created.append((schedule_id, schedule, memo or {}))
        self.schedules[schedule_id] = Described(
            schedule_id, memo or {}, paused=schedule.state.paused
        )
        return ScheduleHandle(self, schedule_id)

    def get_schedule_handle(self, schedule_id: str) -> ScheduleHandle:
        return ScheduleHandle(self, schedule_id)

    async def list_schedules(self, *_: Any, **__: Any):
        described = list(self.schedules.values())

        async def iterator():
            for entry in described:
                yield entry

        return iterator()


def scheduler(client: ScheduleClient, **kwargs: Any) -> TemporalScheduler:
    return TemporalScheduler(
        "unused", task_queue="trellis-runs", connection=Connection(client), **kwargs
    )


def spec(**changes: Any) -> ScheduleSpec:
    return ScheduleSpec(
        **{
            "tenant_id": TENANT,
            "agent_id": "refunds",
            "name": "nightly sweep",
            "cadence": "weekdays",
            "timezone": "Europe/Berlin",
            "on_behalf_of": "user-7",
            "input": "sweep",
            **changes,
        }
    )


async def test_a_firing_starts_a_run_on_behalf_of_the_person_who_set_the_schedule() -> None:
    client = ScheduleClient()
    created = await scheduler(client).create(spec())
    schedule_id, temporal, memo = client.created[0]

    action = temporal.action
    assert isinstance(action, ScheduleActionStartWorkflow)
    started = action.args[0]
    assert started.on_behalf_of == "user-7", "a 6 a.m. run acts as the person, never wider"
    assert started.user_id == "user-7"
    assert started.tenant_id == TENANT and started.agent_id == "refunds"
    assert started.metadata["schedule_id"] == schedule_id
    assert action.task_queue == "trellis-runs"

    assert list(temporal.spec.cron_expressions) == ["0 0 * * 1-5"]
    assert temporal.spec.time_zone_name == "Europe/Berlin"
    assert temporal.state.paused is False

    assert memo["tenant_id"] == TENANT
    assert memo[SPEC_MEMO_KEY]["cadence"] == "weekdays", "the contract's own spec, kept verbatim"
    assert created.cadence == "weekdays" and created.enabled is True


async def test_a_manual_schedule_is_created_paused_and_refuses_to_be_enabled() -> None:
    client = ScheduleClient()
    created = await scheduler(client).create(spec(cadence="manual"))
    _, temporal, _ = client.created[0]
    assert temporal.state.paused is True
    assert not temporal.spec.cron_expressions, "manual fires only when triggered"
    assert created.enabled is False

    with pytest.raises(ValueError, match="trigger it"):
        await scheduler(client).set_enabled(created.schedule_id, True)


async def test_a_disabled_spec_is_created_paused() -> None:
    client = ScheduleClient()
    created = await scheduler(client).create(spec(enabled=False))
    assert client.created[0][1].state.paused is True
    assert created.enabled is False


async def test_enabled_is_read_back_from_temporal_not_from_the_memo() -> None:
    """A memo cannot be updated on an existing schedule, so the thing that decides whether it
    fires has exactly one source of truth."""
    client = ScheduleClient()
    instance = scheduler(client)
    created = await instance.create(spec())
    disabled = await instance.set_enabled(created.schedule_id, False)
    assert disabled.enabled is False
    assert client.notes == [("pause", "disabled by the platform")]
    assert (await instance.get(created.schedule_id)).enabled is False

    enabled = await instance.set_enabled(created.schedule_id, True)
    assert enabled.enabled is True
    assert client.notes[-1] == ("unpause", "enabled by the platform")


async def test_a_cadence_under_the_floor_never_reaches_temporal() -> None:
    from trellis.contracts.errors import ConfigurationError

    client = ScheduleClient()
    with pytest.raises(ConfigurationError, match="more often than"):
        await scheduler(client).create(spec(cadence="*/5 * * * * *"))
    assert client.created == [], "nothing was registered"


async def test_only_this_tenants_schedules_are_listed() -> None:
    client = ScheduleClient()
    mine = await scheduler(client).create(spec(name="mine"))
    await scheduler(client).create(spec(name="theirs", tenant_id="other"))
    listed = await scheduler(client).list_for_tenant(TENANT)
    assert [s.schedule_id for s in listed] == [mine.schedule_id]
    assert await scheduler(client).list_for_tenant(TENANT, limit=0) == []


async def test_a_schedule_somebody_else_wrote_is_not_ours() -> None:
    """A namespace shared with another product must not leak into a tenant's list."""
    client = ScheduleClient()
    client.schedules["foreign"] = Described("foreign", {"tenant_id": TENANT})
    assert await scheduler(client).list_for_tenant(TENANT) == []
    assert await scheduler(client).get("foreign") is None


async def test_an_unknown_schedule_is_none_and_deleting_it_is_not_an_error() -> None:
    client = ScheduleClient()
    instance = scheduler(client)
    created = await instance.create(spec())
    await instance.delete(created.schedule_id)
    assert client.deleted == [created.schedule_id]
    assert await instance.get(created.schedule_id) is None


async def test_setting_a_schedule_nobody_has_is_a_keyerror() -> None:
    client = ScheduleClient()
    with pytest.raises(KeyError):
        await scheduler(client).set_enabled("sch_missing", False)


def test_a_schedule_round_trips_through_its_memo() -> None:
    """What ``list_for_tenant`` reads back is the record the platform wrote, id included."""
    original = Schedule.from_spec(spec())
    payload = original.model_dump(mode="json")
    revived = Schedule.model_validate({**payload, "schedule_id": original.schedule_id})
    assert revived == original
