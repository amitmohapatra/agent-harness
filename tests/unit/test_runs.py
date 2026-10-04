from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunStart,
    RunStatus,
    ScheduleSpec,
)
from trellis.harness.clients.runs import HttpRuns, LeaseLost, LocalRuns, RunStoreError


def start(run_id: str = "run_1", agent: str = "a") -> RunStart:
    return RunStart(run_id=run_id, tenant_id="t", agent_id=agent, user_id="u", input="hi")


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


# --------------------------------------------------------------------------- in process
async def test_a_run_moves_through_the_contract_state_machine() -> None:
    runs = LocalRuns()
    assert (await runs.started(start())).status is RunStatus.RUNNING
    assert (await runs.started(start())).status is RunStatus.RUNNING  # idempotent
    paused = await runs.paused(interrupt(), checkpoint={"answers": {}})
    assert paused.status is RunStatus.PAUSED and paused.checkpoint == {"answers": {}}
    [waiting] = await runs.inbox("t", "role:ops")
    assert (waiting.run_id, waiting.assignee, waiting.status) == (
        "run_1",
        "role:ops",
        paused.status,
    )
    assert await runs.inbox("t", "role:other") == []
    assert [r.run_id for r in await runs.inbox("t", None)] == ["run_1"]
    answer = resolution()
    resumed = await runs.resumed(answer)
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2
    assert resumed.last_resolution == answer and resumed.checkpoint == {"answers": {}}
    done = await runs.finished("run_1", RunStatus.SUCCESS, output="ok")
    assert done.final and done.output == "ok" and done.checkpoint is None
    with pytest.raises(RunStoreError):
        await runs.finished("run_1", RunStatus.ERROR)


async def test_an_answer_to_another_question_is_refused() -> None:
    runs = LocalRuns()
    await runs.started(start())
    await runs.paused(interrupt())
    wrong = resolution().model_copy(update={"interrupt_id": "run_1.1.9"})
    with pytest.raises(RunStoreError):
        await runs.resumed(wrong)


async def test_queued_runs_are_claimed_once_by_agent_and_leases_expire() -> None:
    runs = LocalRuns()
    await runs.queued(start("run_a", agent="a"))
    await runs.queued(start("run_b", agent="b"))
    claimed = await runs.claim("w1", ["b"], 30)
    assert claimed is not None and claimed.run_id == "run_b" and claimed.status is RunStatus.RUNNING
    assert await runs.claim("w2", ["b"], 30) is None
    await runs.heartbeat("run_b", "w1", 30)
    with pytest.raises(RunStoreError):
        await runs.heartbeat("run_b", "w2", 30)
    lapsed = await runs.claim("w1", ["a"], -1)  # a lease already over
    assert lapsed is not None
    again = await runs.claim("w3", ["a"], 30)
    assert again is not None and again.run_id == "run_a" and again.attempt == 2


async def test_a_resumed_durable_run_goes_back_to_the_queue() -> None:
    runs = LocalRuns()
    await runs.queued(start())
    await runs.claim("w", ["a"], 30)
    await runs.paused(interrupt())
    assert (await runs.resumed(resolution())).status is RunStatus.QUEUED
    claimed = await runs.claim("w", ["a"], 30)
    assert claimed is not None and claimed.last_resolution is not None


async def test_a_cancel_ends_a_paused_run_and_a_lapsed_worker_cannot_write() -> None:
    runs = LocalRuns()
    await runs.started(start())
    await runs.paused(interrupt())
    cancel = resolution().model_copy(update={"decision": InterruptDecision.CANCEL})
    assert (await runs.resumed(cancel)).status is RunStatus.CANCELLED
    await runs.queued(start("run_2"))
    await runs.claim("w1", ["a"], 30)
    with pytest.raises(LeaseLost):
        await runs.finished("run_2", RunStatus.SUCCESS, worker_id="w2")
    assert (await runs.finished("run_2", RunStatus.SUCCESS, worker_id="w1")).final


async def test_a_due_schedule_queues_a_run_when_a_worker_claims() -> None:
    runs = LocalRuns()
    schedule = await runs.schedule(
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
    assert await runs.claim("w", ["a"], 30) is None
    runs._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    fired = await runs.claim("w", ["a"], 30)
    assert fired is not None and fired.input == "report" and fired.user_id == "u"
    assert fired.metadata["schedule_id"] == schedule.schedule_id


# --------------------------------------------------------------------------- agent-runs
def record_json(status: str = "RUNNING", **fields: object) -> dict[str, object]:
    return {"run_id": "run_1", "tenant_id": "t", "agent_id": "a", "status": status, **fields}


@respx.mock
async def test_the_http_store_speaks_the_agent_runs_wire() -> None:
    base = "http://runs.test"
    checkpoint = {"answers": {"k": []}}
    paused_json = record_json("PAUSED", awaiting=interrupt().awaiting(), checkpoint=checkpoint)
    started = respx.post(f"{base}/v1/runs").mock(
        return_value=httpx.Response(201, json=record_json("QUEUED"))
    )
    claim = respx.post(f"{base}/v1/runs/claim").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "run": record_json(),
                    "lease": {
                        "run_id": "run_1",
                        "worker_id": "w",
                        "expires_at": "2026-09-30T00:00:00Z",
                    },
                },
            ),
            httpx.Response(204),
        ]
    )
    pause = respx.post(f"{base}/v1/runs/run_1/pause").mock(
        return_value=httpx.Response(200, json=paused_json)
    )
    resume = respx.post(f"{base}/v1/runs/run_1/resume").mock(
        return_value=httpx.Response(200, json=record_json("QUEUED", attempt=2))
    )
    finish = respx.post(f"{base}/v1/runs/run_1/finish").mock(
        return_value=httpx.Response(200, json=record_json("SUCCESS", output="ok"))
    )
    heartbeat = respx.post(f"{base}/v1/runs/run_1/heartbeat").mock(
        side_effect=[
            httpx.Response(200, json={}),
            httpx.Response(409, json={"detail": "lease lost"}),
        ]
    )
    summary = {
        "run_id": "run_1",
        "agent_id": "a",
        "status": "PAUSED",
        "awaiting": interrupt().awaiting(),
        "assignee": "role:ops",
        "deadline": None,
        "updated_at": "2026-09-30T00:00:00Z",
    }
    inbox = respx.get(f"{base}/v1/runs").mock(return_value=httpx.Response(200, json=[summary]))
    respx.get(f"{base}/v1/runs/run_1").mock(return_value=httpx.Response(200, json=paused_json))
    schedule = respx.post(f"{base}/v1/schedules").mock(
        return_value=httpx.Response(
            201,
            json={
                "schedule_id": "s1",
                "tenant_id": "t",
                "agent_id": "a",
                "name": "n",
                "cadence": "@daily",
                "on_behalf_of": "u",
            },
        )
    )
    runs = HttpRuns(base, "key")

    assert (await runs.queued(start())).status is RunStatus.QUEUED
    body = json.loads(started.calls[0].request.content)
    assert body["queue"] is True and body["run_id"] == "run_1" and body["input"] == "hi"
    headers = started.calls[0].request.headers
    assert headers["X-Api-Key"] == "key" and headers["X-Trellis-Tenant"] == "t"

    claimed = await runs.claim("w", ["a"], 30)
    assert claimed is not None and claimed.status is RunStatus.RUNNING
    assert json.loads(claim.calls[0].request.content) == {
        "worker_id": "w",
        "agent_ids": ["a"],
        "lease_seconds": 30,
    }
    assert await runs.claim("w", ["a"], 30) is None
    await runs.heartbeat("run_1", "w", 30)
    with pytest.raises(LeaseLost):
        await runs.heartbeat("run_1", "w", 30)
    assert heartbeat.call_count == 2

    paused = await runs.paused(interrupt(), checkpoint=checkpoint, worker_id="w")
    assert pause.calls[0].request.url.params["worker_id"] == "w"
    sent = json.loads(pause.calls[0].request.content)
    assert sent["interrupt"]["question"] == "ok?" and sent["checkpoint"] == checkpoint
    assert paused.checkpoint == checkpoint
    fetched = await runs.get("run_1")
    assert fetched is not None and fetched.checkpoint == checkpoint

    assert (await runs.resumed(resolution())).attempt == 2
    assert json.loads(resume.calls[0].request.content)["answer"] == "yes"
    await runs.finished("run_1", RunStatus.SUCCESS, output="ok", worker_id="w")
    assert json.loads(finish.calls[0].request.content) == {"status": "SUCCESS", "output": "ok"}
    assert finish.calls[0].request.url.params["worker_id"] == "w"

    [waiting] = await runs.inbox("t", "role:ops")
    assert waiting.awaiting is not None and waiting.assignee == "role:ops"
    assert inbox.calls[0].request.url.params["assignee"] == "role:ops"
    assert inbox.calls[0].request.url.params["status"] == "PAUSED"
    spec = ScheduleSpec(tenant_id="t", agent_id="a", name="n", cadence="@daily", on_behalf_of="u")
    assert (await runs.schedule(spec)).schedule_id == "s1"
    assert schedule.called
    await runs.aclose()


@respx.mock
async def test_a_refused_or_unreachable_store_raises() -> None:
    respx.post("http://runs.test/v1/runs").mock(return_value=httpx.Response(409, text="duplicate"))
    respx.get("http://runs.test/v1/runs/nope").mock(return_value=httpx.Response(404))
    runs = HttpRuns("http://runs.test", None)
    with pytest.raises(RunStoreError, match="409"):
        await runs.started(start())
    assert await runs.get("nope") is None
    respx.post("http://runs.test/v1/runs").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(RunStoreError, match="unreachable"):
        await runs.started(start())


@respx.mock
async def test_a_schedule_is_one_post_the_service_upserts() -> None:
    base = "http://runs.test"
    spec = ScheduleSpec(
        tenant_id="t", agent_id="a", name="a for u", cadence="daily", on_behalf_of="u", input="v"
    )
    stored = {**spec.model_dump(mode="json"), "schedule_id": "s1"}
    route = respx.post(f"{base}/v1/schedules").mock(
        side_effect=[httpx.Response(201, json=stored), httpx.Response(200, json=stored)]
    )
    runs = HttpRuns(base, "key")
    first, again = await runs.schedule(spec), await runs.schedule(spec)
    assert first.schedule_id == again.schedule_id == "s1"
    assert route.call_count == 2 and len(respx.calls) == 2  # no listing, no PATCH
    await runs.aclose()


async def test_local_schedules_upsert_on_agent_person_cadence_and_input() -> None:
    runs = LocalRuns()

    def spec(**changes: object) -> ScheduleSpec:
        fields = {"tenant_id": "t", "agent_id": "a", "name": "n", "cadence": "daily"}
        return ScheduleSpec(**{**fields, "on_behalf_of": "u", "input": {"x": 1}, **changes})

    first = await runs.schedule(spec())
    assert (await runs.schedule(spec(name="renamed"))).schedule_id == first.schedule_id
    assert (await runs.schedule(spec(input={"x": 2}))).schedule_id != first.schedule_id
    assert (await runs.schedule(spec(on_behalf_of="v"))).schedule_id != first.schedule_id


@respx.mock
async def test_artifacts_are_uploaded_with_their_checksum_and_read_back() -> None:
    base = "http://runs.test"
    ref = {
        "artifact_id": "art_1",
        "type": "blob",
        "uri": "/v1/artifacts/art_1",
        "mime_type": "application/json",
        "checksum": "sha256:x",
        "size_bytes": 2,
    }
    upload = respx.post(f"{base}/v1/runs/run_1/artifacts").mock(
        return_value=httpx.Response(201, json=ref)
    )
    respx.get(f"{base}/v1/artifacts/art_1").mock(return_value=httpx.Response(200, content=b"[]"))
    respx.get(f"{base}/v1/artifacts/gone").mock(return_value=httpx.Response(404))
    runs = HttpRuns(base, "key")
    stored = await runs.put_artifact("run_1", b"[]", worker_id="w")
    assert stored.artifact_id == "art_1"
    request = upload.calls[0].request
    assert request.content == b"[]" and request.headers["content-type"] == "application/json"
    assert request.url.params["worker_id"] == "w"
    assert request.url.params["checksum"].startswith("sha256:")
    assert await runs.artifact("art_1", "t") == b"[]"
    assert await runs.artifact("gone", "t") is None
    await runs.aclose()


async def test_local_artifacts_are_kept_per_tenant() -> None:
    runs = LocalRuns()
    await runs.started(start())
    ref = await runs.put_artifact("run_1", b'{"a":1}')
    assert ref.size_bytes == 7
    assert await runs.artifact(ref.artifact_id, "t") == b'{"a":1}'
    assert await runs.artifact(ref.artifact_id, "other") is None


@respx.mock
async def test_a_failed_finish_carries_its_error_and_the_whole_inbox_names_no_assignee() -> None:
    from trellis.contracts import AgentError

    base = "http://runs.test"
    finish = respx.post(f"{base}/v1/runs/run_1/finish").mock(
        return_value=httpx.Response(200, json=record_json("ERROR"))
    )
    inbox = respx.get(f"{base}/v1/runs").mock(return_value=httpx.Response(200, json=[]))
    runs = HttpRuns(base, None)
    error = AgentError(code="Boom", message="it broke")
    await runs.finished("run_1", RunStatus.ERROR, error=error)
    sent = json.loads(finish.calls[0].request.content)
    assert sent["status"] == "ERROR" and sent["error"]["message"] == "it broke"
    assert "x-api-key" not in finish.calls[0].request.headers  # no key configured, none sent
    assert await runs.inbox("t", None) == []
    assert "assignee" not in inbox.calls[0].request.url.params
    await runs.aclose()


async def test_a_resume_of_a_run_the_store_never_had_is_refused() -> None:
    with pytest.raises(RunStoreError, match="no run run_1"):
        await LocalRuns().resumed(resolution())


async def test_a_manual_schedule_never_fires_on_its_own() -> None:
    runs = LocalRuns()
    manual = await runs.schedule(
        ScheduleSpec(tenant_id="t", agent_id="a", name="m", cadence="manual", on_behalf_of="u")
    )
    assert manual.next_fire_at is None
    assert await runs.claim("w", ["a"], 30) is None
