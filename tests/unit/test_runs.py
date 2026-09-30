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
from trellis.harness.journal import JOURNAL_KEY


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
    paused = await runs.paused(interrupt(), journal={"answers": {}})
    assert paused.status is RunStatus.PAUSED and paused.metadata[JOURNAL_KEY] == {"answers": {}}
    assert [r.run_id for r in await runs.list_paused("t", assignee="role:ops")] == ["run_1"]
    assert await runs.list_paused("t", assignee="role:other") == []
    answer = resolution()
    resumed = await runs.resumed(answer)
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2
    assert resumed.last_resolution == answer
    done = await runs.finished("run_1", RunStatus.SUCCESS, output="ok")
    assert done.final and done.output == "ok"
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
    paused_json = record_json("PAUSED", awaiting=interrupt().awaiting())
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
    inbox = respx.get(f"{base}/v1/runs").mock(return_value=httpx.Response(200, json=[paused_json]))
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

    paused = await runs.paused(interrupt(), journal={"answers": {"k": []}}, worker_id="w")
    assert pause.calls[0].request.url.params["worker_id"] == "w"
    assert json.loads(pause.calls[0].request.content)["question"] == "ok?"
    # agent-runs keeps no journal: this client carries it for the runs it paused
    assert paused.metadata[JOURNAL_KEY] == {"answers": {"k": []}}
    fetched = await runs.get("run_1")
    assert fetched is not None and fetched.metadata[JOURNAL_KEY] == {"answers": {"k": []}}

    assert (await runs.resumed(resolution())).attempt == 2
    assert json.loads(resume.calls[0].request.content)["answer"] == "yes"
    await runs.finished("run_1", RunStatus.SUCCESS, output="ok", worker_id="w")
    assert json.loads(finish.calls[0].request.content) == {"status": "SUCCESS", "output": "ok"}
    assert finish.calls[0].request.url.params["worker_id"] == "w"

    assert (await runs.list_paused("t", assignee="role:ops"))[0].awaiting is not None
    assert inbox.calls[0].request.url.params["assignee"] == "role:ops"
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
