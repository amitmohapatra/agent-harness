"""What the pipeline sends agent-runs, and what agent-runs answers, against agent-runs'
committed OpenAPI document — driven by real runs, not hand-made bodies: runs that pause for an
approval, ask with a table large enough to travel as an artifact, fail, are queued for a worker
that saves its progress and resumes them, and a schedule. The store behind the wire is
``LocalRuns`` (the same behaviour as agent-runs), so every answer is a real state transition;
every request and every answer is checked against the document. And agent-runs' schemas are
the contracts' models: what it answers parses into ``RunRecord``, ``Schedule``, ``Interrupt``,
``ArtifactRef``…, and the enums agree."""

from __future__ import annotations

import enum
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from tests.support.openapi import OpenAPI, runs_contract
from trellis import Harness, Runtime, tool
from trellis.contracts import (
    AgentError,
    ArtifactRef,
    ErrorCategory,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
    Schedule,
    ScheduleSpec,
)
from trellis.harness.clients.runs import (
    Conflict,
    HttpRuns,
    LeaseLost,
    LocalRuns,
    NotFound,
    RunStoreError,
    RunSummary,
)

TENANT = "default"


# --------------------------------------------------------------------------- the schemas
#: agent-runs' component schema, and the model the harness parses it with
MODELS: dict[str, type[BaseModel]] = {
    "RunRecord": RunRecord,
    "Schedule": Schedule,
    "ScheduleSpec": ScheduleSpec,
    "Interrupt": Interrupt,
    "InterruptResolution": InterruptResolution,
    "ArtifactRef": ArtifactRef,
    "AgentError": AgentError,
    "RunSummary": RunSummary,
}
ENUMS: dict[str, type[enum.Enum]] = {
    "RunStatus": RunStatus,
    "InterruptReason": InterruptReason,
    "InterruptDecision": InterruptDecision,
    "ErrorCategory": ErrorCategory,
}


@pytest.mark.parametrize("name", list(MODELS))
def test_agent_runs_schemas_parse_into_the_contracts_models(name: str) -> None:
    schema = runs_contract().document["components"]["schemas"][name]
    model = MODELS[name]
    documented = set(schema.get("properties", {}))
    fields = set(model.model_fields)
    required = set(schema.get("required", []))
    # everything the model needs is always there
    assert {n for n, f in model.model_fields.items() if f.is_required()} <= required, name
    if model.model_config.get("extra") == "ignore":  # a listing: the harness reads a subset
        assert fields <= documented, (name, fields - documented)
    else:  # every property agent-runs may send is a field (a model may forbid extras)
        assert documented == fields, (name, documented ^ fields)


@pytest.mark.parametrize("name", list(ENUMS))
def test_agent_runs_enums_are_the_contracts_enums(name: str) -> None:
    schema = runs_contract().document["components"]["schemas"][name]
    assert set(schema["enum"]) == {member.value for member in ENUMS[name]}


# --------------------------------------------------------------------------- the wire
class RunsService:
    """agent-runs' routes over ``LocalRuns``, checking every exchange against the document."""

    def __init__(self, contract: OpenAPI) -> None:
        self.contract = contract
        self.store = LocalRuns()
        self.violations: list[str] = []
        self.seen: list[tuple[str, str]] = []

    def client(self) -> HttpRuns:
        transport = httpx.MockTransport(self.handle)
        client = httpx.AsyncClient(transport=transport, base_url="http://runs.test")
        return HttpRuns("http://runs.test", "key", client=client)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.violations.extend(self.contract.request(request))
        try:
            status, body = await self._answer(request)
        except RunStoreError as exc:  # LocalRuns' refusal, as agent-runs words it
            status, body = _problem(request, exc)
        response = _response(request, status, body)
        self.violations.extend(self.contract.response(request, response))
        self.seen.append((request.method, _template(request.url.path)))
        return response

    async def _answer(self, request: httpx.Request) -> tuple[int, Any]:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if path.startswith("/v1/runs/") and path != "/v1/runs/claim":
            run_id, _, action = path.removeprefix("/v1/runs/").partition("/")
            return await self._run(request, run_id, action, body)
        return await self._collection(request, body)

    async def _collection(self, request: httpx.Request, body: Any) -> tuple[int, Any]:
        method, path, store = request.method, request.url.path, self.store
        if (method, path) == ("POST", "/v1/runs"):
            start = RunStart.model_validate({k: v for k, v in body.items() if k != "queue"})
            made = await (store.queued if body["queue"] else store.started)(start)
            return 201, made.model_dump(mode="json")
        if (method, path) == ("GET", "/v1/runs"):
            summaries = await store.inbox(TENANT, request.url.params.get("assignee"))
            return 200, [s.model_dump(mode="json") for s in summaries]
        if (method, path) == ("POST", "/v1/runs/claim"):
            claimed = await store.claim(body["worker_id"], body["agent_ids"], body["lease_seconds"])
            if claimed is None:
                return 204, None
            return 200, {"run": claimed.model_dump(mode="json"), "lease": _lease(claimed, body)}
        if (method, path) == ("POST", "/v1/schedules"):
            made = await store.schedule(ScheduleSpec.model_validate(body))
            return 201, made.model_dump(mode="json")
        artifact_id = path.removeprefix("/v1/artifacts/")
        data = await store.artifact(artifact_id, TENANT)
        if data is None:
            raise NotFound(f"no artifact {artifact_id}")
        return 200, data

    async def _run(
        self, request: httpx.Request, run_id: str, action: str, body: Any
    ) -> tuple[int, Any]:
        store, worker = self.store, request.url.params.get("worker_id")
        if action == "":
            record = await store.get(run_id)
            if record is None:
                raise NotFound(f"no run {run_id}")
            return 200, record.model_dump(mode="json")
        if action == "pause":
            asked = Interrupt.model_validate(body["interrupt"])
            paused = await store.paused(asked, checkpoint=body["checkpoint"], worker_id=worker)
            return 200, paused.model_dump(mode="json")
        if action == "resume":
            resumed = await store.resumed(InterruptResolution.model_validate(body))
            return 200, resumed.model_dump(mode="json")
        if action == "finish":
            error = AgentError.model_validate(body["error"]) if body.get("error") else None
            status = RunStatus(body["status"])
            done = await store.finished(
                run_id, status, output=body["output"], error=error, worker_id=worker
            )
            return 200, done.model_dump(mode="json")
        if action == "heartbeat":
            checkpoint = body.get("checkpoint")
            await store.heartbeat(
                run_id, body["worker_id"], body["lease_seconds"], checkpoint=checkpoint
            )
            record = await store.get(run_id)
            assert record is not None
            return 200, _lease(record, body)
        assert action == "artifacts", action
        ref = await store.put_artifact(run_id, request.content, worker_id=worker)
        return 201, ref.model_dump(mode="json")


#: the problem code of each refusal of LocalRuns, and its status
PROBLEMS: dict[type[RunStoreError], tuple[str, int]] = {
    LeaseLost: ("LEASE_LOST", 409),
    Conflict: ("CONFLICT", 409),
    NotFound: ("NOT_FOUND", 404),
}


def _problem(request: httpx.Request, exc: RunStoreError) -> tuple[int, dict[str, Any]]:
    code, status = PROBLEMS[type(exc)]
    return status, {
        "type": f"urn:trellis:problem:{code.lower().replace('_', '-')}",
        "title": code.replace("_", " ").capitalize(),
        "status": status,
        "detail": str(exc),
        "instance": request.url.path,
        "code": code,
        "retryable": False,
    }


def _lease(record: RunRecord, body: dict[str, Any]) -> dict[str, Any]:
    until = datetime.now(UTC) + timedelta(seconds=body["lease_seconds"])
    return {
        "run_id": record.run_id,
        "worker_id": body["worker_id"],
        "expires_at": until.isoformat(),
    }


def _response(request: httpx.Request, status: int, body: Any) -> httpx.Response:
    if body is None:
        return httpx.Response(status, request=request)
    if isinstance(body, bytes):
        return httpx.Response(
            status, content=body, headers={"content-type": "application/json"}, request=request
        )
    media = "application/problem+json" if status >= 400 else "application/json"
    return httpx.Response(
        status, content=json.dumps(body), headers={"content-type": media}, request=request
    )


def _template(path: str) -> str:
    return re.sub(r"/(run_|art_)[^/]+", "/{id}", path)


@pytest.fixture
async def wired() -> AsyncIterator[tuple[Harness, RunsService]]:
    service = RunsService(runs_contract())
    async with Harness() as h:
        h.runs = service.client()
        yield h, service


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


@tool(side_effects="write")
def charge(order: str) -> str:
    """Charge an order."""
    return f"charged {order}"


async def approver(input: str, agent: Runtime) -> Any:
    return await agent.tools.call("refund", order=input)


async def reviewer(input: str, agent: Runtime) -> Any:
    rows = [{"sku": f"S-{i}", "note": "x" * 200} for i in range(120)]  # over 16 KiB: an artifact
    return await agent.ask("Check the lines", table=rows, assignee="role:ops")


async def failing(input: str, agent: Runtime) -> Any:
    raise TimeoutError("the supplier did not answer")


async def billing(input: dict[str, Any], agent: Runtime) -> Any:
    receipt = await agent.tools.call("charge", order=input["order"])
    size = await agent.ask("Which size?", options=["S", "L"])
    return f"{receipt}; {size}"


async def test_every_run_the_pipeline_records_speaks_the_runs_contract(
    wired: tuple[Harness, RunsService],
) -> None:
    h, service = wired
    approving = h.wrap(approver, id="approver", tools=[refund])
    paused = await approving.run("o-1", user="ada")
    assert paused.interrupt is not None and paused.interrupt.tool_call is not None
    assert (await approving.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")).answer
    rejected = await approving.run("o-2", user="ada")
    assert rejected.interrupt is not None
    await approving.resume(rejected.interrupt.interrupt_id, "reject", answer="no", reviewer="cfo")

    reviewing = h.wrap(reviewer, id="reviewer")
    asked = await reviewing.run("review", user="ada")
    assert asked.interrupt is not None and asked.interrupt.payload_ref is not None
    assert await h.runs.artifact(asked.interrupt.payload_ref.artifact_id, TENANT) is not None
    [waiting] = await h.inbox("role:ops")
    assert waiting.awaiting == asked.interrupt
    cancelled = await reviewing.resume(asked.interrupt.interrupt_id, "cancel", reviewer="ada")
    assert cancelled.status is RunStatus.CANCELLED

    failed = await h.wrap(failing, id="failing").run("x", user="ada")
    assert failed.error is not None and failed.error.retryable  # a timeout may pass
    record = await h.runs.get(failed.run_id)
    assert record is not None and record.error == failed.error

    queued = h.wrap(billing, id="billing", tools=[charge])
    handle = await queued.start({"order": "o-3"}, user="ada")
    worker = h.worker([queued], concurrency=1)
    assert await worker.run_once()  # the charge is saved as progress, then it asks
    first = await handle.result(timeout=5)
    assert first.interrupt is not None
    again = await queued.resume(first.interrupt.interrupt_id, "answer", answer="L", reviewer="a")
    assert again.status is RunStatus.QUEUED
    assert await worker.run_once()
    assert (await handle.result(timeout=5)).answer == "charged o-3; L"
    schedule = await queued.schedule("daily", {"order": "o-4"}, on_behalf_of="ada")
    assert schedule.cadence == "daily"
    assert await h.runs.get("run_missing") is None

    assert service.violations == [], "\n".join(service.violations)
    assert {
        ("POST", "/v1/runs"),
        ("POST", "/v1/runs/{id}/pause"),
        ("POST", "/v1/runs/{id}/resume"),
        ("POST", "/v1/runs/{id}/finish"),
        ("POST", "/v1/runs/{id}/artifacts"),
        ("GET", "/v1/artifacts/{id}"),
        ("GET", "/v1/runs"),
        ("GET", "/v1/runs/{id}"),
        ("POST", "/v1/runs/claim"),
        ("POST", "/v1/runs/{id}/heartbeat"),
        ("POST", "/v1/schedules"),
    } <= set(service.seen)
