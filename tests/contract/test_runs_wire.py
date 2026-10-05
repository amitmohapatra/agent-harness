"""What the pipeline sends agent-runs, and what agent-runs answers, against agent-runs'
committed OpenAPI document — driven by real runs, not hand-made bodies: runs that pause for an
approval, ask with a table large enough to travel as an artifact, fail, are queued for a worker
that saves its progress and resumes them, and a schedule. The store behind the wire is
``LocalRuns`` (the same behaviour as agent-runs), so every answer is a real state transition;
every request and every answer is checked against the document. The harness's client is
``trellis.runs.RunsClient``, which keeps no tenant between calls: the service here acts as a
platform key's would, knowing a run's tenant only from the ``X-Trellis-Tenant`` each call that
names a run (and each listing) sends, so a call that forgot its ``tenant=`` is a violation. And
agent-runs' schemas are the contracts' models: what it answers parses into ``RunRecord``,
``Schedule``, ``Interrupt``, ``ArtifactRef``…, and the enums agree."""

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
from trellis.harness.journal import JOURNAL_REF, MAX_CHECKPOINT_BYTES
from trellis.harness.runs import LocalRuns
from trellis.memory.models import KeyInfo
from trellis.runs import (
    Lease,
    NotFoundError,
    PayloadTooLargeError,
    RunsClient,
    RunsError,
    RunSummary,
)

TENANT = "default"
#: the header a call names its tenant with (a platform key's only way to)
TENANT_HEADER = "X-Trellis-Tenant"


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
    """agent-runs' routes over ``LocalRuns``, checking every exchange against the document, and
    that every call naming a run (and every listing) names the run's tenant."""

    def __init__(self, contract: OpenAPI) -> None:
        self.contract = contract
        self.store = LocalRuns()
        self.violations: list[str] = []
        self.seen: list[tuple[str, str]] = []
        #: the tenant each call named
        self.tenants: set[str | None] = set()

    def client(self) -> RunsClient:
        transport = httpx.MockTransport(self.handle)
        client = httpx.AsyncClient(transport=transport, base_url="http://runs.test")
        return RunsClient("http://runs.test", api_key="key", http_client=client)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.violations.extend(self.contract.request(request))
        self.tenants.add(request.headers.get(TENANT_HEADER))
        try:
            status, body, headers = await self._answer(request)
        except RunsError as exc:  # LocalRuns' refusal, as agent-runs words it
            status, body, headers = *_problem(request, exc), {}
        response = _response(request, status, body, headers)
        self.violations.extend(self.contract.response(request, response))
        self.seen.append((request.method, _template(request.url.path)))
        return response

    def _tenant(self, request: httpx.Request) -> str | None:
        """The tenant the call names; a call that names none is a violation."""
        tenant = request.headers.get(TENANT_HEADER)
        if tenant is None:
            self.violations.append(f"{request.method} {request.url.path}: no {TENANT_HEADER}")
        return tenant

    async def _answer(self, request: httpx.Request) -> tuple[int, Any, dict[str, str]]:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if path.startswith("/v1/runs/") and path != "/v1/runs/claim":
            run_id, _, action = path.removeprefix("/v1/runs/").partition("/")
            status, answer = await self._run(request, run_id, action, body)
            return status, answer, {}
        return await self._collection(request, body)

    async def _collection(
        self, request: httpx.Request, body: Any
    ) -> tuple[int, Any, dict[str, str]]:
        method, path, store = request.method, request.url.path, self.store
        if (method, path) == ("POST", "/v1/runs"):
            start = RunStart.model_validate({k: v for k, v in body.items() if k != "queue"})
            made = await store.start(start, queue=body["queue"])
            return 201, made.model_dump(mode="json"), {}
        if (method, path) == ("GET", "/v1/runs"):
            return self._page(request, await self._listed(request))
        if (method, path) == ("POST", "/v1/runs/claim"):
            claimed = await store.claim(
                body["worker_id"], body["agent_ids"], lease_seconds=body["lease_seconds"]
            )
            if claimed is None:
                return 204, None, {}
            return 200, claimed.model_dump(mode="json"), {}
        if (method, path) == ("POST", "/v1/schedules"):
            made = await store.schedules.create(ScheduleSpec.model_validate(body))
            return 201, made.model_dump(mode="json"), {}
        artifact_id = path.removeprefix("/v1/artifacts/")
        data = await store.artifacts.download(artifact_id, tenant=self._tenant(request))
        if data is None:
            raise NotFoundError(f"no artifact {artifact_id}", code="NOT_FOUND", status=404)
        return 200, data, {}

    async def _listed(self, request: httpx.Request) -> list[RunSummary]:
        params = request.url.params
        status = params.get("status")
        listed = self.store.iterate(
            status=RunStatus(status) if status else None,
            assignee=params.get("assignee"),
            tenant=self._tenant(request),
        )
        return [summary async for summary in listed]

    @staticmethod
    def _page(request: httpx.Request, rows: list[RunSummary]) -> tuple[int, Any, dict[str, str]]:
        """One page of ``rows`` from the request's ``cursor``, linking the next."""
        limit = int(request.url.params["limit"])
        offset = int(request.url.params.get("cursor", "0"))
        page = rows[offset : offset + limit]
        more = offset + limit < len(rows)
        link = {"link": f'</v1/runs?cursor={offset + limit}>; rel="next"'} if more else {}
        return 200, [r.model_dump(mode="json") for r in page], link

    async def _run(
        self, request: httpx.Request, run_id: str, action: str, body: Any
    ) -> tuple[int, Any]:
        store, worker = self.store, request.url.params.get("worker_id")
        tenant = self._tenant(request)
        if action == "":
            record = await store.get(run_id, tenant=tenant)
            if record is None:
                raise NotFoundError(f"no run {run_id}", code="NOT_FOUND", status=404)
            return 200, record.model_dump(mode="json")
        if action in ("pause", "heartbeat"):
            _bounded(body.get("checkpoint"))
        if action == "pause":
            asked = Interrupt.model_validate(body["interrupt"])
            paused = await store.pause(asked, checkpoint=body["checkpoint"], worker_id=worker)
            return 200, paused.model_dump(mode="json")
        if action == "resume":
            resolution = InterruptResolution.model_validate(body)
            resumed = await store.resume(resolution, tenant=tenant)
            return 200, resumed.model_dump(mode="json")
        if action == "finish":
            error = AgentError.model_validate(body["error"]) if body.get("error") else None
            status = RunStatus(body["status"])
            done = await store.finish(
                run_id, status, output=body["output"], error=error, worker_id=worker, tenant=tenant
            )
            return 200, done.model_dump(mode="json")
        if action in ("heartbeat", "cancel", "release"):
            return 200, (await self._held(run_id, action, body, tenant)).model_dump(mode="json")
        assert action == "artifacts", action
        ref = await store.artifacts.upload(
            run_id,
            request.content,
            mime_type=request.headers["content-type"],
            worker_id=worker,
            tenant=tenant,
        )
        return 201, ref.model_dump(mode="json")

    async def _held(
        self, run_id: str, action: str, body: Any, tenant: str | None
    ) -> RunRecord | Lease:
        """A worker's heartbeat or release, or a cancel."""
        if action == "heartbeat":
            return await self.store.heartbeat(
                run_id,
                body["worker_id"],
                lease_seconds=body["lease_seconds"],
                checkpoint=body.get("checkpoint"),
                tenant=tenant,
            )
        if action == "cancel":
            return await self.store.cancel(run_id, reason=body.get("reason"), tenant=tenant)
        return await self.store.release(
            run_id, body["worker_id"], checkpoint=body.get("checkpoint"), tenant=tenant
        )


def _bounded(checkpoint: dict[str, Any] | None) -> None:
    """agent-runs' ``bounded_checkpoint``: a checkpoint over 1 MiB of compact JSON is ``413``."""
    size = len(json.dumps(checkpoint, separators=(",", ":"), ensure_ascii=False).encode())
    if size > MAX_CHECKPOINT_BYTES:
        raise PayloadTooLargeError(
            f"checkpoint is {size} bytes", code="PAYLOAD_TOO_LARGE", status=413
        )


def _problem(request: httpx.Request, exc: RunsError) -> tuple[int, dict[str, Any]]:
    return exc.status, {
        "type": f"urn:trellis:problem:{exc.code.lower().replace('_', '-')}",
        "title": exc.code.replace("_", " ").capitalize(),
        "status": exc.status,
        "detail": exc.message,
        "instance": request.url.path,
        "code": exc.code,
        "retryable": False,
    }


def _response(
    request: httpx.Request, status: int, body: Any, headers: dict[str, str]
) -> httpx.Response:
    if body is None:
        return httpx.Response(status, request=request)
    if isinstance(body, bytes):
        return httpx.Response(
            status, content=body, headers={"content-type": "application/json"}, request=request
        )
    media = "application/problem+json" if status >= 400 else "application/json"
    return httpx.Response(
        status,
        content=json.dumps(body),
        headers={"content-type": media, **headers},
        request=request,
    )


def _template(path: str) -> str:
    return re.sub(r"/(run_|art_)[^/]+", "/{id}", path)


@pytest.fixture
async def wired() -> AsyncIterator[tuple[Harness, RunsService]]:
    service = RunsService(runs_contract())
    async with Harness(runs=service.client()) as h:
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
    ref = asked.interrupt.payload_ref
    assert await h.runs.artifacts.download(ref.artifact_id, tenant=TENANT) is not None
    [waiting] = await h.inbox("role:ops")
    assert waiting.awaiting == asked.interrupt
    cancelled = await reviewing.resume(asked.interrupt.interrupt_id, "cancel", reviewer="ada")
    assert cancelled.status is RunStatus.CANCELLED

    failed = await h.wrap(failing, id="failing").run("x", user="ada")
    assert failed.error is not None and failed.error.retryable  # a timeout may pass
    record = await h.runs.get(failed.run_id, tenant=TENANT)
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
    assert await h.runs.get("run_missing", tenant=TENANT) is None

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


@tool(side_effects="write")
def export(order: str) -> str:
    """Export an order's history."""
    return "x" * (MAX_CHECKPOINT_BYTES + 1)  # a journal larger than a checkpoint may be


async def test_a_runs_limits_version_and_cancel_speak_the_runs_contract(
    wired: tuple[Harness, RunsService],
) -> None:
    h, service = wired
    deadline = datetime.now(UTC) + timedelta(hours=1)
    queued = h.wrap(billing, id="billing", tools=[charge], version="2026.10")
    handle = await queued.start({"order": "o-8"}, user="ada", timeout=60, deadline=deadline)
    record = await handle.status()
    assert (record.timeout_seconds, record.deadline, record.agent_version) == (
        60,
        deadline,
        "2026.10",
    )
    assert (await handle.cancel(reason="a duplicate")).status is RunStatus.CANCELLED
    timed = await h.wrap(failing, id="failing").run("x", user="ada", timeout=30)
    assert timed.status is RunStatus.ERROR  # within its time: the error, not a timeout
    assert service.violations == [], "\n".join(service.violations)
    assert ("POST", "/v1/runs/{id}/cancel") in service.seen


async def test_a_journal_larger_than_a_checkpoint_travels_as_a_run_artifact(
    wired: tuple[Harness, RunsService],
) -> None:
    h, service = wired

    async def exporting(input: dict[str, Any], agent: Runtime) -> Any:
        history = await agent.tools.call("export", order=input["order"])
        size = await agent.ask("Which size?", options=["S", "L"])
        return f"{len(history)}; {size}"

    queued = h.wrap(exporting, id="exporting", tools=[export])
    handle = await queued.start({"order": "o-7"}, user="ada")
    worker = h.worker([queued], concurrency=1)
    assert await worker.run_once()  # saved as progress, then paused: both by reference
    first = await handle.result(timeout=5)
    assert first.interrupt is not None
    record = await h.runs.get(handle.run_id, tenant=TENANT)
    assert record is not None and record.checkpoint is not None
    assert set(record.checkpoint) == {JOURNAL_REF}
    await queued.resume(first.interrupt.interrupt_id, "answer", answer="S", reviewer="ada")
    assert await worker.run_once()
    assert (await handle.result(timeout=5)).answer == f"{MAX_CHECKPOINT_BYTES + 1}; S"
    assert service.violations == [], "\n".join(service.violations)
    assert ("GET", "/v1/artifacts/{id}") in service.seen


async def test_the_inbox_reads_every_page_and_says_when_it_stops(
    wired: tuple[Harness, RunsService],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from trellis.harness import harness as harness_module

    h, service = wired

    async def asking(input: str, agent: Runtime) -> Any:
        return await agent.ask(f"ok {input}?", assignee="role:ops")

    agent = h.wrap(asking, id="asking")
    for n in range(3):
        await agent.run(str(n), user="ada")
    monkeypatch.setattr(harness_module, "INBOX_LIMIT", 1)
    assert len(await h.inbox("role:ops")) == 3  # three pages of one, newest first
    listings = [c for c in service.seen if c == ("GET", "/v1/runs")]
    assert len(listings) == 3
    monkeypatch.setattr(harness_module, "INBOX_MAX_PAGES", 2)
    with caplog.at_level("WARNING", logger="trellis.harness"):
        assert len(await h.inbox("role:ops")) == 2
    assert "holds at least 2 paused runs" in caplog.text
    assert await h.inbox("role:nobody") == []
    assert service.violations == [], "\n".join(service.violations)


async def test_a_platform_keys_runs_name_their_tenant_on_every_call(
    wired: tuple[Harness, RunsService], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A platform key has no tenant of its own: nothing remembers one between calls, so
    every call carries the tenant the caller named, from the run's record or runtime."""
    h, service = wired

    async def platform() -> KeyInfo:
        return KeyInfo(key_id="k", tenant_id=None, principal="platform", role="service")

    monkeypatch.setattr(h, "key", platform)
    approving = h.wrap(approver, id="approver", tools=[refund])
    paused = await approving.run("o-1", user="ada", tenant="acme")
    assert paused.interrupt is not None
    [waiting] = await h.inbox(tenant="acme")
    assert waiting.run_id == paused.run_id
    answer = paused.interrupt.interrupt_id
    assert (await approving.resume(answer, "approve", reviewer="cfo", tenant="acme")).answer
    await h.feedback(paused.run_id, "confirm", tenant="acme")

    reviewing = h.wrap(reviewer, id="reviewer")
    queued = h.wrap(billing, id="billing", tools=[charge])
    asked = await reviewing.run("review", user="ada", tenant="acme")
    assert asked.interrupt is not None and asked.interrupt.payload_ref is not None
    handle = await queued.start({"order": "o-5"}, user="ada", tenant="acme")
    assert await h.worker([queued], concurrency=1).run_once()
    first = await handle.result(timeout=5)
    assert first.interrupt is not None
    await queued.schedule("daily", {"order": "o-6"}, on_behalf_of="ada", tenant="acme")
    assert service.violations == [], "\n".join(service.violations)
    assert service.tenants <= {"acme", None}  # a claim names none: the key's own queue
    assert "acme" in service.tenants
