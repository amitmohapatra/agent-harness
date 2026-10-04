"""The harness against its neighbours' committed OpenAPI documents (CI checks the siblings out
next to this repository): every request it sends must be an operation the service documents
with a body its schema accepts, and every answer the test doubles give must be one the service
documents. The memory fake is checked on every exchange of every test (``tests/conftest.py``);
here every call the harness makes is driven once, and the checker itself is shown to catch a
mismatch. agent-runs is checked the same way when its document is there."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from tests.support.memory import CONTRACT, ROUTES, FakeMemoryService
from tests.support.openapi import RUNS_OPENAPI, OpenAPI
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
    ScheduleSpec,
)
from trellis.harness.clients import runs as runs_module
from trellis.harness.clients.memory import Memory
from trellis.harness.clients.runs import HttpRuns


# --------------------------------------------------------------------------- memory service
def test_every_route_the_fake_serves_is_a_documented_operation() -> None:
    for method, pattern, name in ROUTES:
        path = pattern.replace("(?P<name>[^/]+)", "tool_search").replace(
            "(?P<document_id>[^/]+)", "doc_1"
        )
        assert CONTRACT.operation(method, path) is not None, (name, method, path)


async def test_every_memory_call_the_harness_makes_speaks_the_contract(
    memory_service: FakeMemoryService,
) -> None:
    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return order

    async def works(input: str, agent: Runtime) -> Any:
        await agent.tools.call("memory_search", query="refunds")
        await agent.tools.call("memory_remember", content="prefers email")
        await agent.tools.call("tool_search", task="refund")
        return await agent.tools.call("refund", order="o1")

    memory_service.claims = 3
    settings = Settings(memory_url="http://memory.test", api_key="test", bifrost_virtual_key="vk")
    async with Harness(config=settings) as h:
        h.memory = Memory("http://memory.test", "test", client=memory_service.client())
        agent = h.wrap(works, id="contract", tools=[refund])
        paused = await agent.run("refund o1", user="ada", thread="thr_1")
        assert paused.interrupt is not None
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
        assert done.status is RunStatus.SUCCESS
        await h.feedback(done.run_id, "correct", correction="refund o2")
        await h.add_document(b"Returns: 30 days.", user="ada", title="Returns", wait=None)
        await h.writes.drain()
    called = {c.name for c in memory_service.calls}
    assert called >= {
        "key",
        "context",
        "agent_tools",
        "call_agent_tool",
        "tool_hints",
        "record_tool",
        "feedback",
        "messages",
        "catalog",
        "put_catalog",
        "model_key",
        "add_document",
        "document",
    }
    assert memory_service.violations == []


def test_the_checker_catches_a_body_the_contract_refuses() -> None:
    request = httpx.Request(
        "POST",
        "http://memory.test/v1/feedback",
        json={"target_kind": "run", "verdict": "maybe"},  # no target_id, no such verdict
    )
    found = CONTRACT.request(request)
    assert any("target_id" in v for v in found) and any("maybe" in v for v in found)
    answer = httpx.Response(200, json={"rendered": "x"}, request=request)
    assert CONTRACT.response(request, answer)  # 200 is not how feedback answers a new record
    unknown = httpx.Request("GET", "http://memory.test/v1/nowhere")
    assert CONTRACT.request(unknown) == ["GET /v1/nowhere: no such operation"]
    teapot = httpx.Response(418, request=request)
    assert "not documented" in CONTRACT.response(request, teapot)[0]
    plain = httpx.Response(503, text="down", headers={"content-type": "text/plain"})
    assert CONTRACT.response(request, plain) == []  # not JSON: nothing to check
    wrong_media = httpx.Response(
        200, content=b"{}", headers={"content-type": "application/problem+json"}
    )
    tools = httpx.Request("GET", "http://memory.test/v1/tools")
    assert "answers application/problem+json" in CONTRACT.response(tools, wrong_media)[0]
    assert CONTRACT.response(tools, httpx.Response(304)) == []  # a conditional GET's answer
    multipart = httpx.Request(
        "POST", "http://memory.test/v1/verify", content=b"x", headers={"content-type": "text/csv"}
    )
    assert CONTRACT.request(multipart) == []
    xml = httpx.Request(
        "POST",
        "http://memory.test/v1/verify",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert CONTRACT.request(xml)  # an empty object misses what verify requires


# --------------------------------------------------------------------------- agent-runs
def _runs_contract() -> OpenAPI:
    if not RUNS_OPENAPI.exists():
        pytest.skip(f"agent-runs commits no OpenAPI document at {RUNS_OPENAPI}")
    return OpenAPI.load(RUNS_OPENAPI, strict_query=True)


@respx.mock
async def test_every_runs_call_the_harness_makes_speaks_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _runs_contract()
    violations: list[str] = []
    start = RunStart(run_id="run_1", tenant_id="t", agent_id="a", thread_id="thr", user_id="u")
    record = RunRecord.from_start(start, status=RunStatus.RUNNING)
    asked = Interrupt(interrupt_id="run_1.1.1", tenant_id="t", run_id="run_1", question="ok?")
    paused = record.model_copy(update={"status": RunStatus.PAUSED, "awaiting": asked})
    lease = {"run_id": "run_1", "worker_id": "w", "expires_at": "2026-10-04T00:01:00Z"}
    artifact = {
        "artifact_id": "art_1",
        "type": "blob",
        "uri": "/v1/artifacts/art_1",
        "mime_type": "application/json",
        "checksum": "sha256:0",
        "size_bytes": 2,
        "created_at": "2026-10-04T00:00:00Z",
    }
    spec = ScheduleSpec(tenant_id="t", agent_id="a", name="n", cadence="daily", on_behalf_of="u")
    schedule = {
        **spec.model_dump(mode="json"),
        "schedule_id": "sch_1",
        "created_at": "2026-10-04T00:00:00Z",
        "updated_at": "2026-10-04T00:00:00Z",
    }
    summary = {
        "run_id": "run_1",
        "agent_id": "a",
        "status": "PAUSED",
        "awaiting": asked.model_dump(mode="json"),
        "assignee": None,
        "deadline": None,
        "updated_at": "2026-10-04T00:00:00Z",
    }

    def answer(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        bodies: dict[tuple[str, str], tuple[int, Any]] = {
            ("POST", "/v1/runs"): (201, record.model_dump(mode="json")),
            ("POST", "/v1/runs/claim"): (
                200,
                {"run": record.model_dump(mode="json"), "lease": lease},
            ),
            ("POST", "/v1/runs/run_1/heartbeat"): (200, lease),
            ("POST", "/v1/runs/run_1/pause"): (200, paused.model_dump(mode="json")),
            ("POST", "/v1/runs/run_1/resume"): (200, record.model_dump(mode="json")),
            ("POST", "/v1/runs/run_1/finish"): (
                200,
                record.model_copy(update={"status": RunStatus.SUCCESS}).model_dump(mode="json"),
            ),
            ("GET", "/v1/runs/run_1"): (200, record.model_dump(mode="json")),
            ("GET", "/v1/runs"): (200, [summary]),
            ("POST", "/v1/runs/run_1/artifacts"): (201, artifact),
            ("POST", "/v1/schedules"): (201, schedule),
        }
        status, body = bodies[(method, path)]
        response = httpx.Response(status, json=body, request=request)
        violations.extend(contract.request(request) + contract.response(request, response))
        return response

    respx.route(host="runs.test").mock(side_effect=answer)
    runs = HttpRuns("http://runs.test", "key")
    await runs.started(start)
    await runs.queued(start)
    assert await runs.claim("w", ["a"], 60) is not None
    await runs.heartbeat("run_1", "w", 60)
    await runs.heartbeat("run_1", "w", 60, checkpoint={"calls": {"k": ["paid"]}})
    await runs.paused(asked, checkpoint={"answers": {}}, worker_id="w")
    resolution = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.ANSWER, answer="yes"
    )
    await runs.resumed(resolution)
    await runs.finished("run_1", RunStatus.SUCCESS, output={"ok": True}, worker_id="w")
    await runs.get("run_1")
    await runs.inbox("t", "role:ops")
    await runs.put_artifact("run_1", json.dumps([1]).encode(), worker_id="w")
    await runs.schedule(spec)
    await runs.aclose()
    assert violations == []


def test_the_runs_problem_document_reads_as_the_runs_client_reads_it() -> None:
    contract = _runs_contract()
    schemas = contract.document["components"]["schemas"]
    assert (
        "LEASE_LOST" in schemas["ErrorCode"]["enum"] and "CONFLICT" in schemas["ErrorCode"]["enum"]
    )
    problem = {
        "type": "urn:trellis:problem:lease-lost",
        "title": "Lease lost",
        "status": 409,
        "detail": "w1 no longer holds run_1",
        "instance": "/v1/runs/run_1/finish",
        "code": "LEASE_LOST",
        "retryable": False,
    }
    request = httpx.Request("POST", "http://runs.test/v1/runs/run_1/finish")
    response = httpx.Response(
        409, json=problem, headers={"content-type": "application/problem+json"}, request=request
    )
    assert contract.response(request, response) == []
    assert isinstance(runs_module.refusal(response), runs_module.LeaseLost)
