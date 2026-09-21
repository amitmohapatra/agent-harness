"""Runs are recorded as a side effect of the turn, never as a step in it.

The failure being prevented is the inverted priority: an agent that refused to answer a
customer because its own bookkeeping service was unreachable. Recording is best-effort by
default, and the tests below pin both halves of that — it records when it can, and the turn
survives when it cannot.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from universal_agent_contracts.messages import AgentResponse, AgentStatus

from universal_agent_harness import AgentHarness
from universal_agent_harness.runs import RunStoreClient

CONFIG = {"memory": {"enabled": False}}


def store(handler, **kwargs: Any) -> RunStoreClient:
    return RunStoreClient(
        "http://runs.test",
        api_key="k",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://runs.test"
        ),
        **kwargs,
    )


def recording():
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "path": request.url.path,
                "tenant": request.headers.get("x-tenant-id"),
                "key": request.headers.get("x-api-key"),
                "body": json.loads(request.content) if request.content else None,
            }
        )
        return httpx.Response(201, json={"run_id": "r1"})

    return calls, handler


def build(runs: Any) -> AgentHarness:
    return AgentHarness(runs=runs, defaults={"tenant_id": "acme"}, config=CONFIG)


async def test_a_completed_turn_opens_and_closes_a_run() -> None:
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok({"intent": "refund"})

    await agent({})
    await harness.drain()

    assert calls[0]["path"] == "/v1/runs"
    assert calls[-1]["path"].endswith("/transition")
    assert calls[-1]["body"]["status"] == "SUCCESS"
    assert calls[-1]["body"]["output"] == {"intent": "refund"}


async def test_the_run_id_is_its_own_idempotency_key() -> None:
    """Run ids are derived, not random, so a retried turn reopens nothing rather than
    opening a second run for the same work."""
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok(None)

    await agent({})
    await harness.drain()
    opened = calls[0]["body"]
    assert opened["idempotency_key"] == opened["run_id"]


async def test_the_tenant_and_key_travel_on_every_write() -> None:
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok(None)

    await agent({})
    await harness.drain()
    assert {c["tenant"] for c in calls} == {"acme"}
    assert {c["key"] for c in calls} == {"k"}


async def test_a_failed_turn_is_recorded_as_failed() -> None:
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse(status=AgentStatus.ERROR, data=None)

    await agent({})
    await harness.drain()
    assert calls[-1]["body"]["status"] == "ERROR"


async def test_an_unreachable_runs_service_does_not_fail_the_turn() -> None:
    """Bookkeeping being down is a worse reason to fail a customer's request than almost
    any other. The turn completes and the gap is logged."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    harness = build(store(handler))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok("answered anyway")

    result = await agent({})
    assert result.data == "answered anyway"  # the turn never waited on the recorder
    await harness.drain()  # and the failure surfaces only here


async def test_required_inverts_that_for_workflows_that_need_the_record() -> None:
    """Some workflows would rather fail than run unrecorded. That has to be opt-in."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    harness = build(store(handler, required=True))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok("x")

    await agent({})  # the turn itself still completes
    with pytest.raises(RuntimeError, match="agent-runs is required"):
        await harness.drain()


async def test_a_refused_transition_is_not_treated_as_an_outage() -> None:
    """409 is the service protecting the record from a repeated or illegal transition. It
    did its job; logging it as unavailable would train people to ignore real outages."""
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/transition"):
            seen.append(409)
            return httpx.Response(409, text="cannot move from SUCCESS to SUCCESS")
        return httpx.Response(201, json={"run_id": "r1"})

    harness = build(store(handler, required=True))

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok("x")

    result = await agent({})
    await harness.drain()  # must not raise, even with required=True
    assert result.data == "x"
    assert seen == [409]


async def test_an_unconfigured_deployment_records_nothing() -> None:
    harness = AgentHarness(defaults={"tenant_id": "acme"}, config=CONFIG)
    assert harness.runs.name == "noop"

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        return AgentResponse.ok("x")

    assert (await agent({})).data == "x"
