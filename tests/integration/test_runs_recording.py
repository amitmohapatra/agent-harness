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
from universal_agent_contracts import AgentPaused
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


# ------------------------------------------------------- pausing, and telling someone


async def test_a_plain_agent_can_pause_and_the_question_is_recorded() -> None:
    """The framework-agnostic human-in-the-loop path, with no LangGraph anywhere.

    Two things had to be true for this and neither was: an ordinary coroutine needed a way
    to say "a person has to answer this" (pause detection matched four LangGraph class
    names and nothing else), and the question had to survive the trip (only the exception's
    class name reached the store, so the inbox said "GraphInterrupt" and nothing about what
    was actually being asked).
    """
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="refund-bot")
    async def agent(state, runtime):
        raise AgentPaused(
            "Approve a EUR 240 refund for order 91?",
            expects={"type": "boolean"},
            payload={"order_id": 91},
        )

    with pytest.raises(AgentPaused):
        await agent({})
    await harness.drain()

    transitions = [c for c in calls if c["path"].endswith("/transition")]
    assert transitions, "a paused run must be recorded"
    awaiting = transitions[-1]["body"]["awaiting"]
    assert transitions[-1]["body"]["status"] == "PAUSED"
    assert awaiting["question"] == "Approve a EUR 240 refund for order 91?"
    assert awaiting["expects"] == {"type": "boolean"}
    assert awaiting["payload"] == {"order_id": 91}
    # The class name stays too: a UI may want to know what *kind* of pause it was.
    assert awaiting["reason"] == "AgentPaused"


async def test_a_paused_run_is_not_reported_as_finished() -> None:
    """PAUSED is not terminal. Reporting it as finished is what made "waiting for a human"
    look like a completed turn."""
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="a")
    async def agent(state, runtime):
        raise AgentPaused("Approve?")

    with pytest.raises(AgentPaused):
        await agent({})
    await harness.drain()

    statuses = [c["body"].get("status") for c in calls if c["path"].endswith("/transition")]
    assert "PAUSED" in statuses
    assert not {"SUCCESS", "ERROR"} & set(statuses), statuses


async def test_a_webhook_url_is_registered_when_the_deployment_sets_one() -> None:
    """Without it a UI has to poll to notice that the 3am job is waiting on an approval."""
    calls, handler = recording()
    harness = build(store(handler, webhook_url="https://ui.example/hooks/runs"))

    @harness.agent(agent_id="a")
    async def agent(state, runtime):
        return AgentResponse.ok({})

    await agent({})
    await harness.drain()

    opened = next(c for c in calls if c["path"] == "/v1/runs")
    assert opened["body"]["webhook_url"] == "https://ui.example/hooks/runs"


async def test_no_webhook_url_is_sent_when_none_is_configured() -> None:
    """agent-runs forbids unknown fields, and an explicit null is not the same as not
    asking to be told."""
    calls, handler = recording()
    harness = build(store(handler))

    @harness.agent(agent_id="a")
    async def agent(state, runtime):
        return AgentResponse.ok({})

    await agent({})
    await harness.drain()

    opened = next(c for c in calls if c["path"] == "/v1/runs")
    assert "webhook_url" not in opened["body"]
