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
from trellis.contracts import AgentPaused
from trellis.contracts.messages import AgentResponse, AgentStatus

from trellis.harness import AgentHarness
from trellis.harness.runs import RunStoreClient

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
    # the interrupt as the contracts spell it: a question, with its own id
    assert awaiting["reason"] == "QUESTION" and awaiting["interrupt_id"].startswith("int_")


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


# ------------------------------------------------------- answering, reading, the port shape


async def test_an_answer_reaches_the_store_after_the_pause_and_moves_the_run_on() -> None:
    """``resumed`` posts to the resume route with the answer on it, after the PAUSED
    transition it answers (the pause is recorded off the turn; the answer waits for it)."""
    from trellis.contracts import InterruptDecision, InterruptResolution

    calls, handler = recording()
    harness = build(store(handler))
    ctx = harness.context_factory.build(agent_id="deploy", overrides={"turn_id": "t1"})

    @harness.agent(agent_id="deploy")
    async def agent(state, runtime):
        raise AgentPaused("Which region?", expects={"type": "string"})

    with pytest.raises(AgentPaused):
        await agent({}, context=ctx)
    interrupt = harness.resolutions.announced("acme")[-1]
    answer = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.ANSWER,
        answer="eu",
    )
    await harness.resume(interrupt, answer, context=ctx)
    paths = [c["path"] for c in calls]
    paused = next(
        i for i, c in enumerate(calls) if c["body"] and c["body"].get("status") == "PAUSED"
    )
    resumed = paths.index(f"/v1/runs/{interrupt.run_id}/resume")
    assert paused < resumed
    assert calls[resumed]["tenant"] == "acme"
    assert calls[resumed]["body"]["answer"]["decision"] == "ANSWER"
    assert calls[resumed]["body"]["answer"]["answer"] == "eu"


async def test_a_failed_turn_carries_its_error_and_a_refused_one_is_rejected_when_raising() -> None:
    from trellis.contracts import PolicyDeniedError

    from trellis.harness import CallablePolicyProvider

    calls, handler = recording()
    harness = AgentHarness(
        runs=store(handler), defaults={"tenant_id": "acme"}, config=CONFIG, error_mode="return"
    )

    @harness.agent(agent_id="triage")
    async def agent(state, runtime):
        raise RuntimeError("boom")

    await agent({})
    await harness.drain()
    last = calls[-1]["body"]
    assert last["status"] == "ERROR" and "boom" in last["error"]["message"]

    calls.clear()
    raising = AgentHarness(
        runs=store(handler),
        defaults={"tenant_id": "acme"},
        config=CONFIG,
        error_mode="raise",
        tools=[lambda: "x"],
        policy=CallablePolicyProvider(tool=lambda c, call: "no tools today"),
    )

    @raising.agent(agent_id="triage")
    async def denied(state, runtime):
        return await runtime.tools.call("<lambda>")

    with pytest.raises(PolicyDeniedError):
        await denied({})
    await raising.drain()
    assert calls[-1]["body"]["status"] == "REJECTED"  # not the error category, not RUNNING


async def test_get_and_list_paused_read_the_inbox() -> None:
    good = {
        "run_id": "r1",
        "tenant_id": "acme",
        "agent_id": "deploy",
        "status": "PAUSED",
        "awaiting": {
            "interrupt_id": "int_1",
            "tenant_id": "acme",
            "run_id": "r1",
            "reason": "QUESTION",
            "question": "Which region?",
            "created_at": "2026-09-28T10:00:00Z",
        },
    }
    legacy = {  # written by 0.2.0: no interrupt id, the pause's class name as the reason
        "run_id": "r0",
        "tenant_id": "acme",
        "agent_id": "deploy",
        "status": "PAUSED",
        "awaiting": {
            "reason": "AgentPaused",
            "question": "Approve?",
            "expects": {"type": "boolean"},
        },
    }
    unreadable = {"run_id": "r2", "tenant_id": "acme", "agent_id": "deploy"}  # no status
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "path": request.url.path,
                "params": dict(request.url.params),
                "tenant": request.headers.get("x-tenant-id"),
            }
        )
        if request.url.path == "/v1/runs/r1":
            return httpx.Response(200, json=good)
        if request.url.path == "/v1/runs/r404":
            return httpx.Response(404, text="no run")
        if request.url.path == "/v1/runs/r500":
            return httpx.Response(500, text="down")
        return httpx.Response(200, json={"runs": [legacy, unreadable, good]})

    client = store(handler)
    record = await client.get("r1")
    assert record is not None and record.awaiting is not None
    assert record.awaiting.question == "Which region?" and record.awaiting.interrupt_id == "int_1"
    assert await client.get("r404") is None
    inbox = await client.list_paused("acme", limit=5)
    assert [r.run_id for r in inbox] == ["r0", "r1"]  # the unreadable row is dropped
    assert inbox[0].awaiting is not None and inbox[0].awaiting.question == "Approve?"
    assert inbox[0].awaiting.expects == {"type": "boolean"}
    listing = seen[-1]
    assert listing["params"] == {"status": "PAUSED", "limit": "5"} and listing["tenant"] == "acme"
    strict = store(handler, required=True)
    with pytest.raises(RuntimeError, match="agent-runs is required"):
        await strict.get("r500")
    bound = store(handler, tenant_id="acme")
    with pytest.raises(RuntimeError, match="bound to tenant"):
        await bound.list_paused("globex")


async def test_the_tenant_a_run_was_opened_under_names_its_later_records() -> None:
    """The port's ``finished`` and ``resumed`` carry no tenant; the client remembers it."""
    from trellis.contracts import RunStart, RunStatus

    calls, handler = recording()
    client = store(handler)  # not bound to a tenant
    await client.started(RunStart(run_id="r9", tenant_id="globex", agent_id="a"))
    await client.finished("r9", RunStatus.SUCCESS, output={"ok": True})
    assert [c["tenant"] for c in calls] == ["globex", "globex"]
    assert calls[-1]["body"] == {"status": "SUCCESS", "output": {"ok": True}}
