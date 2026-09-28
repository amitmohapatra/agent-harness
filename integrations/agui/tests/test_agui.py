"""AG-UI: every harness event has its protocol shape, and one route drives, pauses and
resumes with the deployment's identity, never the client's."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from trellis.contracts import (
    AgentError,
    AgentExecutionContext,
    AgentPaused,
    ErrorCategory,
    Interrupt,
    InterruptReason,
    RunEvent,
    RunEventType,
    RunOutcome,
    ToolCall,
)

from trellis.harness import (
    AgentHarness,
    CallablePolicyProvider,
    CollectingEventSink,
    LocalToolClient,
)
from trellis.harness.interrupts import ANSWER
from trellis.harness_agui import AGUIEventType, RunAgentInput, agui_router, encode, translate
from trellis.harness_agui.router import Surface
from trellis.harness_agui.sse import decode

CTX = AgentExecutionContext.create(
    tenant_id="acme", user_id="u1", agent_id="ref", thread_id="thr_1"
)


def _event(kind: RunEventType, **fields: Any) -> RunEvent:
    return RunEvent.of(CTX, kind, 3, **fields)


def test_every_harness_event_has_its_agui_shape() -> None:
    error = AgentError(code="X", message="boom", category=ErrorCategory.MODEL)
    cases = {
        RunEventType.RUN_STARTED: (_event(RunEventType.RUN_STARTED), AGUIEventType.RUN_STARTED, {}),
        RunEventType.STEP_STARTED: (
            _event(RunEventType.STEP_STARTED, step="agent"),
            AGUIEventType.STEP_STARTED,
            {"stepName": "agent"},
        ),
        RunEventType.STEP_FINISHED: (
            _event(RunEventType.STEP_FINISHED, step="agent"),
            AGUIEventType.STEP_FINISHED,
            {"stepName": "agent"},
        ),
        RunEventType.TEXT_MESSAGE_START: (
            _event(RunEventType.TEXT_MESSAGE_START, message_id="m1", data={"role": "assistant"}),
            AGUIEventType.TEXT_MESSAGE_START,
            {"messageId": "m1", "role": "assistant"},
        ),
        RunEventType.TEXT_MESSAGE_CONTENT: (
            _event(RunEventType.TEXT_MESSAGE_CONTENT, message_id="m1", data={"delta": "Hi"}),
            AGUIEventType.TEXT_MESSAGE_CONTENT,
            {"messageId": "m1", "delta": "Hi"},
        ),
        RunEventType.TEXT_MESSAGE_END: (
            _event(RunEventType.TEXT_MESSAGE_END, message_id="m1"),
            AGUIEventType.TEXT_MESSAGE_END,
            {"messageId": "m1"},
        ),
        RunEventType.TOOL_CALL_START: (
            _event(RunEventType.TOOL_CALL_START, tool_call_id="c1", data={"tool": "refund"}),
            AGUIEventType.TOOL_CALL_START,
            {"toolCallId": "c1", "toolCallName": "refund"},
        ),
        RunEventType.TOOL_CALL_ARGS: (
            _event(RunEventType.TOOL_CALL_ARGS, tool_call_id="c1", data={"args": {"a": 1}}),
            AGUIEventType.TOOL_CALL_ARGS,
            {"toolCallId": "c1", "delta": '{"a": 1}'},
        ),
        RunEventType.TOOL_CALL_END: (
            _event(RunEventType.TOOL_CALL_END, tool_call_id="c1"),
            AGUIEventType.TOOL_CALL_END,
            {"toolCallId": "c1"},
        ),
        RunEventType.TOOL_CALL_RESULT: (
            _event(RunEventType.TOOL_CALL_RESULT, tool_call_id="c1", data={"output": {"ok": True}}),
            AGUIEventType.TOOL_CALL_RESULT,
            {"toolCallId": "c1", "content": '{"ok": true}', "role": "tool"},
        ),
        RunEventType.STATE_SNAPSHOT: (
            _event(RunEventType.STATE_SNAPSHOT, data={"snapshot": {"n": 1}}),
            AGUIEventType.STATE_SNAPSHOT,
            {"snapshot": {"n": 1}},
        ),
        RunEventType.STATE_DELTA: (
            _event(RunEventType.STATE_DELTA, data={"delta": [{"op": "add"}]}),
            AGUIEventType.STATE_DELTA,
            {"delta": [{"op": "add"}]},
        ),
        RunEventType.MESSAGES_SNAPSHOT: (
            _event(RunEventType.MESSAGES_SNAPSHOT, data={"messages": [{"role": "user"}]}),
            AGUIEventType.MESSAGES_SNAPSHOT,
            {"messages": [{"role": "user"}]},
        ),
        RunEventType.RAW: (
            _event(RunEventType.RAW, data={"event": {"x": 1}, "source": "graph"}),
            AGUIEventType.RAW,
            {"event": {"x": 1}, "source": "graph"},
        ),
        RunEventType.CUSTOM: (
            _event(RunEventType.CUSTOM, data={"name": "recommendation", "actions": ["a"]}),
            AGUIEventType.CUSTOM,
            {"name": "recommendation", "value": {"actions": ["a"]}},
        ),
        RunEventType.CONTEXT_LOADED: (
            _event(RunEventType.CONTEXT_LOADED, data={"has_context": True}),
            AGUIEventType.CUSTOM,
            {"name": "context_loaded", "value": {"has_context": True}},
        ),
        RunEventType.RUN_ERROR: (
            _event(RunEventType.RUN_ERROR, error=error),
            AGUIEventType.RUN_ERROR,
            {"message": "boom", "code": "X"},
        ),
        RunEventType.RUN_FINISHED: (
            RunEvent.finished(CTX, RunOutcome.SUCCESS, 3, result={"answer": 1}),
            AGUIEventType.RUN_FINISHED,
            {"outcome": {"type": "success"}, "result": {"answer": 1}},
        ),
    }
    assert set(cases) | {RunEventType.INTERRUPT} == set(RunEventType)
    for kind, (event, expected_type, members) in cases.items():
        agui = translate(event)
        assert agui is not None, kind
        wire = agui.wire()
        assert (
            wire["type"] == expected_type.value
            and wire["runId"] == CTX.agent_run_id
            and wire["threadId"] == "thr_1"
        )
        for key, value in members.items():
            assert wire[key] == value, (kind, key)
    line = encode(agui)
    assert (
        line.startswith("data: ")
        and line.endswith("\n\n")
        and decode(line)[0]["type"] == "RUN_FINISHED"
    )


def test_a_pause_and_a_cancellation_are_outcome_objects_and_failures_are_errors() -> None:
    """``RunFinished.outcome`` is the protocol's discriminated object: ``{type: "interrupt",
    interrupts: [...]}``, ``{type: "cancelled"}``; a bare string is not an outcome."""
    call = ToolCall(tool="refund", args={"amount": 240}, idempotency_key="k1")
    approval = Interrupt(
        tenant_id="acme",
        run_id=CTX.agent_run_id,
        reason=InterruptReason.APPROVAL,
        question="Approve the tool call refund?",
        expects={"type": "boolean"},
        payload={"reason": "over the limit"},
        tool_call=call,
    )
    assert translate(RunEvent.interrupt(CTX, approval, 4)) is None  # folded into the finish
    paused = translate(RunEvent.finished(CTX, RunOutcome.INTERRUPT, 5, interrupt=approval))
    assert paused is not None
    outcome = paused.wire()["outcome"]
    assert outcome["type"] == "interrupt" and len(outcome["interrupts"]) == 1
    entry = outcome["interrupts"][0]
    assert entry["id"] == approval.interrupt_id and entry["reason"] == "approval"
    assert entry["message"] == "Approve the tool call refund?" and entry["toolCallId"] == "k1"
    assert entry["responseSchema"] == {"type": "boolean"}
    assert entry["metadata"]["payload"] == {"reason": "over the limit"}
    assert entry["metadata"]["tool_call"]["args"] == {"amount": 240}
    cancelled = translate(RunEvent.finished(CTX, RunOutcome.CANCELLED, 6))
    assert cancelled is not None and cancelled.wire()["outcome"] == {"type": "cancelled"}
    error = AgentError(code="X", message="boom", category=ErrorCategory.MODEL)
    for failure in (RunOutcome.ERROR, RunOutcome.REJECTED, RunOutcome.TIMEOUT):
        failed = translate(RunEvent.finished(CTX, failure, 7, error=error))
        assert failed is not None and failed.type is AGUIEventType.RUN_ERROR
        assert failed.message == "boom" and failed.code == "X"


# ---------------------------------------------------------------------------- the route


async def greeter(payload: Any, runtime: Any) -> Any:
    answer = runtime.state.get("resolutions", {}).get(ANSWER)
    if isinstance(payload, str) and payload.startswith("ask") and answer is None:
        raise AgentPaused("Which region?", expects={"type": "string"})
    if isinstance(payload, str) and payload == "whoami":
        return {
            "user": runtime.context.user_id,
            "tenant": runtime.context.tenant_id,
            "workspace": runtime.context.workspace_id,
            "forwarded": runtime.state["request"].metadata.get("forwarded_props"),
            "tools": runtime.state["request"].metadata.get("frontend_tools"),
        }
    return f"hello {payload}" if answer is None else f"deploying to {answer.answer}"


def _app(**router_options: Any) -> tuple[FastAPI, AgentHarness]:
    harness = AgentHarness(memory=None, event_sinks=[CollectingEventSink()], error_mode="return")
    app = FastAPI()
    options = {"tenant_id": "acme", **router_options}
    app.include_router(agui_router(harness, agent=greeter, agent_id="greeter", **options))
    return app, harness


async def _post(app: FastAPI, body: dict[str, Any]) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui"
    ) as client:
        response = await client.post("/agui/run", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return decode(response.text)


async def _status(app: FastAPI, body: dict[str, Any]) -> int:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui"
    ) as client:
        return (await client.post("/agui/run", json=body)).status_code


async def test_the_route_streams_a_run_then_a_pause_then_its_resume() -> None:
    app, _harness = _app()
    events = await _post(
        app,
        {
            "threadId": "thr_a",
            "runId": "run_client_1",
            "messages": [{"id": "1", "role": "user", "content": "world"}],
        },
    )
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert all(e["threadId"] == "thr_a" and e["runId"] == "run_client_1" for e in events)
    assert events[-1]["outcome"] == {"type": "success"} and events[-1]["result"] == "hello world"
    assert "STEP_STARTED" in types and "STEP_FINISHED" in types

    paused = await _post(
        app,
        {"threadId": "thr_b", "messages": [{"id": "1", "role": "user", "content": "ask me"}]},
    )
    finish = paused[-1]
    assert finish["type"] == "RUN_FINISHED" and finish["outcome"]["type"] == "interrupt"
    interrupt = finish["outcome"]["interrupts"][0]
    assert interrupt["message"] == "Which region?" and interrupt["reason"] == "question"
    assert interrupt["responseSchema"] == {"type": "string"}

    resumed = await _post(
        app,
        {
            "threadId": "thr_b",
            "messages": [],
            "resume": [{"interruptId": interrupt["id"], "status": "resolved", "payload": "eu"}],
        },
    )
    assert resumed[-1]["type"] == "RUN_FINISHED" and resumed[-1]["outcome"] == {"type": "success"}
    assert resumed[-1]["result"] == "deploying to eu"
    assert resumed[-1]["runId"] == finish["runId"]  # the same run continued
    again = await _post(
        app,
        {"threadId": "thr_b", "resume": [{"interruptId": interrupt["id"], "status": "resolved"}]},
    )
    assert again[-1]["type"] == "RUN_ERROR" and again[-1]["code"] == "UNKNOWN_INTERRUPT"


async def test_a_resume_is_the_thread_and_callers_own_and_a_bad_one_is_refused() -> None:
    app, harness = _app()
    paused = await _post(
        app, {"threadId": "thr_c", "messages": [{"id": "1", "role": "user", "content": "ask"}]}
    )
    interrupt_id = paused[-1]["outcome"]["interrupts"][0]["id"]
    other_thread = await _post(
        app,
        {"threadId": "thr_other", "resume": [{"interruptId": interrupt_id, "status": "resolved"}]},
    )
    assert other_thread[-1]["code"] == "UNKNOWN_INTERRUPT"
    assert harness.resolutions.announced("acme")  # still waiting for its own thread
    assert (
        await _status(
            app, {"threadId": "thr_c", "resume": [{"interruptId": interrupt_id, "status": "lost"}]}
        )
        == 422
    )
    bad = await _post(
        app,
        {
            "threadId": "thr_c",
            "resume": [{"interruptId": interrupt_id, "status": "resolved", "decision": "approve"}],
        },
    )
    assert bad[-1]["type"] == "RUN_ERROR" and bad[-1]["code"] == "BAD_RESUME"
    assert harness.resolutions.announced("acme")  # a malformed answer keeps the pause
    cancelled = await _post(
        app,
        {"threadId": "thr_c", "resume": [{"interruptId": interrupt_id, "status": "cancelled"}]},
    )
    assert cancelled[-1]["type"] == "RUN_FINISHED"
    assert cancelled[-1]["outcome"] == {"type": "cancelled"}
    assert not harness.resolutions.announced("acme")


async def test_identity_is_the_deployments_never_the_clients() -> None:
    app, _harness = _app()
    events = await _post(
        app,
        {
            "threadId": "thr_id",
            "messages": [{"id": "1", "role": "user", "content": "whoami"}],
            "forwardedProps": {"userId": "mallory", "tenantId": "globex"},
            "tools": [{"name": "confirm", "description": "Ask the person", "parameters": {}}],
        },
    )
    seen = events[-1]["result"]
    assert seen["user"] is None and seen["tenant"] == "acme" and seen["workspace"] is None
    assert seen["forwarded"] == {"userId": "mallory", "tenantId": "globex"}  # data, not identity
    assert seen["tools"][0]["name"] == "confirm"  # offered to the agent, executed by no one

    def from_auth(request: Request, body: RunAgentInput) -> dict[str, Any]:
        return {
            "tenant_id": request.headers["x-tenant"],
            "user_id": request.headers.get("x-user"),
            "workspace_id": "ws1",
        }

    harness = AgentHarness(memory=None, error_mode="return")
    app = FastAPI()
    app.include_router(
        agui_router(harness, agent=greeter, agent_id="greeter", context_factory=from_auth)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui"
    ) as client:
        response = await client.post(
            "/agui/run",
            json={"threadId": "t", "messages": [{"id": "1", "role": "user", "content": "whoami"}]},
            headers={"x-tenant": "globex", "x-user": "u7"},
        )
    seen = decode(response.text)[-1]["result"]
    assert seen == {
        "user": "u7",
        "tenant": "globex",
        "workspace": "ws1",
        "forwarded": None,
        "tools": None,
    }
    with pytest.raises(ValueError, match="context_factory or a tenant_id"):
        agui_router(harness, agent=greeter, agent_id="greeter")


async def test_an_approval_is_answered_with_true_false_or_edited_arguments() -> None:
    refunds: list[int] = []

    def refund(amount: int) -> dict[str, Any]:
        refunds.append(amount)
        return {"refunded": amount}

    harness = AgentHarness(
        memory=None,
        tools=LocalToolClient({"refund": refund}),
        policy=CallablePolicyProvider(tool=lambda c, call: "require_approval"),
        error_mode="return",
    )

    async def agent(payload: Any, runtime: Any) -> Any:
        return (await runtime.tools.call("refund", amount=240)).output

    app = FastAPI()
    app.include_router(agui_router(harness, agent=agent, agent_id="refunder", tenant_id="acme"))
    for payload, expected in ((True, [240]), ({"amount": 200}, [200]), (False, [])):
        refunds.clear()
        paused = await _post(
            app,
            {
                "threadId": f"thr_{expected}",
                "messages": [{"id": "1", "role": "user", "content": "go"}],
            },
        )
        entry = paused[-1]["outcome"]["interrupts"][0]
        assert entry["reason"] == "approval" and entry["toolCallId"]
        resumed = await _post(
            app,
            {
                "threadId": f"thr_{expected}",
                "resume": [{"interruptId": entry["id"], "status": "resolved", "payload": payload}],
            },
        )
        assert resumed[-1]["type"] == "RUN_FINISHED", resumed[-1]
        assert refunds == expected
        tool_results = [e for e in resumed if e["type"] == "TOOL_CALL_RESULT"]
        assert tool_results and tool_results[-1]["toolCallId"] == entry["toolCallId"]


async def test_a_run_id_that_is_not_an_identifier_is_refused() -> None:
    app, _harness = _app()
    body = {"threadId": "t", "runId": "not an id", "messages": []}
    assert await _status(app, body) == 422


async def test_the_router_works_on_a_harness_without_a_collecting_sink() -> None:
    harness = AgentHarness(memory=None, error_mode="return")
    app = FastAPI()
    app.include_router(agui_router(harness, agent=greeter, agent_id="greeter", tenant_id="acme"))
    events = await asyncio.wait_for(
        _post(app, {"threadId": "t", "messages": [{"id": "1", "role": "user", "content": "x"}]}),
        timeout=5,
    )
    assert events[-1]["type"] == "RUN_FINISHED" and events[-1]["result"] == "hello x"


async def test_a_client_that_disconnects_does_not_cancel_the_run() -> None:
    """The webhook notifier exists for the client that hung up; the run it left goes on."""
    sink = CollectingEventSink()
    harness = AgentHarness(memory=None, event_sinks=[sink], error_mode="return")
    gate = asyncio.Event()

    async def slow(payload: Any, runtime: Any) -> str:
        await gate.wait()
        return "finished anyway"

    surface = Surface(harness, agent=slow, agent_id="slow", tenant_id="acme", factory=None)
    body = RunAgentInput(thread_id="thr_gone", messages=[{"role": "user", "content": "go"}])
    stream = surface.stream(body, {"tenant_id": "acme"})
    first = decode(await stream.__anext__())[0]
    assert first["type"] == "RUN_STARTED"
    await stream.aclose()  # the client went away
    gate.set()
    for _ in range(50):
        await asyncio.sleep(0.01)
        finished = [e for e in sink.for_run(first["runId"]) if e.type is RunEventType.RUN_FINISHED]
        if finished:
            break
    assert finished and finished[-1].data["result"] == "finished anyway"


async def test_a_run_that_dies_before_its_first_event_ends_the_stream() -> None:
    harness = AgentHarness(memory=None, error_mode="return")

    class Broken:
        def wrap(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("unused")

    surface = Surface(harness, agent=greeter, agent_id="greeter", tenant_id="acme", factory=None)

    async def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("before anything was emitted")

    surface.runner = exploding
    body = RunAgentInput(thread_id="thr_dead", messages=[{"role": "user", "content": "go"}])
    chunks = [chunk async for chunk in surface.stream(body, {"tenant_id": "acme"})]
    events = decode("".join(chunks))
    assert events[-1]["type"] == "RUN_ERROR" and events[-1]["code"] == "RUN_ABORTED"
