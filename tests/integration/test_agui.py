"""The AG-UI surface over a real FastAPI app: run, pause, resume, reconnect, artifacts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import AgentError, RunEvent, RunEventType, RunOutcome, new_id
from trellis.harness.agui.events import AGUIEvent, AGUIEventType
from trellis.harness.agui.hub import MAX_EVENTS_PER_RUN, Hub
from trellis.harness.agui.sse import decode
from trellis.harness.agui.translate import translate
from trellis.harness.identity import Identity

PATH = "/agui"
RELEASE = asyncio.Event()


@tool(side_effects="irreversible")
async def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


async def agent_fn(input: Any, agent: Runtime) -> Any:
    if input == "ask":
        return f"you said {await agent.ask('Proceed?', options=['yes', 'no'])}"
    if input == "refund":
        return await agent.tools.call("refund", order="o1")
    if input == "table":
        rows = [{"n": i, "text": "x" * 400} for i in range(60)]  # past the inline limit
        return await agent.ask("Check these rows", table=rows)
    if input == "wait":
        await RELEASE.wait()
    return f"echo {input}"


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    harness = Harness(config=Settings())
    app = FastAPI()
    harness.wrap(agent_fn, id="chat", tools=[refund]).serve_chat(app, identity=user_of)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://agui", headers={"x-user": "u1"}
    ) as http:
        yield http
    await harness.aclose()


def body(text: str | None = None, **extra: Any) -> dict[str, Any]:
    messages = [{"id": "m1", "role": "user", "content": text}] if text else []
    return {"threadId": "t1", "messages": messages, **extra}


async def post(
    client: httpx.AsyncClient, payload: dict[str, Any]
) -> list[tuple[int, dict[str, Any]]]:
    response = await client.post(f"{PATH}/run", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return decode(response.text)


def finished(events: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    return events[-1][1]


async def test_a_run_streams_numbered_events_ending_in_its_result(
    client: httpx.AsyncClient,
) -> None:
    events = await post(client, body("hello", runId="run-a"))
    assert [n for n, _ in events] == list(range(len(events)))
    assert events[0][1]["type"] == "RUN_STARTED"
    last = finished(events)
    assert last["type"] == "RUN_FINISHED"
    assert last["runId"] == "run-a"
    assert last["outcome"] == {"type": "success"}
    assert last["result"] == "echo hello"


async def test_a_question_pauses_and_a_resume_entry_answers_it(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("ask")))
    assert paused["outcome"]["type"] == "interrupt"
    entry = paused["outcome"]["interrupts"][0]
    assert entry["reason"] == "choice"
    assert entry["message"] == "Proceed?"

    resumed = await post(client, body(resume=[{"interruptId": entry["id"], "payload": "yes"}]))
    first_number = resumed[0][0]
    assert first_number > 0  # the run keeps counting across attempts
    last = finished(resumed)
    assert last["outcome"] == {"type": "success"}
    assert last["result"] == "you said yes"


async def test_an_approval_is_answered_with_true(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("refund")))
    entry = paused["outcome"]["interrupts"][0]
    assert entry["reason"] == "approval"
    assert entry["metadata"]["tool_call"]["tool"] == "refund"

    rejected = await client.post(
        f"{PATH}/run", json=body(resume=[{"interruptId": entry["id"], "payload": "maybe"}])
    )
    assert rejected.status_code == 409

    resumed = await post(client, body(resume=[{"interruptId": entry["id"], "payload": True}]))
    assert finished(resumed)["result"] == "refunded o1"


async def test_a_cancelled_resume_ends_the_run_as_cancelled(client: httpx.AsyncClient) -> None:
    entry = finished(await post(client, body("ask")))["outcome"]["interrupts"][0]
    events = await post(client, body(resume=[{"interruptId": entry["id"], "status": "cancelled"}]))
    assert finished(events)["outcome"] == {"type": "cancelled"}


async def test_an_unknown_interrupt_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(
        f"{PATH}/run", json=body(resume=[{"interruptId": "run_nope.1.1", "payload": 1}])
    )
    assert response.status_code == 404


async def test_a_client_reconnects_after_the_last_event_it_saw(client: httpx.AsyncClient) -> None:
    events = await post(client, body("hello", runId="run-r"))
    replay = await client.get(f"{PATH}/runs/run-r/events", headers={"Last-Event-ID": "0"})
    assert decode(replay.text) == events[1:]
    by_query = await client.get(f"{PATH}/runs/run-r/events", params={"after": 0})
    assert decode(by_query.text) == events[1:]
    stranger = await client.get(f"{PATH}/runs/run-r/events", headers={"x-user": "u2"})
    assert stranger.status_code == 404


async def test_a_reconnect_follows_a_live_run_to_its_end(client: httpx.AsyncClient) -> None:
    RELEASE.clear()
    first = asyncio.create_task(client.post(f"{PATH}/run", json=body("wait", runId="run-live")))
    await asyncio.sleep(0.1)  # the run is open, waiting on RELEASE
    follower = asyncio.create_task(client.get(f"{PATH}/runs/run-live/events"))
    await asyncio.sleep(0.05)
    assert not follower.done()
    RELEASE.set()
    followed = decode((await follower).text)
    assert followed == decode((await first).text)
    assert finished(followed)["result"] == "echo wait"


async def test_a_large_table_travels_as_an_artifact(client: httpx.AsyncClient) -> None:
    paused = finished(await post(client, body("table")))
    reference = paused["outcome"]["interrupts"][0]["metadata"]["payload_ref"]
    run_id = paused["runId"]
    content = await client.get(f"{PATH}/runs/{run_id}/artifacts/{reference['artifact_id']}")
    assert content.status_code == 200
    assert content.headers["content-type"] == "application/json"
    assert len(content.json()["table"]) == 60
    missing = await client.get(f"{PATH}/runs/{run_id}/artifacts/art_missing")
    assert missing.status_code == 404


async def test_a_large_table_is_served_by_another_replica_and_only_while_awaited() -> None:
    first, replica = Harness(config=Settings()), Harness(config=Settings())
    replica.runs = first.runs  # two processes, one run store
    apps = [FastAPI(), FastAPI()]
    first.wrap(agent_fn, id="chat").serve_chat(apps[0], identity=user_of)
    replica.wrap(agent_fn, id="chat").serve_chat(apps[1], identity=user_of)
    clients = [
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=a), base_url="http://agui", headers={"x-user": "u1"}
        )
        for a in apps
    ]
    try:
        paused = finished(decode((await clients[0].post(f"{PATH}/run", json=body("table"))).text))
        entry = paused["outcome"]["interrupts"][0]
        route = f"{PATH}/runs/{paused['runId']}/artifacts/"
        route += entry["metadata"]["payload_ref"]["artifact_id"]
        served = await clients[1].get(route)
        assert served.status_code == 200 and len(served.json()["table"]) == 60
        await clients[1].post(
            f"{PATH}/run", json=body(resume=[{"interruptId": entry["id"], "payload": "ok"}])
        )
        await asyncio.sleep(0.05)
        assert (await clients[1].get(route)).status_code == 404
    finally:
        for c in clients:
            await c.aclose()
        await first.aclose()
        await replica.aclose()


async def test_a_small_payload_travels_inline_and_the_checkpoint_stays_small(
    harness: Harness,
) -> None:
    async def reviewer(input: Any, agent: Runtime) -> Any:
        big = "line\n" * 5000
        return await agent.ask("Accept?", diff=(big, big + "more"), expects={"type": "string"})

    paused = await harness.wrap(reviewer, id="reviewer").run("go", user="u")
    assert paused.interrupt is not None and paused.interrupt.payload is None
    ref = paused.interrupt.payload_ref
    assert ref is not None and ref.size_bytes is not None and ref.size_bytes > 16 * 1024
    record = await harness.runs.get(paused.run_id)
    assert record is not None and len(str(record.checkpoint)) < 16 * 1024


# --------------------------------------------------------------------------- translate


def _context() -> Any:
    return Identity(
        tenant="t", user="u", agent_id="a", run_id=new_id("run_"), thread="th"
    ).context()


def test_run_error_is_folded_into_the_finish_event() -> None:
    context = _context()
    error = AgentError(code="Boom", message="it broke")
    assert translate(RunEvent.of(context, RunEventType.RUN_ERROR, 0, error=error)) is None
    finish = translate(RunEvent.finished(context, RunOutcome.ERROR, 1, error=error))
    assert finish is not None
    assert finish.type is AGUIEventType.RUN_ERROR
    assert (finish.message, finish.code) == ("it broke", "Boom")


def test_context_loaded_becomes_a_custom_event() -> None:
    event = translate(RunEvent.of(_context(), RunEventType.CONTEXT_LOADED, 0, data={"chars": 3}))
    assert event is not None
    assert (event.type, event.name, event.value) == (
        AGUIEventType.CUSTOM,
        "context_loaded",
        {"chars": 3},
    )


def test_tool_events_carry_their_call() -> None:
    context = _context()
    start = translate(RunEvent.tool(context, RunEventType.TOOL_CALL_START, "c1", 0, tool="refund"))
    result = translate(
        RunEvent.tool(context, RunEventType.TOOL_CALL_RESULT, "c1", 1, output={"ok": 1})
    )
    assert start is not None and result is not None
    assert (start.tool_call_id, start.tool_call_name) == ("c1", "refund")
    assert (result.role, result.content) == ("tool", '{"ok": 1}')


# --------------------------------------------------------------------------- hub


async def test_a_buffer_keeps_its_last_events_and_numbers_them() -> None:
    buffer = Hub().open("r", "u")
    for n in range(MAX_EVENTS_PER_RUN + 5):
        buffer.publish(AGUIEvent(type=AGUIEventType.CUSTOM, value=n))
    buffer.publish(AGUIEvent(type=AGUIEventType.RUN_FINISHED), final=True)
    read = [(n, e.value) async for n, e in buffer.read(-1)]
    assert len(read) == MAX_EVENTS_PER_RUN
    assert read[0][0] == 6
    assert read[-2] == (MAX_EVENTS_PER_RUN + 4, MAX_EVENTS_PER_RUN + 4)


def test_the_hub_forgets_finished_runs_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness.agui import hub as module

    monkeypatch.setattr(module, "MAX_RUNS", 2)
    runs = Hub()
    live = runs.open("live", "u")
    done = runs.open("done", "u")
    done.publish(AGUIEvent(type=AGUIEventType.RUN_FINISHED), final=True)
    runs.open("new", "u")
    assert runs.get("done") is None
    assert runs.get("live") is live


def test_every_harness_event_has_its_agui_members() -> None:
    context = _context()

    def one(kind: RunEventType, **fields: Any) -> AGUIEvent:
        found = translate(RunEvent.of(context, kind, 0, **fields))
        assert found is not None
        return found

    assert one(RunEventType.RUN_STARTED).type is AGUIEventType.RUN_STARTED
    assert one(RunEventType.STEP_STARTED, step="plan").step_name == "plan"
    start = one(RunEventType.TEXT_MESSAGE_START, message_id="m1", data={})
    assert (start.message_id, start.role, start.delta) == ("m1", "assistant", None)
    content = one(RunEventType.TEXT_MESSAGE_CONTENT, message_id="m1", data={"delta": "hi"})
    assert (content.delta, content.role) == ("hi", None)
    end = one(RunEventType.TEXT_MESSAGE_END, message_id="m1")
    assert (end.message_id, end.delta) == ("m1", None)
    args = one(RunEventType.TOOL_CALL_ARGS, tool_call_id="c1", data={"args": {"order": "o1"}})
    assert args.delta == '{"order": "o1"}'
    assert one(RunEventType.TOOL_CALL_END, tool_call_id="c1").tool_call_id == "c1"
    assert one(RunEventType.STATE_SNAPSHOT, data={"step": 2}).snapshot == {"step": 2}
    assert one(RunEventType.STATE_SNAPSHOT, data={"snapshot": [1]}).snapshot == [1]
    patch = [{"op": "replace", "path": "/step", "value": 3}]
    assert one(RunEventType.STATE_DELTA, data={"delta": patch}).delta == patch
    messages = [{"role": "user", "content": "hi"}]
    assert one(RunEventType.MESSAGES_SNAPSHOT, data={"messages": messages}).messages == messages
    raw = one(RunEventType.RAW, data={"event": {"x": 1}, "source": "langgraph"})
    assert (raw.event, raw.source) == ({"x": 1}, "langgraph")
    custom = one(RunEventType.CUSTOM, data={"name": "tool_notice", "tool": "note"})
    assert (custom.name, custom.value) == ("tool_notice", {"tool": "note"})


def test_endings_that_are_not_errors_or_successes() -> None:
    context = _context()
    cancelled = translate(RunEvent.finished(context, RunOutcome.CANCELLED, 0))
    assert cancelled is not None and cancelled.wire()["outcome"] == {"type": "cancelled"}
    rejected = translate(RunEvent.finished(context, RunOutcome.REJECTED, 0))
    assert rejected is not None and rejected.type is AGUIEventType.RUN_ERROR
    assert (rejected.message, rejected.code) == ("rejected", "REJECTED")


def test_an_event_the_protocol_has_no_type_for_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module("trellis.harness.agui.translate")
    monkeypatch.setattr(module, "CUSTOM_EVENTS", {})
    loaded = RunEvent.of(_context(), RunEventType.CONTEXT_LOADED, 0, data={"chars": 1})
    assert translate(loaded) is None


def test_values_that_are_not_json_travel_as_their_text() -> None:
    from trellis.harness.agui.translate import _json
    from trellis.harness.tools.convert import text_of

    odd = {("a", "b"): 1}  # a key JSON cannot carry
    assert _json(odd) == str(odd) and text_of(odd) == str(odd)
    assert _json("as is") == "as is"


def test_an_sse_body_skips_lines_that_are_not_events() -> None:
    body = ': a comment\nid: 3\nevent: message\ndata: {"type": "RUN_STARTED"}\n\nretry: 10\n\n'
    assert decode(body) == [(3, {"type": "RUN_STARTED"})]


# --------------------------------------------------------------------------- serving edges


def serve(identity: Any = user_of, **wrap: Any) -> tuple[Harness, httpx.AsyncClient]:
    harness = Harness(config=Settings())
    app = FastAPI()
    harness.wrap(agent_fn, id="chat", tools=[refund], **wrap).serve_chat(app, identity=identity)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
    )
    return harness, http


async def test_without_an_identity_every_caller_is_anonymous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="trellis.agui"):
        harness, http = serve(identity=None)
    assert "every request runs as user 'anonymous'" in caplog.text
    async with http:
        events = await post(http, body("hello", runId="run-anon"))
        assert finished(events)["result"] == "echo hello"
        record = await harness.runs.get("run-anon")
        assert record is not None and record.user_id == "anonymous"
    await harness.aclose()


async def test_an_identity_may_be_async_and_must_name_someone() -> None:
    async def by_token(request: Request) -> str:
        return request.headers.get("authorization", "").removeprefix("Bearer ")

    harness, http = serve(identity=by_token)
    async with http:
        refused = await http.post(f"{PATH}/run", json=body("hi"))
        assert refused.status_code == 401
        http.headers["authorization"] = "Bearer ada"
        events = await post(http, body("hi", runId="run-ada"))
        assert finished(events)["result"] == "echo hi"
        record = await harness.runs.get("run-ada")
        assert record is not None and record.user_id == "ada"
    await harness.aclose()


async def test_a_run_id_must_be_new_and_well_formed(client: httpx.AsyncClient) -> None:
    await post(client, body("hello", runId="run-once"))
    again = await client.post(f"{PATH}/run", json=body("hello", runId="run-once"))
    assert again.status_code == 422
    odd = await client.post(f"{PATH}/run", json=body("hello", runId="run once/../x"))
    assert odd.status_code == 422


async def test_without_a_user_message_the_agent_gets_the_state(client: httpx.AsyncClient) -> None:
    payload = {
        "threadId": "t1",
        "messages": [{"id": "m0", "role": "assistant", "content": "earlier"}],
        "state": "from state",
    }
    assert finished(await post(client, payload))["result"] == "echo from state"


@pytest.mark.parametrize(
    ("resume", "result"),
    [
        ({"payload": False}, "refund was not run: the approver rejected it"),
        ({"payload": {"order": "o2"}}, "refunded o2"),
        (
            {"payload": "too much", "decision": "reject"},
            "refund was not run: the approver rejected it (too much)",
        ),
    ],
    ids=["false-rejects", "object-edits", "decision-named"],
)
async def test_an_approval_is_rejected_edited_or_decided_outright(
    client: httpx.AsyncClient, resume: dict[str, Any], result: str
) -> None:
    entry = finished(await post(client, body("refund")))["outcome"]["interrupts"][0]
    events = await post(client, body(resume=[{"interruptId": entry["id"], **resume}]))
    assert finished(events)["result"] == result


async def test_a_run_the_harness_could_not_finish_is_aborted_for_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trellis.runs import RunsError

    harness, http = serve()

    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise RunsError("agent-runs refused the finish")

    monkeypatch.setattr(harness.runs, "finish", refused)
    async with http:
        last = finished(await post(http, body("hello")))
    assert last["type"] == "RUN_ERROR" and last["code"] == "RUN_ABORTED"
    assert last["message"] == "agent-runs refused the finish"
    await harness.aclose()


async def test_an_artifact_the_store_no_longer_has_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, http = serve()

    async def gone(artifact_id: str, *, tenant: str | None = None) -> None:
        return None

    async with http:
        paused = finished(await post(http, body("table")))
        reference = paused["outcome"]["interrupts"][0]["metadata"]["payload_ref"]
        monkeypatch.setattr(harness.runs.artifacts, "download", gone)
        route = f"{PATH}/runs/{paused['runId']}/artifacts/{reference['artifact_id']}"
        assert (await http.get(route)).status_code == 404
        assert (await http.get(f"{PATH}/runs/run_unknown/artifacts/art_1")).status_code == 404
    await harness.aclose()


# --------------------------------------------------------------------------- the contract


async def test_a_refusal_is_a_problem_document(client: httpx.AsyncClient) -> None:
    entry = finished(await post(client, body("refund")))["outcome"]["interrupts"][0]
    refused = await client.post(
        f"{PATH}/run", json=body(resume=[{"interruptId": entry["id"], "payload": "maybe"}])
    )
    assert refused.status_code == 409
    assert refused.headers["content-type"] == "application/problem+json"
    problem = refused.json()
    assert problem["code"] == "CONFLICT" and problem["detail"].startswith("BAD_RESUME:")
    assert problem["instance"] == f"{PATH}/run" and problem["retryable"] is False
    missing = await client.get(f"{PATH}/runs/run_nope/events")
    assert (missing.status_code, missing.json()["code"]) == (404, "NOT_FOUND")
    gone = await client.get(f"{PATH}/runs/run_nope/artifacts/art_1")
    assert (gone.status_code, gone.json()["code"]) == (404, "NOT_FOUND")
    client.headers["x-user"] = ""
    nobody = await client.post(f"{PATH}/run", json=body("hi"))
    assert (nobody.status_code, nobody.json()["code"]) == (401, "AUTHENTICATION")


async def test_a_decision_or_a_role_outside_the_vocabulary_is_a_validation_error(
    client: httpx.AsyncClient,
) -> None:
    entry = finished(await post(client, body("refund")))["outcome"]["interrupts"][0]
    bad = await client.post(
        f"{PATH}/run",
        json=body(resume=[{"interruptId": entry["id"], "decision": "perhaps"}]),
    )
    assert bad.status_code == 422  # not a 409 BAD_RESUME: the request itself is invalid
    odd = await client.post(
        f"{PATH}/run", json={"threadId": "t1", "messages": [{"role": "robot", "content": "x"}]}
    )
    assert odd.status_code == 422
    approved = await post(
        client, body(resume=[{"interruptId": entry["id"], "decision": "approve"}])
    )  # any case
    assert finished(approved)["result"] == "refunded o1"


def test_the_routes_are_in_the_openapi_document() -> None:
    harness = Harness(config=Settings())
    app = FastAPI()
    agent = harness.wrap(agent_fn, id="chat", tools=[refund])
    agent.serve_chat(app, identity=user_of)
    agent.serve_chat(app, path="/agui2", identity=user_of)  # a second mount: one tag
    doc = app.openapi()
    assert doc["info"]["title"] == "chat agent" and doc["info"]["version"] != "0.1.0"
    assert [t["name"] for t in doc["tags"]] == ["agui"]
    run = doc["paths"][f"{PATH}/run"]["post"]
    assert list(run["responses"]["200"]["content"]) == ["text/event-stream"]
    for status in ("401", "404", "409"):
        assert list(run["responses"][status]["content"]) == ["application/problem+json"]
    assert set(run["responses"]["422"]["content"]) == {
        "application/problem+json",
        "application/json",
    }
    events = doc["paths"][f"{PATH}/runs/{{run_id}}/events"]["get"]
    assert list(events["responses"]["200"]["content"]) == ["text/event-stream"]
    artifact = doc["paths"][f"{PATH}/runs/{{run_id}}/artifacts/{{artifact_id}}"]["get"]
    assert "application/octet-stream" in artifact["responses"]["200"]["content"]
    schemas = doc["components"]["schemas"]
    assert schemas["Resume"]["properties"]["decision"]["anyOf"][0] == {
        "$ref": "#/components/schemas/InterruptDecision"
    }
    assert "user" in schemas["Role"]["enum"]


def test_an_app_that_named_itself_keeps_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    from trellis.harness import agui

    harness = Harness(config=Settings())
    named = FastAPI(title="Procurement", version="3.1")
    harness.wrap(agent_fn, id="chat").serve_chat(named, identity=user_of)
    assert (named.openapi()["info"]["title"], named.version) == ("Procurement", "3.1")
    versioned = FastAPI(version="2.0")  # titled by the surface, its own version kept
    harness.wrap(agent_fn, id="versioned").serve_chat(versioned, identity=user_of)
    assert (versioned.title, versioned.version) == ("versioned agent", "2.0")

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(agui.metadata, "version", missing)
    unnamed = FastAPI()
    harness.wrap(agent_fn, id="other").serve_chat(unnamed, identity=user_of)
    assert unnamed.version == "0"
