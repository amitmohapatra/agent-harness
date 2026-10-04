"""The A2A surface piece by piece: who is calling, where a push may go and how it is signed,
the task store, run events as task updates, the executor's refusals and endings, and what a
remote agent's reply reduces to."""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from a2a.client import Client
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import (
    Artifact,
    CancelTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import InvalidRequestError
from fastapi import FastAPI

from tests.surfaces.test_a2a import (
    URL,
    asgi,
    caller,
    connect,
    send,
    states,
    status_texts,
    task_id_of,
)
from trellis import Harness, Runtime, Settings
from trellis.contracts import (
    AgentError,
    AgentExecutionContext,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunRecord,
    RunStart,
    RunStatus,
    ToolError,
)
from trellis.harness.agent import Agent
from trellis.harness.runs import LocalRuns
from trellis.harness.surfaces.a2a import agent_card, push
from trellis.harness.surfaces.a2a import client as a2a_client
from trellis.harness.surfaces.a2a import executor as executor_module
from trellis.harness.surfaces.a2a.executor import RunExecutor
from trellis.harness.surfaces.a2a.identity import (
    ANONYMOUS,
    IDENTITY_HEADER,
    HeaderIdentity,
    IdentityRefused,
    header,
    identity_headers,
)
from trellis.harness.surfaces.a2a.push import (
    PushNotifier,
    TargetRefused,
    check_addresses,
    validate_url,
)
from trellis.harness.surfaces.a2a.tasks import RunTaskStore, task_from_run
from trellis.harness.surfaces.a2a.translate import (
    Update,
    text_and_data,
    update_for,
    value_part,
    values,
)
from trellis.runs.webhooks import verify_signature

TENANT = "default"


async def tenant() -> str:
    """The tenant a task store reads runs in (a harness's ``tenant``)."""
    return TENANT


def call_context(user: str | None = "u1", raw: str | None = None) -> ServerCallContext:
    if raw is not None:
        return ServerCallContext(state={"headers": {IDENTITY_HEADER: raw}})
    headers = identity_headers(TENANT, user) if user is not None else {}
    return ServerCallContext(state={"headers": headers})


# --------------------------------------------------------------------------- identity


async def test_the_identity_header_names_the_user_of_the_keys_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = Harness(config=Settings())
    resolve = HeaderIdentity(harness)
    with pytest.raises(IdentityRefused, match="not served here"):
        resolve(call_context("u1"))  # the key not asked yet: no tenant is served
    await harness.key()
    assert resolve(call_context("u1")) == "u1"
    with caplog.at_level(logging.WARNING, logger="trellis.a2a"):
        assert resolve(call_context(None)) == ANONYMOUS
        assert resolve(call_context(None)) == ANONYMOUS
    assert caplog.text.count("run as user 'anonymous'") == 1  # warned once per server
    await harness.aclose()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("x" * 5000, "too long"),
        ("{not json", "not JSON"),
        ('["a list"]', "must be a JSON object"),
        ('{"tenant_id": "other", "user_id": "u"}', "tenant 'other' is not served here"),
        ('{"tenant_id": "default"}', "names no user"),
    ],
    ids=["too-long", "not-json", "not-an-object", "another-tenant", "no-user"],
)
async def test_an_identity_header_it_cannot_act_on_is_refused(raw: str, reason: str) -> None:
    harness = Harness(config=Settings())
    await harness.key()
    with pytest.raises(IdentityRefused, match=reason):
        HeaderIdentity(harness)(call_context(raw=raw))
    await harness.aclose()


def test_a_context_without_headers_has_no_identity_header() -> None:
    assert header(ServerCallContext(state={}), IDENTITY_HEADER) == ""
    assert header(ServerCallContext(state={"headers": "garbage"}), IDENTITY_HEADER) == ""
    assert header(ServerCallContext(state={"headers": {"other": "x"}}), IDENTITY_HEADER) == ""


# --------------------------------------------------------------------------- the card


async def test_the_card_describes_what_the_target_says_about_itself() -> None:
    class Described:
        description = "  Finds spare parts.  "

        async def __call__(self, input: Any, agent: Runtime) -> Any:
            return input

    class Silent:
        async def __call__(self, input: Any, agent: Runtime) -> Any:
            return input

    harness = Harness(config=Settings())
    assert agent_card(harness.wrap(Described(), id="parts"), URL).description == (
        "Finds spare parts."
    )
    assert agent_card(harness.wrap(Silent(), id="quiet"), URL).description == "the quiet agent"
    await harness.aclose()


# --------------------------------------------------------------------------- push


def test_a_push_url_is_normalised_and_checked_without_resolving_it() -> None:
    assert validate_url(" https://hooks.example.com ") == "https://hooks.example.com/"
    assert validate_url("https://[::ffff:93.184.216.34]/h") == "https://[::ffff:93.184.216.34]/h"
    for refused, reason in (
        ("https://example.com/" + "a" * 2048, "too long"),
        ("https://[::1]/hook", "private address"),
        ("https://[::ffff:10.0.0.1]/hook", "private address"),
        ("https://printer.local/", "local host"),
        ("https://example.com/#frag", "no credentials or fragment"),
    ):
        with pytest.raises(TargetRefused, match=reason):
            validate_url(refused)


@pytest.fixture
def resolving(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Name resolution answered from a table (a name missing from it does not resolve)."""
    table: dict[str, Any] = {}

    def getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        if host not in table:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in table[host]]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return table


async def test_every_address_a_push_host_resolves_to_must_be_public(
    resolving: dict[str, Any],
) -> None:
    resolving["public.example"] = ["93.184.216.34"]
    resolving["mixed.example"] = ["93.184.216.34", "10.0.0.7"]
    resolving["empty.example"] = []
    await check_addresses("https://public.example/hook")
    with pytest.raises(TargetRefused, match="private address"):
        await check_addresses("https://mixed.example/hook")
    with pytest.raises(TargetRefused, match="private address"):
        await check_addresses("https://empty.example/hook")
    with pytest.raises(TargetRefused, match="does not resolve"):
        await check_addresses("https://nowhere.example/hook")
    notifier = PushNotifier(configs=None)  # type: ignore[arg-type]
    assert await notifier.validate_url("https://public.example/hook")
    assert not await notifier.validate_url("https://mixed.example/hook")


async def test_a_push_is_retried_until_it_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = [httpx.Response(500), httpx.ConnectError("reset"), httpx.Response(202)]
    seen: list[httpx.Request] = []

    def receive(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    pauses: list[float] = []

    async def no_wait(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(push.asyncio, "sleep", no_wait)
    notifier = PushNotifier(
        configs=None,  # type: ignore[arg-type]
        client=httpx.AsyncClient(transport=httpx.MockTransport(receive)),
    )
    assert await notifier.deliver("https://93.184.216.34/hook", "tok", "run-1", b"{}")
    assert len(seen) == 3 and pauses == [0.2, 0.4]
    assert all(verify_signature("tok", r.headers["X-Trellis-Signature"], b"{}") for r in seen)
    assert seen[0].headers["X-Trellis-Event"] == "a2a.task_update"


async def test_a_push_that_is_never_accepted_or_may_not_go_is_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(push.asyncio, "sleep", no_wait)
    attempts: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(410)

    notifier = PushNotifier(
        configs=None,  # type: ignore[arg-type]
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    with caplog.at_level(logging.WARNING, logger="trellis.a2a.push"):
        assert not await notifier.deliver("https://93.184.216.34/hook", "tok", "run-1", b"{}")
        assert not await notifier.deliver("https://10.1.1.1/hook", "tok", "run-2", b"{}")
    assert len(attempts) == push.ATTEMPTS
    assert "A2A push attempt 3 for run-1 failed: HTTP 410" in caplog.text
    assert "A2A push target refused for run-2" in caplog.text


# --------------------------------------------------------------------------- tasks


def record(status: RunStatus, **fields: Any) -> RunRecord:
    start = RunStart(
        run_id="run_1", tenant_id=TENANT, agent_id="greeter", user_id="u1", thread_id="th"
    )
    return RunRecord.from_start(start).model_copy(update={"status": status, **fields})


def test_a_failed_run_is_a_failed_task_with_its_error() -> None:
    task = task_from_run(
        record(RunStatus.ERROR, input="go", error=AgentError(code="Boom", message="it broke"))
    )
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert task.status.message.parts[0].text == "it broke"
    assert task.history[0].parts[0].text == "go"
    assert task.context_id == "th"


def test_a_finished_run_carries_its_result_as_the_tasks_artifact() -> None:
    task = task_from_run(record(RunStatus.SUCCESS, output={"n": 1}))
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    [artifact] = task.artifacts
    assert (artifact.artifact_id, artifact.name) == ("run_1-result", "result")
    assert values(artifact.parts) == [{"n": 1.0}]
    assert list(task.history) == []


async def test_the_task_store_lists_and_deletes_what_it_holds() -> None:
    runs = LocalRuns()
    store = RunTaskStore(runs, agent_id="greeter", user_of=lambda context: "u1", tenant=tenant)
    context = ServerCallContext()
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.TASK_STATE_WORKING))
    await store.save(task, context)
    listed = await store.list(ListTasksRequest(), context)
    assert [t.id for t in listed.tasks] == ["t1"]
    await store.delete("t1", context)
    assert await store.get("t1", context) is None


async def test_a_task_being_opened_or_of_another_agent_is_not_rebuilt() -> None:
    runs = LocalRuns()
    await runs.start(
        RunStart(run_id="run_x", tenant_id=TENANT, agent_id="other", user_id="u1", input="hi")
    )
    await runs.finish("run_x", RunStatus.SUCCESS, output="done")
    store = RunTaskStore(runs, agent_id="greeter", user_of=lambda context: "u1", tenant=tenant)
    assert await store.get("run_x", ServerCallContext()) is None  # another agent's run
    store.opening("run_y")
    assert await store.get("run_y", ServerCallContext()) is None


# --------------------------------------------------------------------------- translate


def event(kind: RunEventType, **fields: Any) -> RunEvent:
    context = AgentExecutionContext.create(tenant_id=TENANT, agent_id="a", agent_run_id="run_1")
    return RunEvent.of(context, kind, 0, **fields)


def test_text_and_progress_keep_the_task_working() -> None:
    text = update_for(
        event(RunEventType.TEXT_MESSAGE_CONTENT, message_id="m1", data={"delta": "hel"})
    )
    assert text is not None and text.parts[0].text == "hel"
    assert not text.terminal
    assert (
        update_for(event(RunEventType.TEXT_MESSAGE_CONTENT, message_id="m1", data={"delta": ""}))
        is None
    )
    assert update_for(event(RunEventType.TEXT_MESSAGE_START, message_id="m1")) is None
    progress = update_for(
        event(RunEventType.TOOL_CALL_START, tool_call_id="c1", data={"tool": "refund"})
    )
    assert progress is not None and progress.state == TaskState.TASK_STATE_WORKING
    assert values(progress.parts) == [
        {"event": "TOOL_CALL_START", "tool_call_id": "c1", "tool": "refund"}
    ]


def test_an_ending_without_an_error_says_how_it_ended() -> None:
    timeout = update_for(event(RunEventType.RUN_FINISHED, outcome=RunOutcome.TIMEOUT))
    assert timeout is not None and timeout.state == TaskState.TASK_STATE_FAILED
    assert timeout.parts[0].text == "the run ended timeout"


def test_values_and_parts_convert_both_ways() -> None:
    assert value_part(object()).WhichOneof("content") == "text"  # not JSON: its text
    assert value_part({"a": 1}).WhichOneof("content") == "data"
    assert text_and_data(None) == ("", {})
    message = new_message(
        [
            new_text_part("first"),
            new_data_part([1, 2]),  # data that is no object merges nothing
            Part(url="https://files.example/report.pdf"),
            Part(raw=b"\x89PNG", media_type="image/png"),  # bytes carry no value here
            new_data_part({"answer": 42}),
            new_text_part("second"),
        ],
        role=Role.ROLE_USER,
    )
    assert text_and_data(message) == ("first\nsecond", {"answer": 42.0})
    assert values(message.parts) == [
        "first",
        [1.0, 2.0],
        "https://files.example/report.pdf",
        {"answer": 42.0},
        "second",
    ]


# --------------------------------------------------------------------------- the executor


class Queue:
    """An event queue that keeps what the executor enqueues."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)

    def states(self) -> list[Any]:
        return [e.status.state for e in self.events if isinstance(e, TaskStatusUpdateEvent)]

    def texts(self) -> list[str]:
        return [
            p.text
            for e in self.events
            if isinstance(e, TaskStatusUpdateEvent)
            for p in e.status.message.parts
        ]


async def waiter(input: Any, agent: Runtime) -> Any:
    if input == "ask":
        return await agent.ask("Which region?")
    await asyncio.Event().wait()
    return "never"


@pytest.fixture
async def executor() -> AsyncIterator[tuple[RunExecutor, Agent]]:
    harness = Harness(config=Settings())
    await harness.key()  # what execute() asks first: the tenant the header is checked against
    agent = harness.wrap(waiter, id="waiter")
    resolve = HeaderIdentity(harness)
    tasks = RunTaskStore(harness.runs, agent_id="waiter", user_of=resolve, tenant=harness.tenant)
    yield RunExecutor(agent, resolve, tasks), agent
    await harness.aclose()


def request(
    task_id: str,
    *,
    state: Any = None,
    text: str = "hi",
    user: str | None = "u1",
    raw: str | None = None,
) -> RequestContext:
    message = new_message([new_text_part(text)], context_id="c1", role=Role.ROLE_USER)
    task = (
        Task(id=task_id, context_id="c1", status=TaskStatus(state=state))
        if state is not None
        else None
    )
    return RequestContext(
        call_context(user, raw),
        request=SendMessageRequest(message=message),
        task_id=task_id,
        context_id="c1",
        task=task,
    )


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (TaskState.TASK_STATE_COMPLETED, "this task has ended"),
        (TaskState.TASK_STATE_WORKING, "still working"),
    ],
    ids=["ended", "working"],
)
async def test_a_message_to_a_task_that_cannot_take_one_is_refused(
    executor: tuple[RunExecutor, Agent], state: Any, reason: str
) -> None:
    run, _ = executor
    with pytest.raises(InvalidRequestError, match=reason):
        await run.execute(request("t1", state=state), Queue())  # type: ignore[arg-type]


async def test_a_new_task_never_takes_the_id_of_an_existing_run(
    executor: tuple[RunExecutor, Agent],
) -> None:
    run, agent = executor
    await agent.harness.runs.start(
        RunStart(run_id="run_taken", tenant_id=TENANT, agent_id="waiter", user_id="eve")
    )
    with pytest.raises(InvalidRequestError, match="no task run_taken"):
        await run.execute(request("run_taken"), Queue())  # type: ignore[arg-type]


async def test_a_caller_with_a_refused_identity_is_refused(
    executor: tuple[RunExecutor, Agent],
) -> None:
    run, _ = executor
    with pytest.raises(InvalidRequestError, match="not JSON"):
        await run.execute(request("t1", raw="{nope"), Queue())  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError, match="not JSON"):
        await run.cancel(request("t1", raw="{nope"), Queue())  # type: ignore[arg-type]


async def test_an_answer_to_someone_elses_question_keeps_the_task_waiting(
    executor: tuple[RunExecutor, Agent],
) -> None:
    run, agent = executor
    queue = Queue()
    await run.execute(request("run_q", text="ask"), queue)  # type: ignore[arg-type]
    assert queue.states()[-1] == TaskState.TASK_STATE_INPUT_REQUIRED
    again = Queue()
    await run.execute(
        request("run_q", state=TaskState.TASK_STATE_INPUT_REQUIRED, text="eu", user="mallory"),
        again,  # type: ignore[arg-type]
    )
    assert again.states() == [TaskState.TASK_STATE_INPUT_REQUIRED]
    assert again.texts() == ["this task is not yours to answer"]
    record = await agent.harness.runs.get("run_q")
    assert record is not None and record.status is RunStatus.PAUSED


async def test_cancelling_a_task_no_run_holds_still_ends_it_once(
    executor: tuple[RunExecutor, Agent],
) -> None:
    run, _ = executor
    queue = Queue()
    await run.cancel(request("run_none"), queue)  # type: ignore[arg-type]
    await run.cancel(request("run_none"), queue)  # type: ignore[arg-type]
    assert queue.states() == [TaskState.TASK_STATE_CANCELED]  # the terminal state, sent once


async def test_a_stream_that_breaks_fails_the_task_once(
    executor: tuple[RunExecutor, Agent], caplog: pytest.LogCaptureFixture
) -> None:
    run, _ = executor

    async def broken() -> AsyncIterator[RunEvent]:
        yield event(RunEventType.RUN_STARTED)
        raise RuntimeError("the harness broke")

    queue = Queue()
    with caplog.at_level(logging.WARNING, logger="trellis.a2a"):
        await run._stream(queue, "t1", "c1", broken())  # type: ignore[arg-type]
    assert queue.states()[-1] == TaskState.TASK_STATE_FAILED
    assert queue.texts()[-1] == "the run ended without a result"
    assert "A2A task t1 failed: the harness broke" in caplog.text
    finished = Update(TaskState.TASK_STATE_COMPLETED, result="late")
    await run._apply(None, "t1", finished)  # type: ignore[arg-type]  # already settled: nothing


async def test_a_stream_cancelled_from_outside_stays_cancelled(
    executor: tuple[RunExecutor, Agent],
) -> None:
    run, _ = executor
    opened = asyncio.Event()

    async def endless() -> AsyncIterator[RunEvent]:
        yield event(RunEventType.RUN_STARTED)
        opened.set()
        await asyncio.Event().wait()

    queue = Queue()
    streaming = asyncio.create_task(run._stream(queue, "t1", "c1", endless()))  # type: ignore[arg-type]
    await opened.wait()
    streaming.cancel()  # the server going away, not cancel(): the task is not settled here
    with pytest.raises(asyncio.CancelledError):
        await streaming
    assert TaskState.TASK_STATE_CANCELED not in queue.states()
    assert run._settle("t1")


def test_settled_tasks_are_remembered_up_to_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_module, "MAX_SETTLED", 1)
    run = RunExecutor(None, lambda c: "u", None)  # type: ignore[arg-type]
    assert run._settle("a") and not run._settle("a")
    assert run._settle("b")
    assert run._settle("a")  # forgotten once the bound pushed it out


# --------------------------------------------------------------------------- over the wire


WORKING = asyncio.Event()


async def slow_greeter(input: Any, agent: Runtime) -> Any:
    if input == "wait":
        WORKING.set()
        await asyncio.Event().wait()
    if input == "ask":
        return f"deploying to {await agent.ask('Which region?')}"
    return f"hello {input}"


@pytest.fixture
async def wire() -> AsyncIterator[tuple[Client, Harness]]:
    harness = Harness(config=Settings())
    app = FastAPI()
    harness.wrap(slow_greeter, id="greeter").serve_a2a(app, URL)
    async with asgi(app) as http:
        yield await connect(http), harness
    await harness.aclose()


async def test_a_working_task_can_be_cancelled(wire: tuple[Client, Harness]) -> None:
    client, harness = wire
    assert isinstance(harness.runs, LocalRuns)
    WORKING.clear()
    # the in-process transport answers once the stream ends: the send runs in the background
    sent = asyncio.create_task(send(client, "wait"))
    await WORKING.wait()
    await asyncio.sleep(0.1)  # the SDK has stored the working task
    [task_id] = list(harness.runs._runs)
    cancelled = await client.cancel_task(CancelTaskRequest(id=task_id), context=caller())
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED
    assert states(await sent)[-1] == TaskState.TASK_STATE_CANCELED
    record = await harness.runs.get(task_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_cancel_word_answers_a_question_by_ending_the_run(
    wire: tuple[Client, Harness],
) -> None:
    client, harness = wire
    task_id = task_id_of(await send(client, "ask"))
    ended = await send(client, "cancel", task_id=task_id)
    assert states(ended)[-1] == TaskState.TASK_STATE_CANCELED
    record = await harness.runs.get(task_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_an_unreadable_decision_asks_again(wire: tuple[Client, Harness]) -> None:
    client, _ = wire
    task_id = task_id_of(await send(client, "ask"))
    again = await send(client, task_id=task_id, data={"decision": "perhaps"})
    assert states(again)[-1] == TaskState.TASK_STATE_INPUT_REQUIRED
    assert "Which region?" in status_texts(again)
    done = await send(client, "eu", task_id=task_id)
    assert states(done)[-1] == TaskState.TASK_STATE_COMPLETED


# --------------------------------------------------------------------------- the client


def test_a_reply_reduces_to_its_artifacts_or_its_text() -> None:
    reply = a2a_client._Reply()
    assert reply.output is None
    reply.absorb(
        StreamResponse(
            task=Task(
                id="t1",
                context_id="c",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
                artifacts=[Artifact(artifact_id="a1", parts=[new_text_part("part one")])],
            )
        )
    )
    reply.absorb(
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c",
                artifact=Artifact(artifact_id="a2", parts=[new_data_part({"n": 2})]),
            )
        )
    )
    reply.absorb(StreamResponse(message=Message(message_id="m", parts=[new_text_part("a note")])))
    reply.absorb(StreamResponse())  # a response with no payload says nothing
    assert reply.task_id == "t1"
    assert reply.output == ["part one", {"n": 2.0}]
    assert reply.text == "a note"
    texts_only = a2a_client._Reply()
    texts_only.absorb(
        StreamResponse(message=Message(message_id="m", parts=[new_text_part("only text")]))
    )
    assert texts_only.output == "only text"


async def test_a_remote_agent_outside_a_run_or_unreachable_is_a_tool_error() -> None:
    card = agent_card(Harness(config=Settings()).wrap(slow_greeter, id="greeter"), URL)
    with pytest.raises(ToolError, match="inside a harness run"):
        await a2a_client._exchange(card, "hi")

    class Down:
        def send_message(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise ConnectionError("refused")

    message = new_message([new_text_part("hi")], role=Role.ROLE_USER)
    with pytest.raises(ToolError, match="ConnectionError: refused"):
        await a2a_client._send(Down(), message, None)  # type: ignore[arg-type]
    async with a2a_client._http() as http:
        assert http.timeout.read == a2a_client.TIMEOUT_SECONDS


async def test_a_remote_failure_is_the_tool_error_the_calling_model_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(input: Any, agent: Runtime) -> Any:
        raise RuntimeError("out of stock")

    remote = Harness(config=Settings())
    app = FastAPI()
    remote.wrap(failing, id="greeter").serve_a2a(app, URL)
    monkeypatch.setattr(a2a_client, "_http", lambda: asgi(app))

    async def delegate(input: Any, agent: Runtime) -> Any:
        return await agent.tools.call("greeter", message=input)

    from trellis import a2a

    harness = Harness(config=Settings())
    result = await harness.wrap(delegate, id="caller", tools=[a2a(URL)]).run("x", user="u1")
    assert result.status is RunStatus.SUCCESS
    assert "greeter failed" in result.answer and "TASK_STATE_FAILED" in result.answer
    assert "out of stock" in result.answer
    await harness.aclose()
    await remote.aclose()
