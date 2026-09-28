"""Serving a harness agent over A2A: the card, the stream, the pause, the resume, and every
refusal that has to happen before a run starts."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from a2a.types import GetTaskRequest, SendMessageRequest, TaskState
from a2a.utils.errors import A2AError
from a2a_support import (
    ACME,
    AGENT_URL,
    TENANT,
    StaticDirectory,
    artifacts,
    asgi_client,
    caller,
    data_parts,
    final,
    greeter,
    harness,
    message,
    request,
    sdk_client,
    send,
    states,
    task_id_of,
    texts,
)
from google.protobuf.json_format import MessageToDict
from trellis.contracts.runs import RunStatus

from trellis.harness_a2a import A2AServer, FixedIdentity, TrustedHeaderIdentity

pytestmark = pytest.mark.anyio if False else []  # asyncio_mode=auto: no marker needed


def build(**options: Any) -> tuple[A2AServer, Any]:
    instance = harness()
    options.setdefault("identity", TrustedHeaderIdentity(allowed_tenants={TENANT}))
    server = A2AServer(instance, agent=greeter, agent_id="greeter", url=AGENT_URL, **options)
    return server, instance


async def test_the_card_is_served_at_the_well_known_path() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        response = await http.get(server.card_path)
    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "greeter"
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"][0]["url"] == AGENT_URL
    assert "trusted-identity" in card["capabilities"]["extensions"][0]["uri"]
    await server.aclose()


async def test_a_run_streams_working_then_completes_with_its_result() -> None:
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("world"), caller(ACME))
    assert states(responses)[0] is TaskState.TASK_STATE_SUBMITTED
    assert TaskState.TASK_STATE_WORKING in states(responses)
    assert final(responses) is TaskState.TASK_STATE_COMPLETED
    assert artifacts(responses) == ["hello world"]
    # the run the harness recorded is the task
    run_id = task_id_of(responses)
    assert instance.event_sinks[0].for_run(run_id, TENANT)
    await server.aclose()


async def test_identity_is_the_headers_and_the_task_id_is_the_run_id() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("whoami"), caller(ACME))
    seen = artifacts(responses)[0]
    assert seen["tenant"] == "acme" and seen["user"] == "u1" and seen["workspace"] == "ws1"
    assert seen["run"] == task_id_of(responses)
    assert seen["thread"] == seen["metadata"]["a2a_context_id"]
    assert seen["metadata"]["a2a"] is True
    assert seen["metadata"]["a2a_identity_extension"] is True
    await server.aclose()


async def test_progress_events_reach_the_caller_as_data_parts() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("world"), caller(ACME))
    kinds = {entry["event"] for entry in data_parts(responses) if "event" in entry}
    assert {"STEP_STARTED", "STEP_FINISHED"} <= kinds
    await server.aclose()


async def test_a_pause_becomes_input_required_and_the_next_message_resumes_the_run() -> None:
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        paused = await send(client, request("ask me"), caller(ACME))
        assert final(paused) is TaskState.TASK_STATE_INPUT_REQUIRED
        assert "Which region?" in texts(paused)
        assert any(entry.get("expects") == {"type": "string"} for entry in data_parts(paused))
        task_id = task_id_of(paused)
        assert [i.run_id for i in instance.resolutions.announced(TENANT)] == [task_id]

        resumed = await send(
            client, SendMessageRequest(message=message("eu", task_id=task_id)), caller(ACME)
        )
        assert final(resumed) is TaskState.TASK_STATE_COMPLETED
        assert artifacts(resumed) == ["deploying to eu"]
        assert task_id_of(resumed) == task_id  # the same task, the same run
        assert not instance.resolutions.announced(TENANT)
    await server.aclose()


async def test_a_resume_from_another_caller_is_refused_and_the_pause_survives() -> None:
    """The important half is the second assertion: a refused answer must not end the wait."""
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        paused = await send(client, request("ask me"), caller(ACME))
        task_id = task_id_of(paused)
        # another user of the same tenant does not even see the task: a different owner
        stranger = {"tenant_id": TENANT, "user_id": "mallory", "workspace_id": "ws1"}
        with pytest.raises(A2AError):
            await send(
                client, SendMessageRequest(message=message("eu", task_id=task_id)), caller(stranger)
            )
        # the same user in another workspace shares the task's owner, and is still refused —
        # asked again rather than answered, so the wait survives
        other_workspace = {"tenant_id": TENANT, "user_id": "u1", "workspace_id": "ws-other"}
        refused = await send(
            client,
            SendMessageRequest(message=message("eu", task_id=task_id)),
            caller(other_workspace),
        )
        assert final(refused) is TaskState.TASK_STATE_INPUT_REQUIRED
        assert any("not yours to answer" in text for text in texts(refused))
        assert [i.run_id for i in instance.resolutions.announced(TENANT)] == [task_id]
        # and its own caller can still answer
        resumed = await send(
            client, SendMessageRequest(message=message("eu", task_id=task_id)), caller(ACME)
        )
        assert final(resumed) is TaskState.TASK_STATE_COMPLETED
    await server.aclose()


async def test_a_foreign_tenant_is_refused_before_a_run_starts() -> None:
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        with pytest.raises(A2AError):
            await send(client, request("world"), caller({"tenant_id": "globex"}))
        with pytest.raises(A2AError):  # no identity at all
            await send(client, request("world"), caller(None))
    assert instance.event_sinks[0].events == []  # nothing ran
    await server.aclose()


async def test_a_body_that_claims_another_tenant_is_refused() -> None:
    """A2A requests carry a ``tenant`` of their own, and the SDK's JSON-RPC dispatcher lifts it
    straight off the body onto the call context. It is never identity here, and a body that
    contradicts the authenticated caller is refused rather than served."""
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        with pytest.raises(A2AError):
            await send(client, request("whoami", tenant="globex"), caller(ACME))
        assert instance.event_sinks[0].events == []  # nothing ran
        # naming its own tenant is fine, and changes nothing about who the caller is
        responses = await send(client, request("whoami", tenant=TENANT), caller(ACME))
    assert artifacts(responses)[0]["tenant"] == TENANT
    await server.aclose()


async def test_a_transport_tenant_that_contradicts_the_caller_is_refused() -> None:
    """A2A's REST routes carry a tenant in the path, and the SDK puts it on the call context. A
    deployment whose routing and whose authentication disagree is refused, not guessed at."""
    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.utils.errors import InvalidRequestError

    from trellis.harness_a2a import identity_headers

    server, _ = build()
    call_context = ServerCallContext(state={"headers": identity_headers(ACME)}, tenant="globex")
    with pytest.raises(InvalidRequestError, match="different tenant"):
        server.executor._identity(RequestContext(call_context=call_context))
    ok = ServerCallContext(state={"headers": identity_headers(ACME)}, tenant=TENANT)
    assert server.executor._identity(RequestContext(call_context=ok)) == ACME
    await server.aclose()


async def test_a_task_id_that_is_not_an_identifier_is_refused() -> None:
    """The run id is derived from the task id, and a rewritten id would strand the record."""
    server, _ = build(task_store=None)
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        with pytest.raises(A2AError):
            await send(
                client,
                SendMessageRequest(message=message("world", task_id="../../etc/passwd")),
                caller(ACME),
            )
    await server.aclose()


async def test_a_terminal_task_is_immutable() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        done = await send(client, request("world"), caller(ACME))
        task_id = task_id_of(done)
        with pytest.raises(A2AError):
            await send(
                client, SendMessageRequest(message=message("again", task_id=task_id)), caller(ACME)
            )
        # a refinement is a new task on the same context
        again = await send(
            client,
            SendMessageRequest(message=message("again", context_id="ctx-1")),
            caller(ACME),
        )
        assert final(again) is TaskState.TASK_STATE_COMPLETED
        assert task_id_of(again) != task_id
    await server.aclose()


async def test_an_unknown_task_id_is_not_a_new_run_under_that_id() -> None:
    """The SDK answers ``TaskNotFound`` for a task the caller does not own, which is what stops a
    caller claiming another caller's run id."""
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        with pytest.raises(A2AError):
            await send(
                client,
                SendMessageRequest(message=message("world", task_id="run-nobody-has")),
                caller(ACME),
            )
    assert instance.event_sinks[0].events == []
    await server.aclose()


async def test_a_failing_agent_fails_the_task_with_its_error() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("boom"), caller(ACME))
    assert final(responses) is TaskState.TASK_STATE_FAILED
    assert any("boom" in text for text in texts(responses))
    await server.aclose()


async def test_a_task_can_be_read_back_and_a_stranger_reads_nothing() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        done = await send(client, request("world"), caller(ACME))
        task_id = task_id_of(done)
        task = await client.get_task(GetTaskRequest(id=task_id), context=caller(ACME))
        assert task.id == task_id and task.status.state is TaskState.TASK_STATE_COMPLETED
        with pytest.raises(A2AError):
            await client.get_task(
                GetTaskRequest(id=task_id),
                context=caller({"tenant_id": TENANT, "user_id": "mallory"}),
            )
    await server.aclose()


async def test_a_single_tenant_server_needs_no_header() -> None:
    instance = harness()
    server = A2AServer(
        instance,
        agent=greeter,
        agent_id="greeter",
        url=AGENT_URL,
        identity=FixedIdentity(TENANT, user_id="service"),
    )
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("whoami"), caller(None))
    seen = artifacts(responses)[0]
    assert seen["tenant"] == TENANT and seen["user"] == "service"
    await server.aclose()


async def test_without_a_resolver_the_harness_defaults_are_the_identity() -> None:
    instance = harness()
    server = A2AServer(instance, agent=greeter, agent_id="greeter", url=AGENT_URL)
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        responses = await send(client, request("whoami"), caller(None))
        assert artifacts(responses)[0]["tenant"] == TENANT
        paused = await send(client, request("ask me"), caller(None))
        resumed = await send(
            client,
            SendMessageRequest(message=message("eu", task_id=task_id_of(paused))),
            caller(None),
        )
        assert artifacts(resumed) == ["deploying to eu"]
    await server.aclose()


async def test_a_non_streaming_caller_gets_the_finished_task() -> None:
    server, _ = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http, streaming=False)
        responses = await send(client, request("world"), caller(ACME))
    assert len(responses) == 1
    assert final(responses) is TaskState.TASK_STATE_COMPLETED
    assert artifacts(responses) == ["hello world"]
    await server.aclose()


async def test_an_approval_pause_takes_a_decision_and_a_bad_answer_keeps_the_pause() -> None:
    """Policy approvals arrive as an ``APPROVAL`` interrupt, which needs a decision, not prose."""
    from trellis.harness import CallablePolicyProvider, LocalToolClient

    async def spend(amount: int) -> str:
        """Spend money, which is exactly the kind of tool a person should approve."""
        return f"spent {amount}"

    def approve_tools(_context: Any, _call: Any) -> str:
        return "require_approval"

    instance = harness(
        tools=LocalToolClient({"spend": spend}),
        policy=CallablePolicyProvider(tool=approve_tools),
        error_mode="return",
    )

    async def spender(payload: Any, runtime: Any) -> Any:
        outcome = await runtime.tools.call("spend", amount=5)
        return outcome.output

    server = A2AServer(
        instance,
        agent=spender,
        agent_id="spender",
        url=AGENT_URL,
        identity=TrustedHeaderIdentity(allowed_tenants={TENANT}),
    )
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        paused = await send(client, request("spend it"), caller(ACME))
        assert final(paused) is TaskState.TASK_STATE_INPUT_REQUIRED
        task_id = task_id_of(paused)
        confused = await send(
            client,
            SendMessageRequest(message=message("maybe later", task_id=task_id)),
            caller(ACME),
        )
        # prose does not answer an approval: the agent says so and keeps waiting
        assert final(confused) is TaskState.TASK_STATE_INPUT_REQUIRED
        assert any("approve, reject, cancel" in text for text in texts(confused))
        assert instance.resolutions.announced(TENANT)  # the pause survived the bad answer
        approved = await send(
            client,
            SendMessageRequest(message=message("approve", task_id=task_id)),
            caller(ACME),
        )
        assert final(approved) is TaskState.TASK_STATE_COMPLETED
        assert artifacts(approved) == ["spent 5"]
    await server.aclose()


async def test_cancelling_a_pause_ends_the_task_without_running_the_agent() -> None:
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        paused = await send(client, request("ask me"), caller(ACME))
        task_id = task_id_of(paused)
        cancelled = await send(
            client, SendMessageRequest(message=message("cancel", task_id=task_id)), caller(ACME)
        )
        assert final(cancelled) is TaskState.TASK_STATE_CANCELED
        assert not instance.resolutions.announced(TENANT)
    await server.aclose()


async def test_the_card_url_is_published_to_the_registry() -> None:
    instance = harness()
    directory = StaticDirectory([])
    server = A2AServer(
        instance, agent=greeter, agent_id="greeter", url=AGENT_URL, directory=directory
    )
    assert await server.publish_card() is True
    assert directory.published[0].metadata["card_url"] == server.card_url
    assert server.card_url.endswith("/.well-known/agent-card.json")
    await server.aclose()


async def test_the_task_store_rebuilds_a_task_from_the_run_store() -> None:
    """A task this process never saw is answerable because the run is durable."""
    from trellis.contracts.runs import RunRecord, RunStart

    from trellis.harness_a2a.tasks import HarnessTaskStore

    class OneRun:
        name = "memory"

        def __init__(self, record: RunRecord) -> None:
            self.record = record

        async def get(self, run_id: str) -> RunRecord | None:
            return self.record if run_id == self.record.run_id else None

    record = RunRecord.from_start(
        RunStart(
            run_id="run-1",
            tenant_id=TENANT,
            agent_id="greeter",
            thread_id="ctx-1",
            user_id="u1",
            input="world",
        )
    ).model_copy(update={"status": RunStatus.SUCCESS, "output": "hello world"})
    identity = TrustedHeaderIdentity(allowed_tenants={TENANT})
    store = HarnessTaskStore(OneRun(record), identity=identity)
    assert store.durable is True

    from a2a.server.context import ServerCallContext

    from trellis.harness_a2a import identity_headers

    def context(fields: dict[str, Any]) -> ServerCallContext:
        return ServerCallContext(state={"headers": identity_headers(fields)})

    task = await store.get("run-1", context(ACME))
    assert task is not None
    assert task.status.state is TaskState.TASK_STATE_COMPLETED
    assert task.artifacts[0].parts[0].text == "hello world"
    assert task.history[0].parts[0].text == "world"
    # another tenant, and another user, get nothing at all
    assert await store.get("run-1", context({"tenant_id": "globex", "user_id": "u1"})) is None
    assert await store.get("run-1", context({"tenant_id": TENANT, "user_id": "mallory"})) is None
    assert await store.get("missing", context(ACME)) is None


async def test_a_decision_that_does_not_fit_the_question_keeps_the_pause() -> None:
    """``harness.resume`` refuses a decision the interrupt cannot take. That refusal must not end
    the wait either: ``claim`` has already taken the pause off the registry."""
    server, instance = build()
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        paused = await send(client, request("ask me"), caller(ACME))
        task_id = task_id_of(paused)
        wrong = await send(
            client,
            SendMessageRequest(
                message=message("eu", task_id=task_id, data={"decision": "approve"})
            ),
            caller(ACME),
        )
        assert final(wrong) is TaskState.TASK_STATE_INPUT_REQUIRED
        assert any("not answered with" in text for text in texts(wrong))
        assert [i.run_id for i in instance.resolutions.announced(TENANT)] == [task_id]
        # and the answer it can take still works
        answered = await send(
            client, SendMessageRequest(message=message("eu", task_id=task_id)), caller(ACME)
        )
        assert final(answered) is TaskState.TASK_STATE_COMPLETED
        assert artifacts(answered) == ["deploying to eu"]
    await server.aclose()


async def test_cancelling_a_running_task_ends_it_once() -> None:
    """A cancel arriving while the stream is finishing must not race the stream into the SDK's
    "already terminal" error, and the run behind the task has to be cancelled."""
    import asyncio

    from a2a.types import CancelTaskRequest

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(payload: Any, runtime: Any) -> Any:
        started.set()
        await release.wait()
        return "never"

    instance = harness()
    server = A2AServer(
        instance,
        agent=slow,
        agent_id="slow",
        url=AGENT_URL,
        identity=TrustedHeaderIdentity(allowed_tenants={TENANT}),
    )
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        stream = asyncio.create_task(send(client, request("work"), caller(ACME)))
        await asyncio.wait_for(started.wait(), timeout=5)
        task_id = next(iter(server.executor._running))
        for _ in range(50):  # the framework saves the task on its own schedule
            try:
                await client.get_task(GetTaskRequest(id=task_id), context=caller(ACME))
                break
            except A2AError:
                await asyncio.sleep(0.05)
        cancelled = await client.cancel_task(CancelTaskRequest(id=task_id), context=caller(ACME))
        assert cancelled.status.state is TaskState.TASK_STATE_CANCELED
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(stream, timeout=5)
        assert server.executor._running == {}  # the run's handle is gone
    await server.aclose()


async def test_a_rebuilt_paused_task_tells_the_caller_what_it_asked_and_nothing_else() -> None:
    """``Interrupt.awaiting()`` carries the tool call's arguments unredacted. What a caller gets is
    the question and the shape of the answer."""
    from trellis.contracts.runs import Interrupt, InterruptReason, RunRecord, RunStart
    from trellis.contracts.tool import ToolCall

    from trellis.harness_a2a.tasks import task_from_run

    record = RunRecord.from_start(
        RunStart(run_id="run-2", tenant_id=TENANT, agent_id="greeter", thread_id="ctx-2")
    ).model_copy(
        update={
            "status": RunStatus.PAUSED,
            "awaiting": Interrupt(
                tenant_id=TENANT,
                run_id="run-2",
                reason=InterruptReason.APPROVAL,
                question="Approve the charge?",
                expects={"type": "boolean"},
                tool_call=ToolCall(tool="charge", args={"api_key": "sk-live-SECRET"}),
            ),
        }
    )
    task = task_from_run(record)
    assert task.status.state is TaskState.TASK_STATE_INPUT_REQUIRED
    parts = task.status.message.parts
    assert parts[0].text == "Approve the charge?"
    asked = MessageToDict(parts[1].data)
    assert asked["expects"] == {"type": "boolean"} and asked["reason"] == "APPROVAL"
    assert "sk-live-SECRET" not in str(task), "the unredacted interrupt must not leave the process"
    assert "tool_call" not in asked and "tenant_id" not in asked


async def test_a_rebuilt_failed_task_carries_its_error() -> None:
    from trellis.contracts.errors import AgentError, ErrorCategory
    from trellis.contracts.runs import RunRecord, RunStart

    from trellis.harness_a2a.tasks import task_from_run

    record = RunRecord.from_start(
        RunStart(run_id="run-3", tenant_id=TENANT, agent_id="greeter")
    ).model_copy(
        update={
            "status": RunStatus.ERROR,
            "error": AgentError(code="BOOM", message="it broke", category=ErrorCategory.MODEL),
        }
    )
    task = task_from_run(record)
    assert task.status.state is TaskState.TASK_STATE_FAILED
    assert task.status.message.parts[0].text == "it broke"
    assert MessageToDict(task.status.message.parts[1].data) == {"code": "BOOM"}


async def test_the_task_store_lists_and_deletes_the_callers_own_tasks() -> None:
    from a2a.server.context import ServerCallContext
    from a2a.types import ListTasksRequest
    from trellis.contracts.runs import RunRecord, RunStart

    from trellis.harness_a2a import identity_headers
    from trellis.harness_a2a.tasks import HarnessTaskStore, task_from_run

    store = HarnessTaskStore(None, identity=TrustedHeaderIdentity(allowed_tenants={TENANT}))
    assert store.durable is False  # no run store: the documented in-memory fallback
    mine = ServerCallContext(state={"headers": identity_headers(ACME)})
    theirs = ServerCallContext(
        state={"headers": identity_headers({"tenant_id": TENANT, "user_id": "mallory"})}
    )
    task = task_from_run(
        RunRecord.from_start(RunStart(run_id="run-4", tenant_id=TENANT, agent_id="greeter"))
    )
    await store.save(task, mine)
    assert [t.id for t in (await store.list(ListTasksRequest(), mine)).tasks] == ["run-4"]
    assert (await store.list(ListTasksRequest(), theirs)).tasks == []
    await store.delete("run-4", mine)
    assert await store.get("run-4", mine) is None


async def test_a_deployments_default_user_does_not_lock_a_caller_out_of_its_own_pause() -> None:
    """The run gets the harness's default user (the context factory back-fills it), so the resume
    check has to compare the same fields or nobody could ever answer."""
    instance = harness(defaults={"tenant_id": TENANT, "user_id": "service-account"})
    server = A2AServer(
        instance,
        agent=greeter,
        agent_id="greeter",
        url=AGENT_URL,
        identity=TrustedHeaderIdentity(allowed_tenants={TENANT}),
    )
    async with asgi_client(server.app()) as http:
        client = sdk_client(server, http)
        tenant_only = {"tenant_id": TENANT}
        paused = await send(client, request("ask me"), caller(tenant_only))
        resumed = await send(
            client,
            SendMessageRequest(message=message("eu", task_id=task_id_of(paused))),
            caller(tenant_only),
        )
        assert final(resumed) is TaskState.TASK_STATE_COMPLETED
        assert artifacts(resumed) == ["deploying to eu"]
    await server.aclose()
