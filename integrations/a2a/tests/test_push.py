"""Push notifications: signed like every other harness webhook, and pointed only where a webhook
may point."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.types import Task, TaskPushNotificationConfig, TaskState, TaskStatus
from a2a.utils.errors import A2AError
from a2a_support import (
    ACME,
    AGENT_URL,
    TENANT,
    asgi_client,
    caller,
    greeter,
    harness,
    request,
    sdk_client,
    send,
    task_id_of,
)
from trellis.memory.webhooks import EVENT_HEADER, SIGNATURE_HEADER, verify_signature

from trellis.harness import WebhookEventSink
from trellis.harness.events.webhook import DEFAULT_TYPES
from trellis.harness_a2a import A2AServer, HarnessPushNotifier, TrustedHeaderIdentity
from trellis.harness_a2a.push import EVENT_NAME, TOKEN_HEADER

SECRET = "whsec_test"
TARGET = "https://receiver.example.com/hooks/a2a"


def recorder() -> tuple[list[httpx.Request], httpx.AsyncClient]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handle))


def notifier(client: httpx.AsyncClient, **options: Any) -> HarnessPushNotifier:
    options.setdefault("verify_targets", False)  # no DNS in a unit test
    return HarnessPushNotifier(
        InMemoryPushNotificationConfigStore(), secret=SECRET, client=client, **options
    )


def task(state: TaskState = TaskState.TASK_STATE_COMPLETED) -> Task:
    return Task(id="run-1", context_id="ctx-1", status=TaskStatus(state=state))


async def test_a_delivery_is_signed_the_way_every_harness_webhook_is() -> None:
    seen, client = recorder()
    push = notifier(client)
    assert await push.deliver(TARGET, "run-1", {"task": {"id": "run-1"}}, token="tok") is True
    request = seen[0]
    assert str(request.url) == TARGET
    assert verify_signature(SECRET, request.headers[SIGNATURE_HEADER], request.content)
    assert request.headers[EVENT_HEADER] == EVENT_NAME
    assert request.headers[TOKEN_HEADER] == "tok"
    assert json.loads(request.content) == {"task": {"id": "run-1"}}
    assert push.delivered == 1
    await client.aclose()


@pytest.mark.parametrize(
    "url",
    [
        "http://receiver.example.com/hook",  # not https
        "https://user:pass@receiver.example.com/hook",  # credentials in the URL
        "https://127.0.0.1/hook",  # a private address
        "https://localhost/hook",  # a local name
        "https://receiver.example.com/hook#fragment",
    ],
)
async def test_a_target_a_webhook_may_not_point_at_is_refused(url: str) -> None:
    seen, client = recorder()
    push = notifier(client)
    assert await push.deliver(url, "run-1", {"task": {}}) is False
    assert await push.validate_url(url) is False
    assert seen == []  # nothing left the process
    assert push.refused == 2 and push.delivered == 0
    await client.aclose()


async def test_every_registered_target_of_a_task_is_notified() -> None:
    seen, client = recorder()
    push = notifier(client)
    context = ServerCallContext()
    for index, url in enumerate((TARGET, "https://second.example.com/hook")):
        await push._configs.set_info(
            "run-1",
            TaskPushNotificationConfig(id=f"cfg-{index}", task_id="run-1", url=url, token="tok"),
            context,
        )
    await push.send_notification("run-1", task())
    assert {str(r.url) for r in seen} == {TARGET, "https://second.example.com/hook"}
    assert json.loads(seen[0].content)["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    await push.send_notification("another-run", task())  # no config: no delivery
    assert len(seen) == 2
    await client.aclose()


async def test_a_failing_receiver_is_retried_then_counted_not_raised() -> None:
    attempts: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    push = notifier(client, attempts=2)
    assert await push.deliver(TARGET, "run-1", {"task": {}}) is False
    assert len(attempts) == 2 and push.failed == 1
    await client.aclose()


async def test_a_notifier_agrees_with_the_deployments_run_webhooks() -> None:
    """One secret and one target policy, so a receiver verifies A2A pushes and run events alike."""
    sink = WebhookEventSink(
        "https://receiver.example.com/runs",
        secret=SECRET,
        types=DEFAULT_TYPES,
        attempts=5,
        verify_targets=False,
    )
    push = HarnessPushNotifier.from_sink(sink, InMemoryPushNotificationConfigStore())
    assert push.secret == sink.secret and push.attempts == 5
    assert push.verify_targets is False
    await push.aclose()
    await sink.aclose()


async def test_a_push_url_a_webhook_may_not_point_at_is_refused_at_registration() -> None:
    """The SDK takes our validator, so a bad URL is refused when it is registered — not later,
    quietly, when a notification would have gone to it."""
    instance = harness()
    store = InMemoryPushNotificationConfigStore()
    _seen, client = recorder()
    server = A2AServer(
        instance,
        agent=greeter,
        agent_id="greeter",
        url=AGENT_URL,
        identity=TrustedHeaderIdentity(allowed_tenants={TENANT}),
        push_config_store=store,
        push_notifier=HarnessPushNotifier(
            store, secret=SECRET, client=client, verify_targets=False
        ),
    )
    assert server.card.capabilities.push_notifications is True
    async with asgi_client(server.app()) as http:
        remote = sdk_client(server, http)
        responses = await send(remote, request("ask me"), caller(ACME))
        task_id = task_id_of(responses)
        with pytest.raises(A2AError, match=r"(?i)push|url|not.*safe|invalid"):
            await remote.create_task_push_notification_config(
                TaskPushNotificationConfig(
                    id="cfg-1", task_id=task_id, url="http://169.254.169.254/latest/meta-data"
                ),
                context=caller(ACME),
            )
        # a target a webhook may point at is registered without complaint
        accepted = await remote.create_task_push_notification_config(
            TaskPushNotificationConfig(id="cfg-2", task_id=task_id, url=TARGET),
            context=caller(ACME),
        )
        assert accepted.url == TARGET
    await server.aclose()
    await client.aclose()
